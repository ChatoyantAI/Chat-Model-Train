"""Join verified source records into a NEW index, without inventing any text.

The live extracts must come from scoped SELECTs through the reader account.
An existing character greeting is kept separate from a saved conversation opening.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

from .conversation_filter_data import DDL, insert_conversation


def extracted_rows(path, prefix):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            group = json.loads(line)
            if group["name"].startswith(prefix):
                yield from group["rows"]


def enrich(packet, conversation, first_message, profile, profile_source, live_card):
    packet["role_profile"].pop("prompt", None)
    packet["context_basis"]["known_gaps"] = [
        g
        for g in packet["context_basis"]["known_gaps"]
        if "Greeting" not in g and "角色详细资料" not in g
    ]
    for key in (
        "name",
        "intro",
        "description",
        "gender",
        "scenario",
        "personality",
        "qa_examples",
        "example_reply",
    ):
        if profile.get(key):
            packet["role_profile"][key] = profile[key]
    packet["character_name"] = packet["role_profile"].get("name") or "未命名角色"
    packet["provenance"]["profile_source"] = profile_source
    packet["provenance"]["profile_language"] = profile.get("language")
    packet["provenance"]["profile_updated_at"] = str(profile.get("updated_at") or "")
    if not any(
        packet["role_profile"].get(k) for k in ("intro", "description", "personality")
    ):
        packet["context_basis"]["known_gaps"].append("角色详细资料缺失，不应猜测人设")
    # User-approved reconstruction: take the existing role-card opening directly.
    # No generation, translation or inferred rewrite. Preserve the source locale.
    greeting = live_card.get("greeting") or profile.get("greeting") or ""
    source = "character_descriptions.greeting" if greeting else "missing"
    historical = False
    if not greeting and conversation.get("initial_greeting"):
        greeting, source, historical = (
            conversation["initial_greeting"],
            "conversations.initial_greeting",
            True,
        )
    if (
        not greeting
        and first_message
        and first_message.get("role") == 1
        and first_message.get("status") == 1
    ):
        first_user = min(
            (
                m["source_increase_id"]
                for m in packet["messages"]
                if m["role"] == "user" and m.get("source_increase_id")
            ),
            default=0,
        )
        if first_message["increase_id"] < first_user:
            greeting = first_message.get("content") or ""
            if greeting:
                source, historical = "messages.first_assistant", True
                packet["provenance"]["greeting_message_increase_id"] = first_message[
                    "increase_id"
                ]
                # The first assistant remains historical background, never a scored turn.
                for message in packet["messages"]:
                    if (
                        message.get("source_increase_id")
                        == first_message["increase_id"]
                    ):
                        if message["scope"] == "candidate":
                            raise ValueError("首条历史开场不能同时作为待评候选")
                        message["displayed_as_greeting"] = True
    packet["greeting"] = {
        "id": "greeting",
        "content": greeting,
        "source": source,
        "historical_snapshot": historical,
    }
    packet["provenance"]["greeting_source"] = source
    packet["provenance"]["greeting_language"] = (
        conversation.get("initial_greeting_language")
        if historical
        else live_card.get("language")
    )
    packet["context_basis"]["greeting_is_historical_snapshot"] = historical
    packet["context_basis"]["notes"] = [
        n
        for n in packet["context_basis"]["notes"]
        if "开场" not in n and "Greeting" not in n
    ]
    packet["context_basis"]["notes"].append(
        "按本数据集约定，Greeting 直接读取角色资料中已有的开场原文，并作为评估上下文。没有生成、翻译或自行补写。优先对应语言，资料不存在时使用已有英文资料，并保留来源语言。会话没有另存 Greeting 不算数据缺失。"
    )
    if not greeting:
        packet["context_basis"]["known_gaps"].append(
            "角色资料和可用会话来源中均没有开场正文"
        )
    return packet


def main():
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    for name in ("input", "output", "conversations", "recovery", "profiles"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--english")
    args = ap.parse_args()
    output = Path(args.output)
    building = output.with_suffix(".building.sqlite3")
    if output.exists() or building.exists():
        raise SystemExit("拒绝覆盖既有索引，请指定新的输出路径")
    conversations = {
        r["id"]: r for r in extracted_rows(args.conversations, "conversations_")
    }
    first = {
        r["conversation_id"]: r
        for r in extracted_rows(args.recovery, "first_messages_")
    }
    live_cards = {
        (r["character_id"], r["language"]): r
        for r in extracted_rows(args.recovery, "character_openings_")
    }
    if args.english:
        live_cards.update(
            {
                (r["character_id"], r["language"]): r
                for r in extracted_rows(args.english, "character_openings_")
            }
        )
    # Match exact requested locale only. Never invent a translation or use user's
    # current account language to overwrite this dataset's generation language.
    src = sqlite3.connect(f"file:{Path(args.input).as_posix()}?mode=ro", uri=True)
    wanted = {
        (
            character,
            conversations.get(cid, {}).get("initial_greeting_language") or language,
        )
        for cid, character, language in src.execute(
            "SELECT conversation_id,character_id,language FROM conversations"
        )
    }
    offline = {}
    for batch in pq.ParquetFile(args.profiles).iter_batches(batch_size=4096):
        for r in batch.to_pylist():
            key = (r["character_id"], r["language"])
            if key in wanted:
                offline[key] = r
    out = sqlite3.connect(building)
    out.executescript(DDL)
    report = Counter()
    try:
        for seq, blob in src.execute(
            "SELECT seq,payload_zlib FROM conversations ORDER BY seq"
        ):
            packet = json.loads(zlib.decompress(blob))
            conv = conversations.get(packet["conversation_id"], {})
            locale = conv.get("initial_greeting_language") or packet["language"]
            key = (packet["character_id"], locale)
            live = live_cards.get(key, {})
            if not live or not live.get("greeting"):
                live = live_cards.get((packet["character_id"], "en")) or live
            profile = live or offline.get(key) or {}
            packet = enrich(
                packet,
                conv,
                first.get(packet["conversation_id"]),
                profile,
                "reader.character_descriptions"
                if live
                else "canonical.character_descriptions",
                live,
            )
            insert_conversation(out, seq, packet)
            report["conversations"] += 1
            report["candidates"] += packet["candidate_count"]
            report["messages"] += len(packet["messages"])
            report[packet["provenance"]["greeting_source"]] += 1
            report["role_card_language_fallback"] += bool(
                profile.get("language") and profile["language"] != locale
            )
            for gap in packet["context_basis"]["known_gaps"]:
                report[gap] += 1
            if seq % 1000 == 0:
                out.commit()
                print(
                    json.dumps(
                        {"indexed": seq, "greetings": dict(report)}, ensure_ascii=False
                    ),
                    flush=True,
                )
        manifest = json.loads(
            src.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0]
        )
        for name, column in (("characters", "character_id"), ("users", "user_key")):
            report[name] = out.execute(
                f"SELECT COUNT(DISTINCT {column}) FROM conversations"
            ).fetchone()[0]
        digest = hashlib.sha256()
        for seq, fingerprint in out.execute(
            "SELECT seq,context_hash FROM conversations ORDER BY seq"
        ):
            digest.update(f"{seq}:{fingerprint}\n".encode())
        manifest.update(
            report=dict(report),
            dataset_sha256=digest.hexdigest(),
            enrichment_sources={
                "conversations": hashlib.sha256(
                    Path(args.conversations).read_bytes()
                ).hexdigest(),
                "recovery": hashlib.sha256(
                    Path(args.recovery).read_bytes()
                ).hexdigest(),
            },
            ready=True,
        )
        out.execute(
            "INSERT INTO metadata VALUES ('manifest',?)",
            (json.dumps(manifest, ensure_ascii=False),),
        )
        out.commit()
    finally:
        src.close()
        out.close()
    building.rename(output)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
