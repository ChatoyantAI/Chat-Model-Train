"""Materialize the coarse-filter pool as complete, scoped offline conversations.

Only writes a new index. Original Parquet files are never modified. Call as a
module with --source, --candidates, --personas, --output. Needs pyarrow at import
time only inside main(), not in the web service.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

from .llm_judge import _split_platform_guidelines, _platform_output_requirements

DATA_SCHEMA = "conversation-filter-data-v1"
DDL = """
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE conversations(
 seq INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL UNIQUE,
 user_key TEXT NOT NULL, character_id TEXT NOT NULL, character_name TEXT NOT NULL,
 language TEXT NOT NULL, languages_json TEXT NOT NULL, candidate_count INTEGER NOT NULL,
 turn_count INTEGER NOT NULL, text_chars INTEGER NOT NULL, context_hash TEXT NOT NULL,
 payload_zlib BLOB NOT NULL
);
CREATE INDEX idx_conversation_language_seq ON conversations(language,seq);
CREATE INDEX idx_conversation_character_seq ON conversations(character_id,seq);
CREATE INDEX idx_conversation_user_seq ON conversations(user_key,seq);
"""


def text(value):
    return "" if value is None else str(value)


def pack_conversation(
    bundle: dict, candidates: list[dict], persona: dict | None = None
) -> dict:
    by_id = {r["target_increase_id"]: r for r in candidates}
    turns = sorted(bundle["turns"], key=lambda t: t["turn_order"])
    target_orders = [
        t["turn_order"] for t in turns if t["assistant_increase_id"] in by_id
    ]
    if len(target_orders) != len(candidates):
        raise ValueError("候选回复与来源轮次没有一一匹配")
    # No material beyond the last candidate is needed by this conversation unit.
    turns = [t for t in turns if t["turn_order"] <= max(target_orders)]
    messages = []
    query_mismatches = target_mismatches = 0
    for index, turn in enumerate(turns, 1):
        candidate = by_id.get(turn["assistant_increase_id"])
        original_user, blocks = _split_platform_guidelines(
            text(turn.get("user_content"))
        )
        if candidate:
            if candidate["query_increase_id"] != turn["user_increase_id"]:
                raise ValueError("候选 User / Assistant 不属于同一原始轮次")
            user, additional = _split_platform_guidelines(text(candidate["query_text"]))
            blocks.extend(b for b in additional if b not in blocks)
            query_mismatches += user.strip() != original_user.strip()
            target_mismatches += (
                text(candidate["target_text"]).strip()
                != text(turn.get("assistant_content")).strip()
            )
        else:
            user = original_user
        requirements = _platform_output_requirements(blocks)
        if turn.get("user_increase_id") is not None or user:
            message = {
                "id": f"U{index}",
                "role": "user",
                "scope": "context_only",
                "content": user,
                "source_increase_id": turn.get("user_increase_id"),
            }
            if requirements:
                message["platform_output_requirements"] = requirements
            messages.append(message)
        answer = (
            text(candidate["target_text"])
            if candidate
            else text(turn.get("assistant_content"))
        )
        if turn.get("assistant_increase_id") is not None or answer:
            message = {
                "id": f"A{index}",
                "role": "assistant",
                "scope": "candidate" if candidate else "context_only",
                "content": answer,
                "source_increase_id": turn.get("assistant_increase_id"),
                "source_model": candidate.get("model")
                if candidate
                else turn.get("assistant_model"),
            }
            if candidate:
                message["candidate_message_id"] = candidate["message_id"]
                message["generation_language"] = candidate["language"]
            messages.append(message)
    languages = Counter(text(r["language"]) for r in candidates)
    main_language = languages.most_common(1)[0][0]
    locale_matches = (
        bool(bundle.get("has_exact_localized_character_description"))
        and bundle.get("source_user_language") == main_language
    )
    prefix = "localized_" if locale_matches else "character_"
    profile = {}
    for field in (
        "name",
        "intro",
        "description",
        "gender",
        "scenario",
        "personality",
        "qa_examples",
    ):
        value = bundle.get(prefix + field) or bundle.get("character_" + field)
        if value:
            profile[field] = value
    if bundle.get("character_age") is not None:
        profile["age"] = bundle["character_age"]
    greeting = text(bundle.get("initial_greeting"))
    # A role card's current opening is not a historical conversation snapshot.
    # Never silently substitute it or synthesize one when the snapshot is absent.
    greeting_source = "initial_greeting" if greeting else "missing"
    gaps = []
    if not any(profile.get(k) for k in ("intro", "description", "personality")):
        gaps.append("角色详细资料缺失，不应猜测人设")
    if greeting_source != "initial_greeting":
        gaps.append("未找到会话原始 Greeting；不以角色卡备选或生成内容替代")
    if bundle.get("persona_id") and not persona:
        gaps.append("会话关联了 persona，但没有可匹配的离线资料")
    return {
        "schema_version": DATA_SCHEMA,
        "conversation_id": bundle["conversation_id"],
        "character_id": bundle["character_id"],
        "user_key": hashlib.sha256(text(bundle["user_id"]).encode()).hexdigest(),
        "character_name": text(profile.get("name")) or "未命名角色",
        "languages": dict(languages),
        "language": main_language,
        "candidate_count": len(candidates),
        "turn_count": len(turns),
        "role_profile": profile,
        "persona": persona,
        "greeting": {"id": "greeting", "content": greeting},
        "messages": messages,
        "context_basis": {
            "mode": "offline_conversation_snapshot",
            "not_original_prompt_replay": True,
            "notes": [
                "角色与 persona 来自离线资料快照，不能保证与生成时版本逐字相同。",
                "展示完整原始轮次至最后候选；不注入后续摘要，不重建逐轮在线裁剪窗口。",
                "候选 User/Assistant 正文以粗筛文件为准，其余轮次从原始来源补回。",
                "平台要求只从已有 guideline 提取；未恢复的目标语言或格式不编造。",
            ],
            "known_gaps": gaps,
        },
        "provenance": {
            "greeting_source": greeting_source,
            "profile_source": prefix,
            "query_text_differences": query_mismatches,
            "target_text_differences": target_mismatches,
            "source_models": dict(Counter(text(r.get("model")) for r in candidates)),
            "summary_ids_not_injected": sorted(
                {r["summary_id"] for r in candidates if r.get("summary_id") is not None}
            ),
        },
    }


def insert_conversation(connection, seq: int, packet: dict):
    serialized = json.dumps(
        packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    content_hash = hashlib.sha256(serialized).hexdigest()
    connection.execute(
        "INSERT INTO conversations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            seq,
            packet["conversation_id"],
            packet["user_key"],
            packet["character_id"],
            packet["character_name"],
            packet["language"],
            json.dumps(packet["languages"]),
            packet["candidate_count"],
            packet["turn_count"],
            sum(len(m["content"]) for m in packet["messages"]),
            content_hash,
            zlib.compress(serialized),
        ),
    )


def main():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--personas")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    building = output.with_suffix(".building.sqlite3")
    if output.exists() or building.exists():
        raise SystemExit("输出已存在；拒绝覆盖，请使用一个新的索引路径")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows = pq.read_table(args.candidates).to_pylist()
    groups = defaultdict(list)
    for r in candidate_rows:
        groups[r["conversation_id"]].append(r)
    seqs = {cid: n for n, cid in enumerate(sorted(groups), 1)}
    wanted = pa.array(sorted(groups), type=pa.large_string())
    persona_ids = {r["persona_id"] for r in candidate_rows if r.get("persona_id")}
    persona_records = defaultdict(list)
    if args.personas and persona_ids:
        # Read only dataset-scoped roleplay profile fields, never account contact data.
        p = pq.read_table(
            args.personas,
            columns=[
                "persona_id",
                "user_id",
                "conversation_id",
                "nickname",
                "gender",
                "age",
                "likes",
                "dislikes",
                "personality",
                "pinned_memories",
                "updated_at",
            ],
        )
        p = p.filter(
            pc.is_in(
                p["persona_id"],
                value_set=pa.array(sorted(persona_ids), type=pa.large_string()),
            )
        )
        for r in p.to_pylist():
            persona_records[r["persona_id"]].append(r)
    columns = [
        "conversation_id",
        "user_id",
        "character_id",
        "persona_id",
        "initial_greeting",
        "has_exact_localized_character_description",
        "source_user_language",
        "character_age",
    ]
    for prefix in ("character_", "localized_"):
        columns.extend(
            prefix + field
            for field in (
                "name",
                "intro",
                "description",
                "gender",
                "scenario",
                "personality",
                "qa_examples",
                "greeting",
            )
        )
    columns.extend(
        "turns.list.element." + field
        for field in (
            "turn_order",
            "assistant_increase_id",
            "user_increase_id",
            "user_content",
            "assistant_content",
            "assistant_model",
        )
    )
    report = Counter()
    connection = sqlite3.connect(building)
    try:
        connection.executescript(DDL)
        for file_index, file in enumerate(
            sorted(Path(args.source).glob("conversation_bucket=*/*.parquet")), 1
        ):
            table = pq.ParquetFile(file).read(columns=columns)
            table = table.filter(pc.is_in(table["conversation_id"], value_set=wanted))
            for bundle in table.to_pylist():
                records = [
                    p
                    for p in persona_records.get(bundle.get("persona_id"), [])
                    if p["user_id"] == bundle["user_id"]
                    and (
                        not p["conversation_id"]
                        or p["conversation_id"] == bundle["conversation_id"]
                    )
                ]
                records.sort(
                    key=lambda p: (
                        p["conversation_id"] == bundle["conversation_id"],
                        str(p["updated_at"]),
                    ),
                    reverse=True,
                )
                persona = (
                    {
                        k: records[0][k]
                        for k in (
                            "nickname",
                            "gender",
                            "age",
                            "likes",
                            "dislikes",
                            "personality",
                            "pinned_memories",
                        )
                        if records[0][k] is not None
                    }
                    if records
                    else None
                )
                packet = pack_conversation(
                    bundle, groups[bundle["conversation_id"]], persona
                )
                insert_conversation(connection, seqs[bundle["conversation_id"]], packet)
                report["conversations"] += 1
                report["candidates"] += packet["candidate_count"]
                report["messages"] += len(packet["messages"])
                report["query_text_differences"] += packet["provenance"][
                    "query_text_differences"
                ]
                report["target_text_differences"] += packet["provenance"][
                    "target_text_differences"
                ]
                for gap in packet["context_basis"]["known_gaps"]:
                    report[gap] += 1
            connection.commit()
            if file_index % 16 == 0:
                print(
                    json.dumps({"shards": file_index, **report}, ensure_ascii=False),
                    flush=True,
                )
        assert report["conversations"] == len(groups)
        assert report["candidates"] == len(candidate_rows)
        report["characters"] = connection.execute(
            "SELECT COUNT(DISTINCT character_id) FROM conversations"
        ).fetchone()[0]
        report["users"] = connection.execute(
            "SELECT COUNT(DISTINCT user_key) FROM conversations"
        ).fetchone()[0]
        manifest = {
            "schema_version": DATA_SCHEMA,
            "source_file": args.candidates,
            "source_bytes": Path(args.candidates).stat().st_size,
            "source_sha256": hashlib.sha256(
                Path(args.candidates).read_bytes()
            ).hexdigest(),
            "report": dict(report),
            "languages": dict(
                connection.execute(
                    "SELECT language,COUNT(*) FROM conversations GROUP BY language"
                )
            ),
            "ready": True,
        }
        connection.execute(
            "INSERT INTO metadata VALUES ('manifest',?)",
            (json.dumps(manifest, ensure_ascii=False),),
        )
        connection.execute("PRAGMA optimize")
        connection.commit()
    finally:
        connection.close()
    building.rename(output)
    print(
        json.dumps({"output": str(output), "manifest": manifest}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
