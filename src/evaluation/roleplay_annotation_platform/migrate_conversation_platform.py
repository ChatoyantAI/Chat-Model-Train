"""Build a new platform-assembled index and optionally clean stored request fields.

Never writes production data or changes scores. Run with service stopped when
cleaning its results database. The caller must back up that database first.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

from .conversation_filter_data import DDL, insert_conversation
from .conversation_platform_context import assemble_platform_context, CONTEXT_VERSION


def clean_attempts(connection):
    count = 0
    for row_id, request_json, original_hash, raw_json in connection.execute(
        "SELECT id,request_json,request_hash,raw_json FROM attempts"
    ).fetchall():
        request = json.loads(request_json)
        changed = False
        for message in request.get("messages", []):
            prefix = "请评价以下对话的候选回复组：\n"
            if message.get("role") != "user" or not message.get(
                "content", ""
            ).startswith(prefix):
                continue
            packet = json.loads(message["content"][len(prefix) :])
            profile = packet.get("role_profile", {})
            if "prompt" in profile:
                profile.pop("prompt")
                message["content"] = prefix + json.dumps(packet, ensure_ascii=False)
                changed = True
        if changed:
            serialized = json.dumps(request, ensure_ascii=False)
            raw = json.loads(raw_json)
            raw["request_display_sanitized"] = True
            raw.setdefault("original_request_hash", original_hash)
            connection.execute(
                "UPDATE attempts SET request_json=?,request_hash=?,raw_json=? WHERE id=?",
                (
                    serialized,
                    hashlib.sha256(serialized.encode()).hexdigest(),
                    json.dumps(raw, ensure_ascii=False),
                    row_id,
                ),
            )
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--results")
    args = ap.parse_args()
    import pyarrow.parquet as pq

    target = Path(args.output)
    if target.exists():
        raise SystemExit("Refuse to overwrite existing index")
    runtime = {
        row["conversation_id"]: row
        for line in Path(args.runtime).read_text(encoding="utf-8").splitlines()
        for row in json.loads(line)["rows"]
    }
    language = {
        row["target_increase_id"]: row["language"]
        for row in pq.read_table(
            args.candidates, columns=["target_increase_id", "language"]
        ).to_pylist()
    }
    src = sqlite3.connect(f"file:{Path(args.input).as_posix()}?mode=ro", uri=True)
    out = sqlite3.connect(target)
    out.executescript(DDL)
    stats = Counter()
    try:
        for seq, blob in src.execute(
            "SELECT seq,payload_zlib FROM conversations ORDER BY seq"
        ):
            packet = json.loads(zlib.decompress(blob))
            if packet["conversation_id"] not in runtime:
                raise ValueError(
                    f"Missing configuration for conversation sequence {seq}"
                )
            for message in packet["messages"]:
                if message["scope"] == "candidate":
                    message["generation_language"] = language.get(
                        message.get("source_increase_id"), packet["language"]
                    )
            packet = assemble_platform_context(
                packet, runtime[packet["conversation_id"]]
            )
            insert_conversation(out, seq, packet)
            stats["conversations"] += 1
            stats["candidates"] += packet["candidate_count"]
            stats["defaulted_settings"] += bool(
                packet["platform_assembly"]["defaulted_settings"]
            )
            for system in packet["system_contexts"]:
                stats[system["template"]] += 1
            if seq % 1000 == 0:
                out.commit()
                print(json.dumps(dict(stats)), flush=True)
        manifest = json.loads(
            src.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0]
        )
        digest = hashlib.sha256()
        for seq, fingerprint in out.execute(
            "SELECT seq,context_hash FROM conversations ORDER BY seq"
        ):
            digest.update(f"{seq}:{fingerprint}\n".encode())
        manifest.update(
            dataset_sha256=digest.hexdigest(),
            platform_assembly_version=CONTEXT_VERSION,
            platform_assembly_report=dict(stats),
            runtime_source_sha256=hashlib.sha256(
                Path(args.runtime).read_bytes()
            ).hexdigest(),
        )
        out.execute(
            "INSERT INTO metadata VALUES ('manifest',?)",
            (json.dumps(manifest, ensure_ascii=False),),
        )
        out.commit()
    finally:
        src.close()
        out.close()
    if args.results:
        with sqlite3.connect(args.results) as results:
            print("cleaned historical requests", clean_attempts(results), flush=True)
    print(
        json.dumps(
            {
                "done": True,
                "stats": dict(stats),
                "dataset_sha256": manifest["dataset_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
