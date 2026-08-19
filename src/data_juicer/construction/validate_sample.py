#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import orjson


SCHEMA_VERSION = "lusy-rp-dialogue-data-juicer-v2.2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = 0
    ids: set[str] = set()
    with args.path.open("rb") as handle:
        for line in handle:
            if args.limit is not None and rows >= args.limit:
                break
            row = orjson.loads(line)
            rows += 1
            cid = row["conversation_id"]
            if cid in ids:
                raise SystemExit(f"duplicate conversation: {cid}")
            ids.add(cid)
            if row.get("schema_version") != SCHEMA_VERSION:
                raise SystemExit(f"schema mismatch: {cid}")
            messages = row["messages"]
            if messages[0]["role"] != "system":
                raise SystemExit(f"bad system prefix: {cid}")
            memory_mode = row["memory"]["mode"]
            greeting_index = 2 if memory_mode == "effective_memory_tail" else 1
            if messages[greeting_index]["role"] != "assistant":
                raise SystemExit(f"bad greeting role: {cid}")
            if len(messages) != greeting_index + 1 + 2 * row["turn_count"]:
                raise SystemExit(f"message accounting mismatch: {cid}")
            for index, message in enumerate(messages[greeting_index + 1 :]):
                role = "user" if index % 2 == 0 else "assistant"
                if message["role"] != role or not str(message["content"]).strip():
                    raise SystemExit(f"message contract failed: {cid}:{index}")
            contract = row.get("data_juicer_contract") or {}
            if contract.get("primary_field") != "messages":
                raise SystemExit(f"Data-Juicer contract missing: {cid}")
            if contract.get("loss_mask_assigned") is not False:
                raise SystemExit(f"loss mask unexpectedly assigned: {cid}")
            if not row.get("source_policy_audit") or not row.get("lineage"):
                raise SystemExit(f"v2.2 audit metadata missing: {cid}")
            if row["lineage"].get("branch_id") != cid:
                raise SystemExit(f"branch identity mismatch: {cid}")
            excluded = set(row["render_audit"]["excluded_fields"])
            if not {"character_description", "character_prompt"}.issubset(excluded):
                raise SystemExit(f"excluded field audit missing: {cid}")
            if row["render_audit"].get("historical_prompt_exact") is not False:
                raise SystemExit(f"historical prompt contract failed: {cid}")
    if args.expected is not None and rows != args.expected:
        raise SystemExit(f"expected {args.expected}, validated {rows}")
    print(json.dumps({"status": "ok", "validated_rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
