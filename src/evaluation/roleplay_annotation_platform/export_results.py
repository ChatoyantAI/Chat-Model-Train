from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def export_results(database: Path, output: Path, username: str | None = None) -> int:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    query = (
        "SELECT username, seq, evaluation_id, rubric_version, status, gates_json, scores_json, "
        "gate_notes, overall_notes, created_at, updated_at FROM annotations"
    )
    parameters: tuple[str, ...] = ()
    if username:
        query += " WHERE username = ?"
        parameters = (username,)
    query += " ORDER BY username, seq"

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in connection.execute(query, parameters):
            record = dict(row)
            record["gates"] = json.loads(record.pop("gates_json"))
            record["scores"] = json.loads(record.pop("scores_json"))
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    connection.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blind annotation results to JSONL")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--username")
    args = parser.parse_args()
    count = export_results(args.database, args.output, args.username)
    print(f"exported {count} records to {args.output}")


if __name__ == "__main__":
    main()
