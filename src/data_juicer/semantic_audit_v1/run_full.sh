#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816}"
SOURCE_DIR="${SOURCE_DIR:-$DATASET_ROOT/01_constructed/dialogues}"
INDEPENDENT_ROOT="${INDEPENDENT_ROOT:-$DATASET_ROOT/02_data_juicer/01_independent_v1}"
RUN_ROOT="${RUN_ROOT:-$DATASET_ROOT/02_data_juicer/02_semantic_audit_v1}"
TURN_DIR="$RUN_ROOT/01_turn_view"
STATUS_PATH="$RUN_ROOT/07_dashboard/run_status.json"
BENCHMARK_PATH="$RUN_ROOT/07_dashboard/embedding_benchmark.json"
MODEL="${MODEL:-BAAI/bge-m3}"
REVISION="${REVISION:-5617a9f61b028005a4858fdac845db406aefb181}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DERIVE_WORKERS="${DERIVE_WORKERS:-8}"

mkdir -p "$RUN_ROOT/07_dashboard"

record_failure() {
  "$PYTHON" "$SCRIPT_DIR/set_status.py" \
    --path "$STATUS_PATH" --status failed --stage failed \
    --message "semantic embedding pipeline exited with an error"
}
trap record_failure ERR

"$PYTHON" "$SCRIPT_DIR/derive_turns.py" \
  --source-dir "$SOURCE_DIR" \
  --retained-parquet "$INDEPENDENT_ROOT/02_exact_dedup/retained.parquet" \
  --output-dir "$TURN_DIR" \
  --status-path "$STATUS_PATH" \
  --workers "$DERIVE_WORKERS"

if [[ ! -f "$BENCHMARK_PATH" ]]; then
  "$PYTHON" "$SCRIPT_DIR/benchmark_embeddings.py" \
    --turn-shard "$TURN_DIR/bucket-000.parquet" \
    --output "$BENCHMARK_PATH" \
    --status-path "$STATUS_PATH" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --device cuda:0 \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --sample-size 2048
fi

"$PYTHON" "$SCRIPT_DIR/embed_turns.py" \
  --turn-dir "$TURN_DIR" \
  --output-root "$RUN_ROOT" \
  --status-path "$STATUS_PATH" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --device cuda:0 \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --fields assistant,user

trap - ERR

