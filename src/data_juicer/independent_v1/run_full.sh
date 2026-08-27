#!/usr/bin/env bash
set -euo pipefail

TOOLS_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SOURCE_DIR:?SOURCE_DIR must point to frozen 01_constructed/dialogues}"
: "${RUN_ROOT:?RUN_ROOT must point to a new versioned run directory}"
: "${DJ_ROOT:?DJ_ROOT must point to the pinned Data-Juicer checkout}"
: "${TOKENIZER_MODEL:?TOKENIZER_MODEL must point to the multilingual SentencePiece model}"

PYTHON="$DJ_ROOT/.venv/bin/python"
DJ_PROCESS="$DJ_ROOT/.venv/bin/dj-process"
STATUS_PATH="$RUN_ROOT/07_dashboard/run_status.json"
DERIVE_WORKERS=${DERIVE_WORKERS:-8}

export CUDA_VISIBLE_DEVICES=
export DATA_JUICER_USE_STDLIB_JSON=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

on_error() {
  local exit_code=$?
  local failed_line=$1
  trap - ERR
  "$PYTHON" "$TOOLS_ROOT/run_status.py" \
    --status-path "$STATUS_PATH" \
    fail --message "pipeline failed at shell line ${failed_line} with exit ${exit_code}" || true
  exit "$exit_code"
}
trap 'on_error "$LINENO"' ERR

mkdir -p \
  "$RUN_ROOT/00_input_manifest/configs" \
  "$RUN_ROOT/00_input_manifest/models" \
  "$RUN_ROOT/01_derived_profile" \
  "$RUN_ROOT/02_exact_dedup" \
  "$RUN_ROOT/03_minhash_branch" \
  "$RUN_ROOT/04_simhash_branch" \
  "$RUN_ROOT/05_repetition_profile" \
  "$RUN_ROOT/06_candidate_samples" \
  "$RUN_ROOT/07_dashboard"

if [[ ! -s "$STATUS_PATH" ]]; then
  "$PYTHON" "$TOOLS_ROOT/run_status.py" --status-path "$STATUS_PATH" init --expected-buckets 256
fi

if [[ ! -e "$RUN_ROOT/00_input_manifest/models/xlm-roberta-base-sentencepiece.bpe.model" ]]; then
  ln -s "$TOKENIZER_MODEL" "$RUN_ROOT/00_input_manifest/models/xlm-roberta-base-sentencepiece.bpe.model"
fi

for config in "$TOOLS_ROOT"/configs/*.yaml; do
  target="$RUN_ROOT/00_input_manifest/configs/$(basename "$config")"
  if [[ ! -e "$target" ]]; then
    cp "$config" "$target"
  fi
done

if [[ -n "${SOURCE_MANIFEST:-}" && -s "${SOURCE_MANIFEST}" && ! -e "$RUN_ROOT/00_input_manifest/CONSTRUCTION_MANIFEST.json" ]]; then
  cp "$SOURCE_MANIFEST" "$RUN_ROOT/00_input_manifest/CONSTRUCTION_MANIFEST.json"
fi

if [[ ! -e "$RUN_ROOT/00_input_manifest/IMPLEMENTATION_SHA256SUMS.txt" ]]; then
  sha256sum "$TOOLS_ROOT"/*.py "$TOOLS_ROOT"/run_full.sh "$TOOLS_ROOT"/configs/*.yaml \
    > "$RUN_ROOT/00_input_manifest/IMPLEMENTATION_SHA256SUMS.txt"
fi

cd "$RUN_ROOT"

"$PYTHON" "$TOOLS_ROOT/run_status.py" \
  --status-path "$STATUS_PATH" stage derive_views running --message "reading frozen buckets; fork metadata is ignored"
"$PYTHON" "$TOOLS_ROOT/derive_views.py" \
  --source-dir "$SOURCE_DIR" \
  --run-root "$RUN_ROOT" \
  --workers "$DERIVE_WORKERS" \
  --expected-buckets 256
derived_rows=$("$PYTHON" "$TOOLS_ROOT/artifact_info.py" "$RUN_ROOT/01_derived_profile/parquet")
"$PYTHON" "$TOOLS_ROOT/run_status.py" \
  --status-path "$STATUS_PATH" stage derive_views completed --output-records "$derived_rows"
"$PYTHON" "$TOOLS_ROOT/summarize_run.py" --run-root "$RUN_ROOT"

run_data_juicer_stage() {
  local stage_name=$1
  local config_name=$2
  local output_path=$3
  local input_rows=$4
  local output_rows

  if output_rows=$("$PYTHON" "$TOOLS_ROOT/artifact_info.py" "$output_path" 2>/dev/null); then
    "$PYTHON" "$TOOLS_ROOT/run_status.py" \
      --status-path "$STATUS_PATH" stage "$stage_name" completed \
      --input-records "$input_rows" --output-records "$output_rows" --message "resumed existing valid Parquet artifact"
  else
    "$PYTHON" "$TOOLS_ROOT/run_status.py" \
      --status-path "$STATUS_PATH" stage "$stage_name" running --input-records "$input_rows"
    "$DJ_PROCESS" --config "$RUN_ROOT/00_input_manifest/configs/$config_name"
    output_rows=$("$PYTHON" "$TOOLS_ROOT/artifact_info.py" "$output_path")
    "$PYTHON" "$TOOLS_ROOT/run_status.py" \
      --status-path "$STATUS_PATH" stage "$stage_name" completed \
      --input-records "$input_rows" --output-records "$output_rows"
  fi
  STAGE_OUTPUT_ROWS=$output_rows
}

run_data_juicer_stage \
  exact 10_exact.yaml "$RUN_ROOT/02_exact_dedup/retained.parquet" "$derived_rows"
exact_rows=$STAGE_OUTPUT_ROWS
"$PYTHON" "$TOOLS_ROOT/summarize_run.py" --run-root "$RUN_ROOT"

run_data_juicer_stage \
  minhash 20_minhash.yaml "$RUN_ROOT/03_minhash_branch/retained.parquet" "$exact_rows"
minhash_rows=$STAGE_OUTPUT_ROWS
"$PYTHON" "$TOOLS_ROOT/summarize_run.py" --run-root "$RUN_ROOT"

run_data_juicer_stage \
  simhash 30_simhash.yaml "$RUN_ROOT/04_simhash_branch/retained.parquet" "$exact_rows"
simhash_rows=$STAGE_OUTPUT_ROWS
"$PYTHON" "$TOOLS_ROOT/summarize_run.py" --run-root "$RUN_ROOT"

run_data_juicer_stage \
  repetition 40_repetition.yaml "$RUN_ROOT/05_repetition_profile/scored.parquet" "$exact_rows"
repetition_rows=$STAGE_OUTPUT_ROWS

"$PYTHON" "$TOOLS_ROOT/run_status.py" \
  --status-path "$STATUS_PATH" stage summarize running \
  --input-records "$repetition_rows" --message "building compact dashboard evidence"
"$PYTHON" "$TOOLS_ROOT/summarize_run.py" --run-root "$RUN_ROOT"
"$PYTHON" "$TOOLS_ROOT/run_status.py" \
  --status-path "$STATUS_PATH" stage summarize completed --output-records "$repetition_rows"
"$PYTHON" "$TOOLS_ROOT/run_status.py" --status-path "$STATUS_PATH" complete
