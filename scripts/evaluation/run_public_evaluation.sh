#!/usr/bin/env bash
set -euo pipefail
script_path=$(readlink -f "${BASH_SOURCE[0]}")
repo_root=$(cd "$(dirname "$script_path")/../.." && pwd)
eval_python=${EVALUATION_PYTHON:-$HOME/.cache/chat-model-train/eval-venv/bin/python}
if [[ ! -x "$eval_python" ]]; then
  echo 'Run python3 scripts/evaluation/setup_public_evaluation.py first (or set EVALUATION_PYTHON).' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$eval_python" "$repo_root/scripts/evaluation/run_public_evaluation.py" "$@"
