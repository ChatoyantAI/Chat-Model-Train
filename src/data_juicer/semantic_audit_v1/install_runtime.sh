#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
UV_BIN="${UV_BIN:-/home/gyr/.local/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv is not available at $UV_BIN" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv --python /home/gyr/.local/bin/python3.12 "$VENV_DIR"
fi

"$UV_BIN" pip install --python "$VENV_DIR/bin/python" -r "$SCRIPT_DIR/requirements.txt"
"$VENV_DIR/bin/python" -c 'import numpy, pyarrow, torch, transformers; print(torch.__version__, transformers.__version__, pyarrow.__version__, numpy.__version__)'

