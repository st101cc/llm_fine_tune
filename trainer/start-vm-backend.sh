#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export USE_TF=0
export USE_FLAX=0
export FORGETUNE_DATA_ROOT="$PWD/data"
# Override the VM's shared, non-writable Hugging Face caches before Python imports.
export HF_HOME="$FORGETUNE_DATA_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$FORGETUNE_DATA_ROOT/logs" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
if curl --fail --silent http://127.0.0.1:18000/hardware >/dev/null; then
  echo "Backend already running on 127.0.0.1:18000"
  exit 0
fi
PY=.venv/bin/python
[ -x "$PY" ] || PY=.venv311/bin/python
# Async LangGraph review checkpoints require Python 3.11+ on this deployment.
"$PY" -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ is required'"
nohup "$PY" -m uvicorn app:app --app-dir trainer --host 127.0.0.1 --port 18000 >> "$FORGETUNE_DATA_ROOT/logs/backend.log" 2>&1 < /dev/null &
echo "Started backend PID $!"
