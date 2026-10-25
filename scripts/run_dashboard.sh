#!/usr/bin/env bash
# Start the MemGuard dashboard for viewing (from VS Code Remote-SSH: forward the port in the Ports tab).
#
#   scripts/run_dashboard.sh                 # http://localhost:8501, models on CPU
#   GPU=3 PORT=8502 scripts/run_dashboard.sh # models on GPU 3
#
# The file watcher is off: it imports every loaded module to find files to watch, which floods the log
# with transformers' optional torchvision imports. Restart the script after editing the code.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PORT=${PORT:-8501}
ENV=${ENV:-aegis}
STREAMLIT="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/envs/$ENV/bin/streamlit"
[ -x "$STREAMLIT" ] || { echo "streamlit not found at $STREAMLIT; run scripts/setup_env.sh" >&2; exit 1; }
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPU DEVICE=cuda; fi
cd "$ROOT"
exec "$STREAMLIT" run app.py --server.headless true --server.port "$PORT" --server.fileWatcherType none \
  --browser.gatherUsageStats false "$@"
