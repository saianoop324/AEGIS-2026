#!/usr/bin/env bash
# Build the MemGuard environments with conda.
#
#   scripts/setup_env.sh                    # aegis env: core + tests (CPU or GPU)
#   scripts/setup_env.sh --with-baselines   # also NeMo Guardrails, datasets, garak (Experiments 13-15)
#   scripts/setup_env.sh --with-vllm        # also the aegis-vllm env that serves the agent LLM
#
# TORCH_INDEX picks the torch wheel build for aegis; the default cu128 runs on NVIDIA drivers >= 570.
# Set TORCH_INDEX=https://download.pytorch.org/whl/cpu on a machine without a GPU.
# VLLM_CUDA picks the vllm build for aegis-vllm: cu129 (default, drivers >= 525) or cu130 (>= 580).
set -euo pipefail

ENV=${ENV:-aegis}
VLLM_ENV=${VLLM_ENV:-aegis-vllm}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
VLLM_CUDA=${VLLM_CUDA:-cu129}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

WITH_BASELINES=0 WITH_VLLM=0
for arg in "$@"; do
  case $arg in
    --with-baselines) WITH_BASELINES=1 ;;
    --with-vllm) WITH_VLLM=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

CONDA_BASE=$(conda info --base)
env_exists() { [ -x "$CONDA_BASE/envs/$1/bin/python" ]; }

env_exists "$ENV" || conda create -y -n "$ENV" python=3.11
PY="$CONDA_BASE/envs/$ENV/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch --index-url "$TORCH_INDEX"
"$PY" -m pip install -r "$ROOT/requirements-dev.txt"
if [ $WITH_BASELINES = 1 ]; then "$PY" -m pip install -r "$ROOT/requirements-baselines.txt"; fi

# fetch the embedding and NLI models once, so the first gate call does not stall on a download
cd "$ROOT"
"$PY" - <<'EOF'
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import settings
SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
AutoTokenizer.from_pretrained(settings.DEBERTA_NLI_MODEL_NAME)
AutoModelForSequenceClassification.from_pretrained(settings.DEBERTA_NLI_MODEL_NAME)
print("models cached")
EOF

if [ $WITH_VLLM = 1 ]; then
  env_exists "$VLLM_ENV" || conda create -y -n "$VLLM_ENV" python=3.12
  VPY="$CONDA_BASE/envs/$VLLM_ENV/bin/python"
  "$VPY" -m pip install --upgrade pip
  if [ "$VLLM_CUDA" = cu130 ]; then
    "$VPY" -m pip install vllm==0.32.0
  else
    # PyPI's vllm 0.32.0 is built for CUDA 13.0 (driver >= 580); this build runs on 12.x drivers (>= 525)
    # vllm 0.32.0 pins these three; all must come from the same CUDA build or torchvision ops fail to load
    "$VPY" -m pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
      --index-url "https://download.pytorch.org/whl/$VLLM_CUDA"
    "$VPY" -m pip install "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+$VLLM_CUDA-cp38-abi3-manylinux_2_28_x86_64.whl" \
      --extra-index-url "https://download.pytorch.org/whl/$VLLM_CUDA"
    # PyPI's torchcodec is the CUDA 13 build and fails to import without libnvrtc.so.13; the agent
    # only sends text, so the CPU build is enough
    "$VPY" -m pip install --force-reinstall --no-deps torchcodec==0.17.0 --index-url https://download.pytorch.org/whl/cpu
  fi
fi

[ -f "$ROOT/.env" ] || echo "Next: cp .env.example .env and edit it."
echo "Done. Run the tests with: conda activate $ENV && python -m pytest tests/ -q"
