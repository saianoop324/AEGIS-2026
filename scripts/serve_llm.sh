#!/usr/bin/env bash
# Serve the agent LLM with vLLM as an OpenAI-compatible server with tool calling, the way
# Experiments 10-14 ran Qwen3.5-9B (tool parser qwen3_coder, text-only, thinking off).
#
#   scripts/serve_llm.sh                         # Qwen/Qwen3.5-9B as "qwen3.5-9b" on GPU 0, port 8011
#   GPU=1 PORT=8012 MODEL=Qwen/Qwen3.5-4B NAME=qwen3.5-4b scripts/serve_llm.sh   # write-gate classifier
#
# Runs from the aegis-vllm conda env (scripts/setup_env.sh --with-vllm). Point the agent at it with
# NVIDIA_NIM_API_BASE=http://localhost:$PORT/v1 NVIDIA_NIM_MODEL=$NAME NVIDIA_NIM_API_KEY=local
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
NAME=${NAME:-qwen3.5-9b}
GPU=${GPU:-0}
PORT=${PORT:-8011}
MAX_LEN=${MAX_LEN:-16384}
GPU_UTIL=${GPU_UTIL:-0.85}
VLLM_ENV=${VLLM_ENV:-aegis-vllm}

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
VLLM="$CONDA_BASE/envs/$VLLM_ENV/bin/vllm"
[ -x "$VLLM" ] || { echo "vllm not found at $VLLM; run scripts/setup_env.sh --with-vllm" >&2; exit 1; }

# JIT steps call ninja from the env without activating it. The FlashInfer sampler JIT also needs a
# matching nvcc; off by default (Experiment 2's fix), vLLM then samples with PyTorch.
export PATH="$CONDA_BASE/envs/$VLLM_ENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

CUDA_VISIBLE_DEVICES=$GPU exec "$VLLM" serve "$MODEL" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --language-model-only \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  "$@"
