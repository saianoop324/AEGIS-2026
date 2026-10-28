#!/bin/bash
# ==============================================================================
# MemGuard - Start & Serve LLM (Qwen2.5-7B-Instruct) via vLLM
# Automatically activates virtual environment and configures GPU inference
# ==============================================================================

set -e

# Automatically activate venv if present and not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        echo "🔄 Activating virtual environment (venv)..."
        source venv/bin/activate
    elif [ -f "../venv/bin/activate" ]; then
        echo "🔄 Activating virtual environment (../venv)..."
        source ../venv/bin/activate
    fi
fi

# Model Configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8011
GPU_UTIL=0.75
MAX_LEN=4096

# Optimization & Compatibility Flags
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=========================================================="
echo " 🚀 Starting MemGuard Local LLM Server"
echo " Model:    ${MODEL}"
echo " Port:     http://0.0.0.0:${PORT}"
echo " GPU Util: ${GPU_UTIL} | Max Tokens: ${MAX_LEN}"
echo "=========================================================="

exec vllm serve "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name "${MODEL}" meta/llama-3.1-70b-instruct

