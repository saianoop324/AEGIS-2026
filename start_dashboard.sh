#!/bin/bash
# ==============================================================================
# MemGuard - Start Streamlit Observability Dashboard
# Automatically activates virtual environment and connects to local vLLM
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

# Connect to local vLLM server by default
export NVIDIA_NIM_API_BASE="http://localhost:8011/v1"
export NVIDIA_NIM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export NVIDIA_NIM_API_KEY="local"

PORT=8501

echo "=========================================================="
echo " 🛡️ Starting MemGuard Observability Dashboard"
echo " Web UI:   http://0.0.0.0:${PORT}"
echo " LLM Host: ${NVIDIA_NIM_API_BASE}"
echo " Model:    ${NVIDIA_NIM_MODEL}"
echo "=========================================================="

exec streamlit run app.py --server.port "${PORT}" --server.address 0.0.0.0

