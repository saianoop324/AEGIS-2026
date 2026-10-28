#!/bin/bash
# ==============================================================================
# MemGuard - Automated Dependency Installer & Environment Verification
# ==============================================================================

set -e

echo "=========================================================="
echo " 1. Upgrading Pip..."
echo "=========================================================="
python3 -m pip install --upgrade pip setuptools wheel

echo "=========================================================="
echo " 2. Installing Dependencies from requirements.txt..."
echo "=========================================================="
pip install -r requirements.txt

echo "=========================================================="
echo " 3. Running Pre-Flight CUDA & Engine Verification..."
echo "=========================================================="
python3 -c "
import torch
import transformers
import vllm

cuda_ok = torch.cuda.is_available()
device = torch.cuda.get_device_name(0) if cuda_ok else 'CPU (No GPU detected)'

print(f'✅ PyTorch:      {torch.__version__} (CUDA Available: {cuda_ok} - {device})')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ vLLM Engine:  {vllm.__version__}')

if not cuda_ok:
    print('⚠️ WARNING: CUDA is not available. Check your NVIDIA drivers.')
else:
    print('🚀 Environment is 100% verified and ready!')
"

