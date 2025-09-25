#!/usr/bin/env bash
echo "=== GPU Information ==="
nvidia-smi || true
echo ""
echo "=== Python Version ==="
python3 -V || true
echo ""
echo "=== PyTorch & SGLang Version ==="
python3 -c "import torch,sglang; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('sglang', sglang.__version__)" || true
echo ""
echo "=== Environment Variables ==="
env | egrep 'NCCL|CUDA|CU|TORCH|SGLANG' || true
echo ""
echo "=== System Info ==="
uname -a || true
echo ""
echo "=== Available GPUs ==="
python3 -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" || true 