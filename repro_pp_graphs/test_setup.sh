#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== DeepSeek-V3 PP CUDA Graphs 问题复现环境测试 ==="
echo ""

# 1. 检查环境
echo "1. 检查环境..."
bash "$ROOT/collect_env.sh" | head -20
echo ""

# 2. 检查模型路径设置
echo "2. 检查配置文件..."
echo "基础配置:"
cat "$ROOT/server_args_base.txt"
echo ""
echo "CUDA Graphs开启配置:"
cat "$ROOT/server_args_graphs_on.txt" || echo "（空配置 - 默认开启）"
echo ""
echo "CUDA Graphs关闭配置:"
cat "$ROOT/server_args_graphs_off.txt"
echo ""

# 3. 检查模型是否存在（不下载，只检查）
echo "3. 检查模型可用性..."
MODEL_PATH=$(grep "model-path" "$ROOT/server_args_base.txt" | cut -d' ' -f2)
echo "配置的模型: $MODEL_PATH"

python3 -c "
import os
from transformers import AutoTokenizer
try:
    # 只检查tokenizer，不下载完整模型
    tokenizer = AutoTokenizer.from_pretrained('$MODEL_PATH', cache_dir=os.path.expanduser('~/.cache/huggingface'))
    print('✓ 模型可访问')
except Exception as e:
    print(f'⚠️ 模型访问问题: {e}')
    print('建议: 确保网络连接正常或模型已下载到本地')
" || echo "⚠️ 无法验证模型，请确保transformers库已安装"

echo ""

# 4. 检查端口可用性
echo "4. 检查端口可用性..."
for port in 8000 8001; do
    if ! nc -z 127.0.0.1 $port 2>/dev/null; then
        echo "✓ 端口 $port 可用"
    else
        echo "⚠️ 端口 $port 被占用"
    fi
done
echo ""

# 5. 检查依赖
echo "5. 检查Python依赖..."
python3 -c "
import sys
deps = ['torch', 'sglang', 'requests', 'pathlib']
missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f'✓ {dep}')
    except ImportError:
        missing.append(dep)
        print(f'✗ {dep}')

if missing:
    print(f'缺少依赖: {missing}')
    sys.exit(1)
else:
    print('所有依赖已满足')
"
echo ""

# 6. 显存检查
echo "6. 显存检查..."
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(min(4, torch.cuda.device_count())):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f'GPU {i}: {props.name}, {memory_gb:.1f}GB')
    
    if torch.cuda.device_count() >= 4:
        print('✓ 有足够的GPU运行4-way PP')
    elif torch.cuda.device_count() >= 2:
        print('⚠️ 只有2张GPU，建议修改配置为 --pipeline-parallel-size 2')
    else:
        print('✗ GPU数量不足')
else:
    print('✗ CUDA不可用')
"
echo ""

echo "=== 环境检查完成 ==="
echo ""
echo "下一步:"
echo "1. 如果所有检查都通过，运行: bash run_repro.sh"
echo "2. 如果需要nsys深度分析，运行: bash run_repro_nsys.sh"
echo "3. 如果GPU不足4张，修改 server_args_base.txt 中的 --pipeline-parallel-size"
echo "4. 如果模型太大，可以换成更小的模型如 meta-llama/Meta-Llama-3.1-8B-Instruct" 