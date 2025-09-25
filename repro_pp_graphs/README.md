# DeepSeek-V3 Pipeline Parallelism CUDA Graphs 延迟问题复现脚本

## 问题描述

基于 [GitHub Issue #10865](https://github.com/sgl-project/sglang/issues/10865)，此脚本用于复现 DeepSeek-V3 在使用 Pipeline Parallelism (PP) 时，开启 CUDA Graphs 导致的异常延迟问题：

- **Decode + CUDA Graphs**: 最后一个PP rank向PP0发送token ID耗时 ~100ms
- **Prefill/关闭CUDA Graphs**: 同样的发送操作仅耗时 ~3ms
- 这会显著影响 TPOT (Time Per Output Token) 和整体吞吐量

## 脚本结构

```
repro_pp_graphs/
├─ run_repro.sh                 # 主执行脚本
├─ server_args_base.txt         # 基础服务器参数
├─ server_args_graphs_on.txt    # 开启CUDA Graphs配置（问题态）
├─ server_args_graphs_off.txt   # 关闭CUDA Graphs配置（对照组）
├─ client_benchmark.py          # 客户端压测脚本
├─ prompts.txt                  # 测试提示词
├─ collect_env.sh              # 环境信息收集
└─ README.md                   # 本说明文件
```

## 环境要求

- **GPU**: 4×H100 (或调整为2×H100，修改 `--pipeline-parallel-size 2`)
- **模型**: Llama-3.1-70B (可替换为DeepSeek-V3)
- **SGLang**: 支持Pipeline Parallelism和CUDA Graphs的版本
- **依赖**: PyTorch, requests, pathlib

## 使用方法

### 1. 运行完整A/B测试

```bash
cd /sgl-workspace/sglang/repro_pp_graphs
bash run_repro.sh
```

### 2. 查看结果

测试完成后会生成：

- **日志文件**: `logs/` 目录
  - `CASE_A_decode_graphs_server.log` - CUDA Graphs开启时的服务器日志
  - `CASE_A_decode_graphs_client.log` - 对应的客户端测试结果
  - `CASE_B_no_graphs_server.log` - CUDA Graphs关闭时的服务器日志  
  - `CASE_B_no_graphs_client.log` - 对应的客户端测试结果
  - `env.txt` - 环境信息

- **性能追踪**: `trace/` 目录
  - `CASE_A_decode_graphs_chrome_trace.json` - CUDA Graphs场景的Chrome trace
  - `CASE_B_no_graphs_chrome_trace.json` - 无CUDA Graphs场景的Chrome trace

### 3. 分析性能追踪

使用 [ui.perfetto.dev](https://ui.perfetto.dev) 打开生成的 JSON 文件：

1. 在搜索框输入 `cudagraph` 定位decode阶段
2. 观察 `send_tensor_dict` 或类似通信函数的耗时
3. 对比两个场景的差异

**预期结果**:
- Case A (CUDA Graphs): ~100ms 发送延迟 (含~30ms与decode重叠，~70ms空转)
- Case B (无CUDA Graphs): ~3ms 发送延迟

## 配置调整

### 使用2张GPU

修改 `server_args_base.txt`:
```
--pipeline-parallel-size 2
```

### 使用DeepSeek-V3

修改 `server_args_base.txt`:
```
--model-path deepseek-ai/DeepSeek-V3
```

### 使用其他模型

如果70B模型太大，可以使用：
- Mixtral-8x7B (MoE模型，活跃参数较少)
- Llama-3.1-8B (调整为单GPU测试)

### 调整压测参数

修改 `client_benchmark.py` 中的默认参数：
- `--batch`: 批次大小
- `--max_new_tokens`: 生成token数量

## 故障排除

### 1. 服务器启动失败

检查 `logs/*_server.log` 中的错误信息：
- 显存不足：减少 `--mem-fraction-static` 或使用更小模型
- 模型下载失败：确保网络连接和HuggingFace访问

### 2. 客户端连接失败

- 检查端口是否被占用
- 增加服务器启动等待时间

### 3. 性能追踪文件为空

确保设置了环境变量：
```bash
export SGLANG_ENABLE_TORCH_PROFILER=1
```

## 后续优化验证

基于此脚本可以进一步验证优化方案：

1. **独立通信流**: 将最终发送移到graph回放之外
2. **事件同步**: 使用CUDA Events替代全局同步
3. **批量通信**: 使用 `batch_isend_irecv` 减少启动开销

## 环境检查

运行环境检查脚本：
```bash
bash collect_env.sh
```

这会输出GPU信息、Python版本、PyTorch/SGLang版本等关键信息。

## 支持的扩展

如需要更详细的CUDA分析，可以添加 nsys 版本：
```bash
nsys profile -o trace/nsys_profile.qdrep python3 -m sglang.launch_server ...
```

这将生成 `.qdrep` 文件，可以直接查看CUDA Graph与NCCL的流/事件依赖关系。 