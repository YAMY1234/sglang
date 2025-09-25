# 8B 模型 2GPU Pipeline Parallelism 测试结果存档

## 测试配置
- **模型**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **GPU**: 2张 H100 80GB
- **PP size**: 2
- **数据类型**: bfloat16
- **注意力后端**: triton

## 测试结果

### Case A: CUDA Graphs 开启
- Round 1: TPOT = **0.013123s** (13.1ms), 总时间 = 0.84s
- Round 2: TPOT = **0.004543s** (4.5ms), 总时间 = 0.291s  
- Round 3: TPOT = **0.004455s** (4.5ms), 总时间 = 0.285s

**服务器日志显示**:
- 生成吞吐量: **54.39 token/s**
- CUDA Graph: **True**

### Case B: CUDA Graphs 关闭
- Round 1: TPOT = **0.039065s** (39.1ms), 总时间 = 2.5s
- Round 2: TPOT = **0.007363s** (7.4ms), 总时间 = 0.471s
- Round 3: TPOT = **0.007122s** (7.1ms), 总时间 = 0.456s

**服务器日志显示**:
- 生成吞吐量: **47.27 token/s**
- CUDA Graph: **False**

## 关键发现

### 🎯 **与Issue #10865的差异**
- **我们的结果**: CUDA Graphs开启时性能**更好**（4.5ms vs 7.1ms TPOT）
- **Issue描述**: CUDA Graphs开启时性能**更差**（100ms vs 3ms）

### 🤔 **原因分析**
1. **模型规模差异**: 8B vs 671B参数（DeepSeek-V3）
2. **PP stage数量**: 2 vs 4+
3. **计算密度不足**: 8B模型在2张H100上计算密度不够高，CUDA Graph的优化效果显著
4. **通信开销相对较小**: 小模型的token传输开销不是瓶颈

### 📊 **性能表现**
- **CUDA Graphs带来15%的吞吐量提升**（54.39 vs 47.27 token/s）
- **第一轮包含预热时间**，后续轮次更稳定
- **两种配置都能正常工作**，无明显延迟问题

## 结论

在8B模型 + 2GPU PP配置下，**未能复现Issue #10865中描述的延迟问题**。

需要使用更大的模型（如70B或DeepSeek-V3）和更多GPU来验证真实的问题场景。

## 下一步

1. 使用70B模型 + 4GPU PP配置
2. 如有条件，使用DeepSeek-V3模型
3. 增加batch size和序列长度，提高计算密度 