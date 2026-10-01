# DUET on SGLang（SEED 集成仓库）

## 这是什么

本仓库是 SGLang 的私有集成拷贝，承载 **DUET / SEED** 在 SGLang 上的服务实现：一个模型无关的公共层（`python/sglang/srt/duet/`：spec 契约、发布件取件与校验、开关与数值档位、残差码、状态截秩、slot 侧车）加三个模型适配器（`python/sglang/srt/models/lightning_duet/`、`kimi_linear_duet/`、`flash_next_duet/`）。任何 DUET/SEED 发布件（`spec.json` + `manifest.json` + `duet_components.safetensors`）在这里以**一条命令**起服务，默认生产数值档，用同一套守卫与评测验收。设计文档：twinstar 仓库 `docs/168-duet-in-sglang-design-v1-20260930.md`；验证手册：`docs/165-duet-ckpt-handbook-20260930.md`。

- `main` = 集成分支（原 fork `YAMY1234/sglang` 的 `line/duet-sgl-unified-20260930`）。三模型一条命令起服务并过守卫后切出稳定分支 `duet-main` 对外。
- 评测贡献放 `benchmark/duet_eval/`（端点评测、单一判分器、配对统计、NLL 窗采集）。

## 怎么起服务（一条命令）

```bash
python -m sglang.launch_server --model-path <基座目录> --duet-release <HF repo[@revision] | 本地发布目录> [--duet-numerics {production,reference}]
```

- `--duet-release` 是主开关：发布件被钉 commit 取件（只取三文件、拒绝符号链接），`spec.json` 的 `model` 决定适配器，张量契约由 spec + 基座 config 推导并逐张量校验。不给 `--duet-release` = 原生模型类，对 stock 逐位。
- `--duet-numerics production`（缺省）= 生产档；`reference` = 守卫 / 评测用的精度档（fp32 emitter、exact prefix、warm 截秩、eager）。适配器尚未验证过的档位**启动即拒绝**；资格测试用 `--duet-allow-unvalidated-profile`（env `SGLANG_DUET_ALLOW_UNVALIDATED=1`）显式越权，`/server_info.duet.profile_validated=false`。
- 其它开关（均有 `SGLANG_DUET_*` 环境变量对应）：`--prefill-layer-trim/--no-prefill-layer-trim`、`--prefill-saving-policy {latent-only,latent-and-kv,latent-and-ssm,kv-and-ssm}`、`--decode-ssm-r N`、`--decode-ssm-w N`（0 = 精确状态 / 只剪一次）、`--duet-emitter-precision {fp32,bf16}`、`--duet-code-precision {fp32,tf32}`、`--duet-prefix-state {exact,factored}`。
- 自检：`GET /server_info` 的 `duet` 键回显发布目录、spec 摘要、适配器包与是否 in-tree、数值档位与各开关生效值。
- 示例（Lightning，精度档）：

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE= python -m sglang.launch_server \
  --model-path /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --duet-release mamingyuan2001/<lightning-duet-release> --duet-numerics reference \
  --tp-size 1 --dtype bfloat16 --trust-remote-code --disable-cuda-graph --disable-overlap-schedule \
  --disable-radix-cache --chunked-prefill-size -1
```

## 怎么提 PR 与分支命名

- 只接 PR，不直推 `main`（个人私有仓无法设分支保护，靠约定）。PR 目标：`main`；稳定分支 `duet-main` 切出后只接 PR。
- 分支命名：`line/<线名>-<日期>`（如 `line/infork-K-20261001`、`line/duet-infork-p4-lightning-20260930`）；明远的评测贡献 `line/duet-eval-<日期>`，目标 `main`。
- 每个 PR：描述写清改动范围（公共层 / 哪个适配器 / benchmark）、CPU 测试结果（`test/registered/unit/duet/` 从该目录 `python -m unittest <module>`）、GPU 守卫结果（若有）；不带 Generated / Co-Authored 尾注。
- 公共层 `python/sglang/srt/duet/**` 的接口改动先在 PR 里说明影响的适配器；适配器包只改自己的目录；`arg_groups/`、`entrypoints/` 等通用代码不得 import 任何模型适配器（经公共层的 `adapters.run_resolution_hook` / `adapters.describe` 扩展点分发）。
- 评审归属见 `CODEOWNERS`。
