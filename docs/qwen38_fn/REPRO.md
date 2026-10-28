# Qwen3.8 Flash Next (FN) on GB300: TP4 c1 decode, reproduction guide

This guide reproduces the per-user decode speed of Qwen3.8 Flash Next (NVFP4) on GB300 NVL72 at tensor parallel 4 and concurrency 1, comparing upstream SGLang `main` against our optimized branch. All numbers below are measured, not projected.

## 1. Code

| Role | Repository / branch | Commit |
|---|---|---|
| Baseline | `sgl-project/sglang` main | [`449617bc94a5`](https://github.com/sgl-project/sglang/commit/449617bc94a5d4ae23ad62d0da2f842e711ca983) |
| Measured candidate ("757f") | `YAMY1234/sglang` | [`757f1b9c7a1d`](https://github.com/YAMY1234/sglang/commit/757f1b9c7a1d16ae2a378975bb3cfea909b312a1) |
| Current parent ("2a92" = 757f + QSA attention core) | `YAMY1234/sglang` branch `perf/q38cc/base-tp4` | [`2a92c89d22d6`](https://github.com/YAMY1234/sglang/commit/2a92c89d22d660197949fc36e6f4fb6b66332e14) |
| 2a92 rebased onto newer main `4517a5ff1bb` | `YAMY1234/sglang` branch `qwen38-fn-mega` | `0a79c70fe0f4` |
| FN on the NVIDIA megakernel plugin (in progress, not benchmarked yet) | `YAMY1234/sgl-nvidia-mega-fork` branch `qwen38-fn-mega` | `e67057406c3a` |

Compare views:
- SGLang, current parent vs main: https://github.com/sgl-project/sglang/compare/main...YAMY1234:sglang:perf/q38cc/base-tp4
- SGLang, rebased tree vs main: https://github.com/sgl-project/sglang/compare/main...YAMY1234:sglang:qwen38-fn-mega
- Plugin: https://github.com/sgl-project/sgl-nvidia-mega/compare/main...YAMY1234:sgl-nvidia-mega-fork:qwen38-fn-mega

## 2. Environment

- Hardware: GB300 NVL72, one 4-GPU tray per server (TP4), concurrency 1.
- Container: `lmsysorg/sglang:nightly-dev-cu13-20261015-a6a8c560` (the image both arms were measured on). The SGLang source is mounted and put first on `PYTHONPATH`; the image's own SGLang is not used.
- Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` (target and MTP draft). The checkpoint quantizes the target routed experts to NVFP4, the MTP experts to block FP8 and the PLE embedding to FP8; everything else is BF16.
- KV cache: BF16. Speculative decoding: MTP (NEXTN), 3 draft steps, top-k 1, 4 verify tokens per request.

## 3. Acceptance length

All performance numbers in this guide use a **fixed simulated acceptance length of 2.32** on both arms, so that decode speed does not depend on prompt content:

```bash
export SGLANG_SIMULATE_ACC_LEN=2.32
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
```

Unset these three variables for natural acceptance (and always for accuracy runs). On GSM8K the natural acceptance length of both the candidate and its parent is about 3.20, higher than 2.32; AgentX with natural acceptance has not been measured yet.

## 4. Launch the server (identical arguments for both arms)

```bash
python3 -m sglang.launch_server \
  --served-model-name nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --speculative-draft-model-revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --trust-remote-code \
  --quantization modelopt_mixed \
  --context-length 262144 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --expert-parallel-size 1 \
  --attention-backend trtllm_mha \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --linear-attn-verify-backend triton \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-track-interval 256 \
  --max-mamba-cache-size 480 \
  --mem-fraction-static 0.88 \
  --max-running-requests 96 \
  --page-size 64 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --moe-runner-backend flashinfer_trtllm \
  --moe-a2a-backend none \
  --disable-shared-experts-fusion \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-moe-runner-backend flashinfer_trtllm \
  --enable-linear-replayssm-spec \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 96 \
  --cuda-graph-bs-decode 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 \
  --kv-cache-dtype bfloat16 \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --ple-offload-embedding \
  --ple-offload-backend pinned \
  --enable-metrics \
  --enable-cache-report \
  --reasoning-parser auto \
  --tokenizer-worker-num 1 \
  --scheduler-recv-interval 1 \
  --decode-log-interval 10 \
  --stream-interval 20 \
  --watchdog-timeout 1800 \
  --random-seed 35 \
  --model-path $MODEL_PATH \
  --host 0.0.0.0 \
  --port 30000
```

### Environment, baseline (main)

```bash
# main 449617bc94a5 imports flash_mla, which the 10-15 image lacks: install sgl-flash-mla 1.0.0
# into a side directory and put it on PYTHONPATH after the checkout
pip install --no-deps --target $DEPS_DIR sgl-flash-mla==1.0.0
export PYTHONPATH=$SGLANG_REPO/python:$DEPS_DIR   # checkout of main 449617bc94a5
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 NO_COLOR=1
export SGLANG_TIMEOUT_KEEP_ALIVE=1800
export SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=0
export SGLANG_SIMULATE_ACC_LEN=2.32 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
```

Note: the measured baseline arm pinned the Python radix-tree core (last line); the candidate arm left it at its default. Besides `sgl-flash-mla` for the baseline, this is the only environment difference beyond the candidate's own switches below.

### Environment, candidate (2a92; 757f is identical without `SGLANG_Q38C_ATTN_CORE_TP4=1`)

```bash
export AUTO_NUMA_BIND=0
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NO_COLOR=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SGLANG_REPO/python
export PYTHONUNBUFFERED=1
export Q38_ATTN_CORE_V1=1
export Q38_ATTN_GATE_V2=1
export Q38_GDN_COMMIT_BV32=1
export Q38_GDN_FRONT=1
export Q38_GDN_INPROJ_CONV=1
export Q38_GDN_NORM_EARLY=1
export Q38_GDN_OUTPROJ_BATCHSPLIT=1
export Q38_GDN_PDL=1
export Q38_GDN_STATE_PREFETCH=1
export Q38_MTP_FULL_INPUT_FOLD=1
export Q38_MTP_POSTVERIFY_FUSED_ACCEPT=1
export Q38_MTP_STEP_INPUT_FOLD=1
export Q38_QSA_EARLY_INDEXER=0
export Q38_QSA_PDL_INDEX=1
export Q38_SHARED_OUT=1
export SGLANG_AUTO_NUMA_BIND=0
export SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH=1
export SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND=1
export SGLANG_ENABLE_METADATA_GLUE_GRAPH=0
export SGLANG_ENABLE_PP_SPEC=0
export SGLANG_FLASHINFER_AUTOTUNE_CACHE=1
export SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=0
export SGLANG_PERFLL_GLUE_META_SPLIT=1
export SGLANG_PERFLL_HOST_D2H_RING=1
export SGLANG_PERFLL_HOST_DECODE_ALLOC=1
export SGLANG_PERFLL_HOST_ENV_FREEZE=1
export SGLANG_PERFLL_HOST_FIELD_COPY=1
export SGLANG_PERFLL_HOST_FILL_PLAN=1
export SGLANG_PERFLL_HOST_GRAPH_INPUTS=1
export SGLANG_PERFLL_HOST_HC_PIGGYBACK=1
export SGLANG_PERFLL_HOST_INDEX_OPS=1
export SGLANG_PERFLL_HOST_JIT_INDEX=1
export SGLANG_PERFLL_HOST_LINK_SPIN=1
export SGLANG_PERFLL_HOST_OUT_POOL=1
export SGLANG_PERFLL_HOST_PIN_MAIN=1
export SGLANG_PERFLL_HOST_RESULT_COPY=1
export SGLANG_PERFLL_HOST_SAMPLING_COPY=1
export SGLANG_PERFLL_HOST_SB_SNAPSHOT=1
export SGLANG_PERFLL_HOST_SIM_PREDRAW=1
export SGLANG_PERFLL_HOST_WAIT_EVENTS=1
export SGLANG_PERFLL_HOST_ZMQ_EVENTS=1
export SGLANG_PERFLL_STEP2=3
export SGLANG_PERFLL_STEP5A=1
export SGLANG_PERFLL_STEP5A_BONUS_I64=1
export SGLANG_PERFLL_STEP5A_DRAFT_LOCS=1
export SGLANG_PERFLL_STEP5A_SEQ_LENS=1
export SGLANG_PERFPF_GC_LOG_MS=20
export SGLANG_PERFPF_GDN_GRAPH_FLASHINFER=1
export SGLANG_PLUGINS=q38_shared_down_tgv
export SGLANG_PST_SPEC_EXTEND_LAYOUT=1
export SGLANG_Q35MF_BK_VEC_COPY=1
export SGLANG_Q35MF_COMMIT_CUDA=1
export SGLANG_Q35MF_DRAFT_SHARED_METADATA=1
export SGLANG_Q35MF_EDGE_CHAIN_TREE=1
export SGLANG_Q35MF_EDGE_PACKED_VERIFY_BCAST=1
export SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING=1
export SGLANG_Q35MF_FUSED_ACCEPT=1
export SGLANG_Q35MF_MKD_XCHG=0
export SGLANG_Q35MF_MKH_N3B=1
export SGLANG_Q35MF_MKH_N3C=1
export SGLANG_Q35MF_SPEC_RELAY_BYPASS=1
export SGLANG_Q38C_ATTN_CORE_TP4=1
export SGLANG_Q38C_GDN_FRONT12=1
export SGLANG_Q38C_HC_CLUSTER_M4ONLY=1
export SGLANG_Q38C_LOOKUP_NATIVE=1
export SGLANG_Q38C_LOOP_GLUE=1
export SGLANG_Q38C_MK_GDN=1
export SGLANG_Q38C_MK_MOE=1
export SGLANG_Q38C_MK_MOE_AOT=$AOT_DIR/mkm
export SGLANG_Q38C_MK_MOE_SF_PACK=1
export SGLANG_Q38C_MK_MOE_SF_PACK_AOT=$AOT_DIR/sf-pack
export SGLANG_Q38C_MK_MTP=1
export SGLANG_Q38C_MK_QSA=1
export SGLANG_Q38C_QSA_PDL_FIX=1
export SGLANG_Q38C_ROUTER_AOT=$AOT_DIR/router
export SGLANG_Q38C_ROUTER_FUSED=1
export SGLANG_Q38C_ROUTER_LAYERS=all
export SGLANG_Q38C_SCORE_Q4=1
export SGLANG_Q38C_SCORE_Q4_N16=1
export SGLANG_Q38C_TOPK_ZERO_START=1
export SGLANG_Q38_HC_COMBINE_PARTS=2
export SGLANG_Q38_HC_SMALLM16=1
export SGLANG_Q38_HC_SMALLM4=0
export SGLANG_Q38_ROUTER_FRONT=0
export SGLANG_Q38_ROUTER_HYBRID=0
export SGLANG_Q38_ROUTER_NATIVE_REDUCE=0
export SGLANG_SIMULATE_ACC_LEN=2.32
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
export SGLANG_TIMEOUT_KEEP_ALIVE=1800
```

Also make sure these are **unset** on the candidate: `Q38_NATIVE_DIAGNOSTICS Q38_GDN_ATTEST_DIR Q38_SHADOW_DIR Q38_GDN_HISTORY_EARLY SGLANG_PERFLL_RS_D1 SGLANG_PERFLL_RS_D1_MAX_BS Q38_MOE_C1_PIN_FILE`.

### Build the MoE / router AOT bundles (candidate only)

The candidate loads three prebuilt bundles; without them the server fails at the first MoE call or at router initialization (`KeyError`, no JIT and no fallback). Build them once on any ARM64 CPU host, with no GPU, checkpoint or private files, using the same image as the server:

```bash
git clone --branch perf/q38cc/base-tp4 https://github.com/YAMY1234/sglang.git sglang-2a92
cd sglang-2a92 && git checkout --detach 2a92c89d22d660197949fc36e6f4fb6b66332e14
cp docs/qwen38_fn/build_q38_aot.py .   # from the q38cc/fn-tp4-repro branch
docker run --rm --platform linux/arm64 -v "$PWD:/work" -w /work \
  lmsysorg/sglang:nightly-dev-cu13-20261015-a6a8c560 \
  bash -lc 'export CUDA_VISIBLE_DEVICES="" TRITON_CACHE_DIR=/tmp/c/triton XDG_CACHE_HOME=/tmp/c/xdg;
            python3 build_q38_aot.py --source /work --output /work/aot-2a92'
```

It writes `aot-2a92/mkm/{q35-port.cubin,finalize.cubin,manifest.json}`, `aot-2a92/router/{route-m1.cubin,route-m8.cubin,manifest.json}` and `aot-2a92/sf-pack/{q35-sf-pack.cubin,manifest.json}` (NVRTC `sm_103a`, plus the ordered top-10 finalize compiled for capability 10.3). The same sources are byte-identical at 757f, so the same bundles serve both commits. Then set, in the candidate environment above:

```bash
export AOT_DIR=$PWD/aot-2a92
export SGLANG_Q38C_MK_MOE_AOT=$AOT_DIR/mkm SGLANG_Q38C_ROUTER_AOT=$AOT_DIR/router SGLANG_Q38C_MK_MOE_SF_PACK_AOT=$AOT_DIR/sf-pack
```

`Q38_ROOT` in the environment list is read only by our internal launch helpers; leave it unset when launching `sglang.launch_server` directly with `--model-path`.

## 5. Benchmarks

Every comparison runs both arms **on the same node, one after the other, in AB order on one node and BA order on a second node**; numbers are reported per node and averaged. One TP4 server per 4-GPU tray.

### 5.1 Static (decode speed at fixed context)

Concurrency 1, greedy, `ignore_eos`, streamed through SGLang's native `/generate` endpoint with `input_ids`. TPS/user is measured from the decode phase only: output tokens divided by the time between the first and the last streamed token.

- **8K:** ISL 8192, OSL 1024, 12 timed requests (3 sets of 4), same prompt each time.
- **224K:** ISL 229,376 (an 8192-token prompt tiled), OSL 1024. One prefill-only request (`max_new_tokens=1`) to fill the prefix cache, one warm decode request, then 4 timed requests that hit the prefix cache.

```python
# minimal client: decode TPS/user for one request
import json, time, urllib.request
def decode_tps(port, input_ids, osl=1024):
    body = {"input_ids": input_ids, "stream": True,
            "sampling_params": {"temperature": 0, "top_p": 1, "max_new_tokens": osl,
                                "ignore_eos": True, "skip_special_tokens": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first = last = None
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            n = json.loads(line[6:])["meta_info"]["completion_tokens"]
            now = time.perf_counter()
            first = first or (now, n)
            last = (now, n)
    return (last[1] - first[1]) / (last[0] - first[0])
```

With the simulated acceptance length the result does not depend on the prompt text; any fixed 8192-token prompt reproduces it.

### 5.2 AgentX (agentic-coding traces)

InferenceX AgentX client in **canonical mode, 3600 s**, concurrency 1, dataset `semianalysisai/cc-traces-weka-062126-256k` (train split), 10 warm-up requests per lane, the same fixed benchmark id for both arms (`757f54362e39`), server started with the arguments above (HiCache off). Headline metric: **P50 and P90 TPS/user** (per-request interactivity = 1000 / full-response ITL in ms; P90 is the slow tail, i.e. the 10th percentile of per-request TPS/user).

Note: concurrency 1 means one root lane; the coding agent's sub-agent requests can overlap it, so the server sometimes runs 2–3 requests at once (about 7.5% of decode time in our runs).

### 5.3 Accuracy (GSM8K)

InferenceX `lm_eval` runner, full GSM8K test set (1,319 questions), concurrency 64, few-shot seed 1234, context length 16,384, max output 12,288 tokens, **simulated acceptance unset** (natural acceptance).

## 6. Expected results (GB300, TP4, c1, simulated AL 2.32 unless noted)

**AgentX canonical 3600 s, TPS/user** (two nodes, AB+BA):

| Node | Arm | P50 TPS/user | P90 TPS/user | Completed requests |
|---|---|---|---|---|
| A | main | 380.7 | 344.8 | 671 |
| A | 757f | 549.2 | 429.0 | 760 |
| B | main | 383.7 | 347.1 | 655 |
| B | 757f | 551.8 | 424.2 | 799 |
| **mean** | main | **382.2** | **345.9** | |
| **mean** | 757f | **550.5** | **426.6** | |
| | **757f / main** | **1.440x** | **1.233x** | |

0 errored requests in all arms (11 warm-up records dropped per arm).

**Static TPS/user** (two nodes, AB+BA, P50 of the timed requests):

| Context | main (node A / B) | 757f (node A / B) | 757f / main |
|---|---|---|---|
| 8K (ISL 8192, OSL 1024) | 420.2 / 420.3 | 581.3 / 582.3 | 1.384x |
| 224K (ISL 229,376, OSL 1024) | 386.8 / 386.4 | 568.1 / 568.1 | 1.469x |

**2a92 vs 757f, AgentX canonical 3600 s** (two other nodes, AB+BA): P50 TPS/user 551.8 / 551.3 → 561.5 / 557.5; P90 400.7 / 430.0 → 433.3 / 432.7.

**Accuracy (GSM8K full 1319, natural acceptance):** 757f strict-match 1288 / 1292 correct on two nodes, against 1291 / 1292 for the commit it was built on (`dad57dfd`; main was not part of this accuracy pair); natural acceptance length 3.197 / 3.200. No empty or garbled replies.

**Newer baseline** (upstream main `4517a5ff1bb` on `lmsysorg/sglang:nightly-dev-cu13-20261028-586355a8`, same arguments): Static 8K 421.6 / 422.2 TPS/user, 224K 387.7 / 387.0 (two boots, A/A within 0.2%), i.e. unchanged from the baseline above.

## 7. Notes and known limits

- The P90 gap (1.23x vs 1.44x at P50) comes from rounds where 2–3 requests overlap: several fused kernels in 757f only cover one request (MoE router, MKM routed experts, SF-pack, QSA score/top-k), so those rounds fall back to the stock kernels. Multi-request coverage is the next work item.
- `main 4517a5ff1bb` and newer require `sgl-flash-mla==1.0.0` and FlashInfer 0.7.2; use `nightly-dev-cu13-20261028-586355a8` or later for the rebased branch `qwen38-fn-mega`. The 10-15 nightly cannot capture CUDA graphs with that tree.
- Set `SGLANG_JIT_CACHE_DIR` (and `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`) to a local disk on the newer tree; the default `~/.cache/sglang/jit` can hit a home-directory quota during graph capture.
