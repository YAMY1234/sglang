# Kimi-K3.1-Pro on VR200: the k10b max-throughput configuration (concurrency 930)

This is our max-throughput configuration for Kimi-K3.1-Pro on VR200 (SM107) with SGLang, run with InferenceX AgentX at concurrency 930. It uses PD disaggregation, fully asynchronous PP8 prefill, a hierarchical KV cache with a remote DRAM tier, and DSpark speculative decoding (block size 5) with the draft KV ring.

Precision: this is the "basis B" configuration, i.e. the precision set of NVIDIA's NVFP4 R3 checkpoint (MXFP8 KDA projections and shared experts, FP4 routed experts, BF16 elsewhere, FP8 KV and FP8 indexer; details in section 3). It differs from the previous k10p configuration only by `SGLANG_K31_BASISB_MXFP8=1` on both roles and by the SGLang tree.

Placeholders: `<WORK_DIR>` (shared work directory), `<NODE_LOCAL_DIR>` (node-local scratch), `<MODEL_DIR>`, `<DSPARK_SRC_DIR>`, `<DRAFT_DIR>`, `<L3_REGISTRY_DIR>` (a directory every node can read and write), `<RDMA_DEVICES>` (comma-separated RDMA devices, e.g. `mlx5_0,mlx5_1`) and `<RDMA_DEVICES_UCX>` (the same as `mlx5_0:1,mlx5_1:1`).

## 1. Code

- SGLang branch: https://github.com/YAMY1234/sglang/tree/kimi31pro-rubin at `4e1d1706fc15b84595dbc0f1cc575efbee2ecfc7` or later (later commits only add docs). Its code is exactly `167e81cefdb20f9fd771918d77b0f4b0b20eb57c` (the measured SGLang tree, also at https://github.com/YAMY1234/sglang/tree/k31-next-b-l3g-dspkring); on top of that the branch only adds docs (`docs/kimi31pro_rubin/`) and the run helpers in `tools/kimi31pro_rubin/`. `git diff 167e81cefd 4e1d1706fc -- python sgl-kernel` is empty.
- The container below ships SGLang Python sources at `afcd7d6b487a9cbe73e25941971b4cc107e6f9ae`, the first commit of the branch. The measured run replaced the container's runtime files (`python/sglang`, 150 files) with the measured tree's. We checked every file's before and after blob against those two commits. No separate patch is needed: install the branch's `python/sglang` over the container's. Nothing under `sgl-kernel/` changes.

```bash
git clone https://github.com/YAMY1234/sglang.git <WORK_DIR>/sglang
cd <WORK_DIR>/sglang && git checkout kimi31pro-rubin
# inside each container, before starting a worker (copy keeps the image's compiled extension files):
cp -r <WORK_DIR>/sglang/python/sglang/. /sgl-workspace/sglang/python/sglang/
```

### Rust radix-tree extension (prefill only)

Prefill sets `SGLANG_HICACHE_DEVICE_WRITE_AHEAD=1`. That flag needs the Rust tree-core bindings `peek_device_eviction_candidates` and `set_write_ahead_spare`, which are in commit `18f0ac81a20ff6e5fa5798729e29eef53777155d` on the same fork but not in this branch's `rust/` sources. Build the extension in the container (same torch and Rust toolchain), then copy it into the prefill containers only; decode keeps the image's extension.

```bash
git clone https://github.com/YAMY1234/sglang.git <WORK_DIR>/sglang-treecore
cd <WORK_DIR>/sglang-treecore && git checkout 18f0ac81a20ff6e5fa5798729e29eef53777155d
cd python && SGLANG_BUILD_RUST_EXTS=sglang.srt.mem_cache.rust_tree_core.mem_cache python3 setup.py build_rust --inplace
# prefill containers:
cp sglang/srt/mem_cache/rust_tree_core/mem_cache.cpython-312-aarch64-linux-gnu.so \
   /sgl-workspace/sglang/python/sglang/srt/mem_cache/rust_tree_core/
```

The measured build's sha256 is `abb128734a4b5af909e80dceedcc637d36c1386287fb569c001cc23b307de05e`. A rebuild with another toolchain may differ.

### DeepGEMM (decode only)

- Install `sgl-deep-gemm==0.2.8` from PyPI in the decode containers: `pip install --no-deps sgl-deep-gemm==0.2.8`. The aarch64 wheel's sha256 is `eae77ab8b7e1870fc2a6939ec2c81776a755bff7f8402cb30d3565d998f870e4`.
- `SGLANG_K31_DG_MEGA_PATCH=p157_156` serves MegaMoE from patched headers that ship in the branch, at `python/sglang/srt/dg_mega_patch_data/p157_156/`. They come from `sgl-project/DeepGEMM` PRs 156 and 157, backported to 0.2.8. SGLang builds a mirror of the installed `deep_gemm` in `SGLANG_DG_MEGA_PATCH_ROOT`; it applies only if the installed headers are 0.2.8's.

## 2. Container

`lmsysorg/sglang:dev-kimi-k31-pro-cu134` (aarch64, CUDA 13, includes Dynamo 1.6.0), pinned by digest:

```bash
docker pull lmsysorg/sglang@sha256:a42079a8d1299b94824eab821de5f261c016797434cf69dc1fb3b15bdfddfc0e
```

## 3. Checkpoints

- Target: `moonshotai/Kimi-K3.1-Pro` at revision `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` (FP4 MoE checkpoint).
- Draft: `Inferact/Kimi-K3.1-Pro-DSpark` at revision `e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f`, served through an SGLang-readable view (below).
- Precision (basis B): the target checkpoint is loaded as is, and `SGLANG_K31_BASISB_MXFP8=1` casts the KDA projections (q, k, v, g, b, f_a, o) and the shared-expert gate, up and down projections to MXFP8 at load time. Those tensors are byte-identical to the MXFP8 tensors stored in NVIDIA's NVFP4 R3 checkpoint (`nvidia/Kimi-K3.1-Pro-NVFP4`), and R3's NVFP4 routed experts dequantize to exactly this checkpoint's MXFP4 values. Everything R3 keeps in BF16 stays BF16. KV cache and indexer cache are FP8. MoE combine stays BF16 (no low-precision combine). The one difference from R3: routed-expert activations run in MXFP8 (MXFP4 weights x MXFP8 activations), which is higher precision than R3's NVFP4 activations.

```bash
huggingface-cli download moonshotai/Kimi-K3.1-Pro --revision d9e99b0792a5e93b78ea9bea62a68167e36ddd9e --local-dir <MODEL_DIR>
huggingface-cli download Inferact/Kimi-K3.1-Pro-DSpark --revision e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f --local-dir <DSPARK_SRC_DIR>
python3 mk_dspark_view.py <DSPARK_SRC_DIR> <DRAFT_DIR>
```

`mk_dspark_view.py` copies the weights and writes a flat Qwen3-style `config.json`: the transformer layer config moves to the top level, and `target_layer_ids = aux_hidden_state_layer_ids - 1`.

```python
import json, os, shutil, sys
src, dst = sys.argv[1:3]
c = json.load(open(os.path.join(src, "config.json")))
assert c.get("speculators_model_type") == "dspark" and c["architectures"] == ["DSparkDraftModel"]
t = c["transformer_layer_config"]; aux = c["aux_hidden_state_layer_ids"]; ids = [a - 1 for a in aux]
v = dict(t)
v.update({
    "architectures": ["DSparkDraftModel"], "model_type": "qwen3", "dtype": c.get("dtype", "bfloat16"), "torch_dtype": c.get("dtype", "bfloat16"),
    "num_target_layers": 93, "block_size": c["block_size"], "mask_token_id": c["mask_token_id"], "draft_vocab_size": c["draft_vocab_size"],
    "markov_rank": c["markov_rank"], "markov_head_type": c["markov_head_type"], "enable_confidence_head": c["enable_confidence_head"],
    "confidence_head_with_markov": c["confidence_head_with_markov"], "sample_from_anchor": c["sample_from_anchor"],
    "tie_word_embeddings": False, "rope_theta": t["rope_parameters"]["rope_theta"],
    "dflash_config": {"target_layer_ids": ids, "block_size": c["block_size"], "mask_token_id": c["mask_token_id"]},
    "dspark_view": {"source": os.path.realpath(src), "aux_hidden_state_layer_ids": aux, "target_layer_ids": ids},
})
os.makedirs(dst, exist_ok=True)
for f in ("model.safetensors", "README.md", "LICENSE"):
    if os.path.exists(os.path.join(src, f)):
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
json.dump(v, open(os.path.join(dst, "config.json"), "w"), indent=2, sort_keys=True)
print("VIEW_OK", dst, "target_layer_ids", ids)
```

## 4. Layout (11 nodes, 40 GPUs, VR200 with 4 GPUs per node)

| Role | Nodes | Parallelism |
|---|---|---|
| Prefill x3 | 6 (2 per worker) | PP8, TP1, EP1; static layer split `12,12,12,12,12,12,11,10` |
| Decode x1 | 4 | TP16 = attention DP4 x (TP4 + DCP4), MoE EP16 (MegaMoE), DP LM head |
| Frontend | 1 | Dynamo 1.6.0 KV-aware frontend, etcd, NATS, benchmark client |

The L3 tier is a DRAM pool on the decode nodes: one pool service per NUMA node, so eight in total. Each one registers in `<L3_REGISTRY_DIR>` and is reached through NIXL over `<RDMA_DEVICES>` (segment 331 GiB, growing to 502 GiB, slab 512 MiB). The service is `tools/kimi31pro_rubin/l3_dram_pool/nixl_dram_l3.py`; its README has the run command.

## 5. Environment

Both roles also need three simulated-acceptance variables (section 7) and the Dynamo transport settings:

```bash
export SGLANG_SIMULATE_ACC_LEN=3.81 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
export DYN_REQUEST_PLANE=tcp DYN_RESPONSE_PLANE=tcp
```

Prefill:

```bash
export SGLANG_K31_BASISB_MXFP8='1'
export SGLANG_DSPARK_DRAFT_KV_RING='1'
export SGLANG_HICACHE_COW_PER_LAYER='1'
export SGLANG_LOG_MAMBA_DEMOTE_DROPS='1'
export SGLANG_HICACHE_MAMBA_BACKUP_BEFORE_DEMOTE='1'
export SGLANG_ATTN_RES_FAST_INHERIT='1'
export SGLANG_PP_PD_EARLY_SEND_WHEN_READY='1'
export SGLANG_PP_CE_PUSH_GATED='2'
export SGLANG_PP_CE_PUSH_LOG_INTERVAL='300'
export SGLANG_PP_PARTITION_AUTO_BURN_SCALE='1.25'
export SGLANG_PP_PARTITION_AUTO_KV_FLOOR='1.0'
export SGLANG_PP_PARTITION_AUTO='0'
export SGLANG_PP_CE_PUSH_SLOT_MB='auto'
export SGLANG_PP_CE_PUSH='1'
export SGLANG_HICACHE_L3_INFLIGHT_ADMISSION_WAITS='8'
export SGLANG_HICACHE_DEVICE_WRITE_AHEAD='1'
export SGLANG_OPT_HICACHE_BATCH_WRITE_BACK_COPY='1'
export SGLANG_MAMBA_CHUNK_CKPT='none'
export SGLANG_MAMBA_BRANCH_TRACK_PREFER_END='1'
export SGLANG_K31_PREFILL_MOE_CUTE_FC1='1'
export SGLANG_K31_CONV1D_FWD_WIDE='1'
export SGLANG_K31_KDA_SEED_CU_SEQLENS='1'
export SGLANG_HICACHE_PRECOMPILE_TRANSFER_MAMBA='1'
export SGLANG_DISAGG_DCP_PACK_MAX_TOKENS='116480'
export SGLANG_DISAGG_DCP_SEND_STATS_INTERVAL='60'
export SGLANG_DISAGG_DCP_SEND_CALLS='3'
export SGLANG_DISAGG_TRANSFER_SHARD_BY_DST_RR='1'
export SGLANG_DISAGG_TRANSFER_TIME_STATS_INTERVAL='60'
export SGLANG_DSA_FUSED_INDEXER_TOPK='1'
export SGLANG_DSA_FUSED_INDEXER_TOPK_MIN_MEAN_KEYS='40960'
export PYTHONUNBUFFERED='1'
export PYTHONNOUSERSITE='1'
export CUDA_MODULE_LOADING='EAGER'
export SGLANG_CACHE_DIR='<NODE_LOCAL_DIR>/sglang-cache'
export SGLANG_TIMEOUT_KEEP_ALIVE='900'
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT='1800'
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL='True'
export NCCL_MNNVL_ENABLE='1'
export NCCL_CUMEM_ENABLE='1'
export NVSHMEM_REMOTE_TRANSPORT='none'
export WITH_NVIDIA_PEERMEM='0'
export PYTORCH_ALLOC_CONF='expandable_segments:False'
export SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB='32768'
export SGLANG_DCP_PUSH_COMBINE_MAX_ROWS='1536'
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT='3600'
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT='3600'
export SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION='0.07'
export SGLANG_NUMA_BIND_V2='0'
export SGLANG_WARMUP_BF16_GEMM_MAX_M='32768'
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK='4096'
export DG_MEGA_MOE_SM107_K64='0'
export SGLANG_PP_LAYER_PARTITION='12,12,12,12,12,12,11,10'
export SGLANG_K31_KDA_PREFILL_FI_SM107='1'
export SGLANG_PP_PD_PREFILL_RELEASE_RELAY='1'
export SGLANG_PP_PD_PREFILL_EARLY_SEND='1'
export SGLANG_MAMBA_PATH_CAP_COUNTS_HOST='1'
export SGLANG_HICACHE_MAMBA_HOST_SCALE='0.67'
export SGLANG_HICACHE_HOST_USAGE_LOG_INTERVAL='60'
export SGLANG_KDA_PREFILL_CONV_QKV_PLANES='1'
export SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE='1'
export SGLANG_HICACHE_WB_EVICT_BATCH='1'
export SGLANG_HICACHE_LOAD_BLOCK_QUOTA='32'
export SGLANG_HICACHE_MAMBA_BACKUP_BLOCK_QUOTA='16'
export UCX_MEM_MMAP_HOOK_MODE='none'
export UCX_NET_DEVICES='<RDMA_DEVICES_UCX>'
export SGLANG_HICACHE_L3_WRITE_ON_HOST_EVICT='1'
export SGLANG_HICACHE_L3_WRITE_MAMBA_HOST_LEAVES='1'
export SGLANG_HICACHE_L3_BELIEF_TTL_S='300'
export SGLANG_HICACHE_L3_WRITE_STATE_ON_TOMBSTONE='1'
export SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV='0'
export SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM='1'
export SGLANG_HICACHE_L3_SAVED_TOKEN_THRESHOLD='1'
export SGLANG_HICACHE_L3_READ_PAST_TREE_KV='1'
export SGLANG_HICACHE_L3_LIFETIME_LEDGER='0'
export SGLANG_SCHED_GC_FREEZE_AFTER='8100'
export SGLANG_SCHED_GC_FREEZE_EVERY='70'
export SGLANG_HICACHE_L3_TOMBSTONE_ASYNC='1'
export SGLANG_HICACHE_L3_ACK_LAG_STEPS='4'
export SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT='300'
export SGLANG_HICACHE_L3_TRACE='0'
export SGLANG_HICACHE_L3_LOOKUP_LOG_N='0'
export SGLANG_HICACHE_L3_WB_ACK_CREDIT='1'
export SGLANG_HICACHE_L3_WB_WALK_EVERY='16'
export SGLANG_PP_PD_DEFER_EARLY_SETTLE='1'
export SGLANG_DISAGG_DCP_CHUNK_PACK='1'
export SGLANG_PP_COMM_OVERLAP='1'
export SGLANG_PP_DECOUPLE_HOST='1'
export SGLANG_PP_SYNC_WINDOW='1'
export SGLANG_PP_PD_LOCAL_RESULT='1'
export SGLANG_PP_CHECK_PROXY_BATCH='1'
export SGLANG_HICACHE_PIN_LOAD_INDICES='1'
export SGLANG_HICACHE_NIXL_DRAM_STAGING_MIB='512'
export SGLANG_DSA_MLA_PREFILL_STRIDED_OUT='1'
```

Decode:

```bash
export SGLANG_K31_BASISB_MXFP8='1'
export SGLANG_DSPARK_DRAFT_KV_RING_STAGING_SLOTS='72'
export SGLANG_OPT_K3_SP_TP4_WIDE_PUSH='1'
export SGLANG_DSPARK_SIMULATE_FOLD='1'
export SGLANG_DSPARK_DRAFT_KV_RING='1'
export SGLANG_PD_DECODE_COMPACT_PROMPT='1'
export SGLANG_K31_SM107_DG_BF16='1'
export SGLANG_DCP4_PUSH_OWNER_MERGE='1'
export SGLANG_DSA_DCP_TOPK_PUSH_LPT='1'
export SGLANG_DSA_DCP_TOPK_PUSH_SAMPLED='1'
export SGLANG_DCP4_PUSH_MERGE_WARPS='4'
export SGLANG_K31_DG_MEGA_PATCH='p157_156'
export SGLANG_OPT_KDA_PENDING_FOLD='1'
export SGLANG_DG_MEGA_PATCH_ROOT='<NODE_LOCAL_DIR>/dg-mega-patch'
export SGLANG_DCP4_PUSH_CAND_K='2048'
export SGLANG_DISAGG_DECODE_IGNORE_ROUTED_DP_RANK='1'
export SGLANG_DISAGG_DECODE_LB_ROWS_EPS='0.03'
export SGLANG_K31_TM_GC_FREEZE_AFTER='64'
export SGLANG_K31_DSA_PAGED_MQA_ONEBLOCK='1'
export SGLANG_K31_KDA_ONORM_FOLD_ANY_HEADS='1'
export PYTHONUNBUFFERED='1'
export PYTHONNOUSERSITE='1'
export SGLANG_CACHE_DIR='<NODE_LOCAL_DIR>/sglang-cache'
export SGLANG_TIMEOUT_KEEP_ALIVE='900'
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT='1800'
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL='True'
export NCCL_MNNVL_ENABLE='1'
export NCCL_CUMEM_ENABLE='1'
export NVSHMEM_REMOTE_TRANSPORT='none'
export WITH_NVIDIA_PEERMEM='0'
export PYTORCH_ALLOC_CONF='expandable_segments:False'
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK='1536'
export SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB='32768'
export SGLANG_DCP_PUSH_COMBINE_MAX_ROWS='1536'
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT='3600'
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT='3600'
export SGLANG_TRTLLM_MLA_WORKSPACE_MB='256'
export SGLANG_K31_KDA_PREFILL_FI_SM107='1'
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_RAISE_DECODE_BLOCK_M='1'
export SGLANG_K31_MOE_START='early'
export SGLANG_K31_SM107_GEMM_TACTICS='1'
export SGLANG_OPT_DFLASH_RING_LOCS_FUSED='1'
export SGLANG_K31_SM107_BFA_SIDE_MAX_TOKENS='1344'
export SGLANG_K31_SM107_GATE_SIDE_MAX_TOKENS='1344'
export SGLANG_DSA_DCP_TOPK_PUSH='1'
export SGLANG_DSA_DCP_ATTN_INPLACE_OUT='1'
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_DECODE_BLOCK_M_96_MIN_TOKENS='0'
export SGLANG_OPT_KDA_REPLAYSSM_FAST_FOLD='1'
export SGLANG_OPT_KDA_MTP_VERIFY_P1_HOIST='1'
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE='1'
export SGLANG_K3_SP_COLLECTIVE='1'
export K31HT_DCP_COMBINE='v3'
export K31HT_SOFTMAX_SPLIT='1'
```

## 6. Launch

Frontend node (etcd and NATS started as Dynamo 1.6.0 requires):

```bash
export DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS='3600'
export DYN_NATS_REQUEST_TIMEOUT_SECS='14400'
export DYN_ROUTER_TEMPERATURE='0'
export DYN_TCP_CHANNEL_BUFFER='128'
export DYN_TCP_REQUEST_TIMEOUT='60'
export DYN_TOKENIZER_CACHE='1'
export DYN_TOKENIZER_CACHE_BYTES='8000000000'
python3 -m dynamo.frontend \
  --router-mode kv \
  --router-session-affinity-ttl-secs 14400 \
  --router-kv-overlap-score-credit 8 \
  --router-host-cache-hit-weight 1 \
  --router-decode-active-request-weight 0 \
  --dyn-chat-processor sglang \
  --dyn-preprocess-workers 4 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --trust-remote-code
```

Prefill: three workers. Each spans 2 nodes and runs one command per node (`--node-rank 0` and `1`, with `--dist-init-addr` on the worker's first node):

```bash
python3 -m dynamo.sglang --model-path <MODEL_DIR> --served-model-name moonshotai/Kimi-K3.1-Pro \
  --disaggregation-mode prefill --nnodes 2 --node-rank <0|1> --dist-init-addr <PREFILL_NODE0_IP>:<PORT> \
  --request-plane tcp --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:<KV_EVENTS_PORT>"}' \
  --trust-remote-code \
  --weight-loader-disable-mmap \
  --weight-loader-drop-cache-after-load \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}' \
  --watchdog-timeout 3600 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --tensor-parallel-size 1 \
  --ep-size 1 \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.84 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --mamba-full-memory-ratio 1.2 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-track-interval 256 \
  --mamba-ssm-dtype float32 \
  --mamba-max-states-per-path 3 \
  --disaggregation-decode-allocation-policy early \
  --schedule-policy shortest-prefill-first \
  --enable-hierarchical-cache \
  --hicache-write-policy write_back \
  --hicache-size 250 \
  --enable-cache-report \
  --enable-metrics \
  --gc-threshold 700 10 1000 \
  --disaggregation-transfer-backend mooncake \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path '<DRAFT_DIR>' \
  --speculative-num-draft-tokens 6 \
  --speculative-dspark-block-size 5 \
  --speculative-use-block-verification \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --pipeline-parallel-size 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --hicache-storage-backend nixl_dram \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-prefetch-retry-max-attempts 0 \
  --hicache-storage-prefetch-retry-poll-interval 0 \
  --hicache-storage-backend-extra-config '{"nixl_dram_registry_dir":"<L3_REGISTRY_DIR>","nixl_dram_expected_services":8,"prefetch_threshold":8192,"nixl_dram_ucx_devices":"<RDMA_DEVICES>","nixl_dram_reg_chunk_mib":2048}'
```

Decode: one worker on 4 nodes (`--node-rank 0` to `3`):

```bash
python3 -m dynamo.sglang --model-path <MODEL_DIR> --served-model-name moonshotai/Kimi-K3.1-Pro \
  --disaggregation-mode decode --nnodes 4 --node-rank <0..3> --dist-init-addr <DECODE_NODE0_IP>:<PORT> \
  --request-plane tcp \
  --trust-remote-code \
  --weight-loader-disable-mmap \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}' \
  --watchdog-timeout 3600 \
  --soft-watchdog-timeout 300 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --tensor-parallel-size 16 \
  --dcp-size 4 \
  --dcp-comm-backend fi_a2a \
  --attn-dp-size 4 \
  --ep-size 16 \
  --moe-a2a-backend megamoe \
  --enable-dp-lm-head \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --mem-fraction-static 0.916 \
  --max-running-requests 464 \
  --gc-warning-threshold-secs 0.3 \
  --load-balance-method total_tokens \
  --cuda-graph-max-bs-decode 116 \
  --cuda-graph-bs-decode 8 16 24 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 \
  --disaggregation-decode-extra-slots 16 \
  --mamba-ssm-dtype float32 \
  --mamba-max-states-per-path 1 \
  --stream-interval 30 \
  --enable-cache-report \
  --enable-metrics \
  --gc-threshold 700 10 1000 \
  --disaggregation-transfer-backend mooncake \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path '<DRAFT_DIR>' \
  --speculative-num-draft-tokens 6 \
  --speculative-dspark-block-size 5 \
  --speculative-use-block-verification \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --enable-linear-replayssm-spec \
  --linear-replayssm-cache-len 8
```

L3 DRAM pools: start them on the decode nodes before the workers, using the command in `tools/kimi31pro_rubin/l3_dram_pool/README.md`, and wait for `L3B_READY` from all 8.

## 7. Benchmark

- InferenceX AgentX at `1570a54f18398e742342ae00eda3924774a42959` (https://github.com/SemiAnalysisAI/InferenceX): `inferencex-e2e/benchmarks/srt_agentic.sh`, run on the frontend node through `tools/kimi31pro_rubin/bench/` (see its README for the exact commands).
- Settings: `MODEL=moonshotai/Kimi-K3.1-Pro`, `CONC=930`, `DURATION=3600`, `THINKING_MODE=thinking_on`, `SCENARIO_TYPE=agentic-coding`, `IS_AGENTIC=1`, `KV_OFFLOADING=dram`, `KV_OFFLOAD_BACKEND=hicache`, `TOTAL_CPU_DRAM_GB=12472`, `RESULT_FILENAME=agentx_c930`, `AIPERF_HTTP_TCP_USER_TIMEOUT=900000`, `AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang:`, `AIPERF_MAX_CONTEXT_LENGTH=996579`, `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true`.
- Client: aiperf `SemiAnalysisAI/aiperf` at `754356e9a39acc6cc6afb242d123bb57c3fb6f75` (as InferenceX pins it), with `src/aiperf/credit/sticky_router.py` taken from upstream aiperf `9adff5e48418e2e9be41758a22c409132cc4b8e6` ("fix(router): pin SPAWN subagents under their own id"). The fix only changes routing in the client and avoids client host OOM.
- Speculative acceptance follows InferenceX's synthetic-acceptance rule. The golden acceptance length for this target and draft at block size 5 with thinking on is 3.81 (InferenceX main, `kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark__probabilistic-block.yaml`). Hence `SGLANG_SIMULATE_ACC_LEN=3.81` on both roles; `SGLANG_DSPARK_SIMULATE_FOLD=1` applies it inside the graphed verify.
- Warm-up: before the client starts, the measured run sent synthetic requests through the frontend so that every decode DP rank decodes at every running-batch size the window uses. These were waves of random-token prompts at 1k, 8k and 32k tokens (plus one 64k and one 128k per 8 requests), 128 output tokens, and 1 to 20 running per rank. The prompts cannot hit the client's prefix cache. Disclosed as a server-side warm-up. Script: `tools/kimi31pro_rubin/decode_warmup/d_warmup.py --dp 4`.

## 8. What differs from the measured run

- The measured run used a batch launcher that renders the same worker, frontend and service commands as above. Host-level helpers (metrics taps and a page-cache drop on prefill nodes after weight load) are not included; they do not change the server configuration.

## 9. Notes on the run

- The binding resource at this concurrency is prefill; higher concurrency only adds prefill queueing.
- The harness may report requests as failed with `InvalidInferenceResultError` ("No responses with actual content"). These are not generation failures: they are short replies cut by the trace's `max_tokens` in the middle of a tool-call header, which the frontend's `kimi_k3` tool-call parser withholds until the header is complete (about 14 tokens). A frontend without a tool-call parser returns the same tokens as content.

## Open items

1. **Launch flag spellings:** the `--kv-events-config` endpoint port and the `dynamo.frontend` flag names follow the measured launch. They have not been checked against the Dynamo 1.6.0 documentation yet.
