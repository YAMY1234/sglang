# Kimi-K3.1-Pro on VR200: branch `kimi31pro-rubin`

This branch is the SGLang tree used to serve Kimi-K3.1-Pro with prefill/decode (PD) disaggregation on NVIDIA VR200 (SM107). It adds fully asynchronous pipeline-parallel (PP) prefill, a hierarchical KV cache with a remote DRAM L3 tier, Mamba/KDA state handling for that cache, DSA/MLA decode kernels for decode context parallelism (DCP), MegaMoE tuning, and a draft KV ring for DSpark speculative decoding.

Almost every change sits behind an environment flag that is off by default, so the tree behaves like its base unless the flags are set. The [Configuration](#configuration) section lists the flags and server arguments of the deployment this branch was cut for.

## Base

- Upstream: `sgl-project/sglang` `main` at `db0dfc7228` ("[Perf] dsv41: Reorder the Engram all-to-all output with a Triton kernel (#46925)"), committed 2026-10-27.
- On top: 189 commits (2 merges) that change about 466 files, then one documentation commit that adds this file.
- History is kept as developed, not squashed. Commits titled `overlay ...`, `vNp ...`, `... rebuild` and `compose: ...` are squashed imports of feature branches; their messages name the source branch or range.

```bash
git log --oneline db0dfc7228..kimi31pro-rubin
git diff --stat db0dfc7228 kimi31pro-rubin
```

Ranges below are written `A^..B` so that they include `A` and can be passed to `git log`.

## Feature groups

### Model and SM107 platform

- `afcd7d6b48^..5e4664aad1` (3 commits):
  - `afcd7d6b48`: Kimi-K3.1 model support (KDA linear attention, DSA sparse MLA, MoE, DFLASH and DSpark draft paths, memory pools, tests).
  - `overlay v1`: SM107 kernels: the vendored FlashInfer CuTe DSL KDA prefill and the SM107 entries of the sparse MLA decode kernels.
  - `5e4664aad1`: Rust radix-tree core on the container's torch, plus the relaxed write-back eviction fence.
- `ed9f659792^..19da015a8a`: rebuilt deltas of earlier feature branches: SM107 dense GEMM tactics, KDA prefill conv planes, HiCache batching, DSA DCP top-k push, MegaMoE decode settings, KDA ReplaySSM folds, K3 sequence-parallel (SP) collectives with a VR200 TP4 table.

### Fully asynchronous PP prefill

- `aa41e031a4^..aa08893b0b` and the `overlay v3pp` commit after it: PP0 decides KV releases for all stages in one relay pass; windowed relay waits; each stage sends its last KV chunk when its own forward completes.
- `118cb31ebe`, `8f490007b2`, `d198c19771`: deferred early-send settle, applied identically on every stage.
- `be9ea558c1^..d04fbf5f90` (merged at `b0c7c5898d`): stages stop meeting on the host every micro-batch: local results without an output ring, bounded send windows, background receive threads, optional late tensor receive, and a per-epoch state digest to check that stages agree.
- `ba96350c27^..d4edcebfcc`: activations go to the next stage by copy-engine push into a per-stage input ring (CUDA VMM fabric memory) instead of NCCL, with per-stage slot sizing and NCCL fallback.
- `1c0efd2aa2^..ec56d087aa`: optional per-GPU-speed layer split at boot (off in the deployment, which uses a static split).
- `31215da352^..5cb8760a65`: host lead bound, gated copy-engine pushes (every stage, `GATED=2`), and early sends handed to the transfer worker only once their forward is done.
- `250998ec05^..6a88d73fd0`, `344d70b227`: pitched-row copy of the inherited attention-residual bank on later stages, and a PDL ordering fix in the fused attention-residual kernel.
- `8c086078ac`: optional host-side check that each proxy message matches the micro-batch about to run.

### PD transfer and Mooncake

- `5ede66689e`, `852b28cd38`, `fd0e536f9f`: DCP-1 prefill to DCP-N decode: NumPy-built Mooncake descriptors, one packed gather per chunk for every decode DCP rank, optional high-priority gather stream.
- `2019fecaf6`, `7ac21df3d9`, `1e54c9a808`: transfer sharding by room or by destination, larger DCP pack chunks, fewer Mooncake calls per chunk.
- `f558a8837b`, `e4c5c15d99`: one device stream per transfer worker thread, and a prefill transfer timeout that also bounds the workers' CUDA waits. Optional non-spinning waits come with `SGLANG_DISAGG_TRANSFER_POLL_WAIT_US`.
- `5ba31e25f8`, `b14ef83bfe`, `0c09901b51`, `fc862f4f31`: decode side: optional cap on KV reserved ahead of prefill, compact prompts to the decode scheduler, decode-side DP choice, and a rows-aware DP load balancer within a KV guard.
- `0f45e473f8`, `325cb39cc7`, `85445aabd3`, `7d7f290e38`: transfer diagnostics (settle timing, send stats, per-chunk transfer time, in-flight room dump on SIGUSR2). Off by default.

### HiCache L2/L3

- `4d18220ac6`, `fe2eedc83c`, `031b3b62ef`: between-step batching of host copies, batched eviction waits, device write-ahead of the LRU tail so demotes do not wait for D2H copies.
- `9019bb43f6`: L3 tier: storage write-back and write-on-host-evict (L3 holds the pages leaving L2), Mamba host leaves and tombstone states written to L3, L3 reads past the tree's stateless KV, and the `nixl_dram` storage backend (client side).
- `516c05b37e`, `476f33b7b8`, `5cff824ee6`, `19188cad5d`: write-behind ack credit, lagged ack release on PP0, asynchronous tombstone writes, bounded admission waits for in-flight all-or-nothing reads.
- `c67069b8c5`, `d37ca524df` and its v2 follow-up, `d0f420f929`: `nixl_dram` staging region instead of registering the host pool, GIL-free grouped copies, and a layout-aware client that follows an L3 pool that grows after boot.
- `638d81ae3b^..d6eb08dd41`, `5a599ce027^..cb7e6d7ca6`, `0fd00d5647^..d7e8ffd4c4`: pinned index uploads without stream syncs, a load-back event ring sized for the PP depth, and a fix for coalesced copies across chunked host registrations.
- `a4c3fbf958^..f81a36e39f`, `eb8dcf776a`: host and L3 counters for microbenchmarks (off by default).

### Mamba/KDA state

- `5893f88128`: the per-path Mamba state cap can count host copies; Mamba host pool share and host usage log.
- `184c788038`, `7072d1f8bd`, `a3e363ab58`: which chunk states reach the radix tree, branch-point tracking, message-boundary branch points, and an admission fix for split chunks. Branch-point splitting comes with `SGLANG_MAMBA_BRANCH_TRACK_SPLIT`.
- `e10d7115ff`, `5caac27069^..7ce73afa26`, `88d940466f^..c3471f3191`: copy device-only state to host before a demote, per-layer copy-on-write during load-back, and a fix so a refused host-state load-back continues from the device anchor's state instead of a cleared slot.
- `1d22f9dc46`, `f078467ea5^..7ec4d702cf`: ReplaySSM verify: CUDA fast fold, phase-1-hoisted verify kernel, and folding accepted rows inside the next verify (pending fold), all bit-identical.
- `12c74435b1`, `150271b64f`, `fdeb17d0de`, `f1618b0887`: KDA prefill conv layout and wider tiles, host-seeded `cu_seqlens`, init-time precompiles, output-norm fold for any head count.

### DSA/MLA decode

- `92902f3dbe`, `d5b7affdaa^..f9e7c94700`, `d9f24332ed`, `6748b7a994`: DCP top-k push (each rank pushes merge candidates), DCP4 candidate count, fallback publishing, in-place DCP attention output.
- `590670e9fe`: decode switches ported from open upstream PRs (staged merge, DCP combine v3, split softmax, draft fusions); all off by default.
- `2e768f9359^..17c53e7696`, `1d1e70147e^..deea84270d` (inside `4ad181321b^..deea84270d`): wide DCP4 merge, sampled-threshold top-k for long rows, longest-row-first launch order, owner-rank merge, with bench-only trace instantiations.
- `e49f6aa2d4`: verify-step paged MQA logits with one Q block per request (bitwise equal to the DeepGEMM logits).
- `98559cd243`, `decd1700e9`: prefill: fused indexer top-k without the score matrix, MLA head slice passed to the V-BMM without a copy.

### MegaMoE, DeepGEMM and dense GEMMs

- `0e634f17d1`: MegaMoE decode BLOCK_M rules and weight localization across the GPU's two dies.
- `SGLANG_K31_DG_MEGA_PATCH` commit: serve MegaMoE from a bundled DeepGEMM header patch set through a node-local mirror of the installed package.
- `6b2e56e6f3^..a7e732a5b7`: KDA `qkvg`/`o_proj` on DeepGEMM `bf16_gemm_nt` at the decode row counts where cuBLAS is slowest.
- `753408db93`: prefill MoE with CuTe DSL grouped GEMMs (FC1 with fused activation and MXFP8 quantization).
- `b73ea6dc33`, `bfb9ab8cb0`, `04974f472a^..32b9abd50f`: K3 SP collectives and GEMM reduce-scatter, a 192-block TP4 push for mid-size batches, a VR200 table for the MoE front epilogue.

### DSpark speculative decoding (draft KV ring)

- `276e116e3a^..376a4fd85d`: page-aligned draft KV ring for the all-sliding-window DSpark draft: the draft pool keeps window plus verify-block rows per request instead of one row per target slot.
  - `276e116e3a`: the ring itself, PD transfer of the last window, and a staging ring on head-sharded decode.
  - The next two commits, up to `e58941ac70`: fused block locations with a sink page; the window starts at the request's prefix-hit floor.
  - `a4eeebcaf6`: staging slots: the staging ring holds N requests instead of one block per request-pool row.
  - `f4e145e871`, `376a4fd85d`: bounded unstage (deduplicated per row, this rank's heads only, fixed buffer) and dropping a request's pending unstage when it releases its row, so pending work never exceeds the live requests.

### DFLASH

- `6d9fe8a432`: run the DFLASH sliding-window draft without DCP (whole window ring on every rank, own heads only).
- `590670e9fe`: DFLASH draft-layer switches (packed top-k gather, fused epilogues, lean all-reduce launch).
- The DSpark ring work above makes the fused ring-location kernel shared by the DFLASH and DSpark rings.

### Scheduler, GC and diagnostics

- `b8c813653a`: prefill KV exhaustion fix (park a chunked continuation whose budget is spent, keep a truncated chunk's tail in the budget, defer a batch whose KV demand exceeds free plus evictable pages).
- `9b4d9f44de^..2bb098ef46`, `9dc456e6e9`, `608a7e698b`: shortest-prefill-first: periodic re-match, match-time logging, bounded anti-starvation aging.
- `548066c22f`, `e61d40e368`, `43020ab173`, `53bc0e7696`: `gc.freeze` in the scheduler after N batches and in the tokenizer manager after N requests.
- `2c121aa426^..0c748e49bc`, `6e40b32e98`: PP bubble diagnostics (per-phase host/wait partition, stack sampler, slow-event log) and an all-thread stack dump on SIGUSR2. Off by default.

### Misc fixes and tests

- Test fixes and test doubles: `207ef871df`, `a164f8d1ee`, `0e37d53ac9`, `8e5e707f69`, `0e83bb21b6`, `64f10306e1`, `7f0c2c1b08`, `4f68b24490`.
- Microbenchmark-only fake KV sender paths: `b8f2b13cbb`, `b0466ae4f4`.

## Deployment layout

| Role | Workers | GPUs per worker | Parallelism |
|---|---|---|---|
| Prefill | 3 | 8 (2 nodes x 4 GPUs) | PP8, TP1, EP1; static layer split `12,12,12,12,12,12,11,10` |
| Decode | 1 | 16 (4 nodes x 4 GPUs) | attention DP4 x (TP4 + DCP4), MoE EP16 over MegaMoE, DP LM head |

Both roles run DSpark with a block size of 5 draft tokens (K5, verify window 6) and the draft KV ring.

Why this decode layout:

- Attention TP4: the non-expert weights (MLA and KDA projections, shared experts, embeddings) are split four ways, which leaves room for KV on each GPU. With attention TP1 every DP rank would hold a full copy.
- DCP4 across the same four ranks: MLA stores one latent KV per token that all heads share, so plain TP4 would replicate the whole latent KV on all four ranks. DCP4 shards it by token, so each token's KV is stored once per attention DP group.
- Attention DP4 gives four independent decode batches, and EP16 spreads the routed experts over all 16 GPUs.

Prefill uses PP8 with TP1 and EP1, so the only cross-GPU traffic is the stage-to-stage activation hand-off; the async PP changes above keep stages from waiting on each other's host loop.

Outside this branch:

- Routing: an external KV-aware router sends requests to the prefill and decode workers.
- L3 DRAM pools: `nixl_dram` is a client. It connects to DRAM pool services (two per decode node, one per NUMA node) that register in `nixl_dram_registry_dir`. The service is not part of this branch.
- Rust radix-tree extension: prefill sets `SGLANG_HICACHE_DEVICE_WRITE_AHEAD=1`, which needs the `peek_device_eviction_candidates` and `set_write_ahead_spare` bindings. They are not in this branch's `rust/sglang-radix-tree` sources. The deployment builds the extension from `18f0ac81a2` (on fork branch `k31-v7pc-dop-l3v6-p14b-dp-kda1tok-wbuncov-pinidx-mlacompact-wahead`). Otherwise, leave the flag unset or set `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`.
- DeepGEMM: decode uses `sgl-deep-gemm` 0.2.8. `SGLANG_K31_DG_MEGA_PATCH=p157_156` only applies on top of that version, and MegaMoE weight localization needs DeepGEMM 0.2.5 or newer.

## Configuration

Placeholders: `<target-model-path>`, `<dspark-draft-model-path>`, `<node-local dir>`, `<registry dir>`, `<rdma devices>`. Defaults are the values in this tree (`python/sglang/srt/environ.py`, `python/sglang/srt/arg_groups/fields/`). The launcher also passes `--disaggregation-mode`, host, port and bootstrap arguments.

### Server arguments

| Argument | Prefill | Decode | Default | Meaning |
|---|---|---|---|---|
| `model-path` | `<target-model-path>` | same | | Kimi-K3.1-Pro FP4 checkpoint |
| `pipeline-parallel-size` | 8 | | 1 | PP stages per prefill worker |
| `tensor-parallel-size` | 1 | 16 | 1 | TP world per worker |
| `attn-dp-size` | | 4 | 1 | attention DP groups inside TP (attention TP = 16 / 4) |
| `dcp-size` | | 4 | 1 | decode context parallelism (shards KV by token) |
| `dcp-comm-backend` | | `fi_a2a` | auto | FlashInfer MNNVL all-to-all for the DCP reduction |
| `ep-size` | 1 | 16 | 1 | expert parallelism |
| `moe-a2a-backend` | | `megamoe` | `none` | DeepGEMM MegaMoE dispatch and combine |
| `moe-runner-backend` | `flashinfer_mxfp4` | | `auto` | MXFP4 MoE runner on prefill |
| `enable-dp-lm-head` | | true | false | vocabulary-parallel LM head inside the attention TP group |
| `dsa-prefill-backend`, `dsa-decode-backend` | pinned | pinned | auto-detect | both roles pin the same sparse MLA backend explicitly |
| `kv-cache-dtype` | `fp8_e4m3` | `fp8_e4m3` | `auto` | KV cache dtype |
| `page-size` | 64 | 64 | model default | tokens per KV page |
| `mem-fraction-static` | 0.84 | 0.916 | auto | static memory fraction (weights and pools) |
| `chunked-prefill-size`, `max-prefill-tokens` | 16384 | | auto, 16384 | prefill chunk and batch token budget |
| `max-running-requests` | | 464 | auto | 116 per attention DP rank |
| `cuda-graph-bs-decode` | | 8 to 32 step 8, 36 to 116 step 4 | auto | captured decode batch sizes |
| `cuda-graph-max-bs-decode` | | 116 | auto | largest captured decode batch |
| `load-balance-method` | | `total_tokens` | `auto` | decode DP load balancing |
| `schedule-policy` | `shortest-prefill-first` | | `fcfs` | prefill scheduling policy |
| `disaggregation-transfer-backend` | `mooncake` | `mooncake` | `mooncake` | KV transfer backend |
| `disaggregation-decode-allocation-policy` | `early` | | `early` | decode reserves KV at bootstrap |
| `disaggregation-decode-extra-slots` | | 16 | 0 | extra request slots for in-transfer requests |
| `enable-hierarchical-cache` | true | | false | HiCache (host L2) |
| `hicache-write-policy` | `write_back` | | `write_through` | write to host on device eviction |
| `hicache-size` | 250 | | 0 | host pool size in GB |
| `hicache-storage-backend` | `nixl_dram` | | none | L3 in remote DRAM over NIXL |
| `hicache-storage-backend-extra-config` | see below | | none | `nixl_dram` client settings |
| `hicache-storage-prefetch-policy` | `wait_complete` | | `timeout` | wait for L3 prefetches to finish |
| `hicache-storage-prefetch-retry-max-attempts`, `-poll-interval` | 0, 0 | | 8, 8 | no availability re-checks |
| `mamba-radix-cache-strategy` | `extra_buffer` | | `auto` | Mamba radix cache strategy |
| `mamba-track-interval` | 256 | | 256 | Mamba state tracking interval |
| `mamba-full-memory-ratio` | 1.2 | | auto | Mamba state memory relative to full KV |
| `mamba-ssm-dtype` | `float32` | `float32` | model config | SSM state dtype |
| `mamba-max-states-per-path` | 3 | 1 | -1 | cached Mamba states kept per root-to-tail path |
| `speculative-algorithm` | `DSPARK` | `DSPARK` | none | speculative decoding algorithm |
| `speculative-draft-model-path` | `<dspark-draft-model-path>` | same | none | DSpark draft weights |
| `speculative-dspark-block-size` | 5 | 5 | from checkpoint | draft tokens per block |
| `speculative-num-draft-tokens` | 6 | 6 | block size + 1 | verify window |
| `speculative-use-block-verification` | true | true | false | block verification of sampled drafts |
| `speculative-draft-kv-cache-dtype` | `fp8_e4m3` | `fp8_e4m3` | target's | draft KV dtype |
| `enable-linear-replayssm-spec` | | true | false | ReplaySSM verify for KDA layers |
| `linear-replayssm-cache-len` | | 8 | 16 | ReplaySSM ring length (at least the verify window) |
| `stream-interval` | | 30 | 1 | streaming interval in tokens |
| `watchdog-timeout` | 3600 | 3600 | 300 | hard watchdog (s) |
| `soft-watchdog-timeout` | | 300 | none | soft watchdog that dumps stacks (s) |
| `gc-threshold` | 700 10 1000 | 700 10 1000 | none | Python GC thresholds |
| `gc-warning-threshold-secs` | | 0.3 | 0 | log long GC pauses |

Both roles also set `trust-remote-code`, `reasoning-parser kimi_k3`, `tool-call-parser kimi_k3`, `weight-loader-disable-mmap`, `model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}'`, `enable-cache-report` and `enable-metrics`. Prefill also sets `weight-loader-drop-cache-after-load`.

`hicache-storage-backend-extra-config` on prefill:

```json
{"nixl_dram_registry_dir": "<registry dir>", "nixl_dram_expected_services": 8,
 "prefetch_threshold": 8192, "nixl_dram_ucx_devices": "<rdma devices>",
 "nixl_dram_reg_chunk_mib": 2048}
```

### Environment flags: both roles

Defaults marked † depend on `SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION` (off by default); the off value is shown.

| Flag | Value | Default | Meaning |
|---|---|---|---|
| `SGLANG_DSPARK_DRAFT_KV_RING` | 1 | 0 | keep the DSpark draft KV in a per-request, page-aligned ring of window plus verify rows |
| `SGLANG_K31_KDA_PREFILL_FI_SM107` | 1 | 0 | admit SM107 to the vendored FlashInfer CuTe DSL KDA prefill |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | prefill 4096, decode 1536 | 8192 | MegaMoE buffer capacity in tokens per rank |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | True | unset | Mooncake custom memory pool for NVLink/MNNVL transfers |
| `SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB` | 32768 | 16384 † | CustomAllReduceV2 workspace cap per direction |
| `SGLANG_DCP_PUSH_COMBINE_MAX_ROWS` | 1536 | 1024 † | rows per DCP push-combine call |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | 3600 | 300 | PD bootstrap timeout (s) |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | 3600 | 300 | PD waiting timeout (s) |
| `SGLANG_TIMEOUT_KEEP_ALIVE` | 900 | 5 | HTTP keep-alive (s) |
| `SGLANG_CACHE_DIR` | `<node-local dir>` | `~/.cache/sglang` | kernel and JIT cache directory |

### Environment flags: prefill

| Flag | Value | Default | Meaning |
|---|---|---|---|
| **Pipeline** | | | |
| `SGLANG_PP_LAYER_PARTITION` | `12,12,12,12,12,12,11,10` | unset | layers per stage |
| `SGLANG_PP_COMM_OVERLAP` | 1 | 0 | PP tensor communication on its own stream |
| `SGLANG_PP_PD_PREFILL_RELEASE_RELAY` | 1 | 0 | PP0 decides KV releases from every stage's status and relays them in one pass |
| `SGLANG_PP_SYNC_WINDOW` | 1 | 0 | leave `pp_size + pp_async_batch_depth` rounds of relay sends un-waited |
| `SGLANG_PP_PD_PREFILL_EARLY_SEND` | 1 | 0 | each stage sends its last KV chunk when its own forward completes |
| `SGLANG_PP_PD_EARLY_SEND_WHEN_READY` | 1 | 0 | hand an early chunk to the transfer worker only after its forward has finished |
| `SGLANG_PP_PD_DEFER_EARLY_SETTLE` | 1 | 0 | an early chunk in flight does not block the result; checkpoint frees wait for the transfer |
| `SGLANG_PP_PD_LOCAL_RESULT` | 1 | 0 | no output ring; only the last stage processes the output and sends first-token metadata |
| `SGLANG_PP_DECOUPLE_HOST` | 1 | 0 | stages stop meeting on the host every micro-batch (send windows, receive threads) |
| `SGLANG_PP_CE_PUSH` | 1 | 0 | copy-engine push of activations into the next stage's input ring instead of NCCL |
| `SGLANG_PP_CE_PUSH_GATED` | 2 | 0 | a gate thread enqueues every stage's push copies once the forward ended and the slot is free |
| `SGLANG_PP_CE_PUSH_SLOT_MB` | `auto` | `1600` | ring slot size; `auto` fits the previous stage's largest proxy |
| `SGLANG_PP_CE_PUSH_LOG_INTERVAL` | 300 | 0 | seconds between push and fallback log lines |
| `SGLANG_PP_CHECK_PROXY_BATCH` | 1 | 0 | assert that each proxy message belongs to the micro-batch about to run |
| `SGLANG_PP_PARTITION_AUTO` | 0 | 0 | layer split from measured GPU speed (off) |
| `SGLANG_PP_PARTITION_AUTO_BURN_SCALE` | 1.25 | 1.0 | auto split only; no effect while it is off |
| `SGLANG_PP_PARTITION_AUTO_KV_FLOOR` | 1.0 | 1.0 | auto split only; no effect while it is off |
| `SGLANG_ATTN_RES_FAST_INHERIT` | 1 | 1 | pitched-row copy of the inherited attention-residual bank (bitwise identical) |
| **PD transfer** | | | |
| `SGLANG_DISAGG_DCP_CHUNK_PACK` | 1 | 0 | one gather packs a chunk for every decode DCP rank |
| `SGLANG_DISAGG_DCP_PACK_MAX_TOKENS` | 116480 | 0 | tokens per DCP relayout chunk (0: prefill chunk size) |
| `SGLANG_DISAGG_DCP_SEND_CALLS` | 3 | 0 | Mooncake calls per decode DCP rank per chunk (0: one per KV entry) |
| `SGLANG_DISAGG_TRANSFER_SHARD_BY_DST_RR` | 1 | 0 | shard transfers round-robin by destination, one worker per decode DP group |
| `SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT` | 300 | 0 | fail a room stuck this long after its last chunk; bounds worker CUDA waits |
| `SGLANG_DISAGG_DCP_SEND_STATS_INTERVAL` | 60 | 0 | send statistics log interval (s) |
| `SGLANG_DISAGG_TRANSFER_TIME_STATS_INTERVAL` | 60 | 0 | transfer timing log interval (s) |
| **HiCache L2** | | | |
| `SGLANG_HICACHE_DEVICE_WRITE_AHEAD` | 1 | 0 | back up the device LRU tail before eviction (needs the Rust bindings above) |
| `SGLANG_OPT_HICACHE_BATCH_WRITE_BACK_COPY` | 1 | 0 | batch small page copies and merge adjacent host rows |
| `SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE` | 1 | 0 | eviction backups wait only for work issued before the in-flight forward |
| `SGLANG_HICACHE_WB_EVICT_BATCH` | 1 | 0 | queue all eviction copies and wait once per walk |
| `SGLANG_HICACHE_LOAD_BLOCK_QUOTA` | 32 | 0 | grid cap for host-to-device load kernels (0: kernel default) |
| `SGLANG_HICACHE_MAMBA_BACKUP_BLOCK_QUOTA` | 16 | 0 | grid cap for the Mamba state backup kernel (0: 2 blocks) |
| `SGLANG_HICACHE_PIN_LOAD_INDICES` | 1 | 0 | upload load indices through pooled pinned buffers, no stream sync |
| `SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM` | 1 | 0 | host pressure does not first reclaim redundant Full host copies |
| `SGLANG_HICACHE_COW_PER_LAYER` | 1 | 0 | per-layer Mamba copy-on-write during a load-back |
| `SGLANG_HICACHE_MAMBA_HOST_SCALE` | 0.67 | unset | scale of the Mamba host pool's share of `hicache-size` |
| `SGLANG_HICACHE_MAMBA_BACKUP_BEFORE_DEMOTE` | 1 | 0 | copy device-only Mamba state to host before a demote instead of dropping it |
| `SGLANG_HICACHE_PRECOMPILE_TRANSFER_MAMBA` | 1 | 0 | build the host Mamba transfer kernel at pool creation |
| `SGLANG_HICACHE_HOST_USAGE_LOG_INTERVAL` | 60 | 0 | host usage log interval (s) |
| `SGLANG_LOG_MAMBA_DEMOTE_DROPS` | 1 | 0 | count demoted states in the host usage line |
| **HiCache L3** | | | |
| `SGLANG_HICACHE_NIXL_DRAM_STAGING_MIB` | 512 | 0 | bounce L3 transfers through one registered staging region |
| `SGLANG_HICACHE_L3_WRITE_ON_HOST_EVICT` | 1 | 0 | write L3 from the coldest L2 pages ahead of host eviction, not at L2 admission |
| `SGLANG_HICACHE_L3_WRITE_MAMBA_HOST_LEAVES` | 1 | 0 | also write cold Mamba host leaves ahead of eviction |
| `SGLANG_HICACHE_L3_WRITE_STATE_ON_TOMBSTONE` | 1 | 0 | write an evicted host Mamba state to L3 while its Full KV stays |
| `SGLANG_HICACHE_L3_TOMBSTONE_ASYNC` | 1 | 0 | issue those writes without waiting; free the slot at the synced ack drain |
| `SGLANG_HICACHE_L3_ACK_LAG_STEPS` | 4 | 0 | PP0 publishes only acks that were ready this many steps ago |
| `SGLANG_HICACHE_L3_WB_ACK_CREDIT` | 1 | 0 | acked write-behind backups count as clean, avoiding full tail walks |
| `SGLANG_HICACHE_L3_WB_WALK_EVERY` | 16 | 4 | scheduler steps between write-behind walks |
| `SGLANG_HICACHE_L3_BELIEF_TTL_S` | 300 | 900 | how long a written page is assumed present in L3 |
| `SGLANG_HICACHE_L3_READ_PAST_TREE_KV` | 1 | 0 | query L3 from the end of the tree's stateless KV, not from the deepest state |
| `SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV` | 0 | 0 | recover checkpoints inside the resident KV span (off) |
| `SGLANG_HICACHE_L3_SAVED_TOKEN_THRESHOLD` | 1 | 0 | admit a storage checkpoint by total prefill saved, including tree KV |
| `SGLANG_HICACHE_L3_INFLIGHT_ADMISSION_WAITS` | 8 | 0 | `best_effort` only; no effect under `wait_complete` |
| `SGLANG_HICACHE_L3_LIFETIME_LEDGER`, `SGLANG_HICACHE_L3_TRACE`, `SGLANG_HICACHE_L3_LOOKUP_LOG_N` | 0 | 0 | L3 diagnostics (off) |
| **Mamba/KDA** | | | |
| `SGLANG_MAMBA_CHUNK_CKPT` | `none` | `all` | mid-prefill chunk states donated to the radix tree |
| `SGLANG_MAMBA_BRANCH_TRACK_PREFER_END` | 1 | 0 | track the input end when the branch point falls in the last forward |
| `SGLANG_MAMBA_PATH_CAP_COUNTS_HOST` | 1 | 0 | `mamba-max-states-per-path` also counts host copies |
| `SGLANG_KDA_PREFILL_CONV_QKV_PLANES` | 1 | 0 | KDA prefill conv writes Q, K, V as separate planes |
| `SGLANG_K31_CONV1D_FWD_WIDE` | 1 | 0 | wider prefill causal-conv tiles (bitwise equal) |
| `SGLANG_K31_KDA_SEED_CU_SEQLENS` | 1 | 0 | seed the KDA `cu_seqlens` memo from host lengths, no readback |
| **Attention and MoE** | | | |
| `SGLANG_DSA_FUSED_INDEXER_TOPK` | 1 | 0 | prefill top-k from a fused scorer without the score matrix (exact) |
| `SGLANG_DSA_FUSED_INDEXER_TOPK_MIN_MEAN_KEYS` | 40960 | 65536 | use the fused path above this mean row length |
| `SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION` | 0.07 | 0.2 | free-memory share budgeted for indexer logits |
| `SGLANG_DSA_MLA_PREFILL_STRIDED_OUT` | 1 | 1 | pass the MLA head slice to the V-BMM without a copy |
| `SGLANG_K31_PREFILL_MOE_CUTE_FC1` | 1 | 0 | routed experts as CuTe DSL grouped GEMMs on SM107 prefill (EP1 only) |
| **Runtime** | | | |
| `SGLANG_SCHED_GC_FREEZE_AFTER` | 8100 | 0 | scheduler `gc.freeze()` after this many batches |
| `SGLANG_SCHED_GC_FREEZE_EVERY` | 70 | 0 | then re-freeze every N batches |
| `SGLANG_WARMUP_BF16_GEMM_MAX_M` | 32768 | 16384 | largest M of the startup BF16 GEMM warm-up |
| `SGLANG_NUMA_BIND_V2` | 0 | 1 | built-in NUMA binding off |

### Environment flags: decode

| Flag | Value | Default | Meaning |
|---|---|---|---|
| **DSpark** | | | |
| `SGLANG_DSPARK_DRAFT_KV_RING_STAGING_SLOTS` | 72 | 0 | staging ring for incoming draft rows holds this many requests (0: one block per request row) |
| `SGLANG_OPT_DFLASH_RING_LOCS_FUSED` | 1 | 0 | one-launch ring locations (shared by the DFLASH and DSpark rings) |
| `SGLANG_DSPARK_SIMULATE_FOLD` | 1 | 0 | only with `SGLANG_SIMULATE_ACC_LEN` set: apply the simulated accept inside the graphed verify |
| `K31HT_SOFTMAX_SPLIT` | 1 | 0 | vocab-split temperature softmax (same distribution within FP32 rounding, not bitwise) |
| **DSA/MLA with DCP** | | | |
| `SGLANG_DSA_DCP_TOPK_PUSH` | 1 | 0 | local top-k pushes only the merge's candidates |
| `SGLANG_DCP4_PUSH_CAND_K` | 2048 | 768 | candidates each rank pushes (2048 equals top-k) |
| `SGLANG_DCP4_PUSH_MERGE_WARPS` | 4 | 0 | warps per rank in the merge when candidates equal top-k |
| `SGLANG_DCP4_PUSH_OWNER_MERGE` | 1 | 0 | each row merges on its owner rank; a quarter of the push bytes |
| `SGLANG_DSA_DCP_TOPK_PUSH_SAMPLED` | 1 | 0 | long rows select candidates from a sampled threshold in one pass (same set) |
| `SGLANG_DSA_DCP_TOPK_PUSH_LPT` | 1 | 0 | launch sampled top-k blocks longest row first |
| `SGLANG_DSA_DCP_ATTN_INPLACE_OUT` | 1 | 0 | split DCP sparse MLA calls write the output in place |
| `SGLANG_K31_DSA_PAGED_MQA_ONEBLOCK` | 1 | 0 | verify paged MQA logits with one Q block per request |
| `K31HT_DCP_COMBINE` | `v3` | `0` | DCP push-combine kernel v3 (bitwise equal) |
| **KDA** | | | |
| `SGLANG_OPT_KDA_REPLAYSSM_FAST_FOLD` | 1 | 0 | CUDA row-parallel ReplaySSM fold (bit-identical) |
| `SGLANG_OPT_KDA_MTP_VERIFY_P1_HOIST` | 1 | 0 | phase-1-hoisted KDA verify kernel (bit-identical) |
| `SGLANG_OPT_KDA_PENDING_FOLD` | 1 | 0 | fold accepted rows inside the next verify instead of at commit (bit-identical) |
| `SGLANG_K31_KDA_ONORM_FOLD_ANY_HEADS` | 1 | 0 | fold the KDA output gated RMSNorm into verify for any head count |
| **MoE and GEMMs** | | | |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE` | 1 | 0 | split MegaMoE weights across the GPU's two dies |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_RAISE_DECODE_BLOCK_M` | 1 | 0 | larger DeepGEMM BLOCK_M for FP8 x FP4 calls in decode graphs |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_DECODE_BLOCK_M_96_MIN_TOKENS` | 0 | 0 | with the raise: always use BLOCK_M 96 |
| `SGLANG_K31_DG_MEGA_PATCH` | `p157_156` | `""` | MegaMoE from the bundled DeepGEMM header patch set |
| `SGLANG_DG_MEGA_PATCH_ROOT` | `<node-local dir>` | `/tmp/sglang-dg-mega-patch` | where the patched DeepGEMM mirror is built |
| `SGLANG_K31_MOE_START` | `early` | unset | shared experts fork before the MoE front; front and routed chain on the priority stream |
| `SGLANG_K31_SM107_GEMM_TACTICS` | 1 | 0 | SM107 per-M dense GEMM tactics and side-stream GEMMs |
| `SGLANG_K31_SM107_DG_BF16` | 1 | 0 | KDA `qkvg`/`o_proj` on DeepGEMM at the rows where cuBLAS is slowest (bitwise equal) |
| `SGLANG_K31_SM107_BFA_SIDE_MAX_TOKENS` | 1344 | 1152 | largest graphed rows whose KDA `[f_a, b]` GEMM runs on the side stream |
| `SGLANG_K31_SM107_GATE_SIDE_MAX_TOKENS` | 1344 | 128 | largest graphed rows whose MLA output gate runs on the side stream |
| `SGLANG_K3_SP_COLLECTIVE` | 1 | auto, off on SM107 | fused SP reduce-scatter and all-gather over the MNNVL push workspace |
| `SGLANG_OPT_K3_SP_TP4_WIDE_PUSH` | 1 | 0 | 192-block TP4 push for 448 to 767 tokens (same reduction order) |
| **PD and scheduling** | | | |
| `SGLANG_PD_DECODE_COMPACT_PROMPT` | 1 | 0 | send only a prompt's length and last tokens to the decode scheduler |
| `SGLANG_DISAGG_DECODE_IGNORE_ROUTED_DP_RANK` | 1 | 0 | decode picks the DP rank itself, ignoring the router's choice |
| `SGLANG_DISAGG_DECODE_LB_ROWS_EPS` | 0.03 | 0 | pick the DP with the fewest requests among those within mean KV + eps |
| `SGLANG_K31_TM_GC_FREEZE_AFTER` | 64 | 0 | tokenizer manager `gc.freeze()` after this many finished requests |

Decode also raises the sparse MLA backend's workspace to 256 MB (default 150 MB †).

Other environment: both roles set `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`, `NVSHMEM_REMOTE_TRANSPORT=none`, `WITH_NVIDIA_PEERMEM=0`, `PYTORCH_ALLOC_CONF=expandable_segments:False`, `TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800`, `PYTHONUNBUFFERED=1` and `PYTHONNOUSERSITE=1`. Prefill also sets `CUDA_MODULE_LOADING=EAGER`, `UCX_MEM_MMAP_HOOK_MODE=none`, `UCX_NET_DEVICES=<rdma devices>` and `DG_MEGA_MOE_SM107_K64=0` (read by DeepGEMM, not by this tree).

## Related branches not included

All on the same fork.

- `k31-next-b-l3g-dspkring` @ `167e81cefd`: the second precision basis: the same stack with load-time MXFP8 for the shared experts and the KDA projections.
- `k31-next-a-mxfp4idx` @ `95b839b72c`: this branch plus an MXFP4 DSA index-K cache (`SGLANG_K31_INDEXER_MXFP4`, off by default) and index-K dump tooling.
- `k31-moedec-ac` @ `8b48f372d7`: this branch plus MegaMoE decode levers: large-bucket BLOCK_M, shared `gate_up` beside the router, DeepGEMM dispatch rows for the latent `up_proj` and shared `down_proj`.
- `k31-pfmoe-a4w4` @ `c4482ae29c`: A4W4 prefill MoE (routed experts as NVFP4 W4A4 on FlashInfer's CuTe DSL MoE, `SGLANG_K31_PREFILL_MOE_A4W4`), branched before the draft KV ring.
- `k31-dspark-draft-ring` @ `5b12d13060`: development line of the DSpark draft KV ring, including a smaller unstage chunk buffer that is not in this branch.
