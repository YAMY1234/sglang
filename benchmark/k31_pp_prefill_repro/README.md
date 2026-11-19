# Kimi-K3.1-Pro prefill on VR200: PP8 vs TEP8+DCP8

This directory reproduces a prefill-only throughput comparison between two ways of running one 8-GPU Kimi-K3.1-Pro prefill server on VR200 (Rubin):

- **PP8**: 8 pipeline stages, TP1 per stage.
- **TEP8+DCP8**: TP8, decode context parallel 8 (DCP: the MLA KV cache and the DSA index keys are split by token over the 8 ranks), EP8 with the fused MegaMoE kernel.

Result: at 128k context with an 89.8% prefix hit, PP8 prefills **79.4k new tok/s** and TEP8+DCP8 **60.5k** (**1.31×**). At 32k context with no prefix hit the ratio is **1.43×**. At a moderate Poisson load, TEP8+DCP8 has the lower TTFT.

The TEP8+DCP8 arm uses the public SGLang stack with its best public flags, picked by a screen beforehand. Results with additional unpublished TEP8 optimizations may differ.

## Contents

| File | Purpose |
|---|---|
| `make_arms.py` | Defines every server flag and env variable of both arms. Writes `arms.json` and prints the field diff. |
| `arms.json` | Generated output (`pp8`, `tep8`, and the screened-out `tep8-prims`). |
| `launch_server.py` | Execs one node's `sglang.launch_server` for one arm, optionally under `nsys launch`. |
| `pfab_client.py` | Load client: warm, probe, closed loop, Poisson. Writes one JSON line per request plus a summary. |
| `run_arm.sh` | Per-node driver. Node 0 runs rank 0 and the client; node 1 follows via marker files. It executes a plan file. |
| `plans/` | `bench-{pp8,tep8}.txt` (main bench), `screen-tep8.txt` (DSA backend screen), `profile-{pp8,tep8}.txt` + `profile.sched` (nsys). |
| `slurm_example.sh` | Example sbatch for one arm with a container runtime plugin (pyxis/enroot). |
| `parse_results.py` | Per-phase client numbers, the same window from the rank-0 server log, and a `server_args` diff of two runs. |
| `profile/` | `nsys_window.sh` (per-node nsys start/stop and py-spy sidecar), `classify_gpu.py` (exclusive GPU-time classes from the nsys SQLite), `breakdown.py` (GPU-ms per 1k new tokens). |

## Hardware

- VR200, 4 GPUs per node. **Each arm uses 2 nodes (8 GPUs)** in one multi-node NVLink domain; the env sets `MC_FORCE_MNNVL=1` and `NCCL_MNNVL_ENABLE=1`.
- Nodes must be exclusive. The driver kills leftover GPU processes between server boots.
- In the reference measurement, both arms ran at the same time on different node pairs of the same cluster.

## Software

| Layer | Version |
|---|---|
| Container image | `lmsysorg/sglang:dev-kimi-k31-pro-cu134@sha256:a42079a8d1299b94824eab821de5f261c016797434cf69dc1fb3b15bdfddfc0e` (PyTorch 2.15.0+cu134) |
| SGLang | This branch: base `afcd7d6b` (the image's SGLang commit) plus the commits listed below. Run it from the checkout with `PYTHONPATH=<repo>/python`. |
| Rust TreeCore (`mem_cache` extension) | Built from this branch's `rust/sglang-radix-tree` (Mamba path cap counting host copies). |
| DeepGEMM | `sgl-deep-gemm==0.2.8` from PyPI (aarch64 wheel sha256 `eae77ab8b7e1870fc2a6939ec2c81776a755bff7f8402cb30d3565d998f870e4`). It is put ahead of the image's copy on `PYTHONPATH`. |
| Model | `moonshotai/Kimi-K3.1-Pro`, revision `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` |
| Draft model (DFLASH) | `RadixArk/Kimi-K3.1-Dflash2-Preview`, revision `d0d8bc2e3c6d0a195d2e00a49017e71ca8d881f1` |
| Nsight Systems (profiling only) | 2026.5.1 CLI |

Commits on top of `afcd7d6b`:

1. `rust_tree_core: Mamba path cap can count host copies (Rust part)`.
2. `K3.1 VR200 (SM107) prefill overlay v7pcr`: the Python changes both arms run (60 files).
3. `schedule_batch: insert fake-bootstrap requests into the radix tree (benchmark only)`. See "Prefill-only setup".
4. `PD fake sender: optional non-blocking index readback (rig only, default off)`. Flag `SGLANG_DISAGG_FAKE_SENDER_ASYNC_INDICES`.
5. `PD fake sender: PP PD early send under the same rig flag`. Same flag.
6. This directory.

**Commits 4 and 5 were not active in the reference numbers.** The flag defaults to off, and the reference runs predate the fix. With the flag off, the fake KV sender pays one blocking device-to-host index copy per send that a real Mooncake sender does not pay, and PP early send is disabled for fake-host requests. To reproduce the numbers below, leave the flag off. To bring the prefill-only setup closer to real serving, set `SGLANG_DISAGG_FAKE_SENDER_ASYNC_INDICES=1`. Its effect on this A/B has not been measured.

Setup, once:

```bash
# SGLang source: this branch
git clone -b k31-pp8-prefill-repro https://github.com/YAMY1234/sglang.git $SGLANG_REPO

# checkpoints at the pinned revisions
huggingface-cli download moonshotai/Kimi-K3.1-Pro --revision d9e99b0792a5e93b78ea9bea62a68167e36ddd9e --local-dir $MODEL_PATH
huggingface-cli download RadixArk/Kimi-K3.1-Dflash2-Preview --revision d0d8bc2e3c6d0a195d2e00a49017e71ca8d881f1 --local-dir $DRAFT_MODEL_PATH

# DeepGEMM 0.2.8 into its own directory (inside the image, or with --platform manylinux2014_aarch64 --only-binary=:all:)
pip install --no-deps --target $DEEP_GEMM_DIR sgl-deep-gemm==0.2.8

# Rust TreeCore: build once inside the image into a cache that run_arm.sh copies in (RUST_EXT_CACHE)
SGLANG_CACHE_DIR=$WORK/rust-ext-cache PYTHONPATH=$SGLANG_REPO/python \
  python3 -c "import sglang.srt.mem_cache.rust_tree_core.extension as e; print(e.bindings.__file__)"
```

- The Rust build uses `cargo` from the image and needs crates.io access unless the crates are already cached.
- If no Rust extension can be loaded or built, SGLang **silently falls back to the Python TreeCore**, which is slower. Check that every server log has `Tree Core: RustUnifiedTreeCore`.

## Server configuration

`make_arms.py` is the source of truth. `launch_server.py` adds `--model-path $MODEL_PATH --host 0.0.0.0 --port 30000 --dist-init-addr <node0>:<port> --nnodes 2 --node-rank <0|1> --disaggregation-mode prefill` and replaces `@DRAFT_MODEL_PATH@` with `$DRAFT_MODEL_PATH`.

### Arguments, identical in both arms (32)

`--disaggregation-transfer-backend mooncake`, `--disaggregation-decode-allocation-policy early`, `--served-model-name moonshotai/Kimi-K3.1-Pro`, `--trust-remote-code`, `--reasoning-parser kimi_k3`, `--tool-call-parser kimi_k3`, `--kv-cache-dtype fp8_e4m3`, `--page-size 64`, `--speculative-algorithm DFLASH`, `--speculative-draft-model-path $DRAFT_MODEL_PATH`, `--speculative-draft-attention-backend cutedsl_mla`, `--speculative-draft-kv-cache-dtype fp8_e4m3`, `--speculative-num-draft-tokens 6`, `--speculative-use-block-verification`, `--dsa-prefill-backend trtllm`, `--dsa-decode-backend trtllm`, `--enable-hierarchical-cache`, `--hicache-size 230`, `--hicache-write-policy write_back`, `--mamba-radix-cache-strategy extra_buffer`, `--mamba-track-interval 256`, `--mamba-full-memory-ratio 1.2`, `--mamba-ssm-dtype float32`, `--schedule-policy shortest-prefill-first`, `--enable-cache-report`, `--enable-metrics`, `--gc-threshold 700 10 1000`, `--watchdog-timeout 3600`, `--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}'`, `--weight-loader-disable-mmap`. `--weight-loader-drop-cache-after-load` and `--weight-loader-prefetch-checkpoints` are off.

### Arguments that differ (11)

| | PP8 | TEP8+DCP8 |
|---|---|---|
| `--pipeline-parallel-size` | `8` | – |
| `--tensor-parallel-size` | `1` | `8` |
| `--dcp-size` | – | `8` |
| `--dcp-comm-backend` | – | `fi_a2a` |
| `--ep-size` | `1` | `8` |
| `--moe-runner-backend` | `flashinfer_mxfp4` | – |
| `--moe-a2a-backend` | – | `megamoe` |
| `--chunked-prefill-size` | `16384` | `32768` |
| `--max-prefill-tokens` | `16384` | `32768` |
| `--mem-fraction-static` | `0.90` | `0.88` |
| `--mamba-max-states-per-path` | `3` | `1` |

### Environment, identical in both arms (30)

```
CUDA_MODULE_LOADING=EAGER  PYTORCH_ALLOC_CONF=expandable_segments:False  TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
PYTHONUNBUFFERED=1  PYTHONNOUSERSITE=1
MC_FORCE_MNNVL=1  NCCL_MNNVL_ENABLE=1  NCCL_CUMEM_ENABLE=1  NVSHMEM_REMOTE_TRANSPORT=none  WITH_NVIDIA_PEERMEM=0
SGLANG_NUMA_BIND_V2=0  SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600  SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600  SGLANG_TIMEOUT_KEEP_ALIVE=900
SGLANG_WARMUP_BF16_GEMM_MAX_M=32768  SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB=32768  SGLANG_DCP_PUSH_COMBINE_MAX_ROWS=1536
SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.2  SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096  DG_MEGA_MOE_SM107_K64=0
SGLANG_K31_KDA_PREFILL_FI_SM107=1  SGLANG_KDA_PREFILL_CONV_QKV_PLANES=1
SGLANG_MAMBA_PATH_CAP_COUNTS_HOST=1  SGLANG_HICACHE_MAMBA_HOST_SCALE=0.65
SGLANG_HICACHE_WB_EVICT_BATCH=1  SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE=1
SGLANG_HICACHE_LOAD_BLOCK_QUOTA=64  SGLANG_HICACHE_MAMBA_BACKUP_BLOCK_QUOTA=16  SGLANG_HICACHE_HOST_USAGE_LOG_INTERVAL=60
```

### Environment that differs (9)

| | PP8 | TEP8+DCP8 |
|---|---|---|
| `SGLANG_PP_LAYER_PARTITION` | `12,12,12,12,12,12,11,10` | – |
| `SGLANG_PP_PD_PREFILL_RELEASE_RELAY` | `1` | – |
| `SGLANG_PP_PD_PREFILL_EARLY_SEND` | `1` | – |
| `SGLANG_K3_SP_COLLECTIVE` | – | `1` |
| `SGLANG_K3_SP_ATTN_RES` | – | `1` |
| `SGLANG_K3_SP_GEMM_RS_MIN_TOKENS` | – | `16384` |
| `SGLANG_DSA_DCP_PREFILL_FUSED_GATHER` | – | `1` |
| `SGLANG_K31_DSA_PREFILL_ROW_SPLIT` | – | `1` |
| `SGLANG_K31_DSA_PER_WINDOW_LOGITS` | – | `1` |

`run_arm.sh` also points every JIT and compile cache (`SGLANG_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, FlashInfer, TileLang, TVM-FFI, DeepGEMM, CuTe AOT) at a fresh node-local directory per run. No JIT or autotune cache carries over between runs.

Notes on the two arms:
- **TEP8+DCP8 flags**: SP collectives fuse the TP reduce-scatter into the GEMM and the attention-residual all-gather. DCP prefill uses a fused gather. The DSA prefill splits rows across the DCP group and uses per-window logits. This is the best public TEP8+DCP8 configuration we have.
- **DSA backend screen** (`plans/screen-tep8.txt`): the rule was fixed in advance. Use `prims_ts` only if it beats `trtllm` by more than 2% with no failed request. Measured `prims_ts / trtllm = 0.999`, so both arms use `trtllm`. The `tep8-prims` arm in `arms.json` is the screened-out variant.
- `mem-fraction-static` and `mamba-max-states-per-path` are each arm's serving values.
  - TEP8+DCP8 uses 0.88, as its serving recipes do; its SP buffers are allocated before the KV pool. PP8 uses 0.90.
  - The measured hit is 89.8% ± 0.1 pp in both arms, so the Mamba state cap does not change the hit on this workload.
- **Engagement checks** in the server logs (`server-key-<host>.log`):
  - Both arms: `Tree Core: RustUnifiedTreeCore`, `Mamba path cap counts host copies (cap N, ...)`, `HiCache host usage` lines.
  - TEP8+DCP8: `K3 SP collective enabled (TP8, ...)`, `K3 SP attention-residual carry enabled`, `DSA prefill: sparse MLA rows split across the DCP group (dcp=8, ...)`, `DSA DCP prefill: fused KV / index-K gather`.
  - PP8: `rows split across` must **not** appear (the row split is gated on DCP > 1).
- **Second check**: `parse_results.py RESULTS RUN_A RUN_B --args-diff RUN_A RUN_B` diffs the resolved `server_args` of the two arms. In the reference runs, 533 of 551 fields are identical. The other 18 are all derived from the layout: parallel sizes, chunk, memory fraction, DSA backend, the DCP overrides (`dcp_replicate_q_proj`, `speculative_attention_mode=decode`), the overlap-schedule disable that PP forces, seed and addresses.

## Prefill-only setup (fake KV sender)

- Each arm is one PD **prefill** server (`--disaggregation-mode prefill`) with **no decode side**.
- Every request carries the fake bootstrap host `2.2.2.2`, its own `bootstrap_room` and `max_new_tokens 1`. The server therefore pairs it with the fake KV sender: the KV "transfer" completes at once.
- The measurement therefore covers prefill compute, scheduling and the cache tiers, with **no KV transfer cost and no decode**.
- Why this setup:
  - PP with DFLASH speculative decoding only runs as a PD prefill server; an aggregated server with `pp_size > 1` is rejected. So both arms run as PD prefill.
  - Leaving out decode and transfer isolates the prefill engine. Transfer cost differs by layout (a DCP1→DCP8 fan-out for PP8, a same-DCP page copy for TEP8+DCP8) and needs a separate end-to-end test.
- Commit 3 lets fake-host requests insert into the radix tree. Without it, the prefill-only server sees no prefix hits and no HiCache traffic, so the 90%-hit workload could not be run.

## Workload, client and procedure

Requests are random token ids with a fixed seed. Both arms get the identical request sequence.

| Name | Request length | Shared prefix | New tokens per request | Target hit | Prefix groups | Load |
|---|---|---|---|---|---|---|
| **W128_90** (main) | 131,072 | 117,760 | 13,312 | 89.8% | 16 | closed loop, concurrency 64, window 900 s |
| W32_0 | 32,768 | 0 | 32,768 | 0% | – | closed loop, concurrency 32, window 300 s |
| TTFT | as W128_90 | | | | | Poisson 3.246 req/s, window 900 s |

- **Prefix sizes**: multiples of 256, aligned to the Mamba track interval. The 16 prefixes (about 1.9M tokens) stay resident in device KV in both arms, so this measures compute, not cache-tier capacity.
- **TTFT rate**: 3.246 req/s = 0.7 × min(PP8, TEP8 saturated new tok/s from the screen) ÷ 13,312.
- **Procedure per point**: `/flush_cache`, warm the 16 prefixes once (`warm`), then the measured phase.
  - The first 45 s of each measured phase (`LEAD`) are excluded.
  - Each server boot also runs a probe request first. The first request after boot compiles the Triton KDA kernels (about 3–4 s per layer); the probe and warm absorb this, so it lands before any timed window.
- **Order** (`plans/bench-*.txt`, fixed in advance, identical for both arms):
  - boot 1: H1, H2 (W128_90), TTFT, S32a (W32_0);
  - restart;
  - boot 2: H3 (W128_90), S32b (W32_0).
  - The full plan runs in about 105 minutes per arm, including two boots (PP8 boots in about 950 s, TEP8+DCP8 in about 550–850 s).
- **Metric**: **new tok/s** = (Σ input − Σ cached tokens) ÷ window length, over requests that complete inside the window. The client takes `cached_tokens` from `meta_info`.
  - Cross-check: Σ`#new-token` over the rank-0 `Prefill batch` lines in the same window (PP0 for PP8, TP0 for TEP8). The two must agree within 2%; they agreed within 0.06%.
- **Validity of a sample**:
  - hit within 89.8% ± 1.5 pp (W32_0: under 0.5%);
  - zero failed requests;
  - at least 95% of the window's `Prefill batch` lines have `#queue-req ≥ 1` (saturated);
  - no OOM, no retraction.

Run (Slurm with a container plugin; submit both arms together):

```bash
export SGLANG_REPO MODEL_PATH DRAFT_MODEL_PATH DEEP_GEMM_DIR WORK
sbatch -A <account> -p <partition> $SGLANG_REPO/benchmark/k31_pp_prefill_repro/slurm_example.sh bench-pp8
sbatch -A <account> -p <partition> $SGLANG_REPO/benchmark/k31_pp_prefill_repro/slurm_example.sh bench-tep8
# afterwards
python3 $SGLANG_REPO/benchmark/k31_pp_prefill_repro/parse_results.py $WORK <bench-pp8-run> <bench-tep8-run> \
  --args-diff <bench-pp8-run> <bench-tep8-run>
```

Without Slurm, start the driver on both nodes at the same time inside the container. `RUN` must be a directory both nodes see.

```bash
# node 0
NODE_RANK=0 SGLANG_REPO=... MODEL_PATH=... DRAFT_MODEL_PATH=... PYTHONPATH=$DEEP_GEMM_DIR RUST_EXT_CACHE=... \
  bash run_arm.sh $RUN plans/bench-pp8.txt
# node 1: same command with NODE_RANK=1
```

Outputs are in `RUN/s<boot>/`:
- `<TAG>.jsonl` (per request) and `<TAG>.jsonl.sum.json` (summary);
- `phases.tsv` (phase start epochs, used to window the server log);
- `prefill-<host>.log` (batch lines) and `server-key-<host>.log`;
- `server-<host>.log` (the full log);
- `boot.tsv`.

## Expected results

Reference runs: both arms in parallel, on different node pairs, the plans in `plans/bench-*.txt`, commits 1–3 (fake-sender flag off).

| Point | PP8 new tok/s | TEP8+DCP8 new tok/s | Hit (both) |
|---|---|---|---|
| H1 (boot 1) | 79,354 | 60,079 | 89.8% |
| H2 (boot 1) | 79,872 | 60,987 | 89.8% |
| H3 (boot 2) | 79,044 | 60,328 | 89.8% |
| **W128_90 mean** | **79,423** | **60,465** | |
| S32a / S32b | 92,843 / 92,733 | 64,771 / 64,771 | 0% |
| TTFT point (λ = 3.246 req/s) | 44,078 | 44,063 | 89.8% |

- **W128_90**: R = 79,423 / 60,465 = **1.314**. Worst pairing (PP8 min ÷ TEP8 max) 1.296; best 1.329.
  - Noise floor, (max − min) / mean over the 3 samples: PP8 1.04%, TEP8 1.50%.
  - Client and server-log numbers agree within 0.06%.
  - Every saturated window had `#queue-req ≥ 1` on 100% of the rank-0 batch lines. There were no failed requests.
- **W32_0**: R = 92,788 / 64,771 = **1.43**.
- **TTFT** at the same Poisson rate (both arms unsaturated, about 44.1k new tok/s):
  - PP8: p50 1.63 s, p90 1.97 s;
  - TEP8+DCP8: p50 0.75 s, p90 1.61 s.
  - A PP8 request crosses 8 stages in sequence, so PP8 wins on throughput but not on single-request latency.
- Scope: one 8-GPU prefill server, prefill only (fake KV transfer, one output token), these two workloads. Real PD serving adds KV transfer and decode, which this setup does not measure.

## Profiling (GPU time per token)

The profile explains the gap: TEP8+DCP8 spends **130 GPU-ms per 1k new tokens** against **104** for PP8 (1.25×). About 24 of the 26 ms difference is TP and DCP communication.

| GPU-ms per 1k new tokens (sum over 8 GPUs) | PP8 | TEP8+DCP8 |
|---|---:|---:|
| Compute, excluding MoE and fused collectives | 62.7 | 65.7 |
| MoE (TEP8 includes EP dispatch/combine) | 25.9 | 28.9 |
| Fused TP collectives (GEMM+RS; attn-residual+AG) | 0 | 18.4 |
| NCCL-only (send/recv, all-gather, all-reduce, barriers) | 6.1 | 11.6 |
| GPU idle (incl. copy-only) | 9.2 | 5.1 |
| **Total** | **103.8** | **129.8** |

Capture:
- Each arm runs `plans/profile-{pp8,tep8}.txt`: the same boot, warm and W128_90 c64 point as the bench, with a 420 s window, then W32_0 c32.
- `profile/nsys_window.sh` reads `plans/profile.sched`. On every node it records nsys from +240 s to +254 s of the W128_90 phase (14 s, inside the window).
  - py-spy (optional) dumps the first scheduler's main thread from +340 s to +400 s, after nsys has stopped.
  - The unprofiled part of the window serves as the overhead reference. In the reference runs the step rate inside the nsys window matched the unprofiled window (PP8 4.78 vs 4.75 steps/s, TEP8 1.89 vs 1.86).

```bash
# the server runs under `nsys launch`; the sidecar starts and stops collection on that session
NSYS=/path/to/nsys NSYS_SCHED=$B/plans/profile.sched  NODE_RANK=<0|1> ... bash $B/run_arm.sh $RUN $B/plans/profile-pp8.txt
```

Equivalent nsys commands per node:

```bash
nsys launch --session-new=pf-$(hostname -s) --trace=cuda,nvtx,osrt python3 -m sglang.launch_server ...   # launch_server.py
nsys start --session=pf-$(hostname -s) -o $RUN/prof/nsys/P128-$(hostname -s) --force-overwrite=true \
  --sample=process-tree --sampling-period=16000000 --cpuctxsw=process-tree                    # at phase start + 240 s
nsys stop --session=pf-$(hostname -s)                                                          # 14 s later
nsys export --type=sqlite --include-blobs=true -o $RUN/prof/nsys/P128-$(hostname -s).sqlite \
  $RUN/prof/nsys/P128-$(hostname -s).nsys-rep                                                   # after the server stops
```

Analysis (stdlib only, any machine with the SQLite files):

```bash
B=$SGLANG_REPO/benchmark/k31_pp_prefill_repro
for R in $RUN_PP8 $RUN_TEP8; do
  mkdir -p $R/prof/ana
  for f in $R/prof/nsys/P128-*.sqlite; do
    python3 $B/profile/classify_gpu.py $f --anchor write_req_to_token_pool --out $R/prof/ana/gpu-$(basename ${f%.sqlite}).json
  done
done
python3 $B/profile/breakdown.py --arm pp8 16384 $RUN_PP8/prof/ana/gpu-P128-*.json --arm tep8 32768 $RUN_TEP8/prof/ana/gpu-P128-*.json
```

How the breakdown is computed:
- `classify_gpu.py` sweeps each GPU's kernel and memcpy intervals over the common steady window (0.5 s trimmed at each end). Each instant counts in exactly one class, in this priority order: MoE > other compute > fused TP collectives > NCCL send/recv > all-gather > all-reduce > barriers > copy-only > idle. The classes therefore sum to wall time.
- Kernels are classed by name regex (`CLASSES`).
- **Steps** = count of the `write_req_to_token_pool` kernel, which runs once per forward on every rank.
- `breakdown.py` divides each class by steps, sums over the 8 GPUs, and divides by the tokens per step: 16,384 for PP8 and 32,768 for TEP8. Every batch in a saturated window is a full chunk.
- Caveat: the fused TP collective kernels also do GEMM work that PP8 counts as compute. The communication cost proper is therefore between +5.5 and +24 GPU-ms per 1k tokens.
