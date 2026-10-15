# DSV4.1-Flash low-latency decode (GB300, TP4): branch `dsv41-lowlat-full`

This branch ports the muLLM DeepSeek-V4.1-Flash low-latency decode work into SGLang.
Every item is a separate commit behind its own opt-in env flag; with all flags off the
code paths are those of the base commit.

- **Base:** SGLang main `72730d6360a`.
- **Target:** c=1 decode, ISL 8K / OSL 1K, DSpark speculative decoding (block 5),
  synthetic acceptance 3.51, one GB300 node, TP4.
- **muLLM reference: 905 tok/s/user at c=1.**
- **This branch, all flags on: 899.7 tok/s/user at c=1** (step 3.93 ms; same-node table below).

## 1. The 26 muLLM items

### Ported (16), one opt-in flag each

| # | item | flag(s) | commits |
|---|---|---|---|
| 1 | attention-metadata build replayed from a CUDA graph | `SGLANG_OPT_DSV41_ATTN_METADATA_GRAPH` | `5070588f15d`, `a23fa9b2721` |
| 5 | wo_b GEMM epilogue publishes into the attention all-reduce mailbox | `SGLANG_OPT_DSV41_WO_B_PUBLISH` (needs `SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8`) | `e61ca5d1492`, `68e83a2b2ab` |
| 7 | DSpark aux hidden-state mean on the mHC stats stream (≤ 16 tokens) | `SGLANG_OPT_DSV41_AUX_MEAN_SIDE_STREAM` | `c01c5f1680d`, `df4a4ca0f39` |
| 8 | low-latency MXFP8 attention projections at 1–8 rows, pinned GB300 tactics; optional row-tiled path past 8 rows (off in `full.env`, section 4.3) | `SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8`, `SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8_PINNED_TACTICS` (+ `_MAX_ROWS`) | `3606eb81383`, `8e984b38993`, `501778fd2c2`, `92a4007953b` |
| 9 | low-M kernels for the compressor projections (1–16 rows in `full.env`) and the indexer `weights_proj` | `SGLANG_OPT_DSV41_LOW_LATENCY_COMPRESSOR` (+ `_MAX_ROWS`), `SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER_WEIGHTS` | `6d0b6328767`, `cdde701be38`, `566192e461b` |
| 10 | indexer query inputs on side streams | `SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER` | `d8d953512e4`, `2b79a8fe76e` |
| 13 | Engram lookup + WKV on a prefetch stream | `SGLANG_OPT_DSV41_PREFETCH_ENGRAM_WKV` (with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`) | `17f016a2231` |
| 14 | low-batch routed-expert GEMMs (TP MoE, 1–8 rows; 1–48 in `full.env`, several 8-row tiles per expert) | `SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM` (+ `_ADMIT`, `_FINALIZE`, `_MAX_ROWS`; needs `--ep-size 1`) | `be284254390`, `57b7307f5d6`, `c88807c7f53`, `ffb5baddeea` |
| 15 | the same low-batch kernels for the DSpark draft MoE (row bound from #14's `_MAX_ROWS`) | `SGLANG_OPT_DSV41_DRAFT_FAST_PATHS` | `cd7744e679e`, `ffb5baddeea` |
| 16 | draft stages on the target's fused all-reduce + mHC posts | `SGLANG_OPT_DSV41_DRAFT_FUSED_MHC_ALL_REDUCE` | `919e577da24` |
| 18 | draft context-KV projection on a side stream in the verify graph | `SGLANG_OPT_DSV41_DSPARK_CONTEXT_SIDE_STREAM` | `f0f3efa6c37` |
| 19 | post-verify Engram history commit inside the verify CUDA graph | `SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH` | `49f76718055` |
| 19 | temperature-sampling verify accept (softmax, rejection, bonus, accept-site TP broadcast, Engram + hidden commits) in one CUDA graph per batch size | `SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH` | `25eead7b3d5` |
| 20 | DSpark step inputs in one launch | `SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE` | `f24c6755e6c` |
| 22 | whole Markov chain in one vocab-sharded cooperative kernel per rank | `SGLANG_OPT_DSV41_DSPARK_SHARDED_MARKOV` (+ `_TIMEOUT_MS`) | `b49735e6d98`, `ba868ad8791`, `08fab16418c` |
| 24 | low-M FlashInfer cute-dsl LM head | `SGLANG_OPT_DSV41_LOW_LATENCY_LM_HEAD` | `8015e326ce3` |
| 26 | lean step prep (skip unchanged host re-uploads) | `SGLANG_OPT_DSV41_LEAN_PREP_INPUTS` | `ddfd88b1c21` |

Also on the branch: `SGLANG_OPT_DSV41_SMALL_M_HC_PRENORM` (`c229cb8b7a6`), muLLM's
1–16-row mHC coefficient path (one TF32 DeepGEMM prenorm GEMM), and `SGLANG_SPEC_TP_SYNC=rng`,
an existing SGLang setting that limits the speculative TP sync to the init and RNG sites.

Three more flags support temperature sampling:

- `SGLANG_OPT_DSV41_PRECOMPILE_SAMPLING_ACCEPT` (`de0f578c6dc`): compiles the sampling and mixed
  accept kernels for every decode batch size at startup (0.6 s), so the first temperature request
  does not stall on Triton compiles (1.0–1.5 s each before).
- `SGLANG_SIMULATE_ACC_SKIP_SAMPLING_ACCEPT` (`a65a799063a`, benchmark only, default off, not in
  `full.env`): under `SGLANG_SIMULATE_ACC_LEN` it skips the sampling accept whose length the
  simulation overwrites, as HIP always does, and accepts greedily. The muLLM port's synthetic
  acceptance mode still runs its full rejection sampler (target stats, per-draft test, bonus
  resample) and only replaces the decision, so keep this flag off when comparing with it.
- `SGLANG_OPT_DSV41_BLOCKWISE_REJECTION` (`d8fba160deb`, `8ab303f6c3d`): the muLLM GPU runner's
  block-wise rejection sampler (`rejection_sampler_utils.py` @ `b5a4d16f`, classic chain
  verification) for the eager accept, the sampling-accept graph and the precompile. Per-vocab-block
  target / draft stats, a one-warp accept, a block-parallel Gumbel-max resample; no probability
  tensors. Draws are counter-based (server key, device step counter, row, token), so TP ranks agree
  without a broadcast and `full.env` sets `SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample`. Top-k,
  top-p and block verification keep the classic kernel. Section 4.2.
  `SGLANG_DSV41_DEBUG_ACCEPT_TP_AGREEMENT` (`c2fb6527e7a`, debug only, host-blocking) all-gathers every
  sampling accept over TP and logs mismatches.

### Already covered in SGLang (10)

| # | item | what covers it in SGLang |
|---|---|---|
| 2 | lean sparse-MLA index table | the DSv4 FlashMLA path takes the SWA and compressed indices as separate arrays: there is no merged per-layer table and no per-layer zeros / qlen launch |
| 3 | per-KV-group sparse index builds | one `DSV4Metadata` per step; there are no per-group builds |
| 4 | MXFP8 attention input from the fused all-reduce + mHC | the fused kernel already emits it (`models/deepseek_v4_mhc.py`) |
| 6 | deferred finalize at the special boundaries | residual-only fused boundaries already exist (`models/deepseek_v4.py`, `models/deepseek_v4_mhc.py`) |
| 11 | candidate top-k over the valid prefix | radix top-k over each row's valid prefix (`kernels/ops/attention/dsv4/candidate_blocks.py`) |
| 12 | candidate expansion + schedule on a side stream | `layers/attention/dsv4/v41_indexer/sparse_table.py` already does this, top-k included |
| 17 | all draft layers' context KV in one GEMM | `commit_kv_proj_fused` stacks every stage's `wkv` into one MXFP8 GEMM (`kernels/ops/speculative/dspark/dspark_draft_model.py`) |
| 21 | fused Markov sampling | fused Triton step (`kernels/ops/speculative/dspark/markov_sample.py`) |
| 23 | draft logits without all-gather | `SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD` (default on) keeps the draft logits vocab-local; #22 reads that shard |
| 25 | one-shot logits all-gather | the target logits all-gather is already a one-shot NVLS multimem gather inside the verify graph |

## 2. Environment

- **Hardware:** one GB300 node, 4 GPUs (SM103). The pinned MXFP8 tactics (#8) are for GB300.
- **Container:** `lmsysorg/sglang:nightly-dev-cu13-20261012-e6072935` (arm64). Imported on an
  aarch64 node with
  `enroot import -o lmsysorg+sglang+nightly-dev-cu13-20261012-e6072935-arm64.sqsh "docker://lmsysorg/sglang:nightly-dev-cu13-20261012-e6072935"`.
  CMH copy: `/lustre/fsw/portfolios/coreai/users/yangminl/dsv41-lowlat-cmh/containers/lmsysorg+sglang+nightly-dev-cu13-20261012-e6072935-arm64.sqsh`.
- **Model:** DeepSeek-V4.1-Flash. CMH example:
  `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weiliangl/models/DeepSeek-V4.1-Flash`.
- **Running from source:** this checkout is mounted read-only at `/src` with `PYTHONPATH=/src/python`.
  The image sets `SGLANG_RUST_BUILD_MODE=never`, and a source mount needs `SGLANG_RUST_BUILD_MODE=auto`
  plus `CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup`. Keep `SGLANG_CACHE_DIR` on a shared
  persistent path so JIT kernels and FlashInfer autotune results are reused. A cold first start
  takes 16–24 min (JIT compile + autotune).

## 3. Reproduce

Everything below is in `benchmark/dsv41_lowlat/`: `dsv41-lowlat.sbatch` (one node runs the arms one
after another), `full.env`, `base.env`, `main.env`, `ab_bench.py` (bench client), `l6_gate.py`
(GSM8K gate) and `gsm8k200.jsonl` (the 200 pre-tokenized GSM8K prompts). Set the cluster paths at
the top of the sbatch (`ROOT`, `D`, `BENCH`, `MODEL`, `IMG`, `NSYS`).

**(a) Flags.** `benchmark/dsv41_lowlat/full.env`. Lines starting with `--` are extra server arguments;
the rest is server environment:

```
--ep-size 1
--cuda-graph-backend-prefill breakable
--cuda-graph-bs-prefill 128 256 512 1024 2048 3072 4096 6144 8192
--cuda-graph-prefill-max-context 262144
--cuda-graph-max-seq-len-prefill 262144
SGLANG_OPT_DSV41_PREFILL_ASYNC_H2D=1
SGLANG_OPT_DSV41_PRECOMPILE_PREFILL=1
SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM=1
SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM_MAX_ROWS=48
SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8=1
SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8_PINNED_TACTICS=1
SGLANG_OPT_DSV41_LOW_LATENCY_LM_HEAD=1
SGLANG_OPT_DSV41_LOW_LATENCY_COMPRESSOR=1
SGLANG_OPT_DSV41_LOW_LATENCY_COMPRESSOR_MAX_ROWS=16
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample
SGLANG_OPT_DSV41_DRAFT_FUSED_MHC_ALL_REDUCE=1
SGLANG_OPT_DSV41_SMALL_M_HC_PRENORM=1
SGLANG_OPT_DSV41_ATTN_METADATA_GRAPH=1
SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1
SGLANG_OPT_DSV41_LEAN_PREP_INPUTS=1
SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH=1
SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH=1
SGLANG_OPT_DSV41_PRECOMPILE_SAMPLING_ACCEPT=1
SGLANG_OPT_DSV41_BLOCKWISE_REJECTION=1
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
SGLANG_OPT_DSV41_AUX_MEAN_SIDE_STREAM=1
SGLANG_OPT_DSV41_PREFETCH_ENGRAM_WKV=1
SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER_WEIGHTS=1
SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER=1
SGLANG_OPT_DSV41_WO_B_PUBLISH=1
SGLANG_OPT_DSV41_DRAFT_FAST_PATHS=1
SGLANG_OPT_DSV41_DSPARK_CONTEXT_SIDE_STREAM=1
SGLANG_OPT_DSV41_DSPARK_SHARDED_MARKOV=1
SGLANG_OPT_DSV41_DSPARK_SHARDED_MARKOV_TIMEOUT_MS=20000
```

**(b) Server launch.** The launch recorded for the FULL arm (job 4492679, `launch.txt`), inside the
container. The flags from (a) are appended to the environment, and the env file's `--ep-size 1`
follows the default `--ep-size 4` (the last value wins):

```
PYTHONPATH=/src/python SGLANG_RUST_BUILD_MODE=auto SGLANG_CACHE_DIR=<shared cache> \
CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup \
SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token \
<flags from full.env> \
python3 -m sglang.launch_server --model-path /model --trust-remote-code --tp 4 --ep-size 4 --mem-fraction-static 0.8 --cuda-graph-max-bs-decode 8 --max-running-requests 16 --chunked-prefill-size 8192 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --watchdog-timeout 7200 --enable-cache-report --enable-metrics --host 0.0.0.0 --port 7900 --ep-size 1 --cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill 128 256 512 1024 2048 3072 4096 6144 8192 --cuda-graph-prefill-max-context 262144 --cuda-graph-max-seq-len-prefill 262144
```

The three `SGLANG_SIMULATE_ACC_*` variables give the synthetic acceptance (3.51 accepted tokens per
step on average) used for every performance number. The accuracy launch (d) drops them.

**(c) Benchmark.** The same-node table below came from one job with three arms:

```
env D=<out dir> BENCH=<checkout>/benchmark/dsv41_lowlat \
    ARMS="FULL:<this checkout>:$BENCH/full.env base:<dsv41-lowlat-moe checkout>:$BENCH/base.env main:<main 72730d6360a checkout>:$BENCH/main.env" \
    OSL=1024 ROUNDS=2 PROF=1 ACC=0 CONC=1,2,4,8 MAXBS=8 \
    sbatch --time=02:00:00 --export=ALL <account/partition/qos> benchmark/dsv41_lowlat/dsv41-lowlat.sbatch
```

Per arm, after the server is ready, the job runs the client twice: round 2 goes through the
concurrencies in reverse order. Then it records one bs=1 nsys window:

```
python3 ab_bench.py decode --label FULL --decode-conc 1,2,4,8 --decode-osl 1024 --out $OUT/bench
python3 ab_bench.py decode --label FULL --decode-conc 1,2,4,8 --decode-osl 1024 --reverse --out $OUT/bench
```

ISL is 8,192 random token ids (seeded per point, identical across arms). Each concurrency C gets
2×C warm-up requests, then 5×C timed requests, streaming, `ignore_eos`. Read the `SUMMARY` lines:
`tps_user_p50` (tok/s/user) and `step_p50_ms` (time per decode step, i.e. per verify + draft round;
tokens per step = `tok_per_chunk`). The client sends temperature 0 unless `BENCH_TEMPERATURE` is
set (section 4.1 uses `BENCH_TEMPERATURE=1.0`); `l6_gate.py` likewise reads `GATE_TEMPERATURE`.

**(d) GSM8K-200 greedy (cold), plus real acceptance at c=1.** A launch without the
`SGLANG_SIMULATE_ACC_*` variables (`PERF=0 ACC=1` in the sbatch):

```
python3 l6_gate.py --prompts gsm8k200.jsonl --gsm-cold-only --base-url $BASE_URL --out $OUT/gate
python3 ab_bench.py decode --label FULL-realacc --decode-conc 1 --out $OUT/bench
```

Harness: `l6_gate.py` sends the 200 pre-tokenized prompts to `/generate` (temperature 0,
512 new tokens, concurrency 64) and compares the last number in the output with the reference.
It is not `sgl-eval`. It was kept so the score compares with earlier runs of this work, which used
the same harness.

**(e) Baselines.** Identical launch and client. Either the branch `dsv41-lowlat-moe` @ `c88807c7f53`
with `base.env` (only #14 on, `--ep-size 1`), or SGLang main `72730d6360a` with `main.env` (no flags,
TP4 / EP4). On this branch, unsetting every flag in `full.env` gives the base-commit behaviour.

## 4. Results

The three arms. All three use the same container, model, launch command, bench client and node,
TP4 on one GB300 node, DSpark on, both SWA replay flags off.

| Arm | Code | Flags | What it measures |
|---|---|---|---|
| **FULL** | this branch, code head `68e83a2b2ab` | `full.env`: all 16 ported items on, `--ep-size 1` | everything ported |
| **base** | `c88807c7f53` (an ancestor of this branch: main `72730d6360a` + 4 commits, three for #14 and one for #8 whose flag stays off) | `base.env`: only #14 routed-expert GEMMs, `--ep-size 1` | the first port only; FULL − base = the other 15 items |
| **main** | SGLang main `72730d6360a`, unmodified | `main.env`: no flags, default EP4 | stock SGLang; FULL − main = the whole port |

Same node (nvl72d170-T17, job 4492679), three arms run one after another (FULL, base, main), two
rounds each (round 1 / round 2). tok/s/user p50 and decode step p50 in ms. ISL 8K / OSL 1K, synthetic acceptance
3.51, both SWA replay flags off, decode CUDA graphs up to bs 8.

| c | FULL tok/s/user | FULL step | base tok/s/user | base step | main tok/s/user | main step |
|---|---|---|---|---|---|---|
| 1 | 899.5 / 899.9 | 3.919 / 3.938 | 788.6 / 794.0 | 4.427 / 4.369 | 716.9 / 706.2 | 4.952 / 4.977 |
| 2 | 682.0 / 659.4 | 5.178 / 5.188 | 632.7 / 631.5 | 5.495 / 5.517 | 608.2 / 605.0 | 5.752 / 5.790 |
| 4 | 560.2 / 568.1 | 6.120 / 6.017 | 505.7 / 536.1 | 6.371 / 6.330 | 484.6 / 500.2 | 6.639 / 6.741 |
| 8 | 366.1 / 371.1 | 7.847 / 7.922 | 345.0 / 351.9 | 8.211 / 8.181 | 354.6 / 356.2 | 8.465 / 8.513 |

- **c=1: FULL 899.7 tok/s/user, step 3.929 ms (means of the two rounds); base 791.3 / 4.398 ms
  ⇒ −470 µs per step, +13.7 %; main 711.6 / 4.965 ms ⇒ −1.04 ms per step, +26.4 %.**
  muLLM reference: 905 tok/s/user at c=1.
- FULL vs base, step time: c=2 −323 µs, c=4 −282 µs, c=8 −311 µs.
- **GSM8K-200 greedy (cold), FULL: 196/200** (job 4491842, nvl72d149-T16, `l6_gate.py`). The base
  flags scored 196/200 in an earlier run with the same harness.
- **Real acceptance, FULL, c=1:** 5.38 tokens per step, 1,473.6 tok/s/user, step 3.972 ms (same job).

The table above is greedy (temperature 0) and predates the two sampling flags; both leave greedy
steps unchanged (3.928 vs 3.922 ms below).

### 4.1 Temperature 1.0 (sampling accept in the graph)

Same node (nvl72d009-T18, job 4501829), same source (`a65a799063a`) for both arms. **FULL** =
`full.env` without the two sampling flags, **+flags** = with them. ISL 8K / OSL 1K, c=1, synthetic
acceptance 3.51, AgentX server controls (`--context-length 1048576`, auto reasoning / tool parsers,
thinking on). tok/s/user p50 and step p50:

| arm | T 1.0 round 1 | T 1.0 round 2 | T 0 |
|---|---|---|---|
| FULL | 788.3 / 4.401 ms | 779.2 / 4.454 ms | 890.2 / 3.922 ms |
| +flags | 807.9 / 4.323 ms | 805.5 / 4.326 ms | 888.5 / 3.928 ms |

- **At T 1.0 the flags save 103 µs per step (+2.9 % tok/s/user); greedy is unchanged.**
- **AgentX c=1, 900 s, +flags** (same server, same harness as the FULL AgentX run of job 4495162):
  P90 interactivity 607.3 (FULL 602.7), P50 TPOT 1.26 ms (1.29), 148 requests, acceptance 3.51.
- **GSM8K-200 cold, +flags, real acceptance** (job 4500271): greedy 198/200, temperature 1.0 195/200.
- **Where T 1.0 still costs time.** Even with the flags, a T 1.0 step is 378 µs slower than a T 0 step
  (nsys, bs 1, clean steps, critical-path walk; job 4503775). The extra time splits three ways:
  - ≈ 90 µs: the sampling kernels, all on the critical path:
    - `speculative_sampling_classic_kernel`: 32 µs. One CTA per request walks the vocab twice.
    - Target and draft softmax: 38 µs. Each writes a full-vocab probability row.
    - Three NCCL broadcasts at the accept site (`SGLANG_SPEC_TP_SYNC=rng`): 14 µs.
    - Uniform draws and the bonus gather: 6 µs.
  - ≈ 110 µs: more idle time on the critical path (host side).
  - ≈ 180 µs: the routed-expert GEMMs (`dsv41_routed_expert_gemm1` / `2`) run longer, while the dense
    projections stay flat. On random-token prompts, sampled tokens route to more distinct experts
    than the repetitive greedy continuation does. This is a property of the workload, not of the
    sampler.
- The muLLM port's rejection sampler does this work in a block-parallel way. Per step it runs a
  per-vocab-block target stats kernel, a one-warp accept kernel, and a block-parallel Gumbel-max
  resample. It writes no probability tensors and needs no TP broadcast, because its draws are
  counter-based (seed, position).

### 4.2 Temperature 1.0: block-wise rejection sampler

**FULL** = `full.env` before this item (with the two sampling flags of 4.1, `SGLANG_SPEC_TP_SYNC=rng`);
**+block-wise** = plus `SGLANG_OPT_DSV41_BLOCKWISE_REJECTION=1` and
`SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample` (the current `full.env`). Same source (`8ab303f6c3d`),
same node (nvl72d132-T18) for jobs 4506292 and 4507852, recipe as 4.1. tok/s/user p50 and step p50:

| job | arm | T 1.0 round 1 | T 1.0 round 2 | T 0 |
|---|---|---|---|---|
| 4506292 | FULL | 813.1 / 4.343 ms | 804.9 / 4.344 ms | 897.1 / 3.955 ms |
| 4506292 | +block-wise | 834.4 / 4.222 ms | 829.9 / 4.225 ms | 877.6 / 3.998 ms |
| 4507852 | +block-wise | 838.8 / 4.217 ms | 832.4 / 4.212 ms | 877.6 / 3.961, 879.1 / 4.006 ms |
| 4507852 | FULL | 810.4 / 4.324 ms | 805.2 / 4.325 ms | 894.3 / 3.920, 896.1 / 3.955 ms |

- **At T 1.0 the sampler saves 110–120 µs per step (+2.6 % / +3.4 % tok/s/user).** Greedy steps do not
  run the new code. The T 0 rounds spread by 35–45 µs within an arm: synthetic acceptance changes the
  committed text, and with it the expert routing.
- **Sampling kernels per T 1.0 step (bs 1, torch profiler):** 98.5 µs → 14.9 µs. Before: classic rejection
  38.9, softmax 39.2, three NCCL broadcasts 14.6, uniform draws + bonus gather 5.9. After: stats 5.6,
  resample 4.1, accept 3.2, insert 2.0; no collective.
- **AgentX c=1, 900 s, same node:** +block-wise P90 interactivity 637.9, P50 / P90 TPOT 1.23 / 1.57 ms,
  steady TPOT (median chunk gap / acceptance) p50 1.216 ms. FULL: 605.2, 1.26 / 1.65 ms, steady 1.246 ms.
  ⇒ +5.4 % P90 interactivity, −0.030 ms per token (steady). Both runs valid, 148 requests, acceptance 3.51.
- **Accuracy and rank agreement** (job 4506291, real acceptance, broadcast off): GSM8K-200 cold greedy
  196/200, temperature 1.0 195/200. The debug check all-gathered 7,000 sampling accepts over the 4 ranks:
  0 mismatches. Acceptance on random-token prompts at T 1.0 is 1.05 tokens per step (classic: 1.06).
- **Correctness tests** (`test_dspark_blockwise_rejection.py`, GB300): the joint law of (accepted length,
  bonus token) matches the analytic chain speculative-sampling law (χ², 512 K draws) for the new and the
  classic sampler, at T 1 / 1, 0.7 / 1.3 and with a one-hot draft. With drafts drawn from q, the first
  emitted token follows p. A CUDA graph replay equals the eager call bit for bit.
- **What T 1.0 still costs** (+226 µs per step over the arm's own T 0): mostly the expert-GEMM content
  effect of random-token prompts (section 4.1). Re-reading job 4503775's windows with GPU idle defined
  as no kernel on any stream gives +36 µs of idle at T 1.0 (not the ≈ 110 µs of 4.1's path walk). The
  host was not late: kernels after a gap were launched before it began. The block-wise sampler removes
  the broadcast waits in that idle.

### 4.3 Past 8 rows: routed-expert GEMMs to 48 rows, compressor to 16

**FULL** = `full.env` without the two `_MAX_ROWS` lines; **NEW** = with them (the current `full.env`):
`SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM_MAX_ROWS=48` (#14 and #15) and
`SGLANG_OPT_DSV41_LOW_LATENCY_COMPRESSOR_MAX_ROWS=16` (#9 compressor). The routed-expert GEMM pair
already serves one expert and one row limit per CTA; a new route kernel (`route_wide.cu`, 9–64 rows)
gives an expert chosen by more than 8 rows ceil(count / 8) consecutive 8-row CTAs (`ffb5baddeea`).
Same source (`92a4007953b`), same node, job 4510101: FULL, NEW, FULL again, two rounds each (round 2
reversed), ISL 8K / OSL 1K, synthetic acceptance 3.51. Step p50 in ms (round 1 / round 2):

| c (rows) | FULL | NEW | FULL again | NEW vs mean FULL |
|---|---|---|---|---|
| 1 (6) | 3.911 / 3.916 | 3.944 / 3.908 | 3.942 / 3.961 | −0.007 (−0.2 %) |
| 2 (12) | 5.178 / 5.134 | 4.878 / 4.879 | 5.157 / 5.159 | **−0.278 (−5.4 %)** |
| 4 (24) | 5.971 / 6.025 | 5.709 / 5.732 | 5.965 / 5.990 | −0.267 (−4.5 %) |
| 8 (48) | 7.838 / 7.886 | 7.573 / 7.569 | 7.873 / 7.973 | −0.322 (−4.1 %) |

- tok/s/user p50, mean FULL → NEW: c=1 896 → 894, c=2 675 → 715 (+5.9 %), c=4 564 → 601 (+6.5 %),
  c=8 368 → 378 (+2.6 %). Output tok/s: c=2 +6.2 %, c=4 +3.4 %, c=8 0.0 %.
- **Where the 12-row step goes** (nsys, dependency-aware path, job 4507455): the FlashInfer routed MoE
  (`bmm` pair + routing kernel, 419 + 91 µs on path) becomes the NVRTC pair (307 µs). Expert GEMM time
  per layer is about the same (21 vs 22.5 µs); the saving is the routing kernel (185 → 108 µs) and
  less MoE on the path. The compressor part adds nothing measurable (3 calls per step at 9–16 rows).
- **GSM8K-200 greedy (cold):** NEW 196/200, FULL 197/200 (job 4510102, same node; real acceptance at
  c=1: NEW 5.62 tokens per step, step 3.969 ms; FULL 5.31, 3.994 ms).
- **Not in `full.env`: #8 past 8 rows** (`SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8_MAX_ROWS`, `92a4007953b`).
  FlashInfer's `cutedsl_low_latency` kernel runs one CTA per 8-row tile and reads each tile's activation
  scales from a 128-row atom of its own (its `reorder_scale_factors_per_cta`), so plain
  `mxfp8_quantize` scales give garbage past 8 rows. The flag quantizes 9..N BF16 rows into that layout
  (bit-equal to `mxfp8_quantize`) and lifts the runner's 8-row check. It is correct but neutral (step
  −0.06 ms at 12 rows, +0.01 at 24; job 4509909). `wqkv_a` and often `wo_b` get pre-quantized input
  in the plain layout and stay on cute-dsl. More importantly, the dense GEMMs past 8 rows run beside
  the mHC prenorm + sinkhorn chain, which becomes the critical path when they shrink (−308 µs dense on
  path, +266 µs mHC). Past 16 rows that chain is the next lever: in FULL, mHC grows by 0.92 ms from
  6 to 24 rows when the small-M path ends.
- **Kept at 8:** the low-latency MXFP8 path stays on cute-dsl past 8 rows (above). The indexer `wk` /
  `weights_proj` `ll_bf16` kernels lose to the JIT small GEMMs past 8 rows (+0.27 / +0.61 µs at 12
  rows). Swap-AB sparse decode attention costs about what FlashMLA split-KV does at 12 rows. The LM head
  keeps its 16-row bound (+7.5 µs at 24 rows).

### 4.4 Past 8 rows: Gluon sparse decode attention to 24 rows

`SGLANG_OPT_DSV41_SWAPAB_MAX_ROWS=24` (`f279bb82e21`) keeps decode / verify on the 16-head Gluon
swap-AB attention up to 24 rows. Past its bound, FlashMLA's sparse FP8 kernel runs on 64 query heads,
48 of them zero padding at TP4, at a flat ≈ 21 µs per layer from 12 to 48 rows. Gluon runs one CTA per
(row, 64-token KV tile): 10.5 µs at 12 rows, 15.1 at 24, slower than FlashMLA from 32 rows (second
wave). Error at the BF16 floor at 9–64 rows (FlashMLA 1.1–1.3× the floor). Same source, same node,
job 4514775: FULL (`full.env` without the line), NEW, FULL again, two rounds each (round 2 reversed),
ISL 8K / OSL 1K, synthetic acceptance 3.51. Step p50 in ms (round 1 / round 2):

| c (rows) | FULL | NEW | FULL again | NEW vs mean FULL |
|---|---|---|---|---|
| 1 (6) | 3.924 / 4.006 | 3.950 / 3.915 | 3.908 / 3.904 | −0.003 (−0.1 %) |
| 2 (12) | 4.817 / 4.894 | 4.482 / 4.526 | 4.863 / 4.900 | **−0.365 (−7.5 %)** |
| 4 (24) | 5.808 / 5.685 | 5.581 / 5.659 | 5.772 / 5.801 | −0.147 (−2.5 %) |
| 8 (48) | 7.542 / 7.538 | 7.657 / 7.557 | 7.540 / 7.521 | +0.072 (+1.0 %, code path unchanged) |

- tok/s/user p50, mean FULL → NEW: c=2 717 → 771 (+7.5 %); c=1 and c=4 within ±1 %; c=8 381 → 389
  (+2.0 %, one FULL-again round at 368). Output tok/s: c=2
  +5.6 %, c=4 +0.7 %. At c=4 the step gain does not reach TPOT (prefill of arriving requests dominates
  its spread).
- **Where the 12-row step goes** (nsys, dependency-aware path): FlashMLA 864 µs + its schedule kernel
  32 µs leave the path; Gluon partial + combine add 432; `wo_a_fused` loses 50 µs of PDL overlap
  (the Triton `_combine` does not trigger an early launch) and path idle grows 28 µs. Net ≈ −386 µs;
  traced step −412, client −365.
- **GSM8K-200 greedy (cold):** NEW 196/200 at the default 16 running requests (mostly 66–96 rows, so
  FlashMLA), 195/200 at `--max-running-requests 4` (every step ≤ 24 rows on Gluon); FULL 197/200 (4.3).

## 5. Known limits

- Every run has both SWA replay flags off (`--enable-encoder-swa-bounded-replay` and
  `--enable-decoder-swa-bounded-replay` not set).
- Decode CUDA graphs are captured up to bs 8 (`--cuda-graph-max-bs-decode 8`). The 1–8-row kernels
  (#8, #22), #14 and #15 (1–48 rows in `full.env`), the compressor part of #9 (1–16 rows) and the
  1–16-token paths (#5's mailbox, #7, #13, small-M mHC) fall back to the default kernels above their
  bounds. Past 16 rows the small-M mHC path ends, and the mHC prenorm + sinkhorn chain becomes the
  critical path (section 4.3).
- `--ep-size 1` (TP MoE) is required by #14 and #15; main runs its default EP4.
- #7 keeps muLLM's 16-token bound. Without it, c=8 regressed by about 0.9 ms per step.
- #5 with the pinned tactics (#8) needs commit `68e83a2b2ab`: the wo_b publisher's precompile runs
  under the pinned tactics, otherwise warmup fails with `Invalid low-latency GEMM tactic`.
- Fixed in `a3fe950efe6`: with `--max-running-requests 1`, warmup failed with the same error. The
  6-row decode autotune pass filed an unpinned tactic under FlashInfer's bucket 8. The pins now cover
  every autotune pass. mrr 1 / 2 / 4 / 16 all start; c=1 at mrr 1 is 3.99 ms per step vs 3.93–3.98 ms
  at mrr 16 (two launches each), within the launch-to-launch spread of re-tuned tactics.
- c=4 tok/s/user is bimodal across rounds and arms (about 505–510 or 536–568) at a nearly equal step
  time; compare c=4 by step time.
- At temperature 1.0, with the block-wise sampler, the sampling accept costs ≈ 15 µs of kernels per
  step (section 4.2). Steps with top-k / top-p, logit adjustments, grammar, mixed greedy and
  sampled rows, compact verify, or more than 8 rows keep the eager accept.
- The performance numbers use synthetic acceptance. Real acceptance on random-token prompts is higher
  (5.38 tokens per step at c=1).
