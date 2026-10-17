P4 sigmoid RMSNorm epilogue (default off)

Run in the GB300 CUDA 13 service container with the released model mounted at `/model`:

```bash
bash p4/run_gate.sh 0
bash p4/micro.sh 0
```

`P4_OUT` selects the evidence directory. Gate writes build/resource logs and `gate.json`; micro requires a complete bitwise certificate for the same extension path before timing. The source hash in the extension name pins the CUDA binary. The timing estimator is the median of 50 samples in alternating order, after a 512 MiB L2 flush outside each event interval. Each graph contains 36 distinct state slices and service-sized BF16 out-projection spacers (hidden size 4096). TP2 buckets are 1/2/4/8; TP1 buckets 1/2/3 follow the production whitelist. Raw samples are retained.

The module gate loads the checked-in service layout metadata when available;
`P4_Z_LAYOUTS` selects a newer capture catalog. It covers BF16 bias, every production K1 bucket, INVALIDATE on/off, 64 changing-input eager/graph steps, padding, slot-0 replacement, repeated expiry and every actual per-layer norm weight. For an immutable original-K1 crosscheck, set `P4_BASELINE_TREE` to the archived v7 source tree. To also replay service snapshots, set `P4_SERVICE_DIR` to their directory. A synthetic-only certificate explicitly reports that no service snapshot was supplied.

For integration, use `p4/hjob.sh` with holder headers `--gpus 1 --excl --minutes 15`; `qsubmit.sh` records HOLDER_SOURCE, from which the script resolves its immutable source tree. `p4/job.sbatch` / `micro.sbatch` are bounded normal-QOS alternatives; CMH physically requires a four-GPU node even though these development jobs run on one logical GPU. Record allocated GPU-hours accordingly.

The production switch is `SGLANG_GDN_FACTORED_STEP_FUSE_NORM=1`. Fusion is limited to the timed, admitted buckets: HV24/TP2 B1/B2/B4/B8 and HV48/TP1 B1/B2/B3. Both matched current-kernel runs pass the unchanged timing gate. Unmeasured TP2 B3/B5/B6/B7 keep the independent norm and original K1 CUDA route. Earlier B2/B4 failures remain in docs/231; B4 was added only after two current-kernel measured-layout passes. The epilogue keeps the third iteration's 32-lane form and adds bounded token/head-stride indexing; the warp3/writeback variants were rejected. It requires the existing qualified K1 CUDA route, a sigmoid norm-before-gate with BF16 weights, BF16 z with inner stride 1 and 16-byte aligned head starts (token/head strides are read in the kernel), and the model shared patches. Unsupported buckets and other forward modes retain their norm call. The backend follows the decode and linear backend wrappers; failure to load the extension does not permit skipping normalization. Activation: `FACTORED_STEP_FUSE_NORM_ACTIVE gate=sigmoid ...`.

Actual request capture (diagnostic service only): enable fusion, set `SGLANG_GDN_P4_CAPTURE_DIR`, `SGLANG_GDN_P4_CAPTURE_FLAGFILE`, and optionally `SGLANG_GDN_P4_CAPTURE_LIMIT` (default 36 per `(B, HV)` per process). Use `--disable-cuda-graph`; after readiness, create the flag file, send real requests, then remove the flag. The hook skips graph capture, records only selected slots plus slot 0, and saves current post-conv inputs, z, norm weight and state. Do not enable this hook in timing. Replay with:

```bash
python p4/gate.py --service-only --service-dir /path/to/dumps --out service-gate.json --model /model
```

Shared commits: `5af94800f57` (radix + qwen3_next + the actual served qwen3_5 forward), `dd8d55ba1de` (integration review: early-return on extend). Integration owns their merge and the formal service C1/C8 A/B gates.

The stride-aware repair (lead 20:09) uses no producer layout change and no z copy
in the fused arm. The separate arm retains the model's existing reshape and any
copy it causes; that cost is included in the separate-chain time. Snapshots use `p4-service-step-v2` and preserve z shape,
stride and storage offset. Service replay writes `z-layouts.json` beside its
certificate; rerun `P4_Z_LAYOUTS=/path/to/z-layouts.json bash p4/run_gate.sh 0`,
then `bash p4/micro.sh 0`. The gate adds those exact stride/offset patterns to
all B/INVALIDATE/eager/graph cases; micro uses the primary patterns from that
same certificate, records metadata and per-arm layout-copy bytes. Fused copy time is zero;
separate copy time is included in the chain and is not separately isolated. Batch
sizes absent in the capture reuse the captured strides and are identified by
`measured_shape`. The built-in token/head-gap patterns are labelled synthetic
preflight and cannot substitute for a measured-layout admission result. The
norm reference mirrors the production standalone norm's 2D z reshape, including
the original copy for token-strided views. The fused model skips that unused
reshape (shared patch a15fbde8c2a). Prior contiguous-layout timing remains historical
evidence; stride-aware service admission requires requalification.

TP1 remains synthetic-only for this stage. Micro retains both its contiguous
control and token-gap stress layout; the contiguous result qualifies its
buckets for the current strided kernel, independently of the TP2 service layout.

The default fused projection/Conv1D route hands the backend `(qkvz, ba)` before
z exists. The metadata promise supports this producer; `forward_decode` passes
its generated z to the same strided K1 epilogue and returns it for shape handling.
Diagnostic capture logs `P4_PRODUCED_Z` even when fusion is rejected, so source
layout metadata cannot silently disappear behind the deferred producer.

Evidence: 180 synthetic cases, 156 actual service-replay cases (144 snapshots
from TP2 B1/B2 plus dispatch checks), then 358 combined cases with the measured
layout; all bitwise. Twelve CUDA instances use 128 registers and no spill. The
current TP2 layout is stride (3072,128,1), offset 0. B4/B8 timing sweeps use this
captured pattern; they are not B4/B8 service activation captures. TP1 is synthetic
only. Two independent 50-sample matched runs pass for every admitted bucket;
formal C1/C8 service A/B and integration approval remain integration-line work.
