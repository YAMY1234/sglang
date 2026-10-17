P4 sigmoid RMSNorm epilogue (default off)

Run in the GB300 CUDA 13 service container with the released model mounted at `/model`:

```bash
bash p4/run_gate.sh 0
bash p4/micro.sh 0
```

`P4_OUT` selects the evidence directory. Gate writes build/resource logs and `gate.json`; micro requires a complete bitwise certificate for the same extension path before timing. The source hash in the extension name pins the CUDA binary. The timing estimator is the median of 50 samples in alternating order, after a 512 MiB L2 flush outside each event interval. Each graph contains 36 distinct state slices and service-sized BF16 out-projection spacers (hidden size 4096). TP2 buckets are 1/2/4/8; TP1 buckets 1/2/3 follow the production whitelist. Raw samples are retained.

The module gate covers BF16 bias, every production K1 bucket, INVALIDATE on/off, 64 changing-input eager/graph steps, padding, slot-0 replacement, repeated expiry and every actual per-layer norm weight. For an immutable original-K1 crosscheck, set `P4_BASELINE_TREE` to the archived v7 source tree. To also replay service snapshots, set `P4_SERVICE_DIR` to their directory. A synthetic-only certificate explicitly reports that no service snapshot was supplied.

For integration, use `p4/hjob.sh` with holder headers `--gpus 1 --excl --minutes 15`; `qsubmit.sh` records HOLDER_SOURCE, from which the script resolves its immutable source tree. `p4/job.sbatch` / `micro.sbatch` are bounded normal-QOS alternatives; CMH physically requires a four-GPU node even though these development jobs run on one logical GPU. Record allocated GPU-hours accordingly.

The production switch is `SGLANG_GDN_FACTORED_STEP_FUSE_NORM=1`. Fusion is limited to the timed, admitted buckets: HV24/TP2 B1/B2/B8 and HV48/TP1 B1/B2/B3. B4 and the unmeasured TP2 buckets keep the independent norm and the original K1 CUDA route; five iterations did not meet the unchanged B4 timing gate. The selected kernel is the simpler third iteration (source identical to 31b0e05da95), not the rejected warp3/writeback variants. It requires the existing qualified K1 CUDA route, a sigmoid norm-before-gate with BF16 weights, contiguous BF16 z, and the model shared patches. Unsupported buckets and other forward modes retain their norm call. The backend unwraps HybridLinearAttnBackend; failure to load the extension does not permit skipping normalization. Activation: `FACTORED_STEP_FUSE_NORM_ACTIVE gate=sigmoid ...`.

Actual request capture (diagnostic service only): enable fusion, set `SGLANG_GDN_P4_CAPTURE_DIR`, `SGLANG_GDN_P4_CAPTURE_FLAGFILE`, and optionally `SGLANG_GDN_P4_CAPTURE_LIMIT` (default 36 per `(B, HV)` per process). Use `--disable-cuda-graph`; after readiness, create the flag file, send real requests, then remove the flag. The hook skips graph capture, records only selected slots plus slot 0, and saves current post-conv inputs, z, norm weight and state. Do not enable this hook in timing. Replay with:

```bash
python p4/gate.py --service-only --service-dir /path/to/dumps --out service-gate.json --model /model
```

Shared commits: `5af94800f57` (radix + qwen3_next + the actual served qwen3_5 forward), `dd8d55ba1de` (integration review: early-return on extend). Integration owns their merge and the formal service C1/C8 A/B gates.
