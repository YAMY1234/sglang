# R12 harness

This directory contains the submit-time harness for R12 job7b/job7c.

- `render_r12_launches.sh`: freezes both ranks of every prefill/decode service generation and every router command, including the normal resident-decode path and the whole-group restart fallback.
- `preflight_r12.sh`: performs local/AGA syntax, token-diff, fallback-path, and analyzer-fixture checks.
- `check_feature_support.py`: fails closed on model/KV-pool-sensitive launch
  features.  For Kimi-K3 it proves that the target is a hybrid MLA pool,
  records the fixed-source non-MLA staging guards, and permits only
  `SGLANG_DISAGG_STAGING_BUFFER=0` on both prefill and decode.  Register and
  prove any future model-sensitive env/flag here before enabling it.
- `run_r12_job7.sbatch`: common four-node job; select job7b with `R12_PP_CHUNK=8192` or job7c with `R12_PP_CHUNK=16384`.
- `run_r12_role.sh` and `r12_launch_lib.sh`: shared two-node rank launch construction and actual-vs-rendered byte comparison.
- `analyze_r12.py` / `summarize_r12.py`: point and job summaries. Scheduler
  `Prefill batch` cadence is the mechanism basis (PP0--PP3 and TEP TP0--TP7).

Local preflight:

```bash
bash r12_harness/preflight_r12.sh r12_harness/local-dry-run
```

The frozen product source `cb0b3498fcc2f398229b0b8cb9df0a5825e1438a`
does not contain R7 instrumentation. The harness does not inject or synthesize
events: it reports `trace_mode=native_logs`, `TRACE_COMPLETENESS=N/A_NO_TRACE`,
and `kv_transfer=TRACE_UNAVAILABLE`. The hard request gate is exactly 480/480
completed with zero incomplete requests.

Kimi-K3 uses `HybridLinearKVPool(use_mla=True)` with an inner
`MLATokenToKVPool`.  Fixed cb0b and upstream/main explicitly support the
disaggregation staging feature only for non-MLA pools, so this comparison sets
`SGLANG_DISAGG_STAGING_BUFFER=0` identically on prefill and decode.  Both the
CPU preflight and every allocated job emit `feature-support-proof.json` before
server launch.

The image's native `sglang-kernel==0.4.7` is below cb0b's hard minimum. The
job therefore installs `sglang-kernel==0.4.8` into job-local `pipdeps/` while
keeping `flashinfer-python==0.6.18`; the EXIT trap removes that directory.

Submit both jobs with `--segment=4` and no nodelist under AGA's
`topology/block` plugin. Before source extraction, each job validates one rack,
records it under `runtime/rack-<job>.txt`, excludes that rack from a pending
peer, and requeues the higher job ID if simultaneous starts collide on a rack.
The local `scontrol update` field name is `ExcNodeList`.
