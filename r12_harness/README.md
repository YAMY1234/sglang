# R12 harness

This directory contains the submit-time harness for R12 job7b/job7c.

- `render_r12_launches.sh`: freezes both ranks of every prefill/decode service generation and every router command for the whole-group restart lifecycle.
- `preflight_r12.sh`: performs local/AGA syntax, declaration-dependency static sweep, proof diff, feature support, and zero-uncovered-function checks.
- `check_feature_support.py`: fails closed on model/KV-pool-sensitive launch
  features.  For Kimi-K3 it proves that the target is a hybrid MLA pool,
  records the fixed-source non-MLA staging guards, and permits only
  `SGLANG_DISAGG_STAGING_BUFFER=0` on both prefill and decode.  Register and
  prove any future model-sensitive env/flag here before enabling it.
- `run_r12_job7.sbatch`: common four-node job; select job7b with `R12_PP_CHUNK=8192` or job7c with `R12_PP_CHUNK=16384`.
- `run_r12_role.sh` and `r12_launch_lib.sh`: shared two-node rank launch construction and actual-vs-rendered byte comparison.
- `r12_workflow.sh`: the shared JOB_START-to-R12_SUMMARY control graph used
  by both allocated jobs and the CPU-only walkthrough.
- `prewarm_r12_cache.sh`: materializes Kimi remote tokenizer code with one
  process per service/role/node-rank cache before local TP workers start; its
  `encoding_k3.py` and `tokenization_kimi.py` hashes are retained as proof.
- `walkthrough_r12.sh`: runs the real B/A/B-repeat/A-repeat whole-group
  functions under `set -euo pipefail`, stubbing only external processes. A
  `trap DEBUG` execution list is compared with the sourced `declare -F` delta;
  uncovered functions must be zero before summary and EXIT cleanup pass.
- `static_sweep_r12.py`: rejects any same-line `local`/`declare`/`export`/
  `readonly` declaration whose later assignment references an earlier one.
- `analyze_r12.py` / `summarize_r12.py`: point and job summaries. Scheduler
  `Prefill batch` cadence is the mechanism basis (PP0--PP3 and TEP TP0--TP7).

Local preflight:

```bash
bash r12_harness/preflight_r12.sh r12_harness/local-dry-run
bash r12_harness/walkthrough_r12.sh /new/path/walkthrough-v2
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

Every service generation starts prefill, decode, and router together and stops
all three before the next generation. The whole job reuses one bootstrap port;
renderer and walkthrough proofs reject split P/D ports, resident decode,
handoff probes, fallback generations, and multiple bootstrap generations.
Startup readiness has no harness timeout: it waits until health passes or an
external service process exits and emits a progress line every 60 seconds.
An exited group is stopped and retried at most twice with the same frozen
configuration. Explicit OOM evidence is reported as `NEED_LEAD`; the harness
never changes `mem-fraction-static` automatically.

The wrappers never enforce a GPU-hour or elapsed-time budget: they contain no
budget watchdog and never cancel the Slurm job or interrupt a running service
or benchmark because of time.  Qwen E2E requests 01:30:00 and Kimi requests
02:00:00.  The only time-aware workflow branch is a pre-start check that may
skip the optional Kimi A-C16-repeat when fewer than 30 minutes remain.
