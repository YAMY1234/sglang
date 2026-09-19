# X1 · DeepSeek-V4-Flash prefill-only PP anomaly decomposition

## 0. Arm table

All rows use upstream `main@3a64faa1f22a`, the original `DeepSeek-V4-Flash-MTP`
draft shard, no DSpark, chunk/max-prefill 8192, one discarded 64-prompt warmup,
then three 128-prompt measurements. `tok/s/GPU`, TTFT and KV capacity are pending.

| Arm | Topology | GPUs | `SGLANG_PP_COMM_OVERLAP` | breakable prefill graph | tok/s/GPU (3-run median; range) | TTFT P50 / P99 | capture line | actual PP max micro-batch | KV capacity | job |
|---|---|---:|---|---|---:|---:|---|---:|---:|---:|
| A | TP4 / EP4 | 4 | unset | automatic (non-PP) | 10,320.77 (958.48) | 6,269.42 / 6,678.82 ms | **NO** — `DeepSeek-V4 (heavy capture-pool memory pressure)` → prefill backend disabled | n/a | 23,122,944 tokens | 796765 |
| B | DEP4 | 4 | unset | automatic (non-PP) | 10,046.95 (2,349.79) | 5,680.95 / 8,137.62 ms | **NO** — same DSV4 auto-disable | n/a | 22,888,960 tokens | 796780 |
| C | TP2 / EP2 | 2 | unset | automatic (non-PP) | pending | pending | pending | n/a | pending | pending |
| D | TP1 / EP1 × PP2 | 2 | unset | breakable | pending | pending | pending | pending | pending | pending |
| E-ov1 | TP2 / EP2 × PP2 | 4 | `1` | breakable | pending | pending | pending | pending | pending | pending |
| E-base | TP2 / EP2 × PP2 | 4 | unset | breakable | pending | pending | pending | pending | pending | pending |
| E-no-BCG | TP2 / EP2 × PP2 | 4 | unset | omitted | pending | pending | pending | pending | pending | pending |
| F-ov1 | TP1 / EP1 × PP4 | 4 | `1` | breakable | pending | pending | pending | pending | pending | pending |
| F-base | TP1 / EP1 × PP4 | 4 | unset | breakable | pending | pending | pending | pending | pending | pending |
| F-no-BCG | TP1 / EP1 × PP4 | 4 | unset | omitted | pending | pending | pending | pending | pending | pending |

## 1. Status lines

- 2026-09-19 14:53 PDT | PREP / first report checkpoint | job n/a | submitted n/a / started 14:46 / waited 0 min, reasonable (no X1 GPU job submitted); live AGA login safety check: user slice 49/300 tasks and 3.95/32.21 GB, batch idle=1361; M5 jobs are a different line and are not counted as X1 jobs | read protocol v2 and M5/docs 13/25/50 evidence; fixed X1 source, target/draft provenance and 10-row switch matrix; confirmed nominal ETA 18:16 PDT (3.5 h from instruction): setup 20 min + five two-job waves at about 18 min=90 min + profile/trace analysis 45 min + report/push 25 min + retry/queue buffer 30 min = 210 min; actual 7 min vs expected 20-min setup, 13 min ahead
- 2026-09-19 14:57 PDT | A · TP4/EP4 | job 796765 | submitted 14:56:40 / started 14:56:55 / waited 0.25 min, reasonable (`Reason=None`, `LastSchedEval=14:56:55`, `Priority=131562`, start-time batch idle=1349) | `qos=short`, one node × 4 GPU, MTP-only, chunk=8192, warmup=64 discarded, formal=128×3; source clone verified at `3a64faa1f22a`; attempted paired B submission was rejected before job creation by the account-wide `QOSMaxSubmitJobPerUserLimit`, so X1 has one active job and will fill the second slot when the shared submit count falls | ETA 18:16 PDT; actual 11 min vs expected 20-min setup, 9 min ahead, but the account-wide submit cap may consume queue buffer
- 2026-09-19 15:00 PDT | B · DEP4 | job 796780 | submitted 14:58:57 / started 14:59:36 / waited 0.65 min, reasonable (`Reason=None`, `LastSchedEval=14:59:36`, `Priority=131562`, pre-start batch idle=1338) | second submit succeeded after one shared-QOS slot opened; one node × 4 GPU, same source/MTP/workload as A; X1 now has exactly two running jobs | ETA 18:16 PDT; actual 14 min vs expected first pair starting by 15:06, 6 min ahead
- 2026-09-19 15:13 PDT | A · TP4/EP4 complete | job 796765 | submitted 14:56:40 / started 14:56:55 / waited 0.25 min, reasonable / ended 15:12:28 (`COMPLETED 0:0`, elapsed 15m33s) | three runs=`9364.80,10323.28,10320.77 tok/s/GPU`, median 10320.77, range 958.48; TTFT P50/P99 medians=6269.42/6678.82 ms; KV=23,122,944; server-info says prefill backend=`disabled` and no prefill capture begin/end exists | ETA 18:16 PDT; actual 27 min vs planned setup+first pair 38 min, 11 min ahead
- 2026-09-19 15:14 PDT | B · DEP4 measurements complete / postprocess FAIL | job 796780 | submitted 14:58:57 / started 14:59:36 / waited 0.65 min, reasonable / ended 15:13:51 (`FAILED 127:0`, elapsed 14m15s) | the discarded warmup and all three formal runs completed before failure: `8735.59,11085.38,10046.95 tok/s/GPU`, median 10046.95, range 2349.79; TTFT P50/P99=5680.95/8137.62 ms; KV=22,888,960. Failure occurred only after run 3 because I replaced the still-open sbatch file while adding the profile parser, so bash read a corrupted next token (`on3`); summary was deterministically reconstructed from the three saved JSONL files and ready server-info. No server/workload failure and no rerun needed | ETA 18:16 PDT; actual 28 min vs planned first-pair completion around 15:24, 10 min ahead
- 2026-09-19 15:15 PDT | C · TP2/EP2 and D · TP1/EP1×PP2 submitted | jobs 796886 / 796887 | both submitted 15:15:07 / starts — (pending) / waited 0.3 min at checkpoint, reasonable (both `Reason=None`; C `LastSchedEval=15:15:23`, D `LastSchedEval=15:15:07`, both `Priority=131562`, batch idle=1327) | AGA short QOS rejects a 2-GPU allocation with `QOSMinGRES`; both jobs therefore reserve the required four GPUs but each inner `srun` exposes and uses exactly two GPUs. The two 2-GPU arms were submitted together; X1 active jobs=2 | ETA 18:16 PDT; actual 29 min vs expected second wave by 15:24, 9 min ahead
- 2026-09-19 15:16 PDT | C / D started | jobs 796886 / 796887 | both submitted 15:15:07 / both started 15:15:59 / both waited 0.87 min, reasonable (`Reason=None`, `LastSchedEval=15:15:59`, `Priority=131562`) | C on `nvl72d097-T17`, D on `nvl72d032-T16`; each inner step uses two GPUs and X1 running count is exactly two | ETA 18:16 PDT; actual 30 min vs expected second-wave start 15:24, 8 min ahead

## 2. Exact commands

Pending source staging and final script capture. The immutable inputs are:

- Source: clone `https://github.com/sgl-project/sglang.git`, checkout exactly
  `3a64faa1f22a86abd37a759c84267d929e820d5b`.
- Target: `$U/pp-verify-20260917/dsv41-flash/models/DeepSeek-V4-Flash-DSpark`.
- Draft: `$U/pp-verify-20260917/dsv41-flash/models/DeepSeek-V4-Flash-MTP`.
  The lead text calls this the shard “in hf-cache”; bounded AGA inspection shows the
  materialized shard at this `models/` path, which is also the path used by the
  prior passing jobs 773019/773020. The `hf-cache/hub` directory only contains
  the target repository cache.
- Container and isolated kernel dependency: the same image and
  `sglang-kernel==0.4.7` installation used by M5.
- MTP flags: `--speculative-draft-model-path /draft --speculative-algorithm EAGLE
  --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4`; every PP arm also exports
  `SGLANG_ENABLE_PP_SPEC=1`.
- Slurm: one node, `qos=short`, time limit at most 1:55, and no
  `--nice`/`--hold`/`--dependency`/`--begin`.

## 3. Decomposition and conclusion

Pending controlled measurements. The registered comparisons are:

1. C vs A isolates the TP-degree effect (TP2 vs TP4, both PP1).
2. D vs C isolates adding one PP boundary on the same two GPUs while changing
   only TP2/EP2 to TP1/EP1×PP2.
3. E vs D measures the TP2×PP2 combination against TP1×PP2; F vs E then measures
   the PP-level/TP trade from TP2×PP2 to TP1×PP4 at fixed four GPUs.
4. E-ov1 and F-ov1 each collect at least 60 seconds of per-stage trace. The
   analysis will split receiving wait, exposed CPU scheduling, and GPU kernel
   time without changing product code.
5. If E remains anomalously low, the trace/log audit will compare per-GPU expert
   weight ownership and MoE kernels for EP2 against the EP1/EP4 rows, including
   whether dispatch selected different MegaMoE, CuteDSL, or FlashInfer paths.

No SGLang product code is changed by this task.

## 4. NEED_LEAD

None at this checkpoint.
