# X1 · DeepSeek-V4-Flash prefill-only PP anomaly decomposition

## 0. Arm table

Unless a row is explicitly marked consolidated-v5, authoritative rows use
upstream `main@3a64faa1f22a`. All use the original `DeepSeek-V4-Flash-MTP`
draft shard, no DSpark, chunk/max-prefill 32768, one
discarded 64-prompt warmup, then three 128-prompt measurements. The chunk value
was corrected by lead protocol v2.1 at 16:19 PDT; the earlier chunk-8192 data are
retained below only as invalidated provenance and are excluded from conclusions.

| Arm | Topology | GPUs | source | mem frac | chunk | `SGLANG_PP_COMM_OVERLAP` | breakable prefill graph | tok/s/GPU (3-run median; range) | TTFT P50 / P99 | formal prefill graph True/False | actual PP max micro-batch | KV capacity | job |
|---|---|---:|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|
| A | TP4 / EP4 | 4 | main | 0.9 | 32768 | unset | automatic (non-PP) | **21,392.27** (36.33) | 3,013.18 / 3,508.23 ms | NO — DSV4 auto-disable | n/a | 23,122,944 | 797385 |
| B | DEP4 | 4 | main | 0.9 | 32768 | unset | automatic (non-PP) | **41,874.50** (9,774.57) | 1,358.00 / 2,014.07 ms | NO — DSV4 auto-disable | n/a | 22,888,960 | 797386 |
| C | TP2 / EP2 | 2 | main | 0.9 | 32768 | unset | automatic (non-PP) | **36,762.18** (22.12) | 3,530.21 / 4,008.73 ms | NO — DSV4 auto-disable | n/a | 18,454,528 | 797479 |
| D | TP1 / EP1 × PP2 | 2 | main | 0.9 | 32768 | unset | breakable; cap 4096 recovery | **29,075.02** (102.99) | 4,485.22 / 4,878.73 ms | captured; hot-step ratio not instrumented | 128 | 34,293,248 | 797549 (default-cap failure 797480) |
| E-ov1-mem09 (feasibility) | TP2 / EP2 × PP2 | 4 | main | 0.9 | 32768 | `1` | breakable; cap 4096 | **infeasible** | n/a | captured, then warmup OOM | 128 | 43,634,176 | 797652 (default-cap failure 797550) |
| F-ov1-mem09 (extra evidence) | TP1 / EP1 × PP4 | 4 | main | 0.9 | 32768 | `1` | breakable; cap 4096 | **39,385.86** (1,730.17) | 1,533.69 / 2,066.00 ms | **0 / 416** | 64 | 77,515,520 | 797653 |
| E-ov1 | TP2 / EP2 × PP2 | 4 | main | 0.5 | 32768 | `1` | breakable; cap 4096 | **18,046.81** (20.15) | 3,589.16 / 3,951.86 ms | YES capture; **0 / 204** hot steps | 128 | 19,723,520 | 797742 |
| F-ov1 | TP1 / EP1 × PP4 | 4 | main | 0.5 | 32768 | `1` | breakable; cap 4096 | **38,334.15** (2,870.14) | 1,559.69 / 2,068.12 ms | YES capture; **0 / 416** hot steps | 64 | 34,345,984 | 797745 |
| E-base | TP2 / EP2 × PP2 | 4 | main | 0.5 | 32768 | unset | breakable; cap 4096 | pending | pending | pending | pending | pending | pending |
| F-base | TP1 / EP1 × PP4 | 4 | main | 0.5 | 32768 | unset | breakable; cap 4096 | pending | pending | pending | pending | pending | pending |
| E-no-BCG | TP2 / EP2 × PP2 | 4 | main | 0.5 | 32768 | unset | omitted | pending | pending | pending | pending | pending | pending |
| F-no-BCG | TP1 / EP1 × PP4 | 4 | main | 0.5 | 32768 | unset | omitted | pending | pending | pending | pending | pending | pending |
| E-v5-cap8k | TP2 / EP2 × PP2 | 4 | v5 | 0.5 | 32768 | `1` | breakable; cap 8192 | pending | pending | pending | pending | pending | pending |
| F-v5-cap8k | TP1 / EP1 × PP4 | 4 | v5 | 0.5 | 32768 | `1` | breakable; cap 8192 | pending | pending | pending | pending | pending | pending |
| E-v5-cap32k | TP2 / EP2 × PP2 | 4 | v5 | 0.5 | 32768 | `1` | breakable; cap 32768 | pending | pending | pending | pending | pending | pending |
| F-v5-cap32k | TP1 / EP1 × PP4 | 4 | v5 | 0.5 | 32768 | `1` | breakable; cap 32768 | pending | pending | pending | pending | pending | pending |

Invalidated v2-chunk8k data (do not compare or use for conclusions):

| Arm | chunk | tok/s/GPU runs → median | TTFT P50 / P99 median | graph capture | actual PP max micro-batch | KV capacity | job / terminal state |
|---|---:|---|---:|---|---:|---:|---|
| A | 8192 | 9,364.80 / 10,323.28 / 10,320.77 → 10,320.77 | 6,269.42 / 6,678.82 ms | NO, DSV4 auto-disable | n/a | 23,122,944 | 796765 / COMPLETED |
| B | 8192 | 8,735.59 / 11,085.38 / 10,046.95 → 10,046.95 | 5,680.95 / 8,137.62 ms | NO, DSV4 auto-disable | n/a | 22,888,960 | 796780 / postprocess-only FAIL after all runs |
| C | 8192 | 21,397.03 / 21,439.67 / 21,513.70 → 21,439.67 | 6,046.45 / 6,304.99 ms | NO, DSV4 auto-disable | n/a (resolved live value 256) | 18,454,528 | 796886 / COMPLETED |
| D | 8192 | 24,535.11 / 24,541.59 / 24,530.77 → 24,535.11 | 5,306.27 / 5,513.28 ms | YES, begin/end on both stages | 128 | 34,293,248 | 796887 / COMPLETED |

## 1. Status lines

- 2026-09-19 14:53 PDT | PREP / first report checkpoint | job n/a | submitted n/a / started 14:46 / waited 0 min, reasonable (no X1 GPU job submitted); live AGA login safety check: user slice 49/300 tasks and 3.95/32.21 GB, batch idle=1361; M5 jobs are a different line and are not counted as X1 jobs | read protocol v2 and M5/docs 13/25/50 evidence; fixed X1 source, target/draft provenance and 10-row switch matrix; confirmed nominal ETA 18:16 PDT (3.5 h from instruction): setup 20 min + five two-job waves at about 18 min=90 min + profile/trace analysis 45 min + report/push 25 min + retry/queue buffer 30 min = 210 min; actual 7 min vs expected 20-min setup, 13 min ahead
- 2026-09-19 14:57 PDT | A · TP4/EP4 | job 796765 | submitted 14:56:40 / started 14:56:55 / waited 0.25 min, reasonable (`Reason=None`, `LastSchedEval=14:56:55`, `Priority=131562`, start-time batch idle=1349) | `qos=short`, one node × 4 GPU, MTP-only, chunk=8192, warmup=64 discarded, formal=128×3; source clone verified at `3a64faa1f22a`; attempted paired B submission was rejected before job creation by the account-wide `QOSMaxSubmitJobPerUserLimit`, so X1 has one active job and will fill the second slot when the shared submit count falls | ETA 18:16 PDT; actual 11 min vs expected 20-min setup, 9 min ahead, but the account-wide submit cap may consume queue buffer
- 2026-09-19 15:00 PDT | B · DEP4 | job 796780 | submitted 14:58:57 / started 14:59:36 / waited 0.65 min, reasonable (`Reason=None`, `LastSchedEval=14:59:36`, `Priority=131562`, pre-start batch idle=1338) | second submit succeeded after one shared-QOS slot opened; one node × 4 GPU, same source/MTP/workload as A; X1 now has exactly two running jobs | ETA 18:16 PDT; actual 14 min vs expected first pair starting by 15:06, 6 min ahead
- 2026-09-19 15:13 PDT | A · TP4/EP4 complete | job 796765 | submitted 14:56:40 / started 14:56:55 / waited 0.25 min, reasonable / ended 15:12:28 (`COMPLETED 0:0`, elapsed 15m33s) | three runs=`9364.80,10323.28,10320.77 tok/s/GPU`, median 10320.77, range 958.48; TTFT P50/P99 medians=6269.42/6678.82 ms; KV=23,122,944; server-info says prefill backend=`disabled` and no prefill capture begin/end exists | ETA 18:16 PDT; actual 27 min vs planned setup+first pair 38 min, 11 min ahead
- 2026-09-19 15:14 PDT | B · DEP4 measurements complete / postprocess FAIL | job 796780 | submitted 14:58:57 / started 14:59:36 / waited 0.65 min, reasonable / ended 15:13:51 (`FAILED 127:0`, elapsed 14m15s) | the discarded warmup and all three formal runs completed before failure: `8735.59,11085.38,10046.95 tok/s/GPU`, median 10046.95, range 2349.79; TTFT P50/P99=5680.95/8137.62 ms; KV=22,888,960. Failure occurred only after run 3 because I replaced the still-open sbatch file while adding the profile parser, so bash read a corrupted next token (`on3`); summary was deterministically reconstructed from the three saved JSONL files and ready server-info. No server/workload failure and no rerun needed | ETA 18:16 PDT; actual 28 min vs planned first-pair completion around 15:24, 10 min ahead
- 2026-09-19 15:15 PDT | C · TP2/EP2 and D · TP1/EP1×PP2 submitted | jobs 796886 / 796887 | both submitted 15:15:07 / starts — (pending) / waited 0.3 min at checkpoint, reasonable (both `Reason=None`; C `LastSchedEval=15:15:23`, D `LastSchedEval=15:15:07`, both `Priority=131562`, batch idle=1327) | AGA short QOS rejects a 2-GPU allocation with `QOSMinGRES`; both jobs therefore reserve the required four GPUs but each inner `srun` exposes and uses exactly two GPUs. The two 2-GPU arms were submitted together; X1 active jobs=2 | ETA 18:16 PDT; actual 29 min vs expected second wave by 15:24, 9 min ahead
- 2026-09-19 15:16 PDT | C / D started | jobs 796886 / 796887 | both submitted 15:15:07 / both started 15:15:59 / both waited 0.87 min, reasonable (`Reason=None`, `LastSchedEval=15:15:59`, `Priority=131562`) | C on `nvl72d097-T17`, D on `nvl72d032-T16`; each inner step uses two GPUs and X1 running count is exactly two | ETA 18:16 PDT; actual 30 min vs expected second-wave start 15:24, 8 min ahead
- 2026-09-19 15:32 PDT | C / D v2-chunk8k complete | jobs 796886 / 796887 | both submitted 15:15:07 / both started 15:15:59 / both waited 0.87 min, reasonable (`Reason=None`, `LastSchedEval=15:15:59`, `Priority=131562`) / D ended 15:30:56 (`COMPLETED`, 14m57s), C ended 15:31:37 (`COMPLETED`, 15m38s) | saved complete three-run evidence; D really captured breakable prefill graphs on PP0/PP1 and resolved PP max micro-batch to 128 | ETA 18:16 PDT; actual 46 min vs second-wave completion planned around 15:42, 10 min ahead
- 2026-09-19 16:20 PDT | protocol v2.1 correction / invalidation checkpoint | jobs 796886 / 796887 terminal; X1 active jobs=0 | both had submitted 15:15:07 / started 15:15:59 / waited 0.87 min, reasonable; no queued X1 job | lead #108 establishes that chunk 8192 underfills this 8K-ISL/C=32 workload and dominates the PP effect. A–D are marked `v2-chunk8k（作废）`; script default and future matrix changed to chunk/max-prefill 32768, with every other control retained. ETA corrected to 19:50 PDT (original 18:16 + 94 min for four mandatory reruns and report churn); actual elapsed 94 min vs original planned 94 min to ETA, on the old schedule but new work adds five two-job waves
- 2026-09-19 16:24 PDT | A / B v2.1 reruns submitted | jobs 797385 / 797386 | both submitted 16:23:30 / starts pending / waited 0.5 min at checkpoint, reasonable (both `Reason=None`, `LastSchedEval=16:23:30`, `Priority=131562`; batch idle=33 conventional plus 2 starred) | one node × 4 GPU each, `qos=short`, explicit `CHUNK=32768`; X1 submitted/running count exactly two | ETA 19:50 PDT; actual 98 min vs revised plan first rerun wave submitted by 16:25, 1 min ahead
- 2026-09-19 16:25 PDT | A / B v2.1 reruns started | jobs 797385 / 797386 | both submitted 16:23:30 / both started 16:24:15 / both waited 0.75 min, reasonable (`Reason=None`, `LastSchedEval=16:24:15`, `Priority=131562`) | A on `nvl72d181-T17`, B on `nvl72d078-T10`; X1 running count exactly two | ETA 19:50 PDT; actual 99 min vs revised first-wave start planned by 16:27, 2 min ahead
- 2026-09-19 16:39 PDT | A / B v2.1 complete | jobs 797385 / 797386 | both submitted 16:23:30 / started 16:24:15 / waited 0.75 min, reasonable / B ended 16:37:49 (`COMPLETED`, 13m34s), A ended 16:38:57 (`COMPLETED`, 14m42s) | A runs=`21392.27,21403.20,21366.87`, median 21392.27; B runs=`41874.50,33243.91,43018.47`, median 41874.50; both logs confirm DSV4 auto-disabled prefill graph | ETA 19:50 PDT; actual 113 min vs revised first-wave completion planned 16:42, 3 min ahead
- 2026-09-19 16:41 PDT | C / D v2.1 reruns submitted | jobs 797479 / 797480 | both submitted 16:40:19 / starts pending / waited 0.7 min at checkpoint, reasonable (both `Reason=None`, `LastSchedEval=16:40:19`, `Priority=131562`) | four-GPU short-QOS allocations with nested two-GPU `srun`; explicit chunk 32768; X1 submitted count exactly two | ETA 19:50 PDT; actual 115 min vs revised second-wave submission planned 16:42, 1 min ahead
- 2026-09-19 16:41 PDT | C / D v2.1 reruns started | jobs 797479 / 797480 | both submitted 16:40:19 / both started 16:40:29 / both waited 0.17 min, reasonable (`Reason=None`, `LastSchedEval=16:40:29`, `Priority=131562`) | C on `nvl72d078-T10`, D on `nvl72d094-T06`; inner steps expose exactly two GPUs and X1 running count exactly two | ETA 19:50 PDT; actual 115 min vs revised second-wave start planned 16:43, 2 min ahead
- 2026-09-19 16:56 PDT | C complete / D default graph-cap infeasible | jobs 797479 / 797480 | both submitted 16:40:19 / started 16:40:29 / waited 0.17 min, reasonable; C ended 16:55:41 (`COMPLETED`, 15m12s), D ended 16:53:09 (`FAILED 1:0`, 12m40s) | C median 36,762.18 tok/s/GPU. D completed breakable capture on both stages but PP1 OOMed during discarded warmup: only 686 MiB free when a 2.00 GiB allocation was requested; no formal result exists. Captured tiers stopped at 8192 even though 32K workload batches dominate, so the graph pool consumed memory without serving those main batches | ETA 19:50 PDT; actual 130 min vs revised second-wave completion planned 16:58, 2 min ahead; recovery is overlapped with the E profile wave
- 2026-09-19 16:58 PDT | D feasibility retry + E-ov1 profile started | jobs 797549 / 797550 | both submitted 16:57:09 / both started 16:57:16 / both waited 0.12 min, reasonable (`Reason=Prolog` at 3-second checkpoint, `LastSchedEval=16:57:16`, `Priority=131562`) | D remains `breakable` but bounds the unused capture tier to 4096; E uses default capture cap, comm overlap=1 and ≥60 s profile. D on `nvl72d078-T10`, E on `nvl72d093-T05`; X1 running count exactly two | ETA 19:50 PDT; actual 132 min vs revised recovery/profile-wave start planned 17:00, 2 min ahead
- 2026-09-19 17:10 PDT | D recovery complete / E default-cap capture failure | jobs 797549 / 797550 | both submitted 16:57:09 / started 16:57:16 / waited 0.12 min, reasonable; E ended 17:07:46 (`FAILED 1:0`, 10m30s), D ended 17:09:54 (`COMPLETED`, 12m38s) | D cap=4096 captured PP0/PP1 (12.16/12.67 GiB) and produced stable runs=`29075.02,29000.92,29103.91`, median 29075.02. E default cap failed *during capture*, before ready/benchmark: PP1 TP ranks fell to about 10 MiB and raised CUDA OOM / `markCaptureEnd called with no captures in progress`. This confirms default breakable cap is infeasible for both D runtime and E capture under mem-fraction 0.9 | ETA 19:50 PDT; actual 144 min vs recovery-wave completion planned 17:15, 5 min ahead
- 2026-09-19 17:11 PDT | E-ov1 / F-ov1 cap=4096 profiles submitted | jobs 797652 / 797653 | both submitted 17:10:53 / starts pending / waited 0.3 min at checkpoint, reasonable (both `Reason=None`, `LastSchedEval=17:10:53`, `Priority=131562`) | both use comm overlap=1, explicit breakable, uniform capture cap 4096, and ≥60 s profiling; X1 submitted count exactly two | ETA 19:50 PDT; actual 145 min vs revised profile-pair submission planned 17:16, 5 min ahead
- 2026-09-19 17:12 PDT | E-ov1 / F-ov1 cap=4096 profiles started | jobs 797652 / 797653 | both submitted 17:10:53 / both started 17:11:10 / both waited 0.28 min, reasonable (`Reason=None`, `LastSchedEval=17:11:10`, `Priority=131562`) | E on `nvl72d078-T10`, F on `nvl72d193-T02`; X1 running count exactly two | ETA 19:50 PDT; actual 146 min vs revised profile-pair start planned 17:17, 5 min ahead
- 2026-09-19 17:15 PDT | lead #118 graph-coverage extension accepted | active jobs 797652 / 797653 | both submitted 17:10:53 / started 17:11:10 / waited 0.28 min, reasonable (`Reason=None`, `LastSchedEval=17:11:10`, `Priority=131562`) | retain current main cap=4096 pair as the X1 baseline. Add two controlled E/F waves on fork `pp-verify/consolidated-v5@dbe4c93ac3c`: cap 8192 versus cap 32768 at the same lowered memory fraction, and count every logged prefill step's `cuda graph: True/False`. No current job is cancelled and X1 running count remains two | ETA corrected to 20:30 PDT (prior 19:50 +40 min for source staging, two waves and graph-hit analysis); actual 149 min vs revised profile-pair plan 151 min, 2 min ahead before extension
- 2026-09-19 17:24 PDT | E-ov1 main cap=4096 warmup OOM / F-ov1 profiling | jobs 797652 / 797653 | both submitted 17:10:53 / started 17:11:10 / waited 0.28 min, reasonable; E ended 17:22:15 (`FAILED 1:0`, 11m05s), F remains running on `nvl72d193-T02` | E completed cap-4096 capture on both PP stages but failed in the discarded 32K warmup: PP1 ranks requested 2.00 GiB with only about 0.70/1.19 GiB free; no formal E result exists. F completed three formal runs and entered its 60 s trace; its 32K hot-step lines are `cuda graph: False` while four 6-token startup probes are the only `True` lines seen so far. To obtain a controlled and feasible E/F switch matrix, all authoritative E/F reruns will use the same `mem-fraction-static=0.5`; the 0.9 attempts remain explicit feasibility evidence | ETA 20:30 PDT; actual 158 min vs extension plan expecting the first profile by 17:30, 6 min ahead, with one memory-control rerun wave added inside the existing retry buffer
- 2026-09-19 17:31 PDT | F-ov1 mem-fraction 0.9 profile complete | job 797653 | submitted 17:10:53 / started 17:11:10 / waited 0.28 min, reasonable / ended 17:30:47 (`COMPLETED`, 19m37s) | runs=`39385.86,38350.17,40080.35`, median 39385.86; TTFT P50/P99=1533.69/2066.00 ms; formal hot-step graph ratio=0/416. The 64.83 s trace was saved for every PP stage; the original parser globbed before later-stage serialization completed, so this extra mem-0.9 trace is retained as raw evidence and the corrected wait is applied to the controlled pair | ETA 20:30 PDT; actual 165 min vs extended profile-wave completion planned by 17:38, 7 min ahead
- 2026-09-19 17:33 PDT | controlled E-ov1 / F-ov1 mem-fraction 0.5 profiles started | jobs 797742 / 797745 | E submitted 17:31:35 / started 17:31:39 / waited 0.07 min, reasonable; F submitted 17:31:47 / started 17:32:39 / waited 0.87 min, reasonable (`Reason` briefly `Nodes required ... DOWN, DRAINED or reserved`, `LastSchedEval=17:32:05`, `Priority=131562`, batch idle=77) | both main, cap=4096, comm overlap=1, explicit breakable and >=60 s trace; launcher now waits for all rank trace files before parsing; X1 running count exactly two | ETA 20:30 PDT; actual 167 min vs extension schedule expecting controlled-pair start around 17:32, 1 min behind, within retry buffer
- 2026-09-19 17:46 PDT | controlled E-ov1 / F-ov1 formal measurements complete, profiling | jobs 797742 / 797745 | E submitted 17:31:35 / started 17:31:39 / waited 0.07 min; F submitted 17:31:47 / started 17:32:39 / waited 0.87 min; both waits reasonable and both remain running only to finish the required traces | E runs=`18036.92,18046.81,18057.07`, median 18046.81, only 20.15 range; F runs=`38334.15,37030.33,39900.47`, median 38334.15. Exact LF-based log windows give E graph True/False=`0/204` and F=`0/416`; capture did occur, but no 32K formal step replayed it. The first counter implementation used Python universal newlines while its offsets came from `wc -l`; tqdm's bare CR capture progress shifted the offsets, so the report uses the corrected byte/LF recount and the launcher is corrected for subsequent waves | ETA 20:30 PDT; actual 180 min vs extension schedule expecting the profile data around 17:52, 6 min ahead
- 2026-09-19 18:00 PDT | lead #120 task-status publication checkpoint | jobs 797742 / 797745 | E submitted 17:31:35 / started 17:31:39 / waited 0.07 min; F submitted 17:31:47 / started 17:32:39 / waited 0.87 min; both waits reasonable and both remain `RUNNING` only for four-rank gzip trace parsing | the report had in fact been pushed incrementally through `4e2e8ea6ef4` on the X1 line branch `pp-verify/X1-v4flash-anomaly`; the requested fork task-status ref did not yet exist. Per #120 this checkpoint creates/pushes `pp-mtp-verification-20260917` immediately, without waiting for the parsers and without opening a PR. X1 active jobs remain exactly two | ETA 20:30 PDT; actual 194 min vs extension schedule expecting graph-cap work to begin around 18:00, on schedule except that trace parsing still occupies this pair

## 2. Exact commands

The immutable inputs are:

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
- Protocol v2.1 workload: `--chunked-prefill-size 32768
  --max-prefill-tokens 32768`, random ISL 8192, OSL 1, concurrency 32, one
  discarded 64-prompt warmup, and three formal 128-prompt runs.
- Frozen AGA launcher for A/B/C and the default-cap D attempt:
  `$U/pp-perf-20260919/X1-v4flash-anomaly/run_x1_prefill.sbatch`, SHA-256
  `e9c664d8e5e8765f6a3cdaeb51f172711d3fffddb7fdf795934d52520d89448e`.
- Post-OOM launcher used from D recovery onward adds only an optional
  `--cuda-graph-max-bs-prefill` control plus readback fields; SHA-256
  `df9f45ef9d4e69063e2ece99642d4069db9d9c3132fb77d982fb22c02d5b2018`.
- Controlled-matrix launcher adds source/memory controls, per-run graph-hit
  counts, and waits for all PP trace writers; SHA-256
  `eb9e2c8a950204967d615993ad8704aff000148eba6686cdb536ef8614880dd1`.
- Streaming trace parser (rank/stage aggregate plus every scheduler-step
  timeline), SHA-256
  `7f3e9d1bbed26abb7c93127ee3bf5fd733539f2ea6c2d3744d7823c77e0d2fc7`.

Exact allocation/submit matrix (the short QOS has a four-GPU minimum; C/D
reserve four but the nested `srun --gpus-per-node=2` exposes exactly two):

```bash
sbatch --parsable --export=ALL,ARM=A,GPUS=4,COMM=unset,BCG=1,PROFILE=0,CACHE_KEY=X1-v21-A,CHUNK=32768 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=B,GPUS=4,COMM=unset,BCG=1,PROFILE=0,CACHE_KEY=X1-v21-B,CHUNK=32768 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=C,GPUS=2,COMM=unset,BCG=1,PROFILE=0,CACHE_KEY=X1-v21-C,CHUNK=32768 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=D,GPUS=2,COMM=unset,BCG=1,PROFILE=0,CACHE_KEY=X1-v21-D,CHUNK=32768 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=E,GPUS=4,COMM=1,BCG=1,PROFILE=1,CACHE_KEY=X1-v21-E,CHUNK=32768,CG_MAX=4096 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=F,GPUS=4,COMM=1,BCG=1,PROFILE=1,CACHE_KEY=X1-v21-F,CHUNK=32768,CG_MAX=4096 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=E,GPUS=4,COMM=unset,BCG=1,PROFILE=0,CACHE_KEY=X1-v21-E,CHUNK=32768,CG_MAX=4096 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=F,GPUS=4,COMM=unset,BCG=1,PROFILE=0,CACHE_KEY=X1-v21-F,CHUNK=32768,CG_MAX=4096 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=E,GPUS=4,COMM=unset,BCG=0,PROFILE=0,CACHE_KEY=X1-v21-E,CHUNK=32768 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=F,GPUS=4,COMM=unset,BCG=0,PROFILE=0,CACHE_KEY=X1-v21-F,CHUNK=32768 run_x1_prefill.sbatch

# Recovery after the default-cap D warmup OOM; backend remains breakable.
sbatch --parsable --export=ALL,ARM=D,GPUS=2,COMM=unset,BCG=1,PROFILE=0,CACHE_KEY=X1-v21-D,CHUNK=32768,CG_MAX=4096 run_x1_prefill.sbatch

# The first E attempt above showed that mem-fraction 0.9 leaves insufficient
# runtime headroom after capture.  The authoritative E/F switch matrix uses
# this identical feasibility control on both arms and every switch setting:
# add SOURCE=main,MEM_FRACTION=0.5 to each of the six E/F commands.

# Lead #118 source/capture-coverage pair (same memory and overlap controls):
sbatch --parsable --export=ALL,ARM=E,GPUS=4,COMM=1,BCG=1,PROFILE=0,CACHE_KEY=X1-v5-E,CHUNK=32768,CG_MAX=8192,SOURCE=v5,MEM_FRACTION=0.5 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=F,GPUS=4,COMM=1,BCG=1,PROFILE=0,CACHE_KEY=X1-v5-F,CHUNK=32768,CG_MAX=8192,SOURCE=v5,MEM_FRACTION=0.5 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=E,GPUS=4,COMM=1,BCG=1,PROFILE=0,CACHE_KEY=X1-v5-E,CHUNK=32768,CG_MAX=32768,SOURCE=v5,MEM_FRACTION=0.5 run_x1_prefill.sbatch
sbatch --parsable --export=ALL,ARM=F,GPUS=4,COMM=1,BCG=1,PROFILE=0,CACHE_KEY=X1-v5-F,CHUNK=32768,CG_MAX=32768,SOURCE=v5,MEM_FRACTION=0.5 run_x1_prefill.sbatch
```

The launcher installs the pinned dependency into a per-job directory, then runs
this common server command; the final line is exactly one of the topology rows
shown below.

```bash
python3 -m pip install -q --target "$PIPDEPS" --no-deps sglang-kernel==0.4.7
python3 -m sglang.launch_server \
  --model-path /model --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.9 --max-running-requests 256 \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --cuda-graph-max-bs-decode 256 --page-size 256 --swa-full-tokens-ratio 0.1 \
  --moe-a2a-backend megamoe --enable-w4a4-mxfp4-megamoe \
  --speculative-draft-model-path /draft --speculative-algorithm EAGLE \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 "${TOPOLOGY[@]}"

# A: --tp-size 4 --ep-size 4
# B: --enable-dp-attention --enable-dp-lm-head --dp-size 4 --tp-size 4
#    --ep-size 4 --load-balance-method total_tokens
# C: --tp-size 2 --ep-size 2
# D: --tp-size 1 --ep-size 1 --pp-size 2 --cuda-graph-backend-prefill breakable
# E: --tp-size 2 --ep-size 2 --pp-size 2 [--cuda-graph-backend-prefill breakable]
# F: --tp-size 1 --ep-size 1 --pp-size 4 [--cuda-graph-backend-prefill breakable]
```

For D/E/F the launcher exports `SGLANG_ENABLE_PP_SPEC=1`; only `*-ov1`
additionally exports `SGLANG_PP_COMM_OVERLAP=1`. Each row uses exactly this
discarded warmup and three formal invocations (labels `1`, `2`, `3`):

```bash
python3 -m sglang.bench_serving --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model deepseek-ai/DeepSeek-V4-Flash --tokenizer /model \
  --dataset-name random --random-input-len 8192 --random-output-len 1 \
  --random-range-ratio 1 --max-concurrency 32 --request-rate inf --seed 42 \
  --disable-tqdm --num-prompts 64 --flush-cache --output-file bench-warmup.jsonl
# repeat the same command three times with --num-prompts 128 and distinct files
```

## 3. Decomposition and conclusion

The authoritative chunk-32768 comparisons are:

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
6. Lead #118 hypothesis: default prefill capture tops out below a 32K PP batch,
   so a successfully captured graph can still have a zero/near-zero hot-path
   hit rate. On consolidated-v5 (`dbe4c93ac3c`, which contains the non-first
   stage replay fix), compare E/F at capture caps 8192 and 32768 with otherwise
   identical source, memory fraction, comm-overlap and workload. Report both
   throughput and the server-log ratio of prefill steps marked
   `cuda graph: True` versus `False`.

The expert-weight comparison is fixed before looking at performance. The target
config has 43 layers, 256 routed experts/layer, hidden size 4096, expert
intermediate size 2048, and `expert_dtype=fp4`. Upstream `get_pp_indices`
places the remainder on the final stages, hence PP2 owns `[21,22]` layers and
PP4 owns `[10,11,11,11]`. With even EP sharding, the routed expert-layer copies
per GPU are therefore:

| Arm | routed expert-layer copies/GPU | raw routed FP4 payload/GPU (scales/metadata excluded) |
|---|---:|---:|
| A/B (PP1, EP4) | 43 × 64 = 2,752 | 32.25 GiB |
| C (PP1, EP2) | 43 × 128 = 5,504 | 64.50 GiB |
| D (PP2, EP1) | 21/22 × 256 = 5,376 / 5,632 | 63.00 / 66.00 GiB |
| E (PP2, EP2) | 21/22 × 128 = 2,688 / 2,816 | 31.50 / 33.00 GiB |
| F (PP4, EP1) | 10/11 × 256 = 2,560 / 2,816 | 30.00 / 33.00 GiB |

The payload uses three expert matrices per copy:
`3 × 4096 × 2048 × 0.5 byte = 12 MiB`. Thus E and F are deliberately almost
matched in routed-expert payload; C and D are likewise almost matched. Any
large E/F difference cannot be explained by resident routed weights alone and
must be checked against PP bubbles/communication and the actual MoE kernels.

Interim topology decomposition from the completed authoritative rows:

- C/A = **+71.85% per GPU** (although the two-GPU C aggregate is 14.08% below
  the four-GPU A aggregate): reducing TP/EP degree from 4 to 2 is a large
  per-GPU win.
- D/C = **-20.91% per GPU**: introducing a PP boundary and changing TP2/EP2 to
  TP1/EP1×PP2 on the same two GPUs costs about one fifth here, not one half.
- E/D = **-37.93% per GPU**: adding a second TP/EP lane to each of the two PP
  stages is the first large negative interaction.
- F/E = **+112.42% per GPU**: at fixed four GPUs, replacing two TP2/EP2 stages
  with four TP1/EP1 stages more than doubles throughput. Both captured the same
  4096 ceiling and both had zero hot-path graph hits, so this factor-of-two gap
  cannot be attributed to graph replay. The trace and MoE-kernel split below
  will identify whether it is chiefly EP2 collective/kernel cost or exposed PP
  scheduling/receive time.

No SGLang product code is changed by this task.

## 4. NEED_LEAD

None at this checkpoint.
