# X1 · DeepSeek-V4-Flash prefill-only PP anomaly decomposition

## 0. Arm table

All rows use upstream `main@3a64faa1f22a`, the original `DeepSeek-V4-Flash-MTP`
draft shard, no DSpark, chunk/max-prefill 8192, one discarded 64-prompt warmup,
then three 128-prompt measurements. `tok/s/GPU`, TTFT and KV capacity are pending.

| Arm | Topology | GPUs | `SGLANG_PP_COMM_OVERLAP` | breakable prefill graph | tok/s/GPU (3-run median; range) | TTFT P50 / P99 | capture line | actual PP max micro-batch | KV capacity | job |
|---|---|---:|---|---|---:|---:|---|---:|---:|---:|
| A | TP4 / EP4 | 4 | unset | automatic (non-PP) | pending | pending | pending | n/a | pending | pending |
| B | DEP4 | 4 | unset | automatic (non-PP) | pending | pending | pending | n/a | pending | pending |
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
