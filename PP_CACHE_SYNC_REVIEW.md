# PP HiCache phase-one candidate: review guide

The phase-one candidate passed a complete historical one-hour run. The measured source is frozen at `c8729902be29336c10f504eec44f0dadbac4150d` on `fork/q35-highx-pp-common-commit`; it remains unchanged. This independent review branch rebuilds the exact milestone trees as seven logical commits, then updates documentation. Its runtime and tests are byte-identical to the measured source. Base: `2974fe57e3cb7c5b1c0126293ba1101fd74389ae`. No PR is open.

## Read in this order

| Step | Review commit | Problem / repair | Original milestone |
|---|---|---|---|
| 1 | `add7223cea39` | fix(cache): keep PP ACK drains bounded and validate chain message order | `0f3be807e6f9` |
| 2 | `e1542414b284` | feat(cache): model operation-identified common PP commit prefixes | `421026cbe888` |
| 3 | `3f32bde42f84` | feat(cache): carry PP readiness over an independent asynchronous channel | `ddaf0024c1b0` |
| 4 | `b90088122767` | feat(cache): gate phase-one ACK and belief effects on common PP commit | `712d4414fa74` |
| 5 | `3fc78d4d58ec` | fix(cache): batch bounded belief proposals without relaxing age protection | `d4aabd209f2b` |
| 6 | `6def88ea3dc8` | fix(cache): intersect restorable hybrid prefetch boundaries across ranks | `75e40cf7c7d3` |
| 7 | `173e8e11fbde` | fix(cache): pipeline belief proposals within bounded credits and payload queues | `c8729902be29` |

Each commit message states the unresolved scope at that step. Intermediate commits are review milestones, not independently qualified deployments. The final seven-commit tree is exactly the measured tree; the following documentation commit changes only README and these review/design notes.

## Protocol and scope

PP0 count broadcasts can exceed downstream ACK availability. The v3.2 layer drains available local entries with debt and checks an ordered header before receiving a site-specific payload; it avoids the observed empty-queue drain while preserving forward pipeline cadence. A whole-PP collective inside the prefill scheduler can deadlock and is not used here. A peer that never reaches another receive still requires the watchdog.

The default-off phase-one guard adds canonical operation identities and a contiguous shared commit prefix. Physical ACK receipt stays local; backup/release and belief side effects wait for all ranks' readiness from the previous round. Independent asynchronous control traffic carries bounded READY payloads; missing identities, bad epochs and excessive age fail with evidence rather than silently dropping work. Downstream belief mutations are proposals, including misses under storage-capacity pressure. Same-key add/delete order is retained; no cross-rank semantic deduplication or interval merging is claimed.

Inspect `pp_commit.py` (identities/frontier), `pp_commit_transport.py` (independent control transport), `pp_commit_bridge.py` and `pp_belief_proposals.py` (producer integration/credits), `unified_radix_cache.py` (side effects), and `prefetch_boundaries.py` (legal hybrid endpoint intersection). Proposal capacity8192, in-flight1536/origin, adaptive batches16/32/64 and explicit frame bounds remain; 120-second protection is unchanged. Default flag `SGLANG_HICACHE_PP_COMMON_COMMIT=False`; enable only for the scoped PP>1 prefill configuration. Some preceding debt/boundary repairs are not flag-gated.

## Validation and limits

- CPU: 85 tests plus five subtests; conditional sustained-load model: 72 positive and three negative scenarios. Real Gloo CLOSED-frame retirement is regression-tested.
- Historical internal run: 5P PP4→3D DEP4, C1500, 32 GPUs, 3600 seconds, original3/lane warmup, injected AL4.8. X71.7495, historical AIPerf Y154568.41/GPU, ITL P90 13.9374ms, hit93.1903%. No real-MTP EVAL and no external golden/10-lane qualification.
- Against the v3.2 baseline: X+1.61%, Y−1.22%, TTFT P50+12.69%; different allocations, so this is an observed difference, not isolated causal overhead.
- 20/20 ranks had all five common kinds, R8 and actual L3 reads during the timed window. Zero stall, sequence errors, belief mismatch, residual unassigned, request errors or mid-window cancellation. The 161 request/2 session cancellations all occurred after the sending window in grace; they remain disclosed.
- Proposal peak1162, overflow1098, peak age12.397s; full arrivals and completions each1,000,580, all drained. Physical debt peak218/18 cycles, final0. Measurements validate this load, not a universal128/s/origin service guarantee or arbitrary fault tolerance.
- Phase two remains unimplemented: prefetch result insertion, L2 write/load completion effects, LRU touch and rehydrate attachment. SWA has CPU-only coverage; cache linker is rejected. Zero sampled mismatch under this configuration is not complete tree-state equivalence proof.

[Result and CPU evidence](https://github.com/YAMY1234/task-status/tree/q35perf/b-highx/qwen35-gb300-agentx-vs-trtllm/subtasks/b-highx/design160) · [frozen recipe and runtime manifest](https://github.com/YAMY1234/task-status/tree/q35perf/b-highx/qwen35-gb300-agentx-vs-trtllm/subtasks/b-highx/repro101/common160-full) · [phase-two design boundary](PP_CACHE_COMMIT_BOUNDARY_DESIGN.md).
