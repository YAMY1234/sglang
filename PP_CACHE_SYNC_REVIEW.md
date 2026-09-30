# PP cache queue synchronization: review notes

Status: experimental, reviewable for protocol progress; **not a production cache-consistency fix**. The measured engine is frozen at `188576241030e65f1441e79d9281c2885dfe1def`. This review commit changes documentation only. Historical base: `2974fe57e3cb7c5b1c0126293ba1101fd74389ae`.

## Problem and repair

PP0 reduces completion counts within its attention groups and forwards the result down the pipeline. Other PP stages can have fewer local ACKs, so draining the forwarded count with `Queue.get()` can block indefinitely. A captured downstream backup-ACK wait and persistent per-stage drain debt support this failure mode; the precise first divergent enqueue key is still unknown.

The experimental repair keeps forward pipeline ordering. Each stage receives the original count, drains at most its locally available count, and records the shortfall as local debt for later rounds. This avoids the observed empty-queue drain wait; it does not establish identical ACK identities or tree state.

Each logical synchronization now sends a fixed int64 header containing sequence, site, element count, and original element size, followed by a sequence-prefixed int64 payload on a site-specific tag. A skipped/reordered next call raises an explicit mismatch before receiving the wrong payload. No whole-PP collective is added inside the prefill scheduler. A peer that never enters another receive still needs timeout/stack diagnostics.

## Read order

1. `python/sglang/srt/mem_cache/unified_radix_cache.py`: `_pp_drain_counts`, `_pp_sync`, and the seven tagged call sites.
2. `python/sglang/srt/distributed/communication_tags.py`: header and site tags.
3. `test/registered/unit/mem_cache/test_pp_drain_debt.py`: missing ACK, debt recovery/warnings, chain order, and mismatched-site/sequence tests.
4. [Required common commit boundary](PP_CACHE_COMMIT_BOUNDARY_DESIGN.md): follow-up design, not implemented.

The last protocol change is 56 engine lines relative to its predecessor; the full experimental series relative to the historical base is larger and must be reviewed as a whole. Do not mistake the last-commit size for the entire patch.

## Validation and limits

- 14 CPU cases and a real three-rank Gloo test passed, including a deliberately skipped synchronization.
- Two-node PP2 smoke exercised writeback on all four ranks (324 issued operations / 1514 backup drains each), without an active long stall or sequence mismatch.
- The 32-GPU reproduction completed its 3600-second window: historical AIPerf Y156479 tok/s/GPU, X70.61, hit93.204%; 0 errors or mid-window cancellations. Original 3/lane warmup and injected AL4.8 are intentionally historical/internal. 147 requests / 2 sessions were cancelled only at end grace; no EVAL or external-curve qualification.
- No active >120-second stall or sequence mismatch was observed. Two post-window snapshots were in `_drain_async_work`; a single snapshot does not prove either a permanent stall or harmlessness.
- Active ACK debt reached148, longest observed age1104 cycles, and a continuous lower bound301 seconds. Nine ranks retained nonzero debt at shutdown. Queue progress is **not** eventual cache agreement.
- Original rank-local prefetch/termination conditions still exist. The guard detects divergence at the next executed synchronization; it does not prove all control flow identical. Local belief cardinality is a bounded LRU gauge, not a key-set digest.
- Digest validation retains its original attention/TP meaning. Correct common commit ordering, enqueue identity checks, bounded backpressure, reset epochs, and correctness qualification are unfinished.

Evidence and exact recipe/source qualifications: [highx final assessment](https://github.com/YAMY1234/task-status/blob/462b15ffaf38bcd159ef9f60eda950ac9b9e4595/qwen35-gb300-agentx-vs-trtllm/subtasks/b-highx/evidence/924239/final-assessment142.json). No PR is opened; the engine SHA and benchmark recipe remain unchanged.
