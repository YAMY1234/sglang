# Phase-one common cache commit boundary

`SGLANG_HICACHE_PP_COMMON_COMMIT=1` opts into this experiment; its default is false. The measured v3.2 baseline remains `188576241030e65f1441e79d9281c2885dfe1def`. This branch adds a logical commit layer; CPU checks do not establish GPU performance or production eligibility.

The scope is backup ACK completion, queued release ACK page ownership, and explicit belief insert/delete. Prefetch result insertion/revocation, D2H/H2D completion, and lookup LRU touches are unchanged phase-two work. An LRU eviction during a committed insertion can still remove different keys across stages: set digests and eviction/touch counters expose that case. The current opt-in path supports the Python unified cache, CP1 and an attached storage controller. Linker, buffer-only, runtime storage clear/detach and reset with in-flight work fail closed.

## Protocol and ownership

Producer identities use content/request, pool, logical range and per-key generation, never process-local operation/node IDs or physical page addresses. PP0 assigns `(epoch, seq)` manifest positions. An ACK prepares a callback and retains its resource ownership; only a confirmed contiguous prefix applies callbacks. Queue counts and v3.2 debt still govern physical drains. Release pages retain that physical granularity but one logical range is prepared only after all its fragments arrive. Reordered identities cannot satisfy a missing operation; no NOOP is inferred from queue length.

Every cache-event round broadcasts a bounded manifest/commit frame along the original PP direction, with distinct LENGTH/FRAME tags and v3.2 headers/sequences. PP0's TP root supplies the same manifest to its TP replicas. Each stage reduces its contiguous readiness over the existing attention groups. Previous-round READY reports return to PP0 on a separate startup-created Gloo group. Only background threads block on that group; scheduler publication and mailbox polling do not wait for reverse traffic. Coalescing bounds buffers without dropping wire sequence validation. No reverse work joins `_drain_async_work`.

Logical backup completion jointly removes `ongoing_backup`, unlocks its host references, clears write-behind ownership and adds the acknowledged content's beliefs. Mamba no longer publishes belief at enqueue; a private pending-write set suppresses duplicate writes without affecting prefix matches. Explicit invalidation is staged. Release fragments keep their host slots unavailable until commit. Belief fingerprints compare the same committed prefix, not arbitrary queue lengths; TP and PP checks are reported separately from local LRU effects.

Admission closes through the common frame when pending resource limits approach their bounds. A permanent missing identity prevents further commits: eight unchanged rounds warn with operation identities; 120 seconds without completion fails explicitly. Downstream-only backup/release ACKs behind PP0's drain budget are checked too, so unrelated progress cannot conceal an orphan. Resource bounds are 8192 prepared operations and a conservative 512 GiB sum of pinned operation spans (duplicate references may overcount). At process exit, IO is stopped without relabeling local teardown as a common commit.

## Validation and remaining gates

CPU coverage includes three-stage normal/late/permanently missing operations, equal-count different-key ACKs, reordering, duplicate generations, reset, bounded ownership, fragmented release, default-off consumers and belief mutation. Native Gloo tests exercise the full bridge plus v3.2 chain at PP3/TP1 and PP3/TP2 with a deliberately late TP ACK. The existing 14 v3.2 regression cases remain required.

A broader legacy cache test import is blocked in the local shared CPU environment by Transformers already registering `qwen3_asr`; no dependency or unrelated engine workaround is included. The focused suite loads the checked-in state machine/bridge and actual producer/consumer methods, and uses real CPU tensors/Gloo rather than changing model code to make imports pass.

Next gates: PP2/TP2 + DEP4, C200, real write-back (`wb_issued_ops > 0`), normal 1 hour; then the frozen 32-GPU historical 3600-second recipe, normal 3 hours. Target Y is 156K ±2%, commit stalls zero, and belief mismatch zero or explicitly diagnosed. Historical 3/lane and injected AL4.8 remain internal-only. Prefetch/LRU expansion requires the agreed phase-two trigger; this patch does not claim full cache consensus before those measurements.

## First writeback smoke: repeated no-effect belief deletes

The first PP2 run entered real warmup but the new 120-second guard rejected three downstream-only belief-delete effects after common position147. Both PP1 ranks had empty belief state and zero pinned bytes; PP0 had no corresponding operation. Two CPU regressions reproduced this as pending3/1. The operation factory had incorrectly numbered repeated local miss feedback as distinct logical mutations.

The follow-up preserves the common boundary for real backup/release ACKs. It drops a deletion only when membership is absent and there is no staged explicit add; repeated identical belief effects coalesce until commit only if no opposite effect intervenes. Add-delete-add-delete ordering remains tested. A later physical backup ACK is a later positive observation. LRU membership is read without touching LRU order. All50 focused CPU tests pass; native PP3TP1/PP3TP2 Gloo is included. PP2/writeback and full32GPU validation remain required. Diagnostic snapshot histories are now bounded on every PP rank.
