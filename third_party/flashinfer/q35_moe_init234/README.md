# Pinned FlashInfer SM100-family initialization experiment

Default off: `SGLANG_FLASHINFER_MOE_COMPACT_INIT=1` on this SGLang fork.
Apply `apply.py PRIVATE_FLASHINFER_PACKAGE --receipt RECEIPT` during private
container setup. The installed files must match source-sha256.json exactly;
shared installations/caches must not be patched. A/B use the same overlay,
source, tactics and observer; only this switch differs.

The two int32 tile initializations become one Triton fill, retaining all zero
padding. SM100 GEMM2 speculatively reads an expert entry before its live-tile
branch, so we do not leave arbitrary tail data. Only the actual MNNVL dispatch
format opts into sparse output. Standard/reduce-scatter fallback remains dense.
Sparse zero runs on the existing aux stream between the original events, before
atomic finalize. Rows with at least one local expert are cleared, even if their
routing weight is zero. All other rows are unspecified and never consumed by
MNNVL combine. No arithmetic, precision or routing choice changes.

Qualification: `python test_compact_init.py --output gate.json` on one SM100 GPU.
Includes empty input, odd/tail tiles, sentinel512, local offsets0/480, buffer
poisoning and graph reuse. Metadata and initialization write sets must be exact.
Full actual-library GEMMs use same-shape synthetic weights and fixed tactics;
atomic BF16 output must lie within three stock repeats' elementwise envelope
plus one BF16 ULP. NaN propagation into live output is forbidden. This does not
replace real-model GSM8K. No GPU gate has been run merely by applying the patch.

Source proof and prior design are in task-status B-fork design234/moe-coverage.md.
Expected net saving0–0.10ms/round, potentially negative: masks cost work and the
old dense memset overlaps GEMM1. Static B8/B16 ABAB precedes AgentX validation.

SM100 family includes SM103/GB300: use major==10, matching pinned
FlashInfer utils.is_sm100a_supported. Rubin/major11 remains on stock.

R3 qualification replaces the invalid three-stock envelope gate: R2 stock's
independent holdout itself exceeded that envelope at346/4096 coordinates.
Isolate each weighted BF16 route contribution using the same stock graph;
require two repeats bitwise, then check stock/candidate sums against the
BF16 gamma_(k-1)*sumabs bound, plus reference conversion/subnormal error.
Zero/single-route outputs remain bitwise. Keep old envelope counts diagnostic.
Production code and promotion gates do not change.
