"""GLM-5.3-Flash DUET serving adapter (spec.model "glm5-next"; twinstar docs/79, phase F of docs/72).

`--duet-release <dir>` (sglang.srt.duet.adapters) registers this package over `Glm5NextForConditionalGeneration`: the class in
`model.py` is the stock GLM-5.3 model (TP 4, DSA + KDA + mHC + FP8 MoE) plus the DUET prompt path and the KDA state cut.
`geometry.py` (torch-free) derives the release's tensor contract from the spec and the base config; `emitter_math.py` is the
emitters' trained arithmetic; `state_policy.py` the KDA state cut on the dense pool with its schedule on the host.

Prompt (every extend batch, chunk-aware): the stock layers 0..k-1 run on every prompt token but each request's last one; the
4 x 4096 mHC residual entering layer k is coded (r NVFP4 coordinates + m exact bf16 coordinates, the token's input streams as side
information) and decoded; the emitters write every deeper layer's prompt memory from it -- DSA layers: the MLA latent c into the
MLA pool and the indexer key / gate score through the layer's own pooled-FP8 write path (closed pools + the open-pool tail);
KDA layers: conv windows + recurrent state into the Mamba pool (fla's chunk_kda, the training kernel).  At a request's last chunk
every KDA state is cut to its stored form (explicit sink + rank r, cold) and the last prompt token runs as decode step 1 through
all 45 layers.  Decode: the cut every W steps (warm-started randomized subspace iteration, never an fp64 eigendecomposition of the
full state), after the step, on replayed and eager steps.  Attention caches are not compressed.
"""

__all__ = ["geometry", "emitter_math", "state_policy", "model"]
