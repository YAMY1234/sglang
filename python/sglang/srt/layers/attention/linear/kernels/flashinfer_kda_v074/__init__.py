"""FlashInfer v0.7.4's flashinfer/kda_kernels/kda_chunked_bt16.py (the CuTe DSL BT=16
KDA prefill kernels), vendored verbatim; license header kept (BSD-3-Clause, NVIDIA).

github.com/flashinfer-ai/flashinfer v0.7.4 (77039051b0be921788dc2784ec41423b558878ec),
git blob 65837616b001b3fdd5856bc936526930f4ae100e: 0.7.2's file plus flashinfer#6218
(Q/K L2 norm as rsqrt(sum + 1e-6)), #8684 (decomp-chain update_ready deadlock) and
#8599 (non-finite V padding). kda_flashinfer_prefill loads it as
flashinfer.kda_kernels.kda_chunked_bt16 (SGLANG_K31_KDA_PREFILL_FI_SM107=1).
"""

raise ImportError(
    "vendored FlashInfer source, loaded as flashinfer.kda_kernels.kda_chunked_bt16 by "
    "sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill with "
    "SGLANG_K31_KDA_PREFILL_FI_SM107=1"
)
