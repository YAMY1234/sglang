"""FlashInfer prims_ts files at 784bfaf5 that differ from the image's 26bb6dc9 tree.

Verbatim copies (license headers kept; Apache-2.0, the FlashInfer team) from
github.com/flashinfer-ai/flashinfer at 784bfaf5bda80259d11a41955ab8c979914197f1
(flashinfer-ai/flashinfer#8186), each at its path under flashinfer/attention/prims_ts/:
mla_decode.py, kernels/mla_decode/helpers/nvfp4_kv.py and
kernels/mla_decode/throughput_2cta/nvfp4_kv_load.py.
Their relative imports resolve only as flashinfer.attention.prims_ts.*, under which
sglang.srt.layers.attention.dsa.prims_ts loads them (SGLANG_ENABLE_PRIMS_TS_SM107=1).
"""

raise ImportError(
    "vendored FlashInfer sources, loaded as flashinfer.attention.prims_ts.* by "
    "sglang.srt.layers.attention.dsa.prims_ts with SGLANG_ENABLE_PRIMS_TS_SM107=1"
)
