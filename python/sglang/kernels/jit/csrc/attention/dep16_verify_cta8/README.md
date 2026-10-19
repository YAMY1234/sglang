# Native verify launcher host headers

The two host headers are copied from the Apache-2.0 FlashInfer0.7.1 distribution.
The full distributed license and third-party notices are retained in
`third_party/flashinfer/LICENSE`. Original SHA-256:

- `fmhaKernels.cuh`: `7f2ea5116d4929cfb5350614671c7524da29df49db969f668d2e672bc5c0369c`
- `fmhaRunner.cuh`: `da0e9ac5fe699b8c4586a8d2a87e18a2994343ae1875420776e7d889394d3035`

The modified header fixes KV parallelism and keeps partial statistics disjoint from
partial outputs for the resulting grid. Physical SM queries, CUDA sources, cubins,
reduction order and counter lifecycle use the installed FlashInfer implementation.
The Python loader verifies the original header ABI before selecting this module.
The opt-in module is prepared before CUDA-graph capture; stock dispatch does not
import or build it.
