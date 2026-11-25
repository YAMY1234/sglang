# GLM-5.2 NVFP4 PD serving: files outside the Python package

Everything the GLM-5.2 serving image builds or loads besides the SGLang package itself.

| Path | What | Used by |
|---|---|---|
| `natives/build_natives.py` | Builds both prefill natives on CPU in the serving image (`torch.utils.cpp_extension`, no CUDA context) and writes `BUILD.json` | the image build |
| `natives/topk-overflow-prefix/native_binding.cc` | Registers `sgl_kernel::fast_topk_transform_fused_overflow_prefix`; the kernel sources are `python/sglang/kernels/aot/{csrc/elementwise/topk_overflow_prefix.cu,include/sgl_kernel/dsa/legacy_radix_topk_overflow_prefix*_v2.cuh}` | prefill with `SGLANG_DSA_TOPK_OVERFLOW_PREFIX=1` (`TOPK_OVERFLOW_NATIVE_SO`); the loader checks the binary against `python/sglang/srt/layers/attention/dsa/topk_overflow_native.sha256` |
| `natives/hcb-prefill-port/` | The DSA prefill top-k selector port (`hcb_prefill::{topk,gather,remap}`); `trt_selector.cuh` is derived from TensorRT-LLM (header inside); `legacy_radix_topk.cuh` is `python/sglang/kernels/aot/include/sgl_kernel/dsa/legacy_radix_topk.cuh` | prefill with `SGLANG_HCB_PREFILL_TRT_PORT=1` (`SGLANG_HCB_PREFILL_PORT_SO`) |
| `moe-configs/draft-config/` | Triton fused-MoE configs for the draft MoE shape (E=256, N=2048) on VR NVL72 | prefill with `SGLANG_MOE_CONFIG_DIR=<repo>/glm52/moe-configs/draft-config` |

The HiCache NIXL DRAM L3 pool service that the prefill `nixl_dram` storage backend talks to is in the package:
`python/sglang/srt/mem_cache/storage/nixl_dram/nixl_dram_l3.py`.

Build (inside the serving image, CPU only):

```bash
CUDA_VISIBLE_DEVICES= CUDA_HOME=/usr/local/cuda MAX_JOBS=4 \
  python3 -B glm52/natives/build_natives.py <repo root> <out dir> [<dir of reference binaries>]
```
