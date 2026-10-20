// L2 prefetch of a weight, one prefetch.global.L2::evict_last per 128-byte line spread over many CTAs, issued on a side
// stream while the GPU waits on a latency-bound all-reduce so the next projection reads its weight from L2. Hints only.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

__global__ void l2_line_prefetch_kernel(const char* base, int64_t bytes) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x * 128;
  for (int64_t off = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 128; off < bytes; off += stride) {
    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(base + off));
  }
}

struct L2LinePrefetch {
  static void run(const tvm::ffi::TensorView weight, int64_t bytes, int64_t ctas) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({-1, -1}).with_device(device).verify(weight);
    CHECK_HOST(bytes > 0 && ctas > 0) << "prefetch needs a positive size and grid";
    LaunchKernel(static_cast<uint32_t>(ctas), 256, device.unwrap())(
        l2_line_prefetch_kernel, static_cast<const char*>(weight.data_ptr()), bytes);
  }
};

}  // namespace sglang
