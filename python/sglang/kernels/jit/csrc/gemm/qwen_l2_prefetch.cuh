// Bulk L2 prefetch of a byte range (cp.async.bulk.prefetch.L2), meant for a side stream while the main stream waits
// on another rank: the projection GEMM that follows reads its weights from L2 instead of HBM. Fire and forget: the
// kernel only issues the prefetches, so it neither orders nor delays later readers.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace qwen_l2_prefetch {

inline constexpr uint32_t kThreads = 128;
// Bytes per prefetch instruction; the single-GPU T1 measured 16 KiB with one CTA per SM.
inline constexpr uint64_t kChunk = 16384;

__global__ __launch_bounds__(kThreads) void l2_prefetch_kernel(const char* base, uint64_t nbytes) {
  const uint64_t chunks = (nbytes + kChunk - 1) / kChunk;
  const uint64_t stride = uint64_t{gridDim.x} * kThreads;
  for (uint64_t i = uint64_t{blockIdx.x} * kThreads + threadIdx.x; i < chunks; i += stride) {
    const uint64_t offset = i * kChunk;
    // Bulk prefetch sizes are multiples of 16 bytes; a ragged tail below 16 bytes is skipped.
    const uint32_t size = static_cast<uint32_t>(min(kChunk, nbytes - offset)) & ~15u;
    if (size != 0) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + offset), "r"(size) : "memory");
    }
  }
}

struct L2Prefetch {
  static void run(const tvm::ffi::TensorView bytes, int64_t ctas) {
    using namespace host;
    auto N_ = SymbolicSize{"bytes"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N_}).with_dtype<uint8_t>().with_device(device).verify(bytes);
    const auto nbytes = static_cast<uint64_t>(N_.unwrap());
    CHECK_HOST(reinterpret_cast<uintptr_t>(bytes.data_ptr()) % 16 == 0) << "l2_prefetch: base must be 16B aligned";
    CHECK_HOST(ctas > 0) << "l2_prefetch: ctas " << ctas;
    if (nbytes < 16) return;
    LaunchKernel(static_cast<uint32_t>(ctas), kThreads, device.unwrap())(
        l2_prefetch_kernel, static_cast<const char*>(bytes.data_ptr()), nbytes);
  }
};

}  // namespace qwen_l2_prefetch

using qwen_l2_prefetch::L2Prefetch;

}  // namespace sglang
