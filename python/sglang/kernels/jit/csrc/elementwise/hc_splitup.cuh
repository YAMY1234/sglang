#pragma once

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

namespace sglang::hc_splitup {

constexpr int kHidden = 10240;
constexpr int kLowrank = 320;
constexpr int kSplit = 16;
constexpr int kSlice = kHidden / kSplit;
constexpr int kColumns = 32;
constexpr int kThreads = 128;
// One 16-byte pad rotates successive ldmatrix rows across the 32 banks.
constexpr int kStride = kSlice + 8;
constexpr int kSmemBytes = (kColumns + 16) * kStride * 2;

struct Params {
  const uint16_t* input;
  const uint16_t* weight;
  float* partial;
  int rows;
  int64_t* trace;
};

__device__ __forceinline__ uint32_t shared_address(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Each CTA owns N32 x K640. Weight loading is independent of the preceding
// producer, and no cluster or grid-wide synchronization is needed.
template <bool kTrace>
__global__ __launch_bounds__(kThreads) void hc_splitup_down_partial(const Params __grid_constant__ p) {
  using namespace device;
  if constexpr (kTrace) {
    if (threadIdx.x == 0) {
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(p.trace[blockIdx.x * 5]));
      uint32_t smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      p.trace[blockIdx.x * 5 + 4] = smid;
    }
  }
  PDLTriggerSecondary<true>();
  extern __shared__ __align__(128) uint16_t smem[];
  uint16_t* sw = smem;
  uint16_t* sx = smem + kColumns * kStride;
  const int tid = threadIdx.x;
  const int split = blockIdx.x / (kLowrank / kColumns);
  const int column = (blockIdx.x % (kLowrank / kColumns)) * kColumns;

#pragma unroll
  for (int it = 0; it < kColumns * kSlice / (8 * kThreads); ++it) {
    const int vec = it * kThreads + tid;
    const int n = vec / (kSlice / 8);
    const int k = (vec % (kSlice / 8)) * 8;
    const auto dst = shared_address(sw + n * kStride + k);
    const auto src = p.weight + (column + n) * kHidden + split * kSlice + k;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    if (it % 4 == 3) asm volatile("cp.async.commit_group;" ::: "memory");
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  if constexpr (kTrace) {
    if (tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(p.trace[blockIdx.x * 5 + 1]));
  }
  PDLWaitPrimary<true>();
  if constexpr (kTrace) {
    if (tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(p.trace[blockIdx.x * 5 + 2]));
  }

#pragma unroll
  for (int vec = tid; vec < 16 * kSlice / 8; vec += kThreads) {
    const int row = vec / (kSlice / 8);
    const int k = (vec % (kSlice / 8)) * 8;
    const uint4 value = row < p.rows
        ? *reinterpret_cast<const uint4*>(p.input + row * kHidden + split * kSlice + k)
        : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(sx + row * kStride + k) = value;
  }
  __syncthreads();

  const int lane = tid % 32;
  const int warp = tid / 32;
  const int arow = (lane / 8 % 2) * 8 + lane % 8;
  const int acol = (lane / 16) * 8;
  const int brow = warp * 8 + lane % 8;
  const int bcol = (lane / 8 % 2) * 8;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < kSlice; k += 16) {
    uint32_t a[4], b[2];
    const auto ap = shared_address(sx + arow * kStride + k + acol);
    const auto bp = shared_address(sw + brow * kStride + k + bcol);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(ap));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(bp));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  const int row = lane / 4;
  const int n = column + warp * 8 + (lane % 4) * 2;
  if (row < p.rows) {
    float* out = p.partial + (split * p.rows + row) * kLowrank + n;
    out[0] = acc[0];
    out[1] = acc[1];
  }
  if (row + 8 < p.rows) {
    float* out = p.partial + (split * p.rows + row + 8) * kLowrank + n;
    out[0] = acc[2];
    out[1] = acc[3];
  }
  if constexpr (kTrace) {
    __syncthreads();
    if (tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(p.trace[blockIdx.x * 5 + 3]));
  }
}

struct HcSplitUp {
  static void down(tvm::ffi::TensorView input, tvm::ffi::TensorView weight, tvm::ffi::TensorView partial) {
    run<false>(input, weight, partial, partial);
  }
  static void down_trace(tvm::ffi::TensorView input, tvm::ffi::TensorView weight,
                         tvm::ffi::TensorView partial, tvm::ffi::TensorView trace) {
    run<true>(input, weight, partial, trace);
  }
  template <bool kTrace>
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView weight,
                  tvm::ffi::TensorView partial, tvm::ffi::TensorView trace) {
    using namespace host;
    SymbolicSize rows{"rows"};
    SymbolicDevice dev;
    dev.set_options<kDLCUDA>();
    TensorMatcher({rows, kHidden}).with_dtype<bf16_t>().with_device(dev).verify(input);
    TensorMatcher({kLowrank, kHidden}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({kSplit, rows, kLowrank}).with_dtype<float>().with_device(dev).verify(partial);
    CHECK_HOST(rows.unwrap() > 0 && rows.unwrap() <= 16);
    if constexpr (kTrace) TensorMatcher({160, 5}).with_dtype<int64_t>().with_device(dev).verify(trace);
    const Params p{static_cast<const uint16_t*>(input.data_ptr()),
                   static_cast<const uint16_t*>(weight.data_ptr()),
                   static_cast<float*>(partial.data_ptr()), static_cast<int>(rows.unwrap()),
                   kTrace ? static_cast<int64_t*>(trace.data_ptr()) : nullptr};
    runtime::set_max_dynamic_smem_per_device<hc_splitup_down_partial<kTrace>>(dev.unwrap(), kSmemBytes);
    LaunchKernel(kLowrank / kColumns * kSplit, kThreads, dev.unwrap(), kSmemBytes)
        .enable_pdl(true)(hc_splitup_down_partial<kTrace>, p);
  }
};

}  // namespace sglang::hc_splitup
