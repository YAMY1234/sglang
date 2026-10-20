// One-row draft input norm + qkv GEMV: x = bf16(h * rsqrt(mean(h^2) + eps) * (1 + gamma)) (GemmaRMSNorm, fp32),
// y[n] = bf16(sum_k W[n, k] * x[k]) (fp32 accumulation). One warp per row with the whole row held in registers,
// loaded before the PDL wait together with gamma (weights are static), so the GEMV's weight stream runs under the
// producer of h. Every CTA normalizes the full row itself (same reduction order in every CTA).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

// Five 128-thread CTAs per SM (<= 102 registers a thread) keep all 2560 TP8 rows resident in one wave.
constexpr int kMinBlocks = 5;

struct MkdQkvParams {
  const bf16_t* h;                   // [K], written by the previous grid
  const bf16_t* __restrict__ gamma;  // [K], raw GemmaRMSNorm weight (the kernel adds 1)
  const bf16_t* __restrict__ w;      // [N, K]
  bf16_t* y;                         // [N]
  float eps;
  int32_t n;
};

/**
 * \brief kRows warps per CTA; warp w of CTA c owns row kRows * c + w (the grid covers every row in one wave).
 * \tparam kK    reduction length (a multiple of 256 and of 8 * kRows * 32)
 * \tparam kRows warps per CTA
 */
template <int kK, int kRows, bool kUsePDL>
__global__ void __launch_bounds__(kRows * 32, kMinBlocks) mkd_qkv_kernel(const MkdQkvParams __grid_constant__ p) {
  using namespace device;
  constexpr int kThreads = kRows * 32;
  constexpr int kVecs = kK / 8 / 32;
  constexpr int kPer = kK / 8 / kThreads;
  static_assert(kK % 256 == 0 && kK % (8 * kThreads) == 0, "K must tile the CTA");
  __shared__ alignas(16) bf16_t s_x[kK];
  __shared__ alignas(16) bf16_t s_g[kK];
  __shared__ float s_red[kRows];
  const int warp = static_cast<int>(threadIdx.x / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  const int row = static_cast<int>(blockIdx.x) * kRows + warp;
  PDLTriggerSecondary<kUsePDL>();
  uint4 wv[kVecs];
  if (row < p.n) {
    const uint4* wr = reinterpret_cast<const uint4*>(p.w + static_cast<int64_t>(row) * kK);
#pragma unroll
    for (int j = 0; j < kVecs; ++j)
      wv[j] = __ldg(wr + j * 32 + lane);
  }
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = static_cast<int>(threadIdx.x) + j * kThreads;
    reinterpret_cast<uint4*>(s_g)[i] = __ldg(reinterpret_cast<const uint4*>(p.gamma) + i);
  }
  PDLWaitPrimary<kUsePDL>();
  uint4 hv[kPer];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    hv[j] = reinterpret_cast<const uint4*>(p.h)[static_cast<int>(threadIdx.x) + j * kThreads];
    const bf16_t* a = reinterpret_cast<const bf16_t*>(&hv[j]);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float f = cast<fp32_t>(a[e]);
      ss = fmaf(f, f, ss);
    }
  }
  ss = warp::reduce_sum(ss);
  if (lane == 0) s_red[warp] = ss;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int r = 0; r < kRows; ++r)
    total += s_red[r];
  const float inv = rsqrtf(total / static_cast<float>(kK) + p.eps);
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = static_cast<int>(threadIdx.x) + j * kThreads;
    const uint4 gv = reinterpret_cast<const uint4*>(s_g)[i];
    const bf16_t* a = reinterpret_cast<const bf16_t*>(&hv[j]);
    const bf16_t* g = reinterpret_cast<const bf16_t*>(&gv);
    uint4 xv;
    bf16_t* x = reinterpret_cast<bf16_t*>(&xv);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      x[e] = cast<bf16_t>(cast<fp32_t>(a[e]) * inv * (1.0f + cast<fp32_t>(g[e])));
    reinterpret_cast<uint4*>(s_x)[i] = xv;
  }
  __syncthreads();
  if (row >= p.n) return;
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    const uint4 xv = reinterpret_cast<const uint4*>(s_x)[j * 32 + lane];
    const bf16_t* a = reinterpret_cast<const bf16_t*>(&wv[j]);
    const bf16_t* b = reinterpret_cast<const bf16_t*>(&xv);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      acc = fmaf(cast<fp32_t>(a[e]), cast<fp32_t>(b[e]), acc);
  }
  acc = warp::reduce_sum(acc);
  if (lane == 0) p.y[row] = cast<bf16_t>(acc);
}

template <int kK, int kRows, bool kUsePDL>
struct MkdQkv {
  static void
  run(const tvm::ffi::TensorView h,
      const tvm::ffi::TensorView gamma,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView y,
      double eps) {
    using namespace host;
    auto N = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kK}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(h);
    TensorMatcher({kK}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(gamma);
    TensorMatcher({N, kK}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(device).verify(y);
    const int n = static_cast<int>(N.unwrap());
    const auto params = MkdQkvParams{
        .h = static_cast<const bf16_t*>(h.data_ptr()),
        .gamma = static_cast<const bf16_t*>(gamma.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .y = static_cast<bf16_t*>(y.data_ptr()),
        .eps = static_cast<float>(eps),
        .n = n,
    };
    const auto kernel = mkd_qkv_kernel<kK, kRows, kUsePDL>;
    LaunchKernel(static_cast<uint32_t>((n + kRows - 1) / kRows), kRows * 32, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
