#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

/// \brief Operands of the MegaMoE epilogue. Row strides are in elements.
struct MegaMoeEpilogueParams {
  const void* __restrict__ routed;  // [T, H] the MegaMoE top-k reduce output
  const void* __restrict__ shared;  // [T, H] the shared-expert output
  void* __restrict__ residual;      // [T, H], updated in place
  const void* __restrict__ weight;  // [H] the next layer's input RMSNorm weight
  void* __restrict__ out;           // [T, H] the normalized rows
  int64_t routed_stride;
  int64_t shared_stride;
  int64_t residual_stride;
  int64_t out_stride;
  float eps;
};

/**
 * \brief One CTA per token, one 32-byte BF16 vector per thread. x = routed + shared, as
 * the in-place BF16 add rounds it; then RMSNorm's JIT fused add (elementwise/rmsnorm.cuh,
 * FUSED_ADD, one row per CTA, one vector per thread): residual = T(x + residual) and
 * out = T(residual * w * rstd), with that kernel's two FMA chains per thread, xor-butterfly
 * warp sum, in-order cross-warp sum and rounding, so both outputs equal the unfused chain's.
 */
template <int64_t kDim, bool kUsePDL>
__global__
__launch_bounds__(kDim / 16) void megamoe_epilogue_kernel(const MegaMoeEpilogueParams __grid_constant__ params) {
  using namespace device;
  using T2 = packed_t<bf16_t>;
  constexpr uint32_t kPacks = 8;
  using vec_t = AlignedVector<T2, kPacks>;
  constexpr uint32_t kThreads = kDim / (2 * kPacks);
  constexpr uint32_t kWarps = kThreads / kWarpThreads;
  constexpr float kInvDim = 1.0f / static_cast<float>(kDim);
  // `fma.rn.f32.bf16` is a Blackwell instruction; elsewhere it is a convert pair plus an FMA
  constexpr bool kMixedFMA = SGL_ARCH_BLACKWELL_OR_GREATER;
  static_assert(kDim % (2 * kPacks) == 0 && kThreads % kWarpThreads == 0 && kThreads <= 1024);
  using SumOp = ReductionTrait<ReductionOp::SUM, T2>;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid % kWarpThreads;
  const uint32_t warp_id = tid / kWarpThreads;
  const int64_t row = blockIdx.x;
  __shared__ float s_partial[kWarps];

  vec_t weight;
  weight.load(params.weight, tid);  // never written by the producer kernels

  PDLWaitPrimary<kUsePDL>();

  vec_t x, shared, res;
  x.load(assume_global(pointer::offset<bf16_t>(params.routed, row * params.routed_stride)), tid);
  shared.load(assume_global(pointer::offset<bf16_t>(params.shared, row * params.shared_stride)), tid);
  const auto residual_ptr = assume_global(pointer::offset<bf16_t>(params.residual, row * params.residual_stride));
  res.load(residual_ptr, tid);
#pragma unroll
  for (uint32_t j = 0; j < kPacks; ++j) {
    x[j] = SumOp::reduce(x[j], shared[j]);
    x[j] = SumOp::reduce(x[j], res[j]);
  }
  x.store(residual_ptr, tid);

  float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kPacks; ++j) {
    if constexpr (kMixedFMA) {
      const auto& e = unpack(x[j]);
      acc0 = math::fma_f32_bf16(e[0], e[0], acc0);
      acc1 = math::fma_f32_bf16(e[1], e[1], acc1);
    } else {
      const auto v = cast<fp32x2_t>(x[j]);
      acc0 = fmaf(v.x, v.x, acc0);
      acc1 = fmaf(v.y, v.y, acc1);
    }
  }
  float sum = warp::reduce_sum<kWarpThreads>(acc0 + acc1);
  if (lane_id == 0) s_partial[warp_id] = sum;
  __syncthreads();
  sum = 0.0f;
#pragma unroll
  for (uint32_t w = 0; w < kWarps; ++w) {
    sum += s_partial[w];
  }
  const float r = math::rsqrt(sum * kInvDim + params.eps);

  vec_t out;
#pragma unroll
  for (uint32_t j = 0; j < kPacks; ++j) {
    if constexpr (kMixedFMA) {
      const auto& e = unpack(x[j]);
      const auto& g = unpack(weight[j]);
      out[j] = cast<T2>(fp32x2_t{math::fma_f32_bf16(e[0], g[0], -0.0f) * r, math::fma_f32_bf16(e[1], g[1], -0.0f) * r});
    } else {
      const auto v = cast<fp32x2_t>(x[j]);
      const auto g = cast<fp32x2_t>(weight[j]);
      out[j] = cast<T2>(fp32x2_t{v.x * g.x * r, v.y * g.y * r});
    }
  }
  out.store(assume_global(pointer::offset<bf16_t>(params.out, row * params.out_stride)), tid);

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kDim, bool kUsePDL>
struct MegaMoeEpilogueKernel {
  static constexpr uint32_t kThreads = kDim / 16;
  static constexpr uint32_t kVecBytes = 32;

  /// \brief `residual = T(T(routed + shared) + residual)` in place and `out` its RMSNorm.
  static void
  run(const tvm::ffi::TensorView routed,
      const tvm::ffi::TensorView shared,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView out,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLCUDA>();
    for (const auto& t : {routed, shared, residual, out}) {
      TensorMatcher({N, kDim})  //
          .with_strides({-1, 1})
          .with_dtype<bf16_t>()
          .with_device(device_sym)
          .ensure_alignment(kVecBytes)
          .verify(t);
    }
    TensorMatcher({kDim})  // weight
        .with_dtype<bf16_t>()
        .with_device(device_sym)
        .ensure_alignment(kVecBytes)
        .verify(weight);

    const auto params = MegaMoeEpilogueParams{
        .routed = routed.data_ptr(),
        .shared = shared.data_ptr(),
        .residual = residual.data_ptr(),
        .weight = weight.data_ptr(),
        .out = out.data_ptr(),
        .routed_stride = routed.stride(0),
        .shared_stride = shared.stride(0),
        .residual_stride = residual.stride(0),
        .out_stride = out.stride(0),
        .eps = eps,
    };
    const int64_t num_tokens = N.unwrap();
    if (num_tokens == 0) return;
    CHECK_HOST(num_tokens <= INT32_MAX) << "too many tokens: " << num_tokens;
    LaunchKernel(static_cast<uint32_t>(num_tokens), kThreads, device_sym.unwrap())  //
        .enable_pdl(kUsePDL)(megamoe_epilogue_kernel<kDim, kUsePDL>, params);
  }
};

}  // namespace sglang
