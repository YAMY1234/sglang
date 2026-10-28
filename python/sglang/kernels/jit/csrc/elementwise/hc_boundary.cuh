#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <cuda_bf16.h>

// HC4/H2560/rank320 boundary ABI. The enclosing kernel owns PDL, the
// side-stream event joins, the native output reduction and all tile ownership.
// No launch, CTA barrier, grid barrier, or hidden scratch lives in this header.
namespace sglang::device::hc4 {
constexpr int kHidden = 2560;
constexpr int kBranches = 4;
constexpr int kWidth = kBranches * kHidden;
constexpr int kStatTile = 64;
constexpr int kStatParts = kHidden / kStatTile;

__device__ __forceinline__ float inject_gate(const float* partials, int row, int branch) {
  float total = 0.0f;
#pragma unroll
  for (int split = 0; split < 8; ++split)
    total = __fadd_rn(total, partials[(row * 8 + split) * 4 + branch]);
  // Exactly the stock hc_combine apply's scalar expression/math primitive.
  return 2.0f / (1.0f + math::exp(-total / 4));
}

__device__ __forceinline__ bf16_t combine(bf16_t block, bf16_t residual, float gate) {
  return cast<bf16_t>(fmaf(gate, cast<float>(block), cast<float>(residual)));
}

// All 32 lanes must participate. Each lane owns columns lane and lane+32
// of one logical 64-column tile; only lane 0's result is the complete sum.
__device__ __forceinline__ float sumsq64(bf16_t x0, bf16_t x1) {
  const float a = cast<float>(x0), b = cast<float>(x1);
  float sum = __fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b));
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2)
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, delta));
  return sum;
}

__device__ __forceinline__ float inv_rms(const float* stats, int row, int branch, float eps) {
  float sum = 0.0f;
#pragma unroll
  for (int part = 0; part < kStatParts; ++part)
    sum = __fadd_rn(sum, stats[(row * 4 + branch) * kStatParts + part]);
  return math::rsqrt(__fadd_rn(__fmul_rn(sum, 1.0f / kHidden), eps));
}

__device__ __forceinline__ bf16_t normalize(bf16_t x, bf16_t raw_gamma, float inv) {
  const float scale = __fadd_rn(1.0f, cast<float>(raw_gamma));
  return cast<bf16_t>(__fmul_rn(__fmul_rn(cast<float>(x), inv), scale));
}
}  // namespace sglang::device::hc4
