#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <cuda_bf16.h>

// Call libdevice's accurate exponential explicitly. A caller built with
// --use_fast_math otherwise replaces expf with __expf, changing the gate at
// BF16 rounding ties even though this header's scalar expression is unchanged.
extern "C" __device__ float __nv_expf(float);

// HC4/H2560/rank320 boundary ABI. The enclosing kernel owns PDL, the
// side-stream event joins, the native output reduction and all tile ownership.
// No launch, CTA barrier, grid barrier, or hidden scratch lives in this header.
namespace sglang::device::hc4 {
constexpr int kHidden = 2560;
constexpr int kBranches = 4;
constexpr int kWidth = kBranches * kHidden;
constexpr int kStatTile = 64;
constexpr int kStatParts = kHidden / kStatTile;

__device__ __forceinline__ float add_rn(float a, float b) {
  float out;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}

__device__ __forceinline__ float mul_rn(float a, float b) {
  float out;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}

__device__ __forceinline__ float inject_gate(const float* partials, int row, int branch) {
  float total = 0.0f;
#pragma unroll
  for (int split = 0; split < 8; ++split)
    total = add_rn(total, partials[(row * 8 + split) * 4 + branch]);
  const float denominator = add_rn(1.0f, __nv_expf(mul_rn(-total, 0.25f)));
  float gate;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(gate) : "f"(2.0f), "f"(denominator));
  return gate;
}

__device__ __forceinline__ bf16_t combine(bf16_t block, bf16_t residual, float gate) {
  float out;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out)
      : "f"(gate), "f"(cast<float>(block)), "f"(cast<float>(residual)));
  return cast<bf16_t>(out);
}

// All 32 lanes must participate. Each lane owns columns lane and lane+32
// of one logical 64-column tile; only lane 0's result is the complete sum.
__device__ __forceinline__ float sumsq64(bf16_t x0, bf16_t x1) {
  const float a = cast<float>(x0), b = cast<float>(x1);
  float sum = add_rn(mul_rn(a, a), mul_rn(b, b));
#pragma unroll
  for (int delta = 16; delta > 0; delta /= 2)
    sum = add_rn(sum, __shfl_down_sync(0xffffffffu, sum, delta));
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
