// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl_nvidia_mega/qwen3_5/sm103/draft/src/draft_gemv.cu
// at 5474b2cdbc61. FN: N=2560, K=6144, TP1, BF16 gate and output.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

namespace sglang {

struct Q38cMtpGatedOprojParams {
  const bf16_t* x;
  const bf16_t* gate;
  const bf16_t* __restrict__ weight;
  bf16_t* output;
  int64_t gate_head_stride;
};

__device__ __forceinline__ uint4 q38c_gate_mul8(const uint4 xv, const uint4 gv) {
  const auto* x = reinterpret_cast<const bf16_t*>(&xv);
  const auto* g = reinterpret_cast<const bf16_t*>(&gv);
  uint4 out;
  auto* o = reinterpret_cast<bf16_t*>(&out);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float t;
    // Match Triton's FP32 sigmoid sequence and BF16 materialization.
    asm("mul.rn.f32 %0, %1, 0fBFB8AA3B;\n\t"
        "ex2.approx.f32 %0, %0;\n\t"
        "add.rn.f32 %0, %0, 0f3F800000;\n\t"
        "div.full.f32 %0, 0f3F800000, %0;\n\t"
        "mul.rn.f32 %0, %2, %0;"
        : "=&f"(t)
        : "f"(__bfloat162float(g[e])), "f"(__bfloat162float(x[e])));
    o[e] = __float2bfloat16_rn(t);
  }
  return out;
}

__global__ __launch_bounds__(128) void q38c_mtp_gated_oproj_kernel(
    const Q38cMtpGatedOprojParams __grid_constant__ params) {
  constexpr int kK = 6144;
  constexpr int kVecs = kK / (8 * 32);
  constexpr int kRows = 4;
  __shared__ alignas(16) bf16_t staged_x[kK];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row = blockIdx.x * kRows + warp;
  uint4 weights[kVecs];
  const auto* wr = reinterpret_cast<const uint4*>(params.weight + int64_t(row) * kK);
#pragma unroll
  for (int j = 0; j < kVecs; ++j) weights[j] = __ldg(wr + j * 32 + lane);

  // Name every loaded component, so ptxas cannot sink any static weight load
  // past the PDL wait. No synthetic volatile load is added to the data path.
  uint32_t dep = 0;
#pragma unroll
  for (int j = 0; j < kVecs; ++j)
    dep ^= weights[j].x ^ weights[j].y ^ weights[j].z ^ weights[j].w;
  asm volatile(
      "{ .reg .pred a, b; setp.eq.u64 a, %1, 0; setp.eq.u32 b, %0, 1; "
      "and.pred a, a, b; @a trap; }"
      :: "r"(dep), "l"(params.weight) : "memory");
  device::PDLWaitPrimary<true>();

  for (int i = threadIdx.x; i < kK / 8; i += 128) {
    const auto xv = reinterpret_cast<const uint4*>(params.x)[i];
    const int k = i * 8;
    const auto* gp = params.gate + (k / 256) * params.gate_head_stride + k % 256;
    const auto gv = *reinterpret_cast<const uint4*>(gp);
    reinterpret_cast<uint4*>(staged_x)[i] = q38c_gate_mul8(xv, gv);
  }
  __syncthreads();
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < kVecs; ++j) {
    const auto xv = reinterpret_cast<const uint4*>(staged_x)[j * 32 + lane];
    const auto* a = reinterpret_cast<const bf16_t*>(&weights[j]);
    const auto* b = reinterpret_cast<const bf16_t*>(&xv);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      acc = fmaf(__bfloat162float(a[e]), __bfloat162float(b[e]), acc);
  }
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1)
    acc = acc + __shfl_xor_sync(0xffffffffu, acc, mask, 32);
  if (lane == 0) params.output[row] = __float2bfloat16_rn(acc);
  device::PDLTriggerSecondary<true>();
}

struct Q38cMtpGatedOproj {
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView gate,
                  tvm::ffi::TensorView weight, tvm::ffi::TensorView output) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({1, 6144}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({1, 24, 256}).with_dtype<bf16_t>().with_device(dev).verify(gate);
    TensorMatcher({2560, 6144}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({1, 2560}).with_dtype<bf16_t>().with_device(dev).verify(output);
    const Q38cMtpGatedOprojParams params{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .gate = static_cast<const bf16_t*>(gate.data_ptr()),
        .weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .output = static_cast<bf16_t*>(output.data_ptr()),
        .gate_head_stride = gate.stride(1),
    };
    LaunchKernel(640, 128, dev.unwrap())
        .enable_pdl(true)(q38c_mtp_gated_oproj_kernel, params);
  }
};

}  // namespace sglang
