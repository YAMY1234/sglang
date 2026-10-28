// SPDX-License-Identifier: Apache-2.0
// FN adaptation of Qwen3.5 draft_fc.cu @5474b2cdbc61:
// two separately rounded FCs over one embedding and four normalized HC branches.
// No Qwen3.5 TP mailbox or ordinary residual arithmetic is reused.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <cuda_bf16.h>

namespace sglang {
struct Q38MkInputParams {
  const bf16_t *embedding, *hidden, *norm_e, *norm_h, *weight_e, *weight_h;
  float* partial;
  float eps_e, eps_h;
  int rank, world;
};

__device__ __forceinline__ float q38_mk_warp_sum(float x) {
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, mask);
  return x;
}

template <int H>
__device__ void q38_mk_norm(const bf16_t* x, const bf16_t* gamma,
                           bf16_t* output, float eps, float* reductions) {
  float sum = 0.0f;
  for (int j = threadIdx.x; j < H; j += 128) {
    float v = __bfloat162float(x[j]);
    sum = fmaf(v, v, sum);
  }
  sum = q38_mk_warp_sum(sum);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (!lane) reductions[warp] = sum;
  __syncthreads();
  const float inv = rsqrtf((((reductions[0] + reductions[1]) + reductions[2]) + reductions[3]) / H + eps);
  for (int j = threadIdx.x; j < H; j += 128) {
    const float v = __bfloat162float(x[j]) * inv;
    output[j] = __float2bfloat16_rn(v * (1.0f + __bfloat162float(gamma[j])));
  }
  __syncthreads();
}

template <int World>
__global__ __launch_bounds__(128) void q38_mk_mtp_input_partial_kernel(
    const Q38MkInputParams __grid_constant__ p) {
  constexpr int H = 2560, Slice = H / World, Vectors = (Slice + 255) / 256;
  __shared__ alignas(16) bf16_t embedding[H], hidden[4 * H];
  __shared__ float reduction[4];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 4 + warp;
  const int first = p.rank * Slice;
  uint4 we[Vectors], wh[Vectors];
#pragma unroll
  for (int v = 0; v < Vectors; ++v) {
    int col = v * 256 + lane * 8;
    we[v] = col < Slice ? *reinterpret_cast<const uint4*>(p.weight_e + int64_t(row) * H + first + col) : uint4{};
    wh[v] = col < Slice ? *reinterpret_cast<const uint4*>(p.weight_h + int64_t(row) * H + first + col) : uint4{};
  }
  // Pin all static weight operands before the producer wait.
  uint32_t dep = 0;
#pragma unroll
  for (int v = 0; v < Vectors; ++v)
    dep ^= we[v].x ^ we[v].y ^ we[v].z ^ we[v].w ^ wh[v].x ^ wh[v].y ^ wh[v].z ^ wh[v].w;
  asm volatile("{ .reg .pred a,b; setp.eq.u64 a,%1,0; setp.eq.u32 b,%0,1; and.pred a,a,b; @a trap; }"
               :: "r"(dep), "l"(p.weight_e) : "memory");
  device::PDLTriggerSecondary<true>();
  device::PDLWaitPrimary<true>();
  q38_mk_norm<H>(p.embedding, p.norm_e, embedding, p.eps_e, reduction);
  q38_mk_norm<4 * H>(p.hidden, p.norm_h, hidden, p.eps_h, reduction);
  float acc[5] = {};
#pragma unroll
  for (int v = 0; v < Vectors; ++v) {
    const int col = v * 256 + lane * 8;
    if (col < Slice) {
      const uint4 xe = *reinterpret_cast<const uint4*>(embedding + first + col);
      const bf16_t* a = reinterpret_cast<const bf16_t*>(&we[v]);
      const bf16_t* b = reinterpret_cast<const bf16_t*>(&xe);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[0] = fmaf(__bfloat162float(a[j]), __bfloat162float(b[j]), acc[0]);
#pragma unroll
      for (int branch = 0; branch < 4; ++branch) {
        const uint4 xh = *reinterpret_cast<const uint4*>(hidden + branch * H + first + col);
        const bf16_t* a = reinterpret_cast<const bf16_t*>(&wh[v]);
        const bf16_t* b = reinterpret_cast<const bf16_t*>(&xh);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[branch + 1] = fmaf(__bfloat162float(a[j]), __bfloat162float(b[j]), acc[branch + 1]);
      }
    }
  }
#pragma unroll
  for (int branch = 0; branch < 5; ++branch) {
    const float value = q38_mk_warp_sum(acc[branch]);
    if (!lane) p.partial[branch * H + row] = value;
  }
}

struct Q38MkMtpInput {
  static void run(tvm::ffi::TensorView embedding, tvm::ffi::TensorView hidden,
                  tvm::ffi::TensorView norm_e, tvm::ffi::TensorView norm_h,
                  tvm::ffi::TensorView we, tvm::ffi::TensorView wh,
                  tvm::ffi::TensorView partial, float eps_e, float eps_h, int rank, int world) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({1,2560}).with_dtype<bf16_t>().with_device(dev).verify(embedding);
    TensorMatcher({1,10240}).with_dtype<bf16_t>().with_device(dev).verify(hidden);
    TensorMatcher({2560}).with_dtype<bf16_t>().with_device(dev).verify(norm_e);
    TensorMatcher({10240}).with_dtype<bf16_t>().with_device(dev).verify(norm_h);
    TensorMatcher({2560,2560}).with_dtype<bf16_t>().with_device(dev).verify(we);
    TensorMatcher({2560,2560}).with_dtype<bf16_t>().with_device(dev).verify(wh);
    TensorMatcher({5,2560}).with_dtype<float>().with_device(dev).verify(partial);
    CHECK_HOST((world == 1 || world == 4) && rank >= 0 && rank < world);
    Q38MkInputParams p{static_cast<const bf16_t*>(embedding.data_ptr()),
      static_cast<const bf16_t*>(hidden.data_ptr()), static_cast<const bf16_t*>(norm_e.data_ptr()),
      static_cast<const bf16_t*>(norm_h.data_ptr()), static_cast<const bf16_t*>(we.data_ptr()),
      static_cast<const bf16_t*>(wh.data_ptr()), static_cast<float*>(partial.data_ptr()),
      eps_e, eps_h, rank, world};
    LaunchKernel launch(640, 128, dev.unwrap());
    launch.enable_pdl(true);
    if (world == 4) launch(q38_mk_mtp_input_partial_kernel<4>, p);
    else launch(q38_mk_mtp_input_partial_kernel<1>, p);
  }
};
}  // namespace sglang
