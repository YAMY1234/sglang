#pragma once
#include "grouped_gemma_rmsnorm.cuh"

namespace sglang {

struct Q38cMtpInputAddNormParams {
  const bf16_t* embedding;
  const bf16_t* hidden;
  const bf16_t* weight;
  bf16_t* combined;
  bf16_t* normalized;
  float eps;
};

__global__ __launch_bounds__(160) void q38c_mtp_input_add_norm_kernel(
    const Q38cMtpInputAddNormParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<bf16_t>;
  using Storage = AlignedVector<Float2, 8>;
  constexpr uint32_t kThreads = 160;
  constexpr uint32_t kWarps = 5;
  const uint32_t bid = blockIdx.x;
  const uint32_t branch = bid % 4;
  const uint32_t row = bid / 4;
  const auto gmem = tile::Memory<Storage>::cta(kThreads);
  __shared__ float smem[kWarpThreads];

  // HC down's weight DMA warp does not wait; its activation DMA warp does.
  PDLTriggerSecondary<true>();
  const auto weight_vec = gmem.load(params.weight + branch * 2560, 0);
  PDLWaitPrimary<true>();
  const auto embed_vec = gmem.load(params.embedding + row * 2560, 0);
  const auto hidden_vec = gmem.load(params.hidden + bid * 2560, 0);
  Storage combined;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [ex, ey] = cast<fp32x2_t>(embed_vec[i]);
    const auto [hx, hy] = cast<fp32x2_t>(hidden_vec[i]);
    // Retain the original torch.add BF16 materialization before HC norm.
    combined[i] = cast<Float2>(fp32x2_t{ex + hx, ey + hy});
  }

  // Same input slices, warp reduction and two broadcasts as grouped Gemma norm.
  float sum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [x, y] = cast<fp32x2_t>(combined[i]);
    sum += x * x + y * y;
  }
  sum = warp::reduce_sum(sum);
  const auto warp_id = threadIdx.x / kWarpThreads;
  smem[warp_id] = sum;
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    const auto local_sum = tx < kWarps ? smem[tx] : 0.0f;
    sum = warp::reduce_sum(local_sum);
    smem[tx] = math::rsqrt(sum / 2560 + params.eps);
  }
  __syncthreads();
  const float inv_rms = smem[warp_id];
  Storage normalized;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [x, y] = cast<fp32x2_t>(combined[i]);
    const auto [wx, wy] = cast<fp32x2_t>(weight_vec[i]);
    normalized[i] = cast<Float2>(
        fp32x2_t{x * inv_rms * (1.0f + wx), y * inv_rms * (1.0f + wy)});
  }
  gmem.store(params.combined + bid * 2560, combined, 0);
  gmem.store(params.normalized + bid * 2560, normalized, 0);
}

struct Q38cMtpInputAddNorm {
  static void run(
      tvm::ffi::TensorView embedding,
      tvm::ffi::TensorView hidden,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView combined,
      tvm::ffi::TensorView normalized,
      float eps) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({rows, 2560}).with_dtype<bf16_t>().with_device(dev).verify(embedding);
    TensorMatcher({rows, 10240}).with_dtype<bf16_t>().with_device(dev)
        .verify(hidden).verify(combined).verify(normalized);
    TensorMatcher({10240}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    const Q38cMtpInputAddNormParams params{
        .embedding = static_cast<const bf16_t*>(embedding.data_ptr()),
        .hidden = static_cast<const bf16_t*>(hidden.data_ptr()),
        .weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .combined = static_cast<bf16_t*>(combined.data_ptr()),
        .normalized = static_cast<bf16_t*>(normalized.data_ptr()),
        .eps = eps,
    };
    LaunchKernel(static_cast<uint32_t>(rows.unwrap()) * 4, 160, dev.unwrap())
        .enable_pdl(true)(q38c_mtp_input_add_norm_kernel, params);
  }
};

}  // namespace sglang
