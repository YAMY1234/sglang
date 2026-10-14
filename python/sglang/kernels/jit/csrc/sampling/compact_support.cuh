#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/warp.cuh>

#include <cmath>
#include <cstdint>

namespace sglang {

/** \brief Count every cutoff tie, including entries outside the selected top-k. */
template <int kTopK, int kThreads, int kTile>
__global__ void compact_count_cutoff(const float* __restrict__ probs,
                                     const float* __restrict__ values,
                                     int32_t* __restrict__ counts,
                                     uint32_t vocab_size,
                                     uint32_t num_tiles) {
  static_assert(kThreads % 32 == 0 && kThreads <= 1024);
  const uint32_t row = blockIdx.y;
  const uint32_t lane = device::get_lane_id();
  const uint32_t warp = threadIdx.x / 32;
  const float cutoff = values[row * kTopK + kTopK - 1];
  int32_t count = 0;
  for (uint32_t offset = threadIdx.x; offset < kTile; offset += kThreads) {
    const uint32_t col = blockIdx.x * kTile + offset;
    if (col < vocab_size) count += probs[static_cast<uint64_t>(row) * vocab_size + col] == cutoff;
  }
  count = device::warp::reduce_sum(count);
  __shared__ int32_t warp_counts[kThreads / 32];
  if (lane == 0) warp_counts[warp] = count;
  __syncthreads();
  if (warp == 0) {
    count = lane < kThreads / 32 ? warp_counts[lane] : 0;
    count = device::warp::reduce_sum(count);
    if (lane == 0) counts[row * num_tiles + blockIdx.x] = count;
  }
}

/**
 * \brief Normalize bounded value groups with an unbounded cutoff multiplicity.
 *
 * At most kTopK-1 tokens are strictly higher than the cutoff. Equal-cutoff
 * tokens are one group, so this does not truncate ties to kTopK entries.
 */
template <int kTopK>
__global__ void compact_normalize(const float* __restrict__ values,
                                  const int32_t* __restrict__ counts,
                                  float* __restrict__ higher_probs,
                                  float* __restrict__ cutoff_params,
                                  uint32_t num_tiles,
                                  float top_p) {
  static_assert(kTopK > 0 && kTopK <= 32);
  const uint32_t row = blockIdx.x;
  const uint32_t lane = device::get_lane_id();
  const float cutoff = values[row * kTopK + kTopK - 1];
  int32_t num_cutoff = 0;
  for (uint32_t i = lane; i < num_tiles; i += 32) num_cutoff += counts[row * num_tiles + i];
  num_cutoff = device::warp::reduce_sum(num_cutoff);
  const float value = lane < kTopK ? values[row * kTopK + lane] : cutoff;
  // Select on the original FP32 softmax output: distinct logits can round to
  // equal probabilities, so moving selection before softmax can drop support.
  const float cutoff_weight = cutoff;
  const float weight = lane < kTopK && value > cutoff ? value : 0.0f;
  const float normalizer = device::warp::reduce_sum(weight) + static_cast<float>(num_cutoff) * cutoff_weight;
  const float inverse_normalizer = 1.0f / normalizer;
  const float prob = weight * inverse_normalizer;
  const float cutoff_prob = cutoff_weight * inverse_normalizer;
  const float cdf = device::warp::inclusive_sum(prob);
  // AIR selects top_p times the actual input mass, rather than assuming its
  // floating-point normalization sums to exactly one.
  const float target_mass = top_p * (device::warp::reduce_sum(prob) + static_cast<float>(num_cutoff) * cutoff_prob);
  const uint32_t first = device::warp::reduce_min(lane < kTopK && weight > 0.0f && cdf >= target_mass ? lane : 32u);
  const float threshold = first < 32 ? device::warp::broadcast<32>(prob, first) : cutoff_prob;
  const float kept = prob >= threshold ? prob : 0.0f;
  const float kept_cutoff = cutoff_prob >= threshold ? cutoff_prob : 0.0f;
  const float kept_sum = device::warp::reduce_sum(kept) + static_cast<float>(num_cutoff) * kept_cutoff;
  const float inverse_kept_sum = 1.0f / kept_sum;
  const bool invalid = !(normalizer > 0.0f) || !(kept_sum > 0.0f);
  if (lane < kTopK) higher_probs[row * kTopK + lane] = invalid ? CUDART_NAN_F : kept * inverse_kept_sum;
  if (lane == 0) {
    cutoff_params[row * 2] = cutoff;
    cutoff_params[row * 2 + 1] = invalid ? CUDART_NAN_F : kept_cutoff * inverse_kept_sum;
  }
}

/**
 * \brief Emit the native dense probability ABI with bounded renormalization.
 *
 * Each CTA owns one column tile. Its base writes join before the same CTA
 * scatters higher-support tokens into that tile; there is no inter-CTA race.
 */
template <int kTopK, int kThreads, int kTile>
__global__ void compact_emit(float* __restrict__ output,
                             const float* __restrict__ probs,
                             const float* __restrict__ values,
                             const int64_t* __restrict__ indices,
                             const float* __restrict__ higher_probs,
                             const float* __restrict__ cutoff_params,
                             uint32_t vocab_size) {
  const uint32_t row = blockIdx.y;
  const uint32_t begin = blockIdx.x * kTile;
  const float cutoff = cutoff_params[row * 2];
  const float cutoff_prob = cutoff_params[row * 2 + 1];
  for (uint32_t offset = threadIdx.x; offset < kTile; offset += kThreads) {
    const uint32_t col = begin + offset;
    if (col < vocab_size) {
      const uint64_t pos = static_cast<uint64_t>(row) * vocab_size + col;
      output[pos] = isnan(cutoff_prob) ? cutoff_prob : probs[pos] == cutoff ? cutoff_prob : 0.0f;
    }
  }
  __syncthreads();
  if (threadIdx.x < kTopK) {
    const uint32_t slot = row * kTopK + threadIdx.x;
    const int64_t col = indices[slot];
    if (values[slot] > cutoff && col >= begin && col < begin + kTile && col < vocab_size) {
      output[static_cast<uint64_t>(row) * vocab_size + col] = higher_probs[slot];
    }
  }
}

/// \brief Validate before invoking the external FlashInfer selection operator.
template <int kTopK>
void compact_check_inputs(tvm::ffi::TensorView probs) {
  using namespace host;
  SymbolicSize rows{"rows"}, vocab{"vocab"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  TensorMatcher({rows, vocab}).with_dtype<float>().with_device<kDLCUDA>(device).verify(probs);
  CHECK_HOST(rows.unwrap() > 0 && rows.unwrap() <= 65535) << "compact support row count outside CUDA grid limits";
  CHECK_HOST(vocab.unwrap() >= kTopK && vocab.unwrap() <= (1 << 24)) << "compact support vocabulary size outside exact count range";
}

/// \brief Validate scratch/selected values and launch the three ordered stages.
template <int kTopK, int kThreads, int kTile>
void compact_support(tvm::ffi::TensorView output,
                     tvm::ffi::TensorView probs,
                     tvm::ffi::TensorView values,
                     tvm::ffi::TensorView indices,
                     tvm::ffi::TensorView counts,
                     tvm::ffi::TensorView higher_probs,
                     tvm::ffi::TensorView cutoff_params,
                     float top_p) {
  using namespace host;
  SymbolicSize rows{"rows"}, vocab{"vocab"}, tiles{"tiles"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  TensorMatcher({rows, vocab}).with_dtype<float>().with_device<kDLCUDA>(device).verify(probs).verify(output);
  TensorMatcher({rows, kTopK}).with_dtype<float>().with_device<kDLCUDA>(device).verify(values).verify(higher_probs);
  TensorMatcher({rows, kTopK}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(indices);
  TensorMatcher({rows, tiles}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(counts);
  TensorMatcher({rows, 2}).with_dtype<float>().with_device<kDLCUDA>(device).verify(cutoff_params);
  CHECK_HOST(top_p > 0.0f && top_p <= 1.0f) << "compact support requires top_p in (0, 1]";
  CHECK_HOST(rows.unwrap() > 0 && rows.unwrap() <= 65535 && vocab.unwrap() >= kTopK && vocab.unwrap() <= (1 << 24));
  CHECK_HOST(tiles.unwrap() == div_ceil(vocab.unwrap(), int64_t{kTile}));
  CHECK_HOST(output.data_ptr() != probs.data_ptr()) << "compact support output must not alias probs";
  const uint32_t r = static_cast<uint32_t>(rows.unwrap());
  const uint32_t v = static_cast<uint32_t>(vocab.unwrap());
  const uint32_t p = static_cast<uint32_t>(tiles.unwrap());
  const DLDevice dev = device.unwrap();
  const auto* src = static_cast<const float*>(probs.data_ptr());
  const auto* selected = static_cast<const float*>(values.data_ptr());
  auto* partials = static_cast<int32_t*>(counts.data_ptr());
  auto* high = static_cast<float*>(higher_probs.data_ptr());
  auto* cut = static_cast<float*>(cutoff_params.data_ptr());
  LaunchKernel(dim3(p, r), kThreads, dev)(compact_count_cutoff<kTopK, kThreads, kTile>, src, selected, partials, v, p);
  LaunchKernel(r, 32, dev)(compact_normalize<kTopK>, selected, partials, high, cut, p, top_p);
  LaunchKernel(dim3(p, r), kThreads, dev)(compact_emit<kTopK, kThreads, kTile>, static_cast<float*>(output.data_ptr()), src, selected, static_cast<const int64_t*>(indices.data_ptr()), high, cut, v);
}

}  // namespace sglang
