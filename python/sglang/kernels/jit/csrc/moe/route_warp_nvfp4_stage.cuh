// MoE routing and FlashInfer MegaMoE NVFP4 input staging in one launch.
//
// At decode, three small launches sit between the router GEMM and the FlashInfer
// SM107 MegaMoE kernel: route_warp_kernel, the padded-row id mask, and FlashInfer's
// fused input stager, which quantizes the BF16 rows to NVFP4 and copies the routes
// into the MegaMoE workspace. Neither the routing nor the quantization reads the
// other's output, so here the first CTAs route one token per warp and write the
// routes to both the router outputs and the workspace, and the remaining CTAs each
// quantize one row into the workspace and mask the workspace's route tail.
//
// Routing is route_warp_kernel's code for single-group sigmoid routing, compiled
// with the same flags, so ids and weights are bit-identical. The quantization
// issues the operation sequence of FlashInfer 0.7.3's NVFP4 stager
// (DataPreprocess.nvfp4_quant_and_process_impl with an offline norm constant and
// scale multiplier): per 16-element block, amax by max.f32 over |x|, the E4M3
// scale from cvt.rn.satfinite.e4m3x2.f32(amax * scale_multiplier), the encode
// scale min(norm_const * rcp.approx.ftz(scale), FLT_MAX) * min(scale * 1e30, 1),
// and cvt.rn.satfinite.e2m1x2.f32 of x * encode scale, low nibble first.

#pragma once

#include "route_warp.cuh"
#include <cfloat>

namespace sglang {

namespace device::route_warp_nvfp4_stage::ptx {

// The E4M3 byte of x (satfinite, round to nearest even).
SGL_DEVICE uint32_t f32_to_e4m3(float x) {
  uint16_t v;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(v) : "f"(0.0f), "f"(x));
  return v & 0xffu;
}

SGL_DEVICE float e4m3_to_f32(uint32_t byte) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(static_cast<uint16_t>(byte)));
  return __half2float(__ushort_as_half(static_cast<uint16_t>(h2 & 0xffffu)));
}

// Eight values to eight E2M1 nibbles, element 2i in the low nibble of byte i.
SGL_DEVICE uint32_t f32x8_to_e2m1x8(const float (&v)[8]) {
  uint32_t out;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}

}  // namespace device::route_warp_nvfp4_stage::ptx

namespace moe::route_warp_nvfp4_stage {

using route_warp::kVecSize;
using route_warp::order_key;
using route_warp::Params;
using route_warp::smem_pitch;
namespace rw_ptx = device::route_warp::ptx;
namespace q_ptx = device::route_warp_nvfp4_stage::ptx;

inline constexpr uint32_t kNumWarps = 4;
inline constexpr uint32_t kBlockSize = kNumWarps * 32;
inline constexpr uint32_t kSfVec = 16;  // NVFP4 elements per E4M3 scale

struct StageParams {
  const bf16_t* __restrict__ x;    // [T, H], row stride `x_stride`
  uint2* __restrict__ x_q;         // [capacity, H / 16] blocks of 16 E2M1 values
  uint8_t* __restrict__ x_sf;      // [capacity, H / 16] E4M3 scales
  int32_t* __restrict__ ids;       // [capacity, K] workspace routes
  float* __restrict__ weights;     // [capacity, K]
  int32_t* __restrict__ tail_ids;  // ids + T * K
  int64_t x_stride;
  uint32_t num_blocks;  // H / 16
  uint32_t tail_elems;  // (capacity - T) * K
  float norm_const;
  float scale_multiplier;
};

// route_warp_kernel<E, K, 1, 1, SIGMOID, T, TB, kBias=true, kUsePDL, false, false, kNumWarps>, with each
// route also written to the workspace (a padded row as id -1 and weight 0, as the stager writes it).
template <uint32_t E, uint32_t K, typename T, typename TB, bool kUsePDL>
SGL_DEVICE void route_rows(const Params& p, const StageParams& s, const uint32_t block) {
  using namespace device;
  constexpr uint32_t L = E / 32;          // experts per lane
  constexpr uint32_t M = std::min(L, K);  // pops one lane can take
  constexpr uint32_t V = kVecSize<T, L>;
  constexpr uint32_t VB = kVecSize<TB, L>;
  static_assert(E % 32 == 0 && L <= 16 && K >= 1 && K <= 32);

  constexpr uint32_t kPitch = smem_pitch(L);
  __shared__ float s_val[kNumWarps][32 * kPitch];
  __shared__ uint32_t s_key[kNumWarps][32 * kPitch];

  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t warp_id = threadIdx.x / 32;
  const uint32_t token = block * kNumWarps + warp_id;
  const uint32_t base = lane_id * L;

  if (token >= p.num_tokens) return;
  const auto bias = static_cast<const TB*>(p.bias);
  PDLWaitPrimary<kUsePDL>();

  const auto out_w = p.out_w + static_cast<int64_t>(token) * K;
  const auto out_i = p.out_i + static_cast<int64_t>(token) * K;
  const auto stage_w = s.weights + static_cast<int64_t>(token) * K;
  const auto stage_i = s.ids + static_cast<int64_t>(token) * K;
  if (p.num_non_padded != nullptr && token >= static_cast<uint32_t>(*p.num_non_padded)) {
    if (lane_id < K) {
      out_w[lane_id] = 0.0f;
      out_i[lane_id] = -1;
      stage_w[lane_id] = 0.0f;
      stage_i[lane_id] = -1;
    }
    return;
  }

  // ---- load and score ----
  const auto row = static_cast<const T*>(p.logits) + static_cast<int64_t>(token) * p.logits_stride + base;
  AlignedVector<T, V> xv[L / V];
#pragma unroll
  for (uint32_t i = 0; i < L / V; ++i) {
    xv[i].load(row, i);
  }
  AlignedVector<TB, VB> bv[L / VB];
#pragma unroll
  for (uint32_t i = 0; i < L / VB; ++i) {
    bv[i].load(bias + base, i);
  }

  float val[L];
  float biased[L];
  uint32_t key[L];
#pragma unroll
  for (uint32_t j = 0; j < L; ++j) {
    auto x = cast<float>(xv[j / V][j % V]);
    val[j] = route_warp::sigmoid(x);
    biased[j] = val[j] + cast<float>(bv[j / VB][j % VB]);
    // NaN ranks last (as DeepGEMM's mega-gate); + 0 turns -0 into +0.
    biased[j] = (biased[j] == biased[j]) ? biased[j] + 0.0f : -1e30f;
    key[j] = order_key(biased[j]);
    s_val[warp_id][lane_id * kPitch + j] = val[j];
  }

  // ---- top-K over the lane heads ----
#pragma unroll
  for (uint32_t j = 0; j < L; ++j) {
    s_key[warp_id][lane_id * kPitch + j] = key[j];
  }
  merge_sort(key);
  uint32_t my_key = 0, my_lane = 0, my_run = 0;
  uint32_t prev_key = 0, prev_lane = 32, run = 0;
#pragma unroll
  for (uint32_t k = 0; k < K; ++k) {
    const auto m = warp::reduce_max(key[0]);
    const auto winner = warp::reduce_min(key[0] == m ? lane_id : 32u);
    const bool pop = lane_id == winner;
#pragma unroll
    for (uint32_t i = 0; i + 1 < M; ++i) {
      key[i] = pop ? key[i + 1] : key[i];
    }
    key[M - 1] = pop ? route_warp::kKeyTaken : key[M - 1];
    run = (m == prev_key && winner == prev_lane) ? run + 1 : 0;
    prev_key = m;
    prev_lane = winner;
    my_key = (lane_id == k) ? m : my_key;
    my_lane = (lane_id == k) ? winner : my_lane;
    my_run = (lane_id == k) ? run : my_run;
  }

  __syncwarp();
  uint32_t expert = 0;
  float w = 0.0f;
  if (lane_id < K) {
    uint32_t hits = 0;
#pragma unroll
    for (uint32_t j = 0; j < L; ++j) {
      hits |= (s_key[warp_id][my_lane * kPitch + j] == my_key ? 1u : 0u) << j;
    }
    // Equal keys of one lane pop in id order.
    for (uint32_t r = 0; r < my_run; ++r) {
      hits &= hits - 1;
    }
    const uint32_t j = __ffs(hits) - 1;
    expert = my_lane * L + j;
    w = s_val[warp_id][my_lane * kPitch + j];
  }

  // ---- weights ----
  if (p.renormalize) {
    const auto sum = warp::reduce_sum<1, std::bit_ceil(K)>(w);
    const auto norm = p.renorm_eps > 0.0f ? sum + p.renorm_eps : (sum > 0.0f ? sum : 1.0f);
    w = rw_ptx::div_full(w, norm);
  }
  if (p.apply_scale) w *= p.scale;
  if (lane_id < K) {
    out_w[lane_id] = w;
    out_i[lane_id] = static_cast<int32_t>(expert);
    stage_w[lane_id] = w;
    stage_i[lane_id] = static_cast<int32_t>(expert);
  }
}

// One row of FlashInfer's NVFP4 stager (offline scale), then this CTA's share of the route tail mask.
template <bool kUsePDL>
SGL_DEVICE void stage_row(const StageParams& s, const uint32_t row, const uint32_t num_rows) {
  using namespace device;
  PDLWaitPrimary<kUsePDL>();
  const auto x_row = s.x + static_cast<int64_t>(row) * s.x_stride;
  const auto q_row = s.x_q + static_cast<int64_t>(row) * s.num_blocks;
  const auto sf_row = s.x_sf + static_cast<int64_t>(row) * s.num_blocks;
  for (uint32_t b = threadIdx.x; b < s.num_blocks; b += kBlockSize) {
    AlignedVector<bf16_t, 8> lo, hi;
    lo.load(x_row + b * kSfVec);
    hi.load(x_row + b * kSfVec + 8);
    float v[kSfVec];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      v[i] = __bfloat162float(lo[i]);
      v[i + 8] = __bfloat162float(hi[i]);
    }
    float amax = fmaxf(fabsf(v[0]), 0.0f);
#pragma unroll
    for (uint32_t i = 1; i < kSfVec; ++i) {
      amax = fmaxf(amax, fabsf(v[i]));
    }
    const uint32_t sf = q_ptx::f32_to_e4m3(amax * s.scale_multiplier);
    const float sf_f = q_ptx::e4m3_to_f32(sf);
    // An all-zero block (scale 0) encodes as zeros, not NaN.
    const float enc = fminf(s.norm_const * rw_ptx::rcp_approx_ftz(sf_f), FLT_MAX) * fminf(sf_f * 1.0e30f, 1.0f);
    float lo8[8], hi8[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      lo8[i] = v[i] * enc;
      hi8[i] = v[i + 8] * enc;
    }
    q_row[b] = make_uint2(q_ptx::f32x8_to_e2m1x8(lo8), q_ptx::f32x8_to_e2m1x8(hi8));
    sf_row[b] = static_cast<uint8_t>(sf);
  }
  // Rows [T, capacity) of the workspace routes must not dispatch: -1, as the stager's fused tail mask.
  for (uint32_t i = row * kBlockSize + threadIdx.x; i < s.tail_elems; i += num_rows * kBlockSize) {
    s.tail_ids[i] = -1;
  }
}

template <uint32_t E, uint32_t K, typename T, typename TB, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize) void route_warp_nvfp4_stage_kernel(
    const __grid_constant__ Params p, const __grid_constant__ StageParams s) {
  const uint32_t num_route_blocks = device::div_ceil(p.num_tokens, kNumWarps);
  if (blockIdx.x < num_route_blocks) {
    route_rows<E, K, T, TB, kUsePDL>(p, s, blockIdx.x);
  } else {
    stage_row<kUsePDL>(s, blockIdx.x - num_route_blocks, p.num_tokens);
  }
}

template <uint32_t E, uint32_t K, typename T, typename TB, bool kUsePDL>
struct RouteWarpNvfp4StageKernel {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_non_padded,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf,
      const tvm::ffi::TensorView stage_ids,
      const tvm::ffi::TensorView stage_weights,
      const float scale,
      const bool renormalize,
      const bool apply_scale,
      const float norm_const,
      const float scale_multiplier) {
    using namespace host;
    constexpr uint32_t L = E / 32;
    auto N_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden"};
    auto C_ = SymbolicSize{"capacity"};
    auto B_ = SymbolicSize{"num_scale_blocks"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLCUDA>();
    TensorMatcher({N_, E})  //
        .with_strides({-1, 1})
        .with_dtype<T>()
        .with_device(device_sym)
        .ensure_alignment(kVecSize<T, L> * sizeof(T))
        .verify(logits);
    TensorMatcher({E})  //
        .with_dtype<TB>()
        .with_device(device_sym)
        .ensure_alignment(kVecSize<TB, L> * sizeof(TB))
        .verify(bias);
    TensorMatcher({N_, K}).with_dtype<float>().with_device(device_sym).verify(out_w);
    TensorMatcher({N_, K}).with_dtype<int32_t>().with_device(device_sym).verify(out_i);
    TensorMatcher({N_, H_})  //
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device_sym)
        .ensure_alignment(16)
        .verify(x);
    TensorMatcher({C_, -1}).with_dtype<uint8_t>().with_device(device_sym).ensure_alignment(16).verify(x_q);
    TensorMatcher({C_, B_}).with_dtype<uint8_t>().with_device(device_sym).verify(x_sf);
    TensorMatcher({C_, K}).with_dtype<int32_t>().with_device(device_sym).verify(stage_ids);
    TensorMatcher({C_, K}).with_dtype<float>().with_device(device_sym).verify(stage_weights);
    const auto num_tokens = N_.unwrap();
    const auto hidden = H_.unwrap();
    const auto capacity = C_.unwrap();
    // The stager's fused path: an exact-width scale plane, so hidden % 64 == 0.
    RuntimeCheck(hidden % 64 == 0, "route_warp_nvfp4_stage: hidden must be a multiple of 64, got ", hidden);
    RuntimeCheck(x_q.size(1) * 2 == hidden, "route_warp_nvfp4_stage: x_q must hold hidden / 2 bytes per row");
    RuntimeCheck(B_.unwrap() * kSfVec == hidden, "route_warp_nvfp4_stage: x_sf must hold hidden / 16 scales per row");
    RuntimeCheck(num_tokens <= capacity, "route_warp_nvfp4_stage: ", num_tokens, " tokens exceed capacity ", capacity);

    auto p = Params{};
    p.logits = logits.data_ptr();
    p.bias = bias.data_ptr();
    p.out_w = static_cast<float*>(out_w.data_ptr());
    p.out_i = static_cast<int32_t*>(out_i.data_ptr());
    if (num_non_padded.has_value()) {
      // A 0-d or [1] device scalar, so no shape matcher.
      const auto& n = num_non_padded.value();
      CHECK_HOST(is_type<int32_t>(n.dtype()) && n.numel() == 1 && n.device().device_type == kDLCUDA);
      p.num_non_padded = static_cast<const int32_t*>(n.data_ptr());
    }
    p.logits_stride = logits.stride(0);
    p.num_tokens = static_cast<uint32_t>(num_tokens);
    p.num_shared = 0;
    p.scale = scale;
    p.renorm_eps = 0.0f;
    p.renormalize = renormalize;
    p.apply_scale = apply_scale;

    auto s = StageParams{};
    s.x = static_cast<const bf16_t*>(x.data_ptr());
    s.x_q = static_cast<uint2*>(x_q.data_ptr());
    s.x_sf = static_cast<uint8_t*>(x_sf.data_ptr());
    s.ids = static_cast<int32_t*>(stage_ids.data_ptr());
    s.weights = static_cast<float*>(stage_weights.data_ptr());
    s.tail_ids = s.ids + num_tokens * K;
    s.x_stride = x.stride(0);
    s.num_blocks = static_cast<uint32_t>(hidden / kSfVec);
    s.tail_elems = static_cast<uint32_t>((capacity - num_tokens) * K);
    s.norm_const = norm_const;
    s.scale_multiplier = scale_multiplier;
    if (num_tokens == 0) return;

    const auto num_blocks = div_ceil(p.num_tokens, kNumWarps) + p.num_tokens;
    LaunchKernel(num_blocks, kBlockSize, device_sym.unwrap())
        .enable_pdl(kUsePDL)(route_warp_nvfp4_stage_kernel<E, K, T, TB, kUsePDL>, p, s);
  }
};

}  // namespace moe::route_warp_nvfp4_stage

}  // namespace sglang
