// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// D-side KV record writer for the "nvfp4.fp8rope" layout read in place by FlashInfer 0.7.6 PrimTS sparse-MLA decode (352 B per token per layer):
//   bytes   0..255  E2M1 NoPE: 512 values in 32 blocks of 16, the even-indexed value of a pair in the LOW nibble (sign bit 3, magnitude code 0..7 = {0, .5, 1, 1.5, 2, 3, 4, 6})
//   bytes 256..319  RoPE as E4M3, unscaled (plain RN cast of the post-RoPE BF16 values)
//   bytes 320..351  one E4M3 scale per NoPE block b at 320 + 8*(b % 4) + b / 4; real scale = E4M3(amax(block) / 6), no global scale
// One warp per token (lane = NoPE block), one launch, no allocation, no host sync, CUDA-graph capturable. loc[t] == skip_index (or < 0) skips the token.
// kMode 0: the numerics of the torch reference quantizer fp8rope_records() (d576_micro_v4.py), bit-exact: scale = amax * fl(1/6) (torch divides by a CPU scalar as a multiply by its reciprocal),
//          q = x / scale (IEEE division, 0 when the E4M3 scale is 0), nearest E2M1 magnitude with ties to the LOWER code.
// kMode 2: kMode 0 with the scale computed as a true IEEE division amax / 6 (in case torch's scalar division is a true division on the reference device).
// kMode 3 (default of the python wrapper): bit-identical to kMode 2 and ~1 us faster: q = v * rcp.approx(scale) feeds the hardware E2M1 convert of |q| (sign = v < 0), and only the elements whose q lies within a
//          few 1e-6 of an E2M1 midpoint (where the approximation error could flip the rounding, and where ties go to the lower code) recompute q = IEEE v / scale and the reference rule.
// kMode 1: TensorRT-LLM style instructions: scale = amax * rcp.approx(6), outScale = amax != 0 ? rcp.approx(scale_e4m3) : 0, hardware cvt.rn.satfinite.e2m1x2 (ties to even).

#pragma once

#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize, SymbolicDType, SymbolicDevice
#include <sgl_kernel/utils.h>   // RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // LaunchKernel, SGL_DEVICE, bf16_t

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

namespace sglang::nvfp4_fp8rope_d {

inline constexpr int kNopeDim = 512;
inline constexpr int kRopeDim = 64;
inline constexpr int kBlock = 16;
inline constexpr int kRecBytes = 352;
inline constexpr int kRopeOff = 256;
inline constexpr int kScaleOff = 320;
inline constexpr int kWarpsPerBlock = 4;
inline constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
inline constexpr float kInvSix = 1.0f / 6.0f;  // float(1/6): what torch's  amax / 6.0  multiplies by

struct WriterParams {
  const bf16_t* __restrict__ k_nope;  // [T, 512] (row stride k_nope_stride_t elements)
  const bf16_t* __restrict__ k_rope;  // [T, 64]  post-RoPE
  const void* __restrict__ loc;       // [T], int32 or int64: record index of the token in dst
  uint8_t* __restrict__ dst;          // [rows, 352] contiguous (one layer's pool)
  int64_t k_nope_stride_t;
  int64_t k_rope_stride_t;
  int64_t skip_index;
  uint32_t num_tokens;
};

SGL_DEVICE float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
SGL_DEVICE float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

SGL_DEVICE float e4m3_to_f32(__nv_fp8_storage_t s) {
  const __half_raw h = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

SGL_DEVICE float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Eight FP32 values -> eight E2M1 (element 0 in the low nibble of byte 0), RN, saturating (TensorRT-LLM fp32_vec_to_e2m1).
SGL_DEVICE uint32_t f32x8_to_e2m1(const float (&v)[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return val;
}

// kMode 0: nearest E2M1 magnitude, ties to the lower code (= argmin |m - LUT| of the torch reference); sign from the scaled value (-0 stays sign-bit-free: x < 0 is false).
SGL_DEVICE uint32_t e2m1_code_ref(float x) {
  const float m = fminf(fabsf(x), 6.0f);
  const uint32_t idx = (m > 0.25f) + (m > 0.75f) + (m > 1.25f) + (m > 1.75f) + (m > 2.5f) + (m > 3.5f) + (m > 5.0f);
  return idx | (x < 0.0f ? 8u : 0u);
}

template <typename LocT, int kMode>
__global__ void __launch_bounds__(kThreadsPerBlock) kernel(const __grid_constant__ WriterParams p) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t t = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (t >= p.num_tokens) return;
  const int64_t slot = static_cast<int64_t>(static_cast<const LocT*>(p.loc)[t]);
  if (slot < 0 || slot == p.skip_index) return;
  uint8_t* __restrict__ rec = p.dst + slot * kRecBytes;

  // NoPE block `lane`: 16 BF16 = two 16-byte loads
  const uint4* src = reinterpret_cast<const uint4*>(p.k_nope + static_cast<int64_t>(t) * p.k_nope_stride_t + lane * kBlock);
  const uint4 a = src[0];
  const uint4 b = src[1];
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float v[16];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[2 * i] = bf16_lo(w[i]);
    v[2 * i + 1] = bf16_hi(w[i]);
    amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
  }

  uint32_t packed[2];
  __nv_fp8_storage_t s8;
  if constexpr (kMode == 3) {
    s8 = __nv_cvt_float_to_fp8(__fdiv_rn(amax, 6.0f), __NV_SATFINITE, __NV_E4M3);
    const float sf = e4m3_to_f32(s8);
    if (sf > 0.0f) {
      const float rs = rcp_approx_ftz(sf);
      float lo[8], hi[8];
      uint32_t sg0 = 0, sg1 = 0;
      uint32_t near_lo = 0, near_hi = 0;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float q = fabsf(v[i] * rs);
        // distance of q to the nearest E2M1 midpoint {.25,.75,1.25,1.75 | 2.5,3.5 | 5} in units where the midpoints of its region are integers
        const float u = q < 2.0f ? q * 2.0f - 0.5f : (q < 4.0f ? q - 2.5f : (q - 5.0f));
        const float eps = q < 2.0f ? 8e-6f : (q < 4.0f ? 8e-6f : 1.6e-5f);
        const float du = fabsf(u - rintf(u));
        const bool is_near = (q < 4.0f ? du < eps : du < eps) && q < 5.5f && q > 0.2f;
        (i < 8 ? lo : hi)[i & 7] = q;
        const uint32_t sgn = v[i] < 0.0f ? 8u : 0u;
        if (i < 8) { sg0 |= sgn << (4 * i); near_lo |= (is_near ? 1u : 0u) << i; } else { sg1 |= sgn << (4 * (i - 8)); near_hi |= (is_near ? 1u : 0u) << (i - 8); }
      }
      packed[0] = f32x8_to_e2m1(lo) | sg0;
      packed[1] = f32x8_to_e2m1(hi) | sg1;
      uint32_t nr = near_lo | (near_hi << 8);
      while (nr) {   // rare: exact reference rule for the elements next to a midpoint
        const int i = __ffs(nr) - 1;
        nr &= nr - 1;
        const uint32_t code = e2m1_code_ref(__fdiv_rn(v[i], sf));
        const int sh = 4 * (i & 7);
        uint32_t& wd = packed[i >> 3];
        wd = (wd & ~(0xFu << sh)) | (code << sh);
      }
    } else {
      packed[0] = 0;
      packed[1] = 0;
    }
  } else if constexpr (kMode == 0 || kMode == 2) {
    s8 = __nv_cvt_float_to_fp8(kMode == 0 ? amax * kInvSix : __fdiv_rn(amax, 6.0f), __NV_SATFINITE, __NV_E4M3);
    const float sf = e4m3_to_f32(s8);
    uint32_t c[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) c[i] = e2m1_code_ref(sf > 0.0f ? __fdiv_rn(v[i], sf) : 0.0f);
    packed[0] = 0;
    packed[1] = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t byte = c[2 * j] | (c[2 * j + 1] << 4);
      packed[j >> 2] |= byte << (8 * (j & 3));
    }
  } else {
    s8 = __nv_cvt_float_to_fp8(amax * rcp_approx_ftz(6.0f), __NV_SATFINITE, __NV_E4M3);
    const float out_scale = amax != 0.0f ? rcp_approx_ftz(e4m3_to_f32(s8)) : 0.0f;
    float lo[8], hi[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      lo[i] = v[i] * out_scale;
      hi[i] = v[8 + i] * out_scale;
    }
    packed[0] = f32x8_to_e2m1(lo);
    packed[1] = f32x8_to_e2m1(hi);
  }
  *reinterpret_cast<uint2*>(rec + 8 * lane) = make_uint2(packed[0], packed[1]);
  rec[kScaleOff + 8 * (lane & 3) + (lane >> 2)] = static_cast<uint8_t>(s8);

  // RoPE: two BF16 per lane -> two E4M3 (unscaled RN cast)
  const uint32_t rw = *reinterpret_cast<const uint32_t*>(p.k_rope + static_cast<int64_t>(t) * p.k_rope_stride_t + 2 * lane);
  const uint32_t r0 = __nv_cvt_float_to_fp8(bf16_lo(rw), __NV_SATFINITE, __NV_E4M3);
  const uint32_t r1 = __nv_cvt_float_to_fp8(bf16_hi(rw), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint16_t*>(rec + kRopeOff + 2 * lane) = static_cast<uint16_t>(r0 | (r1 << 8));
}

// kMode 4 ("split"): two lanes per NoPE block (64 threads per token, 128-thread blocks = 2 tokens), 8 elements per lane = one 16-byte load, amax joined with one shuffle, one 4-byte packed store per lane; the near-midpoint test
// is an integer one: every E2M1 midpoint (and code) has its low 20 float-mantissa bits zero, so |q| is "near" when ((bits + 128) & 0xFFFFF) < 256 (+-128 ulp = 1.5e-5 relative, far above the ~3e-7 rcp error); near elements
// recompute IEEE q = v / scale and the reference rule. Bit-identical to kMode 2 / 3.
inline constexpr int kSplitThreads = 128;
inline constexpr int kSplitTokensPerBlock = 2;

template <typename LocT, bool kRecipScale>
__global__ void __launch_bounds__(kSplitThreads) kernel_split(const __grid_constant__ WriterParams p) {
  const uint32_t idx = threadIdx.x & 63;
  const uint32_t t = blockIdx.x * kSplitTokensPerBlock + (threadIdx.x >> 6);
  if (t >= p.num_tokens) return;
  const int64_t slot = static_cast<int64_t>(static_cast<const LocT*>(p.loc)[t]);
  if (slot < 0 || slot == p.skip_index) return;
  uint8_t* __restrict__ rec = p.dst + slot * kRecBytes;
  const uint32_t blk = idx >> 1;
  const uint32_t half = idx & 1;

  const uint4 a = *reinterpret_cast<const uint4*>(p.k_nope + static_cast<int64_t>(t) * p.k_nope_stride_t + blk * kBlock + half * 8);
  const uint32_t w[4] = {a.x, a.y, a.z, a.w};
  float v[8];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[2 * i] = bf16_lo(w[i]);
    v[2 * i + 1] = bf16_hi(w[i]);
    amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(kRecipScale ? amax * kInvSix : __fdiv_rn(amax, 6.0f), __NV_SATFINITE, __NV_E4M3);
  const float sf = e4m3_to_f32(s8);
  uint32_t packed = 0;
  if (sf > 0.0f) {
    const float rs = rcp_approx_ftz(sf);
    float q[8];
    uint32_t sg = 0, nr = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q[i] = fabsf(v[i] * rs);
      sg |= (v[i] < 0.0f ? 8u : 0u) << (4 * i);
      nr |= (((__float_as_uint(q[i]) + 128u) & 0xFFFFFu) < 256u ? 1u : 0u) << i;
    }
    packed = f32x8_to_e2m1(q) | sg;
    while (nr) {
      const int i = __ffs(nr) - 1;
      nr &= nr - 1;
      const uint32_t code = e2m1_code_ref(__fdiv_rn(v[i], sf));
      const int sh = 4 * i;
      packed = (packed & ~(0xFu << sh)) | (code << sh);
    }
  }
  *reinterpret_cast<uint32_t*>(rec + 8 * blk + 4 * half) = packed;
  if (half == 0) rec[kScaleOff + 8 * (blk & 3) + (blk >> 2)] = static_cast<uint8_t>(s8);

  const float r = __uint_as_float(static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(p.k_rope + static_cast<int64_t>(t) * p.k_rope_stride_t + idx)) << 16);
  rec[kRopeOff + idx] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(r, __NV_SATFINITE, __NV_E4M3));
}

template <int kMode>
struct WriterKernel {
  static void run(
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView loc,
      const tvm::ffi::TensorView k_nope,
      const tvm::ffi::TensorView k_rope,
      int64_t skip_index) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto D_nope = SymbolicSize{"nope_dim"};
    auto D_rope = SymbolicSize{"rope_dim"};
    auto D_rec = SymbolicSize{"record_bytes"};
    auto S_kn_t = SymbolicSize{"k_nope_stride_t"};
    auto S_kr_t = SymbolicSize{"k_rope_stride_t"};
    auto loc_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    D_nope.set_value(kNopeDim);
    D_rope.set_value(kRopeDim);
    D_rec.set_value(kRecBytes);

    TensorMatcher({T, D_nope}).with_strides({S_kn_t, 1}).with_dtype<bf16_t>().with_device(device).verify(k_nope);
    TensorMatcher({T, D_rope}).with_strides({S_kr_t, 1}).with_dtype<bf16_t>().with_device(device).verify(k_rope);
    TensorMatcher({T}).with_strides({1}).with_dtype<int32_t, int64_t>(loc_dtype).with_device(device).verify(loc);
    TensorMatcher({-1, D_rec}).with_strides({D_rec, 1}).with_dtype<uint8_t>().with_device(device).verify(dst);

    const auto aligned = [](const void* ptr, int64_t align) {
      return reinterpret_cast<uintptr_t>(ptr) % static_cast<uintptr_t>(align) == 0;
    };
    RuntimeCheck(aligned(dst.data_ptr(), 16), "record pool base must be 16-byte aligned");
    RuntimeCheck(
        aligned(k_nope.data_ptr(), 16) && (S_kn_t.unwrap() * 2) % 16 == 0, "k_nope base / row stride must be 16-byte aligned");
    RuntimeCheck(
        aligned(k_rope.data_ptr(), 4) && (S_kr_t.unwrap() * 2) % 4 == 0, "k_rope base / row stride must be 4-byte aligned");

    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    if (num_tokens == 0) return;
    const auto params = WriterParams{
        .k_nope = static_cast<const bf16_t*>(k_nope.data_ptr()),
        .k_rope = static_cast<const bf16_t*>(k_rope.data_ptr()),
        .loc = loc.data_ptr(),
        .dst = static_cast<uint8_t*>(dst.data_ptr()),
        .k_nope_stride_t = S_kn_t.unwrap(),
        .k_rope_stride_t = S_kr_t.unwrap(),
        .skip_index = skip_index,
        .num_tokens = num_tokens,
    };
    if constexpr (kMode == 4 || kMode == 5) {
      const auto kernel_ptr = loc_dtype.is_type<int32_t>() ? kernel_split<int32_t, kMode == 5> : kernel_split<int64_t, kMode == 5>;
      LaunchKernel(div_ceil(num_tokens, static_cast<uint32_t>(kSplitTokensPerBlock)), kSplitThreads, device.unwrap())(kernel_ptr, params);
    } else {
      const auto kernel_ptr = loc_dtype.is_type<int32_t>() ? kernel<int32_t, kMode> : kernel<int64_t, kMode>;
      const uint32_t num_blocks = div_ceil(num_tokens, static_cast<uint32_t>(kWarpsPerBlock));
      LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())(kernel_ptr, params);
    }
  }
};

}  // namespace sglang::nvfp4_fp8rope_d
