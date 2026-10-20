// MoE exit of a DEP decode layer in one launch: the gated shared-expert add (gate precomputed by the Triton
// GATE_ONLY launch), the residual add, Gemma RMSNorm (weight + 1) and the static per-tensor FP8 quant of the normed
// row. It reproduces fused_gate_sigmoid_mul_add + flashinfer gemma_fused_add_rmsnorm + static_quant_fp8 element by
// element; the RMS sum follows FlashInfer's CuTe-DSL kernel for hidden 4096 (64 threads per row, thread t holding
// columns 8t + 512b + v, in-thread sum, butterfly over the warp, then the row's two warps added).
// `variant` selects among candidate operation orders for the single-GPU equivalence search (T1).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace qwen_dep_moe_exit {

inline constexpr uint32_t kHidden = 4096;
inline constexpr uint32_t kThreadsPerRow = 64;
inline constexpr uint32_t kRowsPerCta = 2;
inline constexpr uint32_t kThreads = kThreadsPerRow * kRowsPerCta;
inline constexpr uint32_t kVec = 8;
inline constexpr uint32_t kBlocks = kHidden / (kVec * kThreadsPerRow);  // 8

// variant bits
inline constexpr uint32_t kGateMulAdd = 1u;    // gated add as mul then add (else one FMA)
inline constexpr uint32_t kOrderVOuter = 2u;   // in-thread sum over v outer, b inner (else b outer)
inline constexpr uint32_t kSquareFma = 4u;     // in-thread sum as fma(h, h, acc) (else acc + h * h)
inline constexpr uint32_t kInvApprox = 8u;     // 1 / scale as div.full.f32 (else div.rn)
inline constexpr uint32_t kTreeSum = 16u;      // in-thread sum as a halving tree over v + 8b

struct Params {
  const bf16_t* routed;    // [M, kHidden] combine output
  const bf16_t* shared;    // [M, kHidden] shared expert output
  const float* gate;       // [M] sigmoid gate
  bf16_t* residual;        // [M, kHidden] in / out
  bf16_t* out;             // [M, kHidden] normed output (may alias routed)
  uint8_t* out_fp8;        // [M, kHidden] e4m3
  const bf16_t* weight;    // [kHidden] Gemma norm weight
  const float* scale;      // [1] static activation scale of the consuming projection
  bf16_t* dbg_final;       // optional [M, kHidden] gated sum before the residual add
  float eps;
  uint32_t num_tokens;
  uint32_t variant;
};

SGL_DEVICE float bf_lo(uint32_t v) {
  return __uint_as_float(v << 16);
}
SGL_DEVICE float bf_hi(uint32_t v) {
  return __uint_as_float(v & 0xFFFF0000u);
}
SGL_DEVICE uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
SGL_DEVICE float rsqrt_approx(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
SGL_DEVICE float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
SGL_DEVICE uint32_t cvt_e4m3x2(float lo, float hi) {
  uint32_t out;
  asm("{\n"
      ".reg .b16 pair;\n"
      "cvt.rn.satfinite.e4m3x2.f32 pair, %2, %1;\n"
      "cvt.u32.u16 %0, pair;\n"
      "}"
      : "=r"(out)
      : "f"(lo), "f"(hi));
  return out;
}

template <bool kUsePDL>
__global__ __launch_bounds__(kThreads) void moe_exit_kernel(const __grid_constant__ Params p) {
  __shared__ float s_warp[kRowsPerCta][2];
  const uint32_t t = threadIdx.x % kThreadsPerRow;
  const uint32_t rib = threadIdx.x / kThreadsPerRow;
  const uint32_t row = blockIdx.x * kRowsPerCta + rib;
  const bool live = row < p.num_tokens;
  const uint32_t variant = p.variant;

  // Static weight first, then wait for the producer grid.
  uint4 w4[kBlocks];
#pragma unroll
  for (uint32_t b = 0; b < kBlocks; ++b) {
    w4[b] = *reinterpret_cast<const uint4*>(p.weight + b * kVec * kThreadsPerRow + t * kVec);
  }
  device::PDLWaitPrimary<kUsePDL>();

  float h[kBlocks][kVec];
  const float g = live ? p.gate[row] : 0.0f;
#pragma unroll
  for (uint32_t b = 0; b < kBlocks; ++b) {
    const size_t off = static_cast<size_t>(row) * kHidden + b * kVec * kThreadsPerRow + t * kVec;
    uint4 f4 = make_uint4(0, 0, 0, 0), s4 = make_uint4(0, 0, 0, 0), r4 = make_uint4(0, 0, 0, 0);
    if (live) {
      f4 = *reinterpret_cast<const uint4*>(p.routed + off);
      s4 = *reinterpret_cast<const uint4*>(p.shared + off);
      r4 = *reinterpret_cast<const uint4*>(p.residual + off);
    }
    const uint32_t fw[4] = {f4.x, f4.y, f4.z, f4.w};
    const uint32_t sw[4] = {s4.x, s4.y, s4.z, s4.w};
    const uint32_t rw[4] = {r4.x, r4.y, r4.z, r4.w};
    uint32_t res_out[4], fin_out[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      float fin[2];
      const float fv[2] = {bf_lo(fw[i]), bf_hi(fw[i])};
      const float sv[2] = {bf_lo(sw[i]), bf_hi(sw[i])};
      const float rv[2] = {bf_lo(rw[i]), bf_hi(rw[i])};
#pragma unroll
      for (uint32_t k = 0; k < 2; ++k) {
        fin[k] = (variant & kGateMulAdd) ? __fadd_rn(__fmul_rn(g, sv[k]), fv[k]) : __fmaf_rn(g, sv[k], fv[k]);
      }
      fin_out[i] = pack_bf16(fin[0], fin[1]);
      const float x0 = bf_lo(fin_out[i]), x1 = bf_hi(fin_out[i]);
      h[b][2 * i] = __fadd_rn(x0, rv[0]);
      h[b][2 * i + 1] = __fadd_rn(x1, rv[1]);
      res_out[i] = pack_bf16(h[b][2 * i], h[b][2 * i + 1]);
    }
    if (live) {
      *reinterpret_cast<uint4*>(p.residual + off) = make_uint4(res_out[0], res_out[1], res_out[2], res_out[3]);
      if (p.dbg_final != nullptr) {
        *reinterpret_cast<uint4*>(p.dbg_final + off) = make_uint4(fin_out[0], fin_out[1], fin_out[2], fin_out[3]);
      }
    }
  }

  // In-thread sum of squares in the candidate order.
  float acc = 0.0f;
  if (variant & kTreeSum) {
    float q[kBlocks * kVec];
#pragma unroll
    for (uint32_t b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (uint32_t v = 0; v < kVec; ++v) q[v + kVec * b] = __fmul_rn(h[b][v], h[b][v]);
    }
#pragma unroll
    for (uint32_t n = kBlocks * kVec / 2; n >= 1; n /= 2) {
#pragma unroll
      for (uint32_t i = 0; i < n; ++i) q[i] = __fadd_rn(q[i], q[i + n]);
    }
    acc = __fadd_rn(0.0f, q[0]);
  } else if (variant & kOrderVOuter) {
#pragma unroll
    for (uint32_t v = 0; v < kVec; ++v) {
#pragma unroll
      for (uint32_t b = 0; b < kBlocks; ++b) {
        acc = (variant & kSquareFma) ? __fmaf_rn(h[b][v], h[b][v], acc) : __fadd_rn(acc, __fmul_rn(h[b][v], h[b][v]));
      }
    }
  } else {
#pragma unroll
    for (uint32_t b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (uint32_t v = 0; v < kVec; ++v) {
        acc = (variant & kSquareFma) ? __fmaf_rn(h[b][v], h[b][v], acc) : __fadd_rn(acc, __fmul_rn(h[b][v], h[b][v]));
      }
    }
  }
  // Butterfly over the warp (offsets 1, 2, 4, 8, 16), then the row's two warps.
#pragma unroll
  for (uint32_t o = 1; o < 32; o <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  const uint32_t warp_in_row = t / 32;
  if ((threadIdx.x % 32) == 0) s_warp[rib][warp_in_row] = acc;
  __syncthreads();
  const float total = __fadd_rn(s_warp[rib][0], s_warp[rib][1]);
  const float mean = __fdiv_rn(total, static_cast<float>(kHidden));
  const float rstd = rsqrt_approx(__fadd_rn(mean, p.eps));
  const float inv = (variant & kInvApprox) ? div_full(1.0f, p.scale[0]) : __fdiv_rn(1.0f, p.scale[0]);

#pragma unroll
  for (uint32_t b = 0; b < kBlocks; ++b) {
    const size_t off = static_cast<size_t>(row) * kHidden + b * kVec * kThreadsPerRow + t * kVec;
    const uint32_t ww[4] = {w4[b].x, w4[b].y, w4[b].z, w4[b].w};
    uint32_t y_out[4];
    uint32_t q_lo = 0, q_hi = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const float w0 = __fadd_rn(bf_lo(ww[i]), 1.0f), w1 = __fadd_rn(bf_hi(ww[i]), 1.0f);
      const float y0 = __fmul_rn(__fmul_rn(h[b][2 * i], rstd), w0);
      const float y1 = __fmul_rn(__fmul_rn(h[b][2 * i + 1], rstd), w1);
      y_out[i] = pack_bf16(y0, y1);
      const float q0 = fminf(fmaxf(__fmul_rn(bf_lo(y_out[i]), inv), -448.0f), 448.0f);
      const float q1 = fminf(fmaxf(__fmul_rn(bf_hi(y_out[i]), inv), -448.0f), 448.0f);
      const uint32_t pair = cvt_e4m3x2(q0, q1);
      if (i < 2) {
        q_lo |= pair << (16 * i);
      } else {
        q_hi |= pair << (16 * (i - 2));
      }
    }
    if (live) {
      *reinterpret_cast<uint4*>(p.out + off) = make_uint4(y_out[0], y_out[1], y_out[2], y_out[3]);
      *reinterpret_cast<uint2*>(p.out_fp8 + off) = make_uint2(q_lo, q_hi);
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct MoeExit {
  static void
  run(const tvm::ffi::TensorView routed,
      const tvm::ffi::TensorView shared,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView out_fp8,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView scale,
      const tvm::ffi::TensorView dbg_final,
      double eps,
      int64_t variant) {
    using namespace host;
    auto M_ = SymbolicSize{"num_tokens"};
    auto D_ = SymbolicSize{"debug_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M_, kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(routed);
    TensorMatcher({M_, kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(shared);
    TensorMatcher({M_}).with_dtype<fp32_t>().with_device(device).verify(gate);
    TensorMatcher({M_, kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(residual);
    TensorMatcher({M_, kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out);
    TensorMatcher({M_, kHidden}).with_dtype<uint8_t>().with_device(device).ensure_alignment(8).verify(out_fp8);
    TensorMatcher({kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(weight);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(scale);
    TensorMatcher({D_, kHidden}).with_dtype<bf16_t>().with_device(device).verify(dbg_final);
    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;
    Params params{};
    params.routed = static_cast<const bf16_t*>(routed.data_ptr());
    params.shared = static_cast<const bf16_t*>(shared.data_ptr());
    params.gate = static_cast<const float*>(gate.data_ptr());
    params.residual = static_cast<bf16_t*>(residual.data_ptr());
    params.out = static_cast<bf16_t*>(out.data_ptr());
    params.out_fp8 = static_cast<uint8_t*>(out_fp8.data_ptr());
    params.weight = static_cast<const bf16_t*>(weight.data_ptr());
    params.scale = static_cast<const float*>(scale.data_ptr());
    params.dbg_final = D_.unwrap() == M_.unwrap() ? static_cast<bf16_t*>(dbg_final.data_ptr()) : nullptr;
    params.eps = static_cast<float>(eps);
    params.num_tokens = M;
    params.variant = static_cast<uint32_t>(variant);
    LaunchKernel((M + kRowsPerCta - 1) / kRowsPerCta, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(moe_exit_kernel<kUsePDL>, params);
  }
};

}  // namespace qwen_dep_moe_exit

using qwen_dep_moe_exit::MoeExit;

}  // namespace sglang
