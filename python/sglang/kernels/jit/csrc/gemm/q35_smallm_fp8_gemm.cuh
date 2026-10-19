// Small-M FP8 projection GEMM for decode token counts (M <= 64): out[M, N] = (x[M, K] . w[N, K]^T) * alpha,
// e4m3 operands, fp32 accumulation, bf16 output, per-tensor alpha.
//
// Weights stream once: each CTA owns 64 weight rows (4 warps x m16) of one K split and every token (n8 tiles),
// with a cp.async ring of 256-byte K chunks. The weight chunks of the first stages are issued before the PDL
// wait, since the weights do not depend on the producer grid. With kSplit > 1 the last CTA to arrive on an
// N tile sums the fp32 partials in split order (deterministic) and writes the output.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace q35_smallm_fp8_gemm {

inline constexpr uint32_t kThreads = 128;
inline constexpr uint32_t kWarps = kThreads / 32;
inline constexpr uint32_t kRowsPerCta = kWarps * 16;  // weight rows (N) per CTA
inline constexpr uint32_t kChunk = 256;               // K bytes per pipeline stage
inline constexpr uint32_t kRowStride = kChunk + 16;   // padded smem row: ldmatrix rows hit distinct banks

template <uint32_t kNT, uint32_t kStages>
struct Smem {
  static constexpr uint32_t kW = kRowsPerCta * kRowStride;
  static constexpr uint32_t kX = kNT * 8 * kRowStride;
  static constexpr uint32_t kStage = kW + kX;
  static constexpr uint32_t kBytes = kStages * kStage;
};

SGL_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

SGL_DEVICE void cp_async_16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}

SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int kN>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kN) : "memory");
}

SGL_DEVICE void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

SGL_DEVICE void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

SGL_DEVICE void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Params {
  const uint8_t* x;    // [M, K] e4m3
  const uint8_t* w;    // [N, K] e4m3
  const float* alpha;  // [1]
  bf16_t* out;         // [M, N]
  float* part;         // [kSplit, kNT * 8, N] fp32 (kSplit > 1)
  uint32_t* count;     // [N / kRowsPerCta] arrival counters, zero between launches (kSplit > 1)
  uint32_t M, N, K;
};

template <uint32_t kNT, uint32_t kStages>
SGL_DEVICE void load_w(const Params& p, uint8_t* stage, uint32_t n0, uint32_t k0) {
  // 64 rows x 256 bytes = 1024 16-byte pieces, 8 per thread.
#pragma unroll
  for (uint32_t i = 0; i < kRowsPerCta * kChunk / 16 / kThreads; ++i) {
    const uint32_t piece = threadIdx.x + i * kThreads;
    const uint32_t row = piece / (kChunk / 16), col = piece % (kChunk / 16);
    cp_async_16(smem_u32(stage + row * kRowStride + col * 16), p.w + static_cast<size_t>(n0 + row) * p.K + k0 + col * 16,
                16);
  }
}

template <uint32_t kNT, uint32_t kStages>
SGL_DEVICE void load_x(const Params& p, uint8_t* stage, uint32_t k0) {
  using L = Smem<kNT, kStages>;
  uint8_t* sx = stage + L::kW;
  constexpr uint32_t kPieces = kNT * 8 * kChunk / 16;
#pragma unroll
  for (uint32_t i = 0; i < (kPieces + kThreads - 1) / kThreads; ++i) {
    const uint32_t piece = threadIdx.x + i * kThreads;
    if (piece < kPieces) {
      const uint32_t row = piece / (kChunk / 16), col = piece % (kChunk / 16);
      const bool valid = row < p.M;
      // Rows past M are zero-filled (source size 0); the address stays in bounds.
      cp_async_16(smem_u32(sx + row * kRowStride + col * 16),
                  p.x + static_cast<size_t>(valid ? row : 0) * p.K + k0 + col * 16, valid ? 16u : 0u);
    }
  }
}

template <uint32_t kNT, uint32_t kSplit, uint32_t kStages>
__global__ __launch_bounds__(kThreads, 1) void smallm_fp8_gemm_kernel(const __grid_constant__ Params p) {
  using L = Smem<kNT, kStages>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t n0 = blockIdx.x * kRowsPerCta;
  const uint32_t k_per_split = p.K / kSplit;
  const uint32_t kbase = blockIdx.y * k_per_split;
  const uint32_t chunks = k_per_split / kChunk;

  // Weights first (static), then wait for the producer of x.
  for (uint32_t s = 0; s + 1 < kStages; ++s) {
    if (s < chunks) load_w<kNT, kStages>(p, smem + s * L::kStage, n0, kbase + s * kChunk);
  }
  cp_async_commit();
  device::PDLWaitPrimary<true>();
  for (uint32_t s = 0; s + 1 < kStages; ++s) {
    if (s < chunks) load_x<kNT, kStages>(p, smem + s * L::kStage, kbase + s * kChunk);
    cp_async_commit();
  }

  float acc[kNT][4];
#pragma unroll
  for (uint32_t t = 0; t < kNT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;

  // ldmatrix lane addressing: A (16 x 32 bytes of this warp's rows), B (8 tokens x 32 bytes).
  const uint32_t a_row = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
  const uint32_t a_col = (lane >> 4) * 16;
  const uint32_t b_row = lane & 7;
  const uint32_t b_col = ((lane >> 3) & 1) * 16;

  for (uint32_t c = 0; c < chunks; ++c) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    const uint8_t* stage = smem + (c % kStages) * L::kStage;
    const uint32_t sw = smem_u32(stage) + a_row * kRowStride + a_col;
    const uint32_t sx = smem_u32(stage + L::kW) + b_row * kRowStride + b_col;
#pragma unroll
    for (uint32_t kk = 0; kk < kChunk; kk += 32) {
      uint32_t a[4];
      ldmatrix_x4(a, sw + kk);
#pragma unroll
      for (uint32_t t = 0; t < kNT; ++t) {
        uint32_t b[2];
        ldmatrix_x2(b, sx + t * 8 * kRowStride + kk);
        mma_e4m3(acc[t], a, b);
      }
    }
    const uint32_t next = c + kStages - 1;
    if (next < chunks) {
      uint8_t* ns = smem + (next % kStages) * L::kStage;
      load_w<kNT, kStages>(p, ns, n0, kbase + next * kChunk);
      load_x<kNT, kStages>(p, ns, kbase + next * kChunk);
    }
    cp_async_commit();
  }
  device::PDLTriggerSecondary<true>();

  // D fragment: d0 / d1 = (row g, tokens 2q, 2q + 1), d2 / d3 = row g + 8; rows are output columns n.
  const uint32_t g = lane >> 2, q = lane & 3;
  const uint32_t n_lo = n0 + warp * 16 + g, n_hi = n_lo + 8;
  if constexpr (kSplit == 1) {
    const float alpha = p.alpha[0];
#pragma unroll
    for (uint32_t t = 0; t < kNT; ++t) {
      const uint32_t m0 = t * 8 + 2 * q;
      if (m0 < p.M) {
        p.out[static_cast<size_t>(m0) * p.N + n_lo] = __float2bfloat16_rn(__fmul_rn(acc[t][0], alpha));
        p.out[static_cast<size_t>(m0) * p.N + n_hi] = __float2bfloat16_rn(__fmul_rn(acc[t][2], alpha));
      }
      if (m0 + 1 < p.M) {
        p.out[static_cast<size_t>(m0 + 1) * p.N + n_lo] = __float2bfloat16_rn(__fmul_rn(acc[t][1], alpha));
        p.out[static_cast<size_t>(m0 + 1) * p.N + n_hi] = __float2bfloat16_rn(__fmul_rn(acc[t][3], alpha));
      }
    }
  } else {
    constexpr uint32_t kMp = kNT * 8;
    float* part = p.part + static_cast<size_t>(blockIdx.y) * kMp * p.N;
#pragma unroll
    for (uint32_t t = 0; t < kNT; ++t) {
      const uint32_t m0 = t * 8 + 2 * q;
      part[static_cast<size_t>(m0) * p.N + n_lo] = acc[t][0];
      part[static_cast<size_t>(m0 + 1) * p.N + n_lo] = acc[t][1];
      part[static_cast<size_t>(m0) * p.N + n_hi] = acc[t][2];
      part[static_cast<size_t>(m0 + 1) * p.N + n_hi] = acc[t][3];
    }
    __shared__ uint32_t s_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t prev;
      asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;" : "=r"(prev) : "l"(p.count + blockIdx.x) : "memory");
      s_last = (prev == kSplit - 1) ? 1u : 0u;
      if (s_last) p.count[blockIdx.x] = 0;
    }
    __syncthreads();
    if (!s_last) return;
    const float alpha = p.alpha[0];
    // This CTA's 64 columns x M rows, summed over splits in order.
    for (uint32_t i = threadIdx.x; i < p.M * kRowsPerCta; i += kThreads) {
      const uint32_t m = i / kRowsPerCta, n = n0 + i % kRowsPerCta;
      float s = 0.0f;
#pragma unroll
      for (uint32_t k = 0; k < kSplit; ++k) {
        float v;
        asm volatile("ld.global.cg.f32 %0, [%1];"
                     : "=f"(v)
                     : "l"(p.part + (static_cast<size_t>(k) * kMp + m) * p.N + n));
        s = __fadd_rn(s, v);
      }
      p.out[static_cast<size_t>(m) * p.N + n] = __float2bfloat16_rn(__fmul_rn(s, alpha));
    }
  }
}

template <uint32_t kNT, uint32_t kSplit, uint32_t kStages>
struct SmallmFp8Gemm {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView alpha,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView part,
      const tvm::ffi::TensorView count) {
    using namespace host;
    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"n"};
    auto K_ = SymbolicSize{"k"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M_, K_}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(x);
    TensorMatcher({N_, K_}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(alpha);
    TensorMatcher({M_, N_}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto M = static_cast<uint32_t>(M_.unwrap()), N = static_cast<uint32_t>(N_.unwrap()),
               K = static_cast<uint32_t>(K_.unwrap());
    CHECK_HOST(M <= kNT * 8) << "smallm_fp8_gemm: M " << M << " exceeds " << kNT * 8;
    CHECK_HOST(N % kRowsPerCta == 0) << "smallm_fp8_gemm: N " << N;
    CHECK_HOST(K % (kChunk * kSplit) == 0) << "smallm_fp8_gemm: K " << K;
    if (M == 0) return;
    Params params{};
    params.x = static_cast<const uint8_t*>(x.data_ptr());
    params.w = static_cast<const uint8_t*>(w.data_ptr());
    params.alpha = static_cast<const float*>(alpha.data_ptr());
    params.out = static_cast<bf16_t*>(out.data_ptr());
    params.part = static_cast<float*>(part.data_ptr());
    params.count = static_cast<uint32_t*>(count.data_ptr());
    params.M = M;
    params.N = N;
    params.K = K;
    constexpr uint32_t kSmem = Smem<kNT, kStages>::kBytes;
    static const bool attr = [] {
      CHECK_CUDA(cudaFuncSetAttribute(smallm_fp8_gemm_kernel<kNT, kSplit, kStages>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      return true;
    }();
    (void)attr;
    LaunchKernel(dim3(N / kRowsPerCta, kSplit), kThreads, device.unwrap(), kSmem)
        .enable_pdl(true)(smallm_fp8_gemm_kernel<kNT, kSplit, kStages>, params);
  }
};

}  // namespace q35_smallm_fp8_gemm

using q35_smallm_fp8_gemm::SmallmFp8Gemm;

}  // namespace sglang
