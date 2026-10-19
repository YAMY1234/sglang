// SPDX-License-Identifier: Apache-2.0
/*
# SPDX-License-Identifier: Apache-2.0
# Modified by NVIDIA: add an optional forward substitution for verify deltas.

Ported from the vLLM ReplaySSM reference implementation
(github.com/Johnny-Liou/ReplaySSM, commit ``3c85112``,
``vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py``) and the Dao
AI Lab ReplaySSM blog (dao-lab.ai/blog/2026/replayssm). Part B of SGLang RFC
#28511.
*/
// Modified by NVIDIA: provide a cooperative CUDA device stage for the
// compensated BF16 checkpoint path, retaining the triangular inverse and ring
// updates. Matrix and reduction scheduling differ from the Triton implementation;
// numerical equivalence must be checked rather than inferred from this port.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace gdn_stage {

#ifndef GDN_REPLAY_STAGE_FN
#define GDN_REPLAY_STAGE_FN __forceinline__
#endif

using BFloat16 = __nv_bfloat16;

enum class ReplayStorage : int {
  kNoResidual = 0,
  kCompensated = 1,
  kExactRaw = 2,
};

// Activation, checkpoint, and compact-ring storage are BF16. The FP32-checkpoint
// path has different dot operands and deliberately does not use this interface.
// Indices and strides are in elements, not bytes. K == V == 128, ring length 16,
// and packed sequence lengths in [1, 8] are the caller's shape contract.
struct ReplayParams {
  const BFloat16* q = nullptr;
  const BFloat16* k = nullptr;
  const BFloat16* v = nullptr;
  const void* a = nullptr;
  const void* b = nullptr;
  const float* a_log = nullptr;
  const void* dt_bias = nullptr;
  BFloat16* out = nullptr;
  const BFloat16* checkpoint = nullptr;
  BFloat16* d_cache = nullptr;
  BFloat16* k_cache = nullptr;
  float* g_cache = nullptr;
  BFloat16* rawv_cache = nullptr;
  BFloat16* rawk_cache = nullptr;
  float* beta_cache = nullptr;
  const int32_t* query_start_loc = nullptr;
  const void* state_indices = nullptr;
  const void* replay_indices = nullptr;
  const int32_t* write_pos = nullptr;
  const int32_t* cache_base = nullptr;
  const int8_t* is_flush = nullptr;

  int h = 2;
  int hv = 8;
  int null_block_id = -1;
  bool gates_fp32 = false;
  bool dt_bias_fp32 = true;
  bool state_indices_int64 = false;
  bool replay_indices_int64 = false;
  ReplayStorage storage = ReplayStorage::kCompensated;
  float scale = 0.08838834764831844055f;
  int64_t stride_q_t = 256;
  int64_t stride_k_t = 256;
  int64_t stride_v_t = 1024;
  int64_t stride_a_t = 8;
  int64_t stride_b_t = 8;
  int64_t stride_o_t = 1024;
  int64_t stride_state_slot = 8 * 128 * 128;
  int64_t stride_d_slot = 8 * 16 * 128;
  int64_t stride_k_slot = 2 * 16 * 128;
  int64_t stride_g_slot = 8 * 16;
  int64_t stride_rawv_slot = 8 * 16 * 128;
  int64_t stride_rawk_slot = 2 * 16 * 128;
  int64_t stride_beta_slot = 8 * 16;
  int64_t stride_qsl = 1;
  int64_t stride_state_indices = 1;
  int64_t stride_replay_indices = 1;
};

// Scratch belongs to one (request, value head, V tile). It may be aliased with
// another stage's scratch only after every participating thread has finished.
// Padding to 16 token columns is WMMA staging, not extra logical draft tokens.
template <int BV = 32>
struct alignas(32) ReplayScratch {
  static_assert(BV == 16 || BV == 32, "supported V tile sizes are 16 and 32");
  alignas(32) BFloat16 q_hi[16 * 128];
  alignas(32) BFloat16 k_hi[16 * 128];
  alignas(32) BFloat16 k_lo[16 * 128];
  alignas(32) BFloat16 state_tile[BV * 64];
  alignas(32) BFloat16 history_k[16 * 64];
  alignas(32) BFloat16 history_d[BV * 16];
  alignas(32) float scores_q[16 * 16];
  alignas(32) float scores_k[16 * 16];
  alignas(32) float checkpoint_q[BV * 16];
  alignas(32) float checkpoint_k[BV * 16];
  alignas(32) float kk[16 * 16];
  alignas(32) float kq[16 * 16];
  alignas(32) float product[BV * 16];
  alignas(32) float delta[BV * 8];
  float inverse[8 * 8];
  float gates[8];
  float beta[8];
  float prefix[8];
  float exp_prefix[8];
  float q_rnorm[8];
  float k_rnorm[8];
  float history_decay[16];
  float history_total_decay;
};

namespace detail {

template <int Threads, int BarrierId>
__device__ __forceinline__ void group_sync() {
  static_assert(Threads == 64 || Threads == 128,
                "the stage uses two or four complete consumer warps");
  static_assert(BarrierId >= 1 && BarrierId < 16,
                "reserve a named barrier distinct from the CTA barrier");
  asm volatile("bar.sync %0, %1;" : : "r"(BarrierId), "n"(Threads) : "memory");
}

__device__ __forceinline__ int64_t load_index(const void* ptr, int64_t offset,
                                             bool is_int64) {
  return is_int64 ? static_cast<const int64_t*>(ptr)[offset]
                  : static_cast<const int32_t*>(ptr)[offset];
}

__device__ __forceinline__ float load_gate(const void* ptr, int64_t offset,
                                          bool is_fp32) {
  return is_fp32 ? static_cast<const float*>(ptr)[offset]
                 : __bfloat162float(static_cast<const BFloat16*>(ptr)[offset]);
}

__device__ __forceinline__ float exp_f32(float x) {
  const float exponent = __fmul_rn(x, __uint_as_float(0x3fb8aa3bu));
  float result;
  asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(exponent));
  return result;
}

__device__ __forceinline__ float sqrt_f32(float x) {
  float result;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
  return result;
}

__device__ __forceinline__ float div_f32(float numerator, float denominator) {
  float result;
  asm("div.full.f32 %0, %1, %2;"
      : "=f"(result) : "f"(numerator), "f"(denominator));
  return result;
}

// Eight adjacent features per lane, with eight lanes cooperating on a token.
// The multiplication and addition tree is explicit because its rounding is
// visible in the normalized-key low part retained by the checkpoint protocol.
__device__ __forceinline__ float norm_chunk(const BFloat16* values, int64_t offset) {
  const float x0 = __bfloat162float(values[offset + 0]);
  const float x1 = __bfloat162float(values[offset + 1]);
  const float x2 = __bfloat162float(values[offset + 2]);
  const float x3 = __bfloat162float(values[offset + 3]);
  const float x4 = __bfloat162float(values[offset + 4]);
  const float x5 = __bfloat162float(values[offset + 5]);
  const float x6 = __bfloat162float(values[offset + 6]);
  const float x7 = __bfloat162float(values[offset + 7]);
  const float even0 = fmaf(x0, x0, __fmul_rn(x2, x2));
  const float odd0 = fmaf(x1, x1, __fmul_rn(x3, x3));
  const float even1 = fmaf(x4, x4, __fmul_rn(x6, x6));
  const float odd1 = fmaf(x5, x5, __fmul_rn(x7, x7));
  float value = __fadd_rn(__fadd_rn(even0, even1), __fadd_rn(odd0, odd1));
  const unsigned int members = 0xffu << ((threadIdx.x & 31) & ~7);
#pragma unroll
  for (int delta = 4; delta; delta >>= 1) {
    value = __fadd_rn(value, __shfl_xor_sync(members, value, delta, 8));
  }
  return value;
}

__device__ __forceinline__ float sum8(float a0, float a1, float a2, float a3,
                                     float a4, float a5, float a6, float a7) {
  const float s0 = __fadd_rn(a0, a1);
  const float s1 = __fadd_rn(a2, a3);
  const float s2 = __fadd_rn(a4, a5);
  const float s3 = __fadd_rn(a6, a7);
  return __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
}

// A is row-major. B is either a transposed token-major tile (column-major)
// or the row-major historical-score tile. Each warp owns 16 output rows.
// Continuing projections retain their FP32 accumulator across K chunks and
// residual products. Standalone products can start with a zero accumulator.
template <int Rows, int Inner, bool BColumnMajor, bool Accumulate = false>
__device__ __forceinline__ void matrix_product(
    const BFloat16* a, int lda, const BFloat16* b, int ldb, float* dst,
    int local_tid) {
  namespace wmma = nvcuda::wmma;
  const int warp = local_tid >> 5;
  if (warp * 16 < Rows) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, BFloat16, wmma::row_major> af;
    using BLayout = typename std::conditional<
        BColumnMajor, wmma::col_major, wmma::row_major>::type;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, BFloat16, BLayout> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    if constexpr (Accumulate) {
      wmma::load_matrix_sync(cf, dst + warp * 16 * 16, 16, wmma::mem_row_major);
    } else {
      wmma::fill_fragment(cf, 0.0f);
    }
#pragma unroll
    for (int inner = 0; inner < Inner; inner += 16) {
      wmma::load_matrix_sync(af, a + warp * 16 * lda + inner, lda);
      const BFloat16* bp = BColumnMajor ? b + inner : b + inner * ldb;
      wmma::load_matrix_sync(bf, bp, ldb);
      wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(dst + warp * 16 * 16, cf, 16, wmma::mem_row_major);
  }
}

template <int Rows, int Threads, int BarrierId>
__device__ __forceinline__ void add_projection(
    const BFloat16* a, int lda, const BFloat16* b, int ldb, float* accum,
    float* product, int local_tid) {
  matrix_product<Rows, 64, true, true>(a, lda, b, ldb, accum, local_tid);
  const int warp = local_tid >> 5;
  if (warp * 16 < Rows) {
    // Each warp owns these accumulator rows through every projection.
    __syncwarp();
  }
}

}  // namespace detail

// All Threads consumers call this function with dense local_tid in [0,Threads).
// No producer thread calls it. The dedicated and fused wrappers should invoke
// this identical body; neither wrapper may substitute a recurrence for the UT
// inverse. The producer must publish complete q/k/v/ba buffers before entry.
// This stage does not advance cursors or modify the checkpoint. FlushOutput=true
// requires the caller to fold committed history before invoking this stage.
template <int BV = 32, int Threads = 64, int BarrierId = 1,
          bool NormalizeQK = true, bool FlushOutput = false>
__device__ GDN_REPLAY_STAGE_FN void replay_verify_stage(
    const ReplayParams& p, int request, int value_head, int v_begin,
    int local_tid, ReplayScratch<BV>& s) {
  const int warp = local_tid >> 5;
  const int lane = local_tid & 31;
  const int key_head = value_head / (p.hv / p.h);
  const int bos = p.query_start_loc[request * p.stride_qsl];
  const int eos = p.query_start_loc[(request + 1) * p.stride_qsl];
  const int tokens = eos - bos;
  const int64_t state = detail::load_index(
      p.state_indices, request * p.stride_state_indices, p.state_indices_int64);
  const int64_t replay = detail::load_index(
      p.replay_indices, request * p.stride_replay_indices, p.replay_indices_int64);
  if (state <= p.null_block_id) {
    if (!FlushOutput) {
      for (int i = local_tid; i < BV * tokens; i += Threads) {
        const int token = i / BV;
        const int value = v_begin + i % BV;
        if (value < 128) {
          p.out[(bos + token) * p.stride_o_t + value_head * 128 + value] =
              __float2bfloat16_rn(0.0f);
        }
      }
    }
    return;
  }
  const bool flush = p.is_flush[replay] != 0;
  if (flush != FlushOutput) {
    return;
  }
  const int wp = p.write_pos[replay];
  const int base = p.cache_base[replay];
  const bool compensated = p.storage == ReplayStorage::kCompensated;
  const bool exact_raw = p.storage == ReplayStorage::kExactRaw;
  const bool write_key = v_begin == 0 && value_head == key_head * (p.hv / p.h);

  {
    const int t = local_tid >> 3;
    const int feature_lane = local_tid & 7;
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    if (t < tokens && NormalizeQK) {
#pragma unroll
      for (int chunk = 0; chunk < 2; ++chunk) {
        const int feature = chunk * 64 + feature_lane * 8;
        const float qs = detail::norm_chunk(
            p.q, (bos + t) * p.stride_q_t + key_head * 128 + feature);
        const float ks = detail::norm_chunk(
            p.k, (bos + t) * p.stride_k_t + key_head * 128 + feature);
        q_norm = __fadd_rn(q_norm, qs);
        k_norm = __fadd_rn(k_norm, ks);
      }
    }
    if (feature_lane == 0 && t < 8) {
      s.q_rnorm[t] = t < tokens
          ? (NormalizeQK ? detail::div_f32(1.0f, detail::sqrt_f32(q_norm + 1e-6f)) : 1.0f)
          : 0.0f;
      s.k_rnorm[t] = t < tokens
          ? (NormalizeQK ? detail::div_f32(1.0f, detail::sqrt_f32(k_norm + 1e-6f)) : 1.0f)
          : 0.0f;
    }
  }
  if (warp == 0) {
    float g = 0.0f;
    float beta = 0.0f;
    if (lane < tokens) {
      const float av = detail::load_gate(
          p.a, (bos + lane) * p.stride_a_t + value_head, p.gates_fp32);
      const float bv = detail::load_gate(
          p.b, (bos + lane) * p.stride_b_t + value_head, p.gates_fp32);
      const float bias = detail::load_gate(p.dt_bias, value_head, p.dt_bias_fp32);
      const float x = __fadd_rn(av, bias);
      const float softplus = x <= 20.0f ? logf(1.0f + detail::exp_f32(x)) : x;
      g = __fmul_rn(-detail::exp_f32(p.a_log[value_head]), softplus);
      beta = detail::div_f32(1.0f, 1.0f + detail::exp_f32(-bv));
    }
    float prefix = g;
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
      const float prior = __shfl_up_sync(0xffffffffu, prefix, offset);
      if (lane >= offset && lane < 8) {
        prefix = __fadd_rn(prefix, prior);
      }
    }
    if (lane < 8) {
      s.gates[lane] = g;
      s.beta[lane] = beta;
      s.prefix[lane] = prefix;
      s.exp_prefix[lane] = detail::exp_f32(prefix);
    }

    float history_g = 0.0f;
    if (!FlushOutput && lane < wp && lane < 16) {
      const int physical = (base + lane) & 15;
      history_g = p.g_cache[replay * p.stride_g_slot + value_head * 16 + physical];
    }
    float total_g = history_g;
#pragma unroll
    for (int offset = 8; offset; offset >>= 1) {
      total_g = __fadd_rn(total_g,
                        __shfl_xor_sync(0xffffffffu, total_g, offset));
    }
    float history_prefix = history_g;
#pragma unroll
    for (int offset = 1; offset < 16; offset <<= 1) {
      const float prior = __shfl_up_sync(0xffffffffu, history_prefix, offset);
      if (lane >= offset && lane < 16) {
        history_prefix = __fadd_rn(history_prefix, prior);
      }
    }
    if (lane < 16) {
      s.history_decay[lane] = !FlushOutput && lane < wp
                                  ? detail::exp_f32(total_g - history_prefix)
                                  : 0.0f;
    }
    if (lane == 0) {
      s.history_total_decay = detail::exp_f32(total_g);
    }
  }
  detail::group_sync<Threads, BarrierId>();

  for (int i = local_tid; i < 16 * 128; i += Threads) {
    const int token = i / 128;
    const int feature = i % 128;
    float qv = 0.0f;
    float kv = 0.0f;
    if (token < tokens) {
      qv = __bfloat162float(
          p.q[(bos + token) * p.stride_q_t + key_head * 128 + feature]);
      kv = __bfloat162float(
          p.k[(bos + token) * p.stride_k_t + key_head * 128 + feature]);
    }
    const float q_precise = token < 8
        ? __fmul_rn(qv, __fmul_rn(s.q_rnorm[token], p.scale))
        : 0.0f;
    const float k_precise = token < 8 ? __fmul_rn(kv, s.k_rnorm[token]) : 0.0f;
    const BFloat16 k_hi = __float2bfloat16_rn(k_precise);
    const BFloat16 k_lo = __float2bfloat16_rn(token < 8
        ? fmaf(kv, s.k_rnorm[token], -__bfloat162float(k_hi))
        : 0.0f);
    s.q_hi[i] = __float2bfloat16_rn(q_precise);
    s.k_hi[i] = k_hi;
    s.k_lo[i] = k_lo;
    if (write_key && token < tokens && wp + token < 16) {
      const int physical = (base + wp + token) & 15;
      const int64_t k_offset = (key_head * 16 + physical) * 128 + feature;
      p.k_cache[replay * p.stride_k_slot + k_offset] = k_hi;
      if (compensated || exact_raw) {
        p.rawk_cache[replay * p.stride_rawk_slot + k_offset] =
            exact_raw ? __float2bfloat16_rn(kv) : k_lo;
      }
    }
  }
  for (int i = local_tid; i < BV * 16; i += Threads) {
    s.checkpoint_q[i] = 0.0f;
    s.checkpoint_k[i] = 0.0f;
  }
  for (int i = local_tid; i < 16 * 16; i += Threads) {
    s.kk[i] = 0.0f;
    s.kq[i] = 0.0f;
    s.scores_q[i] = 0.0f;
    s.scores_k[i] = 0.0f;
  }
  detail::group_sync<Threads, BarrierId>();

#pragma unroll
  for (int chunk = 0; chunk < 2; ++chunk) {
    const int k_begin = chunk * 64;
    for (int i = local_tid; i < BV * 64; i += Threads) {
      const int value = v_begin + i / 64;
      const int feature = k_begin + i % 64;
      s.state_tile[i] = value < 128
          ? p.checkpoint[state * p.stride_state_slot +
                         value_head * 128 * 128 + value * 128 + feature]
          : __float2bfloat16_rn(0.0f);
    }
    if (!FlushOutput && wp > 0) {
      for (int i = local_tid; i < 16 * 64; i += Threads) {
        const int history = i / 64;
        const int feature = k_begin + i % 64;
        const int physical = (base + history) & 15;
        s.history_k[i] = history < wp
            ? p.k_cache[replay * p.stride_k_slot +
                        (key_head * 16 + physical) * 128 + feature]
            : __float2bfloat16_rn(0.0f);
      }
    }
    detail::group_sync<Threads, BarrierId>();

    detail::add_projection<16, Threads, BarrierId>(
        s.k_hi + k_begin, 128, s.k_hi + k_begin, 128,
        s.kk, s.product, local_tid);
    detail::add_projection<16, Threads, BarrierId>(
        s.k_hi + k_begin, 128, s.q_hi + k_begin, 128,
        s.kq, s.product, local_tid);
    if (compensated) {
      detail::add_projection<16, Threads, BarrierId>(
          s.k_hi + k_begin, 128, s.k_lo + k_begin, 128,
          s.kk, s.product, local_tid);
      detail::add_projection<16, Threads, BarrierId>(
          s.k_lo + k_begin, 128, s.k_hi + k_begin, 128,
          s.kk, s.product, local_tid);
      detail::add_projection<16, Threads, BarrierId>(
          s.k_lo + k_begin, 128, s.k_lo + k_begin, 128,
          s.kk, s.product, local_tid);
      detail::add_projection<16, Threads, BarrierId>(
          s.k_lo + k_begin, 128, s.q_hi + k_begin, 128,
          s.kq, s.product, local_tid);
    }
    detail::add_projection<BV, Threads, BarrierId>(
        s.state_tile, 64, s.q_hi + k_begin, 128,
        s.checkpoint_q, s.product, local_tid);
    detail::add_projection<BV, Threads, BarrierId>(
        s.state_tile, 64, s.k_hi + k_begin, 128,
        s.checkpoint_k, s.product, local_tid);
    if (compensated) {
      detail::add_projection<BV, Threads, BarrierId>(
          s.state_tile, 64, s.k_lo + k_begin, 128,
          s.checkpoint_k, s.product, local_tid);
    }
    if (!FlushOutput && wp > 0) {
      detail::add_projection<16, Threads, BarrierId>(
          s.history_k, 64, s.q_hi + k_begin, 128,
          s.scores_q, s.product, local_tid);
      detail::add_projection<16, Threads, BarrierId>(
          s.history_k, 64, s.k_hi + k_begin, 128,
          s.scores_k, s.product, local_tid);
    }
    // No warp may refill state/history staging for the next chunk until the
    // other warp has finished every projection using the current chunk.
    detail::group_sync<Threads, BarrierId>();
  }

  if (!FlushOutput && wp > 0) {
    // The source reconstructs output history from high parts only. Low D/K
    // parts are retained for the separate checkpoint-commit operation.
    for (int i = local_tid; i < BV * 16; i += Threads) {
      const int value = v_begin + i / 16;
      const int history = i % 16;
      const int physical = (base + history) & 15;
      const float d = value < 128 && history < wp
          ? __bfloat162float(p.d_cache[
                replay * p.stride_d_slot +
                (value_head * 16 + physical) * 128 + value])
          : 0.0f;
      s.history_d[i] = __float2bfloat16_rn(d * s.history_decay[history]);
    }
    // State staging is dead after the K loop; reuse it for the BF16 scores.
    BFloat16* score_q = s.state_tile;
    BFloat16* score_k = s.state_tile + 16 * 16;
    for (int i = local_tid; i < 16 * 16; i += Threads) {
      score_q[i] = __float2bfloat16_rn(s.scores_q[i]);
      score_k[i] = __float2bfloat16_rn(s.scores_k[i]);
    }
    detail::group_sync<Threads, BarrierId>();
    for (int i = local_tid; i < BV * 16; i += Threads) {
      s.checkpoint_q[i] = __fmul_rn(s.history_total_decay, s.checkpoint_q[i]);
      s.checkpoint_k[i] = __fmul_rn(s.history_total_decay, s.checkpoint_k[i]);
    }
    detail::group_sync<Threads, BarrierId>();
    detail::matrix_product<BV, 16, false, true>(
        s.history_d, 16, score_q, 16, s.checkpoint_q, local_tid);
    detail::group_sync<Threads, BarrierId>();
    detail::matrix_product<BV, 16, false, true>(
        s.history_d, 16, score_k, 16, s.checkpoint_k, local_tid);
    detail::group_sync<Threads, BarrierId>();
  }

  for (int i = local_tid; i < 8 * 8; i += Threads) {
    const int row = i / 8;
    const int col = i % 8;
    const float a = row > col && row < tokens && col < tokens
        ? (s.beta[row] * detail::exp_f32(s.prefix[row] - s.prefix[col])) * s.kk[row * 16 + col]
        : 0.0f;
    s.inverse[i] = -a;
  }
  detail::group_sync<Threads, BarrierId>();
#pragma unroll
  for (int row = 2; row < 8; ++row) {
    if (local_tid < 8) {
      const int col = local_tid;
      // Coefficients come from the original -A row, not the evolving inverse.
      float product[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float coefficient = row > j && row < tokens
            ? s.inverse[row * 8 + j]
            : 0.0f;
        product[j] = __fmul_rn(coefficient, s.inverse[j * 8 + col]);
      }
      const float update = detail::sum8(product[0], product[1], product[2], product[3],
                                        product[4], product[5], product[6], product[7]);
      s.product[col] = __fadd_rn(s.inverse[row * 8 + col], update);
    }
    detail::group_sync<Threads, BarrierId>();
    if (local_tid < 8) {
      s.inverse[row * 8 + local_tid] = s.product[local_tid];
    }
    detail::group_sync<Threads, BarrierId>();
  }
  for (int i = local_tid; i < 8 * 8; i += Threads) {
    s.inverse[i] = __fadd_rn(s.inverse[i], i / 8 == i % 8 ? 1.0f : 0.0f);
    const int source_token = i / 8;
    const int token = i % 8;
    // Every inverse row has finished before the preceding named barrier. KK
    // is dead here, so its tile can hold F for all value rows to reuse.
    s.kk[source_token * 16 + token] =
        source_token <= token && token < tokens && source_token < tokens
        ? __fmul_rn(detail::exp_f32(s.prefix[token] - s.prefix[source_token]),
                    s.kq[source_token * 16 + token])
        : 0.0f;
  }
  for (int i = local_tid; i < BV * 8; i += Threads) {
    const int value = v_begin + i / 8;
    const int token = i % 8;
    const float v = value < 128 && token < tokens
        ? __bfloat162float(p.v[(bos + token) * p.stride_v_t + value_head * 128 + value])
        : 0.0f;
    // checkpoint_k is dead after R is formed, so retain R in that buffer.
    s.checkpoint_k[(i / 8) * 16 + token] = s.beta[token] *
        fmaf(-s.exp_prefix[token], s.checkpoint_k[(i / 8) * 16 + token], v);
  }
  detail::group_sync<Threads, BarrierId>();

  for (int i = local_tid; i < BV * 8; i += Threads) {
    const int value_row = i / 8;
    const int output_token = i % 8;
    float d = 0.0f;
#pragma unroll
    for (int token = 0; token < 8; ++token) {
      d = fmaf(s.checkpoint_k[value_row * 16 + token],
               s.inverse[output_token * 8 + token], d);
    }
    s.delta[i] = d;
  }
  detail::group_sync<Threads, BarrierId>();

  for (int i = local_tid; i < BV * 8; i += Threads) {
    const int value_row = i / 8;
    const int token = i % 8;
    const int value = v_begin + value_row;
    float df = 0.0f;
#pragma unroll
    for (int source_token = 0; source_token < 8; ++source_token) {
      const float f = s.kk[source_token * 16 + token];
      df = fmaf(s.delta[value_row * 8 + source_token], f, df);
    }
    if (value < 128 && token < tokens) {
      const float o = fmaf(s.exp_prefix[token], s.checkpoint_q[value_row * 16 + token], df);
      p.out[(bos + token) * p.stride_o_t + value_head * 128 + value] =
          __float2bfloat16_rn(o);
      if (wp + token < 16) {
        const int physical = (base + wp + token) & 15;
        const int64_t offset = (value_head * 16 + physical) * 128 + value;
        const float delta = s.delta[i];
        const BFloat16 hi = __float2bfloat16_rn(delta);
        p.d_cache[replay * p.stride_d_slot + offset] = hi;
        if (compensated || exact_raw) {
          const BFloat16 raw = p.v[(bos + token) * p.stride_v_t + value_head * 128 + value];
          p.rawv_cache[replay * p.stride_rawv_slot + offset] = exact_raw
              ? raw
              : __float2bfloat16_rn(delta - __bfloat162float(hi));
        }
      }
    }
  }
  if (v_begin == 0) {
    for (int token = local_tid; token < tokens && wp + token < 16; token += Threads) {
      const int physical = (base + wp + token) & 15;
      const int offset = value_head * 16 + physical;
      p.g_cache[replay * p.stride_g_slot + offset] = s.gates[token];
      if (exact_raw) {
        p.beta_cache[replay * p.stride_beta_slot + offset] = s.beta[token];
      }
    }
  }
  detail::group_sync<Threads, BarrierId>();
}

}  // namespace gdn_stage
