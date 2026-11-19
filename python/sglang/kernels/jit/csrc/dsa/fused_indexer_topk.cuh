// Fused DSA indexer scorer + threshold candidate filter (FSTK) for SM100-family GPUs.
//
// The scoring core is DeepGEMM's legacy contiguous-KV FP8 MQA logits kernel
// (sm100_mqa_logits: 32 heads x 128, BLOCK_Q 4, SPLIT_KV 256, 3 Q / 5 KV stages, two
// 128-thread math warpgroups), reusing its UMMA instruction sequence and its epilogue
// arithmetic in the same order, so every score is bitwise the one DeepGEMM writes.
// What changes is around the score:
//   - q-blocks run in a caller-given order (longest KV span first, snake over CTAs), so one
//     launch covers the whole batch without a wave tail;
//   - kModeFilter never writes the [rows, num_kv] score matrix: a score >= tau[row] inside
//     the row's window is appended, as (score, window-relative column), to the row's
//     candidate list; the exact top-k is selected from the lists afterwards.
//   - kModeLogits writes the scores exactly as DeepGEMM (used for the bitwise check and as
//     the sample scorer).
//   - kModeNoEpi / kModeNoStore exist for calibration only: no TMEM reads at all, or the
//     full epilogue arithmetic without a store.

#pragma once

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_runtime.h>

// DeepGEMM >= 0.2.8 keeps the original MQA logits kernel and its layout under legacy/.
#if FSTK_DG_LEGACY
#include <deep_gemm/legacy/impls/sm100_mqa_logits.cuh>
namespace fstk_dg_layout = deep_gemm::legacy::layout;
#else
#include <deep_gemm/impls/sm100_mqa_logits.cuh>
namespace fstk_dg_layout = deep_gemm::layout;
#endif

#include <cstdint>

// Top-level namespace: sglang::device / sglang::host must not shadow DeepGEMM names.
namespace fstk {

using namespace deep_gemm;

constexpr uint32_t kNumHeads = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t BLOCK_Q = 4;
constexpr uint32_t SPLIT_KV = 256;
constexpr uint32_t kNumQStages = 3;
constexpr uint32_t kNumKVStages = 5;
constexpr uint32_t kNumSpecializedThreads = 128;
constexpr uint32_t kNumMathThreads = 256;
constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
constexpr uint32_t kNumThreads = kNumSpecializedThreads + kNumMathThreads;

enum Mode : int {
  kModeLogits = 0,
  kModeNoEpi = 1,
  kModeNoStore = 2,
  kModeFilter = 3,
};

struct ScoreParams {
  uint32_t num_q_tokens;
  uint32_t num_kv_tokens;
  uint32_t logits_stride;
  uint32_t num_q_blocks;
  const uint32_t* ks;
  const uint32_t* ke;
  const int32_t* order;  // [num_q_blocks] q-block visit order; nullptr: identity
  float* logits;         // kModeLogits
  const float* tau;      // kModeFilter: per-row threshold
  float* cand_score;     // kModeFilter: [num_q_tokens, cand_cap]
  int32_t* cand_col;     // kModeFilter: [num_q_tokens, cand_cap]
  int32_t* cand_count;   // kModeFilter: [num_q_tokens], zeroed by the caller
  uint32_t cand_cap;
  float* sink;  // kModeNoStore
};

// Snake round-robin over the caller's order: round r gives CTA b the entry
// r*G + b (even r) or r*G + G-1-b (odd r); each round is a bijection, so every
// q-block is visited exactly once.
struct OrderedScheduler {
  const ScoreParams& p;
  uint32_t round = 0;

  CUTLASS_DEVICE explicit OrderedScheduler(const ScoreParams& params) : p(params) {}

  template <bool kLoadSeqBounds = false>
  CUTLASS_DEVICE bool next_q_block(
      uint32_t& q_block_idx,
      uint32_t& kv_token_base,
      uint32_t& num_kv_splits,
      uint32_t* seq_k_start = nullptr,
      uint32_t* seq_k_end = nullptr) {
    const uint32_t grid = gridDim.x;
    while (true) {
      const uint32_t base = round * grid;
      if (base >= p.num_q_blocks) return false;
      const uint32_t pos = (round & 1) ? base + grid - 1 - blockIdx.x : base + blockIdx.x;
      ++round;
      if (pos >= p.num_q_blocks) continue;
      q_block_idx = p.order == nullptr ? pos : static_cast<uint32_t>(__ldg(p.order + pos));
      uint32_t start = cute::numeric_limits<uint32_t>::max();
      uint32_t end = cute::numeric_limits<uint32_t>::min();
#pragma unroll
      for (uint32_t t = 0; t < BLOCK_Q; ++t) {
        const auto row = cute::min(q_block_idx * BLOCK_Q + t, p.num_q_tokens - 1);
        const auto k_start = cute::min(__ldg(p.ks + row), p.num_kv_tokens);
        const auto k_end = cute::min(__ldg(p.ke + row), p.num_kv_tokens);
        if constexpr (kLoadSeqBounds) {
          seq_k_start[t] = k_start;
          seq_k_end[t] = k_end;
        }
        start = cute::min(start, k_start);
        end = cute::max(end, k_end);
      }
      kv_token_base = start / 4 * 4;
      num_kv_splits = math::ceil_div(end - kv_token_base, SPLIT_KV);
      return true;
    }
  }
};

template <int kMode, bool kReuseKV>
__global__ __launch_bounds__(kNumThreads, 1) void fstk_score_kernel(
    const __grid_constant__ ScoreParams params,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using qk_dtype_t = cutlass::float_e4m3_t;
  using reduce_dtype_t = float;
  using logits_dtype_t = float;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  static constexpr uint32_t kNumTmemStages = 3;
  static constexpr uint32_t UMMA_M = 128;
  static constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
  static constexpr uint32_t UMMA_N = math::constexpr_align(BLOCK_QH, 8u);
  static constexpr uint32_t UMMA_K = 32;
  static constexpr uint32_t kNumQKBytesPerToken = kHeadDim;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_QH * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = SPLIT_KV * sizeof(float);
  static constexpr uint32_t kNumWeightBytesPerRow =
      math::constexpr_align(kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), 16u);
  static constexpr uint32_t kNumWeightElementsPerRow =
      kNumWeightBytesPerRow / static_cast<uint32_t>(sizeof(reduce_dtype_t));
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumWeightBytesPerRow;

  using SharedStorage = fstk_dg_layout::MQALogitsSharedStorage<
      kNumHeads,
      kHeadDim,
      false,
      BLOCK_Q,
      SPLIT_KV,
      kNumQStages,
      kNumKVStages,
      kNumTmemStages,
      qk_dtype_t,
      reduce_dtype_t>;
  extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
  auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

  constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
  constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();

  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      smem.full_q_barriers[i].init(1);
      smem.empty_q_barriers[i].init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      smem.full_kv_barriers[i].init(1);
      smem.empty_kv_barriers[i].init(kNumMathThreads);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumTmemStages; ++i) {
      smem.full_tmem_barriers[i].init(1);
      smem.empty_tmem_barriers[i].init(128);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();

  if (warp_idx == kSpecWarpStart + 2) cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
  __syncthreads();

  RingPipeline<kNumQStages> q_pipeline;
  RingPipeline<kNumKVStages> kv_pipeline;
  RingPipeline<kNumTmemStages> tmem_pipeline;

  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = 224;

  cudaGridDependencySynchronize();

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      OrderedScheduler scheduler(params);
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

        const uint32_t q_token_base = q_block_idx * BLOCK_Q;
        tma::copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
            &tensor_map_q,
            &smem.full_q_barriers[q_stage_idx],
            smem.smem_q[q_stage_idx],
            0,
            q_token_base * kNumHeads);
        tma::copy<kNumWeightElementsPerRow, BLOCK_Q, 0>(
            &tensor_map_weights,
            &smem.full_q_barriers[q_stage_idx],
            smem.smem_weights[q_stage_idx],
            0,
            q_token_base);
        smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      OrderedScheduler scheduler(params);
      uint32_t q_block_idx, kv_base, num_kv_splits;
      uint32_t num_loaded = 0;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
#pragma unroll 1
        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
          if (kReuseKV and num_loaded >= kNumKVStages) {
            // Calibration only: the stage keeps the data of an earlier split.
            smem.full_kv_barriers[kv_stage_idx].arrive();
            continue;
          }
          ++num_loaded;
          const uint32_t kv_tma_offset = kv_base + kv_split_idx * SPLIT_KV;
          constexpr uint32_t kKVTmaTokens = 256;
#pragma unroll
          for (uint32_t i = 0; i < SPLIT_KV / kKVTmaTokens; ++i) {
            tma::copy<kHeadDim, kKVTmaTokens, 0>(
                &tensor_map_kv,
                &smem.full_kv_barriers[kv_stage_idx],
                smem.smem_kv[kv_stage_idx] + i * kKVTmaTokens * kNumQKBytesPerToken,
                0,
                kv_tma_offset + i * kKVTmaTokens);
            tma::copy<kKVTmaTokens, 1, 0>(
                &tensor_map_sf_kv,
                &smem.full_kv_barriers[kv_stage_idx],
                smem.smem_sf_kv[kv_stage_idx] + i * kKVTmaTokens,
                kv_tma_offset + i * kKVTmaTokens,
                0);
          }
          smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
        }
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 2) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    OrderedScheduler scheduler(params);
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);

      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

        if (cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

            smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
            ptx::tcgen05_after_thread_sync();

            // Same descriptors and K order as DeepGEMM: the FP32 accumulators are bitwise its.
            auto instr_desc = cute::UMMA::make_instr_desc<
                cutlass::float_e4m3_t,
                cutlass::float_e4m3_t,
                float,
                UMMA_M,
                UMMA_N,
                cute::UMMA::Major::K,
                cute::UMMA::Major::K>();
            auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
#pragma unroll
            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
              auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                  smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
              auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                  smem.smem_q[q_stage_idx], 0, k * UMMA_K);
              ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc);
            }
            ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
          }
        }
        __syncwarp();
      }
      smem.empty_q_barriers[q_stage_idx].arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    constexpr bool kLoadSeqBounds = (kMode == kModeFilter);
    uint32_t seq_k_start[kLoadSeqBounds ? BLOCK_Q : 1];
    uint32_t seq_k_end[kLoadSeqBounds ? BLOCK_Q : 1];
    const auto local_math_warp_idx = warp_idx;
    const auto math_warpgroup_idx = local_math_warp_idx / 4;
    const auto math_thread_idx = local_math_warp_idx * 32 + lane_idx;
    tmem_pipeline.advance(math_warpgroup_idx);

    float weights[BLOCK_Q][kNumHeads];
    float accum[kNumHeads];
    float sink_acc = 0.0f;

    OrderedScheduler scheduler(params);
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.template next_q_block<kLoadSeqBounds>(q_block_idx, kv_base, num_kv_splits, seq_k_start, seq_k_end)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);

      float tau[BLOCK_Q];
      if constexpr (kMode == kModeFilter) {
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          const uint32_t row = q_block_idx * BLOCK_Q + i;
          // Rows past the batch take no candidates.
          tau[i] = row < params.num_q_tokens ? __ldg(params.tau + row) : cute::numeric_limits<float>::infinity();
        }
      }

      if constexpr (kMode != kModeNoEpi) {
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow;
#pragma unroll
          for (uint32_t j = 0; j < kNumHeads; ++j) weights[i][j] = ptx::ld_shared(smem_weights_row + j);
        }
      }

      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
        const uint32_t kv_offset = kv_base + kv_split_idx * SPLIT_KV + math_thread_idx;

        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
        float scale_kv = 0.0f;
        if constexpr (kMode != kModeNoEpi) scale_kv = ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + math_thread_idx);

        CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
        smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
        ptx::tcgen05_after_thread_sync();

        // Release KV smem only after UMMA commits TMEM; earlier release races TMA overwrite
        cutlass::arch::fence_view_async_shared();
        smem.empty_kv_barriers[kv_stage_idx].arrive();

        if constexpr (kMode == kModeNoEpi) {
          ptx::tcgen05_before_thread_sync();
          smem.empty_tmem_barriers[tmem_stage_idx].arrive();
          continue;
        }

#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
          ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr, reinterpret_cast<uint32_t*>(accum));
          cutlass::arch::fence_view_async_tmem_load();
          ptx::tmem_load_32dp32b<kNumHeads / 2>(
              tmem_addr + kNumHeads / 2, reinterpret_cast<uint32_t*>(accum + kNumHeads / 2));
          cutlass::arch::fence_view_async_tmem_load();

          if (i == BLOCK_Q - 1) {
            ptx::tcgen05_before_thread_sync();
            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
          }

          // DeepGEMM's FP32 reduction, operation for operation (bitwise contract).
          auto sum_0 = make_float2(0, 0);
          auto sum_1 = make_float2(0, 0);
          const auto transform = [&](const uint32_t& j, const float2& sum) {
            auto a_0 = make_float2(accum[j], accum[j + 1]);
            auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
            auto b = make_float2(weights[i][j], weights[i][j + 1]);
            return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
          };
#pragma unroll
          for (uint32_t j = 0; j < kNumHeads; j += 4) {
            sum_0 = transform(j, sum_0);
            sum_1 = transform(j + 2, sum_1);
          }
          auto sum = __fadd2_rn(sum_0, sum_1);
          const reduce_dtype_t reduced = (sum.x + sum.y) / 2;
          const logits_dtype_t result = static_cast<logits_dtype_t>(reduced * scale_kv);

          if constexpr (kMode == kModeLogits) {
            const auto q_offset = (q_block_idx * BLOCK_Q + i) * static_cast<uint64_t>(params.logits_stride);
            params.logits[q_offset + kv_offset] = result;
          } else if constexpr (kMode == kModeNoStore) {
            sink_acc += result;
          } else {
            const uint32_t row = q_block_idx * BLOCK_Q + i;
            const uint32_t rel = kv_offset - seq_k_start[i];
            const bool take = rel < (seq_k_end[i] - seq_k_start[i]) and result >= tau[i];
            const uint32_t mask = __ballot_sync(0xffffffffu, take);
            if (mask != 0) {
              uint32_t base = 0;
              if (lane_idx == 0) base = static_cast<uint32_t>(atomicAdd(params.cand_count + row, __popc(mask)));
              base = __shfl_sync(0xffffffffu, base, 0);
              if (take) {
                const uint32_t pos = base + __popc(mask & ((1u << lane_idx) - 1u));
                if (pos < params.cand_cap) {
                  const uint64_t slot = static_cast<uint64_t>(row) * params.cand_cap + pos;
                  params.cand_score[slot] = result;
                  params.cand_col[slot] = static_cast<int32_t>(rel);
                }
              }
            }
          }
        }
      }

      cutlass::arch::fence_view_async_shared();
      smem.empty_q_barriers[q_stage_idx].arrive();
    }

    if constexpr (kMode == kModeNoStore) {
      // Keeps the arithmetic alive; never true for finite inputs.
      if (sink_acc == 1.2345e-38f) params.sink[0] = sink_acc;
    }

    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
#endif
}

// ---------------------------------------------------------------------------
// Exact selection over the candidate lists
// ---------------------------------------------------------------------------

constexpr uint32_t kSelectThreads = 1024;

struct SelectParams {
  const float* cand_score;
  const int32_t* cand_col;
  const int32_t* cand_count;
  const int32_t* lengths;       // per-row window length
  const int32_t* row_to_batch;  // per-row page-table row
  const int32_t* page_table;
  int32_t* out;           // [rows, topk]
  int32_t* fallback;      // [rows] row ids that need the full path
  int32_t* num_fallback;  // [1]
  int64_t page_table_stride;
  uint32_t num_rows;
  uint32_t topk;
  uint32_t cand_cap;
  uint32_t page_bits;
};

SGL_DEVICE uint32_t ordered_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// One CTA per row. Keys are (ordered score, ~column): unique, larger is better, so the
// top-k is every key >= the k-th largest, found by an 8-bit radix select. Equal scores
// at the k-th place resolve to the smaller columns.
__global__ __launch_bounds__(kSelectThreads, 1) void fstk_select_kernel(const SelectParams p) {
  extern __shared__ __align__(16) uint8_t select_smem[];
  auto* keys = reinterpret_cast<uint64_t*>(select_smem);
  __shared__ uint32_t hist[256];
  __shared__ uint32_t s_bin, s_above, s_out;

  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t len = static_cast<uint32_t>(max(p.lengths[row], 0));
  const int32_t* table = p.page_table + static_cast<int64_t>(p.row_to_batch[row]) * p.page_table_stride;
  const uint32_t page_mask = (1u << p.page_bits) - 1u;
  int32_t* out = p.out + static_cast<uint64_t>(row) * p.topk;
  const auto to_slot = [&](uint32_t idx) { return (table[idx >> p.page_bits] << p.page_bits) | (idx & page_mask); };

  if (len <= p.topk) {
    for (uint32_t t = tx; t < p.topk; t += kSelectThreads) out[t] = t < len ? to_slot(t) : -1;
    return;
  }
  const uint32_t n = static_cast<uint32_t>(p.cand_count[row]);
  if (n < p.topk or n > p.cand_cap) {
    if (tx == 0) p.fallback[atomicAdd(p.num_fallback, 1)] = static_cast<int32_t>(row);
    return;
  }
  const uint64_t base = static_cast<uint64_t>(row) * p.cand_cap;
  for (uint32_t t = tx; t < n; t += kSelectThreads) {
    const uint32_t col = static_cast<uint32_t>(p.cand_col[base + t]);
    keys[t] = (static_cast<uint64_t>(ordered_key(p.cand_score[base + t])) << 32) | static_cast<uint64_t>(~col);
  }
  uint64_t prefix = 0, prefix_mask = 0;
  uint32_t remaining = p.topk;
  __syncthreads();
#pragma unroll 1
  for (int shift = 56; shift >= 0; shift -= 8) {
    if (tx < 256) hist[tx] = 0;
    __syncthreads();
    for (uint32_t t = tx; t < n; t += kSelectThreads) {
      const uint64_t k = keys[t];
      if ((k & prefix_mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    if (tx < 32) {
      // Suffix sums over 256 bins, 8 per lane, from the top bin down.
      uint32_t local[8];
      uint32_t lsum = 0;
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        local[b] = hist[255 - (tx * 8 + b)];
        lsum += local[b];
      }
      uint32_t incl = lsum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tx >= static_cast<uint32_t>(o)) incl += v;
      }
      uint32_t above = incl - lsum;
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        if (above < remaining and above + local[b] >= remaining) {
          s_bin = 255 - (tx * 8 + b);
          s_above = above;
        }
        above += local[b];
      }
    }
    __syncthreads();
    remaining -= s_above;
    prefix |= static_cast<uint64_t>(s_bin) << shift;
    prefix_mask |= 0xFFull << shift;
    __syncthreads();
  }
  // prefix is the k-th largest key (keys are unique).
  if (tx == 0) s_out = 0;
  __syncthreads();
  for (uint32_t t = tx; t < n; t += kSelectThreads) {
    const uint64_t k = keys[t];
    if (k >= prefix) {
      const uint32_t pos = atomicAdd(&s_out, 1u);
      out[pos] = to_slot(~static_cast<uint32_t>(k));
    }
  }
}

// ---------------------------------------------------------------------------
// Per-row threshold from sample scores: tau = r-th largest sample score of the row
// ---------------------------------------------------------------------------

struct TauParams {
  const float* sample_logits;  // [rows, sample_stride]
  const uint32_t* sks;         // sample window per row
  const uint32_t* ske;
  const int32_t* lengths;  // full window length per row
  float* tau;
  uint64_t sample_stride;
  uint32_t num_rows;
  uint32_t rank;       // r
  uint32_t topk;
  uint32_t keep_all;   // rows with topk < len <= keep_all keep every score
};

constexpr uint32_t kTauThreads = 512;

__global__ __launch_bounds__(kTauThreads) void fstk_tau_kernel(const TauParams p) {
  __shared__ uint32_t hist[256];
  __shared__ uint32_t s_bin, s_above;
  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t len = static_cast<uint32_t>(max(p.lengths[row], 0));
  if (len <= p.topk) {
    if (tx == 0) p.tau[row] = cute::numeric_limits<float>::infinity();
    return;
  }
  const uint32_t s0 = p.sks[row], s1 = p.ske[row];
  const uint32_t n = s1 > s0 ? s1 - s0 : 0;
  if (len <= p.keep_all or n < p.rank) {
    if (tx == 0) p.tau[row] = -cute::numeric_limits<float>::infinity();
    return;
  }
  const float* in = p.sample_logits + row * p.sample_stride + s0;
  uint32_t prefix = 0, prefix_mask = 0, remaining = p.rank;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tx < 256) hist[tx] = 0;
    __syncthreads();
    for (uint32_t t = tx; t < n; t += kTauThreads) {
      const uint32_t k = ordered_key(in[t]);
      if ((k & prefix_mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    if (tx < 32) {
      uint32_t local[8];
      uint32_t lsum = 0;
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        local[b] = hist[255 - (tx * 8 + b)];
        lsum += local[b];
      }
      uint32_t incl = lsum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tx >= static_cast<uint32_t>(o)) incl += v;
      }
      uint32_t above = incl - lsum;
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        if (above < remaining and above + local[b] >= remaining) {
          s_bin = 255 - (tx * 8 + b);
          s_above = above;
        }
        above += local[b];
      }
    }
    __syncthreads();
    remaining -= s_above;
    prefix |= s_bin << shift;
    prefix_mask |= 0xFFu << shift;
    __syncthreads();
  }
  if (tx == 0) {
    const uint32_t bits = (prefix & 0x80000000u) ? (prefix & 0x7FFFFFFFu) : ~prefix;
    p.tau[row] = __uint_as_float(bits);
  }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

inline CUtensorMap make_map_2d(
    const void* base,
    CUtensorMapDataType dtype,
    uint64_t inner,
    uint64_t outer,
    uint64_t outer_stride_bytes,
    uint32_t box_inner,
    uint32_t box_outer,
    CUtensorMapSwizzle swizzle) {
  CUtensorMap map{};
  uint64_t dim[2] = {inner, outer};
  uint64_t stride[1] = {outer_stride_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t elem_stride[2] = {1, 1};
  const CUresult rc = cuTensorMapEncodeTiled(
      &map,
      dtype,
      2,
      const_cast<void*>(base),
      dim,
      stride,
      box,
      elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(rc == CUDA_SUCCESS, "fstk: cuTensorMapEncodeTiled failed with ", static_cast<int>(rc));
  return map;
}

template <int kMode, bool kReuseKV>
inline void score_impl(
    tvm::ffi::TensorView q,        // [T, 32, 128] e4m3
    tvm::ffi::TensorView kv,       // [N, 128] e4m3
    tvm::ffi::TensorView kv_scale, // [N] fp32
    tvm::ffi::TensorView weights,  // [T, 32] fp32
    tvm::ffi::TensorView ks,       // [T] int32
    tvm::ffi::TensorView ke,       // [T] int32
    tvm::ffi::TensorView order,    // [num_q_blocks] int32 (or empty: identity)
    tvm::ffi::TensorView logits,   // kModeLogits: [T, stride] fp32
    tvm::ffi::TensorView tau,      // kModeFilter: [T] fp32
    tvm::ffi::TensorView cand_score,
    tvm::ffi::TensorView cand_col,
    tvm::ffi::TensorView cand_count,
    int64_t num_ctas) {
  using namespace sglang::host;
  const auto num_q = static_cast<uint32_t>(q.size(0));
  const auto num_kv = static_cast<uint32_t>(kv.size(0));
  RuntimeCheck(q.size(1) == kNumHeads and q.size(2) == kHeadDim, "fstk: q must be [T, 32, 128]");
  RuntimeCheck(kv.size(1) == kHeadDim, "fstk: kv must be [N, 128]");
  RuntimeCheck(weights.size(0) == num_q and weights.size(1) == kNumHeads, "fstk: weights must be [T, 32]");
  RuntimeCheck(num_q > 0 and num_kv > 0, "fstk: empty input");

  ScoreParams params{};
  params.num_q_tokens = num_q;
  params.num_kv_tokens = num_kv;
  params.num_q_blocks = (num_q + BLOCK_Q - 1) / BLOCK_Q;
  params.ks = static_cast<const uint32_t*>(ks.data_ptr());
  params.ke = static_cast<const uint32_t*>(ke.data_ptr());
  params.order = order.size(0) > 0 ? static_cast<const int32_t*>(order.data_ptr()) : nullptr;
  if (params.order != nullptr) RuntimeCheck(order.size(0) == params.num_q_blocks, "fstk: order must list every q-block");
  if constexpr (kMode == kModeLogits) {
    params.logits = static_cast<float*>(logits.data_ptr());
    params.logits_stride = static_cast<uint32_t>(logits.stride(0));
  }
  if constexpr (kMode == kModeNoStore) params.sink = static_cast<float*>(logits.data_ptr());
  if constexpr (kMode == kModeFilter) {
    params.tau = static_cast<const float*>(tau.data_ptr());
    params.cand_score = static_cast<float*>(cand_score.data_ptr());
    params.cand_col = static_cast<int32_t*>(cand_col.data_ptr());
    params.cand_count = static_cast<int32_t*>(cand_count.data_ptr());
    params.cand_cap = static_cast<uint32_t>(cand_score.size(1));
  }

  const auto map_q = make_map_2d(
      q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, kHeadDim, uint64_t(num_q) * kNumHeads, kHeadDim, kHeadDim,
      BLOCK_Q * kNumHeads, CU_TENSOR_MAP_SWIZZLE_128B);
  const auto map_kv = make_map_2d(
      kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, kHeadDim, num_kv, kv.stride(0), kHeadDim, 256,
      CU_TENSOR_MAP_SWIZZLE_128B);
  const auto map_sf_kv = make_map_2d(
      kv_scale.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, num_kv, 1, (uint64_t(num_kv) * 4 + 15) / 16 * 16, 256, 1,
      CU_TENSOR_MAP_SWIZZLE_NONE);
  const auto map_w = make_map_2d(
      weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, kNumHeads, num_q, weights.stride(0) * 4, kNumHeads,
      BLOCK_Q, CU_TENSOR_MAP_SWIZZLE_NONE);

  using SharedStorage = fstk_dg_layout::MQALogitsSharedStorage<
      kNumHeads, kHeadDim, false, BLOCK_Q, SPLIT_KV, kNumQStages, kNumKVStages, 3, cutlass::float_e4m3_t, float>;
  constexpr size_t kSmem = sizeof(SharedStorage);
  constexpr auto kernel = fstk_score_kernel<kMode, kReuseKV>;
  const DLDevice device = q.device();
  runtime::init_per_device(device, [=] { runtime::set_max_dynamic_smem(kernel, kSmem); });
  LaunchKernel(static_cast<uint32_t>(num_ctas), kNumThreads, device, kSmem)
      .enable_pdl(false)(kernel, params, map_q, map_kv, map_sf_kv, map_w);
}

struct FusedIndexerTopK {
  // mode: 0 logits, 1 no-epilogue, 2 no-store, 3 filter; reuse_kv only with mode 1 (calibration).
  static void score(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView kv_scale,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView ks,
      tvm::ffi::TensorView ke,
      tvm::ffi::TensorView order,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView tau,
      tvm::ffi::TensorView cand_score,
      tvm::ffi::TensorView cand_col,
      tvm::ffi::TensorView cand_count,
      int64_t mode,
      int64_t reuse_kv,
      int64_t num_ctas) {
    switch (mode) {
      case kModeLogits:
        return score_impl<kModeLogits, false>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_score, cand_col, cand_count, num_ctas);
      case kModeNoEpi:
        if (reuse_kv)
          return score_impl<kModeNoEpi, true>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_score, cand_col, cand_count, num_ctas);
        return score_impl<kModeNoEpi, false>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_score, cand_col, cand_count, num_ctas);
      case kModeNoStore:
        return score_impl<kModeNoStore, false>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_score, cand_col, cand_count, num_ctas);
      case kModeFilter:
        return score_impl<kModeFilter, false>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_score, cand_col, cand_count, num_ctas);
      default:
        sglang::host::RuntimeCheck(false, "fstk: bad mode ", mode);
    }
  }

  static void select(
      tvm::ffi::TensorView cand_score,
      tvm::ffi::TensorView cand_col,
      tvm::ffi::TensorView cand_count,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView row_to_batch,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView fallback,
      tvm::ffi::TensorView num_fallback,
      int64_t page_size) {
    using namespace sglang::host;
    SelectParams p{};
    p.cand_score = static_cast<const float*>(cand_score.data_ptr());
    p.cand_col = static_cast<const int32_t*>(cand_col.data_ptr());
    p.cand_count = static_cast<const int32_t*>(cand_count.data_ptr());
    p.lengths = static_cast<const int32_t*>(lengths.data_ptr());
    p.row_to_batch = static_cast<const int32_t*>(row_to_batch.data_ptr());
    p.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    p.out = static_cast<int32_t*>(out.data_ptr());
    p.fallback = static_cast<int32_t*>(fallback.data_ptr());
    p.num_fallback = static_cast<int32_t*>(num_fallback.data_ptr());
    p.page_table_stride = page_table.stride(0);
    p.num_rows = static_cast<uint32_t>(out.size(0));
    p.topk = static_cast<uint32_t>(out.size(1));
    p.cand_cap = static_cast<uint32_t>(cand_score.size(1));
    RuntimeCheck(page_size > 0 and (page_size & (page_size - 1)) == 0, "fstk: page_size must be a power of two");
    p.page_bits = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(page_size)));
    if (p.num_rows == 0) return;
    const size_t smem = static_cast<size_t>(p.cand_cap) * sizeof(uint64_t);
    const DLDevice device = out.device();
    runtime::init_per_device(device, [=] { runtime::set_max_dynamic_smem(fstk_select_kernel, smem); });
    LaunchKernel(p.num_rows, kSelectThreads, device, smem)(fstk_select_kernel, p);
  }

  static void tau(
      tvm::ffi::TensorView sample_logits,
      tvm::ffi::TensorView sks,
      tvm::ffi::TensorView ske,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView tau_out,
      int64_t rank,
      int64_t topk,
      int64_t keep_all) {
    TauParams p{};
    p.sample_logits = static_cast<const float*>(sample_logits.data_ptr());
    p.sks = static_cast<const uint32_t*>(sks.data_ptr());
    p.ske = static_cast<const uint32_t*>(ske.data_ptr());
    p.lengths = static_cast<const int32_t*>(lengths.data_ptr());
    p.tau = static_cast<float*>(tau_out.data_ptr());
    p.sample_stride = static_cast<uint64_t>(sample_logits.stride(0));
    p.num_rows = static_cast<uint32_t>(tau_out.size(0));
    p.rank = static_cast<uint32_t>(rank);
    p.topk = static_cast<uint32_t>(topk);
    p.keep_all = static_cast<uint32_t>(keep_all);
    if (p.num_rows == 0) return;
    sglang::host::LaunchKernel(p.num_rows, kTauThreads, tau_out.device())(fstk_tau_kernel, p);
  }
};

}  // namespace fstk
