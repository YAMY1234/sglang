// Fused DSA indexer scorer + threshold candidate filter (FSTK) for SM100-family GPUs.
//
// The scoring core is DeepGEMM's legacy contiguous-KV FP8 MQA logits kernel
// (sm100_mqa_logits: 32 heads x 128, 4 rows per block, 256-key splits, 3 Q / 5 KV stages, two
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
constexpr uint32_t kNumQStages = 3;
constexpr uint32_t kNumSpecializedThreads = 128;
// Candidate sub-lists per row (one per math warp that can see the row).
constexpr uint32_t kNumMathWarps = 8;

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
  // kModeFilter: [num_q_tokens, kNumMathWarps, cand_sub_cap] select keys,
  // (ordered score << 32) | ~window column; one sub-list per math warp of the row's CTA.
  uint64_t* cand_key;
  int32_t* cand_count;  // kModeFilter: [num_q_tokens, kNumMathWarps], zeroed by the caller; may exceed the cap
  uint32_t cand_sub_cap;
  float* sink;  // kModeNoStore
};

// Score layouts. All three issue the same UMMA (kind::f8f6f4, M128, K32 x 4, A = 128 keys,
// B = rows x 32 heads) per element, so every score is bitwise DeepGEMM's; they differ in how
// many rows share a KV tile and how the two math warpgroups split the TMEM accumulators.
//   0: DeepGEMM's (4 rows, N 128, 256-key splits, warpgroup g takes key half g, 3 stages).
//   1: row split (4 rows, N 128, 256-key splits, warpgroup g takes rows 2g, 2g+1 of both halves).
//   2: wide (8 rows, N 256, 128-key splits, warpgroup g takes rows 4g..4g+3): 48 instead of
//      64 SMEM operand bytes per (row, key), the bound of the narrow layouts.
//   3, 4: layouts 0, 2 with two rows' accumulators loaded per TMEM wait (64 registers).
//   5: four math warpgroups, warpgroup g takes row g of both 128-key halves (4 rows, N 128,
//      256-key splits, 4 stages read by all): 32 weight registers per thread, so 4 math
//      warps per SM sub-partition hide the TMEM load latency.
template <int kLayout>
struct LayoutTraits {
  static constexpr int kBase = kLayout == 3 ? 0 : (kLayout == 4 ? 2 : kLayout);
  static constexpr bool kPairLoads = kLayout == 3 or kLayout == 4;
  static constexpr uint32_t kBlockQ = kBase == 2 ? 8 : 4;
  static constexpr uint32_t kSplitKV = kBase == 2 ? 128 : 256;
  static constexpr uint32_t kNumTmemStages = kBase == 0 ? 3 : (kBase == 2 ? 2 : 4);
  static constexpr uint32_t kNumKVStages = kBase == 2 ? 6 : 5;
  // Every math warpgroup reads every accumulator stage (layouts 1, 2, 5) or one each (layout 0).
  static constexpr bool kSharedStage = kBase != 0;
  static constexpr uint32_t kUmmaN = kBlockQ * kNumHeads;
  static constexpr uint32_t kNumMathWG = kBase == 5 ? 4 : 2;
  static constexpr uint32_t kNumMathThreads = kNumMathWG * 128;
  static constexpr uint32_t kNumThreads = kNumSpecializedThreads + kNumMathThreads;
  // The CTA's 64K registers: specialized warps keep 56, math warps get the rest (multiples of 8).
  static constexpr uint32_t kNumMathRegisters = kNumMathWG == 2 ? 224 : 104;
};

// Snake round-robin over the caller's order: round r gives CTA b the entry
// r*G + b (even r) or r*G + G-1-b (odd r); each round is a bijection, so every
// q-block is visited exactly once.
template <uint32_t kBlockQ, uint32_t kSplitKV>
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
      for (uint32_t t = 0; t < kBlockQ; ++t) {
        const auto row = cute::min(q_block_idx * kBlockQ + t, p.num_q_tokens - 1);
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
      num_kv_splits = math::ceil_div(end - kv_token_base, kSplitKV);
      return true;
    }
  }
};

// Order-preserving float -> uint32 (larger key = larger score).
CUTLASS_DEVICE uint32_t ordered_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// The FP32 head reduction of one (row, key), operation for operation DeepGEMM's
// (bitwise contract): pairs of heads, 2*relu as a + |a|, two f32x2 FMA chains.
// kAluHeads: heads [0, kAluHeads) take relu as max(a, 0) on the ALU pipe against weights
// stored doubled (fma(max(a,0), 2w, s) == fma(a+|a|, w, s) bit for bit: same exact product,
// one rounding); the rest keep a + |a| on the FMA pipe. Balances the two pipes.
template <uint32_t kAluHeads = 0>
CUTLASS_DEVICE float dg_reduce(const float* accum, const float* w) {
  static_assert(kAluHeads % 4 == 0 and kAluHeads <= kNumHeads);
  auto sum_0 = make_float2(0, 0);
  auto sum_1 = make_float2(0, 0);
  const auto transform = [&](const uint32_t& j, const float2& sum) {
    auto b = make_float2(w[j], w[j + 1]);
    if (j < kAluHeads) {
      return __ffma2_rn(make_float2(fmaxf(accum[j], 0.0f), fmaxf(accum[j + 1], 0.0f)), b, sum);
    }
    auto a_0 = make_float2(accum[j], accum[j + 1]);
    auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
    return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
  };
#pragma unroll
  for (uint32_t j = 0; j < kNumHeads; j += 4) {
    sum_0 = transform(j, sum_0);
    sum_1 = transform(j + 2, sum_1);
  }
  auto sum = __fadd2_rn(sum_0, sum_1);
  return (sum.x + sum.y) / 2;
}

template <int kMode, bool kReuseKV, int kLayout, uint32_t kAluHeads>
__global__ __launch_bounds__(LayoutTraits<kLayout>::kNumThreads, 1) void fstk_score_kernel(
    const __grid_constant__ ScoreParams params,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using Traits = LayoutTraits<kLayout>;
  using qk_dtype_t = cutlass::float_e4m3_t;
  using reduce_dtype_t = float;
  using Scheduler = OrderedScheduler<Traits::kBlockQ, Traits::kSplitKV>;
  constexpr uint32_t kBlockQ = Traits::kBlockQ;
  constexpr uint32_t kSplitKV = Traits::kSplitKV;
  constexpr uint32_t kNumKVStagesL = Traits::kNumKVStages;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kNumMathWarpGroups = Traits::kNumMathWG;
  constexpr uint32_t kNumMathThreads = Traits::kNumMathThreads;
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  static constexpr uint32_t kNumTmemStages = Traits::kNumTmemStages;
  static constexpr uint32_t UMMA_M = 128;
  static constexpr uint32_t UMMA_N = Traits::kUmmaN;
  static constexpr uint32_t UMMA_K = 32;
  static constexpr uint32_t kMmaPerSplit = kSplitKV / UMMA_M;
  static constexpr uint32_t kNumQKBytesPerToken = kHeadDim;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = UMMA_N * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = kSplitKV * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kSplitKV * sizeof(float);
  static constexpr uint32_t kNumWeightBytesPerRow =
      math::constexpr_align(kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), 16u);
  static constexpr uint32_t kNumWeightElementsPerRow =
      kNumWeightBytesPerRow / static_cast<uint32_t>(sizeof(reduce_dtype_t));
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kBlockQ * kNumWeightBytesPerRow;
  static_assert(kMmaPerSplit * UMMA_M == kSplitKV and UMMA_N <= 256);

  using SharedStorage = fstk_dg_layout::MQALogitsSharedStorage<
      kNumHeads,
      kHeadDim,
      false,
      kBlockQ,
      kSplitKV,
      kNumQStages,
      kNumKVStagesL,
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
    for (uint32_t i = 0; i < kNumKVStagesL; ++i) {
      smem.full_kv_barriers[i].init(1);
      smem.empty_kv_barriers[i].init(kNumMathThreads);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumTmemStages; ++i) {
      smem.full_tmem_barriers[i].init(1);
      smem.empty_tmem_barriers[i].init(Traits::kSharedStage ? kNumMathThreads : 128);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();

  if (warp_idx == kSpecWarpStart + 2) cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
  __syncthreads();

  RingPipeline<kNumQStages> q_pipeline;
  RingPipeline<kNumKVStagesL> kv_pipeline;
  RingPipeline<kNumTmemStages> tmem_pipeline;

  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = Traits::kNumMathRegisters;

  cudaGridDependencySynchronize();

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      Scheduler scheduler(params);
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

        const uint32_t q_token_base = q_block_idx * kBlockQ;
        tma::copy<kHeadDim, kBlockQ * kNumHeads, 0>(
            &tensor_map_q,
            &smem.full_q_barriers[q_stage_idx],
            smem.smem_q[q_stage_idx],
            0,
            q_token_base * kNumHeads);
        tma::copy<kNumWeightElementsPerRow, kBlockQ, 0>(
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
      Scheduler scheduler(params);
      uint32_t q_block_idx, kv_base, num_kv_splits;
      uint32_t num_loaded = 0;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
#pragma unroll 1
        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
          if (kReuseKV and num_loaded >= kNumKVStagesL) {
            // Calibration only: the stage keeps the data of an earlier split.
            smem.full_kv_barriers[kv_stage_idx].arrive();
            continue;
          }
          ++num_loaded;
          const uint32_t kv_tma_offset = kv_base + kv_split_idx * kSplitKV;
          tma::copy<kHeadDim, kSplitKV, 0>(
              &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx], smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
          tma::copy<kSplitKV, 1, 0>(
              &tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx], smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
          smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
        }
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 2) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    Scheduler scheduler(params);
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);

      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

        if (cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t i = 0; i < kMmaPerSplit; ++i) {
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

    constexpr bool kFilter = (kMode == kModeFilter);
    uint32_t seq_k_start[kFilter ? kBlockQ : 1];
    uint32_t seq_k_end[kFilter ? kBlockQ : 1];
    float tau[kFilter ? kBlockQ : 1] = {};
    const auto math_warpgroup_idx = warp_idx / 4;
    float sink_acc = 0.0f;

    // One (row, key) result: store, accumulate, or filter into the row's list. Callers
    // pass the row's window and tau as values so the per-row arrays stay in registers.
    // Each math warp appends to its own sub-list of the row, so the position is a
    // warp-uniform register count plus the ballot prefix: no atomic in the scoring loop.
    // ci: this warp's counter slot for the row (a compile-time index).
    constexpr uint32_t kCnt = Traits::kBase == 0 ? kBlockQ : (Traits::kBase == 5 ? 1 : kBlockQ / 2);
    uint32_t sub_cnt[kFilter ? kCnt : 1] = {};
    // A row is seen by all 8 math warps (layout 0), by its warpgroup's 4 (layouts 1, 2, 5).
    const uint32_t sub_idx = Traits::kBase == 5 ? warp_idx % 4 : warp_idx;
    const auto handle = [&](const uint32_t q_block_idx,
                            const uint32_t li,
                            const uint32_t ci,
                            const uint32_t win_start,
                            const uint32_t win_len,
                            const float row_tau,
                            const uint32_t kv_offset,
                            const float result) {
      const uint32_t row = q_block_idx * kBlockQ + li;
      if constexpr (kMode == kModeLogits) {
        params.logits[row * static_cast<uint64_t>(params.logits_stride) + kv_offset] = result;
      } else if constexpr (kMode == kModeNoStore) {
        sink_acc += result;
      } else if constexpr (kFilter) {
        const uint32_t rel = kv_offset - win_start;
        const bool take = rel < win_len and result >= row_tau;
        const uint32_t mask = __ballot_sync(0xffffffffu, take);
        if (mask != 0) {  // warp-uniform: most warps have no survivor and skip the rest
          const uint32_t base = sub_cnt[kFilter ? ci : 0];
          sub_cnt[kFilter ? ci : 0] = base + __popc(mask);
          if (take) {
            const uint32_t pos = base + __popc(mask & ((1u << lane_idx) - 1u));
            if (pos < params.cand_sub_cap)
              params.cand_key[(static_cast<uint64_t>(row) * kNumMathWarps + sub_idx) * params.cand_sub_cap + pos] =
                  (static_cast<uint64_t>(ordered_key(result)) << 32) | static_cast<uint64_t>(~rel);
          }
        }
      }
    };
    // Publishes this warp's per-row counts of the q-block (li_of(ci): the row of slot ci).
    const auto flush_counts = [&](const uint32_t q_block_idx, const uint32_t li_base) {
      if constexpr (kFilter) {
#pragma unroll
        for (uint32_t c = 0; c < kCnt; ++c) {
          const uint32_t row = q_block_idx * kBlockQ + li_base + c;
          if (lane_idx == 0 and row < params.num_q_tokens)
            params.cand_count[row * kNumMathWarps + sub_idx] = static_cast<int32_t>(sub_cnt[c]);
          sub_cnt[c] = 0;
        }
      }
    };
    const auto load_tau = [&](const uint32_t q_block_idx) {
      if constexpr (kFilter) {
#pragma unroll
        for (uint32_t i = 0; i < kBlockQ; ++i) {
          const uint32_t row = q_block_idx * kBlockQ + i;
          // Rows past the batch take no candidates.
          tau[i] = row < params.num_q_tokens ? __ldg(params.tau + row) : cute::numeric_limits<float>::infinity();
        }
      }
    };
    // Row r of this warpgroup's group of kRows (rows kRows*g + r), by constant indices.
    const auto pick = [&](const auto* arr, const uint32_t r, const uint32_t rows_per_wg) {
      return math_warpgroup_idx != 0 ? arr[rows_per_wg + r] : arr[r];
    };

    if constexpr (Traits::kBase == 0) {
      const auto math_thread_idx = warp_idx * 32 + lane_idx;
      tmem_pipeline.advance(math_warpgroup_idx);
      float weights[kBlockQ][kNumHeads];
      float accum[Traits::kPairLoads ? 2 : 1][kNumHeads];

      Scheduler scheduler(params);
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.template next_q_block<kFilter>(q_block_idx, kv_base, num_kv_splits, seq_k_start, seq_k_end)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.full_q_barriers[q_stage_idx].wait(q_phase);
        load_tau(q_block_idx);
        if constexpr (kMode != kModeNoEpi) {
#pragma unroll
          for (uint32_t i = 0; i < kBlockQ; ++i) {
            const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow;
#pragma unroll
            for (uint32_t j = 0; j < kNumHeads; ++j)
              weights[i][j] = ptx::ld_shared(smem_weights_row + j) * (j < kAluHeads ? 2.0f : 1.0f);
          }
        }

        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
          const uint32_t kv_offset = kv_base + kv_split_idx * kSplitKV + math_thread_idx;

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

          if constexpr (Traits::kPairLoads) {
#pragma unroll
            for (uint32_t i = 0; i < kBlockQ; i += 2) {
              const uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
              ptx::tmem_load_32dp32b<kNumHeads>(tmem_addr, reinterpret_cast<uint32_t*>(accum[0]));
              ptx::tmem_load_32dp32b<kNumHeads>(tmem_addr + kNumHeads, reinterpret_cast<uint32_t*>(accum[1]));
              cutlass::arch::fence_view_async_tmem_load();
              if (i + 2 == kBlockQ) {
                ptx::tcgen05_before_thread_sync();
                smem.empty_tmem_barriers[tmem_stage_idx].arrive();
              }
              const float r0 = dg_reduce<kAluHeads>(accum[0], weights[i]) * scale_kv;
              const float r1 = dg_reduce<kAluHeads>(accum[1], weights[i + 1]) * scale_kv;
#pragma unroll
              for (uint32_t r = 0; r < 2; ++r) {
                const uint32_t ii = i + r;
                handle(
                    q_block_idx,
                    ii,
                    ii,
                    seq_k_start[kFilter ? ii : 0],
                    seq_k_end[kFilter ? ii : 0] - seq_k_start[kFilter ? ii : 0],
                    tau[kFilter ? ii : 0],
                    kv_offset,
                    r == 0 ? r0 : r1);
              }
            }
          } else {
            // Filter: all rows' scores first, appends after; a vote and branch per row
            // between them would stop the rows' loads and FMA chains from overlapping.
            float results[kFilter ? kBlockQ : 1];
#pragma unroll
            for (uint32_t i = 0; i < kBlockQ; ++i) {
              uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
              ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr, reinterpret_cast<uint32_t*>(accum[0]));
              cutlass::arch::fence_view_async_tmem_load();
              ptx::tmem_load_32dp32b<kNumHeads / 2>(
                  tmem_addr + kNumHeads / 2, reinterpret_cast<uint32_t*>(accum[0] + kNumHeads / 2));
              cutlass::arch::fence_view_async_tmem_load();

              if (i == kBlockQ - 1) {
                ptx::tcgen05_before_thread_sync();
                smem.empty_tmem_barriers[tmem_stage_idx].arrive();
              }
              const float result = dg_reduce<kAluHeads>(accum[0], weights[i]) * scale_kv;
              if constexpr (kFilter) {
                results[i] = result;
              } else {
                handle(q_block_idx, i, i, 0, 0, 0.0f, kv_offset, result);
              }
            }
            if constexpr (kFilter) {
#pragma unroll
              for (uint32_t i = 0; i < kBlockQ; ++i)
                handle(q_block_idx, i, i, seq_k_start[i], seq_k_end[i] - seq_k_start[i], tau[i], kv_offset, results[i]);
            }
          }
        }

        flush_counts(q_block_idx, 0);
        cutlass::arch::fence_view_async_shared();
        smem.empty_q_barriers[q_stage_idx].arrive();
      }
    } else if constexpr (Traits::kBase == 5) {
      // Thread t of warpgroup g: TMEM lane t (key t of each 128-key half), row g.
      const uint32_t lane_key = (warp_idx % 4) * 32 + lane_idx;
      const uint32_t my_row = math_warpgroup_idx;
      float weights[kNumHeads];
      float accum[kNumHeads];

      Scheduler scheduler(params);
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.template next_q_block<kFilter>(q_block_idx, kv_base, num_kv_splits, seq_k_start, seq_k_end)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.full_q_barriers[q_stage_idx].wait(q_phase);
        load_tau(q_block_idx);
        uint32_t win_start = 0, win_len = 0;
        float row_tau = 0.0f;
        if constexpr (kFilter) {
          // Constant indices only, so the per-row arrays stay in registers.
#pragma unroll
          for (uint32_t r = 0; r < kBlockQ; ++r) {
            if (my_row == r) {
              win_start = seq_k_start[r];
              win_len = seq_k_end[r] - seq_k_start[r];
              row_tau = tau[r];
            }
          }
        }
        if constexpr (kMode != kModeNoEpi) {
          const auto smem_weights_row = smem.smem_weights[q_stage_idx] + my_row * kNumWeightElementsPerRow;
#pragma unroll
          for (uint32_t j = 0; j < kNumHeads; ++j)
            weights[j] = ptx::ld_shared(smem_weights_row + j) * (j < kAluHeads ? 2.0f : 1.0f);
        }

        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
          float scale_kv[kMmaPerSplit];
          float results[kFilter ? kMmaPerSplit : 1];
#pragma unroll
          for (uint32_t h = 0; h < kMmaPerSplit; ++h)
            scale_kv[h] = kMode != kModeNoEpi ? ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + h * UMMA_M + lane_key) : 0.0f;

#pragma unroll
          for (uint32_t h = 0; h < kMmaPerSplit; ++h) {
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
            ptx::tcgen05_after_thread_sync();
            if (h == kMmaPerSplit - 1) {
              // Every UMMA of the split has committed: the KV stage is free.
              cutlass::arch::fence_view_async_shared();
              smem.empty_kv_barriers[kv_stage_idx].arrive();
            }
            if constexpr (kMode == kModeNoEpi) {
              ptx::tcgen05_before_thread_sync();
              smem.empty_tmem_barriers[tmem_stage_idx].arrive();
              continue;
            }
            ptx::tmem_load_32dp32b<kNumHeads>(tmem_stage_idx * UMMA_N + my_row * kNumHeads, reinterpret_cast<uint32_t*>(accum));
            cutlass::arch::fence_view_async_tmem_load();
            ptx::tcgen05_before_thread_sync();
            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
            const float result = dg_reduce<kAluHeads>(accum, weights) * scale_kv[h];
            if constexpr (kFilter) {
              results[h] = result;
            } else {
              handle(q_block_idx, my_row, 0, 0, 0, 0.0f, kv_base + kv_split_idx * kSplitKV + h * UMMA_M + lane_key, result);
            }
          }
          if constexpr (kFilter) {
            // Appends after both halves are scored (see the layout-0 path).
#pragma unroll
            for (uint32_t h = 0; h < kMmaPerSplit; ++h)
              handle(q_block_idx, my_row, 0, win_start, win_len, row_tau,
                     kv_base + kv_split_idx * kSplitKV + h * UMMA_M + lane_key, results[h]);
          }
        }

        flush_counts(q_block_idx, my_row);
        cutlass::arch::fence_view_async_shared();
        smem.empty_q_barriers[q_stage_idx].arrive();
      }
    } else {
      // Thread t of warpgroup g: TMEM lane t (key t of each 128-key MMA), rows kRows*g + r.
      constexpr uint32_t kRows = kBlockQ / 2;
      const uint32_t lane_key = (warp_idx % 4) * 32 + lane_idx;
      const uint32_t row0 = kRows * math_warpgroup_idx;
      float weights[kRows][kNumHeads];
      float accum[(Traits::kBase == 1 or Traits::kPairLoads) ? 2 : 1][kNumHeads];

      Scheduler scheduler(params);
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.template next_q_block<kFilter>(q_block_idx, kv_base, num_kv_splits, seq_k_start, seq_k_end)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.full_q_barriers[q_stage_idx].wait(q_phase);
        load_tau(q_block_idx);
        uint32_t win_start[kRows] = {}, win_len[kRows] = {};
        float row_tau[kRows] = {};
        if constexpr (kFilter) {
#pragma unroll
          for (uint32_t r = 0; r < kRows; ++r) {
            win_start[r] = pick(seq_k_start, r, kRows);
            win_len[r] = pick(seq_k_end, r, kRows) - win_start[r];
            row_tau[r] = pick(tau, r, kRows);
          }
        }
        if constexpr (kMode != kModeNoEpi) {
#pragma unroll
          for (uint32_t r = 0; r < kRows; ++r) {
            const auto smem_weights_row = smem.smem_weights[q_stage_idx] + (row0 + r) * kNumWeightElementsPerRow;
#pragma unroll
            for (uint32_t j = 0; j < kNumHeads; ++j)
              weights[r][j] = ptx::ld_shared(smem_weights_row + j) * (j < kAluHeads ? 2.0f : 1.0f);
          }
        }

        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
          float scale_kv[kMmaPerSplit];
#pragma unroll
          for (uint32_t h = 0; h < kMmaPerSplit; ++h)
            scale_kv[h] = kMode != kModeNoEpi ? ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + h * UMMA_M + lane_key) : 0.0f;

#pragma unroll
          for (uint32_t h = 0; h < kMmaPerSplit; ++h) {
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
            ptx::tcgen05_after_thread_sync();
            if (h == kMmaPerSplit - 1) {
              // Every UMMA of the split has committed: the KV stage is free.
              cutlass::arch::fence_view_async_shared();
              smem.empty_kv_barriers[kv_stage_idx].arrive();
            }
            if constexpr (kMode == kModeNoEpi) {
              ptx::tcgen05_before_thread_sync();
              smem.empty_tmem_barriers[tmem_stage_idx].arrive();
              continue;
            }
            const uint32_t tmem_addr = tmem_stage_idx * UMMA_N + row0 * kNumHeads;
            const uint32_t kv_offset = kv_base + kv_split_idx * kSplitKV + h * UMMA_M + lane_key;
            if constexpr (Traits::kBase == 1) {
              ptx::tmem_load_32dp32b<kNumHeads>(tmem_addr, reinterpret_cast<uint32_t*>(accum[0]));
              ptx::tmem_load_32dp32b<kNumHeads>(tmem_addr + kNumHeads, reinterpret_cast<uint32_t*>(accum[1]));
              cutlass::arch::fence_view_async_tmem_load();
              ptx::tcgen05_before_thread_sync();
              smem.empty_tmem_barriers[tmem_stage_idx].arrive();
              const float r0 = dg_reduce<kAluHeads>(accum[0], weights[0]) * scale_kv[h];
              const float r1 = dg_reduce<kAluHeads>(accum[1], weights[1]) * scale_kv[h];
              handle(q_block_idx, row0, 0, win_start[0], win_len[0], row_tau[0], kv_offset, r0);
              handle(q_block_idx, row0 + 1, 1, win_start[1], win_len[1], row_tau[1], kv_offset, r1);
            } else if constexpr (Traits::kPairLoads) {
#pragma unroll
              for (uint32_t r = 0; r < kRows; r += 2) {
                const uint32_t addr = tmem_addr + r * kNumHeads;
                ptx::tmem_load_32dp32b<kNumHeads>(addr, reinterpret_cast<uint32_t*>(accum[0]));
                ptx::tmem_load_32dp32b<kNumHeads>(addr + kNumHeads, reinterpret_cast<uint32_t*>(accum[1]));
                cutlass::arch::fence_view_async_tmem_load();
                if (r + 2 == kRows) {
                  ptx::tcgen05_before_thread_sync();
                  smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                }
                const float r0 = dg_reduce<kAluHeads>(accum[0], weights[r]) * scale_kv[h];
                const float r1 = dg_reduce<kAluHeads>(accum[1], weights[r + 1]) * scale_kv[h];
                handle(q_block_idx, row0 + r, r, win_start[r], win_len[r], row_tau[r], kv_offset, r0);
                handle(q_block_idx, row0 + r + 1, r + 1, win_start[r + 1], win_len[r + 1], row_tau[r + 1], kv_offset, r1);
              }
            } else {
              // DeepGEMM's per-row load pattern over this warpgroup's rows; filter appends deferred.
              float results[kFilter ? kRows : 1];
#pragma unroll
              for (uint32_t r = 0; r < kRows; ++r) {
                const uint32_t addr = tmem_addr + r * kNumHeads;
                ptx::tmem_load_32dp32b<kNumHeads / 2>(addr, reinterpret_cast<uint32_t*>(accum[0]));
                cutlass::arch::fence_view_async_tmem_load();
                ptx::tmem_load_32dp32b<kNumHeads / 2>(addr + kNumHeads / 2, reinterpret_cast<uint32_t*>(accum[0] + kNumHeads / 2));
                cutlass::arch::fence_view_async_tmem_load();
                if (r == kRows - 1) {
                  ptx::tcgen05_before_thread_sync();
                  smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                }
                const float result = dg_reduce<kAluHeads>(accum[0], weights[r]) * scale_kv[h];
                if constexpr (kFilter) {
                  results[r] = result;
                } else {
                  handle(q_block_idx, row0 + r, r, 0, 0, 0.0f, kv_offset, result);
                }
              }
              if constexpr (kFilter) {
#pragma unroll
                for (uint32_t r = 0; r < kRows; ++r)
                  handle(q_block_idx, row0 + r, r, win_start[r], win_len[r], row_tau[r], kv_offset, results[r]);
              }
            }
          }
        }

        flush_counts(q_block_idx, row0);
        cutlass::arch::fence_view_async_shared();
        smem.empty_q_barriers[q_stage_idx].arrive();
      }
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

// One radix step: the bin (counted from the top) holding the remaining-th largest key.
// Warp 0 scans kBins histogram bins, kBins/32 per lane; publishes {bin, count above it}.
template <uint32_t kBins>
CUTLASS_DEVICE void radix_pick_bin(const uint32_t* hist, uint32_t remaining, uint32_t* s_bin, uint32_t* s_above) {
  constexpr uint32_t kPer = kBins / 32;
  const uint32_t lane = threadIdx.x;
  if (lane >= 32) return;
  uint32_t lsum = 0;
#pragma unroll 8
  for (uint32_t b = 0; b < kPer; ++b) lsum += hist[kBins - 1 - (lane * kPer + b)];
  uint32_t incl = lsum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= static_cast<uint32_t>(o)) incl += v;
  }
  uint32_t above = incl - lsum;
  if (above < remaining and above + lsum >= remaining) {
#pragma unroll 1
    for (uint32_t b = 0; b < kPer; ++b) {
      const uint32_t c = hist[kBins - 1 - (lane * kPer + b)];
      if (above + c >= remaining) {
        *s_bin = kBins - 1 - (lane * kPer + b);
        *s_above = above;
        break;
      }
      above += c;
    }
  }
}

constexpr uint32_t kSelectThreads = 512;
constexpr uint32_t kSelectWarps = kSelectThreads / 32;

struct SelectParams {
  const uint64_t* cand_key;
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
  uint32_t cand_sub_cap;
  uint32_t page_bits;
};

// Adds 1 to hist[bin] for every lane with a bin; lanes with equal bins share one atomic.
CUTLASS_DEVICE void warp_hist_add(uint32_t* hist, bool active, uint32_t bin) {
  const uint32_t act = __ballot_sync(0xffffffffu, active);
  if (!active) return;
  const uint32_t peers = __match_any_sync(act, bin);
  if ((__ffs(peers) - 1) == static_cast<int>(threadIdx.x % 32)) atomicAdd(hist + bin, __popc(peers));
}

// One CTA per row. Keys are (ordered score, ~column): unique, larger is better, so the
// top-k is every key >= the k-th largest, found by 8-bit radix steps that start at the
// highest bit where the row's keys differ (candidates share their leading score bits).
// Equal scores at the k-th place resolve to the smaller columns.
__global__ __launch_bounds__(kSelectThreads) void fstk_select_kernel(const SelectParams p) {
  extern __shared__ __align__(16) uint8_t select_smem[];
  auto* keys = reinterpret_cast<uint64_t*>(select_smem);
  __shared__ uint32_t hist[256];
  __shared__ uint32_t s_bin, s_above, s_out;
  __shared__ uint64_t s_or[kSelectWarps], s_and[kSelectWarps];

  const uint32_t row = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % 32, warp = tx / 32;
  const uint32_t len = static_cast<uint32_t>(max(p.lengths[row], 0));
  const int32_t* table = p.page_table + static_cast<int64_t>(p.row_to_batch[row]) * p.page_table_stride;
  const uint32_t page_mask = (1u << p.page_bits) - 1u;
  int32_t* out = p.out + static_cast<uint64_t>(row) * p.topk;
  const auto to_slot = [&](uint32_t idx) { return (table[idx >> p.page_bits] << p.page_bits) | (idx & page_mask); };

  if (len <= p.topk) {
    for (uint32_t t = tx; t < p.topk; t += kSelectThreads) out[t] = t < len ? to_slot(t) : -1;
    return;
  }
  // Gather the row's per-warp sub-lists; an overflowing sub-list or too few keys -> fallback.
  __shared__ uint32_t s_off[kNumMathWarps + 1];
  __shared__ uint32_t s_bad;
  if (tx == 0) {
    uint32_t acc = 0, bad = 0;
    for (uint32_t w = 0; w < kNumMathWarps; ++w) {
      const uint32_t c = static_cast<uint32_t>(p.cand_count[row * kNumMathWarps + w]);
      bad |= c > p.cand_sub_cap;
      s_off[w] = acc;
      acc += c;
    }
    s_off[kNumMathWarps] = acc;
    s_bad = bad or acc < p.topk;
    if (s_bad) p.fallback[atomicAdd(p.num_fallback, 1)] = static_cast<int32_t>(row);
  }
  __syncthreads();
  if (s_bad) return;
  const uint32_t n = s_off[kNumMathWarps];
  uint64_t k_or = 0, k_and = ~0ull;
#pragma unroll 1
  for (uint32_t w = 0; w < kNumMathWarps; ++w) {
    const uint64_t* src = p.cand_key + (static_cast<uint64_t>(row) * kNumMathWarps + w) * p.cand_sub_cap;
    const uint32_t c = s_off[w + 1] - s_off[w];
    for (uint32_t t = tx; t < c; t += kSelectThreads) {
      const uint64_t k = src[t];
      keys[s_off[w] + t] = k;
      k_or |= k;
      k_and &= k;
    }
  }
  // Bits set in some key and clear in another: the radix steps start at the highest one.
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    k_or |= __shfl_xor_sync(0xffffffffu, k_or, o);
    k_and &= __shfl_xor_sync(0xffffffffu, k_and, o);
  }
  if (lane == 0) {
    s_or[warp] = k_or;
    s_and[warp] = k_and;
  }
  __syncthreads();
  k_or = 0;
  k_and = ~0ull;
#pragma unroll
  for (uint32_t w = 0; w < kSelectWarps; ++w) {
    k_or |= s_or[w];
    k_and &= s_and[w];
  }
  const uint64_t diff = k_or & ~k_and;
  const int top = 63 - __clzll(diff);  // >= 0: keys are unique and n >= topk >= 1... n > 1 here

  uint64_t prefix = 0, prefix_mask = 0;
  uint32_t remaining = p.topk;
#pragma unroll 1
  for (int hi = top; hi >= 0; hi -= 8) {
    const int shift = max(hi - 7, 0);
    const uint64_t dmask = (1ull << (hi - shift + 1)) - 1;
    if (tx < 256) hist[tx] = 0;
    __syncthreads();
    const uint32_t rounds = (n + kSelectThreads - 1) / kSelectThreads;
    for (uint32_t r = 0; r < rounds; ++r) {
      const uint32_t t = r * kSelectThreads + tx;
      const uint64_t k = t < n ? keys[t] : 0;
      warp_hist_add(hist, t < n and (k & prefix_mask) == prefix, static_cast<uint32_t>((k >> shift) & dmask));
    }
    __syncthreads();
    radix_pick_bin<256>(hist, remaining, &s_bin, &s_above);
    __syncthreads();
    remaining -= s_above;
    prefix |= static_cast<uint64_t>(s_bin) << shift;
    prefix_mask |= dmask << shift;
    if (remaining == hist[s_bin]) break;  // the whole bin is selected
    __syncthreads();
  }
  // The keys >= prefix under the bits fixed so far are exactly the top-k (the last bin is taken whole).
  if (tx == 0) s_out = 0;
  __syncthreads();
  const uint32_t rounds = (n + kSelectThreads - 1) / kSelectThreads;
  for (uint32_t r = 0; r < rounds; ++r) {
    const uint32_t t = r * kSelectThreads + tx;
    const uint64_t k = t < n ? keys[t] : 0;
    const bool take = t < n and (k & prefix_mask) >= prefix;
    const uint32_t m = __ballot_sync(0xffffffffu, take);
    uint32_t base = 0;
    if (lane == 0 and m) base = atomicAdd(&s_out, __popc(m));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (take) out[base + __popc(m & ((1u << lane) - 1u))] = to_slot(~static_cast<uint32_t>(k));
  }
}

// ---------------------------------------------------------------------------
// Per-row threshold from sample scores: tau <= the r-th largest sample score of the row
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
constexpr uint32_t kTauBins = 4096;

// Two 12-bit radix passes over the ordered score bits: tau is the lower edge of the 24-bit
// bin holding the r-th largest sample (at most 2^-15 relative below it).
__global__ __launch_bounds__(kTauThreads) void fstk_tau_kernel(const TauParams p) {
  __shared__ uint32_t hist[kTauBins];
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
  for (int shift = 20; shift >= 8; shift -= 12) {
    for (uint32_t b = tx; b < kTauBins; b += kTauThreads) hist[b] = 0;
    __syncthreads();
    for (uint32_t t = tx; t < n; t += kTauThreads) {
      const uint32_t k = ordered_key(in[t]);
      if ((k & prefix_mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFFFu], 1u);
    }
    __syncthreads();
    radix_pick_bin<kTauBins>(hist, remaining, &s_bin, &s_above);
    __syncthreads();
    remaining -= s_above;
    prefix |= s_bin << shift;
    prefix_mask |= 0xFFFu << shift;
    __syncthreads();
  }
  if (tx == 0) {
    // Low 8 bits zero: the smallest ordered key of the bin, decoded back to a float.
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

template <int kMode, bool kReuseKV, int kLayout, uint32_t kAluHeads>
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
    tvm::ffi::TensorView cand_key,
    tvm::ffi::TensorView cand_count,
    int64_t num_ctas) {
  using namespace sglang::host;
  using Traits = LayoutTraits<kLayout>;
  const auto num_q = static_cast<uint32_t>(q.size(0));
  const auto num_kv = static_cast<uint32_t>(kv.size(0));
  RuntimeCheck(q.size(1) == kNumHeads and q.size(2) == kHeadDim, "fstk: q must be [T, 32, 128]");
  RuntimeCheck(kv.size(1) == kHeadDim, "fstk: kv must be [N, 128]");
  RuntimeCheck(weights.size(0) == num_q and weights.size(1) == kNumHeads, "fstk: weights must be [T, 32]");
  RuntimeCheck(num_q > 0 and num_kv > 0, "fstk: empty input");

  ScoreParams params{};
  params.num_q_tokens = num_q;
  params.num_kv_tokens = num_kv;
  params.num_q_blocks = (num_q + Traits::kBlockQ - 1) / Traits::kBlockQ;
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
    RuntimeCheck(cand_key.size(1) == kNumMathWarps and cand_count.size(1) == kNumMathWarps,
                 "fstk: candidate buffers must be [T, 8, sub_cap] and [T, 8]");
    params.cand_key = static_cast<uint64_t*>(cand_key.data_ptr());
    params.cand_count = static_cast<int32_t*>(cand_count.data_ptr());
    params.cand_sub_cap = static_cast<uint32_t>(cand_key.size(2));
  }

  const auto map_q = make_map_2d(
      q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, kHeadDim, uint64_t(num_q) * kNumHeads, kHeadDim, kHeadDim,
      Traits::kBlockQ * kNumHeads, CU_TENSOR_MAP_SWIZZLE_128B);
  const auto map_kv = make_map_2d(
      kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, kHeadDim, num_kv, kv.stride(0), kHeadDim, Traits::kSplitKV,
      CU_TENSOR_MAP_SWIZZLE_128B);
  const auto map_sf_kv = make_map_2d(
      kv_scale.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, num_kv, 1, (uint64_t(num_kv) * 4 + 15) / 16 * 16,
      Traits::kSplitKV, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  const auto map_w = make_map_2d(
      weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, kNumHeads, num_q, weights.stride(0) * 4, kNumHeads,
      Traits::kBlockQ, CU_TENSOR_MAP_SWIZZLE_NONE);

  using SharedStorage = fstk_dg_layout::MQALogitsSharedStorage<
      kNumHeads, kHeadDim, false, Traits::kBlockQ, Traits::kSplitKV, kNumQStages, Traits::kNumKVStages,
      Traits::kNumTmemStages, cutlass::float_e4m3_t, float>;
  constexpr size_t kSmem = sizeof(SharedStorage);
  constexpr auto kernel = fstk_score_kernel<kMode, kReuseKV, kLayout, kAluHeads>;
  const DLDevice device = q.device();
  runtime::init_per_device(device, [=] { runtime::set_max_dynamic_smem(kernel, kSmem); });
  LaunchKernel(static_cast<uint32_t>(num_ctas), Traits::kNumThreads, device, kSmem)
      .enable_pdl(false)(kernel, params, map_q, map_kv, map_sf_kv, map_w);
}

template <int kLayout, uint32_t kAluHeads>
inline void score_dispatch(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView kv_scale,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView ks,
    tvm::ffi::TensorView ke,
    tvm::ffi::TensorView order,
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView tau,
    tvm::ffi::TensorView cand_key,
    tvm::ffi::TensorView cand_count,
    int64_t mode,
    int64_t reuse_kv,
    int64_t num_ctas) {
#define FSTK_SCORE(M, R) \
  score_impl<M, R, kLayout, kAluHeads>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_key, cand_count, num_ctas)
  switch (mode) {
    case kModeLogits:
      return FSTK_SCORE(kModeLogits, false);
    case kModeNoEpi:
      if (reuse_kv) return FSTK_SCORE(kModeNoEpi, true);
      return FSTK_SCORE(kModeNoEpi, false);
    case kModeNoStore:
      return FSTK_SCORE(kModeNoStore, false);
    case kModeFilter:
      return FSTK_SCORE(kModeFilter, false);
    default:
      sglang::host::RuntimeCheck(false, "fstk: bad mode ", mode);
  }
#undef FSTK_SCORE
}

struct FusedIndexerTopK {
  // mode: 0 logits, 1 no-epilogue, 2 no-store, 3 filter; reuse_kv only with mode 1 (calibration).
  // layout: see LayoutTraits; alu_heads: 0 or 16 (dg_reduce).
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
      tvm::ffi::TensorView cand_key,
      tvm::ffi::TensorView cand_count,
      int64_t mode,
      int64_t reuse_kv,
      int64_t layout,
      int64_t alu_heads,
      int64_t num_ctas) {
#define FSTK_DISPATCH(LY, AH) \
  return score_dispatch<LY, AH>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_key, cand_count, mode, reuse_kv, num_ctas)
    sglang::host::RuntimeCheck(alu_heads == 0 or alu_heads == 16, "fstk: alu_heads must be 0 or 16");
    sglang::host::RuntimeCheck(layout >= 0 and layout <= 5, "fstk: layout must be 0..5");
    if (layout == 0) {
      if (alu_heads) FSTK_DISPATCH(0, 16);
      FSTK_DISPATCH(0, 0);
    }
    if (layout == 1) {
      if (alu_heads) FSTK_DISPATCH(1, 16);
      FSTK_DISPATCH(1, 0);
    }
    if (layout == 3) {
      if (alu_heads) FSTK_DISPATCH(3, 16);
      FSTK_DISPATCH(3, 0);
    }
    if (layout == 4) {
      if (alu_heads) FSTK_DISPATCH(4, 16);
      FSTK_DISPATCH(4, 0);
    }
    if (layout == 5) {
      if (alu_heads) FSTK_DISPATCH(5, 16);
      FSTK_DISPATCH(5, 0);
    }
    if (alu_heads) FSTK_DISPATCH(2, 16);
    FSTK_DISPATCH(2, 0);
#undef FSTK_DISPATCH
  }


  static void select(
      tvm::ffi::TensorView cand_key,
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
    p.cand_key = static_cast<const uint64_t*>(cand_key.data_ptr());
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
    p.cand_sub_cap = static_cast<uint32_t>(cand_key.size(2));
    RuntimeCheck(page_size > 0 and (page_size & (page_size - 1)) == 0, "fstk: page_size must be a power of two");
    p.page_bits = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(page_size)));
    if (p.num_rows == 0) return;
    const size_t smem = static_cast<size_t>(p.cand_sub_cap) * kNumMathWarps * sizeof(uint64_t);
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
