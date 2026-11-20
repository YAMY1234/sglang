// Fused DSA indexer scorer + threshold candidate filter (FSTK) for SM100-family GPUs.
//
// The scoring core is DeepGEMM's legacy contiguous-KV FP8 MQA logits kernel
// (sm100_mqa_logits: 32 heads x 128, 4 rows per block, 256-key splits, 3 Q / 5 KV stages, two
// 128-thread math warpgroups), reusing its UMMA instruction sequence and its epilogue
// arithmetic in the same order, so every score is bitwise the one DeepGEMM writes.
// What changes is around the score:
//   - q-blocks run in a caller-given order (longest KV span first, snake over CTAs), so one
//     launch covers the whole batch without a wave tail;
//   - kModeTau scores every sample_stride-th key and keeps each thread's largest samples in
//     registers; the row's threshold is the rank-th largest of those (no score store);
//   - kModeFilter never writes the [rows, num_kv] score matrix: a score >= tau[row] inside
//     the row's window is appended, as (score, window-relative column), to a per-warp
//     candidate sub-list; in layout 5 a sub-list about to overflow is compacted to a superset
//     of the warp's exact top-k, so no row overflows. The exact top-k is selected afterwards;
//     rows with fewer than k candidates are rescored on the device with tau = -inf.
//   - kModeLogits writes the scores exactly as DeepGEMM (used for the bitwise check).
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

// FSTK_SERVING=1 builds only the layout and passes the prefill path runs.
#ifndef FSTK_SERVING
#define FSTK_SERVING 0
#endif
#ifndef FSTK_SERVING_LAYOUT
#define FSTK_SERVING_LAYOUT 5
#endif

// DeepGEMM >= 0.2.8 keeps the original MQA logits kernel and its layout under legacy/.
#if FSTK_DG_LEGACY
#include <deep_gemm/legacy/impls/sm100_mqa_logits.cuh>
namespace fstk_dg_layout = deep_gemm::legacy::layout;
#else
#include <deep_gemm/impls/sm100_mqa_logits.cuh>
namespace fstk_dg_layout = deep_gemm::layout;
#endif

#include <algorithm>
#include <cstdint>

// Top-level namespace: sglang::device / sglang::host must not shadow DeepGEMM names.
namespace fstk {

using namespace deep_gemm;

constexpr uint32_t kNumHeads = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kNumQStages = 3;
constexpr uint32_t kNumSpecializedThreads = 128;
// Candidate sub-lists per row (one per math warp that can see the row) in layouts 0-4.
constexpr uint32_t kNumMathWarps = 8;
// Layout 5: a row is seen by the 4 warps of one warpgroup.
constexpr uint32_t kNumSubLists5 = 4;
// kModeTau: samples kept per thread (128 threads per row).
constexpr uint32_t kTauKeep = 8;
// kModeTau: hardware barriers kTauBarrierBase + warpgroup. 0 is __syncthreads, 1-7 are CUTLASS's
// reserved ids, and NamedBarrier(n, 0) at the end of the math warps is hardware barrier 8.
constexpr uint32_t kTauBarrierBase = 12;

enum Mode : int {
  kModeLogits = 0,
  kModeNoEpi = 1,
  kModeNoStore = 2,
  kModeFilter = 3,
  // Layout 5 only: per-row threshold from sample scores (no score store).
  kModeTau = 4,
  // Layout 5 only: kModeFilter whose sub-lists compact instead of overflowing (the repair pass).
  kModeFilterCompact = 5,
};

// Device counters of the fused path (uint64, read asynchronously by the host).
enum Stat : int {
  kStatCalls = 0,        // repair-prep launches (one per fused call)
  kStatRepairRows = 1,   // rows whose threshold was above their k-th score, rescored on the device
  kStatMaxRepair = 2,    // most such rows in one call
  kStatCompactions = 3,  // candidate sub-list compactions in the filter pass
  kStatUnrepaired = 4,   // rows the repair select could not settle (must stay 0)
  kStatRepairCalls = 5,  // calls with at least one row repaired
  kNumStats = 8,
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
  // kModeFilter: [num_q_tokens, sub-lists, cand_sub_cap] select keys,
  // (ordered score << 32) | ~window column; one sub-list per math warp that sees the row.
  uint64_t* cand_key;
  // kModeFilter: [num_q_tokens, sub-lists], zeroed by the caller; may exceed the cap in
  // layouts 0-4, never in layout 5 (it compacts).
  int32_t* cand_count;
  uint32_t cand_sub_cap;
  float* sink;  // kModeNoStore
  // Layout 5 filter: a sub-list about to overflow keeps a superset of its top-k (k = topk).
  uint32_t topk;
  unsigned long long* stats;  // optional [kNumStats]
  // Repair pass: rows are compact (row i holds row_map[i]'s query); only rows < *num_rows_dev
  // exist, so the pass is empty when it is 0.
  const int32_t* row_map;
  const int32_t* num_rows_dev;
  // kModeTau: tau_out[row] = the rank-th largest sample score kept by the row's threads (a lower
  // bound of the rank-th largest sample); +inf when lengths[row] <= topk, -inf when
  // lengths[row] <= keep_all or the row has fewer samples.
  float* tau_out;
  const int32_t* lengths;
  uint32_t rank;
  uint32_t keep_all;
  // Windows are [ceil(ks / ks_div), ceil(ke / ks_div)) (the sample pass: ks_div = sample stride).
  uint32_t ks_div;
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
  uint32_t num_q_blocks;

  CUTLASS_DEVICE explicit OrderedScheduler(const ScoreParams& params) : p(params), num_q_blocks(params.num_q_blocks) {
    if (p.num_rows_dev != nullptr) {
      const uint32_t n = static_cast<uint32_t>(max(*p.num_rows_dev, 0));
      num_q_blocks = cute::min(num_q_blocks, (n + kBlockQ - 1) / kBlockQ);
    }
  }

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
      if (base >= num_q_blocks) return false;
      const uint32_t pos = (round & 1) ? base + grid - 1 - blockIdx.x : base + blockIdx.x;
      ++round;
      if (pos >= num_q_blocks) continue;
      q_block_idx = p.order == nullptr ? pos : static_cast<uint32_t>(__ldg(p.order + pos));
      uint32_t start = cute::numeric_limits<uint32_t>::max();
      uint32_t end = cute::numeric_limits<uint32_t>::min();
#pragma unroll
      for (uint32_t t = 0; t < kBlockQ; ++t) {
        const auto row = cute::min(q_block_idx * kBlockQ + t, p.num_q_tokens - 1);
        const auto k_start = cute::min((__ldg(p.ks + row) + p.ks_div - 1) / p.ks_div, p.num_kv_tokens);
        const auto k_end = cute::min((__ldg(p.ke + row) + p.ks_div - 1) / p.ks_div, p.num_kv_tokens);
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

// Inverse of ordered_key.
CUTLASS_DEVICE float ordered_key_to_float(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}

// Ascending sort of one 64-bit value per lane across the warp (bitonic).
CUTLASS_DEVICE uint64_t warp_sort_asc(uint64_t v, uint32_t lane) {
#pragma unroll
  for (uint32_t k = 2; k <= 32; k <<= 1) {
#pragma unroll
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      const uint64_t o = __shfl_xor_sync(0xffffffffu, static_cast<unsigned long long>(v), j);
      const bool keep_min = ((lane & j) == 0) == ((lane & k) == 0);
      v = keep_min ? (v < o ? v : o) : (v < o ? o : v);
    }
  }
  return v;
}

// Loads per lane kept in flight by the compaction passes (the sub-list sits in L2).
constexpr uint32_t kCompactBatch = 16;

// Keys of the first n entries of a warp's sub-list that are >= thr.
CUTLASS_DEVICE uint32_t warp_count_ge(const uint64_t* list, uint32_t n, uint64_t thr, uint32_t lane) {
  const auto* src = reinterpret_cast<const unsigned long long*>(list);
  uint32_t c = 0;
#pragma unroll 1
  for (uint32_t base = 0; base < n; base += 32 * kCompactBatch) {
    uint64_t k[kCompactBatch];
#pragma unroll
    for (uint32_t b = 0; b < kCompactBatch; ++b) {
      const uint32_t t = base + b * 32 + lane;
      k[b] = t < n ? __ldcg(src + t) : 0ull;
    }
#pragma unroll
    for (uint32_t b = 0; b < kCompactBatch; ++b) c += base + b * 32 + lane < n and k[b] >= thr;
  }
  return __reduce_add_sync(0xffffffffu, c);
}

struct Compacted {
  uint64_t thr;
  uint32_t n;
};

// Called by a whole warp whose sub-list (n distinct keys) is about to overflow: picks thr with
// topk <= count(key >= thr) <= keep_max and drops every key below it. Every key of the warp's
// top-k is >= the k-th largest stored key >= thr, so the kept set still holds the warp's top-k,
// and a later key can enter the warp's top-k only if it is >= thr.
// Probes start at a 32-key sample's quantiles, then bisect on the key value: count(>= x) drops
// by at most 1 per unit of x (keys are distinct), so a valid thr exists between the bounds.
__device__ __noinline__ Compacted
fstk_compact(uint64_t* list, uint32_t n, uint32_t topk, uint32_t keep_max, unsigned long long* stats) {
  const uint32_t lane = threadIdx.x % 32;
  __syncwarp();
  const auto* src = reinterpret_cast<const unsigned long long*>(list);
  const uint64_t sample = warp_sort_asc(__ldcg(src + ((2 * lane + 1) * n) / 64), lane);
  const uint32_t target = (topk + keep_max) / 2;
  // count(>= the i-th smallest sample) ~ (32 - i) / 32 * n
  int i = 32 - static_cast<int>((static_cast<uint64_t>(target) * 32 + n / 2) / n);
  uint64_t lo = 0, hi = ~0ull;  // count(>= lo) > keep_max, count(>= hi) < topk
  uint64_t thr = 0;
#pragma unroll 1
  for (uint32_t iter = 0; iter < 72; ++iter) {
    uint64_t cand = __shfl_sync(0xffffffffu, static_cast<unsigned long long>(sample), cute::max(0, cute::min(31, i)));
    if (iter >= 4 or not(cand > lo and cand < hi)) cand = lo + (hi - lo) / 2;
    const uint32_t c = warp_count_ge(list, n, cand, lane);
    if (c > keep_max) {
      lo = cand;
      i += 2;
    } else if (c < topk) {
      hi = cand;
      i -= 2;
    } else {
      thr = cand;
      break;
    }
  }
  // In place: a batch's keys are all loaded before any is stored, and a store never passes the
  // batch being read (out <= the batch's first index + 32 * its round).
  uint32_t out = 0;
#pragma unroll 1
  for (uint32_t base = 0; base < n; base += 32 * kCompactBatch) {
    uint64_t k[kCompactBatch];
#pragma unroll
    for (uint32_t b = 0; b < kCompactBatch; ++b) {
      const uint32_t t = base + b * 32 + lane;
      k[b] = t < n ? __ldcg(src + t) : 0ull;
    }
#pragma unroll
    for (uint32_t b = 0; b < kCompactBatch; ++b) {
      const bool keep = base + b * 32 + lane < n and k[b] >= thr;
      const uint32_t m = __ballot_sync(0xffffffffu, keep);
      if (keep) list[out + __popc(m & ((1u << lane) - 1u))] = k[b];
      out += __popc(m);
    }
  }
  __syncwarp();
  if (stats != nullptr and lane == 0) atomicAdd(stats + kStatCompactions, 1ull);
  return {thr, out};
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

    constexpr bool kCompact = (kMode == kModeFilterCompact);
    constexpr bool kFilter = (kMode == kModeFilter) or kCompact;
    constexpr bool kTau = (kMode == kModeTau);
    static_assert(not(kTau or kCompact) or Traits::kBase == 5, "kModeTau / kModeFilterCompact are layout 5 only");
    constexpr bool kWindowed = kFilter or kTau;
    uint32_t seq_k_start[kWindowed ? kBlockQ : 1];
    uint32_t seq_k_end[kWindowed ? kBlockQ : 1];
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
      // Filter: rows past the batch (or past the repair list) take nothing and publish nothing.
      const uint32_t valid_rows = params.num_rows_dev != nullptr
          ? cute::min(params.num_q_tokens, static_cast<uint32_t>(max(*params.num_rows_dev, 0)))
          : params.num_q_tokens;
      // kModeTau: the kTauKeep largest in-window sample scores this thread has seen, descending.
      float top[kTau ? kTauKeep : 1];
      // kModeTau: per-warpgroup partial sums, double-buffered across calls of wg_sum.
      __shared__ uint32_t s_wg_red[kTau ? 4 : 1][2][4];
      uint32_t red_parity = 0;
      const auto wg_sum = [&](uint32_t v) -> uint32_t {
        v = __reduce_add_sync(0xffffffffu, v);
        if (lane_idx == 0) s_wg_red[kTau ? my_row : 0][red_parity][warp_idx % 4] = v;
        asm volatile("bar.sync %0, 128;" ::"r"(kTauBarrierBase + my_row) : "memory");
        const uint32_t* s = s_wg_red[kTau ? my_row : 0][red_parity];
        red_parity ^= 1;
        return s[0] + s[1] + s[2] + s[3];
      };

      Scheduler scheduler(params);
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.template next_q_block<kWindowed>(q_block_idx, kv_base, num_kv_splits, seq_k_start, seq_k_end)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.full_q_barriers[q_stage_idx].wait(q_phase);
        load_tau(q_block_idx);
        const uint32_t row = q_block_idx * kBlockQ + my_row;
        uint32_t win_start = 0, win_len = 0;
        float row_tau = 0.0f;
        if constexpr (kWindowed) {
          // Constant indices only, so the per-row arrays stay in registers.
#pragma unroll
          for (uint32_t r = 0; r < kBlockQ; ++r) {
            if (my_row == r) {
              win_start = seq_k_start[r];
              win_len = seq_k_end[r] - seq_k_start[r];
              if constexpr (kFilter) row_tau = tau[r];
            }
          }
        }
        // Filter: this warp's count of the row (kModeFilter: may pass the cap, which sends the row
        // to the repair pass); kModeFilterCompact: its sub-list and exact-top-k threshold.
        uint64_t* sub_list = nullptr;
        uint32_t sub_n = 0;
        uint64_t row_thr = 0;
        uint32_t out_row = row;
        if constexpr (kCompact) {
          if (row < valid_rows) {
            out_row = params.row_map != nullptr ? static_cast<uint32_t>(__ldg(params.row_map + row)) : row;
          } else {
            win_len = 0;
          }
          sub_list = params.cand_key + (static_cast<uint64_t>(out_row) * kNumSubLists5 + sub_idx) * params.cand_sub_cap;
        }
        if constexpr (kTau) {
#pragma unroll
          for (uint32_t j = 0; j < kTauKeep; ++j) top[j] = -cute::numeric_limits<float>::infinity();
        }
        // kModeFilter (main pass): one vote per 32 keys, stores only for survivors, no call in the
        // loop; a count past the cap marks the row for the repair pass.
        const auto append_plain = [&](const uint32_t kv_offset, const float result) {
          const uint32_t rel = kv_offset - win_start;
          const bool take = rel < win_len and result >= row_tau;
          const uint32_t mask = __ballot_sync(0xffffffffu, take);
          if (mask != 0) {  // warp-uniform: most warps have no survivor and skip the rest
            const uint32_t base = sub_n;
            sub_n = base + __popc(mask);
            if (take) {
              const uint32_t pos = base + __popc(mask & ((1u << lane_idx) - 1u));
              if (pos < params.cand_sub_cap)
                params.cand_key[(static_cast<uint64_t>(row) * kNumSubLists5 + sub_idx) * params.cand_sub_cap + pos] =
                    (static_cast<uint64_t>(ordered_key(result)) << 32) | static_cast<uint64_t>(~rel);
            }
          }
        };
        // kModeFilterCompact: appends one score per lane to the warp's sub-list if it can be in
        // the row's top-k; compacts first if the sub-list would overflow.
        const auto append = [&](const uint32_t kv_offset, const float result) {
          const uint32_t rel = kv_offset - win_start;
          const bool take0 = rel < win_len and result >= row_tau;
          if (__any_sync(0xffffffffu, take0)) {  // warp-uniform: most warps have no survivor
            const uint64_t key = (static_cast<uint64_t>(ordered_key(result)) << 32) | static_cast<uint64_t>(~rel);
            bool take = take0 and key >= row_thr;
            uint32_t mask = __ballot_sync(0xffffffffu, take);
            if (mask != 0) {
              if (sub_n + __popc(mask) > params.cand_sub_cap) {
                const Compacted c =
                    fstk_compact(sub_list, sub_n, params.topk, params.cand_sub_cap - 64, params.stats);
                sub_n = c.n;
                row_thr = c.thr;
                row_tau = fmaxf(row_tau, ordered_key_to_float(static_cast<uint32_t>(row_thr >> 32)));
                take = take and key >= row_thr;
                mask = __ballot_sync(0xffffffffu, take);
              }
              const uint32_t pos = sub_n + __popc(mask & ((1u << lane_idx) - 1u));
              if (take and pos < params.cand_sub_cap) sub_list[pos] = key;
              // Never taken (fstk_compact always lands): a dropped key is counted, not written out of bounds.
              if (take and pos >= params.cand_sub_cap and params.stats != nullptr)
                atomicAdd(params.stats + kStatUnrepaired, 1ull);
              sub_n = cute::min(sub_n + __popc(mask), params.cand_sub_cap);
            }
          }
        };
        // kModeTau: keeps the thread's kTauKeep largest in-window samples.
        const auto keep_sample = [&](const uint32_t kv_offset, const float result) {
          float v = (kv_offset - win_start) < win_len ? result : -cute::numeric_limits<float>::infinity();
          if (__any_sync(0xffffffffu, v > top[kTauKeep - 1])) {
#pragma unroll
            for (uint32_t j = 0; j < kTauKeep; ++j) {
              const float hi = fmaxf(top[j], v);
              v = fminf(top[j], v);
              top[j] = hi;
            }
          }
        };
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
          float results[kWindowed ? kMmaPerSplit : 1];
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
            if constexpr (kWindowed) {
              results[h] = result;
            } else {
              handle(q_block_idx, my_row, 0, 0, 0, 0.0f, kv_base + kv_split_idx * kSplitKV + h * UMMA_M + lane_key, result);
            }
          }
          // Appends after both halves are scored (see the layout-0 path).
#pragma unroll
          for (uint32_t h = 0; h < kMmaPerSplit; ++h) {
            const uint32_t kv_offset = kv_base + kv_split_idx * kSplitKV + h * UMMA_M + lane_key;
            if constexpr (kCompact)
              append(kv_offset, results[h]);
            else if constexpr (kFilter)
              append_plain(kv_offset, results[h]);
            if constexpr (kTau) keep_sample(kv_offset, results[h]);
          }
        }

        if constexpr (kFilter) {
          if (lane_idx == 0 and row < valid_rows)
            params.cand_count[out_row * kNumSubLists5 + sub_idx] = static_cast<int32_t>(sub_n);
        }
        if constexpr (kTau) {
          // tau = the largest T with at least rank kept samples >= T (bisection on the ordered key).
          constexpr uint32_t kNegInfKey = 0x007FFFFFu;  // ordered_key(-inf)
          uint32_t keys[kTauKeep];
          uint32_t real = 0;
#pragma unroll
          for (uint32_t j = 0; j < kTauKeep; ++j) {
            keys[j] = ordered_key(top[j]);
            real += keys[j] > kNegInfKey;
          }
          const uint32_t total = wg_sum(real);
          uint32_t lo = 0;
          if (total >= params.rank) {
            lo = kNegInfKey + 1;
            uint32_t hi = 0xFFFFFFFFu;
#pragma unroll 1
            while (lo < hi) {
              const uint32_t mid = lo + (hi - lo + 1) / 2;
              uint32_t c = 0;
#pragma unroll
              for (uint32_t j = 0; j < kTauKeep; ++j) c += keys[j] >= mid;
              if (wg_sum(c) >= params.rank)
                lo = mid;
              else
                hi = mid - 1;
            }
          }
          if (warp_idx % 4 == 0 and lane_idx == 0 and row < params.num_q_tokens) {
            const int32_t len = __ldg(params.lengths + row);
            float t = ordered_key_to_float(lo);
            if (len <= static_cast<int32_t>(params.topk))
              t = cute::numeric_limits<float>::infinity();
            else if (len <= static_cast<int32_t>(params.keep_all) or total < params.rank)
              t = -cute::numeric_limits<float>::infinity();
            params.tau_out[row] = t;
          }
        }
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

constexpr uint32_t kSelectThreads = 256;
constexpr uint32_t kSelectWarps = kSelectThreads / 32;
// Keys a row may hold in the select kernel's shared memory: 64 KiB for the main select (several
// CTAs per SM), 128 KiB for the repair select (rows of up to 4 full sub-lists). A row with more
// candidates reads them from global memory in every radix step.
constexpr uint32_t kSelectMaxKeys = 8192;
constexpr uint32_t kSelectRepairKeys = 16384;

struct SelectParams {
  const uint64_t* cand_key;     // [rows, num_sub, cand_sub_cap]
  const int32_t* cand_count;    // [rows, num_sub]
  const int32_t* lengths;       // per-row window length
  const int32_t* row_to_batch;  // per-row page-table row
  const int32_t* page_table;
  int32_t* out;           // [rows, topk]
  int32_t* fallback;      // [rows] row ids with fewer than topk candidates (nullptr: count them unrepaired)
  int32_t* num_fallback;  // [1]
  // Optional: select only rows row_list[0 .. *num_list) (grid-stride); else row = blockIdx.x.
  const int32_t* row_list;
  const int32_t* num_list;
  unsigned long long* stats;  // optional [kNumStats]
  int64_t page_table_stride;
  uint32_t num_rows;
  uint32_t topk;
  uint32_t cand_sub_cap;
  uint32_t num_sub;
  uint32_t page_bits;
  uint32_t smem_keys;  // keys that fit the launch's dynamic shared memory
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
  __shared__ uint32_t s_off[kNumMathWarps + 1];
  __shared__ uint32_t s_bad;

  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % 32, warp = tx / 32;
  const uint32_t page_mask = (1u << p.page_bits) - 1u;
  const uint32_t list_len = p.row_list != nullptr ? static_cast<uint32_t>(max(*p.num_list, 0)) : gridDim.x;

#pragma unroll 1
  for (uint32_t it = blockIdx.x; it < list_len; it += gridDim.x) {
    const uint32_t row = p.row_list != nullptr ? static_cast<uint32_t>(p.row_list[it]) : it;
    const uint32_t len = static_cast<uint32_t>(max(p.lengths[row], 0));
    const int32_t* table = p.page_table + static_cast<int64_t>(p.row_to_batch[row]) * p.page_table_stride;
    int32_t* out = p.out + static_cast<uint64_t>(row) * p.topk;
    const auto to_slot = [&](uint32_t idx) { return (table[idx >> p.page_bits] << p.page_bits) | (idx & page_mask); };

    if (len <= p.topk) {
      for (uint32_t t = tx; t < p.topk; t += kSelectThreads) out[t] = t < len ? to_slot(t) : -1;
      __syncthreads();
      continue;
    }
    // Gather the row's sub-lists; an overflowing sub-list (layouts 0-4) or too few keys -> fallback.
    if (tx == 0) {
      uint32_t acc = 0, bad = 0;
      for (uint32_t w = 0; w < p.num_sub; ++w) {
        const uint32_t c = static_cast<uint32_t>(p.cand_count[row * p.num_sub + w]);
        bad |= c > p.cand_sub_cap;
        s_off[w] = acc;
        acc += c;
      }
      s_off[p.num_sub] = acc;
      s_bad = bad or acc < p.topk;
      if (s_bad) {
        if (p.fallback != nullptr)
          p.fallback[atomicAdd(p.num_fallback, 1)] = static_cast<int32_t>(row);
        else if (p.stats != nullptr)
          atomicAdd(p.stats + kStatUnrepaired, 1ull);
      }
    }
    __syncthreads();
    if (s_bad) {
      __syncthreads();
      continue;
    }
    const uint32_t n = s_off[p.num_sub];
    const bool in_smem = n <= p.smem_keys;
    const uint64_t* row_keys = p.cand_key + static_cast<uint64_t>(row) * p.num_sub * p.cand_sub_cap;
    // The t-th candidate of the row in sub-list order.
    const auto gather = [&](uint32_t t) {
      uint32_t w = 0;
#pragma unroll
      for (uint32_t v = 1; v < kNumMathWarps; ++v) w += v < p.num_sub and t >= s_off[v];
      return row_keys[w * p.cand_sub_cap + (t - s_off[w])];
    };
    const auto key_at = [&](uint32_t t) { return in_smem ? keys[t] : gather(t); };
    uint64_t k_or = 0, k_and = ~0ull;
    // One flat pass over all sub-lists, so every thread's loads are in flight together.
    for (uint32_t t = tx; t < n; t += kSelectThreads) {
      const uint64_t k = gather(t);
      if (in_smem) keys[t] = k;
      k_or |= k;
      k_and &= k;
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
    const int top = 63 - __clzll(diff);  // >= 0: keys are unique and n > topk >= 1

    uint64_t prefix = 0, prefix_mask = 0;
    uint32_t remaining = p.topk;
    const uint32_t rounds = (n + kSelectThreads - 1) / kSelectThreads;
#pragma unroll 1
    for (int hi = top; hi >= 0; hi -= 8) {
      const int shift = max(hi - 7, 0);
      const uint64_t dmask = (1ull << (hi - shift + 1)) - 1;
      hist[tx] = 0;
      __syncthreads();
      for (uint32_t r = 0; r < rounds; ++r) {
        const uint32_t t = r * kSelectThreads + tx;
        const uint64_t k = t < n ? key_at(t) : 0;
        warp_hist_add(hist, t < n and (k & prefix_mask) == prefix, static_cast<uint32_t>((k >> shift) & dmask));
      }
      __syncthreads();
      radix_pick_bin<256>(hist, remaining, &s_bin, &s_above);
      __syncthreads();
      remaining -= s_above;
      prefix |= static_cast<uint64_t>(s_bin) << shift;
      prefix_mask |= dmask << shift;
      const bool whole_bin = remaining == hist[s_bin];
      __syncthreads();
      if (whole_bin) break;
    }
    // The keys >= prefix under the bits fixed so far are exactly the top-k (the last bin is taken whole).
    if (tx == 0) s_out = 0;
    __syncthreads();
    for (uint32_t r = 0; r < rounds; ++r) {
      const uint32_t t = r * kSelectThreads + tx;
      const uint64_t k = t < n ? key_at(t) : 0;
      const bool take = t < n and (k & prefix_mask) >= prefix;
      const uint32_t m = __ballot_sync(0xffffffffu, take);
      uint32_t base = 0;
      if (lane == 0 and m) base = atomicAdd(&s_out, __popc(m));
      base = __shfl_sync(0xffffffffu, base, 0);
      if (take) out[base + __popc(m & ((1u << lane) - 1u))] = to_slot(~static_cast<uint32_t>(k));
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Device-side repair of rows whose threshold was above their k-th score
// ---------------------------------------------------------------------------

struct RepairParams {
  const int32_t* fallback;      // [rows] from the select kernel
  const int32_t* num_fallback;  // [1]
  const uint4* q;               // [rows, 32 x 128 fp8] = 256 uint4 per row
  uint4* q_c;
  const float* w;  // [rows, 32], row stride w_stride
  float* w_c;      // [rows, 32]
  const int32_t* ks;
  const int32_t* ke;
  int32_t* ks_c;
  int32_t* ke_c;
  const float* tau;      // the main pass's thresholds
  float* tau_c;
  int32_t* cand_count;  // [rows, num_sub]: read (overflow or too few), then reset
  unsigned long long* stats;
  uint32_t w_stride;
  uint32_t rows;
  uint32_t num_sub;
  uint32_t cand_sub_cap;
};

constexpr uint32_t kRepairThreads = 256;

// Gathers the listed rows into compact arrays for a compacting filter pass: a row whose sub-list
// overflowed keeps its threshold (it has more than topk keys above it), a row with fewer than topk
// candidates gets tau = -inf. Rows after the list up to the q-block boundary get an empty window at
// the last listed row's start. A no-op when the list is empty.
__global__ __launch_bounds__(kRepairThreads) void fstk_repair_prep_kernel(const RepairParams p) {
  const uint32_t n = cute::min(static_cast<uint32_t>(max(*p.num_fallback, 0)), p.rows);
  const uint32_t n_pad = cute::min((n + 3) / 4 * 4, p.rows);
  const uint32_t tx = threadIdx.x;
  if (blockIdx.x == 0 and tx == 0 and p.stats != nullptr) {
    atomicAdd(p.stats + kStatCalls, 1ull);
    atomicAdd(p.stats + kStatRepairRows, static_cast<unsigned long long>(n));
    atomicMax(p.stats + kStatMaxRepair, static_cast<unsigned long long>(n));
    if (n > 0) atomicAdd(p.stats + kStatRepairCalls, 1ull);
  }
#pragma unroll 1
  for (uint32_t i = blockIdx.x; i < n_pad; i += gridDim.x) {
    if (i < n) {
      const uint32_t src = static_cast<uint32_t>(p.fallback[i]);
      p.q_c[static_cast<uint64_t>(i) * 256 + tx] = p.q[static_cast<uint64_t>(src) * 256 + tx];
      if (tx < 32) p.w_c[i * 32 + tx] = p.w[static_cast<uint64_t>(src) * p.w_stride + tx];
      if (tx == 0) {
        bool overflow = false;
        for (uint32_t w = 0; w < p.num_sub; ++w) overflow |= static_cast<uint32_t>(p.cand_count[src * p.num_sub + w]) > p.cand_sub_cap;
        for (uint32_t w = 0; w < p.num_sub; ++w) p.cand_count[src * p.num_sub + w] = 0;
        p.ks_c[i] = p.ks[src];
        p.ke_c[i] = p.ke[src];
        p.tau_c[i] = overflow ? p.tau[src] : -cute::numeric_limits<float>::infinity();
      }
    } else if (tx == 0) {
      const int32_t start = p.ks[p.fallback[n - 1]];
      p.ks_c[i] = start;
      p.ke_c[i] = start;
      p.tau_c[i] = cute::numeric_limits<float>::infinity();
    }
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

template <typename T>
inline T* opt_ptr(tvm::ffi::TensorView t) {
  return t.size(0) > 0 ? static_cast<T*>(t.data_ptr()) : nullptr;
}

// The optional tensors of a score call (empty tensors when unused).
struct ScoreExtras {
  tvm::ffi::TensorView tau_out;       // kModeTau: [T] fp32
  tvm::ffi::TensorView lengths;       // kModeTau: [T] int32
  tvm::ffi::TensorView row_map;       // repair: [T] int32
  tvm::ffi::TensorView num_rows_dev;  // repair: [1] int32
  tvm::ffi::TensorView stats;         // [kNumStats] int64
  int64_t topk;
  int64_t rank;
  int64_t keep_all;
  int64_t ks_div;
};

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
    const ScoreExtras& x,
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
  params.ks_div = static_cast<uint32_t>(x.ks_div);
  RuntimeCheck(x.ks_div >= 1, "fstk: ks_div must be >= 1");
  params.order = order.size(0) > 0 ? static_cast<const int32_t*>(order.data_ptr()) : nullptr;
  if (params.order != nullptr) RuntimeCheck(order.size(0) == params.num_q_blocks, "fstk: order must list every q-block");
  params.stats = opt_ptr<unsigned long long>(x.stats);
  params.row_map = opt_ptr<const int32_t>(x.row_map);
  params.num_rows_dev = opt_ptr<const int32_t>(x.num_rows_dev);
  RuntimeCheck((params.row_map == nullptr) == (params.num_rows_dev == nullptr), "fstk: row_map needs num_rows");
  RuntimeCheck(params.row_map == nullptr or kMode == kModeFilterCompact, "fstk: row_map is for the compacting filter");
  if constexpr (kMode == kModeLogits) {
    params.logits = static_cast<float*>(logits.data_ptr());
    params.logits_stride = static_cast<uint32_t>(logits.stride(0));
  }
  if constexpr (kMode == kModeNoStore) params.sink = static_cast<float*>(logits.data_ptr());
  if constexpr (kMode == kModeFilter or kMode == kModeFilterCompact) {
    constexpr uint32_t kSub = Traits::kBase == 5 ? kNumSubLists5 : kNumMathWarps;
    params.tau = static_cast<const float*>(tau.data_ptr());
    RuntimeCheck(cand_key.size(1) == kSub and cand_count.size(1) == kSub,
                 "fstk: candidate buffers must be [T, ", kSub, ", sub_cap] and [T, ", kSub, "]");
    params.cand_key = static_cast<uint64_t*>(cand_key.data_ptr());
    params.cand_count = static_cast<int32_t*>(cand_count.data_ptr());
    params.cand_sub_cap = static_cast<uint32_t>(cand_key.size(2));
    params.topk = static_cast<uint32_t>(x.topk);
    if constexpr (Traits::kBase == 5)
      RuntimeCheck(x.topk > 0 and params.cand_sub_cap >= params.topk + 64, "fstk: layout 5 needs sub_cap >= topk + 64");
  }
  if constexpr (kMode == kModeTau) {
    RuntimeCheck(x.tau_out.size(0) == num_q and x.lengths.size(0) == num_q, "fstk: tau mode needs tau_out and lengths");
    params.tau_out = static_cast<float*>(x.tau_out.data_ptr());
    params.lengths = static_cast<const int32_t*>(x.lengths.data_ptr());
    params.topk = static_cast<uint32_t>(x.topk);
    params.rank = static_cast<uint32_t>(x.rank);
    params.keep_all = static_cast<uint32_t>(x.keep_all);
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
    const ScoreExtras& x,
    int64_t mode,
    int64_t reuse_kv,
    int64_t num_ctas) {
#define FSTK_SCORE(M, R) \
  score_impl<M, R, kLayout, kAluHeads>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_key, cand_count, x, num_ctas)
#if FSTK_SERVING
  // The serving build instantiates only the threshold, filter and repair passes.
  sglang::host::RuntimeCheck(
      mode == kModeTau or mode == kModeFilter or mode == kModeFilterCompact, "fstk serving: bad mode ", mode);
  if (mode == kModeTau) return FSTK_SCORE(kModeTau, false);
  if (mode == kModeFilterCompact) return FSTK_SCORE(kModeFilterCompact, false);
  return FSTK_SCORE(kModeFilter, false);
#else
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
    case kModeTau:
      if constexpr (LayoutTraits<kLayout>::kBase == 5) return FSTK_SCORE(kModeTau, false);
      [[fallthrough]];
    case kModeFilterCompact:
      if constexpr (LayoutTraits<kLayout>::kBase == 5) return FSTK_SCORE(kModeFilterCompact, false);
      [[fallthrough]];
    default:
      sglang::host::RuntimeCheck(false, "fstk: bad mode ", mode, " for layout ", kLayout);
  }
#endif
#undef FSTK_SCORE
}

struct FusedIndexerTopK {
  // mode: 0 logits, 1 no-epilogue, 2 no-store, 3 filter, 4 tau, 5 compacting filter (4, 5: layout 5);
  // reuse_kv only with
  // mode 1 (calibration). layout: see LayoutTraits; alu_heads: 0 or 16 (dg_reduce).
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
      tvm::ffi::TensorView tau_out,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView row_map,
      tvm::ffi::TensorView num_rows,
      tvm::ffi::TensorView stats,
      int64_t topk,
      int64_t rank,
      int64_t keep_all,
      int64_t ks_div,
      int64_t mode,
      int64_t reuse_kv,
      int64_t layout,
      int64_t alu_heads,
      int64_t num_ctas) {
    const ScoreExtras x{tau_out, lengths, row_map, num_rows, stats, topk, rank, keep_all, ks_div};
#define FSTK_DISPATCH(LY, AH) \
  return score_dispatch<LY, AH>(q, kv, kv_scale, weights, ks, ke, order, logits, tau, cand_key, cand_count, x, mode, reuse_kv, num_ctas)
#if FSTK_SERVING
    sglang::host::RuntimeCheck(
        layout == FSTK_SERVING_LAYOUT and alu_heads == 0, "fstk serving: only layout ", FSTK_SERVING_LAYOUT);
    FSTK_DISPATCH(FSTK_SERVING_LAYOUT, 0);
#else
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
#endif
#undef FSTK_DISPATCH
  }

  // row_list / num_list empty: one CTA per row of out; else the listed rows (grid-stride), with
  // rows that still lack topk candidates counted in stats instead of listed in fallback.
  static void select(
      tvm::ffi::TensorView cand_key,
      tvm::ffi::TensorView cand_count,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView row_to_batch,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView fallback,
      tvm::ffi::TensorView num_fallback,
      tvm::ffi::TensorView row_list,
      tvm::ffi::TensorView num_list,
      tvm::ffi::TensorView stats,
      int64_t page_size,
      int64_t num_ctas,
      int64_t smem_keys) {
    using namespace sglang::host;
    SelectParams p{};
    p.cand_key = static_cast<const uint64_t*>(cand_key.data_ptr());
    p.cand_count = static_cast<const int32_t*>(cand_count.data_ptr());
    p.lengths = static_cast<const int32_t*>(lengths.data_ptr());
    p.row_to_batch = static_cast<const int32_t*>(row_to_batch.data_ptr());
    p.page_table = static_cast<const int32_t*>(page_table.data_ptr());
    p.out = static_cast<int32_t*>(out.data_ptr());
    p.fallback = opt_ptr<int32_t>(fallback);
    p.num_fallback = opt_ptr<int32_t>(num_fallback);
    p.row_list = opt_ptr<const int32_t>(row_list);
    p.num_list = opt_ptr<const int32_t>(num_list);
    p.stats = opt_ptr<unsigned long long>(stats);
    p.page_table_stride = page_table.stride(0);
    p.num_rows = static_cast<uint32_t>(out.size(0));
    p.topk = static_cast<uint32_t>(out.size(1));
    p.num_sub = static_cast<uint32_t>(cand_key.size(1));
    p.cand_sub_cap = static_cast<uint32_t>(cand_key.size(2));
    RuntimeCheck(p.num_sub >= 1 and p.num_sub <= kNumMathWarps, "fstk: 1..8 candidate sub-lists");
    RuntimeCheck((p.fallback == nullptr) == (p.num_fallback == nullptr), "fstk: fallback needs num_fallback");
    RuntimeCheck((p.row_list == nullptr) == (p.num_list == nullptr), "fstk: row_list needs num_list");
    RuntimeCheck(page_size > 0 and (page_size & (page_size - 1)) == 0, "fstk: page_size must be a power of two");
    p.page_bits = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(page_size)));
    if (p.num_rows == 0) return;
    RuntimeCheck(smem_keys >= 1024 and smem_keys <= kSelectRepairKeys and smem_keys % 1024 == 0,
                 "fstk: select smem_keys must be a multiple of 1024 up to 16384");
    p.smem_keys = static_cast<uint32_t>(smem_keys);
    const uint32_t grid = p.row_list != nullptr ? static_cast<uint32_t>(std::min<int64_t>(num_ctas, p.num_rows)) : p.num_rows;
    const size_t smem = static_cast<size_t>(smem_keys) * sizeof(uint64_t);
    const DLDevice device = out.device();
    runtime::init_per_device(device, [=] {
      runtime::set_max_dynamic_smem(fstk_select_kernel, static_cast<size_t>(kSelectRepairKeys) * sizeof(uint64_t));
    });
    LaunchKernel(grid, kSelectThreads, device, smem)(fstk_select_kernel, p);
  }

  static void repair_prep(
      tvm::ffi::TensorView fallback,
      tvm::ffi::TensorView num_fallback,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView ks,
      tvm::ffi::TensorView ke,
      tvm::ffi::TensorView q_c,
      tvm::ffi::TensorView w_c,
      tvm::ffi::TensorView ks_c,
      tvm::ffi::TensorView ke_c,
      tvm::ffi::TensorView tau,
      tvm::ffi::TensorView tau_c,
      tvm::ffi::TensorView cand_count,
      tvm::ffi::TensorView stats,
      int64_t cand_sub_cap,
      int64_t num_ctas) {
    using namespace sglang::host;
    RepairParams p{};
    p.fallback = static_cast<const int32_t*>(fallback.data_ptr());
    p.num_fallback = static_cast<const int32_t*>(num_fallback.data_ptr());
    p.q = static_cast<const uint4*>(q.data_ptr());
    p.q_c = static_cast<uint4*>(q_c.data_ptr());
    p.w = static_cast<const float*>(weights.data_ptr());
    p.w_c = static_cast<float*>(w_c.data_ptr());
    p.w_stride = static_cast<uint32_t>(weights.stride(0));
    p.ks = static_cast<const int32_t*>(ks.data_ptr());
    p.ke = static_cast<const int32_t*>(ke.data_ptr());
    p.ks_c = static_cast<int32_t*>(ks_c.data_ptr());
    p.ke_c = static_cast<int32_t*>(ke_c.data_ptr());
    p.tau = static_cast<const float*>(tau.data_ptr());
    p.tau_c = static_cast<float*>(tau_c.data_ptr());
    p.cand_count = static_cast<int32_t*>(cand_count.data_ptr());
    p.cand_sub_cap = static_cast<uint32_t>(cand_sub_cap);
    p.stats = opt_ptr<unsigned long long>(stats);
    p.rows = static_cast<uint32_t>(q.size(0));
    p.num_sub = static_cast<uint32_t>(cand_count.size(1));
    RuntimeCheck(q.size(1) * q.size(2) == 256 * 16 and q.stride(0) == 256 * 16, "fstk: q rows must be 4096 contiguous bytes");
    RuntimeCheck(q_c.size(0) >= q.size(0) and ks_c.size(0) >= q.size(0), "fstk: compact buffers too small");
    if (p.rows == 0) return;
    LaunchKernel(static_cast<uint32_t>(std::min<int64_t>(num_ctas, p.rows)), kRepairThreads, q.device())(
        fstk_repair_prep_kernel, p);
  }
};

}  // namespace fstk
