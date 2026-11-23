/// \file topk_impl.cuh
/// \brief DeepSeek-V4 (DSA indexer) top-k implementation classes.
///
/// This header holds ONLY the device-side implementation classes + helpers; the
/// `__global__` kernels and the host dispatcher live in csrc/deepseek_v4/topk_v2.cuh.
///
/// Design notes:
///  - top-k (`topk`) is a *runtime* value (<= kMaxTopK = 2048), never a
///    compile-time constant.
///  - the dispatcher optionally transforms selected raw indices through a page
///    table after the device implementation writes them.
///  - each block reads its own `seq_len` (per-batch ragged lengths) -- the host
///    launches one universal kernel and dispatches per block.
///  - the dispatcher selects cluster size 8 or 16 from the probed occupancy.
///
/// Algorithm: fp16 coarse histogram -> threshold bin -> fp32-boundary collect ->
/// exact radix tie-break, plus an exact-key refinement when the threshold bin
/// overflows the staging buffer (refine_ties).

#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>

#if !defined(USE_ROCM)
// currently only apply cluster for SM90 & SM100, SM120 has poor cluster performance
#define SUPPORT_CLUSTER (SGL_CUDA_ARCH >= 900 && SGL_CUDA_ARCH < 1100)
#else
// AMD doesn't support cluster
#define SUPPORT_CLUSTER false
#endif

#if SUPPORT_CLUSTER
#include <cooperative_groups.h>
#endif

#ifndef SGL_TOPK_ONE_CTA_PER_SM  // 1 when an SM holds one 1024-thread block (set at JIT time from the device)
#define SGL_TOPK_ONE_CTA_PER_SM 0
#endif

// 1: the large-tie radix select resolves equal scores to the smallest indices
// (built only into the module the DSA decode context-parallel merge uses)
#ifndef SGL_TOPK_STABLE_TIES
#define SGL_TOPK_STABLE_TIES 0
#endif

namespace sglang {

namespace device::topk {

/// Hints that `value` is warp-uniform so it can live in a uniform register. The
/// caller must already guarantee that: on ROCm this is the identity, since the
/// 32-bit mask below covers only half of a 64-lane wavefront and there is no
/// uniform register file to hint at.
template <typename T>
SGL_DEVICE T broadcast(T value, uint32_t src = 0) {
#if defined(USE_ROCM)
  static_cast<void>(src);
  return value;
#else
  return __shfl_sync(0xFFFFFFFF, value, src);
#endif
}

template <typename... Smems>
struct MaxSmem {
  static constexpr size_t kSize = std::max({sizeof(Smems)...});
  static constexpr size_t kAlign = std::max({alignof(Smems)...});
  alignas(kAlign) uint8_t storage[kSize];
};

// ---------------------------------------------------------------------------
// Order-preserving float -> integer key extraction
// ---------------------------------------------------------------------------

SGL_DEVICE uint32_t extract_exact_bin(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

constexpr float padding_value() {
  return std::numeric_limits<float>::quiet_NaN();
}

constexpr float infinity_value() {
  return std::numeric_limits<float>::infinity();
}

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  uint32_t b = (uint32_t)__half_as_ushort(__float2half_rn(x)) << 16;
  uint32_t s = (uint32_t)((int32_t)b >> 31);
  return (b ^ (s | 0x80000000u)) >> (32 - kBits);
}

/// fp32 bounds {lo, hi} of coarse bin `bin`: "above" iff v >= hi, "tie" iff lo <= v < hi,
/// agreeing with extract_coarse_bin<kBits> on every non-NaN score. A zero hi moves to
/// the smallest subnormal, since fp32 compares cannot put -0.0 below +0.0.
template <uint32_t kBits>
SGL_DEVICE float2 coarse_bin_bounds(uint32_t bin) {
  constexpr uint32_t kShift = 16 - kBits;
  static_assert(kShift >= 1, "edge keys must be even for the rounding rule below");
  constexpr uint32_t kNegInfBin = 0x03FFu >> kShift;  // bin holding the -inf key
  constexpr uint32_t kInfBin = 0xFC00u >> kShift;     // bin holding the +inf key
  // ordered16 edge keys, {lo, hi} packed in the {low, high} halves.
  const uint32_t keys = (bin << kShift) | ((bin + 1) << (kShift + 16));
  // ordered16 -> fp16 bits per half: ~key below 0x8000 (negative), key ^ 0x8000 above.
  const auto to_fp32 = [](uint32_t k) {
    const uint32_t bits = k ^ 0x80008000u ^ ((~k >> 15) & 0x00010001u) * 0x7FFFu;
    return cast<fp32x2_t>(*reinterpret_cast<const fp16x2_t*>(&bits));
  };
  const auto [v_lo, v_hi] = to_fp32(keys);
  const auto [u_lo, u_hi] = to_fp32(keys - 0x00010001u);
  // Exact midpoints of adjacent fp16 values; the clamp puts one against inf on the
  // +/-65520 overflow point.
  const auto midpoint = [](float a, float b) { return fminf(fmaxf(0.5f * (a + b), -65520.0f), 65520.0f); };
  const int32_t lo = __float_as_int(midpoint(v_lo, u_lo));
  const int32_t hi = __float_as_int(midpoint(v_hi, u_hi));
  // Ties round to even and edge keys are even, so negative edges sit one ulp above
  // the midpoint. A zero hi steps past +0.
  const float lo_edge = __int_as_float(lo + (lo >> 31));
  const float hi_edge = __int_as_float(hi + (hi >> 31) + (hi == 0));
  // Edges inside the NaN key space: every score is >= lo, none is >= hi.
  return {bin <= kNegInfBin ? -infinity_value() : lo_edge, bin >= kInfBin ? infinity_value() : hi_edge};
}

SGL_DEVICE uint32_t warp_sum_bool(bool pred, uint32_t mask = 0xFFFFFFFF) {
#ifdef USE_ROCM
  // The ballot covers the whole hardware wave, which on wave64 holds two of
  // these 32-lane logical warps, so a plain __popc would report the wave's
  // lower half to both of them. Shift the caller's mask onto this warp's half
  // and count all 64 bits. __lane_id() / kWarpSize is 0 on wave32.
  const uint32_t half = __lane_id() / kWarpSize;
  return __popcll(__ballot(pred) & (static_cast<uint64_t>(mask) << (kWarpSize * half)));
#else
  return __popc(__ballot_sync(mask, pred));
#endif
}

struct alignas(8) TieValue {
  float value;
  uint32_t idx;
  inline static constexpr TieValue invalid() {
    return TieValue{padding_value(), 0xFFFFFFFFu};
  }
};

// ---------------------------------------------------------------------------
// Per-batch problem description + page-table transform sink
// ---------------------------------------------------------------------------

struct TopKProblem {
  const float* __restrict__ in;
  int32_t* __restrict__ out;  // page_indices [topk]
  uint32_t topk;
  uint32_t seq_len;
  int32_t bias = 0;
  uint32_t input_start = 0;  // needed by ragged mode

  SGL_DEVICE void emit(uint32_t pos, uint32_t raw_idx) const {
    out[pos] = static_cast<int32_t>(raw_idx) + bias;
  }
};

// ---------------------------------------------------------------------------
// Shared configuration + tie handling (exact radix select on the threshold bin)
// ---------------------------------------------------------------------------

struct TopKConfig {
  static constexpr uint32_t kMaxTopK = 2048;
  static constexpr uint32_t kBlockSize = 1024;
  static constexpr uint32_t kOccupancy = 2;
  static constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;
  // kMaxNumTie must be >= kMaxTopK: the collect pass keeps at most kMaxNumTie
  // threshold-bin candidates, and up to `topk` output slots may have to be
  // filled from them (above_count can be 0, e.g. heavily tied or all-inf
  // scores). A smaller cap leaves slots that handle_tie can only pad, and
  // padded slots inside the first min(seq_len, topk) entries are dereferenced
  // by downstream sparse attention.
  static constexpr uint32_t kMaxNumTie = 2048;
  static constexpr uint32_t kRadixSize = 1 << 8;
  // Radix width of the threshold-bin refinement (refine_ties). Wider
  // than kRadixSize so that ceil(32 / kRefineBits) == 3 rounds cover the whole
  // key: the histogram overlays the tie staging buffer, which that path
  // re-derives anyway, so the extra bins are free. 12 is the widest that fits.
  static constexpr uint32_t kRefineBits = 12;
  static constexpr uint32_t kRefineSize = 1 << kRefineBits;
  static constexpr uint32_t kRefineItems = kRefineSize / kBlockSize;
  static_assert(
      kRefineSize * sizeof(uint32_t) <= kMaxNumTie * sizeof(TieValue),
      "the refinement histogram must fit the tie staging buffer it overlays");
  static_assert(kRefineSize % kBlockSize == 0);
  static constexpr uint32_t kTopKItems = (kMaxTopK + kBlockSize - 1) / kBlockSize;
  // tie candidates owned per thread in the strided handle_tie loops
  static constexpr uint32_t kTieItems = kMaxNumTie / kBlockSize;
  static constexpr bool kStableTies = SGL_TOPK_STABLE_TIES;
  static_assert(kMaxNumTie >= kMaxTopK && kMaxNumTie % kBlockSize == 0 && kBlockSize % kNumWarps == 0);

  struct TieHandleSmem {
    struct MatchBin {
      uint32_t bin;
      uint32_t above_count;
      uint32_t equal_count;
    };
    uint32_t counter;
    uint32_t counter_final;
    MatchBin match;
    uint32_t warp_sum[kNumWarps];
    uint32_t histogram[2][kRadixSize];
  };

  /// Resolve the threshold bin's ties exactly. `base` is the number of strictly
  /// "above" elements already emitted (final output starts at slot `base`);
  /// `topk` here is the number of remaining slots to fill (== global_topk - base).
  SGL_DEVICE static void handle_tie(  //
      const TieValue* tie_buffer,
      const TopKProblem& problem,
      const uint32_t base,
      const uint32_t num_ties,
      const uint32_t topk,
      TieHandleSmem* smem) {
    constexpr auto is_greater = [](const TieValue& a, const TieValue& b) {
      return (a.value > b.value) || (a.value == b.value && a.idx < b.idx);
    };
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);
    static_assert(kNumWarps == kWarpSize);

    if (num_ties <= topk) {
      for (uint32_t t = tx; t < num_ties; t += kBlockSize) {
        problem.emit(base + t, tie_buffer[t].idx);
      }
      for (uint32_t t = num_ties + tx; t < topk; t += kBlockSize) {
        problem.emit(base + t, base + t);
      }
    } else if (num_ties <= kWarpSize) {
      if (lane_id >= num_ties || warp_id >= num_ties) return;  // some threads are idle
      /// NOTE: use long long to avoid mask overflow when num_tie == 32
      const uint32_t mask = (1ull << num_ties) - 1u;
      const auto tie = tie_buffer[lane_id];
      const auto target = tie_buffer[warp_id];
      const auto rank = warp_sum_bool(is_greater(tie, target), mask);
      if (lane_id == 0 && rank < topk) problem.emit(base + rank, target.idx);
    } else if (num_ties <= kWarpSize * 2) {
      // 64 x 64 topk implementation: each thread takes 2 elements
      const auto warp_id_0 = warp_id;
      const auto warp_id_1 = warp_id + kWarpSize;
      const auto lane_id_1 = lane_id + kWarpSize;
      const auto invalid = TieValue::invalid();
      const auto tie_0 = tie_buffer[lane_id];
      const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
      const auto target_0 = tie_buffer[warp_id_0];
      const auto target_1 = tie_buffer[warp_id_1];
      if (true) {  // NOTE: warp_id_0 <= kNumWarps < num_ties
        const auto rank_0 = warp_sum_bool(is_greater(tie_0, target_0));
        const auto rank_1 = warp_sum_bool(is_greater(tie_1, target_0));
        const auto rank = rank_0 + rank_1;
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target_0.idx);
      }
      if (warp_id_1 < num_ties) {
        const auto rank_0 = warp_sum_bool(is_greater(tie_0, target_1));
        const auto rank_1 = warp_sum_bool(is_greater(tie_1, target_1));
        const auto rank = rank_0 + rank_1;
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target_1.idx);
      }
    } else if (num_ties <= kWarpSize * 4) {
      // 128 x 128 topk implementation: each thread takes 4 elements and does local sort + merge
      const auto invalid = TieValue::invalid();
      const TieValue tie[] = {
          tie_buffer[lane_id + 0 * kWarpSize],
          tie_buffer[lane_id + 1 * kWarpSize],
          lane_id + 2 * kWarpSize < num_ties ? tie_buffer[lane_id + 2 * kWarpSize] : invalid,
          lane_id + 3 * kWarpSize < num_ties ? tie_buffer[lane_id + 3 * kWarpSize] : invalid,
      };
      const TieValue target[] = {
          tie_buffer[warp_id + 0 * kWarpSize],
          tie_buffer[warp_id + 1 * kWarpSize],
          tie_buffer[warp_id + 2 * kWarpSize],
          tie_buffer[warp_id + 3 * kWarpSize],
      };
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= 2 && warp_id + i * kWarpSize >= num_ties) break;
        uint32_t rank = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          rank += warp_sum_bool(is_greater(tie[j], target[i]));
        }
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target[i].idx);
      }
    } else if (num_ties <= kBlockSize) {
      // Common case: one candidate per thread.
      return radix_tie_select<1>(tie_buffer, problem, base, num_ties, topk, smem);
    } else {
      // Rare overflow case.
      static_assert(kTieItems == 2);
      return radix_tie_select<2>(tie_buffer, problem, base, num_ties, topk, smem);
    }
  }

  /// Threshold-digit search over the kRefineSize-bin refinement histogram in
  /// `hist`, by the same rule radix_tie_select uses. One thread owns kRefineItems
  /// contiguous bins. Publishes the result in `smem->match`. Block-wide; ends
  /// with a barrier.
  SGL_DEVICE static void refine_find_threshold(  //
      const uint32_t* hist,
      const uint32_t total_active,
      const uint32_t topk_remain,
      TieHandleSmem* smem) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);

    uint32_t counts[kRefineItems];
    uint32_t local_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      counts[i] = hist[tx * kRefineItems + i];
      local_sum += counts[i];
    }
    const auto warp_inc = warp::inclusive_sum(local_sum, lane_id);
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
    __syncthreads();

    // Inclusive prefix over all bins below this thread's first one.
    uint32_t prefix = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
    prefix += warp_inc - local_sum;
#pragma unroll
    for (uint32_t i = 0; i < kRefineItems; ++i) {
      prefix += counts[i];
      const auto above = total_active - prefix;  // active elements in bins ABOVE this one
      if (above < topk_remain && above + counts[i] >= topk_remain) {
        smem->match = {tx * kRefineItems + i, above, counts[i]};
      }
    }
    __syncthreads();
  }

  /// Exact radix select over the tie candidates: each thread owns kItems
  /// strided elements (inactive beyond num_ties). Requires
  /// num_ties <= kItems * kBlockSize. Rounds 0-3 select on the score; equal
  /// scores at the k-th place fill the last slots arbitrarily, unless
  /// kStableTies: then, while they still outnumber the free slots, rounds 4-7
  /// select on the inverted index, so they resolve to the smallest indices (the
  /// `is_greater` order of the small-tie paths).
  template <uint32_t kItems>
  SGL_DEVICE static void radix_tie_select(  //
      const TieValue* tie_buffer,
      const TopKProblem& problem,
      const uint32_t base,
      const uint32_t num_ties,
      const uint32_t topk,
      TieHandleSmem* smem) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);

    bool active[kItems];
    uint32_t key[kItems];
    uint32_t idx[kItems];
    uint32_t write_pos[kItems];
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      const auto t = tx + i * kBlockSize;
      active[i] = t < num_ties;
      const auto tie = active[i] ? tie_buffer[t] : TieValue::invalid();
      key[i] = extract_exact_bin(tie.value);
      idx[i] = tie.idx;
      write_pos[i] = topk;
    }
    uint32_t topk_remain = topk;
    if (tx < kRadixSize) smem->histogram[0][tx] = 0;
    if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
    __syncthreads();
    uint32_t total_active = num_ties;

    constexpr int kRounds = kStableTies ? 8 : 4;
#pragma unroll
    for (int round = 0; round < kRounds; round++) {
      const uint32_t shift = 24 - (round % 4) * 8;
      const auto hist_idx = round % 2;
      const auto histogram = smem->histogram[hist_idx];
      const auto radix_key = [&](uint32_t i) { return round < 4 ? key[i] : ~idx[i]; };

#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (active[i]) atomicAdd(&histogram[(radix_key(i) >> shift) & 0xFFu], 1);
      }
      if (round < kRounds - 1 && tx < kRadixSize) {
        smem->histogram[hist_idx ^ 1][tx] = 0;
      }
      __syncthreads();

      uint32_t hist_val = 0;
      uint32_t warp_inc = 0;
      if (tx < kRadixSize) {
        hist_val = histogram[tx];
        warp_inc = warp::inclusive_sum(hist_val, lane_id);
        if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
      }
      __syncthreads();
      if (tx < kRadixSize) {
        const auto inter = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
        const auto prefix = inter + warp_inc;      // inclusive prefix through this bin
        const auto above = total_active - prefix;  // elements in bins ABOVE this one
        // 3. Find threshold bin
        if (above < topk_remain && above + hist_val >= topk_remain) {
          smem->match = {tx, above, hist_val};
        }
      }
      __syncthreads();

      const auto [threshold_bin, above_count, equal_count] = smem->match;
      total_active = equal_count;
      topk_remain -= above_count;
      // Stable: once the whole threshold bin fits, take it (round 7 always ends
      // here since indices are unique). Otherwise round 3 takes it as is.
      const bool last = kStableTies ? equal_count == topk_remain : round == kRounds - 1;

      // 4. Scatter
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (!active[i]) continue;
        const uint32_t bin = (radix_key(i) >> shift) & 0xFFu;
        if (bin > threshold_bin) {
          write_pos[i] = atomicAdd(&smem->counter, 1);
          active[i] = false;
        } else if (bin < threshold_bin) {
          active[i] = false;
        } else if (last) {
          write_pos[i] = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
        }
        // my_bin == thr && !last: stay active for next round
      }

      if (last || topk_remain == 0) break;
    }

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      if (write_pos[i] < topk) problem.emit(base + write_pos[i], idx[i]);
    }
  }
};

// ---------------------------------------------------------------------------
// Radix base: histogram storage + input iteration + threshold-bin search
// ---------------------------------------------------------------------------

template <uint32_t kHistBits_>
struct TopKRadixBase : TopKConfig {
 public:
  static constexpr uint32_t kVecSize = 4;
  static constexpr uint32_t kHistBits = kHistBits_;
  static constexpr uint32_t kHistSize = 1 << kHistBits;
  using vec_t = AlignedVector<float, kVecSize>;

  struct Smem {
    uint32_t count_eq;
    uint32_t count_gt;
    float v_hi;
    float v_lo;
    union {
      uint32_t warp_sum[kNumWarps];
      uint32_t tie_nonzero;
    };
    // The coarse histogram is dead once find_threshold() has published its
    // boundaries, and the tie machinery only comes alive after that, so the two
    // phases overlay. tie_handle and tie_values are live TOGETHER, so they sit
    // side by side inside the overlay, not in a union with each other.
    union {
      alignas(16) uint32_t histogram[kHistSize];
      struct {
        // refine_ties re-derives the candidates instead of trusting
        // what the collect pass staged, so the buffer is free to carry that
        // path's radix histogram until its staging pass writes values back.
        union {
          TieValue tie_values[kMaxNumTie];
          uint32_t refine_hist[kRefineSize];
        };
        TieHandleSmem tie_handle;
      };
    };
  };

 protected:
  SGL_DEVICE static bool is_zero_bin(float lower) {
#ifdef USE_ROCM
    return __float_as_uint(lower) == 0u;
#else
    return false;
#endif
  }

  template <uint32_t N = 1, typename F>
  SGL_DEVICE static void for_each_input(const float* __restrict__ in, uint32_t seq_len, F&& fn) {
    constexpr auto kStride = N * kBlockSize;
    const auto tx = threadIdx.x;
    const auto num_full = seq_len / kVecSize;  // fully-in-bounds vectors
    const auto kChunk = 128u;
    // lane | rank | warp
    auto vi = N == 1 ? tx : (tx % kChunk) + blockIdx.y * kChunk + (tx / kChunk) * (N * kChunk);
    if (vi < num_full) {
      vec_t next_vec;
      next_vec.load(in, vi);
#pragma unroll 1
      do {
        const auto cur = next_vec;
        vi += kStride;
        if (vi < num_full) next_vec.load(in, vi);
        const auto base = (vi - kStride) * kVecSize;
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          fn(cur[j], base + j);
        }
      } while (vi < num_full);
    }

    if (vi == num_full) {
      const auto base = vi * kVecSize;
      if (base == seq_len) return;
      vec_t cur;
      cur.load(in, vi);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        if (base + j < seq_len) fn(cur[j], base + j);
      }
    }
  }

  template <uint32_t kSize>
  SGL_DEVICE static void init_histogram(uint32_t (&histogram)[kSize], uint32_t tx) {
    static_assert(kSize % kBlockSize == 0);
    constexpr uint32_t kItems = kSize / kBlockSize;
    AlignedVector<uint32_t, kItems> vec;
    vec.fill(0);
    vec.store(histogram, tx);
  }

  /// Same, but scanning a histogram that need not be `smem`'s own -- the cluster
  /// path merges into one rank's copy and scans it there.
  template <typename Smem, typename Fn>
  SGL_DEVICE static void find_threshold(const uint32_t topk, const uint32_t seq_len, Smem* smem, Fn fn) {
    const auto tx = threadIdx.x;
    // the histogram's own size: the cluster path may bin finer than this base
    constexpr uint32_t kItems = sizeof(Smem::histogram) / sizeof(uint32_t) / kBlockSize;
    uint32_t local_exc_sum[kItems + 1];
    AlignedVector<uint32_t, kItems> hist_vec;
    hist_vec.load(smem->histogram, tx);

    local_exc_sum[0] = 0;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      local_exc_sum[i + 1] = hist_vec[i] + local_exc_sum[i];
    }

    const auto local_sum = local_exc_sum[kItems];
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = broadcast(tx / kWarpSize);
    const auto warp_inc_sum = warp::inclusive_sum(local_sum, lane_id);
    const auto warp_exc_sum = warp_inc_sum - local_sum;
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc_sum;

    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    const auto warp_prefix_sum = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
    const auto exc_sum = static_cast<int32_t>(warp_prefix_sum + warp_exc_sum);
    const auto remained = static_cast<int32_t>(seq_len - topk - exc_sum);
    // only 1 lane will execute this
    if (remained >= 0 && remained < static_cast<int32_t>(local_sum)) [[unlikely]] {
      uint32_t target = 0;
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        const auto prev = static_cast<int32_t>(local_exc_sum[i + 0]);
        const auto next = static_cast<int32_t>(local_exc_sum[i + 1]);
        if (remained >= prev && remained < next) target = tx * kItems + i;
      }
      fn(target);
    }

    __syncthreads();
  }

  /// Re-stages the tie buffer exactly when the threshold bin overflows it: radix
  /// rounds narrow the bin to one bucket, then a final scan emits what lies above
  /// it and stages the bucket for the caller's handle_tie.
  SGL_DEVICE static void refine_ties(  //
      const TopKProblem& problem,
      Smem* smem,
      const float v_lo,
      const float v_hi,
      const uint32_t count_eq) {
    constexpr uint32_t kRefineRounds = (32 + kRefineBits - 1) / kRefineBits;
    const auto topk = problem.topk;
    if (smem->count_gt >= topk) [[unlikely]]
      return;
    const auto tx = threadIdx.x;
    const auto handle = &smem->tie_handle;
    const auto hist = smem->refine_hist;
    // for_each_input's elements with one copy of `fn` and reversed lanes, so this
    // cold scan shares no load address (hence no registers) with the fast path.
    const auto scan = [&](auto&& fn) {
      const auto seq_len = problem.seq_len;
      const auto num_vec = (seq_len + kVecSize - 1) / kVecSize;
      uint32_t vi = kBlockSize - 1 - tx;
      vec_t cur;
      if (vi < num_vec) cur.load(problem.in, vi);
#pragma unroll 1
      while (vi < num_vec) {
        const auto next_vi = vi + kBlockSize;
        vec_t next;
        if (next_vi < num_vec) next.load(problem.in, next_vi);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          if (vi * kVecSize + j < seq_len) fn(cur[j], vi * kVecSize + j);
        }
        cur = next;
        vi = next_vi;
      }
    };

    // refine_find_threshold needs 0 < remain <= active; its match rule keeps it.
    uint32_t remain = topk - smem->count_gt;
    uint32_t active = count_eq;
    // Candidates are [bucket_lo, bucket_lo + bucket_mask]; round 0 takes the whole bin.
    uint32_t bucket_lo = 0;
    uint32_t bucket_mask = ~0u;

#pragma unroll
    for (uint32_t round = 0; round < kRefineRounds; ++round) {
      // round 0: bits [20, 32), round 1: bits [8, 20), round 2: bits [0, 8)
      const uint32_t lo = (round + 1) * kRefineBits < 32 ? 32 - (round + 1) * kRefineBits : 0;
#pragma unroll
      for (uint32_t i = 0; i < kRefineItems; ++i) {
        hist[tx * kRefineItems + i] = 0;
      }
      __syncthreads();
      // bucket_lo is aligned to the previous digit, so (key - bucket_lo) >> lo is this digit.
      scan([&](float val, uint32_t) {
        if (!(val >= v_lo && val < v_hi)) return;
        const auto d = extract_exact_bin(val) - bucket_lo;
        if (d <= bucket_mask) atomicAdd(&hist[d >> lo], 1);
      });
      __syncthreads();
      refine_find_threshold(hist, active, remain, handle);
      const auto match = handle->match;
      bucket_lo |= match.bin << lo;
      bucket_mask = (1u << lo) - 1u;
      remain -= match.above_count;
      active = match.equal_count;
      if (lo == 0 || active <= kMaxNumTie) break;
    }

    // Candidates are never NaN, so keys stay <= key(+inf) and bucket_hi cannot wrap.
    // Staging overwrites the histogram, which refine_find_threshold has consumed.
    const auto bucket_hi = bucket_lo + bucket_mask + 1u;
    if (tx == 0) smem->count_eq = 0;
    __syncthreads();
    scan([&](float val, uint32_t idx) {
      if (!(val >= v_lo && val < v_hi)) return;
      const auto key = extract_exact_bin(val);
      if (key >= bucket_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]]
          problem.emit(pos, idx);
      } else if (key >= bucket_lo) {
        const auto slot = atomicAdd(&smem->count_eq, 1);
        if (slot < kMaxNumTie) smem->tie_values[slot] = {val, idx};
      }
    });
    __syncthreads();
  }
};

// ---------------------------------------------------------------------------
// Register path: scores stay resident in registers across both passes (read
// once). Templated on kLocalVecs so the caller picks the smallest covering
// kernel -- a larger kLocalVecs raises kMaxSeqLen but its fixed-unrolled loop
// wastes work on shorter sequences.
// ---------------------------------------------------------------------------

template <uint32_t kLocalVecs_>
struct TopKRegister : TopKRadixBase<12> {
  static constexpr uint32_t kLocalVecs = kLocalVecs_;
  static constexpr uint32_t kMaxSeqLen = kBlockSize * kVecSize * kLocalVecs;
  using Smem = typename TopKRadixBase<12>::Smem;

  template <bool kUsePDL>
  SGL_DEVICE static void forward(const TopKProblem& problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }

    __syncthreads();
    PDLWaitPrimary<kUsePDL>();
    const uint32_t num_full = div_ceil(problem.seq_len, kVecSize);

    // Phase 1: load full vectors + build histogram
    vec_t local_vecs[kLocalVecs];
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi < num_full) local_vecs[i].load(problem.in, vi);
    }

    const auto tail_start = (problem.seq_len - 1) % kVecSize + 1;
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
      if (vi == num_full - 1) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          if (j >= tail_start) local_vecs[i][j] = padding_value();
        }
      }
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(local_vecs[i][j])], 1);
      }
    }
    const auto num_padding = kVecSize - tail_start + problem.input_start;
    if (tx == 0 && num_padding > 0) {
      // Ask the histogram's own binning where the platform's NaN landed.
      atomicSub(&smem->histogram[extract_coarse_bin<kHistBits>(padding_value())], num_padding);
      atomicAdd(&smem->histogram[0], num_padding);
    }
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, num_full * kVecSize, smem, [&](uint32_t threshold_bin) {
      const auto [v_lo, v_hi] = coarse_bin_bounds<kHistBits>(threshold_bin);
      smem->v_hi = v_hi;
      smem->v_lo = v_lo;
    });

    // Phase 3: collect by two fp32 boundaries
    const auto topk = problem.topk;
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    const bool zero_bin = is_zero_bin(v_lo);
    if (zero_bin) {
      // find_threshold has finished using the overlaid warp sums.
      if (tx == 0) smem->tie_nonzero = 0;
      __syncthreads();
    }

#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      const auto base = vi * kVecSize;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const auto idx = base + j;
        const auto val = local_vecs[i][j];
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            problem.emit(pos, idx);
          }
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (zero_bin && __float_as_uint(val) != 0u) atomicOr(&smem->tie_nonzero, 1u);
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      }
    }

    // Phase 4: Handle ties.
    __syncthreads();
    if (smem->count_eq > kMaxNumTie && (!zero_bin || smem->tie_nonzero != 0)) [[unlikely]] {
      // Buffer holds an arrival-order subset; re-stage it on the exact key.
      refine_ties(problem, smem, v_lo, v_hi, smem->count_eq);
    }
    const auto count_gt = smem->count_gt;
    const auto count_eq = smem->count_eq;
    const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
    const auto tie_count = min(count_eq, kMaxNumTie);
    handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
  }
};

// ---------------------------------------------------------------------------
// Streaming path: seq_len > 8192 -- two vectorized passes over global memory
// ---------------------------------------------------------------------------

struct TopKStreaming : TopKRadixBase<12> {
 public:
  static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();

  template <bool kUsePDL>
  SGL_DEVICE static void forward(TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }
    __syncthreads();
    PDLWaitPrimary<kUsePDL>();

    // Phase 1: Load and build histogram
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t) {
      const auto bin = extract_coarse_bin<kHistBits>(val);
      atomicAdd(&smem->histogram[bin], 1);
    });
    const auto num_padding = problem.input_start;
    if (tx == 0 && num_padding != 0) {
      atomicSub(&smem->histogram[extract_coarse_bin<kHistBits>(padding_value())], num_padding);
      atomicAdd(&smem->histogram[0], num_padding);
    }
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, problem.seq_len, smem, [&](uint32_t threshold_bin) {
      const auto [v_lo, v_hi] = coarse_bin_bounds<kHistBits>(threshold_bin);
      smem->v_hi = v_hi;
      smem->v_lo = v_lo;
    });

    // Phase 3: Collect candidates and sort. Classify by two fp32 boundaries derived
    // from the threshold bin instead of recomputing the fp16 bin per element: an
    // element is "above" iff val >= v_hi (bin > threshold) and a "tie" iff
    // v_lo <= val < v_hi (bin == threshold). This drops the F2F + bit-twiddle from
    // the second full pass over the input.
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    const auto topk = problem.topk;
    const bool zero_bin = is_zero_bin(v_lo);
    if (zero_bin) {
      if (tx == 0) smem->tie_nonzero = 0;
      __syncthreads();
    }
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]] {
          problem.emit(pos, idx);
        }
      } else if (val >= v_lo) {
        const auto pos = atomicAdd(&smem->count_eq, 1);
        if (zero_bin && __float_as_uint(val) != 0u) atomicOr(&smem->tie_nonzero, 1u);
        if (pos < kMaxNumTie) [[likely]] {
          smem->tie_values[pos] = {val, idx};
        }
      }
    });

    // Phase 4: Handle ties. Drive the output layout from the *collect* counts so it
    // is self-consistent with the fp32 classification above (rather than the fp16
    // histogram counts), even if rounding moves a boundary element between the
    // "above" and "tie" sets. above_count is < topk by the threshold-bin invariant,
    // so the count_gt guard above effectively never triggers.
    __syncthreads();
    if (smem->count_eq > kMaxNumTie && (!zero_bin || smem->tie_nonzero != 0)) [[unlikely]] {
      // See the register path.
      refine_ties(problem, smem, v_lo, v_hi, smem->count_eq);
    }
    const auto count_gt = smem->count_gt;
    const auto count_eq = smem->count_eq;
    const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
    const auto tie_count = min(count_eq, kMaxNumTie);
    handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
  }
};

#if !defined(USE_ROCM)

// ---------------------------------------------------------------------------
// Sampled-threshold path (CUDA only): a 1/8 sample of the row picks a threshold
// theta, one pass collects every score >= theta into shared memory, and
// TopKStreaming's selection (coarse histogram, threshold bin, handle_tie) runs on
// that buffer. Exact whenever topk <= #{score >= theta} <= kCapacity: the k-th
// largest score is then >= theta.
// ---------------------------------------------------------------------------

struct TopKSampled : TopKRadixBase<12> {
  using Base = TopKRadixBase<12>;
  // A 1/8 sample: 2 adjacent vectors (one 32-byte sector) of every 16.
  static constexpr uint32_t kSampleShift = 3;
  static constexpr uint32_t kSampleGroup = 2u << kSampleShift;
  static constexpr uint32_t kUnroll = 4;  // vectors in flight per thread
  // collect: 8 in flight when the block has the SM to itself (the kernel then runs (1024, 1), up to 64 registers)
  static constexpr uint32_t kCollectUnroll = SGL_TOPK_ONE_CTA_PER_SM ? 8 : 4;
  // Sampled scores >= theta: topk * 9/64 + kSlack, >= 4.8 sigma over topk / 8 for topk <= 2048.
  static constexpr uint32_t kSlack = 40;
  // Sampled scores the threshold bin may add past the target before it is cut finer; keeps
  // #{score >= theta} about 5 sigma under kCapacity at topk 2048.
  static constexpr uint32_t kRefineSlack = 32;
  // Fits the 48 KiB static shared memory next to the fallback path's; see SampledCandidatesSmem.
  static constexpr uint32_t kCapacity = 3584;
  static constexpr uint32_t kCheckEvery = 4;  // collect rounds between overflow checks

  struct Smem : Base::Smem {
    uint32_t n_ge;     // scores >= theta, also past kCapacity
    uint32_t bin;      // coarse bin of the sample's target-th largest score
    uint32_t s_above;  // sampled scores in coarser bins above it
    float theta;
    union {
      TieValue cand[kCapacity];
      int32_t staged[kMaxTopK];  // the tie slots' indices, once the candidates are consumed
    };
  };
  static_assert(sizeof(TieValue) * kCapacity >= sizeof(int32_t) * kMaxTopK);

  /// Emits the top-k above the threshold bin as `emit_above(slot, score, index)` and leaves
  /// the rest as indices (+ problem.bias) in smem->staged[first_staged, topk); returns false,
  /// having emitted nothing, when theta missed.
  /// Bench only (kStamp): thread 0 takes the globaltimer into stamps[k] at a phase boundary (after a barrier).
  template <bool kStamp>
  SGL_DEVICE static void stamp(uint64_t* stamps, const uint32_t k) {
    if constexpr (kStamp) {
      if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamps[k]));
    }
  }

  /// kStamp (bench only): stamps[0..6] = entry, PDL wait done, theta known, collected, select threshold
  /// known, above emitted, ties handled; the production instantiations use kStamp = false.
  template <bool kUsePDL, typename EmitAbove, bool kStamp = false>
  SGL_DEVICE static bool forward(
      const TopKProblem& problem,
      Smem* smem,
      const EmitAbove& emit_above,
      uint32_t& first_staged,
      uint64_t* stamps = nullptr) {
    const auto tx = threadIdx.x;
    stamp<kStamp>(stamps, 0);
    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_gt = 0;
      smem->count_eq = 0;
      smem->n_ge = 0;
      smem->bin = 0;  // kept when the sample is too small: theta -inf, every score qualifies
      smem->s_above = 0;
    }
    __syncthreads();
    PDLWaitPrimary<kUsePDL>();
    stamp<kStamp>(stamps, 1);

    const float theta = sample_threshold(problem, smem);
    stamp<kStamp>(stamps, 2);
    init_histogram(smem->histogram, tx);  // for the candidates; sample_threshold ends on a barrier
    collect(problem, theta, smem);
    __syncthreads();
    stamp<kStamp>(stamps, 3);
    const auto n = smem->n_ge;
    if (n < problem.topk || n > kCapacity) return false;
    select_candidates<kUsePDL, EmitAbove, kStamp>(problem, theta, n, smem, emit_above, first_staged, stamps);
    return true;
  }

 private:
  /// Full vectors sampled, and the vector of sample m.
  SGL_DEVICE static uint32_t num_sampled_vectors(const uint32_t seq_len) {
    const auto num_full = seq_len / kVecSize;
    return num_full / kSampleGroup * 2 + min(num_full % kSampleGroup, 2u);
  }
  SGL_DEVICE static uint32_t sampled_vector(const uint32_t m) {
    return m / 2 * kSampleGroup + m % 2;
  }

  /// `fn(score)` on the sampled vectors, without the input_start head.
  template <typename F>
  SGL_DEVICE static void for_each_sample(const TopKProblem& problem, const uint32_t num_sampled, F&& fn) {
#pragma unroll 1
    for (uint32_t m0 = threadIdx.x; m0 < num_sampled; m0 += kUnroll * kBlockSize) {
      vec_t v[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        if (const auto m = m0 + u * kBlockSize; m < num_sampled) v[u].load(problem.in, sampled_vector(m));
      }
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        const auto m = m0 + u * kBlockSize;
        if (m >= num_sampled) break;
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          if (sampled_vector(m) * kVecSize + j >= problem.input_start) fn(v[u][j]);
        }
      }
    }
  }

  /// The lower edge of the coarse bin holding the sample's target-th largest score; when
  /// that bin would overfill the buffer, a second look at the sample cuts it into
  /// kHistSize equal slices and theta is the edge of the slice holding that score.
  SGL_DEVICE static float sample_threshold(const TopKProblem& problem, Smem* smem) {
    constexpr uint32_t kHistItems = kHistSize / kBlockSize;
    const auto tx = threadIdx.x;
    const auto num_sampled = num_sampled_vectors(problem.seq_len);
    const auto num_samples = num_sampled * kVecSize - problem.input_start;  // vector 0 holds the head
    const uint32_t target = ((problem.topk + (problem.topk >> 3)) >> kSampleShift) + kSlack;
    for_each_sample(
        problem, num_sampled, [&](float val) { atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(val)], 1); });
    __syncthreads();
    find_threshold(target, num_samples, smem, [&](uint32_t bin) { smem->bin = bin; });

    const auto bin = smem->bin;
    const auto s_bin = smem->histogram[bin];
    const float2 bounds = coarse_bin_bounds<kHistBits>(bin);
    const float lo = bounds.x;
    const float hi = bounds.y;
    uint32_t above = 0;
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      if (const auto b = tx * kHistItems + i; b > bin) above += smem->histogram[b];
    }
    above = warp::reduce_sum(above);
    if (tx % kWarpSize == 0 && above != 0) atomicAdd(&smem->s_above, above);
    __syncthreads();  // also ends the histogram reads: the buffer overwrites it next
    const auto s_above = smem->s_above;
    const bool unbounded = lo == -infinity_value() || hi == infinity_value();
    if (s_above + s_bin <= target + kRefineSlack || unbounded) return lo;

    init_histogram(smem->histogram, tx);
    if (tx == 0) smem->theta = lo;
    __syncthreads();
    const float scale = kHistSize / (hi - lo);
    for_each_sample(problem, num_sampled, [&](float val) {
      if (val >= lo && val < hi) {
        atomicAdd(&smem->histogram[min(static_cast<uint32_t>((val - lo) * scale), kHistSize - 1)], 1);
      }
    });
    __syncthreads();
    // s_above < target <= s_above + s_bin, so the slice exists
    find_threshold(
        target - s_above, s_bin, smem, [&](uint32_t slice) { smem->theta = lo + static_cast<float>(slice) / scale; });
    return smem->theta;
  }

  /// Appends every score >= theta to smem->cand (warp-aggregated), counting all of them in n_ge.
  SGL_DEVICE static void collect(const TopKProblem& problem, const float theta, Smem* smem) {
    if constexpr (kCollectUnroll == 8) {
      collect_masked(problem, theta, smem);
    } else {
      collect_per_score(problem, theta, smem);
    }
  }

  /// collect for kCollectUnroll == 8 (one block per SM): the compares pack into a bit mask and only the set bits are
  /// written, each value re-read from L1 (per-score write blocks made the loop issue-bound; a register array indexed
  /// by a runtime bit would go to local memory).
  SGL_DEVICE static void collect_masked(const TopKProblem& problem, const float theta, Smem* smem) {
    constexpr uint32_t kU = 8;
    constexpr uint32_t kStride = kU * kBlockSize;
    static_assert(kU * kVecSize <= 32, "one mask bit per score of a round");
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto num_full = problem.seq_len / kVecSize;
    // warp-uniform trip count: the warp scan below needs every lane
#pragma unroll 1
    for (uint32_t base = 0; base < num_full; base += kStride) {
      vec_t v[kU];
      if (base + kStride <= num_full) {  // block-uniform: a full round needs no bounds
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) v[u].load(problem.in, base + u * kBlockSize + tx);
      } else {
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
          if (const auto vi = base + u * kBlockSize + tx; vi < num_full) {
            v[u].load(problem.in, vi);
          } else {
            v[u].fill(padding_value());  // NaN is never >= theta
          }
        }
      }
      uint32_t mask = 0;
#pragma unroll
      for (uint32_t u = 0; u < kU; ++u) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          mask |= (v[u][j] >= theta ? 1u : 0u) << (u * kVecSize + j);
        }
      }
      const uint32_t count = __popc(mask);
      const auto inc = warp::inclusive_sum(count, lane_id);
      uint32_t pos = 0;
      if (lane_id == kWarpSize - 1 && inc != 0) pos = atomicAdd(&smem->n_ge, inc);
      pos = __shfl_sync(0xFFFFFFFFu, pos, kWarpSize - 1) + inc - count;
#pragma unroll 1
      for (; mask != 0; mask &= mask - 1, ++pos) {
        const auto k = static_cast<uint32_t>(__ffs(mask)) - 1u;
        const auto idx = (base + k / kVecSize * kBlockSize + tx) * kVecSize + k % kVecSize;
        if (pos < kCapacity) smem->cand[pos] = {problem.in[idx], idx};
      }
      // the row is redone on the exact path anyway
      const bool check = base / kStride % kCheckEvery == kCheckEvery - 1;
      if (check && broadcast(*reinterpret_cast<volatile uint32_t*>(&smem->n_ge)) > kCapacity) break;
    }
    if (const auto idx = num_full * kVecSize + tx; idx < problem.seq_len) {
      const auto val = problem.in[idx];
      if (val >= theta) {
        const auto pos = atomicAdd(&smem->n_ge, 1);
        if (pos < kCapacity) smem->cand[pos] = {val, idx};
      }
    }
  }

  /// collect for kCollectUnroll == 4 (two blocks per SM): one predicated write per score.
  SGL_DEVICE static void collect_per_score(const TopKProblem& problem, const float theta, Smem* smem) {
    constexpr uint32_t kStride = kUnroll * kBlockSize;
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto num_full = problem.seq_len / kVecSize;
    // warp-uniform trip count: the warp scan below needs every lane
#pragma unroll 1
    for (uint32_t base = 0; base < num_full; base += kStride) {
      vec_t v[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        if (const auto vi = base + u * kBlockSize + tx; vi < num_full) {
          v[u].load(problem.in, vi);
        } else {
          v[u].fill(padding_value());  // NaN is never >= theta
        }
      }
      uint32_t count = 0;
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          count += v[u][j] >= theta ? 1u : 0u;
        }
      }
      const auto inc = warp::inclusive_sum(count, lane_id);
      uint32_t pos = 0;
      if (lane_id == kWarpSize - 1 && inc != 0) pos = atomicAdd(&smem->n_ge, inc);
      pos = __shfl_sync(0xFFFFFFFFu, pos, kWarpSize - 1) + inc - count;
      if (count != 0) {
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
#pragma unroll
          for (uint32_t j = 0; j < kVecSize; ++j) {
            if (v[u][j] >= theta) {
              if (pos < kCapacity) smem->cand[pos] = {v[u][j], (base + u * kBlockSize + tx) * kVecSize + j};
              ++pos;
            }
          }
        }
      }
      // the row is redone on the exact path anyway
      const bool check = base / kStride % kCheckEvery == kCheckEvery - 1;
      if (check && broadcast(*reinterpret_cast<volatile uint32_t*>(&smem->n_ge)) > kCapacity) break;
    }
    if (const auto idx = num_full * kVecSize + tx; idx < problem.seq_len) {
      const auto val = problem.in[idx];
      if (val >= theta) {
        const auto pos = atomicAdd(&smem->n_ge, 1);
        if (pos < kCapacity) smem->cand[pos] = {val, idx};
      }
    }
  }

  /// One output slot per lane with `pred`; every lane of the warp must call it.
  SGL_DEVICE static uint32_t warp_append(const bool pred, uint32_t* counter) {
    const auto lane_id = threadIdx.x % kWarpSize;
    const auto mask = __ballot_sync(0xFFFFFFFFu, pred);
    uint32_t base = 0;
    if (lane_id == 0 && mask != 0) base = atomicAdd(counter, __popc(mask));
    return __shfl_sync(0xFFFFFFFFu, base, 0) + __popc(mask & ((1u << lane_id) - 1u));
  }

  /// TopKStreaming's phases 2-4 on the n buffered candidates. An overflowing threshold bin
  /// goes to refine_ties, whose rescans of the row keep to the candidates through v_lo >= theta.
  template <bool kUsePDL, typename EmitAbove, bool kStamp = false>
  SGL_DEVICE static void select_candidates(
      const TopKProblem& problem,
      const float theta,
      const uint32_t n,
      Smem* smem,
      const EmitAbove& emit_above,
      uint32_t& first_staged,
      uint64_t* stamps = nullptr) {
    const auto tx = threadIdx.x;
    const auto topk = problem.topk;
    for (uint32_t i = tx; i < n; i += kBlockSize) {
      atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(smem->cand[i].value)], 1);
    }
    __syncthreads();
    find_threshold(topk, n, smem, [&](uint32_t bin) {
      const float2 bounds = coarse_bin_bounds<kHistBits>(bin);
      smem->v_lo = fmaxf(bounds.x, theta);
      smem->v_hi = bounds.y;
    });

    // The tie buffer overlays the histogram, which find_threshold has consumed.
    const auto v_lo = smem->v_lo;
    const auto v_hi = smem->v_hi;
    stamp<kStamp>(stamps, 4);
    PDLTriggerSecondary<kUsePDL>();
#pragma unroll 1
    for (uint32_t i0 = 0; i0 < n; i0 += kBlockSize) {
      const auto i = i0 + tx;
      const auto c = i < n ? smem->cand[i] : TieValue::invalid();
      const bool above = c.value >= v_hi;
      const bool tie = !above && c.value >= v_lo;
      const auto gt = warp_append(above, &smem->count_gt);
      const auto eq = warp_append(tie, &smem->count_eq);
      if (above && gt < topk) emit_above(gt, c.value, c.idx);
      if (tie && eq < kMaxNumTie) smem->tie_values[eq] = c;
    }
    __syncthreads();
    stamp<kStamp>(stamps, 5);
    first_staged = smem->count_gt;
    // The staged slots overlay the candidates, consumed above.
    auto staged = problem;
    staged.out = smem->staged;
    if (smem->count_eq > kMaxNumTie) [[unlikely]] {
      refine_ties(staged, smem, v_lo, v_hi, smem->count_eq);
    }
    const auto count_gt = smem->count_gt;
    const auto count_eq = smem->count_eq;
    const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
    handle_tie(smem->tie_values, staged, count_gt, min(count_eq, kMaxNumTie), remain_topk, &smem->tie_handle);
    if constexpr (kStamp) __syncthreads();
    stamp<kStamp>(stamps, 6);
  }
};

#endif  // !USE_ROCM

// ---------------------------------------------------------------------------
// Cluster path: very long seq_len, small batch. `kClusterSize` blocks cooperate
// on one batch element via distributed shared memory (one cluster per element).
//
// CUDA only: thread-block clusters and distributed shared memory have no CDNA
// equivalent.
//
// Still truncates an overflowing threshold bin, unlike the register, streaming
// and ROCm split paths: the candidate set is split across kClusterSize ranks
// and no single block resolves the row, so refining needs cluster-wide
// histogram and emit counters rather than the block-local ones refine_ties uses.
// ---------------------------------------------------------------------------

#if SUPPORT_CLUSTER

// kBits: the coarse bin width (fp16 key bits). Finer bins overflow the 2048-entry tie buffer less often
// but cost a larger DSMEM fold.
template <uint32_t N, uint32_t kBits = 10>
struct TopKCluster : TopKRadixBase<10> {
 public:
  static constexpr uint32_t kClusterSize = N;
  static constexpr uint32_t kCoarseBits = kBits;
  static constexpr uint32_t kCoarseSize = 1u << kBits;
  static_assert(kBits >= 10 && kBits <= 12 && kCoarseSize % kBlockSize == 0);
  static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();
  struct Smem {
    uint32_t count_eq;
    uint32_t count_gt;
    uint32_t local_start_eq;
    uint32_t local_start_gt;
    float v_lo;
    float v_hi;
    uint32_t warp_sum[kNumWarps];
    union {
      alignas(16) uint32_t histogram[kCoarseSize];
      TieHandleSmem tie_handle;
      int32_t stage_out_idxs[kMaxTopK];
    };
    TieValue tie_values[kMaxNumTie];
  };

  SGL_DEVICE static void barrier_cluster_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  }

  SGL_DEVICE static void barrier_cluster_arrive_release() {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  }

  SGL_DEVICE static void barrier_cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  }

  template <bool kUsePDL>
  SGL_DEVICE static void forward(TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);
    const auto cluster = cooperative_groups::this_cluster();
    const auto this_rank = blockIdx.y;

    init_histogram(smem->histogram, tx);
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
    }
    __syncthreads();
    // Rank 0's shared memory is read by its peers: the zeroed histogram they fold
    // into after bar-0, v_hi / v_lo after bar-2. Those arrives release so the
    // peers' acquire wait orders the reads after the writes at cluster scope;
    // __syncthreads() alone is CTA-scoped. The peers publish nothing at these two
    // barriers and keep the cheaper relaxed arrive.
    if (this_rank == 0) {
      barrier_cluster_arrive_release();  // bar-0 arrive
    } else {
      barrier_cluster_arrive_relaxed();  // bar-0 arrive
    }
    PDLWaitPrimary<kUsePDL>();

    // Phase 1: Load and build histogram over this rank's contiguous chunk.
    for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t) {
      const auto bin = extract_coarse_bin<kCoarseBits>(val);
      atomicAdd(&smem->histogram[bin], 1);
    });

    barrier_cluster_wait();  // bar-0 wait
    __syncthreads();
    if (this_rank != 0) {
      const auto smem_0 = cluster.map_shared_rank(smem, 0);
      // Phase 2. atomic flush all histogram into rank 0
      constexpr uint32_t kFoldItems = kCoarseSize / kBlockSize;
#pragma unroll
      for (uint32_t k = 0; k < kFoldItems; ++k) {
        if (const auto count = smem->histogram[tx + k * kBlockSize]; count != 0) {
          atomicAdd(&smem_0->histogram[tx + k * kBlockSize], count);
        }
      }

      barrier_cluster_arrive_release();  // bar-1 arrive
      barrier_cluster_wait();            // bar-1 wait

      barrier_cluster_arrive_relaxed();  // bar-2 arrive
      barrier_cluster_wait();            // bar-2 wait

      // Phase 4. non-0 rank stage to local smem, then write to rank-0 via DSMEM
      const auto topk = problem.topk;
      const auto v_hi = smem_0->v_hi;
      const auto v_lo = smem_0->v_lo;
      for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            smem->stage_out_idxs[pos] = idx;
          }
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      });
      __syncthreads();
      const auto local_count_gt = smem->count_gt;
      const auto local_count_eq = min(smem->count_eq, kMaxNumTie);
      if (tx == 0) {
        const auto gt = atomicAdd(&smem_0->count_gt, local_count_gt);
        const auto eq = atomicAdd(&smem_0->count_eq, local_count_eq);
        smem->local_start_gt = gt;
        smem->local_start_eq = eq;
      }
      __syncthreads();
      const auto local_start_gt = smem->local_start_gt;
      const auto local_start_eq = smem->local_start_eq;
#pragma unroll
      for (uint32_t i = 0; i < kTieItems; ++i) {
        const auto t = tx + i * kBlockSize;
        if (t < local_count_eq && local_start_eq + t < kMaxNumTie) {
          smem_0->tie_values[local_start_eq + t] = smem->tie_values[t];
        }
      }

      cluster.sync();  // bar 3

      const auto start_write = local_start_gt;
      const auto num_write = local_count_gt;
#pragma unroll
      for (uint32_t i = 0; i < kTopKItems; ++i) {
        if (const auto t = tx + i * kBlockSize; t < num_write && start_write + t < topk) {
          problem.emit(start_write + t, smem->stage_out_idxs[t]);
        }
      }
    } else {
      barrier_cluster_arrive_relaxed();  // bar-1 arrive
      barrier_cluster_wait();            // bar-1 wait

      // Phase 3. rank-0 find threshold and write to local smem for other ranks to read
      find_threshold(problem.topk, problem.seq_len, smem, [&](uint32_t threshold_bin) {
        const auto [v_lo, v_hi] = coarse_bin_bounds<kCoarseBits>(threshold_bin);
        smem->v_hi = v_hi;
        smem->v_lo = v_lo;
      });

      barrier_cluster_arrive_release();  // bar-2 arrive: publishes v_hi / v_lo
      barrier_cluster_wait();            // bar-2 wait

      // Phase 4. rank-0 directly write to output
      const auto topk = problem.topk;
      const auto v_hi = smem->v_hi;
      const auto v_lo = smem->v_lo;
      for_each_input<N>(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
        if (val >= v_hi) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) [[likely]] {
            problem.emit(pos, idx);
          }
        } else if (val >= v_lo) {
          const auto pos = atomicAdd(&smem->count_eq, 1);
          if (pos < kMaxNumTie) [[likely]] {
            smem->tie_values[pos] = {val, idx};
          }
        }
      });

      cluster.sync();     // bar-3

      // Phase 4: Handle ties.
      const auto count_gt = smem->count_gt;
      const auto count_eq = smem->count_eq;
      const auto remain_topk = count_gt < topk ? topk - count_gt : 0;
      const auto tie_count = min(count_eq, kMaxNumTie);
      handle_tie(smem->tie_values, problem, count_gt, tie_count, remain_topk, &smem->tie_handle);
    }
  }
};

#endif  // !USE_ROCM

}  // namespace device::topk

}  // namespace sglang
