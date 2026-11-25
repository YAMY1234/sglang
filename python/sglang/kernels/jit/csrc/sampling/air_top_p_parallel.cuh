// Parallel replacement for flashinfer's deterministic AIR top-p renorm
// (flashinfer/sampling/air_top_p.cuh, sha256 e41572fb, `AirTopPRenormProb<true, float>`).
// The output is meant to be bitwise identical to it for every input:
//   * radix pass 0/1/2 histograms are the same integer (count, uint64 mantissa sum) per bucket;
//   * the bucket selection is AIR's last-block code, copied verbatim (`air_select`);
//   * the norm is AIR's per-thread fold (thread t adds kept i = t, t + 1024, ... in ascending i)
//     followed by the same cub::BlockReduce<float, 1024>;
//   * the output is AIR's elementwise `val >= thr ? val * norm : 0`.
// Must be compiled with -use_fast_math, as flashinfer compiles the AIR module (jit/core.py
// gen_jit_spec default), so every float op lowers to the same instruction.
//
// Launches (one stream, plain launches without PDL, fixed grids, graph-capturable):
//   AirTier1Pass0Kernel     persistent grid G x 512: each CTA loops over a contiguous run of
//                           (row, 4096-element segment) units and adds pass-0 histograms into the
//                           rows' global histograms; no fence, no last-block election
//   AirTier1Select0Kernel   grid rows x 512: AIR's pass-0 selection + sparse decision; clears the
//                           global histogram (the kernel boundary makes every merge visible)
//   AirTier1Pass1Kernel     persistent grid G x 512: pass-1 histogram + ordered compaction of the
//                           elements that can be kept (pass-0 bucket <= chosen)
//   AirTier1NormKernel      grid rows x 1024: AIR's pass-1 selection, pass 2 + selection + norm
//                           (sparse: from the list; dense fallback: AIR's own row loop)
//   AirTier1WriteKernel     grid (segs, rows) x 512: dense output
// G = min(units, SMs x resident blocks per SM); no CTA ever waits for another, so correctness does
// not depend on co-residency. Workspace: laid out by its row capacity. The zero-required regions
// (per-row global histogram, listCount) are zero at the end of every complete call; row state,
// seg_table and list are written before they are read in each call.

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/once.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/object.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace sglang::air_tier1 {

using IdxT = int;
using AccT = float;

// ---- constants shared with AIR ----
inline constexpr int BITS_PER_PASS = 11;
inline constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;
inline constexpr int BLOCK_SIZE = 512;  // AIR radix block size; the selection loops stride by it
inline constexpr int NUM_PASSES = 3;

// ---- Tier 1 layout ----
inline constexpr int kHistThreads = 512;
inline constexpr int kElemsPerThread = 8;  // two float4, index-contiguous per thread
inline constexpr int kSegElems = kHistThreads * kElemsPerThread;  // 4096 = 4 rows of 1024; one loop unit
inline constexpr int kNormThreads = 1024;                         // AIR Apply block size
inline constexpr int kSlotRows = kSegElems / kNormThreads;        // 4
inline constexpr int kMaxSegs = 256;                              // vocab < 2^20
inline constexpr int kListCap = 16384;                            // per-row kept-candidate list
inline constexpr uint32_t kNoBucket = 0xffffffffu;
// Packed shared histogram: count in bits 44..63, mantissa sum in bits 0..43; exact while a
// histogram sees fewer than 2^20 elements (host checks vocab < 2^20).
inline constexpr int kCountShift = 44;
inline constexpr uint64_t kSumMask = (uint64_t{1} << kCountShift) - 1;
// Pass-0 buckets whose elements are all positive finite normals: thr is then such a value and
// `val >= thr` (ftz or not) holds exactly for non-NaN elements in buckets <= b0.
inline constexpr int kSparseMinBucket = 4;
inline constexpr int kSparseMaxBucket = 1019;

struct alignas(64) RowState {
  uint32_t kthValueBits;
  float sum;
  IdxT len;
  uint32_t listCount;
  int sparse;
  float norm;
};

struct Params {
  const float* probs;
  float* out;
  const float* top_p_arr;
  float top_p_val;
  int vocab;
  int num_segs;
  int rows;
  RowState* states;
  unsigned long long* hist_sum;  // [rows][NUM_BUCKETS]
  IdxT* hist_cnt;                // [rows][NUM_BUCKETS]
  int2* seg_table;               // [rows][kMaxSegs]: (offset, count) in the row's list
  int2* list;                    // [rows][kListCap]: (index, value bits)
  uint8_t* fallback;             // [rows]
  unsigned long long* fallback_total;  // cumulative fallback rows on this device, never reset by a call
};

// ======================== AIR helpers (verbatim semantics) ========================

__device__ __forceinline__ int calcStartBit(int pass) {
  int startBit = 32 - (pass + 1) * BITS_PER_PASS;
  return startBit < 0 ? 0 : startBit;
}

__device__ __forceinline__ uint32_t calcMask(int pass) {
  int numBits = calcStartBit(pass - 1) - calcStartBit(pass);
  return (1 << numBits) - 1;
}

// cub::Traits<float>::TwiddleIn / TwiddleOut (cub/util_type.cuh), then AIR's descending flip.
__device__ __forceinline__ uint32_t twiddleIn(float key) {
  uint32_t bits = __float_as_uint(key);
  uint32_t mask = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  bits ^= mask;
  return ~bits;
}

__device__ __forceinline__ float twiddleOut(uint32_t bits) {
  bits = ~bits;
  uint32_t mask = (bits & 0x80000000u) ? 0x80000000u : 0xffffffffu;
  return __uint_as_float(bits ^ mask);
}

__device__ __forceinline__ uint32_t calcMantissa(float value) {
  return __float_as_uint(value) & ((1u << 23) - 1);
}

__device__ __forceinline__ uint32_t calcExponent(float value) {
  return __float_as_uint(value) & ~((1u << 23) - 1);
}

__device__ __forceinline__ float calcFloatValue(uint32_t count, uint32_t exponent, uint64_t bitSum) {
  constexpr uint32_t numTotalBits = 64;
  constexpr uint32_t numMantissa = 23;
  uint64_t extraInMantissa = (bitSum >> numMantissa);
  extraInMantissa = (exponent == 0) ? extraInMantissa : extraInMantissa + count;
  uint32_t numExtra = numTotalBits - __clzll(extraInMantissa);
  int numNorm = (exponent == 0) ? 0 : -1;
  exponent = exponent + ((numExtra + numNorm) << numMantissa);
  uint32_t mantissa;
  if (extraInMantissa != 0) {
    int numMove = numMantissa - (numExtra - 1);
    uint32_t mask = (1u << (numExtra - 1)) - 1;
    extraInMantissa = extraInMantissa & mask;
    if (numMove > 0) {
      extraInMantissa = extraInMantissa << numMove;
      mask = (1u << numMantissa) - 1;
      mantissa = ((bitSum & mask) >> (numExtra - 1)) | extraInMantissa;
    } else {
      mantissa = extraInMantissa >> (-1 * numMove);
    }
  } else {
    mantissa = bitSum;
  }
  uint32_t bitFloat = exponent | mantissa;
  return __uint_as_float(bitFloat);
}

// ======================== histogram plumbing ========================

// Warp-aggregated add of one element (or none: kNoBucket) into a packed shared histogram.
// All 32 lanes must call it together.
__device__ __forceinline__ void histAddWarp(unsigned long long* smem, uint32_t bucket, uint32_t mantissa) {
  const uint32_t peers = __match_any_sync(0xffffffffu, bucket);
  if (bucket != kNoBucket) {
    const uint32_t msum = __reduce_add_sync(peers, mantissa);  // <= 32 * (2^23 - 1)
    if ((threadIdx.x & 31) == static_cast<uint32_t>(__ffs(peers) - 1)) {
      atomicAdd(smem + bucket, (static_cast<unsigned long long>(__popc(peers)) << kCountShift) | msum);
    }
  }
}

// Pass 0 / 1 shared histogram of at most kFlushElems elements of one row (two loop units), as two
// native 32-bit counters per bucket: a = count << 18 | sum(m >> 18), lo = sum(m & 0x3ffff), m the
// 23-bit mantissa. With N <= 8192 elements: sum(m >> 18) <= N * 31 < 2^18, so it never carries into
// the count; a <= 2^31 + N * 31 < 2^32; lo <= N * (2^18 - 1) < 2^31. Both words are exact and the
// bucket's uint64 mantissa sum is (a & 0x3ffff) << 18 + lo, as AIR's. Two words cannot hold 16384
// elements under any split (README), so a buffer is flushed every two units.
// (A 64-bit shared atomicAdd lowers to a CAS spin loop, ATOMS.CAST.SPIN.64, on this target.)
inline constexpr int kUnitsPerFlush = 2;
inline constexpr int kFlushElems = kUnitsPerFlush * kSegElems;  // 8192
inline constexpr int kSegLoBits = 18;
inline constexpr int kSegCountShift = 18;
inline constexpr uint32_t kSegLoMask = (1u << kSegLoBits) - 1;
inline constexpr uint32_t kSegHiMask = (1u << kSegCountShift) - 1;
inline constexpr uint64_t kMantMax = (uint64_t{1} << 23) - 1;
static_assert(static_cast<uint64_t>(kFlushElems) * (kMantMax >> kSegLoBits) < (uint64_t{1} << kSegCountShift),
              "hi sum must stay below the count field");
static_assert((static_cast<uint64_t>(kFlushElems) << kSegCountShift) + kFlushElems * (kMantMax >> kSegLoBits) <
                  (uint64_t{1} << 32),
              "count field");
static_assert(static_cast<uint64_t>(kFlushElems) * kSegLoMask < (uint64_t{1} << 32), "lo field");

struct SegHist {
  uint32_t a[NUM_BUCKETS];
  uint32_t lo[NUM_BUCKETS];
};

// Adds one element (or none: kNoBucket) per lane; all 32 lanes call it together. A warp whose
// lanes share one bucket adds once (full-mask REDUX); otherwise every lane adds its own element
// with native 32-bit shared atomics (same-bucket lanes are replayed by the hardware).
__device__ __forceinline__ void histAddSeg(SegHist& h, uint32_t bucket, uint32_t mantissa) {
  int uniform = 0;
  __match_all_sync(0xffffffffu, bucket, &uniform);
  if (uniform) {
    if (bucket != kNoBucket) {
      const uint32_t hi = __reduce_add_sync(0xffffffffu, mantissa >> kSegLoBits);
      const uint32_t lo = __reduce_add_sync(0xffffffffu, mantissa & kSegLoMask);
      if ((threadIdx.x & 31) == 0) {
        atomicAdd(h.a + bucket, (32u << kSegCountShift) + hi);
        atomicAdd(h.lo + bucket, lo);
      }
    }
  } else if (bucket != kNoBucket) {
    atomicAdd(h.a + bucket, (1u << kSegCountShift) + (mantissa >> kSegLoBits));
    atomicAdd(h.lo + bucket, mantissa & kSegLoMask);
  }
}

// Adds a CTA's buffer into the row's global (sum, count) histogram and zeroes it; the caller has
// synchronised the CTA after the buffer's last add. Integer adds, so the result is independent of
// flush and CTA order (as AIR's own block merge, air_top_p.cuh:269-278).
__device__ __forceinline__ void flush_seg_hist(SegHist& h, unsigned long long* gsum, IdxT* gcnt) {
  for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
    const uint32_t a = h.a[i];
    if (a != 0) {  // count > 0 iff a != 0; lo is 0 when the count is
      const uint32_t lo = h.lo[i];
      h.a[i] = 0;
      h.lo[i] = 0;
      atomicAdd(gsum + i, (static_cast<unsigned long long>(a & kSegHiMask) << kSegLoBits) + lo);
      atomicAdd(gcnt + i, static_cast<IdxT>(a >> kSegCountShift));
    }
  }
}

// Double-buffered: flush interval k adds into hist[k & 1], so a flush needs ONE barrier (before
// it). The buffer flush k zeroes is next written in interval k + 2, after every thread has passed
// flush k + 1's barrier. The flush counters are CTA-uniform registers.
__device__ __forceinline__ void zero_seg_hists(SegHist (&hist)[2]) {
  for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
    hist[0].a[i] = hist[0].lo[i] = hist[1].a[i] = hist[1].lo[i] = 0;
  }
  __syncthreads();
}

// This CTA's contiguous run [begin, end) of the rows x num_segs units, balanced to +-1 unit.
__device__ __forceinline__ void cta_units(const Params& P, int& begin, int& end) {
  const int64_t units = static_cast<int64_t>(P.rows) * P.num_segs;
  begin = static_cast<int>(units * blockIdx.x / gridDim.x);
  end = static_cast<int>(units * (blockIdx.x + 1) / gridDim.x);
}

struct GlobalHist {
  const unsigned long long* sum;
  const IdxT* cnt;
  __device__ __forceinline__ uint64_t value(int i) const { return __ldcg(sum + i); }
  __device__ __forceinline__ IdxT count(int i) const { return __ldcg(cnt + i); }
};

struct PackedSmemHist {
  const unsigned long long* packed;
  __device__ __forceinline__ uint64_t value(int i) const { return packed[i] & kSumMask; }
  __device__ __forceinline__ IdxT count(int i) const { return static_cast<IdxT>(packed[i] >> kCountShift); }
};

// AIR's last-block bucket selection (AirTopPRenormRadixKernel, air_top_p.cuh:328-389), verbatim
// but for: the histogram accessor, `currentSum` passed in (pass > 0: counter->sum; pass 0:
// blockSum * p as in AIR), pass 0 starting kthValueBits from 0 (AIR's Init), and the warp-reduce
// loop restricted to the first BLOCK_SIZE threads (whole warps) when the block is larger.
// Called by every thread of the block; returns nothing, updates *counter from thread 0.
template <class Hist>
__device__ void air_select(
    const Hist histogram, int pass, RowState* counter, AccT* histValueSmem, AccT currentSum, float p) {
  for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
    uint64_t value = histogram.value(i);
    IdxT count = histogram.count(i);
    if (count != 0) {
      uint32_t sb = calcStartBit(pass);
      uint32_t bv = counter->kthValueBits;
      if (pass == 0) bv = i << sb;
      histValueSmem[i] = calcFloatValue(static_cast<uint32_t>(count), calcExponent(twiddleOut(bv)), value);
    } else {
      histValueSmem[i] = 0.0f;
    }
  }
  __syncthreads();
  constexpr int WARP_SIZE = 32;
  constexpr int WARP_COUNT = NUM_BUCKETS / WARP_SIZE;
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  AccT* histPtr = histValueSmem;
  __shared__ AccT warpSum[WARP_COUNT];
  __shared__ cuda::atomic<AccT, cuda::thread_scope_block> blockSum;
  if (threadIdx.x < BLOCK_SIZE) {
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += BLOCK_SIZE)
      reduce_store_async(warp, warpSum + i / WARP_SIZE, histPtr[i], cg::plus<float>{});
  }
  __syncthreads();
  if (threadIdx.x < WARP_SIZE) {
    reduce_store_async(warp, blockSum, warpSum[threadIdx.x], cg::plus<float>{});
    reduce_update_async(warp, blockSum, warpSum[threadIdx.x + WARP_SIZE], cg::plus<float>{});
  }
  __syncthreads();
  if (pass == 0) currentSum = blockSum * p;
  if (threadIdx.x == 0) {
    AccT prev = 0;
    int targetStep = 0;
    for (int i = 0; i < WARP_COUNT; i++) {
      if (warpSum[i]) {
        targetStep = i;
        if ((prev + warpSum[i]) >= currentSum) break;
        prev += warpSum[i];
      }
    }
    int targetIdx = 0;
    for (int i = targetStep * WARP_SIZE; i < NUM_BUCKETS; i++) {
      if (histogram.count(i)) {
        targetIdx = i;
        if ((prev + histPtr[i]) >= currentSum) break;
        prev += histPtr[i];
      }
    }
    counter->sum = currentSum - prev;
    counter->len = histogram.count(targetIdx);
    uint32_t bucket = targetIdx;
    counter->kthValueBits = (pass == 0 ? 0u : counter->kthValueBits) | (bucket << calcStartBit(pass));
  }
  __syncthreads();
}

// Loads the 8 index-contiguous elements of thread `tid` in segment `seg` (callers test
// `base + k < vocab`; vocab % 4 == 0, so a float4 is either whole or absent).
__device__ __forceinline__ void load_segment(
    const float* row, int vocab, int seg, int tid, float (&v)[kElemsPerThread], int& base) {
  base = seg * kSegElems + tid * kElemsPerThread;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int i = base + 4 * h;
    if (i < vocab) {
      const float4 x = *reinterpret_cast<const float4*>(row + i);
      v[4 * h + 0] = x.x;
      v[4 * h + 1] = x.y;
      v[4 * h + 2] = x.z;
      v[4 * h + 3] = x.w;
    }
  }
}

__device__ __forceinline__ void zero_smem_hist(unsigned long long* hist) {
  for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) hist[i] = 0;
  __syncthreads();
}

__device__ __forceinline__ void clear_global_hist(unsigned long long* gsum, IdxT* gcnt) {
  for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
    gsum[i] = 0;
    gcnt[i] = 0;
  }
}

// ======================== kernels ========================

// Loads unit u (row u / num_segs, segment u % num_segs) for thread `tid`; `base` is the index of
// v[0] in its row.
__device__ __forceinline__ void load_unit(const Params& P, int u, float (&v)[kElemsPerThread], int& base) {
  const float* row = P.probs + static_cast<int64_t>(u / P.num_segs) * P.vocab;
  load_segment(row, P.vocab, u % P.num_segs, threadIdx.x, v, base);
}

__global__ void __launch_bounds__(kHistThreads, 2) AirTier1Pass0Kernel(const Params P) {
  __shared__ SegHist hist[2];
  int u, end;
  cta_units(P, u, end);
  if (u >= end) return;  // CTA-uniform; the host never launches more CTAs than units
  zero_seg_hists(hist);
  int buf = 0, pending = 0;
  const int startBit = calcStartBit(0);
  const uint32_t mask = calcMask(0);
  float v[kElemsPerThread] = {}, next[kElemsPerThread] = {};
  int base, next_base = 0;
  load_unit(P, u, v, base);
  for (; u < end; ++u) {
    const int row = u / P.num_segs;
    if (u + 1 < end) load_unit(P, u + 1, next, next_base);  // in flight while this unit is binned
#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) {
      const bool valid = base + k < P.vocab;
      const uint32_t bucket = valid ? ((twiddleIn(v[k]) >> startBit) & mask) : kNoBucket;
      histAddSeg(hist[buf], bucket, valid ? calcMantissa(v[k]) : 0u);
    }
    ++pending;
    if (pending == kUnitsPerFlush || u + 1 == end || (u + 1) / P.num_segs != row) {
      __syncthreads();
      flush_seg_hist(hist[buf], P.hist_sum + static_cast<int64_t>(row) * NUM_BUCKETS,
                     P.hist_cnt + static_cast<int64_t>(row) * NUM_BUCKETS);
      buf ^= 1;
      pending = 0;
    }
#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) v[k] = next[k];
    base = next_base;
  }
}

// AIR's last-block work of pass 0, as its own kernel: the kernel boundary orders every Pass0 merge
// before these reads (no PDL on any launch here).
__global__ void __launch_bounds__(kHistThreads) AirTier1Select0Kernel(const Params P) {
  const int row = blockIdx.x;
  RowState* st = P.states + row;
  unsigned long long* gsum = P.hist_sum + static_cast<int64_t>(row) * NUM_BUCKETS;
  IdxT* gcnt = P.hist_cnt + static_cast<int64_t>(row) * NUM_BUCKETS;
  __shared__ AccT histValue[NUM_BUCKETS];
  const float p = P.top_p_arr ? P.top_p_arr[row] : P.top_p_val;
  air_select(GlobalHist{gsum, gcnt}, 0, st, histValue, AccT{0}, p);
  // Sparse mode: thr is a positive finite normal and the list of pass-0 buckets <= b0 fits.
  const int b0 = static_cast<int>(st->kthValueBits >> calcStartBit(0));
  using BlockReduceInt = cub::BlockReduce<int, kHistThreads>;
  __shared__ typename BlockReduceInt::TempStorage reduce_tmp;
  int mine = 0;
  for (int i = threadIdx.x; i <= b0; i += blockDim.x) mine += __ldcg(gcnt + i);
  const int listed = BlockReduceInt(reduce_tmp).Sum(mine);
  if (threadIdx.x == 0) {
    st->sparse = (b0 >= kSparseMinBucket && b0 <= kSparseMaxBucket && listed <= kListCap) ? 1 : 0;
  }
  __syncthreads();
  clear_global_hist(gsum, gcnt);
}

__global__ void __launch_bounds__(kHistThreads, 2) AirTier1Pass1Kernel(const Params P) {
  __shared__ SegHist hist[2];
  using BlockScan = cub::BlockScan<int, kHistThreads>;
  __shared__ typename BlockScan::TempStorage scan_tmp;
  __shared__ int list_base[2];  // by unit parity: unit u + 1 writes the other slot while u's is read
  int u, end;
  cta_units(P, u, end);
  if (u >= end) return;
  zero_seg_hists(hist);
  int buf = 0, pending = 0;
  const int previousStartBit = calcStartBit(0);
  const int startBit = calcStartBit(1);
  const uint32_t mask = calcMask(1);
  float v[kElemsPerThread] = {}, next[kElemsPerThread] = {};
  int base, next_base = 0;
  load_unit(P, u, v, base);
  for (; u < end; ++u) {
    const int row = u / P.num_segs;
    const int seg = u - row * P.num_segs;
    const RowState* st = P.states + row;
    const bool active = st->len != 0;  // AIR: `if (currentLen == 0) return;`
    if (u + 1 < end) load_unit(P, u + 1, next, next_base);
    if (active) {
      const uint32_t kthValueBits = st->kthValueBits;
      const bool sparse = st->sparse != 0;
      const uint32_t b0 = kthValueBits >> previousStartBit;
      int n_listed = 0;
#pragma unroll
      for (int k = 0; k < kElemsPerThread; ++k) {
        const bool valid = base + k < P.vocab;
        const uint32_t bits = twiddleIn(v[k]);
        const bool in_prefix = valid && ((bits >> previousStartBit) << previousStartBit) == kthValueBits;
        histAddSeg(hist[buf], in_prefix ? ((bits >> startBit) & mask) : kNoBucket, in_prefix ? calcMantissa(v[k]) : 0u);
        n_listed += (sparse && valid && (bits >> previousStartBit) <= b0) ? 1 : 0;
      }
      if (sparse) {
        // Ordered compaction: thread order = index order, one global atomic per unit.
        int offset, total;
        BlockScan(scan_tmp).ExclusiveSum(n_listed, offset, total);
        if (threadIdx.x == 0) {
          const int reserved = static_cast<int>(atomicAdd(&P.states[row].listCount, static_cast<uint32_t>(total)));
          list_base[u & 1] = reserved;
          P.seg_table[static_cast<int64_t>(row) * kMaxSegs + seg] = make_int2(reserved, total);
        }
        __syncthreads();  // also orders this scan's temp-storage reads before the next unit's scan
        int2* out = P.list + static_cast<int64_t>(row) * kListCap + list_base[u & 1] + offset;
#pragma unroll
        for (int k = 0; k < kElemsPerThread; ++k) {
          const bool valid = base + k < P.vocab;
          if (valid && (twiddleIn(v[k]) >> previousStartBit) <= b0) {
            *out++ = make_int2(base + k, __float_as_int(v[k]));
          }
        }
      }
      ++pending;
    }
    if (pending != 0 && (pending == kUnitsPerFlush || u + 1 == end || (u + 1) / P.num_segs != row)) {
      __syncthreads();
      flush_seg_hist(hist[buf], P.hist_sum + static_cast<int64_t>(row) * NUM_BUCKETS,
                     P.hist_cnt + static_cast<int64_t>(row) * NUM_BUCKETS);
      buf ^= 1;
      pending = 0;
    }
#pragma unroll
    for (int k = 0; k < kElemsPerThread; ++k) v[k] = next[k];
    base = next_base;
  }
}

__global__ void __launch_bounds__(kNormThreads) AirTier1NormKernel(const Params P) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* in = P.probs + static_cast<int64_t>(row) * P.vocab;
  RowState* st = P.states + row;
  __shared__ unsigned long long hist[NUM_BUCKETS];
  __shared__ AccT histValue[NUM_BUCKETS];
  __shared__ float slots[kSlotRows][kNormThreads];
  const bool sparse = st->sparse != 0;
  const int2* list = P.list + static_cast<int64_t>(row) * kListCap;
  const int list_count = static_cast<int>(st->listCount);

  // ---- radix pass 1 selection: AIR's last block of pass 1 (skipped with the pass when len == 0,
  // when Pass1 merged nothing). The kernel boundary orders every Pass1 merge before these reads.
  if (st->len != 0) {
    unsigned long long* gsum = P.hist_sum + static_cast<int64_t>(row) * NUM_BUCKETS;
    IdxT* gcnt = P.hist_cnt + static_cast<int64_t>(row) * NUM_BUCKETS;
    air_select(GlobalHist{gsum, gcnt}, 1, st, histValue, st->sum, 0.0f);
    clear_global_hist(gsum, gcnt);
  }

  // ---- radix pass 2 (AIR skips the pass when the previous bucket was empty) ----
  if (st->len != 0) {
    zero_smem_hist(hist);
    const uint32_t kthValueBits = st->kthValueBits;
    const int previousStartBit = calcStartBit(1);
    const int startBit = calcStartBit(2);
    const uint32_t mask = calcMask(2);
    // Rounded up to whole warps so every lane reaches histAddWarp.
    const int n = sparse ? list_count : P.vocab;
    const int n_round = (n + 31) / 32 * 32;
    for (int i = tid; i < n_round; i += kNormThreads) {
      float x = 0.0f;
      bool valid = i < n;
      if (valid) x = sparse ? __int_as_float(list[i].y) : in[i];
      const uint32_t bits = twiddleIn(x);
      const bool in_prefix = valid && ((bits >> previousStartBit) << previousStartBit) == kthValueBits;
      histAddWarp(hist, in_prefix ? ((bits >> startBit) & mask) : kNoBucket, in_prefix ? calcMantissa(x) : 0u);
    }
    __syncthreads();
    air_select(PackedSmemHist{hist}, 2, st, histValue, st->sum, 0.0f);
  }
  const float threshold = twiddleOut(st->kthValueBits);

  // ---- norm: AIR Apply's first loop, thread tid folding i = tid, tid + 1024, ... ascending ----
  float threadSum = 0.0f;
  if (sparse) {
    // A segment is 4 rows of 1024 indices: scatter its kept list entries into slots, then each
    // thread adds its 4 slots in row order. Kept values are > 0, so an empty slot (0) is skipped.
    for (int r = 0; r < kSlotRows; ++r) slots[r][tid] = 0.0f;
    __syncthreads();
    const int2* table = P.seg_table + static_cast<int64_t>(row) * kMaxSegs;
    for (int s = 0; s < P.num_segs; ++s) {
      const int2 seg = table[s];
      for (int k = tid; k < seg.y; k += kNormThreads) {
        const int2 e = list[seg.x + k];
        const float val = __int_as_float(e.y);
        if (val >= threshold) slots[(e.x - s * kSegElems) / kNormThreads][e.x % kNormThreads] = val;
      }
      __syncthreads();
#pragma unroll
      for (int r = 0; r < kSlotRows; ++r) {
        const float val = slots[r][tid];
        if (val > 0.0f) {
          threadSum += static_cast<float>(val);
          slots[r][tid] = 0.0f;
        }
      }
      __syncthreads();
    }
  } else {
    for (int i = tid; i < P.vocab; i += kNormThreads) {
      float val = in[i];
      if (val >= threshold) threadSum += static_cast<float>(val);
    }
  }
  typedef cub::BlockReduce<float, 1024> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp;
  float totalSum = BlockReduce(temp).Sum(threadSum);
  if (tid == 0) {
    st->norm = (totalSum > 1e-8f) ? (1.0f / totalSum) : 1.0f;
    st->listCount = 0;
    if (P.fallback) P.fallback[row] = sparse ? 0 : 1;
    if (!sparse) atomicAdd(P.fallback_total, 1ull);
  }
}

__global__ void __launch_bounds__(kHistThreads) AirTier1WriteKernel(const Params P) {
  const int row = blockIdx.y;
  const RowState* st = P.states + row;
  const float threshold = twiddleOut(st->kthValueBits);
  const float norm = st->norm;
  const int64_t row_off = static_cast<int64_t>(row) * P.vocab;
  const int base = blockIdx.x * kSegElems + threadIdx.x * kElemsPerThread;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int i = base + 4 * h;
    if (i < P.vocab) {
      const float4 x = *reinterpret_cast<const float4*>(P.probs + row_off + i);
      float4 y;
      // AIR Apply: (val >= threshold) ? static_cast<T>(static_cast<float>(val) * norm) : static_cast<T>(0)
      y.x = (x.x >= threshold) ? static_cast<float>(static_cast<float>(x.x) * norm) : static_cast<float>(0);
      y.y = (x.y >= threshold) ? static_cast<float>(static_cast<float>(x.y) * norm) : static_cast<float>(0);
      y.z = (x.z >= threshold) ? static_cast<float>(static_cast<float>(x.z) * norm) : static_cast<float>(0);
      y.w = (x.w >= threshold) ? static_cast<float>(static_cast<float>(x.w) * norm) : static_cast<float>(0);
      *reinterpret_cast<float4*>(P.out + row_off + i) = y;
    }
  }
}

// ======================== host ========================

inline int64_t align256(int64_t x) {
  return (x + 255) / 256 * 256;
}

struct WorkspaceLayout {
  int64_t states, hist_sum, hist_cnt, seg_table, list, total;
  // Offsets come from the allocation's row capacity, not the call's row count: row r's regions
  // stay at one place across calls of any size, so each call only meets the zeros it left.
  explicit WorkspaceLayout(int64_t rows) {
    states = 0;
    hist_sum = states + align256(rows * static_cast<int64_t>(sizeof(RowState)));
    hist_cnt = hist_sum + align256(rows * NUM_BUCKETS * static_cast<int64_t>(sizeof(unsigned long long)));
    seg_table = hist_cnt + align256(rows * NUM_BUCKETS * static_cast<int64_t>(sizeof(IdxT)));
    list = seg_table + align256(rows * kMaxSegs * static_cast<int64_t>(sizeof(int2)));
    total = list + align256(rows * kListCap * static_cast<int64_t>(sizeof(int2)));
  }
};

// Persistent grid: every CTA resident at once (SMs x blocks per SM), never more CTAs than units.
// The occupancy query is memoized per (kernel, device); the first call of a process is eager.
inline uint32_t persistent_grid(void (*kernel)(const Params), int device_id, int64_t units, int64_t grid_limit) {
  const auto key = std::make_pair(reinterpret_cast<const void*>(kernel), device_id);
  const uint32_t per_sm = host::init_once(key, [&] { return host::runtime::get_blocks_per_sm(kernel, kHistThreads); });
  int64_t grid = static_cast<int64_t>(host::runtime::get_sm_count(device_id)) * std::max<uint32_t>(per_sm, 1);
  if (grid_limit > 0) grid = std::min(grid, grid_limit);
  return static_cast<uint32_t>(std::min(grid, units));
}

struct AirTopPParallel {
  static int64_t workspace_bytes(int64_t rows) {
    return WorkspaceLayout(rows).total;
  }

  static void
  run(tvm::ffi::TensorView probs,
      tvm::ffi::TensorView out,
      tvm::ffi::Optional<tvm::ffi::TensorView> top_p_arr,
      double top_p_val,
      tvm::ffi::TensorView workspace,
      int64_t capacity_rows,
      tvm::ffi::TensorView fallback_total,
      tvm::ffi::Optional<tvm::ffi::TensorView> fallback,
      int64_t grid_limit) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto V = SymbolicSize{"vocab"};
    auto device_ = SymbolicDevice{};
    TensorMatcher({R, V}).with_strides({V, 1}).with_device<kDLCUDA>(device_).with_dtype<float>().ensure_alignment(16).verify(
        probs);
    TensorMatcher({R, V}).with_strides({V, 1}).with_device<kDLCUDA>(device_).with_dtype<float>().ensure_alignment(16).verify(
        out);
    if (top_p_arr.has_value()) {
      TensorMatcher({R}).with_strides({1}).with_device<kDLCUDA>(device_).with_dtype<float>().verify(top_p_arr.value());
    }
    if (fallback.has_value()) {
      TensorMatcher({R}).with_device<kDLCUDA>(device_).with_dtype<uint8_t>().verify(fallback.value());
    }
    TensorMatcher({1}).with_device<kDLCUDA>(device_).with_dtype<int64_t>().verify(fallback_total);
    auto W = SymbolicSize{"workspace_bytes"};
    TensorMatcher({W}).with_device<kDLCUDA>(device_).with_dtype<uint8_t>().ensure_alignment(256).verify(workspace);

    const int64_t rows = R.unwrap();
    const int64_t vocab = V.unwrap();
    if (rows == 0) return;
    CHECK_HOST(rows <= 65535) << "air_top_p_parallel: rows " << rows << " > 65535";
    CHECK_HOST(grid_limit >= 0) << "air_top_p_parallel: grid_limit " << grid_limit << " < 0";
    CHECK_HOST(vocab >= NUM_BUCKETS && vocab < (int64_t{1} << 20) && vocab % 4 == 0)
        << "air_top_p_parallel: vocab " << vocab << " must be in [2048, 2^20) and a multiple of 4";
    CHECK_HOST(rows <= capacity_rows) << "air_top_p_parallel: rows " << rows << " > workspace capacity " << capacity_rows;
    const auto layout = WorkspaceLayout(capacity_rows);
    CHECK_HOST(W.unwrap() >= layout.total)
        << "air_top_p_parallel: workspace " << W.unwrap() << " bytes, need " << layout.total;
    const int num_segs = static_cast<int>((vocab + kSegElems - 1) / kSegElems);

    auto* ws = static_cast<uint8_t*>(workspace.data_ptr());
    const auto params = Params{
        .probs = static_cast<const float*>(probs.data_ptr()),
        .out = static_cast<float*>(out.data_ptr()),
        .top_p_arr = top_p_arr.has_value() ? static_cast<const float*>(top_p_arr.value().data_ptr()) : nullptr,
        .top_p_val = static_cast<float>(top_p_val),  // AIR: double top_p_val -> float parameter
        .vocab = static_cast<int>(vocab),
        .num_segs = num_segs,
        .rows = static_cast<int>(rows),
        .states = reinterpret_cast<RowState*>(ws + layout.states),
        .hist_sum = reinterpret_cast<unsigned long long*>(ws + layout.hist_sum),
        .hist_cnt = reinterpret_cast<IdxT*>(ws + layout.hist_cnt),
        .seg_table = reinterpret_cast<int2*>(ws + layout.seg_table),
        .list = reinterpret_cast<int2*>(ws + layout.list),
        .fallback = fallback.has_value() ? static_cast<uint8_t*>(fallback.value().data_ptr()) : nullptr,
        .fallback_total = static_cast<unsigned long long*>(fallback_total.data_ptr()),
    };
    const auto device = device_.unwrap();
    const auto stream = LaunchKernel::resolve_device(device);
    const int64_t units = rows * num_segs;
    const dim3 seg_grid(num_segs, static_cast<uint32_t>(rows));
    // Plain launches (no PDL attribute): each kernel starts after the previous one completed and
    // its writes are visible, which the merged-histogram reads in Select0 and Norm rely on.
    LaunchKernel(persistent_grid(AirTier1Pass0Kernel, device.device_id, units, grid_limit), kHistThreads, stream)(
        AirTier1Pass0Kernel, params);
    LaunchKernel(static_cast<uint32_t>(rows), kHistThreads, stream)(AirTier1Select0Kernel, params);
    LaunchKernel(persistent_grid(AirTier1Pass1Kernel, device.device_id, units, grid_limit), kHistThreads, stream)(
        AirTier1Pass1Kernel, params);
    LaunchKernel(static_cast<uint32_t>(rows), kNormThreads, stream)(AirTier1NormKernel, params);
    LaunchKernel(seg_grid, kHistThreads, stream)(AirTier1WriteKernel, params);
  }
};

}  // namespace sglang::air_tier1
