// Exact global top-k merge for DSA decode context parallelism (DCP).
//
// Every rank holds the local top-k candidates of its token shard (the v2
// candidate epilogue: fp32 score bits, global position; position -1 is empty).
// One CTA per row:
//   1. pushes the rank's best kCandK candidates to every rank through NVLink
//      multicast, and keeps its full local top-k for peers' fallback reads;
//   2. polls the kNumRanks x kCandK candidates of the row;
//   3. selects the k-th key t* among them (score descending, then position
//      ascending: a strict total order over 64-bit keys);
//   4. checks the certificate: a rank may hold more of the global top-k only if
//      it sent kCandK candidates, has more keys, and all of them reach t*. Then
//      its full local top-k replaces its candidates and t* is selected again,
//      which is exact because a rank holds at most k global top-k keys;
//   5. writes this rank's keys >= t* as widened KV slots, -1 after them.
//
// Split mode (kSplit, KV replicated): rank q scored the 64-token pages p with
// p % kNumRanks == q, and every rank writes ALL keys >= t* as KV slots.
//
// Exchange: every 64-bit entry carries an 8-bit epoch tag next to its payload
// (score bits << 32 | position << 8 | tag), written with one 64-bit multimem
// store and read with one 64-bit relaxed load, so a reader that sees the tag of
// the current call also sees its payload; no flag or release is needed. Each
// row slot keeps a call counter; the tag is counter % 255 + 1 and two buffers
// alternate by its parity. A rank overwrites a parity buffer only after it has
// read every peer's entries of the call in between, which each peer wrote after
// it had finished reading that buffer.
//
// All ranks see the same candidates, so t* and the certificate agree on every
// rank and the fallback needs no extra communication.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace dcp_topk_merge {

// One warp per rank (warp q polls rank q's candidates), up to 16 ranks.
constexpr int kMaxWarps = 16;
// Running totals in `stats`: rows whose certificate failed, of them the rows
// this rank published its keys for, of those the rows where it had to select
// among more than kTopK keys, and all merged rows.
enum { kStatFallback, kStatPublished, kStatSelected, kStatRows, kNumStats };
constexpr uint32_t kEmptyPos = 0xFFFFFFu;  // 24-bit position field of an empty entry

struct Smem {
  int hist[2][256];
  int warp_sum[kMaxWarps];
  int digit, above, cnt;
  uint64_t min_key;
  int sent[kMaxWarps], kept[kMaxWarps];
};

SGL_DEVICE uint32_t convert_to_uint32(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// (score bits, position) -> 64-bit key, larger is better; 0 for an empty slot
SGL_DEVICE uint64_t make_topk_key(int32_t score_bits, int32_t pos) {
  if (pos < 0) return 0;
  return (static_cast<uint64_t>(convert_to_uint32(__int_as_float(score_bits))) << 32) | ~static_cast<uint32_t>(pos);
}

SGL_DEVICE uint64_t pack_entry(int32_t score_bits, int32_t pos, uint64_t tag) {
  const uint32_t p = pos >= 0 ? static_cast<uint32_t>(pos) : kEmptyPos;
  return (static_cast<uint64_t>(static_cast<uint32_t>(score_bits)) << 32) | (static_cast<uint64_t>(p) << 8) | tag;
}

SGL_DEVICE uint64_t entry_key(uint64_t e) {
  const uint32_t p = static_cast<uint32_t>(e >> 8) & kEmptyPos;
  return p == kEmptyPos ? 0 : make_topk_key(static_cast<int32_t>(e >> 32), static_cast<int32_t>(p));
}

// A visitor over a per-thread key array, in the form select_kth takes.
template <int kN>
SGL_DEVICE auto visit_keys(const uint64_t (&keys)[kN]) {
  return [&keys](auto f) {
#pragma unroll
    for (int i = 0; i < kN; ++i)
      f(keys[i]);
  };
}

// Exclusive prefix sum over the CTA (one value per thread); `total` is the CTA sum.
template <int kWarps>
SGL_DEVICE int block_exclusive_sum(int v, int& total, Smem& s) {
  const uint32_t lane = device::get_lane_id();
  const int warp = threadIdx.x / 32;
  const int incl = device::warp::inclusive_sum(v, lane);
  if (lane == 31) s.warp_sum[warp] = incl;
  __syncthreads();
  int before = 0;
  total = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    before += w < warp ? s.warp_sum[w] : 0;
    total += s.warp_sum[w];
  }
  __syncthreads();
  return before + incl - v;
}

// Writes the CTA's selected entries in thread order to slots [0, n) through
// emit(slot, i), i the entry within its thread, and pad(slot) to [n, kWidth).
template <int kWidth, int kThreads, int kN, class Emit, class Pad>
SGL_DEVICE void compact(const bool (&sel)[kN], Emit emit, Pad pad, Smem& s) {
  int mine = 0;
#pragma unroll
  for (int i = 0; i < kN; ++i)
    mine += sel[i];
  int n;
  int at = block_exclusive_sum<kThreads / 32>(mine, n, s);
#pragma unroll
  for (int i = 0; i < kN; ++i)
    if (sel[i]) emit(at++, i);
  for (int e = n + threadIdx.x; e < kWidth; e += kThreads)
    pad(e);
}

// The rem-th largest key (rem >= 1) among the keys visited by `visit(f)`; every
// thread visits its own share and calls f the same number of times (k = 0 for
// an empty slot). 8-bit radix rounds from the top byte with double-buffered
// histograms and a one-warp scan; stops once the chosen bucket is exactly the
// remaining quota and returns that bucket's smallest key. 0 when rem < 1.
template <class Visit>
SGL_DEVICE uint64_t select_kth(Visit visit, int rem, Smem& s) {
  if (rem < 1) return 0;
  const int tid = threadIdx.x, warp = tid / 32;
  const uint32_t lane = device::get_lane_id();
  for (int i = tid; i < 256; i += blockDim.x) {
    s.hist[0][i] = 0;
    s.hist[1][i] = 0;
  }
  __syncthreads();
  uint64_t prefix = 0;
  int b = 0;
  for (int shift = 56; shift >= 0; shift -= 8, b ^= 1) {
    int* h = s.hist[b];
    visit([&](uint64_t k) {
      if (k && (shift == 56 || (k >> (shift + 8)) == prefix)) atomicAdd(&h[(k >> shift) & 255], 1);
    });
    for (int i = tid; i < 256; i += blockDim.x)
      s.hist[b ^ 1][i] = 0;  // its last reader finished before the previous round's last barrier
    __syncthreads();
    if (warp == 0) {
      int c[8], tot = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c[i] = h[255 - (lane * 8 + i)];
        tot += c[i];
      }
      int run = device::warp::inclusive_sum(tot, lane) - tot;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int excl = run;
        run += c[i];
        if (run >= rem && excl < rem) {
          s.digit = 255 - (lane * 8 + i);
          s.above = excl;
          s.cnt = c[i];
          s.min_key = ~uint64_t{0};
        }
      }
    }
    __syncthreads();
    rem -= s.above;
    prefix = (prefix << 8) | static_cast<uint64_t>(s.digit);
    if (s.cnt == rem) {
      const uint64_t pre = prefix;
      uint64_t mk = ~uint64_t{0};
      visit([&](uint64_t k) {
        if (k && (k >> shift) == pre) mk = k < mk ? k : mk;
      });
      mk = device::warp::reduce_min(mk);
      if (lane == 0) atomicMin(reinterpret_cast<unsigned long long*>(&s.min_key), mk);
      __syncthreads();
      return s.min_key;
    }
  }
  return prefix;
}

// sel[i] = keys[i] is at least t; nothing is selected when t is 0
template <int kN>
SGL_DEVICE void select_at_least(const uint64_t (&keys)[kN], uint64_t t, bool (&sel)[kN]) {
#pragma unroll
  for (int i = 0; i < kN; ++i)
    sel[i] = t != 0 && keys[i] >= t;
}

// Poll kPer entries (stride `step` from `src`) until all carry `tag`; every
// load of a round is in flight together.
template <int kPer>
SGL_DEVICE void poll_tagged(const uint64_t* src, int step, uint64_t tag, uint64_t (&v)[kPer]) {
  // 64-bit mask: the pushed merge polls kCandK / 32 = 64 entries per lane at 2048 candidates
  static_assert(kPer >= 1 && kPer <= 64);
  uint64_t pending = kPer == 64 ? ~uint64_t{0} : (uint64_t{1} << kPer) - 1;
  while (pending) {
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (pending >> j & 1) v[j] = device::ptx::load_relaxed_sys(src + j * step);
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if ((v[j] & 255) == tag) pending &= ~(uint64_t{1} << j);
  }
}

struct Params {
  const int2* __restrict__ candidates;    // [rows, kTopK] (score bits, global position)
  uint64_t* __restrict__ buffer;          // this rank's symmetric buffer
  uint64_t* mc_buffer;                    // its multicast address
  const int64_t* __restrict__ peer_ptrs;  // [kNumRanks] symmetric buffer of each rank
  int64_t* __restrict__ counters;         // [max_rows] calls per row slot
  const int32_t* __restrict__ global_lens;
  const int32_t* __restrict__ page_table;
  int64_t page_table_stride;
  int32_t* __restrict__ out;             // [rows, kTopK] widened slots, -1 after
  int32_t* __restrict__ fallback;        // [rows] 1 when the certificate failed
  uint64_t* __restrict__ union_scratch;  // [max_rows, kNumRanks * kTopK] fallback union; global, so smem stays small
  unsigned long long* __restrict__ stats;  // [4] running totals, see kStat*
  int32_t rank;
  int32_t span;  // page_size * kNumRanks
  int32_t max_rows;
};

// Fallback of a row whose certificate failed for the ranks in `fail`: the
// k-th key of the union in union_scratch, where segment q holds rank q's full
// local top-k if it failed (polled over NVLink), else its candidates.
template <int kNumRanks, int kTopK, int kCandK>
SGL_DEVICE uint64_t fallback_kth(
    const Params& p,
    int row,
    unsigned fail,
    int64_t full_off,
    uint64_t tag,
    const uint64_t (&cand)[kCandK / 32],
    Smem& s) {
  constexpr int kThreads = 32 * kNumRanks;
  constexpr int kPerThread = kTopK / kThreads;
  constexpr int kPerLane = kCandK / 32;
  const int tid = threadIdx.x, warp = tid / 32;
  const uint32_t lane = device::get_lane_id();
  uint64_t* const keys = p.union_scratch + static_cast<int64_t>(row) * kNumRanks * kTopK;
#pragma unroll 1
  for (int q = 0; q < kNumRanks; ++q) {
    if (fail >> q & 1) {
      const auto* src = reinterpret_cast<const uint64_t*>(p.peer_ptrs[q]) + full_off;
      uint64_t tmp[kPerThread];
      poll_tagged<kPerThread>(src + tid, kThreads, tag, tmp);
#pragma unroll
      for (int i = 0; i < kPerThread; ++i)
        keys[q * kTopK + tid + i * kThreads] = entry_key(tmp[i]);
    } else if (warp == q) {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i)
        keys[q * kTopK + lane + 32 * i] = cand[i];
      for (int e = kCandK + lane; e < kTopK; e += 32)
        keys[q * kTopK + e] = 0;
    }
  }
  __syncthreads();
  auto visit = [&](auto f) {
#pragma unroll 4
    for (int e = tid; e < kNumRanks * kTopK; e += kThreads)
      f(keys[e]);
  };
  int mine = 0, n;
  visit([&](uint64_t k) { mine += k != 0; });
  block_exclusive_sum<kNumRanks>(mine, n, s);
  return select_kth(visit, min(n, kTopK), s);
}

// fallback_kth for at most two failed ranks without the union scratch
// (SGLANG_K31_DCP_MERGE_STAGED, sgl-project/sglang#49922 / #50040): the failed
// ranks' full local top-k polled into registers and every other rank's candidates.
template <int kNumRanks, int kTopK, int kCandK>
SGL_DEVICE uint64_t fallback_kth_regs(
    const Params& p, unsigned fail, int64_t full_off, uint64_t tag, const uint64_t (&cand)[kCandK / 32], Smem& s) {
  constexpr int kThreads = 32 * kNumRanks;
  constexpr int kPerThread = kTopK / kThreads;
  constexpr int kPerLane = kCandK / 32;
  const int tid = threadIdx.x, warp = tid / 32;
  const int q0 = __ffs(fail) - 1;
  const unsigned rest = fail & (fail - 1);
  const int q1 = rest ? __ffs(rest) - 1 : q0;
  const auto* src0 = reinterpret_cast<const uint64_t*>(p.peer_ptrs[q0]) + full_off + tid;
  const auto* src1 = reinterpret_cast<const uint64_t*>(p.peer_ptrs[q1]) + full_off + tid;
  uint64_t f0[kPerThread], f1[kPerThread];
  unsigned pending = rest ? 0xFFFFFFFFu >> (32 - 2 * kPerThread) : 0xFFFFFFFFu >> (32 - kPerThread);
  while (pending) {
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      if (pending >> j & 1) f0[j] = device::ptx::load_relaxed_sys(src0 + j * kThreads);
      if (pending >> (kPerThread + j) & 1) f1[j] = device::ptx::load_relaxed_sys(src1 + j * kThreads);
    }
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      if ((f0[j] & 255) == tag) pending &= ~(1u << j);
      if ((f1[j] & 255) == tag) pending &= ~(1u << (kPerThread + j));
    }
  }
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    f0[j] = entry_key(f0[j]);
    f1[j] = rest ? entry_key(f1[j]) : 0;
  }
  const bool drop = fail >> warp & 1;
  auto visit = [&](auto f) {
#pragma unroll
    for (int i = 0; i < kPerLane; ++i)
      f(drop ? uint64_t{0} : cand[i]);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
      f(f0[j]);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
      f(f1[j]);
  };
  int mine = 0, n;
  visit([&](uint64_t k) { mine += k != 0; });
  block_exclusive_sum<kNumRanks>(mine, n, s);
  return select_kth(visit, min(n, kTopK), s);
}

// Keys of rank q among the first L positions: token interleave, or (kSplit) its
// 64-token pages q, q + kNumRanks, ... below L.
template <int kNumRanks, bool kSplit>
SGL_DEVICE int local_len(int L, int q) {
  if constexpr (kSplit) {
    const int pages = L / 64;
    const int full = pages - q - 1 >= 0 ? (pages - q - 1) / kNumRanks + 1 : 0;
    return full * 64 + (full * kNumRanks + q == pages ? L % 64 : 0);
  } else {
    return max((L - q + kNumRanks - 1) / kNumRanks, 0);
  }
}

// Buffer layout (uint64_t entries): [2 parities][max_rows][kNumRanks * kCandK + kTopK],
// the candidate block of every rank, then this rank's full local top-k.
// kStaged (SGLANG_K31_DCP_MERGE_STAGED=1, default off; sgl-project/sglang#49922 as in
// #50040): stage the candidate push in smem and store it coalesced, store the own
// full top-k after the push, and take the register fallback for <= 2 failed ranks
// without kSplit. Same keys, same threshold, same output. kMinBlocks: the launch
// bounds' minimum blocks per SM (SGLANG_K31_DCP_MERGE_MIN_BLOCKS, staged only); 1
// caps registers no lower than the bound without it at 256 or 512 threads.
template <int kNumRanks, int kTopK, int kCandK, bool kSplit, bool kStaged = false, int kMinBlocks = 1>
__global__ void __launch_bounds__(32 * kNumRanks, kMinBlocks) merge_kernel(const __grid_constant__ Params p) {
  static_assert(kNumRanks == 8 || kNumRanks == 16, "one warp per rank: 8 or 16 ranks");
  constexpr int kThreads = 32 * kNumRanks;
  static_assert(kCandK % 32 == 0 && kTopK % kThreads == 0 && kCandK <= kTopK);
  constexpr int kPerThread = kTopK / kThreads;  // local top-k entries per thread
  constexpr int kPerLane = kCandK / 32;         // candidates per lane
  constexpr int64_t kSlot = static_cast<int64_t>(kNumRanks) * kCandK + kTopK;
  __shared__ Smem s;

  const int row = blockIdx.x, tid = threadIdx.x, warp = tid / 32;
  const int64_t c = p.counters[row];
  const uint64_t tag = static_cast<uint64_t>(c % 255) + 1;
  const int64_t base = ((c & 1) * p.max_rows + row) * kSlot;
  const int64_t full_off = base + static_cast<int64_t>(kNumRanks) * kCandK;

  // ---- this rank's local top-k, kept tagged for peers' fallback reads ----
  int2 v[kPerThread];
  uint64_t own[kPerThread];
  int n_valid = 0, n_own;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    v[i] = p.candidates[static_cast<int64_t>(row) * kTopK + tid + i * kThreads];
    own[i] = make_topk_key(v[i].x, v[i].y);
    n_valid += own[i] != 0;
    if constexpr (!kStaged) {
      device::ptx::store_relaxed_sys(p.buffer + full_off + tid + i * kThreads, pack_entry(v[i].x, v[i].y, tag));
    }
  }
  block_exclusive_sum<kNumRanks>(n_valid, n_own, s);

  // ---- push the best kCandK to every rank ----
  bool sel[kPerThread];
  select_at_least(own, select_kth(visit_keys(own), min(n_own, kCandK), s), sel);
  uint64_t* const dst = p.mc_buffer + base + static_cast<int64_t>(p.rank) * kCandK;
  if constexpr (kStaged) {
    __shared__ uint64_t stage[kCandK];  // this rank's candidate block, pushed coalesced
    compact<kCandK, kThreads>(
        sel,
        [&](int e, int i) { stage[e] = pack_entry(v[i].x, v[i].y, tag); },
        [&](int e) { stage[e] = pack_entry(0, -1, tag); },
        s);
    __syncthreads();
    for (int e = tid; e < kCandK; e += kThreads)
      device::ptx::multimem_store_relaxed(dst + e, stage[e]);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i)
      device::ptx::store_relaxed_sys(p.buffer + full_off + tid + i * kThreads, pack_entry(v[i].x, v[i].y, tag));
  } else {
    compact<kCandK, kThreads>(
        sel,
        [&](int e, int i) { device::ptx::multimem_store_relaxed(dst + e, pack_entry(v[i].x, v[i].y, tag)); },
        [&](int e) { device::ptx::multimem_store_relaxed(dst + e, pack_entry(0, -1, tag)); },
        s);
  }

  // widened slots of the own keys while the pushes travel
  int slot[kPerThread];
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int pos = v[i].y;
    const int page = pos >= 0 ? p.page_table[row * p.page_table_stride + pos / p.span] : 0;
    slot[i] = pos >= 0 ? page * p.span + pos % p.span : -1;
  }

  // ---- threshold over the kNumRanks x kCandK candidates ----
  uint64_t cand[kPerLane];
  poll_tagged<kPerLane>(p.buffer + base + warp * kCandK + device::get_lane_id(), 32, tag, cand);
  int sent = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    cand[i] = entry_key(cand[i]);
    sent += __popc(__ballot_sync(0xffffffff, cand[i] != 0));
  }
  if (device::get_lane_id() == 0) s.sent[warp] = sent;
  __syncthreads();
  int n_cand = 0;
#pragma unroll
  for (int q = 0; q < kNumRanks; ++q)
    n_cand += s.sent[q];
  const int want = min(n_cand, kTopK);
  uint64_t t = select_kth(visit_keys(cand), want, s);
  int kept = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i)
    kept += __popc(__ballot_sync(0xffffffff, cand[i] != 0 && cand[i] >= t));
  if (device::get_lane_id() == 0) s.kept[warp] = kept;
  __syncthreads();

  // ---- certificate and exact fallback ----
  const int L = p.global_lens[row];
  unsigned fail = 0;
#pragma unroll
  for (int q = 0; q < kNumRanks; ++q) {
    if (want > 0 && s.kept[q] == s.sent[q] && local_len<kNumRanks, kSplit>(L, q) > s.sent[q]) fail |= 1u << q;
  }
  if (tid == 0) {
    p.fallback[row] = fail != 0;
    if (fail) atomicAdd(&p.stats[kStatFallback], 1ull);
    if (row == 0) atomicAdd(&p.stats[kStatRows], static_cast<unsigned long long>(gridDim.x));
  }
  if (fail) {
    if constexpr (kStaged && !kSplit) {
      t = __popc(fail) <= 2 ? fallback_kth_regs<kNumRanks, kTopK, kCandK>(p, fail, full_off, tag, cand, s)
                            : fallback_kth<kNumRanks, kTopK, kCandK>(p, row, fail, full_off, tag, cand, s);
    } else {
      t = fallback_kth<kNumRanks, kTopK, kCandK>(p, row, fail, full_off, tag, cand, s);
    }
  }

  int32_t* const out = p.out + static_cast<int64_t>(row) * kTopK;
  const auto pad = [&](int e) { out[e] = -1; };
  if constexpr (kSplit) {
    // ---- every key >= t*: in the candidates (each rank sent all its keys >= t*
    // unless it failed), else in the union the fallback gathered ----
    const auto to_slot = [&](uint64_t k) {
      const int pos = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(k));
      return p.page_table[row * p.page_table_stride + pos / p.span] * p.span + pos % p.span;
    };
    if (!fail) {
      bool csel[kPerLane];
#pragma unroll
      for (int i = 0; i < kPerLane; ++i)
        csel[i] = cand[i] != 0 && cand[i] >= t;
      compact<kTopK, kThreads>(csel, [&](int e, int i) { out[e] = to_slot(cand[i]); }, pad, s);
    } else {
      // at most 64 union keys per thread: the selection fits one 64-bit mask
      constexpr int kUnion = kNumRanks * kTopK / kThreads;
      static_assert(kUnion <= 64);
      const uint64_t* const keys = p.union_scratch + static_cast<int64_t>(row) * kNumRanks * kTopK;
      uint64_t mask = 0;
#pragma unroll 8
      for (int j = 0; j < kUnion; ++j) {
        const uint64_t k = keys[tid + j * kThreads];
        mask |= static_cast<uint64_t>(k != 0 && k >= t) << j;
      }
      int n;
      int at = block_exclusive_sum<kNumRanks>(__popcll(mask), n, s);
      for (; mask; mask &= mask - 1)
        out[at++] = to_slot(keys[tid + (__ffsll(mask) - 1) * kThreads]);
      for (int e = n + tid; e < kTopK; e += kThreads)
        pad(e);
    }
  } else {
    // ---- this rank's keys >= t*, compacted ----
    select_at_least(own, t, sel);
    compact<kTopK, kThreads>(sel, [&](int e, int i) { out[e] = slot[i]; }, pad, s);
  }
  __syncthreads();
  if (tid == 0) p.counters[row] = c + 1;
}

// ---------------------------------------------------------------------------
// Merge of pushed candidates. The local top-k kernel selects only the rank's
// best kCandK keys and pushes them into this layout itself
// (topk_v2.cuh, transform_candidates_dcp_push), so no rank selects, stores or
// pushes from its full local top-k here. Token interleave only (no kSplit).
//   1. poll the kNumRanks x kCandK candidates (warp q: rank q);
//   2. t* and the certificate as merge_kernel; the select starts at the highest
//      bit where the candidate keys differ;
//   3. a rank that fails the certificate stores its keys >= t* (or, past kTopK
//      of them, its local top-k), tagged, in its full slot, which every rank
//      polls: the union of those and the other ranks' candidates holds the
//      global top-k, as in merge_kernel's fallback (publish_local_topk);
//   4. this rank's keys >= t* as widened slots and as local KV rows (slot /
//      kNumRanks, what dcp_localize_compact computes), -1 after, and their count.
// Buffer reuse is merge_kernel's argument: the pushes of call c + 2 into a
// parity buffer come after the pusher read every rank's call c + 1 entries,
// written by each rank after its call-c merge finished.
// ---------------------------------------------------------------------------

struct PushSmem : Smem {
  uint64_t hi, lo;
  int n_out;
};

// After a row slot's kTopK full-slot entries (merge_pushed_kernel): the owner's stamp (call + 1)
// once its full slot holds this call's keys. A full slot is written only when its rank fails, and
// its 8-bit tag repeats every 510 calls, so the tag alone can pass an old failure's keys.
// 16 entries (one used) keep row slots 128-byte aligned, as the candidate blocks were before.
constexpr int kPushExtra = 16;

SGL_DEVICE void fence_acquire_sys() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

SGL_DEVICE uint64_t shr64(uint64_t k, int n) {
  return n >= 64 ? 0 : k >> n;
}

// select_kth with the first round on the highest bits where the keys differ:
// candidate keys share their sign and top exponent bits, and a round there puts
// every key into one or two bins (serialized shared atomics).
template <class Visit>
SGL_DEVICE uint64_t select_kth_adaptive(Visit visit, int rem, PushSmem& s) {
  if (rem < 1) return 0;
  const int tid = threadIdx.x, warp = tid / 32;
  const uint32_t lane = device::get_lane_id();
  uint64_t hi = 0, lo = ~uint64_t{0};
  visit([&](uint64_t k) {
    if (k) {
      hi = k > hi ? k : hi;
      lo = k < lo ? k : lo;
    }
  });
  hi = device::warp::reduce_max(hi);
  lo = device::warp::reduce_min(lo);
  if (tid == 0) {
    s.hi = 0;
    s.lo = ~uint64_t{0};
  }
  for (int i = tid; i < 256; i += blockDim.x) {
    s.hist[0][i] = 0;
    s.hist[1][i] = 0;
  }
  __syncthreads();
  if (lane == 0) {
    atomicMax(reinterpret_cast<unsigned long long*>(&s.hi), hi);
    atomicMin(reinterpret_cast<unsigned long long*>(&s.lo), lo);
  }
  __syncthreads();
  hi = s.hi;
  lo = s.lo;
  if (hi == lo) return hi;  // a single key (keys are unique), so rem == 1
  const int top = 63 - __clzll(static_cast<long long>(hi ^ lo));
  int width = min(top + 1, 8);
  int shift = top + 1 - width;
  uint64_t prefix = shr64(hi, top + 1);  // the bits every key shares
  for (int b = 0;; b ^= 1) {
    int* h = s.hist[b];
    const uint64_t mask = (uint64_t{1} << width) - 1;
    visit([&](uint64_t k) {
      if (k && shr64(k, shift + width) == prefix) atomicAdd(&h[(k >> shift) & mask], 1);
    });
    for (int i = tid; i < 256; i += blockDim.x)
      s.hist[b ^ 1][i] = 0;  // its last reader finished before the previous round's last barrier
    __syncthreads();
    if (warp == 0) {
      int c[8], tot = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c[i] = h[255 - (lane * 8 + i)];
        tot += c[i];
      }
      int run = device::warp::inclusive_sum(tot, lane) - tot;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int excl = run;
        run += c[i];
        if (run >= rem && excl < rem) {
          s.digit = 255 - (lane * 8 + i);
          s.above = excl;
          s.cnt = c[i];
          s.min_key = ~uint64_t{0};
        }
      }
    }
    __syncthreads();
    rem -= s.above;
    prefix = (prefix << width) | static_cast<uint64_t>(s.digit);
    if (shift == 0) return prefix;  // every bit decided
    if (s.cnt == rem) {
      const uint64_t pre = prefix;
      uint64_t mk = ~uint64_t{0};
      visit([&](uint64_t k) {
        if (k && (k >> shift) == pre) mk = k < mk ? k : mk;
      });
      mk = device::warp::reduce_min(mk);
      if (lane == 0) atomicMin(reinterpret_cast<unsigned long long*>(&s.min_key), mk);
      __syncthreads();
      return s.min_key;
    }
    width = min(shift, 8);
    shift -= width;
  }
}

struct PushedParams {
  uint64_t* __restrict__ buffer;          // this rank's symmetric buffer
  const int64_t* __restrict__ peer_ptrs;  // [kNumRanks] symmetric buffer of each rank
  int64_t* __restrict__ counters;         // [max_rows] calls per row slot
  const float* __restrict__ logits;       // [rows, logits_stride] this rank's scores, read on a fallback only
  int64_t logits_stride;
  const int32_t* __restrict__ local_lens;
  const int32_t* __restrict__ global_lens;
  const int32_t* __restrict__ page_table;
  int64_t page_table_stride;
  int32_t* __restrict__ out;          // [rows, kTopK] widened slots, -1 after
  int32_t* __restrict__ out_local;    // [rows, kTopK] local KV rows, -1 after
  int32_t* __restrict__ out_lengths;  // [rows]
  int32_t* __restrict__ fallback;     // [rows] 1 when the certificate failed
  unsigned long long* __restrict__ stats;  // [kNumStats] running totals
  int32_t rank;
  int32_t span;  // page_size * kNumRanks
  int32_t max_rows;
  int64_t* trace = nullptr;  // bench only (merge_pushed_wide_traced_kernel): [rows, 8] per CTA
};

// This rank failed the certificate: its keys >= t_lo, tagged into its full slot
// (kTopK entries, empty-padded) for every rank to poll after the stamp. t_lo, the
// k-th candidate key, is at most the global k-th key, so these hold every global
// top-k key of this rank, and the other ranks' candidates hold theirs (merge_kernel's
// argument). One pass writes them; on the rare row with more than kTopK of them the
// slot is rewritten with this rank's local top-k, selected among them (readers wait
// for the stamp, written after this returns). Returns whether it had to select.
template <int kNumRanks, int kTopK>
SGL_DEVICE bool
publish_local_topk(const PushedParams& p, int row, int64_t full_off, uint64_t tag, uint64_t t_lo, PushSmem& s) {
  constexpr int kThreads = 32 * kNumRanks;
  constexpr int kTile = 16;  // entries per thread per compaction round
  constexpr int kRound = kThreads * kTile;
  const int tid = threadIdx.x;
  const float* const scores = p.logits + static_cast<int64_t>(row) * p.logits_stride;
  const int n = p.local_lens[row];
  const auto key_at = [&](int l) { return make_topk_key(__float_as_int(scores[l]), l * kNumRanks + p.rank); };
  const auto load = [&](int l0, int32_t (&bits)[kTile]) {
#pragma unroll
    for (int j = 0; j < kTile; ++j) {
      const int l = l0 + j * kThreads + tid;
      bits[j] = l < n ? __float_as_int(scores[l]) : 0;
    }
  };
  uint64_t* const dst = p.buffer + full_off;
  // this rank's keys >= t into dst in round order, the first kTopK of them; returns their count,
  // or stops once it passes kTopK when `probe`. The next round's loads are issued before this
  // round's compaction (latency bound in one CTA).
  const auto write_at_least = [&](uint64_t t, bool probe) {
    int written = 0;
    int32_t next[kTile];
    load(0, next);
    for (int l0 = 0; l0 < n; l0 += kRound) {
      int32_t bits[kTile];
#pragma unroll
      for (int j = 0; j < kTile; ++j)
        bits[j] = next[j];
      if (l0 + kRound < n) load(l0 + kRound, next);
      bool sel[kTile];
      int cnt = 0;
#pragma unroll
      for (int j = 0; j < kTile; ++j) {
        const int l = l0 + j * kThreads + tid;
        sel[j] = l < n && make_topk_key(bits[j], l * kNumRanks + p.rank) >= t;
        cnt += sel[j];
      }
      int total;
      int at = written + block_exclusive_sum<kNumRanks>(cnt, total, s);
#pragma unroll
      for (int j = 0; j < kTile; ++j) {
        if (sel[j]) {
          if (at < kTopK)
            device::ptx::store_relaxed_sys(dst + at, pack_entry(bits[j], (l0 + j * kThreads + tid) * kNumRanks + p.rank, tag));
          ++at;
        }
      }
      written += total;
      if (probe && written > kTopK) break;  // block-uniform: the select follows
    }
    return written;
  };
  int written = write_at_least(t_lo, true);
  const bool select = written > kTopK;
  if (select) {
    const auto visit = [&](auto f) {
      for (int l = tid; l < n; l += kThreads) {
        const uint64_t k = key_at(l);
        f(k >= t_lo ? k : uint64_t{0});
      }
    };
    written = write_at_least(select_kth_adaptive(visit, kTopK, s), false);
  }
  for (int e = written + tid; e < kTopK; e += kThreads)
    device::ptx::store_relaxed_sys(dst + e, pack_entry(0, -1, tag));
  return select;
}

template <int kNumRanks, int kTopK, int kCandK, bool kUsePDL>
__global__ void __launch_bounds__(32 * kNumRanks, kNumRanks == 8 ? 4 : 1)
    merge_pushed_kernel(const __grid_constant__ PushedParams p) {
  static_assert(kNumRanks == 4 || kNumRanks == 8 || kNumRanks == 16, "one warp per rank: 4, 8 or 16 ranks");
  constexpr int kThreads = 32 * kNumRanks;
  static_assert(kCandK % 32 == 0 && kTopK % kThreads == 0 && kCandK <= kTopK);
  constexpr int kPerThread = kTopK / kThreads;  // full-slot entries per thread
  constexpr int kPerLane = kCandK / 32;         // candidates per lane
  constexpr int64_t kSlot = static_cast<int64_t>(kNumRanks) * kCandK + kTopK + kPushExtra;
  __shared__ PushSmem s;

  const int row = blockIdx.x, tid = threadIdx.x, warp = tid / 32;
  const uint32_t lane = device::get_lane_id();
  // the candidate push of this call, and the previous call's counter update
  device::PDLWaitPrimary<kUsePDL>();
  const int64_t c = p.counters[row];
  const uint64_t tag = static_cast<uint64_t>(c % 255) + 1;
  const int64_t base = ((c & 1) * p.max_rows + row) * kSlot;
  const int64_t full_off = base + static_cast<int64_t>(kNumRanks) * kCandK;

  // ---- threshold over the kNumRanks x kCandK candidates ----
  uint64_t cand[kPerLane];
  poll_tagged<kPerLane>(p.buffer + base + warp * kCandK + lane, 32, tag, cand);
  int sent = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    cand[i] = entry_key(cand[i]);
    sent += __popc(__ballot_sync(0xffffffff, cand[i] != 0));
  }
  if (lane == 0) s.sent[warp] = sent;
  __syncthreads();
  int n_cand = 0;
#pragma unroll
  for (int q = 0; q < kNumRanks; ++q)
    n_cand += s.sent[q];
  const int want = min(n_cand, kTopK);
  uint64_t t = select_kth_adaptive(visit_keys(cand), want, s);
  int kept = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i)
    kept += __popc(__ballot_sync(0xffffffff, cand[i] != 0 && cand[i] >= t));
  if (lane == 0) s.kept[warp] = kept;
  __syncthreads();

  // ---- certificate and exact fallback ----
  const int L = p.global_lens[row];
  unsigned fail = 0;
#pragma unroll
  for (int q = 0; q < kNumRanks; ++q) {
    // kept == kTopK: rank q's candidates are the whole top-k and its unsent keys are below t*,
    // so with kCandK == kTopK the certificate never fails
    if (want > 0 && s.kept[q] == s.sent[q] && s.kept[q] < kTopK && local_len<kNumRanks, false>(L, q) > s.sent[q])
      fail |= 1u << q;
  }
  if (tid == 0) {
    p.fallback[row] = fail != 0;
    if (fail) atomicAdd(&p.stats[kStatFallback], 1ull);
    if (row == 0) atomicAdd(&p.stats[kStatRows], static_cast<unsigned long long>(gridDim.x));
  }
  // kCandK == kTopK never fails (above): no fallback code, and none of its registers
  constexpr bool kCanFail = kCandK < kTopK;
  const auto extra = [&](int q) {  // rank q's stamp after its full slot of this row slot
    return reinterpret_cast<const uint64_t*>(p.peer_ptrs[q]) + full_off + kTopK;
  };
  const uint64_t stamp = static_cast<uint64_t>(c) + 1;
  const bool failed = fail >> p.rank & 1;
  if (kCanFail && fail) {
    if (failed) {
      const bool selected = publish_local_topk<kNumRanks, kTopK>(p, row, full_off, tag, t, s);
      __syncthreads();
      if (tid == 0) {
        device::ptx::fence_release_sys();
        device::ptx::store_relaxed_sys(p.buffer + full_off + kTopK, stamp);
        atomicAdd(&p.stats[kStatPublished], 1ull);
        if (selected) atomicAdd(&p.stats[kStatSelected], 1ull);
      }
    }
    // every thread polls its entries of each failed rank's full slot once (after that rank's
    // stamp) and, for up to two failed ranks, keeps the keys in registers; with more, the
    // rounds below re-read them over NVLink (unchanged until this rank's call c + 1)
    const bool regs = __popc(fail) <= 2;
    const int q0 = __ffs(fail) - 1;
    uint64_t f0[kPerThread] = {}, f1[kPerThread] = {};
#pragma unroll 1
    for (int q = 0; q < kNumRanks; ++q) {
      if (fail >> q & 1) {
        if (tid == 0) {
          while (device::ptx::load_relaxed_sys(extra(q)) != stamp) {
          }
          fence_acquire_sys();
        }
        __syncthreads();
        uint64_t tmp[kPerThread];
        poll_tagged<kPerThread>(reinterpret_cast<const uint64_t*>(p.peer_ptrs[q]) + full_off + tid, kThreads, tag, tmp);
        if (regs) {
#pragma unroll
          for (int j = 0; j < kPerThread; ++j) {
            if (q == q0) f0[j] = entry_key(tmp[j]);
            else f1[j] = entry_key(tmp[j]);
          }
        }
      }
    }
    const bool drop = fail >> warp & 1;
    const auto visit = [&](auto f) {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i)
        f(drop ? uint64_t{0} : cand[i]);
      if (regs) {
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
          f(f0[j]);
          f(f1[j]);
        }
        return;
      }
#pragma unroll 1
      for (int q = 0; q < kNumRanks; ++q) {
        if (fail >> q & 1) {
          const auto* src = reinterpret_cast<const uint64_t*>(p.peer_ptrs[q]) + full_off + tid;
#pragma unroll
          for (int j = 0; j < kPerThread; ++j)
            f(entry_key(device::ptx::load_relaxed_sys(src + j * kThreads)));
        }
      }
    };
    int mine = 0, n_union;
    visit([&](uint64_t k) { mine += k != 0; });
    block_exclusive_sum<kNumRanks>(mine, n_union, s);
    t = select_kth_adaptive(visit, min(n_union, kTopK), s);
  }

  // ---- this rank's keys >= t*, compacted ----
  int32_t* const out = p.out + static_cast<int64_t>(row) * kTopK;
  int32_t* const out_local = p.out_local + static_cast<int64_t>(row) * kTopK;
  const auto to_slot = [&](uint64_t k) {
    const int pos = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(k));
    return p.page_table[row * p.page_table_stride + pos / p.span] * p.span + pos % p.span;
  };
  const auto emit = [&](int e, uint64_t k) {
    const int slot = to_slot(k);
    out[e] = slot;
    out_local[e] = slot / kNumRanks;
  };
  int n;
  if (kCanFail && failed) {
    // its keys >= t* are in its local top-k, in its own full slot
    const uint64_t* const src = p.buffer + full_off + tid;
    bool sel[kPerThread];
    uint64_t k[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      k[j] = entry_key(device::ptx::load_relaxed_sys(src + j * kThreads));
      sel[j] = t != 0 && k[j] != 0 && k[j] >= t;
    }
    int mine = 0;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
      mine += sel[j];
    int at = block_exclusive_sum<kNumRanks>(mine, n, s);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j)
      if (sel[j]) emit(at++, k[j]);
  } else {
    // its keys >= t* are among its candidates, held by warp `rank`
    if (warp == p.rank) {
      int at = 0;
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        const bool sel = t != 0 && cand[i] != 0 && cand[i] >= t;
        const unsigned m = __ballot_sync(0xffffffff, sel);
        if (sel) emit(at + __popc(m & ((1u << lane) - 1)), cand[i]);
        at += __popc(m);
      }
      if (lane == 0) s.n_out = at;
    }
    __syncthreads();
    n = s.n_out;
  }
  for (int e = n + tid; e < kTopK; e += kThreads) {
    out[e] = -1;
    out_local[e] = -1;
  }
  if (tid == 0) p.out_lengths[row] = n;
  __syncthreads();
  if (tid == 0) p.counters[row] = c + 1;
}


// merge_pushed_kernel for kCandK == kTopK (every rank pushes its whole local top-k, so the
// certificate never fails) with kWarpsPerRank warps per rank instead of one: rank q's
// candidate j sits in thread j % (32 * kWarpsPerRank) of group q, so each thread holds
// kTopK / (32 * kWarpsPerRank) keys and the select's rounds have that many more warps in
// flight. Same buffer protocol, same t*, and this rank's keys >= t* in candidate order, so
// out / out_local / out_lengths equal merge_pushed_kernel's.
/// kTrace (bench only, merge_pushed_wide_traced_kernel): thread 0 writes p.trace[row] = {start, PDL wait done, every
/// rank's candidates in, t* known, end, row} (globaltimer).
template <int kNumRanks, int kTopK, int kWarpsPerRank, bool kUsePDL, bool kTrace>
SGL_DEVICE void merge_pushed_wide_body(const PushedParams& p) {
  constexpr int kRankThreads = 32 * kWarpsPerRank;
  constexpr int kPer = kTopK / kRankThreads;
  constexpr int kWords = kTopK / 32;
  static_assert(kTopK % kRankThreads == 0 && kPer >= 1 && kPer <= 64);
  constexpr int64_t kSlot = static_cast<int64_t>(kNumRanks) * kTopK + kTopK + kPushExtra;
  __shared__ PushSmem s;
  __shared__ uint32_t sel_bits[kWords];
  __shared__ int sel_before[kWords];
  __shared__ int n_cand_smem, n_sel_smem;

  const int row = blockIdx.x, tid = threadIdx.x;
  const int q = tid / kRankThreads, t = tid % kRankThreads;
  const uint32_t lane = device::get_lane_id();
  const auto stamp = [&](int k) {  // straight to global: no register held across the kernel
    if constexpr (kTrace) {
      if (tid == 0) {
        uint64_t now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        p.trace[row * 8 + k] = static_cast<int64_t>(now);
      }
    }
  };
  stamp(0);
  if (tid == 0) n_cand_smem = 0;
  device::PDLWaitPrimary<kUsePDL>();
  stamp(1);
  const int64_t c = p.counters[row];
  const uint64_t tag = static_cast<uint64_t>(c % 255) + 1;
  const int64_t base = ((c & 1) * p.max_rows + row) * kSlot;

  uint64_t cand[kPer];
  poll_tagged<kPer>(p.buffer + base + q * kTopK + t, kRankThreads, tag, cand);
  int mine = 0;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    cand[i] = entry_key(cand[i]);
    mine += cand[i] != 0;
  }
  mine = device::warp::reduce_sum(mine);
  __syncthreads();  // n_cand_smem zeroed
  stamp(2);
  if (lane == 0) atomicAdd(&n_cand_smem, mine);
  __syncthreads();
  const int want = min(n_cand_smem, kTopK);
  const uint64_t ts = select_kth_adaptive(visit_keys(cand), want, s);
  stamp(3);
  if (tid == 0) {
    p.fallback[row] = 0;
    if (row == 0) atomicAdd(&p.stats[kStatRows], static_cast<unsigned long long>(gridDim.x));
  }

  // ---- this rank's keys >= t*, in candidate order: word w of the bitmap is warp w % kWarpsPerRank's ballot at i = w / kWarpsPerRank ----
  const bool own = q == p.rank;
  uint32_t m[kPer];
  if (own) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      m[i] = __ballot_sync(0xffffffff, ts != 0 && cand[i] != 0 && cand[i] >= ts);
      if (lane == 0) sel_bits[i * kWarpsPerRank + t / 32] = m[i];
    }
  }
  __syncthreads();
  if (tid < 32) {
    int cnt[(kWords + 31) / 32], tot = 0;
#pragma unroll
    for (int j = 0; j < (kWords + 31) / 32; ++j) {
      const int w = lane * ((kWords + 31) / 32) + j;
      cnt[j] = w < kWords ? __popc(sel_bits[w]) : 0;
      tot += cnt[j];
    }
    int run = device::warp::inclusive_sum(tot, lane) - tot;
#pragma unroll
    for (int j = 0; j < (kWords + 31) / 32; ++j) {
      const int w = lane * ((kWords + 31) / 32) + j;
      if (w < kWords) sel_before[w] = run;
      run += cnt[j];
    }
    if (lane == 31) n_sel_smem = run;
  }
  __syncthreads();
  int32_t* const out = p.out + static_cast<int64_t>(row) * kTopK;
  int32_t* const out_local = p.out_local + static_cast<int64_t>(row) * kTopK;
  if (own) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (m[i] >> lane & 1) {
        const int e = sel_before[i * kWarpsPerRank + t / 32] + __popc(m[i] & ((1u << lane) - 1));
        const int pos = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(cand[i]));
        const int slot = p.page_table[row * p.page_table_stride + pos / p.span] * p.span + pos % p.span;
        out[e] = slot;
        out_local[e] = slot / kNumRanks;
      }
    }
  }
  const int n = n_sel_smem;
  for (int e = n + tid; e < kTopK; e += kNumRanks * kRankThreads) {
    out[e] = -1;
    out_local[e] = -1;
  }
  if (tid == 0) p.out_lengths[row] = n;
  __syncthreads();
  if (tid == 0) p.counters[row] = c + 1;
  if constexpr (kTrace) {
    stamp(4);
    if (tid == 0) p.trace[row * 8 + 5] = row;
  }
}

template <int kNumRanks, int kTopK, int kWarpsPerRank, bool kUsePDL>
__global__ void __launch_bounds__(32 * kNumRanks * kWarpsPerRank)
    merge_pushed_wide_kernel(const __grid_constant__ PushedParams p) {
  merge_pushed_wide_body<kNumRanks, kTopK, kWarpsPerRank, kUsePDL, false>(p);
}

/// Bench only: the body with its stamps, held to the production kernel's 2 CTAs per SM (64 registers).
template <int kNumRanks, int kTopK, int kWarpsPerRank, bool kUsePDL>
__global__ void __launch_bounds__(32 * kNumRanks * kWarpsPerRank, 2)
    merge_pushed_wide_traced_kernel(const __grid_constant__ PushedParams p) {
  merge_pushed_wide_body<kNumRanks, kTopK, kWarpsPerRank, kUsePDL, true>(p);
}

}  // namespace dcp_topk_merge

template <int64_t kNumRanks, int64_t kTopK, int64_t kCandK, int64_t kSplit, int64_t kStaged = 0, int64_t kMinBlocks = 1>
struct DcpTopKMergeKernel {
  static constexpr auto kernel =
      dcp_topk_merge::merge_kernel<kNumRanks, kTopK, kCandK, kSplit != 0, kStaged != 0, static_cast<int>(kMinBlocks)>;

  /**
   * \brief Merges one batch of decode rows; every rank of the group calls it
   * with the same rows in the same order. See dcp_topk_merge.py for the
   * tensor contract.
   */
  static void
  run(tvm::ffi::TensorView candidates,
      tvm::ffi::TensorView buffer,
      int64_t mc_buffer,
      tvm::ffi::TensorView peer_ptrs,
      tvm::ffi::TensorView counters,
      tvm::ffi::TensorView global_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out_slots,
      tvm::ffi::TensorView out_fallback,
      tvm::ffi::TensorView union_scratch,
      tvm::ffi::TensorView stats,
      int64_t rank,
      int64_t span) {
    using namespace host;

    auto R = SymbolicSize{"rows"};
    auto M = SymbolicSize{"max_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, kTopK, 2}).with_dtype<int32_t>().with_device(device_).verify(candidates);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device_).verify(buffer);
    TensorMatcher({kNumRanks}).with_dtype<int64_t>().with_device(device_).verify(peer_ptrs);
    TensorMatcher({M}).with_dtype<int64_t>().with_device(device_).verify(counters);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(global_lens);
    TensorMatcher({-1, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({R, kTopK}).with_dtype<int32_t>().with_device(device_).verify(out_slots);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(out_fallback);
    TensorMatcher({M, kNumRanks * kTopK}).with_dtype<int64_t>().with_device(device_).verify(union_scratch);
    TensorMatcher({static_cast<int64_t>(dcp_topk_merge::kNumStats)}).with_dtype<int64_t>().with_device(device_).verify(stats);

    const auto rows = static_cast<int32_t>(R.unwrap());
    const auto max_rows = static_cast<int32_t>(M.unwrap());
    RuntimeCheck(rows <= max_rows, "dcp_topk_merge: rows ", rows, " > max_rows ", max_rows);
    RuntimeCheck(
        page_table.size(0) >= rows, "dcp_topk_merge: page_table has ", page_table.size(0), " rows < rows ", rows);
    const int64_t need = 2LL * max_rows * (kNumRanks * kCandK + kTopK);
    RuntimeCheck(buffer.size(0) >= need, "dcp_topk_merge: buffer numel ", buffer.size(0), " < ", need);
    // an entry carries the position in 24 bits, all ones meaning empty
    const int64_t max_pos = page_table.size(1) * span;
    RuntimeCheck(max_pos <= dcp_topk_merge::kEmptyPos, "dcp_topk_merge: positions up to ", max_pos, " exceed 24 bits");
    if (rows == 0) return;

    const auto params = dcp_topk_merge::Params{
        .candidates = static_cast<const int2*>(candidates.data_ptr()),
        .buffer = static_cast<uint64_t*>(buffer.data_ptr()),
        .mc_buffer = reinterpret_cast<uint64_t*>(mc_buffer),
        .peer_ptrs = static_cast<const int64_t*>(peer_ptrs.data_ptr()),
        .counters = static_cast<int64_t*>(counters.data_ptr()),
        .global_lens = static_cast<const int32_t*>(global_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_table_stride = page_table.stride(0),
        .out = static_cast<int32_t*>(out_slots.data_ptr()),
        .fallback = static_cast<int32_t*>(out_fallback.data_ptr()),
        .union_scratch = static_cast<uint64_t*>(union_scratch.data_ptr()),
        .stats = static_cast<unsigned long long*>(stats.data_ptr()),
        .rank = static_cast<int32_t>(rank),
        .span = static_cast<int32_t>(span),
        .max_rows = max_rows,
    };
    LaunchKernel(rows, static_cast<int>(32 * kNumRanks), device_.unwrap())(kernel, params);
  }
};

template <int64_t kNumRanks, int64_t kTopK, int64_t kCandK, bool kUsePDL>
struct DcpTopKMergePushedKernel {
  static constexpr auto kernel = dcp_topk_merge::merge_pushed_kernel<kNumRanks, kTopK, kCandK, kUsePDL>;

  /**
   * \brief Merges one batch of decode rows whose candidates the local top-k
   * pushed (transform_candidates_dcp_push); every rank of the group calls it
   * with the same rows in the same order. See dcp_topk_merge.py.
   */
  static void
  run(tvm::ffi::TensorView logits,
      tvm::ffi::TensorView local_lens,
      tvm::ffi::TensorView buffer,
      tvm::ffi::TensorView peer_ptrs,
      tvm::ffi::TensorView counters,
      tvm::ffi::TensorView global_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out_slots,
      tvm::ffi::TensorView out_local,
      tvm::ffi::TensorView out_lengths,
      tvm::ffi::TensorView out_fallback,
      tvm::ffi::TensorView stats,
      int64_t rank,
      int64_t span) {
    using namespace host;

    auto R = SymbolicSize{"rows"};
    auto M = SymbolicSize{"max_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, -1}).with_strides({-1, 1}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(local_lens);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device_).verify(buffer);
    TensorMatcher({kNumRanks}).with_dtype<int64_t>().with_device(device_).verify(peer_ptrs);
    TensorMatcher({M}).with_dtype<int64_t>().with_device(device_).verify(counters);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(global_lens);
    TensorMatcher({-1, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({R, kTopK}).with_strides({kTopK, 1}).with_dtype<int32_t>().with_device(device_).verify(out_slots);
    TensorMatcher({R, kTopK}).with_strides({kTopK, 1}).with_dtype<int32_t>().with_device(device_).verify(out_local);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(out_lengths);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(out_fallback);
    TensorMatcher({static_cast<int64_t>(dcp_topk_merge::kNumStats)}).with_dtype<int64_t>().with_device(device_).verify(stats);

    const auto rows = static_cast<int32_t>(R.unwrap());
    const auto max_rows = static_cast<int32_t>(M.unwrap());
    RuntimeCheck(rows <= max_rows, "dcp_topk_merge_pushed: rows ", rows, " > max_rows ", max_rows);
    RuntimeCheck(page_table.size(0) >= rows, "dcp_topk_merge_pushed: page_table has fewer rows than ", rows);
    const int64_t need = 2LL * max_rows * (kNumRanks * kCandK + kTopK + dcp_topk_merge::kPushExtra);
    RuntimeCheck(buffer.size(0) >= need, "dcp_topk_merge_pushed: buffer numel ", buffer.size(0), " < ", need);
    const int64_t max_pos = page_table.size(1) * span;
    RuntimeCheck(max_pos <= dcp_topk_merge::kEmptyPos, "dcp_topk_merge_pushed: positions exceed 24 bits");
    if (rows == 0) return;

    const auto params = dcp_topk_merge::PushedParams{
        .buffer = static_cast<uint64_t*>(buffer.data_ptr()),
        .peer_ptrs = static_cast<const int64_t*>(peer_ptrs.data_ptr()),
        .counters = static_cast<int64_t*>(counters.data_ptr()),
        .logits = static_cast<const float*>(logits.data_ptr()),
        .logits_stride = logits.stride(0),
        .local_lens = static_cast<const int32_t*>(local_lens.data_ptr()),
        .global_lens = static_cast<const int32_t*>(global_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_table_stride = page_table.stride(0),
        .out = static_cast<int32_t*>(out_slots.data_ptr()),
        .out_local = static_cast<int32_t*>(out_local.data_ptr()),
        .out_lengths = static_cast<int32_t*>(out_lengths.data_ptr()),
        .fallback = static_cast<int32_t*>(out_fallback.data_ptr()),
        .stats = static_cast<unsigned long long*>(stats.data_ptr()),
        .rank = static_cast<int32_t>(rank),
        .span = static_cast<int32_t>(span),
        .max_rows = max_rows,
    };
    LaunchKernel(rows, static_cast<int>(32 * kNumRanks), device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};


template <int64_t kNumRanks, int64_t kTopK, int64_t kWarpsPerRank, bool kUsePDL>
struct DcpTopKMergePushedWideKernel {
  static constexpr auto kernel = dcp_topk_merge::merge_pushed_wide_kernel<kNumRanks, kTopK, kWarpsPerRank, kUsePDL>;

  /**
   * \brief DcpTopKMergePushedKernel with cand_k == topk and kWarpsPerRank warps per rank;
   * same arguments and outputs. See dcp_topk_merge.py.
   */
  static void
  run(tvm::ffi::TensorView logits,
      tvm::ffi::TensorView local_lens,
      tvm::ffi::TensorView buffer,
      tvm::ffi::TensorView peer_ptrs,
      tvm::ffi::TensorView counters,
      tvm::ffi::TensorView global_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out_slots,
      tvm::ffi::TensorView out_local,
      tvm::ffi::TensorView out_lengths,
      tvm::ffi::TensorView out_fallback,
      tvm::ffi::TensorView stats,
      int64_t rank,
      int64_t span) {
    run_impl<false>(logits, local_lens, buffer, peer_ptrs, counters, global_lens, page_table, out_slots, out_local, out_lengths, out_fallback, stats, rank, span, nullptr);
  }

  /// Bench only: `run`, plus each CTA's {start, PDL wait done, candidates in, t* known, end, row} into trace [rows, 8].
  static void
  run_traced(tvm::ffi::TensorView logits,
      tvm::ffi::TensorView local_lens,
      tvm::ffi::TensorView buffer,
      tvm::ffi::TensorView peer_ptrs,
      tvm::ffi::TensorView counters,
      tvm::ffi::TensorView global_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out_slots,
      tvm::ffi::TensorView out_local,
      tvm::ffi::TensorView out_lengths,
      tvm::ffi::TensorView out_fallback,
      tvm::ffi::TensorView stats,
      int64_t rank,
      int64_t span,
      tvm::ffi::TensorView trace) {
    run_impl<true>(logits, local_lens, buffer, peer_ptrs, counters, global_lens, page_table, out_slots, out_local, out_lengths, out_fallback, stats, rank, span, &trace);
  }

  template <bool kTrace>
  static void
  run_impl(tvm::ffi::TensorView logits,
      tvm::ffi::TensorView local_lens,
      tvm::ffi::TensorView buffer,
      tvm::ffi::TensorView peer_ptrs,
      tvm::ffi::TensorView counters,
      tvm::ffi::TensorView global_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out_slots,
      tvm::ffi::TensorView out_local,
      tvm::ffi::TensorView out_lengths,
      tvm::ffi::TensorView out_fallback,
      tvm::ffi::TensorView stats,
      int64_t rank,
      int64_t span,
      const tvm::ffi::TensorView* trace) {
    using namespace host;

    auto R = SymbolicSize{"rows"};
    auto M = SymbolicSize{"max_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, -1}).with_strides({-1, 1}).with_dtype<float>().with_device(device_).verify(logits);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(local_lens);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device_).verify(buffer);
    TensorMatcher({kNumRanks}).with_dtype<int64_t>().with_device(device_).verify(peer_ptrs);
    TensorMatcher({M}).with_dtype<int64_t>().with_device(device_).verify(counters);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(global_lens);
    TensorMatcher({-1, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({R, kTopK}).with_strides({kTopK, 1}).with_dtype<int32_t>().with_device(device_).verify(out_slots);
    TensorMatcher({R, kTopK}).with_strides({kTopK, 1}).with_dtype<int32_t>().with_device(device_).verify(out_local);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(out_lengths);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(out_fallback);
    TensorMatcher({static_cast<int64_t>(dcp_topk_merge::kNumStats)}).with_dtype<int64_t>().with_device(device_).verify(stats);

    const auto rows = static_cast<int32_t>(R.unwrap());
    const auto max_rows = static_cast<int32_t>(M.unwrap());
    RuntimeCheck(rows <= max_rows, "dcp_topk_merge_pushed_wide: rows ", rows, " > max_rows ", max_rows);
    RuntimeCheck(page_table.size(0) >= rows, "dcp_topk_merge_pushed_wide: page_table has fewer rows than ", rows);
    const int64_t need = 2LL * max_rows * (kNumRanks * kTopK + kTopK + dcp_topk_merge::kPushExtra);
    RuntimeCheck(buffer.size(0) >= need, "dcp_topk_merge_pushed_wide: buffer numel ", buffer.size(0), " < ", need);
    const int64_t max_pos = page_table.size(1) * span;
    RuntimeCheck(max_pos <= dcp_topk_merge::kEmptyPos, "dcp_topk_merge_pushed_wide: positions exceed 24 bits");
    if (rows == 0) return;

    const auto params = dcp_topk_merge::PushedParams{
        .buffer = static_cast<uint64_t*>(buffer.data_ptr()),
        .peer_ptrs = static_cast<const int64_t*>(peer_ptrs.data_ptr()),
        .counters = static_cast<int64_t*>(counters.data_ptr()),
        .logits = static_cast<const float*>(logits.data_ptr()),
        .logits_stride = logits.stride(0),
        .local_lens = static_cast<const int32_t*>(local_lens.data_ptr()),
        .global_lens = static_cast<const int32_t*>(global_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_table_stride = page_table.stride(0),
        .out = static_cast<int32_t*>(out_slots.data_ptr()),
        .out_local = static_cast<int32_t*>(out_local.data_ptr()),
        .out_lengths = static_cast<int32_t*>(out_lengths.data_ptr()),
        .fallback = static_cast<int32_t*>(out_fallback.data_ptr()),
        .stats = static_cast<unsigned long long*>(stats.data_ptr()),
        .rank = static_cast<int32_t>(rank),
        .span = static_cast<int32_t>(span),
        .max_rows = max_rows,
    };
    if constexpr (kTrace) {
      TensorMatcher({R, 8}).with_strides({8, 1}).with_dtype<int64_t>().with_device(device_).verify(*trace);
      auto traced = params;
      traced.trace = static_cast<int64_t*>(trace->data_ptr());
      LaunchKernel(rows, static_cast<int>(32 * kNumRanks * kWarpsPerRank), device_.unwrap())
          .enable_pdl(kUsePDL)(
              dcp_topk_merge::merge_pushed_wide_traced_kernel<kNumRanks, kTopK, kWarpsPerRank, kUsePDL>, traced);
    } else {
      LaunchKernel(rows, static_cast<int>(32 * kNumRanks * kWarpsPerRank), device_.unwrap())
          .enable_pdl(kUsePDL)(kernel, params);
    }
  }
};

}  // namespace sglang
