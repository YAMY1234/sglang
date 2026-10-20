// Fence-free MoE all-to-all for attention-DP + EP decode. dispatch: NCCL-LL lines (8 B payload + the call epoch
// twice per 16-B store) into fixed [source][token] slots; combine: BF16 rows over a 0xFFFF Lamport sentinel (pushers
// map 0xFFFF to 0x7FFF, which no fp32 sum or BF16 output can tell apart), summed in FlashInfer's TOP_K = 10 order.
// Every call writes a header to every target, so no rank starts call c + 2 before all peers finished call c.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {

namespace qwen_dep_ll_a2a {

inline constexpr uint32_t kThreads = 256;
inline constexpr uint32_t kMaxRanks = 16;
inline constexpr uint32_t kMaxTokens = 128;
inline constexpr uint32_t kDispatchStages = 2;
inline constexpr uint32_t kCombineStages = 3;
// Local experts the in-dispatch routing tables can hold: one thread of the last CTA per expert.
inline constexpr uint32_t kMaxLocalExperts = 256;
// Local entries one CTA can receive: 5 id lines (10 ids) per row, rows spread over the grid; kMaxTokens x kMaxRanks
// rows over 128 CTAs is at most 160 ids per CTA.
inline constexpr uint32_t kCtaListCap = 256;
// A peer may legitimately stall for long (e.g. a JIT compile on one rank); only a wait this long is fatal.
inline constexpr uint64_t kTimeoutNs = 300ull * 1000 * 1000 * 1000;

/// \brief Per-call geometry of the dispatch line format and the BF16 combine rows.
/// \tparam kX bytes of hidden state per token (H / 2 for NVFP4, 2 H for BF16). \tparam kS scale bytes (0 for BF16).
template <uint32_t kRanks, uint32_t kTopK, uint32_t kHidden, uint32_t kX, uint32_t kS>
struct Geometry {
  static_assert(kRanks >= 2 && kRanks <= kMaxRanks);
  static_assert(kTopK == 10, "the combine reproduces FlashInfer's TOP_K = 10 reduction order");
  static_assert(kHidden % 128 == 0 && kX % 8 == 0 && kS % 8 == 0);
  static constexpr uint32_t kXBytes = kX;
  static constexpr uint32_t kSBytes = kS;
  static constexpr uint32_t kIdLines = kTopK / 2;
  static constexpr uint32_t kWtLines = kTopK / 2;
  static constexpr uint32_t kXLines = kXBytes / 8;
  static constexpr uint32_t kSLines = kSBytes / 8;
  static constexpr uint32_t kLines = kIdLines + kWtLines + kXLines + kSLines;
  static constexpr uint32_t kChunks = kHidden / 8;

  SGL_DEVICE_HOST static constexpr uint64_t header_offset(uint32_t stage, uint32_t src) {
    return (static_cast<uint64_t>(stage) * kRanks + src) * 16;
  }
  SGL_DEVICE_HOST static constexpr uint64_t slot_offset(uint32_t stage, uint32_t src, uint32_t t, uint32_t slots) {
    return static_cast<uint64_t>(kDispatchStages) * kRanks * 16 +
           ((static_cast<uint64_t>(stage) * kRanks + src) * slots + t) * kLines * 16;
  }
  SGL_DEVICE_HOST static constexpr uint64_t dispatch_bytes(uint32_t slots) {
    return slot_offset(kDispatchStages, 0, 0, slots);
  }
  SGL_DEVICE_HOST static constexpr uint64_t combine_offset(uint32_t stage, uint32_t src, uint32_t t, uint32_t slots) {
    return ((static_cast<uint64_t>(stage) * kRanks + src) * slots + t) * kChunks * 16;
  }
  SGL_DEVICE_HOST static constexpr uint64_t combine_bytes(uint32_t slots) {
    return combine_offset(kCombineStages, 0, 0, slots);
  }
};

struct Peers {
  uint8_t* base[kMaxRanks];
};

/// \brief Cute-DSL routing tables of this EP rank built inside the dispatch (SGLANG_Q35MF_KRNC_EPX): the same six
/// outputs as qwen_ep_tables.cuh (rows ranked within an expert by atomics, as there). count is zero between calls;
/// list holds (expanded index, expert << 16 | rank) pairs of local entries; the last CTA to finish builds the tables
/// and resets count / list_n / done. Disabled when count is null.
struct TableParams {
  int32_t* count;     // [num_local]
  int32_t* list;      // [2 * rows * top_k]
  int32_t* list_n;    // [1]
  int32_t* done;      // [1]
  int32_t* tile_idx_to_expert_idx;
  int32_t* tile_idx_to_mn_limit;
  int32_t* expanded_idx_to_permuted_idx;  // [rows * top_k]
  int32_t* permuted_idx_to_expanded_idx;
  int32_t* total_num_padded_tokens;
  int32_t* num_non_exiting_tiles;
  int32_t local_offset;
  uint32_t num_local;
  uint32_t tile_log2;
};

struct DispatchParams {
  Peers peers;  // each rank's dispatch region
  const uint8_t* __restrict__ x;
  const uint8_t* __restrict__ sf;
  const int32_t* __restrict__ ids;
  const float* __restrict__ wts;
  int32_t* ctr;
  uint8_t* __restrict__ out_x;
  uint8_t* __restrict__ out_sf;
  int32_t* __restrict__ out_ids;
  float* __restrict__ out_wts;
  int32_t* __restrict__ recv_tokens;
  int32_t* err;
  uint32_t tokens;
  uint32_t tmax;
  uint32_t slots;
  uint32_t rank;
  uint32_t experts_per_rank;
  int32_t invalid_id;
  TableParams tab;
};

struct CombineParams {
  Peers peers;  // each rank's combine region
  const bf16_t* __restrict__ partial;
  const int32_t* __restrict__ rids;
  const int32_t* __restrict__ recv_tokens;
  const int32_t* __restrict__ ids;
  bf16_t* __restrict__ out;
  int32_t* ctr;
  int32_t* err;
  uint32_t tokens;
  uint32_t tmax;
  uint32_t slots;
  uint32_t rank;
  uint32_t experts_per_rank;
  uint32_t num_experts;
};

SGL_DEVICE void st_line(void* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(x), "r"(y), "r"(z), "r"(w)
               : "memory");
}

SGL_DEVICE uint4 ld_line(const void* addr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

/// \brief Called once per poll: every 64K polls, read the global timer; after kTimeoutNs of waiting, record
/// `code` in `err` and trap. Never returns early, so a slow peer can never leave stale data in an output.
SGL_DEVICE void wait_tick(uint32_t& spins, uint64_t& since, int32_t* err, int32_t code) {
  if ((++spins & 0xFFFFu) != 0) return;
  uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  if (since == 0) {
    since = now;
  } else if (now - since > kTimeoutNs) {
    atomicExch(err, code);
    __trap();
  }
}

SGL_DEVICE uint32_t canonical_nan2(uint32_t x) {
  const uint32_t m = __vcmpeq2(x, 0xFFFFFFFFu);
  return (x & ~m) | (0x7FFF7FFFu & m);
}

SGL_DEVICE bool has_sentinel(const uint4& v) {
  return (__vcmpeq2(v.x, 0xFFFFFFFFu) | __vcmpeq2(v.y, 0xFFFFFFFFu) | __vcmpeq2(v.z, 0xFFFFFFFFu) |
          __vcmpeq2(v.w, 0xFFFFFFFFu)) != 0u;
}

/// \brief Per-CTA staging of the received ids: local entries are ranked per expert in shared memory and listed;
/// any other entry (another rank's expert or the invalid id) gets permuted index -1 right away.
static_assert(kThreads >= kMaxLocalExperts, "one thread per local expert in the table staging and build");

struct CtaTables {
  int32_t cnt[kMaxLocalExperts];
  int32_t base[kMaxLocalExperts];
  int32_t idx[kCtaListCap];
  int32_t er[kCtaListCap];
  int32_t n;
  int32_t list_base;
};

SGL_DEVICE void table_entry(const TableParams& tab, CtaTables& ct, uint32_t idx, int32_t id) {
  const int32_t e = id - tab.local_offset;
  if (e >= 0 && e < static_cast<int32_t>(tab.num_local)) {
    const int32_t r = atomicAdd(&ct.cnt[e], 1);
    const int32_t slot = atomicAdd(&ct.n, 1);
    ct.idx[slot] = static_cast<int32_t>(idx);
    ct.er[slot] = (e << 16) | r;
  } else {
    tab.expanded_idx_to_permuted_idx[idx] = -1;
  }
}

/// \brief After the CTA's receive loop: one global add per touched expert turns the CTA ranks into expert ranks, one
/// reservation per CTA places its entries on the global list.
SGL_DEVICE void flush_cta_tables(const TableParams& tab, CtaTables& ct) {
  const uint32_t tid = threadIdx.x;
  if (tid < tab.num_local && ct.cnt[tid] > 0) ct.base[tid] = atomicAdd(tab.count + tid, ct.cnt[tid]);
  if (tid == 0 && ct.n > 0) ct.list_base = atomicAdd(tab.list_n, ct.n);
  __syncthreads();
  for (int32_t j = static_cast<int32_t>(tid); j < ct.n; j += kThreads) {
    const int32_t e = ct.er[j] >> 16;
    tab.list[2 * (ct.list_base + j)] = ct.idx[j];
    tab.list[2 * (ct.list_base + j) + 1] = (e << 16) | (ct.base[e] + (ct.er[j] & 0xFFFF));
  }
}

/// \brief Run by the last CTA after every CTA fenced its receive writes: per-expert tiles in ascending local-expert
/// order (qwen_ep_tables.cuh layout), the permutation of the listed entries, then the reset for the next call.
SGL_DEVICE void build_tables(const TableParams& tab) {
  using Scan = cub::BlockScan<int32_t, kThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ int32_t s_offset[kMaxLocalExperts];
  __threadfence();
  const uint32_t e = threadIdx.x;
  const int32_t count = e < tab.num_local ? __ldcg(tab.count + e) : 0;
  const int32_t tile = 1 << tab.tile_log2;
  const int32_t num_tiles = (count + tile - 1) >> tab.tile_log2;
  int32_t tile_offset, total_tiles;
  Scan(scan_storage).ExclusiveSum(num_tiles, tile_offset, total_tiles);
  for (int32_t c = 0; c < num_tiles; ++c) {
    const int32_t end = (tile_offset + c + 1) << tab.tile_log2;
    tab.tile_idx_to_expert_idx[tile_offset + c] = static_cast<int32_t>(e);
    tab.tile_idx_to_mn_limit[tile_offset + c] = min(end, (tile_offset << tab.tile_log2) + count);
  }
  s_offset[e] = tile_offset << tab.tile_log2;
  if (e == 0) {
    tab.num_non_exiting_tiles[0] = total_tiles;
    tab.total_num_padded_tokens[0] = total_tiles << tab.tile_log2;
  }
  __syncthreads();
  const int32_t n = __ldcg(tab.list_n);
  for (int32_t j = static_cast<int32_t>(e); j < n; j += kThreads) {
    const int32_t idx = __ldcg(tab.list + 2 * j);
    const int32_t packed = __ldcg(tab.list + 2 * j + 1);
    const int32_t permuted = s_offset[packed >> 16] + (packed & 0xFFFF);
    tab.expanded_idx_to_permuted_idx[idx] = permuted;
    tab.permuted_idx_to_expanded_idx[permuted] = idx;
  }
  if (e < tab.num_local) tab.count[e] = 0;
  if (e == 0) {
    *tab.list_n = 0;
    *tab.done = 0;
  }
}

template <uint32_t kRanks, uint32_t kTopK, uint32_t kHidden, uint32_t kX, uint32_t kS, bool kUsePDL>
__global__ __launch_bounds__(kThreads) void dispatch_kernel(const __grid_constant__ DispatchParams p) {
  using G = Geometry<kRanks, kTopK, kHidden, kX, kS>;
  __shared__ uint32_t s_mask[kMaxTokens];
  __shared__ uint32_t s_tokens[kRanks];
  __shared__ CtaTables ct;
  const uint32_t b = blockIdx.x, grid = gridDim.x, tid = threadIdx.x;
  const bool tables = p.tab.count != nullptr;
  if (tables) {
    ct.cnt[tid] = 0;
    if (tid == 0) ct.n = 0;
  }
  // The counter was written by an earlier call of this kernel; only the wait orders us after it.
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  const uint32_t call = static_cast<uint32_t>(p.ctr[b]), epoch = call + 1;
  const uint32_t stage = call & 1u;
  if (tid < p.tokens) {
    uint32_t m = 0;
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k)
      m |= 1u << (static_cast<uint32_t>(p.ids[tid * kTopK + k]) / p.experts_per_rank);
    s_mask[tid] = m;
  }
  if (b == 0 && tid < kRanks)
    st_line(p.peers.base[tid] + G::header_offset(stage, p.rank), p.tokens, epoch, p.tokens, epoch);
  __syncthreads();

  const uint32_t sends = p.tokens * kRanks * G::kLines;
  for (uint32_t task = b * kThreads + tid; task < sends; task += grid * kThreads) {
    const uint32_t i = task % G::kLines, j = task / G::kLines, r = j % kRanks, t = j / kRanks;
    const bool routed = s_mask[t] >> r & 1u;
    // A token with no expert on rank r sends it only the id lines (all invalid), which its reader checks first.
    if (!routed && i >= G::kIdLines) continue;
    uint2 d;
    if (i < G::kIdLines) {
      d = routed ? *reinterpret_cast<const uint2*>(p.ids + t * kTopK + 2 * i)
                 : make_uint2(static_cast<uint32_t>(p.invalid_id), static_cast<uint32_t>(p.invalid_id));
    } else if (i < G::kIdLines + G::kWtLines) {
      d = *reinterpret_cast<const uint2*>(p.wts + t * kTopK + 2 * (i - G::kIdLines));
    } else if (i < G::kIdLines + G::kWtLines + G::kXLines) {
      d = *reinterpret_cast<const uint2*>(
          p.x + static_cast<uint64_t>(t) * G::kXBytes + 8 * (i - G::kIdLines - G::kWtLines));
    } else {
      d = *reinterpret_cast<const uint2*>(
          p.sf + static_cast<uint64_t>(t) * G::kSBytes + 8 * (i - G::kIdLines - G::kWtLines - G::kXLines));
    }
    st_line(p.peers.base[r] + G::slot_offset(stage, p.rank, t, p.slots) + i * 16, d.x, epoch, d.y, epoch);
  }

  const uint8_t* mine = p.peers.base[p.rank];
  if (tid < kRanks) {
    const uint8_t* h = mine + G::header_offset(stage, tid);
    uint4 v;
    uint32_t spins = 0;
    uint64_t since = 0;
    for (v = ld_line(h); v.y != epoch || v.w != epoch; v = ld_line(h))
      wait_tick(spins, since, p.err, 1);
    s_tokens[tid] = v.x;
    if (b == 0) p.recv_tokens[tid] = static_cast<int32_t>(v.x);
  }
  __syncthreads();

  // Padding rows (t >= tokens of that source) get invalid ids, as FlashInfer's sanitize does.
  for (uint32_t task = b * kThreads + tid; task < kRanks * p.tmax * kTopK; task += grid * kThreads) {
    const uint32_t s = task / (p.tmax * kTopK), t = task / kTopK % p.tmax;
    if (t >= s_tokens[s]) {
      p.out_ids[task] = p.invalid_id;
      if (tables) p.tab.expanded_idx_to_permuted_idx[task] = -1;
    }
  }
  uint32_t max_tokens = 0;
#pragma unroll
  for (uint32_t s = 0; s < kRanks; ++s)
    max_tokens = max(max_tokens, s_tokens[s]);
  const uint32_t recvs = max_tokens * kRanks * G::kLines;
  for (uint32_t task = b * kThreads + tid; task < recvs; task += grid * kThreads) {
    const uint32_t i = task % G::kLines, j = task / G::kLines, s = j % kRanks, t = j / kRanks;
    if (t >= s_tokens[s]) continue;
    const uint8_t* slot = mine + G::slot_offset(stage, s, t, p.slots);
    uint4 v;
    uint32_t spins = 0;
    uint64_t since = 0;
    const uint8_t* line = slot + i * 16;
    v = ld_line(line);
    if (i >= G::kIdLines) {
      // Line 0 carries the first id: the invalid id means the source sent only the id lines.
      uint4 v0 = ld_line(slot);
      for (; v0.y != epoch || v0.w != epoch; v0 = ld_line(slot))
        wait_tick(spins, since, p.err, 2);
      if (static_cast<int32_t>(v0.x) == p.invalid_id) continue;
    }
    for (; v.y != epoch || v.w != epoch; v = ld_line(line))
      wait_tick(spins, since, p.err, 2);
    const uint2 d = make_uint2(v.x, v.z);
    const uint64_t row = static_cast<uint64_t>(s) * p.tmax + t;
    if (i < G::kIdLines) {
      *reinterpret_cast<uint2*>(p.out_ids + row * kTopK + 2 * i) = d;
      if (tables) {
        table_entry(p.tab, ct, static_cast<uint32_t>(row * kTopK + 2 * i), static_cast<int32_t>(d.x));
        table_entry(p.tab, ct, static_cast<uint32_t>(row * kTopK + 2 * i + 1), static_cast<int32_t>(d.y));
      }
    } else if (i < G::kIdLines + G::kWtLines) {
      *reinterpret_cast<uint2*>(p.out_wts + row * kTopK + 2 * (i - G::kIdLines)) = d;
    } else if (i < G::kIdLines + G::kWtLines + G::kXLines) {
      *reinterpret_cast<uint2*>(p.out_x + row * G::kXBytes + 8 * (i - G::kIdLines - G::kWtLines)) = d;
    } else {
      *reinterpret_cast<uint2*>(p.out_sf + row * G::kSBytes + 8 * (i - G::kIdLines - G::kWtLines - G::kXLines)) = d;
    }
  }
  __syncthreads();
  if (tid == 0) p.ctr[b] = static_cast<int32_t>(epoch);
  if (tables) {
    __shared__ uint32_t s_last;
    flush_cta_tables(p.tab, ct);
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(p.tab.done, 1) == static_cast<int32_t>(grid - 1);
    __syncthreads();
    if (s_last) build_tables(p.tab);
  }
}

template <uint32_t kRanks, uint32_t kTopK, uint32_t kHidden, uint32_t kX, uint32_t kS, bool kUsePDL>
__global__ __launch_bounds__(kThreads) void combine_kernel(const __grid_constant__ CombineParams p) {
  using G = Geometry<kRanks, kTopK, kHidden, kX, kS>;
  __shared__ uint32_t s_tokens[kRanks];
  const uint32_t b = blockIdx.x, grid = gridDim.x, tid = threadIdx.x;
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  const uint32_t call = static_cast<uint32_t>(p.ctr[b]);
  const uint32_t stage = call % kCombineStages;
  if (tid < kRanks) s_tokens[tid] = static_cast<uint32_t>(p.recv_tokens[tid]);
  __syncthreads();
  uint32_t max_tokens = 0;
#pragma unroll
  for (uint32_t s = 0; s < kRanks; ++s)
    max_tokens = max(max_tokens, s_tokens[s]);

  // Push this rank's row of every routed (source, token) to the source.
  const uint32_t pushes = max_tokens * kRanks * G::kChunks;
  for (uint32_t task = b * kThreads + tid; task < pushes; task += grid * kThreads) {
    const uint32_t i = task % G::kChunks, j = task / G::kChunks, s = j % kRanks, t = j / kRanks;
    if (t >= s_tokens[s]) continue;
    const uint64_t row = static_cast<uint64_t>(s) * p.tmax + t;
    if (static_cast<uint32_t>(p.rids[row * kTopK]) >= p.num_experts) continue;
    const uint4 v = reinterpret_cast<const uint4*>(p.partial + row * kHidden)[i];
    st_line(
        p.peers.base[s] + G::combine_offset(stage, p.rank, t, p.slots) + i * 16,
        canonical_nan2(v.x),
        canonical_nan2(v.y),
        canonical_nan2(v.z),
        canonical_nan2(v.w));
  }

  // Sum the first-route contributions of every own token, chunk by chunk.
  uint8_t* mine = p.peers.base[p.rank];
  for (uint32_t task = b * kThreads + tid; task < p.tokens * G::kChunks; task += grid * kThreads) {
    const uint32_t i = task % G::kChunks, t = task / G::kChunks;
    int32_t src[kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      const uint32_t r = static_cast<uint32_t>(p.ids[t * kTopK + k]) / p.experts_per_rank;
      bool first = true;
#pragma unroll
      for (uint32_t q = 0; q < kTopK; ++q) {
        if (q < k && static_cast<uint32_t>(p.ids[t * kTopK + q]) / p.experts_per_rank == r) first = false;
      }
      src[k] = first ? static_cast<int32_t>(r) : -1;
    }
    uint4 v[kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      v[k] =
          src[k] >= 0 ? ld_line(mine + G::combine_offset(stage, src[k], t, p.slots) + i * 16) : make_uint4(0, 0, 0, 0);
    }
    uint32_t spins = 0;
    uint64_t since = 0;
    bool pending = true;
    while (pending) {
      pending = false;
#pragma unroll
      for (uint32_t k = 0; k < kTopK; ++k) {
        if (src[k] >= 0 && has_sentinel(v[k])) {
          v[k] = ld_line(mine + G::combine_offset(stage, src[k], t, p.slots) + i * 16);
          pending = true;
        }
      }
      if (pending) wait_tick(spins, since, p.err, 3);
    }
    float acc[8][kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      const uint32_t w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
      for (uint32_t e = 0; e < 8; ++e) {
        acc[e][k] = src[k] >= 0 ? __uint_as_float((e & 1) ? (w[e >> 1] & 0xFFFF0000u) : (w[e >> 1] << 16)) : 0.0f;
      }
      if (src[k] >= 0) {
        st_line(
            mine + G::combine_offset(stage, src[k], t, p.slots) + i * 16,
            0xFFFFFFFFu,
            0xFFFFFFFFu,
            0xFFFFFFFFu,
            0xFFFFFFFFu);
      }
    }
    alignas(16) bf16_t o[8];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
      float* c = acc[e];
      // FlashInfer moeAlltoAllCombine.cu reduce_contributions<10>.
      c[0] += c[1];
      c[2] += c[3];
      c[4] += c[5];
      c[6] += c[7];
      c[8] += c[9];
      c[0] += c[2];
      c[4] += c[6];
      c[0] += c[4];
      c[0] += c[8];
      o[e] = __float2bfloat16(c[0]);
    }
    reinterpret_cast<uint4*>(p.out + static_cast<uint64_t>(t) * kHidden)[i] = *reinterpret_cast<const uint4*>(o);
  }
  __syncthreads();
  if (tid == 0) p.ctr[b] = static_cast<int32_t>(call + 1);
}

/// \brief Host entry points. Peer region bases come as a CPU int64 tensor [ranks].
/// \tparam kRanks EP ranks. \tparam kTopK experts per token. \tparam kHidden hidden size.
/// \tparam kX, kS payload bytes per token (see Geometry). \tparam kUsePDL PDL launch.
template <uint32_t kRanks, uint32_t kTopK, uint32_t kHidden, uint32_t kX, uint32_t kS, bool kUsePDL>
struct LlA2A {
  using G = Geometry<kRanks, kTopK, kHidden, kX, kS>;

  static Peers peers_from(const tvm::ffi::TensorView peers) {
    using namespace host;
    TensorMatcher({static_cast<int64_t>(kRanks)}).with_dtype<int64_t>().with_device<kDLCPU>().verify(peers);
    Peers out{};
    const int64_t* base = static_cast<const int64_t*>(peers.data_ptr());
    for (uint32_t r = 0; r < kRanks; ++r)
      out.base[r] = reinterpret_cast<uint8_t*>(base[r]);
    return out;
  }

  static void dispatch(
      const tvm::ffi::TensorView peers,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView sf,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView wts,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView out_x,
      const tvm::ffi::TensorView out_sf,
      const tvm::ffi::TensorView out_ids,
      const tvm::ffi::TensorView out_wts,
      const tvm::ffi::TensorView recv_tokens,
      const tvm::ffi::TensorView err,
      int64_t tmax,
      int64_t slots,
      int64_t rank,
      int64_t experts_per_rank,
      int64_t invalid_id) {
    launch_dispatch(
        peers, x, sf, ids, wts, ctr, out_x, out_sf, out_ids, out_wts, recv_tokens, err, tmax, slots, rank,
        experts_per_rank, invalid_id, TableParams{});
  }

  /// \brief dispatch that also writes this rank's cute-DSL routing tables (SGLANG_Q35MF_KRNC_EPX); the tables are
  /// the six tensors of qwen_ep_tables.cuh, count / list / list_n / done its scratch (count, list_n, done zero).
  static void dispatch_tables(
      const tvm::ffi::TensorView peers,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView sf,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView wts,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView out_x,
      const tvm::ffi::TensorView out_sf,
      const tvm::ffi::TensorView out_ids,
      const tvm::ffi::TensorView out_wts,
      const tvm::ffi::TensorView recv_tokens,
      const tvm::ffi::TensorView err,
      const tvm::ffi::TensorView count,
      const tvm::ffi::TensorView list,
      const tvm::ffi::TensorView list_n,
      const tvm::ffi::TensorView done,
      const tvm::ffi::TensorView tile_idx_to_expert_idx,
      const tvm::ffi::TensorView tile_idx_to_mn_limit,
      const tvm::ffi::TensorView expanded_idx_to_permuted_idx,
      const tvm::ffi::TensorView permuted_idx_to_expanded_idx,
      const tvm::ffi::TensorView total_num_padded_tokens,
      const tvm::ffi::TensorView num_non_exiting_tiles,
      int64_t tmax,
      int64_t slots,
      int64_t rank,
      int64_t experts_per_rank,
      int64_t invalid_id,
      int64_t local_offset,
      int64_t num_local,
      int64_t tile_log2) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto R_ = SymbolicSize{"rows"};
    auto T_ = SymbolicSize{"max_tiles"};
    auto P_ = SymbolicSize{"max_permuted"};
    auto L_ = SymbolicSize{"list"};
    TensorMatcher({R_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(out_ids);
    TensorMatcher({R_, static_cast<int64_t>(kTopK)})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(expanded_idx_to_permuted_idx);
    TensorMatcher({num_local}).with_dtype<int32_t>().with_device(device).verify(count);
    TensorMatcher({L_}).with_dtype<int32_t>().with_device(device).verify(list);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(list_n);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(done);
    TensorMatcher({T_}).with_dtype<int32_t>().with_device(device).verify(tile_idx_to_expert_idx);
    TensorMatcher({T_}).with_dtype<int32_t>().with_device(device).verify(tile_idx_to_mn_limit);
    TensorMatcher({P_}).with_dtype<int32_t>().with_device(device).verify(permuted_idx_to_expanded_idx);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(total_num_padded_tokens);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_non_exiting_tiles);
    const int64_t expanded = R_.unwrap() * static_cast<int64_t>(kTopK);
    CHECK_HOST(num_local > 0 && num_local <= kMaxLocalExperts) << "ll_a2a tables: num_local " << num_local;
    CHECK_HOST(tile_log2 >= 3 && tile_log2 <= 10) << "ll_a2a tables: tile_log2 " << tile_log2;
    CHECK_HOST(L_.unwrap() >= 2 * expanded) << "ll_a2a tables: list " << L_.unwrap() << " < 2 x " << expanded;
    const int64_t filled = std::min<int64_t>(num_local, expanded);
    const int64_t max_tiles = filled + (expanded - filled) / (int64_t{1} << tile_log2);
    CHECK_HOST(T_.unwrap() >= max_tiles) << "ll_a2a tables: tile tables too small";
    CHECK_HOST(P_.unwrap() >= (max_tiles << tile_log2)) << "ll_a2a tables: permuted map too small";
    TableParams tab{};
    tab.count = static_cast<int32_t*>(count.data_ptr());
    tab.list = static_cast<int32_t*>(list.data_ptr());
    tab.list_n = static_cast<int32_t*>(list_n.data_ptr());
    tab.done = static_cast<int32_t*>(done.data_ptr());
    tab.tile_idx_to_expert_idx = static_cast<int32_t*>(tile_idx_to_expert_idx.data_ptr());
    tab.tile_idx_to_mn_limit = static_cast<int32_t*>(tile_idx_to_mn_limit.data_ptr());
    tab.expanded_idx_to_permuted_idx = static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
    tab.permuted_idx_to_expanded_idx = static_cast<int32_t*>(permuted_idx_to_expanded_idx.data_ptr());
    tab.total_num_padded_tokens = static_cast<int32_t*>(total_num_padded_tokens.data_ptr());
    tab.num_non_exiting_tiles = static_cast<int32_t*>(num_non_exiting_tiles.data_ptr());
    tab.local_offset = static_cast<int32_t>(local_offset);
    tab.num_local = static_cast<uint32_t>(num_local);
    tab.tile_log2 = static_cast<uint32_t>(tile_log2);
    launch_dispatch(
        peers, x, sf, ids, wts, ctr, out_x, out_sf, out_ids, out_wts, recv_tokens, err, tmax, slots, rank,
        experts_per_rank, invalid_id, tab);
  }

  static void launch_dispatch(
      const tvm::ffi::TensorView peers,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView sf,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView wts,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView out_x,
      const tvm::ffi::TensorView out_sf,
      const tvm::ffi::TensorView out_ids,
      const tvm::ffi::TensorView out_wts,
      const tvm::ffi::TensorView recv_tokens,
      const tvm::ffi::TensorView err,
      int64_t tmax,
      int64_t slots,
      int64_t rank,
      int64_t experts_per_rank,
      int64_t invalid_id,
      const TableParams& tab) {
    using namespace host;
    auto T_ = SymbolicSize{"tokens"};
    auto R_ = SymbolicSize{"rows"};
    auto C_ = SymbolicSize{"ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T_, static_cast<int64_t>(G::kXBytes)}).with_dtype<uint8_t>().with_device(device).verify(x);
    TensorMatcher({T_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({T_, static_cast<int64_t>(kTopK)}).with_dtype<fp32_t>().with_device(device).verify(wts);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(ctr);
    TensorMatcher({R_, static_cast<int64_t>(G::kXBytes)}).with_dtype<uint8_t>().with_device(device).verify(out_x);
    TensorMatcher({R_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(out_ids);
    TensorMatcher({R_, static_cast<int64_t>(kTopK)}).with_dtype<fp32_t>().with_device(device).verify(out_wts);
    TensorMatcher({static_cast<int64_t>(kRanks)}).with_dtype<int32_t>().with_device(device).verify(recv_tokens);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(err);
    if constexpr (G::kSBytes > 0) {  // BF16 payloads carry no scales and never read sf / out_sf
      TensorMatcher({T_, static_cast<int64_t>(G::kSBytes)}).with_dtype<uint8_t>().with_device(device).verify(sf);
      TensorMatcher({R_, static_cast<int64_t>(G::kSBytes)}).with_dtype<uint8_t>().with_device(device).verify(out_sf);
    }
    const int64_t tokens = T_.unwrap();
    CHECK_HOST(tokens <= tmax && tmax <= slots && slots <= kMaxTokens)
        << "ll_a2a dispatch: tokens " << tokens << " tmax " << tmax << " slots " << slots;
    CHECK_HOST(R_.unwrap() == kRanks * tmax) << "ll_a2a dispatch: " << R_.unwrap() << " output rows, tmax " << tmax;
    CHECK_HOST(rank >= 0 && rank < kRanks && experts_per_rank > 0);
    DispatchParams params{};
    params.peers = peers_from(peers);
    params.x = static_cast<const uint8_t*>(x.data_ptr());
    params.sf = static_cast<const uint8_t*>(sf.data_ptr());
    params.ids = static_cast<const int32_t*>(ids.data_ptr());
    params.wts = static_cast<const float*>(wts.data_ptr());
    params.ctr = static_cast<int32_t*>(ctr.data_ptr());
    params.out_x = static_cast<uint8_t*>(out_x.data_ptr());
    params.out_sf = static_cast<uint8_t*>(out_sf.data_ptr());
    params.out_ids = static_cast<int32_t*>(out_ids.data_ptr());
    params.out_wts = static_cast<float*>(out_wts.data_ptr());
    params.recv_tokens = static_cast<int32_t*>(recv_tokens.data_ptr());
    params.err = static_cast<int32_t*>(err.data_ptr());
    params.tokens = static_cast<uint32_t>(tokens);
    params.tmax = static_cast<uint32_t>(tmax);
    params.slots = static_cast<uint32_t>(slots);
    params.rank = static_cast<uint32_t>(rank);
    params.experts_per_rank = static_cast<uint32_t>(experts_per_rank);
    params.invalid_id = static_cast<int32_t>(invalid_id);
    params.tab = tab;
    LaunchKernel(static_cast<uint32_t>(C_.unwrap()), kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(dispatch_kernel<kRanks, kTopK, kHidden, kX, kS, kUsePDL>, params);
  }

  static void combine(
      const tvm::ffi::TensorView peers,
      const tvm::ffi::TensorView partial,
      const tvm::ffi::TensorView rids,
      const tvm::ffi::TensorView recv_tokens,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView err,
      int64_t tmax,
      int64_t slots,
      int64_t rank,
      int64_t experts_per_rank,
      int64_t num_experts) {
    using namespace host;
    auto T_ = SymbolicSize{"tokens"};
    auto R_ = SymbolicSize{"rows"};
    auto C_ = SymbolicSize{"ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, static_cast<int64_t>(kHidden)})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(partial);
    TensorMatcher({R_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(rids);
    TensorMatcher({static_cast<int64_t>(kRanks)}).with_dtype<int32_t>().with_device(device).verify(recv_tokens);
    TensorMatcher({T_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({T_, static_cast<int64_t>(kHidden)})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(out);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(ctr);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(err);
    const int64_t tokens = T_.unwrap();
    CHECK_HOST(tokens <= tmax && tmax <= slots && slots <= kMaxTokens)
        << "ll_a2a combine: tokens " << tokens << " tmax " << tmax << " slots " << slots;
    CHECK_HOST(R_.unwrap() == kRanks * tmax) << "ll_a2a combine: " << R_.unwrap() << " partial rows, tmax " << tmax;
    CHECK_HOST(rank >= 0 && rank < kRanks && experts_per_rank > 0 && num_experts == kRanks * experts_per_rank);
    CombineParams params{};
    params.peers = peers_from(peers);
    params.partial = static_cast<const bf16_t*>(partial.data_ptr());
    params.rids = static_cast<const int32_t*>(rids.data_ptr());
    params.recv_tokens = static_cast<const int32_t*>(recv_tokens.data_ptr());
    params.ids = static_cast<const int32_t*>(ids.data_ptr());
    params.out = static_cast<bf16_t*>(out.data_ptr());
    params.ctr = static_cast<int32_t*>(ctr.data_ptr());
    params.err = static_cast<int32_t*>(err.data_ptr());
    params.tokens = static_cast<uint32_t>(tokens);
    params.tmax = static_cast<uint32_t>(tmax);
    params.slots = static_cast<uint32_t>(slots);
    params.rank = static_cast<uint32_t>(rank);
    params.experts_per_rank = static_cast<uint32_t>(experts_per_rank);
    params.num_experts = static_cast<uint32_t>(num_experts);
    LaunchKernel(static_cast<uint32_t>(C_.unwrap()), kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(combine_kernel<kRanks, kTopK, kHidden, kX, kS, kUsePDL>, params);
  }
};

}  // namespace qwen_dep_ll_a2a

using qwen_dep_ll_a2a::LlA2A;

}  // namespace sglang
