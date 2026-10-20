// Target-verify sampling on the vocab-parallel TP LM head under attention DP (SGLANG_Q35MF_LL16_VPSAMP).
// Candidate push: every rank sends each of its verify rows' next draft token to every rank.
// Shard pass: every rank reduces its vocab shard of every gathered row to per-chunk (max, sum-exp, argmax), the
// logit of the row's next draft token, and one token drawn from the shard's residual (the shard's distribution
// without that draft token), and sends them to the row's owner. Every word is self-validating against a marker
// (e2x_dep_xchg's slot protocol: two halves by a per-row epoch every rank flips per launch), so no fence is needed
// and the owner never reads another rank's logits.
// Decide pass (owner, one CTA per request): the chain acceptance of the one-rank chunked sampler
// (sharded_verify_sampling.py) over the payloads; the final draw picks a shard by residual mass in vocab order and
// takes that shard's drawn token.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

namespace ll16_vp {

inline constexpr uint32_t kMarker = 0xffffffffu;
inline constexpr uint32_t kNanKey = 0xffffffffu;
// A shard with no residual mass draws nothing.
inline constexpr uint32_t kNoToken = 0x7fffffffu;
inline constexpr uint32_t kSpinLimit = 1u << 26;
inline constexpr int kMaxWorld = 16;
// Must match CHUNKS / WORDS / MAX_D / MAX_CHUNK in ll16_vp_verify.py.
inline constexpr int kChunks = 4;
// Per (row, sender): kChunks words (max, sum-exp, argmax id, argmax-is-NaN), then (draft logit, drawn token, 0, 0).
inline constexpr int kWords = (kChunks + 1) * 4;
inline constexpr int kMaxD = 16;
inline constexpr int kMaxChunk = 4096;

SGL_DEVICE uint32_t order_key(float f) {
  uint32_t b = __float_as_uint(f);
  b = b == 0x80000000u ? 0u : b;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

SGL_DEVICE bool better(uint32_t v, int32_t i, uint32_t bv, int32_t bi) {
  return v > bv || (v == bv && i < bi);
}

SGL_DEVICE bool valid(float f) {
  return f == f && f != -INFINITY;
}

// (m, z) = (max, sum of exp(x - m)) over the finite or +inf logits seen so far; z == 0 means none.
SGL_DEVICE void merge(float& m, float& z, float om, float oz) {
  if (oz == 0.f) return;
  if (z == 0.f) {
    m = om;
    z = oz;
  } else if (om > m) {
    z = z * __expf(m - om) + oz;
    m = om;
  } else if (om == m) {
    z += oz;
  } else {
    z += oz * __expf(om - m);
  }
}

SGL_DEVICE bool has_marker(const uint4& v) {
  return v.x == kMarker || v.y == kMarker || v.z == kMarker || v.w == kMarker;
}

// Weight of logit f relative to a maximum m (1 at the maximum itself, so +inf maxima stay finite).
SGL_DEVICE float rel_weight(float f, float m) {
  return valid(f) ? (f == m ? 1.f : __expf(f - m)) : 0.f;
}

// Philox4x32-10 (Salmon et al., SC'11).
SGL_DEVICE uint4 philox(uint4 ctr, uint2 key) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, ctr.x);
    const uint32_t lo0 = 0xD2511F53u * ctr.x;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, ctr.z);
    const uint32_t lo1 = 0xCD9E8D57u * ctr.z;
    ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
  }
  return ctr;
}

SGL_DEVICE void st_relaxed_sys_u32(uint32_t* ptr, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

// Phase stamps for the T1 harness (null trace = off).
SGL_DEVICE void stamp(uint64_t* trace, int64_t slot) {
  if (trace != nullptr && threadIdx.x == 0) {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    trace[slot] = t;
  }
}

}  // namespace ll16_vp

struct Ll16VpCandParams {
  const int64_t* input_ids;   // [rows], this rank's verify rows (d per request)
  const uint64_t* peer_cand;  // [world] every rank's candidate words [2, g_max] (symmetric memory)
  const uint32_t* epoch;      // [g_max]
  uint64_t* rng;              // [2]: seed, launch counter
  int32_t rank;
  int32_t world;
  int32_t rows;
  int32_t d;
  int32_t g_max;
};

/// \brief One CTA: row r's next draft token (input_ids[r + 1] inside its request, else none), plus 1, to every
/// rank's candidate word of gathered row rank * rows + r; also advances the draw counter.
template <int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) ll16_vp_cand_kernel(const Ll16VpCandParams __grid_constant__ p) {
  using namespace device;
  PDLWaitPrimary<kUsePDL>();
  for (int r = static_cast<int>(threadIdx.x); r < p.rows; r += kThreads) {
    const int64_t next = r % p.d < p.d - 1 ? p.input_ids[r + 1] : -1;
    const uint32_t word = static_cast<uint32_t>(next + 1);
    const int64_t g = static_cast<int64_t>(p.rank) * p.rows + r;
    const int64_t at = static_cast<int64_t>(p.epoch[g] & 1u) * p.g_max + g;
    for (int w = 0; w < p.world; ++w)
      ll16_vp::st_relaxed_sys_u32(reinterpret_cast<uint32_t*>(p.peer_cand[w]) + at, word);
  }
  if (threadIdx.x == 0) p.rng[1] += 1;
}

struct Ll16VpShardParams {
  const bf16_t* logits;        // [g, ld], this rank's vocab shard of every gathered row (rank-major)
  uint32_t* slots;             // this rank's [2, g_max, world, kWords]
  const uint64_t* peer_slots;  // [world] every rank's slots (symmetric memory)
  uint32_t* cand;              // this rank's candidate words [2, g_max]
  uint32_t* epoch;             // [g_max]
  uint32_t* counter;           // [g_max], zero between launches
  float2* part;                // [g_max, kChunks] (max, sum-exp) of each chunk, for the row's last CTA
  float* xd;                   // [g_max] the row's draft logit when this shard holds the token
  const uint64_t* rng;         // [2]: seed, launch counter
  uint32_t* stats;             // [rows_max, world, kWords], this rank's rows
  int32_t* meta;               // [4]: [0] = rows per rank of the last exchange
  int64_t ld;
  int32_t vocab;  // shard width
  int32_t chunk;  // elements per chunk, a multiple of 8
  int32_t vocab_start;
  int32_t rank;
  int32_t world;
  int32_t rows;
  int32_t g_max;
  // Release fence before each push (only the proof variant, whose check reads peers' logits).
  int32_t fence;
  uint64_t* trace;  // [g_max, 8] globaltimer stamps or null
};

/// \brief The row's last CTA: one token from this shard's residual of row g (inverse CDF, chunk by chunk mass, then
/// a scan of the chosen chunk), with a uniform from the draw counter. Returns kNoToken when nothing remains.
template <int kThreads>
SGL_DEVICE uint32_t draw_residual(const Ll16VpShardParams& p, const bf16_t* row, int g, int nd_local, float x_d) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kWarps = kThreads / 32;
  constexpr int kPer = ll16_vp::kMaxChunk / 8 / kThreads;
  __shared__ int s_csel;
  __shared__ float s_u;
  __shared__ float s_mw;
  __shared__ float s_wsum[kWarps];
  __shared__ int s_wfirst[kWarps];
  __shared__ int s_wlast[kWarps];
  const int t = static_cast<int>(threadIdx.x);
  const int lane = t % 32;
  const int warp = t / 32;
  if (t == 0) {
    float mc[ll16_vp::kChunks];
    float zc[ll16_vp::kChunks];
    float m = -INFINITY;
#pragma unroll
    for (int c = 0; c < ll16_vp::kChunks; ++c) {
      const float2 pc = __ldcg(p.part + static_cast<int64_t>(g) * ll16_vp::kChunks + c);
      mc[c] = pc.x;
      zc[c] = pc.y;
      if (pc.y != 0.f) m = fmaxf(m, pc.x);
    }
    const int c_x = nd_local >= 0 ? nd_local / p.chunk : -1;
    const float e_x = nd_local >= 0 ? ll16_vp::rel_weight(x_d, m) : 0.f;
    float w[ll16_vp::kChunks];
    float total = 0.f;
#pragma unroll
    for (int c = 0; c < ll16_vp::kChunks; ++c) {
      float wc = zc[c] == 0.f ? 0.f : (mc[c] == m ? zc[c] : zc[c] * __expf(mc[c] - m));
      if (c == c_x) wc = fmaxf(wc - e_x, 0.f);
      w[c] = wc;
      total += wc;
    }
    const uint4 r = ll16_vp::philox(
        make_uint4(
            static_cast<uint32_t>(g),
            static_cast<uint32_t>(p.rank),
            static_cast<uint32_t>(p.rng[1]),
            static_cast<uint32_t>(p.rng[1] >> 32)),
        make_uint2(static_cast<uint32_t>(p.rng[0]), static_cast<uint32_t>(p.rng[0] >> 32)));
    const float u = static_cast<float>(r.x >> 8) * (1.f / 16777216.f) * total;
    int csel = -1;
    int clast = -1;
    float cum = 0.f;
    float u_in = 0.f;
#pragma unroll
    for (int c = 0; c < ll16_vp::kChunks; ++c) {
      if (csel < 0 && w[c] > 0.f && cum + w[c] > u) {
        csel = c;
        u_in = u - cum;
      }
      cum += w[c];
      if (w[c] > 0.f) clast = c;
    }
    if (csel < 0 && clast >= 0) {
      csel = clast;
      u_in = w[clast];
    }
    s_csel = csel;
    s_u = u_in;
    s_mw = m;
  }
  __syncthreads();
  const int csel = s_csel;
  if (csel < 0) return ll16_vp::kNoToken;
  const float m = s_mw;
  const float u_in = s_u;
  const int lo8 = min(csel * p.chunk, p.vocab) / 8;
  const int hi8 = min(csel * p.chunk + p.chunk, p.vocab) / 8;
  // Thread t holds kPer consecutive vectors, so the scan below runs in vocab order.
  vec_t x[kPer];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int v = lo8 + t * kPer + i;
    if (v < hi8) x[i].load(row, v);
  }
  float wv[kPer * 8];
  float tsum = 0.f;
  int last_valid = -1;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int v = lo8 + t * kPer + i;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int idx = v * 8 + e;
      const float we = (v < hi8 && idx != nd_local) ? ll16_vp::rel_weight(cast<fp32_t>(x[i][e]), m) : 0.f;
      wv[i * 8 + e] = we;
      tsum += we;
      if (we > 0.f) last_valid = idx;
    }
  }
  float incl = tsum;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float o = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += o;
  }
  if (lane == 31) s_wsum[warp] = incl;
  __syncthreads();
  float running = incl - tsum;
  for (int w = 0; w < warp; ++w)
    running += s_wsum[w];
  int first = 0x7fffffff;
#pragma unroll
  for (int i = 0; i < kPer * 8; ++i) {
    running += wv[i];
    if (first == 0x7fffffff && wv[i] > 0.f && running > u_in) first = (lo8 + t * kPer) * 8 + i;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    first = min(first, __shfl_xor_sync(0xffffffffu, first, off));
    last_valid = max(last_valid, __shfl_xor_sync(0xffffffffu, last_valid, off));
  }
  if (lane == 0) {
    s_wfirst[warp] = first;
    s_wlast[warp] = last_valid;
  }
  __syncthreads();
  int f = 0x7fffffff;
  int l = -1;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    f = min(f, s_wfirst[w]);
    l = max(l, s_wlast[w]);
  }
  const int pick = f != 0x7fffffff ? f : l;
  return pick >= 0 ? static_cast<uint32_t>(p.vocab_start + pick) : ll16_vp::kNoToken;
}

/**
 * \brief Grid (kChunks, world * rows): CTA (c, g) reduces chunk c of gathered row g on this rank's shard and sends
 * its word to rank g / rows. The row's last CTA (atomic counter) draws the shard's residual token, sends the row
 * word, re-arms the candidate word and flips the row's epoch; on the owner it first waits for every rank's words of
 * row g and copies them to the local stats.
 */
template <int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) ll16_vp_shard_kernel(const Ll16VpShardParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kWarps = kThreads / 32;
  // Vectors per thread: every load of the chunk is issued before any reduction.
  constexpr int kPer = ll16_vp::kMaxChunk / 8 / kThreads;
  __shared__ float s_m[kWarps];
  __shared__ float s_z[kWarps];
  __shared__ uint32_t s_k[kWarps];
  __shared__ int32_t s_i[kWarps];
  __shared__ int s_nd;
  __shared__ bool s_last;

  PDLWaitPrimary<kUsePDL>();

  const int c = static_cast<int>(blockIdx.x);
  const int g = static_cast<int>(blockIdx.y);
  const int t = static_cast<int>(threadIdx.x);
  const int warp = t / 32;
  const int lane = static_cast<int>(get_lane_id());
  if (c == 0) ll16_vp::stamp(p.trace, g * 8 + 0);
  const bf16_t* row = p.logits + static_cast<int64_t>(g) * p.ld;
  // Issued with the chunk loads so the push at the end waits on nothing.
  const int owner = g / p.rows;
  const uint32_t half = p.epoch[g] & 1u;
  uint32_t* const owner_slots = reinterpret_cast<uint32_t*>(p.peer_slots[owner]);
  const int lo8 = min(c * p.chunk, p.vocab) / 8;
  const int hi8 = min(c * p.chunk + p.chunk, p.vocab) / 8;
  vec_t x[kPer];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int v = lo8 + t + i * kThreads;
    if (v < hi8) x[i].load(row, v);
  }
  uint32_t* const cand = p.cand + static_cast<int64_t>(half) * p.g_max + g;
  if (t == 0) {
    uint32_t word;
    uint32_t spins = 0;
    do {
      word = ptx::load_relaxed_sys(cand);
      if (++spins == ll16_vp::kSpinLimit) __trap();
    } while (word == ll16_vp::kMarker);
    s_nd = static_cast<int>(static_cast<int64_t>(word) - 1 - p.vocab_start);
  }
  // Two branch-free passes (max, then sum-exp). A thread visits its ids in increasing order, so a strictly
  // larger key keeps the lowest id among equal ones.
  uint32_t best = 0u;
  int32_t bidx = 0x7fffffff;
  float m = -INFINITY;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int v = lo8 + t + i * kThreads;
    if (v >= hi8) continue;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float f = cast<fp32_t>(x[i][e]);
      const uint32_t key = f == f ? ll16_vp::order_key(f) : ll16_vp::kNanKey;
      const bool take = key > best;
      best = take ? key : best;
      bidx = take ? v * 8 + e : bidx;
      m = ll16_vp::valid(f) ? fmaxf(m, f) : m;
    }
  }
  float z = 0.f;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int v = lo8 + t + i * kThreads;
    if (v >= hi8) continue;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      z += ll16_vp::rel_weight(cast<fp32_t>(x[i][e]), m);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float om = __shfl_xor_sync(0xffffffffu, m, off);
    const float oz = __shfl_xor_sync(0xffffffffu, z, off);
    ll16_vp::merge(m, z, om, oz);
    const uint32_t ov = __shfl_xor_sync(0xffffffffu, best, off);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, bidx, off);
    if (ll16_vp::better(ov, oi, best, bidx)) {
      best = ov;
      bidx = oi;
    }
  }
  if (lane == 0) {
    s_m[warp] = m;
    s_z[warp] = z;
    s_k[warp] = best;
    s_i[warp] = bidx;
  }
  __syncthreads();
  if (c == 0) ll16_vp::stamp(p.trace, g * 8 + 1);

  const int nd_local = s_nd >= 0 && s_nd < p.vocab ? s_nd : -1;
  const int64_t row_words = (static_cast<int64_t>(half) * p.g_max + g) * p.world * ll16_vp::kWords;
  uint32_t* const dst = owner_slots + row_words + p.rank * ll16_vp::kWords;
  if (t == 0) {
    float bm = s_m[0];
    float bz = s_z[0];
    uint32_t k = s_k[0];
    int32_t i = s_i[0];
    for (int w = 1; w < kWarps; ++w) {
      ll16_vp::merge(bm, bz, s_m[w], s_z[w]);
      if (ll16_vp::better(s_k[w], s_i[w], k, i)) {
        k = s_k[w];
        i = s_i[w];
      }
    }
    bm = bz == 0.f ? -INFINITY : bm;
    p.part[static_cast<int64_t>(g) * ll16_vp::kChunks + c] = make_float2(bm, bz);
    if (nd_local >= lo8 * 8 && nd_local < hi8 * 8) {
      const float f = cast<fp32_t>(row[nd_local]);
      p.xd[g] = f == f ? f : -INFINITY;
    }
    const uint4 word = make_uint4(
        __float_as_uint(bm),
        __float_as_uint(bz),
        static_cast<uint32_t>(p.vocab_start + i),
        k == ll16_vp::kNanKey ? 1u : 0u);
    if (p.fence) ptx::fence_release_sys();
    ptx::st_relaxed_16B(word, dst, c);
    if (c == 0) ll16_vp::stamp(p.trace, g * 8 + 2);
    __threadfence();
    s_last = atomicAdd(p.counter + g, 1u) == static_cast<uint32_t>(ll16_vp::kChunks - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  const float x_d = nd_local >= 0 ? __ldcg(p.xd + g) : -INFINITY;
  const uint32_t token = draw_residual<kThreads>(p, row, g, nd_local, x_d);
  if (t == 0) {
    if (p.fence) ptx::fence_release_sys();
    ptx::st_relaxed_16B(make_uint4(__float_as_uint(x_d), token, 0u, 0u), dst, ll16_vp::kChunks);
  }
  if (owner == p.rank) {
    uint4* incoming = reinterpret_cast<uint4*>(p.slots + row_words);
    const int nvec = p.world * (ll16_vp::kWords / 4);
    const int64_t j = g - static_cast<int64_t>(p.rank) * p.rows;
    for (int q = t; q < nvec; q += kThreads) {
      uint4 r;
      uint32_t spins = 0;
      do {
        ptx::ld_relaxed_16B(r, incoming, q);
        if (++spins == ll16_vp::kSpinLimit) __trap();
      } while (ll16_vp::has_marker(r));
      reinterpret_cast<uint4*>(p.stats + j * p.world * ll16_vp::kWords)[q] = r;
      // Re-arm the word this thread read; the senders' next store to it is two launches away.
      const uint4 marker = make_uint4(ll16_vp::kMarker, ll16_vp::kMarker, ll16_vp::kMarker, ll16_vp::kMarker);
      ptx::st_global_16B(marker, incoming, q);
    }
    __syncthreads();
    ll16_vp::stamp(p.trace, g * 8 + 3);
    if (t == 0 && j == 0) p.meta[0] = p.rows;
  }
  if (t == 0) {
    // Every CTA of the row read its candidate word before its count; the owner's next store is two launches away.
    *cand = ll16_vp::kMarker;
    p.counter[g] = 0u;
    p.epoch[g] = half ^ 1u;
  }
  ll16_vp::stamp(p.trace, g * 8 + 4);
}

struct Ll16VpDecideParams {
  const uint32_t* stats;          // [rows_max, world, kWords]
  const int64_t* candidates;      // [bs, d]
  const int64_t* retrieve_index;  // [bs, d]
  const float* coins;             // [bs, d]
  const float* coins_final;       // [bs]
  int32_t* predict;               // [bs * d]
  int32_t* accept_index;          // [bs, accept_width]
  int32_t* num_correct;           // [bs]
  int64_t* argmax;                // [bs * d]
  int32_t vocab;                  // shard width
  int32_t world;
  int32_t d;
  int32_t accept_width;
  int32_t sample;
  uint64_t* trace;  // [bs, 8] globaltimer stamps or null
};

/**
 * \brief Grid (bs). Per request: the argmax of every row (ties to the lowest id, NaN first), and with `sample` the
 * chain acceptance with the stock coins at temperature 1 and the final draw: a shard by residual mass of the last
 * accepted row (vocab order, the stock final coin), then that shard's drawn token.
 */
template <int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) ll16_vp_decide_kernel(const Ll16VpDecideParams __grid_constant__ p) {
  using namespace device;
  // Every per-request input the serial chain reads is staged in shared memory first (one load latency).
  __shared__ float s_mw[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ float s_sw[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ float s_xw[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ uint32_t s_tok[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ uint32_t s_nan[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ uint32_t s_id[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ float s_mass[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ int64_t s_cand[ll16_vp::kMaxD];
  __shared__ int32_t s_ret[ll16_vp::kMaxD];
  __shared__ float s_coin[ll16_vp::kMaxD];
  __shared__ float s_coin_final;
  __shared__ float s_q[ll16_vp::kMaxD];
  __shared__ float s_M[ll16_vp::kMaxD];
  __shared__ float s_S[ll16_vp::kMaxD];
  __shared__ int64_t s_arg[ll16_vp::kMaxD];

  PDLWaitPrimary<kUsePDL>();

  const int b = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(threadIdx.x);
  const int64_t row0 = static_cast<int64_t>(b) * p.d;
  ll16_vp::stamp(p.trace, b * 8 + 0);
  if (t < p.d) {
    s_cand[t] = p.candidates[row0 + t];
    s_ret[t] = static_cast<int32_t>(p.retrieve_index[row0 + t]);
    if (p.sample) s_coin[t] = p.coins[row0 + t];
    if (p.sample && t == 0) s_coin_final = p.coins_final[b];
  }
  for (int pair = t; pair < p.d * p.world; pair += kThreads) {
    const int j = pair / p.world;
    const int w = pair % p.world;
    const uint4* pl = reinterpret_cast<const uint4*>(p.stats + ((row0 + j) * p.world + w) * ll16_vp::kWords);
    uint4 cw[ll16_vp::kChunks + 1];
#pragma unroll
    for (int c = 0; c <= ll16_vp::kChunks; ++c)
      cw[c] = pl[c];
    float m = -INFINITY;
    uint32_t bk = 0u;
    uint32_t bid = 0u;
    uint32_t bnan = 0u;
#pragma unroll
    for (int c = 0; c < ll16_vp::kChunks; ++c) {
      const float mc = __uint_as_float(cw[c].x);
      m = fmaxf(m, mc);
      // Chunks are in vocab order: a later chunk wins only with a strictly larger key.
      const uint32_t key = cw[c].w != 0u ? ll16_vp::kNanKey : ll16_vp::order_key(mc);
      if (c == 0 || key > bk) {
        bk = key;
        bid = cw[c].z;
        bnan = cw[c].w;
      }
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < ll16_vp::kChunks; ++c) {
      const float mc = __uint_as_float(cw[c].x);
      const float zc = __uint_as_float(cw[c].y);
      if (zc != 0.f) s += mc == m ? zc : zc * __expf(mc - m);
    }
    s_mw[j][w] = m;
    s_sw[j][w] = s;
    s_id[j][w] = bid;
    s_nan[j][w] = bnan;
    s_xw[j][w] = __uint_as_float(cw[ll16_vp::kChunks].x);
    s_tok[j][w] = cw[ll16_vp::kChunks].y;
  }
  __syncthreads();
  if (t < p.d) {
    float M = -INFINITY;
    for (int w = 0; w < p.world; ++w)
      M = fmaxf(M, s_mw[t][w]);
    float S = 0.f;
    uint32_t bk = 0u;
    int64_t bid = 0;
    for (int w = 0; w < p.world; ++w) {
      const float mw = s_mw[t][w];
      const float sw = s_sw[t][w];
      if (sw != 0.f) S += mw == M ? sw : sw * __expf(mw - M);
      const uint32_t key = s_nan[t][w] != 0u ? ll16_vp::kNanKey : ll16_vp::order_key(mw);
      // Shards are in vocab order: a later shard wins only with a strictly larger key.
      if (w == 0 || key > bk) {
        bk = key;
        bid = static_cast<int64_t>(s_id[t][w]);
      }
    }
    s_M[t] = M;
    s_S[t] = S;
    s_arg[t] = bid;
    p.argmax[row0 + t] = bid;
  }
  if (!p.sample) return;
  __syncthreads();
  // Every probability the serial chain may need, in parallel: q of each draft token (its logit comes from the one
  // shard that holds it; the others send -inf), and the shard masses of every row.
  if (t < p.d - 1) {
    const float M = s_M[t];
    float xd = -INFINITY;
    for (int w = 0; w < p.world; ++w)
      xd = fmaxf(xd, s_xw[t][w]);
    s_q[t] = xd > -INFINITY ? __expf(xd - (M > -INFINITY ? M : 0.f)) / s_S[t] : 0.f;
  }
  for (int pair = t; pair < p.d * p.world; pair += kThreads) {
    const int r = pair / p.world;
    const int w = pair % p.world;
    const float M = s_M[r];
    const float mw = s_mw[r][w];
    s_mass[r][w] = mw > -INFINITY ? s_sw[r][w] * __expf(mw - (M > -INFINITY ? M : 0.f)) / s_S[r] : 0.f;
  }
  __syncthreads();
  ll16_vp::stamp(p.trace, b * 8 + 1);

  if (t == 0) {
    float coin = s_coin[0];
    int32_t last_ret = s_ret[0];
    p.accept_index[static_cast<int64_t>(b) * p.accept_width] = last_ret;
    int32_t accepted = 0;
    int32_t cur_row = 0;
    bool alive = true;
    int64_t excl = -1;
    float q_excl = 0.f;
    for (int j = 1; j < p.d; ++j) {
      if (j >= p.accept_width) break;
      const float q = s_q[j - 1];
      const int64_t dtok = s_cand[j];
      const bool acc = alive && q > 0.f && (coin < q || q >= 1.f);
      if (acc) {
        p.predict[last_ret] = static_cast<int32_t>(dtok);
        const int32_t ret_j = s_ret[j];
        ++accepted;
        p.accept_index[static_cast<int64_t>(b) * p.accept_width + accepted] = ret_j;
        last_ret = ret_j;
        cur_row = j;
        coin = s_coin[j];
      } else if (alive) {
        excl = dtok;
        q_excl = q;
      }
      alive = alive && acc;
    }
    p.num_correct[b] = accepted;

    // Final draw from row cur_row minus the rejected draft's mass: a shard in vocab order, then its drawn token.
    const int r = cur_row;
    const int excl_owner = excl >= 0 ? static_cast<int>(excl / p.vocab) : -1;
    float total = 0.f;
    for (int w = 0; w < p.world; ++w) {
      float mass = s_mass[r][w];
      if (w == excl_owner) mass = mass - q_excl;
      total += fmaxf(mass, 0.f);
    }
    const float u = s_coin_final * total;
    float cum = 0.f;
    int sel = -1;
    int last_pos = -1;
    for (int w = 0; w < p.world; ++w) {
      float mass = s_mass[r][w];
      if (w == excl_owner) mass = mass - q_excl;
      mass = fmaxf(mass, 0.f);
      if (sel < 0 && mass > 0.f && cum + mass > u) sel = w;
      cum += mass;
      if (mass > 0.f) last_pos = w;
    }
    if (sel < 0) sel = last_pos;
    const uint32_t tok = sel >= 0 ? s_tok[r][sel] : ll16_vp::kNoToken;
    // No residual mass anywhere (an all -inf row): the row's argmax keeps a valid id.
    p.predict[last_ret] = tok != ll16_vp::kNoToken ? static_cast<int32_t>(tok) : static_cast<int32_t>(s_arg[r]);
    ll16_vp::stamp(p.trace, b * 8 + 4);
  }
}

struct Ll16VpPullParams {
  const uint64_t* peer_logits;  // [world]
  const int32_t* meta;          // [0] = rows per rank of the exchange
  float* out;                   // [rows, out_stride]
  int64_t out_stride;
  int64_t ld;
  int32_t vocab;       // shard width
  int32_t vocab_size;  // columns written
  int32_t rank;
};

/// \brief Grid (rows, world): this rank's row r of shard w into out[r, w * vocab : (w + 1) * vocab] as fp32 (the
/// proof's check only; its exchange runs with the release fence).
template <int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) ll16_vp_pull_kernel(const Ll16VpPullParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  PDLWaitPrimary<kUsePDL>();
  const int r = static_cast<int>(blockIdx.x);
  const int w = static_cast<int>(blockIdx.y);
  const int64_t g = static_cast<int64_t>(p.rank) * p.meta[0] + r;
  const bf16_t* src = reinterpret_cast<const bf16_t*>(p.peer_logits[w]) + g * p.ld;
  float* dst = p.out + r * p.out_stride + static_cast<int64_t>(w) * p.vocab;
  const int64_t col0 = static_cast<int64_t>(w) * p.vocab;
  constexpr int kBatch = 8;  // remote loads in flight per thread
  const int nvec = p.vocab / 8;
  for (int base = static_cast<int>(threadIdx.x); base < nvec; base += kThreads * kBatch) {
    vec_t xs[kBatch];
#pragma unroll
    for (int k = 0; k < kBatch; ++k) {
      const int v = base + k * kThreads;
      if (v < nvec) ptx::ld_relaxed_16B(xs[k], src, v);
    }
#pragma unroll
    for (int k = 0; k < kBatch; ++k) {
      const int v = base + k * kThreads;
      if (v >= nvec) continue;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (col0 + v * 8 + e < p.vocab_size) dst[v * 8 + e] = cast<fp32_t>(xs[k][e]);
      }
    }
  }
}

template <bool kUsePDL>
struct Ll16VpVerify {
  static constexpr int kCandThreads = 128;
  static constexpr int kShardThreads = 256;
  static constexpr int kDecideThreads = 128;
  static constexpr int kPullThreads = 256;
  static_assert(ll16_vp::kMaxChunk % (8 * kShardThreads) == 0, "chunk vectors split evenly over the threads");

  static void cand(
      const tvm::ffi::TensorView input_ids,
      const tvm::ffi::TensorView peer_cand,
      const tvm::ffi::TensorView epoch,
      const tvm::ffi::TensorView rng,
      int64_t rank,
      int64_t d) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto WS = SymbolicSize{"world"};
    auto GM = SymbolicSize{"g_max"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R}).with_dtype<int64_t>().with_device(device).verify(input_ids);
    TensorMatcher({WS}).with_dtype<int64_t>().with_device(device).verify(peer_cand);
    TensorMatcher({GM}).with_dtype<int32_t>().with_device(device).verify(epoch);
    TensorMatcher({2}).with_dtype<int64_t>().with_device(device).verify(rng);
    const int64_t rows = R.unwrap();
    const int64_t world = WS.unwrap();
    CHECK_HOST(world >= 1 && world <= ll16_vp::kMaxWorld) << "world " << world;
    CHECK_HOST(rows >= 1 && rows * world <= GM.unwrap()) << "rows " << rows;
    CHECK_HOST(d >= 1 && rows % d == 0) << "rows " << rows << " not a multiple of d " << d;
    CHECK_HOST(rank >= 0 && rank < world) << "rank " << rank;
    const auto params = Ll16VpCandParams{
        .input_ids = static_cast<const int64_t*>(input_ids.data_ptr()),
        .peer_cand = static_cast<const uint64_t*>(peer_cand.data_ptr()),
        .epoch = static_cast<const uint32_t*>(epoch.data_ptr()),
        .rng = static_cast<uint64_t*>(rng.data_ptr()),
        .rank = static_cast<int32_t>(rank),
        .world = static_cast<int32_t>(world),
        .rows = static_cast<int32_t>(rows),
        .d = static_cast<int32_t>(d),
        .g_max = static_cast<int32_t>(GM.unwrap()),
    };
    LaunchKernel(dim3(1), kCandThreads, device.unwrap())
        .enable_pdl(kUsePDL)(ll16_vp_cand_kernel<kCandThreads, kUsePDL>, params);
  }

  static void shard(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView slots,
      const tvm::ffi::TensorView peer_slots,
      const tvm::ffi::TensorView cand,
      const tvm::ffi::TensorView epoch,
      const tvm::ffi::TensorView counter,
      const tvm::ffi::TensorView part,
      const tvm::ffi::TensorView xd,
      const tvm::ffi::TensorView rng,
      const tvm::ffi::TensorView stats,
      const tvm::ffi::TensorView meta,
      const tvm::ffi::TensorView trace,
      int64_t chunk,
      int64_t vocab_start,
      int64_t rank,
      int64_t rows,
      int64_t fence) {
    using namespace host;
    auto G = SymbolicSize{"gathered_rows"};
    auto V = SymbolicSize{"vocab_local"};
    auto LD = SymbolicSize{"logits_row_stride"};
    auto GM = SymbolicSize{"g_max"};
    auto WS = SymbolicSize{"world"};
    auto RM = SymbolicSize{"rows_max"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({G, V}).with_strides({LD, 1}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(
        logits);
    TensorMatcher({2, GM, WS, ll16_vp::kWords})
        .with_dtype<int32_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(slots);
    TensorMatcher({WS}).with_dtype<int64_t>().with_device(device).verify(peer_slots);
    TensorMatcher({2, GM}).with_dtype<int32_t>().with_device(device).verify(cand);
    TensorMatcher({GM}).with_dtype<int32_t>().with_device(device).verify(epoch);
    TensorMatcher({GM}).with_dtype<int32_t>().with_device(device).verify(counter);
    TensorMatcher({GM, ll16_vp::kChunks, 2}).with_dtype<fp32_t>().with_device(device).verify(part);
    TensorMatcher({GM}).with_dtype<fp32_t>().with_device(device).verify(xd);
    TensorMatcher({2}).with_dtype<int64_t>().with_device(device).verify(rng);
    TensorMatcher({RM, WS, ll16_vp::kWords})
        .with_dtype<int32_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(stats);
    TensorMatcher({4}).with_dtype<int32_t>().with_device(device).verify(meta);
    const int64_t world = WS.unwrap();
    const int64_t g = G.unwrap();
    CHECK_HOST(world >= 1 && world <= ll16_vp::kMaxWorld) << "world " << world;
    CHECK_HOST(rows >= 1 && g == world * rows) << "gathered rows " << g << " != world " << world << " x rows " << rows;
    CHECK_HOST(g <= GM.unwrap() && rows <= RM.unwrap()) << "more rows than exchange slots";
    CHECK_HOST(rank >= 0 && rank < world) << "rank " << rank;
    CHECK_HOST(V.unwrap() % 8 == 0 && LD.unwrap() % 8 == 0) << "logits rows must be 16-byte vectors";
    CHECK_HOST(chunk % 8 == 0 && chunk <= ll16_vp::kMaxChunk && chunk * ll16_vp::kChunks >= V.unwrap())
        << "chunk " << chunk;
    const auto params = Ll16VpShardParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .slots = static_cast<uint32_t*>(slots.data_ptr()),
        .peer_slots = static_cast<const uint64_t*>(peer_slots.data_ptr()),
        .cand = static_cast<uint32_t*>(cand.data_ptr()),
        .epoch = static_cast<uint32_t*>(epoch.data_ptr()),
        .counter = static_cast<uint32_t*>(counter.data_ptr()),
        .part = static_cast<float2*>(part.data_ptr()),
        .xd = static_cast<float*>(xd.data_ptr()),
        .rng = static_cast<const uint64_t*>(rng.data_ptr()),
        .stats = static_cast<uint32_t*>(stats.data_ptr()),
        .meta = static_cast<int32_t*>(meta.data_ptr()),
        .ld = LD.unwrap(),
        .vocab = static_cast<int32_t>(V.unwrap()),
        .chunk = static_cast<int32_t>(chunk),
        .vocab_start = static_cast<int32_t>(vocab_start),
        .rank = static_cast<int32_t>(rank),
        .world = static_cast<int32_t>(world),
        .rows = static_cast<int32_t>(rows),
        .g_max = static_cast<int32_t>(GM.unwrap()),
        .fence = static_cast<int32_t>(fence != 0),
        .trace = trace.numel() >= g * 8 ? static_cast<uint64_t*>(trace.data_ptr()) : nullptr,
    };
    LaunchKernel(
        dim3(static_cast<uint32_t>(ll16_vp::kChunks), static_cast<uint32_t>(g)), kShardThreads, device.unwrap())
        .enable_pdl(kUsePDL)(ll16_vp_shard_kernel<kShardThreads, kUsePDL>, params);
  }

  static void decide(
      const tvm::ffi::TensorView stats,
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView retrieve_index,
      const tvm::ffi::TensorView coins,
      const tvm::ffi::TensorView coins_final,
      const tvm::ffi::TensorView predict,
      const tvm::ffi::TensorView accept_index,
      const tvm::ffi::TensorView num_correct,
      const tvm::ffi::TensorView argmax,
      const tvm::ffi::TensorView trace,
      int64_t vocab,
      int64_t sample) {
    using namespace host;
    auto RM = SymbolicSize{"rows_max"};
    auto WS = SymbolicSize{"world"};
    auto B = SymbolicSize{"bs"};
    auto D = SymbolicSize{"draft_token_num"};
    auto AW = SymbolicSize{"accept_width"};
    auto N = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({RM, WS, ll16_vp::kWords})
        .with_dtype<int32_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(stats);
    TensorMatcher({B, D}).with_dtype<int64_t>().with_device(device).verify(candidates);
    TensorMatcher({B, D}).with_dtype<int64_t>().with_device(device).verify(retrieve_index);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(argmax);
    const int64_t bs = B.unwrap();
    const int64_t d = D.unwrap();
    const int64_t world = WS.unwrap();
    CHECK_HOST(N.unwrap() == bs * d) << "argmax rows";
    CHECK_HOST(bs * d <= RM.unwrap()) << "more rows than stats";
    CHECK_HOST(d >= 1 && d <= ll16_vp::kMaxD) << "draft_token_num " << d;
    CHECK_HOST(world >= 1 && world <= ll16_vp::kMaxWorld) << "world " << world;
    const float* coins_ptr = nullptr;
    const float* coins_final_ptr = nullptr;
    int32_t* predict_ptr = nullptr;
    int32_t* accept_ptr = nullptr;
    int32_t* num_ptr = nullptr;
    int64_t accept_width = 1;
    if (sample) {
      TensorMatcher({B, D}).with_dtype<fp32_t>().with_device(device).verify(coins);
      TensorMatcher({B}).with_dtype<fp32_t>().with_device(device).verify(coins_final);
      TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(predict);
      TensorMatcher({B, AW}).with_dtype<int32_t>().with_device(device).verify(accept_index);
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(num_correct);
      coins_ptr = static_cast<const float*>(coins.data_ptr());
      coins_final_ptr = static_cast<const float*>(coins_final.data_ptr());
      predict_ptr = static_cast<int32_t*>(predict.data_ptr());
      accept_ptr = static_cast<int32_t*>(accept_index.data_ptr());
      num_ptr = static_cast<int32_t*>(num_correct.data_ptr());
      accept_width = AW.unwrap();
    }
    const auto params = Ll16VpDecideParams{
        .stats = static_cast<const uint32_t*>(stats.data_ptr()),
        .candidates = static_cast<const int64_t*>(candidates.data_ptr()),
        .retrieve_index = static_cast<const int64_t*>(retrieve_index.data_ptr()),
        .coins = coins_ptr,
        .coins_final = coins_final_ptr,
        .predict = predict_ptr,
        .accept_index = accept_ptr,
        .num_correct = num_ptr,
        .argmax = static_cast<int64_t*>(argmax.data_ptr()),
        .vocab = static_cast<int32_t>(vocab),
        .world = static_cast<int32_t>(world),
        .d = static_cast<int32_t>(d),
        .accept_width = static_cast<int32_t>(accept_width),
        .sample = static_cast<int32_t>(sample != 0),
        .trace = trace.numel() >= bs * 8 ? static_cast<uint64_t*>(trace.data_ptr()) : nullptr,
    };
    LaunchKernel(dim3(static_cast<uint32_t>(bs)), kDecideThreads, device.unwrap())
        .enable_pdl(kUsePDL)(ll16_vp_decide_kernel<kDecideThreads, kUsePDL>, params);
  }

  static void pull(
      const tvm::ffi::TensorView peer_logits,
      const tvm::ffi::TensorView meta,
      const tvm::ffi::TensorView out,
      int64_t ld,
      int64_t vocab,
      int64_t rank) {
    using namespace host;
    auto WS = SymbolicSize{"world"};
    auto R = SymbolicSize{"rows"};
    auto VO = SymbolicSize{"vocab_size"};
    auto OS = SymbolicSize{"out_row_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({WS}).with_dtype<int64_t>().with_device(device).verify(peer_logits);
    TensorMatcher({4}).with_dtype<int32_t>().with_device(device).verify(meta);
    TensorMatcher({R, VO}).with_strides({OS, 1}).with_dtype<fp32_t>().with_device(device).verify(out);
    const int64_t world = WS.unwrap();
    CHECK_HOST(vocab % 8 == 0 && ld % 8 == 0) << "logits rows must be 16-byte vectors";
    CHECK_HOST(VO.unwrap() <= vocab * world) << "vocab_size exceeds the shards";
    if (R.unwrap() == 0) return;
    const auto params = Ll16VpPullParams{
        .peer_logits = static_cast<const uint64_t*>(peer_logits.data_ptr()),
        .meta = static_cast<const int32_t*>(meta.data_ptr()),
        .out = static_cast<float*>(out.data_ptr()),
        .out_stride = OS.unwrap(),
        .ld = ld,
        .vocab = static_cast<int32_t>(vocab),
        .vocab_size = static_cast<int32_t>(VO.unwrap()),
        .rank = static_cast<int32_t>(rank),
    };
    LaunchKernel(dim3(static_cast<uint32_t>(R.unwrap()), static_cast<uint32_t>(world)), kPullThreads, device.unwrap())
        .enable_pdl(kUsePDL)(ll16_vp_pull_kernel<kPullThreads, kUsePDL>, params);
  }
};

}  // namespace sglang
