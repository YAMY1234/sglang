// Target-verify sampling on the vocab-parallel TP LM head under attention DP (SGLANG_Q35MF_LL16_VPSAMP).
// Shard pass: every rank reduces its vocab shard of every gathered row to per-chunk (max, sum-exp) and the
// argmax (torch.argmax rule) and sends that 48B payload to the row's owner, with e2x_dep_xchg's slot protocol
// (marker words, two halves by a per-row epoch every rank flips per launch); the owner's CTA for the row waits
// for every rank, copies the payloads to its local stats and re-arms the slots.
// Decide pass (owner, one CTA per request): the chain acceptance and final draw of the one-rank chunked
// sampler (sharded_verify_sampling.py) over the shard payloads; the draft tokens' logits and the drawn chunk
// are read from the shards that hold them (symmetric memory).

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
inline constexpr uint32_t kSpinLimit = 1u << 26;
inline constexpr int kMaxWorld = 16;
// Must match CHUNKS / WORDS / MAX_D / MAX_CHUNK in ll16_vp_verify.py.
inline constexpr int kChunks = 4;
inline constexpr int kWords = 12;  // m[4], z[4], argmax id, argmax-is-NaN, 0, 0
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

SGL_DEVICE float bf16_bits_to_float(uint32_t bits16) {
  return __uint_as_float(bits16 << 16);
}

SGL_DEVICE float load_peer_bf16(const bf16_t* ptr) {
  uint16_t raw;
  asm volatile("ld.relaxed.sys.global.b16 %0, [%1];" : "=h"(raw) : "l"(ptr) : "memory");
  return bf16_bits_to_float(raw);
}

}  // namespace ll16_vp

struct Ll16VpShardParams {
  const bf16_t* logits;        // [g, ld], this rank's vocab shard of every gathered row (rank-major)
  uint32_t* slots;             // this rank's [2, g_max, world, kWords]
  const uint64_t* peer_slots;  // [world] every rank's slots (symmetric memory)
  uint32_t* epoch;             // [g_max]
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
};

/**
 * \brief Grid (world * rows): CTA g reduces gathered row g on this rank's shard and sends the payload to rank
 * g / rows; on that rank it then waits for every rank's payload of row g and copies them to the local stats.
 */
template <int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) ll16_vp_shard_kernel(const Ll16VpShardParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kWarps = kThreads / 32;
  __shared__ float s_m[kWarps];
  __shared__ float s_z[kWarps];
  __shared__ uint32_t s_k[kWarps];
  __shared__ int32_t s_i[kWarps];
  __shared__ float s_cm[ll16_vp::kChunks];
  __shared__ float s_cz[ll16_vp::kChunks];

  PDLWaitPrimary<kUsePDL>();

  const int g = static_cast<int>(blockIdx.x);
  const int warp = static_cast<int>(threadIdx.x / 32);
  const bf16_t* row = p.logits + static_cast<int64_t>(g) * p.ld;
  uint32_t best = 0u;
  int32_t bidx = 0x7fffffff;
#pragma unroll 1
  for (int c = 0; c < ll16_vp::kChunks; ++c) {
    const int lo = min(c * p.chunk, p.vocab);
    const int hi = min(lo + p.chunk, p.vocab);
    float m = -INFINITY;
    float z = 0.f;
    for (int v = lo / 8 + static_cast<int>(threadIdx.x); v < hi / 8; v += kThreads) {
      vec_t x;
      x.load(row, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float f = cast<fp32_t>(x[e]);
        const uint32_t key = f == f ? ll16_vp::order_key(f) : ll16_vp::kNanKey;
        const int32_t idx = v * 8 + e;
        if (ll16_vp::better(key, idx, best, bidx)) {
          best = key;
          bidx = idx;
        }
        if (f == f && f != -INFINITY) {
          if (f > m) {
            z = z * __expf(m - f) + 1.f;
            m = f;
          } else if (f == m) {
            z += 1.f;
          } else {
            z += __expf(f - m);
          }
        }
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float om = __shfl_xor_sync(0xffffffffu, m, off);
      const float oz = __shfl_xor_sync(0xffffffffu, z, off);
      ll16_vp::merge(m, z, om, oz);
    }
    if (get_lane_id() == 0) {
      s_m[warp] = m;
      s_z[warp] = z;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bm = s_m[0];
      float bz = s_z[0];
      for (int w = 1; w < kWarps; ++w)
        ll16_vp::merge(bm, bz, s_m[w], s_z[w]);
      s_cm[c] = bz == 0.f ? -INFINITY : bm;
      s_cz[c] = bz;
    }
    __syncthreads();
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint32_t ov = __shfl_xor_sync(0xffffffffu, best, off);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, bidx, off);
    if (ll16_vp::better(ov, oi, best, bidx)) {
      best = ov;
      bidx = oi;
    }
  }
  if (get_lane_id() == 0) {
    s_k[warp] = best;
    s_i[warp] = bidx;
  }
  __syncthreads();

  const int owner = g / p.rows;
  const uint32_t half = p.epoch[g] & 1u;
  const int64_t row_words = (static_cast<int64_t>(half) * p.g_max + g) * p.world * ll16_vp::kWords;
  if (threadIdx.x == 0) {
    uint32_t k = s_k[0];
    int32_t i = s_i[0];
    for (int w = 1; w < kWarps; ++w) {
      if (ll16_vp::better(s_k[w], s_i[w], k, i)) {
        k = s_k[w];
        i = s_i[w];
      }
    }
    const uint4 w0 = make_uint4(
        __float_as_uint(s_cm[0]), __float_as_uint(s_cm[1]), __float_as_uint(s_cm[2]), __float_as_uint(s_cm[3]));
    const uint4 w1 = make_uint4(
        __float_as_uint(s_cz[0]), __float_as_uint(s_cz[1]), __float_as_uint(s_cz[2]), __float_as_uint(s_cz[3]));
    const uint4 w2 = make_uint4(static_cast<uint32_t>(p.vocab_start + i), k == ll16_vp::kNanKey ? 1u : 0u, 0u, 0u);
    uint32_t* dst = reinterpret_cast<uint32_t*>(p.peer_slots[owner]) + row_words + p.rank * ll16_vp::kWords;
    // The owner later reads this row's logits from our shard: publish them with the payload.
    ptx::fence_release_sys();
    ptx::st_relaxed_16B(w0, dst, 0);
    ptx::st_relaxed_16B(w1, dst, 1);
    ptx::st_relaxed_16B(w2, dst, 2);
  }
  if (owner != p.rank) {
    if (threadIdx.x == 0) p.epoch[g] = half ^ 1u;
    return;
  }

  uint4* incoming = reinterpret_cast<uint4*>(p.slots + row_words);
  const int nvec = p.world * (ll16_vp::kWords / 4);
  const int64_t j = g - static_cast<int64_t>(p.rank) * p.rows;
  if (static_cast<int>(threadIdx.x) < nvec) {
    uint4 r;
    uint32_t spins = 0;
    do {
      ptx::ld_relaxed_16B(r, incoming, threadIdx.x);
      if (++spins == ll16_vp::kSpinLimit) __trap();
    } while (ll16_vp::has_marker(r));
    reinterpret_cast<uint4*>(p.stats + j * p.world * ll16_vp::kWords)[threadIdx.x] = r;
    // Re-arm the word this thread read; the senders' next store to it is two launches away.
    const uint4 marker = make_uint4(ll16_vp::kMarker, ll16_vp::kMarker, ll16_vp::kMarker, ll16_vp::kMarker);
    ptx::st_global_16B(marker, incoming, threadIdx.x);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    p.epoch[g] = half ^ 1u;
    if (j == 0) p.meta[0] = p.rows;
  }
}

struct Ll16VpDecideParams {
  const uint32_t* stats;          // [rows_max, world, kWords]
  const int32_t* meta;            // [0] = rows per rank of the exchange
  const uint64_t* peer_logits;    // [world] every rank's shard logits (symmetric memory)
  const int64_t* candidates;      // [bs, d]
  const int64_t* retrieve_index;  // [bs, d]
  const float* coins;             // [bs, d]
  const float* coins_final;       // [bs]
  int32_t* predict;               // [bs * d]
  int32_t* accept_index;          // [bs, accept_width]
  int32_t* num_correct;           // [bs]
  int64_t* argmax;                // [bs * d]
  int64_t ld;
  int32_t vocab;  // shard width
  int32_t chunk;
  int32_t rank;
  int32_t world;
  int32_t d;
  int32_t accept_width;
  int32_t sample;
};

/**
 * \brief Grid (bs). Per request: the argmax of every row (ties to the lowest id, NaN first), and with `sample`
 * the chain acceptance with the stock coins at temperature 1 and the final draw from the residual of the last
 * accepted row, as sample_chain_full_vocab.
 */
template <int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) ll16_vp_decide_kernel(const Ll16VpDecideParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kWarps = kThreads / 32;
  constexpr int kMaxPer = ll16_vp::kMaxChunk / 8 / kThreads;
  __shared__ float s_mw[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ float s_sw[ll16_vp::kMaxD][ll16_vp::kMaxWorld];
  __shared__ float s_M[ll16_vp::kMaxD];
  __shared__ float s_S[ll16_vp::kMaxD];
  __shared__ float s_xd[ll16_vp::kMaxD];
  __shared__ int32_t s_sel;
  __shared__ int32_t s_row;
  __shared__ int32_t s_csel;
  __shared__ float s_u2;
  __shared__ int32_t s_excl_local;
  __shared__ int32_t s_last_ret;
  __shared__ float s_warp_sum[kWarps];
  __shared__ int32_t s_warp_first[kWarps];
  __shared__ int32_t s_warp_last[kWarps];

  PDLWaitPrimary<kUsePDL>();

  const int b = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(threadIdx.x);
  const int warp = t / 32;
  const int lane = static_cast<int>(get_lane_id());
  const int64_t row0 = static_cast<int64_t>(b) * p.d;
  for (int pair = t; pair < p.d * p.world; pair += kThreads) {
    const int j = pair / p.world;
    const int w = pair % p.world;
    const uint32_t* pl = p.stats + ((row0 + j) * p.world + w) * ll16_vp::kWords;
    float m = -INFINITY;
#pragma unroll
    for (int c = 0; c < ll16_vp::kChunks; ++c)
      m = fmaxf(m, __uint_as_float(pl[c]));
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < ll16_vp::kChunks; ++c) {
      const float zc = __uint_as_float(pl[ll16_vp::kChunks + c]);
      const float mc = __uint_as_float(pl[c]);
      if (zc != 0.f) s += mc == m ? zc : zc * __expf(mc - m);
    }
    s_mw[j][w] = m;
    s_sw[j][w] = s;
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
      const uint32_t* pl = p.stats + ((row0 + t) * p.world + w) * ll16_vp::kWords;
      const uint32_t key = pl[ll16_vp::kChunks * 2 + 1] != 0u ? ll16_vp::kNanKey : ll16_vp::order_key(mw);
      // Shards are in vocab order: a later shard wins only with a strictly larger key.
      if (w == 0 || key > bk) {
        bk = key;
        bid = static_cast<int64_t>(pl[ll16_vp::kChunks * 2]);
      }
    }
    s_M[t] = M;
    s_S[t] = S;
    p.argmax[row0 + t] = bid;
  }
  if (!p.sample) return;

  const int rows_pad = p.meta[0];
  const int64_t vocab_total = static_cast<int64_t>(p.vocab) * p.world;
  if (t < p.d - 1) {
    // Row t verifies draft token t + 1.
    const int64_t tok = p.candidates[row0 + t + 1];
    float xd = -INFINITY;
    if (tok >= 0 && tok < vocab_total) {
      const int w = static_cast<int>(tok / p.vocab);
      const int64_t g = static_cast<int64_t>(p.rank) * rows_pad + row0 + t;
      const bf16_t* src =
          reinterpret_cast<const bf16_t*>(p.peer_logits[w]) + g * p.ld + (tok - static_cast<int64_t>(w) * p.vocab);
      xd = ll16_vp::load_peer_bf16(src);
      if (xd != xd) xd = -INFINITY;
    }
    s_xd[t] = xd;
  }
  __syncthreads();

  if (t == 0) {
    float coin = p.coins[row0];
    int32_t last_ret = static_cast<int32_t>(p.retrieve_index[row0]);
    p.accept_index[static_cast<int64_t>(b) * p.accept_width] = last_ret;
    int32_t accepted = 0;
    int32_t cur_row = 0;
    bool alive = true;
    int64_t excl = -1;
    float q_excl = 0.f;
    for (int j = 1; j < p.d; ++j) {
      if (j >= p.accept_width) break;
      const int r = j - 1;
      const float M = s_M[r];
      const float m_safe = M > -INFINITY ? M : 0.f;
      const float xd = s_xd[r];
      const float q = xd > -INFINITY ? __expf(xd - m_safe) / s_S[r] : 0.f;
      const int64_t dtok = p.candidates[row0 + j];
      const bool acc = alive && q > 0.f && (coin < q || q >= 1.f);
      if (acc) {
        p.predict[last_ret] = static_cast<int32_t>(dtok);
        const int32_t ret_j = static_cast<int32_t>(p.retrieve_index[row0 + j]);
        ++accepted;
        p.accept_index[static_cast<int64_t>(b) * p.accept_width + accepted] = ret_j;
        last_ret = ret_j;
        cur_row = j;
        coin = p.coins[row0 + j];
      } else if (alive) {
        excl = dtok;
        q_excl = q;
      }
      alive = alive && acc;
    }
    p.num_correct[b] = accepted;

    // Final draw from row cur_row minus the rejected draft's mass: shard, then chunk, in vocab order.
    const int r = cur_row;
    const float M = s_M[r];
    const float m_safe = M > -INFINITY ? M : 0.f;
    const float S = s_S[r];
    const int excl_owner = excl >= 0 ? static_cast<int>(excl / p.vocab) : -1;
    float total = 0.f;
    for (int w = 0; w < p.world; ++w) {
      const float mw = s_mw[r][w];
      float mass = mw > -INFINITY ? s_sw[r][w] * __expf(mw - m_safe) / S : 0.f;
      if (w == excl_owner) mass = mass - q_excl;
      total += fmaxf(mass, 0.f);
    }
    const float u = p.coins_final[b] * total;
    float cum = 0.f;
    int sel = -1;
    float u_sel = 0.f;
    int last_pos = 0;
    float last_mass = 0.f;
    for (int w = 0; w < p.world; ++w) {
      const float mw = s_mw[r][w];
      float mass = mw > -INFINITY ? s_sw[r][w] * __expf(mw - m_safe) / S : 0.f;
      if (w == excl_owner) mass = mass - q_excl;
      mass = fmaxf(mass, 0.f);
      if (sel < 0 && cum + mass > u) {
        sel = w;
        u_sel = u - cum;
      }
      cum += mass;
      if (mass > 0.f) {
        last_pos = w;
        last_mass = mass;
      }
    }
    if (sel < 0) {
      u_sel = last_mass;
      sel = last_pos;
    }
    const int64_t excl_local = excl - static_cast<int64_t>(sel) * p.vocab;
    const bool excl_here = excl >= 0 && excl_local >= 0 && excl_local < p.vocab;
    const uint32_t* pl = p.stats + ((row0 + r) * p.world + sel) * ll16_vp::kWords;
    float cmass[ll16_vp::kChunks];
    for (int c = 0; c < ll16_vp::kChunks; ++c) {
      const float mc = __uint_as_float(pl[c]);
      const float zc = __uint_as_float(pl[ll16_vp::kChunks + c]);
      float mass = mc > -INFINITY ? zc * __expf(mc - m_safe) / S : 0.f;
      if (excl_here && c == static_cast<int>(excl_local / p.chunk)) mass = fmaxf(mass - q_excl, 0.f);
      cmass[c] = mass;
    }
    const int nchunk = (p.vocab + p.chunk - 1) / p.chunk;
    int c_sel = -1;
    int c_last = -1;
    float ccum = 0.f;
    float before = 0.f;
    for (int c = 0; c < ll16_vp::kChunks && c < nchunk; ++c) {
      if (c_sel < 0 && ccum + cmass[c] > u_sel) {
        c_sel = c;
        before = ccum;
      }
      ccum += cmass[c];
      if (cmass[c] > 0.f) c_last = c;
    }
    float u2;
    if (c_sel >= 0) {
      u2 = u_sel - before;
    } else {
      c_sel = c_last > 0 ? c_last : 0;
      u2 = cmass[c_sel];
    }
    s_sel = sel;
    s_row = r;
    s_csel = c_sel;
    s_u2 = u2;
    s_excl_local = excl_here ? static_cast<int32_t>(excl_local) : -1;
    s_last_ret = last_ret;
  }
  __syncthreads();

  // The drawn chunk, read from its shard: inclusive scan of q in vocab order, first index past u2.
  const int sel = s_sel;
  const int r = s_row;
  const float M = s_M[r];
  const float m_safe = M > -INFINITY ? M : 0.f;
  const float S = s_S[r];
  const float u2 = s_u2;
  const int excl_local = s_excl_local;
  const int lo = s_csel * p.chunk;
  const int hi = min(lo + p.chunk, p.vocab);
  const int nvec = (hi - lo) / 8;
  const int per = (nvec + kThreads - 1) / kThreads;
  const int64_t g = static_cast<int64_t>(p.rank) * rows_pad + row0 + r;
  const bf16_t* src = reinterpret_cast<const bf16_t*>(p.peer_logits[sel]) + g * p.ld + lo;
  float q[kMaxPer * 8];
  float tsum = 0.f;
  int last_valid = -1;
#pragma unroll
  for (int i = 0; i < kMaxPer; ++i) {
    const int v = t * per + i;
    if (i < per && v < nvec) {
      vec_t x;
      ptx::ld_relaxed_16B(x, src, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float f = cast<fp32_t>(x[e]);
        const int idx = lo + v * 8 + e;
        float qe = (f == f && f > -INFINITY) ? __expf(f - m_safe) / S : 0.f;
        if (idx == excl_local) qe = 0.f;
        q[i * 8 + e] = qe;
        tsum += qe;
        if (qe > 0.f) last_valid = idx;
      }
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        q[i * 8 + e] = 0.f;
    }
  }
  // Exclusive scan of the per-thread sums (threads hold consecutive ranges).
  float incl = tsum;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float o = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += o;
  }
  if (lane == 31) s_warp_sum[warp] = incl;
  __syncthreads();
  float base = 0.f;
  for (int w = 0; w < warp; ++w)
    base += s_warp_sum[w];
  float running = base + incl - tsum;
  int first = 0x7fffffff;
#pragma unroll
  for (int i = 0; i < kMaxPer; ++i) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int v = t * per + i;
      if (i < per && v < nvec) {
        running += q[i * 8 + e];
        if (first == 0x7fffffff && running > u2) first = lo + v * 8 + e;
      }
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    first = min(first, __shfl_xor_sync(0xffffffffu, first, off));
    last_valid = max(last_valid, __shfl_xor_sync(0xffffffffu, last_valid, off));
  }
  if (lane == 0) {
    s_warp_first[warp] = first;
    s_warp_last[warp] = last_valid;
  }
  __syncthreads();
  if (t == 0) {
    int f = 0x7fffffff;
    int l = -1;
    for (int w = 0; w < kWarps; ++w) {
      f = min(f, s_warp_first[w]);
      l = max(l, s_warp_last[w]);
    }
    const int pick = f != 0x7fffffff ? f : (l >= 0 ? l : p.vocab - 1);
    p.predict[s_last_ret] = static_cast<int32_t>(static_cast<int64_t>(sel) * p.vocab + pick);
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

/// \brief Grid (rows, world): this rank's row r of shard w into out[r, w * vocab : (w + 1) * vocab] as fp32.
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
  for (int v = static_cast<int>(threadIdx.x); v < p.vocab / 8; v += kThreads) {
    vec_t x;
    ptx::ld_relaxed_16B(x, src, v);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      if (col0 + v * 8 + e < p.vocab_size) dst[v * 8 + e] = cast<fp32_t>(x[e]);
    }
  }
}

template <bool kUsePDL>
struct Ll16VpVerify {
  static constexpr int kShardThreads = 256;
  static constexpr int kDecideThreads = 128;
  static constexpr int kPullThreads = 256;
  static_assert(ll16_vp::kMaxChunk % (8 * kDecideThreads) == 0, "chunk vectors split evenly over the threads");

  static void shard(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView slots,
      const tvm::ffi::TensorView peer_slots,
      const tvm::ffi::TensorView epoch,
      const tvm::ffi::TensorView stats,
      const tvm::ffi::TensorView meta,
      int64_t chunk,
      int64_t vocab_start,
      int64_t rank,
      int64_t rows) {
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
    TensorMatcher({GM}).with_dtype<int32_t>().with_device(device).verify(epoch);
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
        .epoch = static_cast<uint32_t*>(epoch.data_ptr()),
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
    };
    LaunchKernel(dim3(static_cast<uint32_t>(g)), kShardThreads, device.unwrap())
        .enable_pdl(kUsePDL)(ll16_vp_shard_kernel<kShardThreads, kUsePDL>, params);
  }

  static void decide(
      const tvm::ffi::TensorView stats,
      const tvm::ffi::TensorView meta,
      const tvm::ffi::TensorView peer_logits,
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView retrieve_index,
      const tvm::ffi::TensorView coins,
      const tvm::ffi::TensorView coins_final,
      const tvm::ffi::TensorView predict,
      const tvm::ffi::TensorView accept_index,
      const tvm::ffi::TensorView num_correct,
      const tvm::ffi::TensorView argmax,
      int64_t ld,
      int64_t vocab,
      int64_t chunk,
      int64_t rank,
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
    TensorMatcher({RM, WS, ll16_vp::kWords}).with_dtype<int32_t>().with_device(device).verify(stats);
    TensorMatcher({4}).with_dtype<int32_t>().with_device(device).verify(meta);
    TensorMatcher({WS}).with_dtype<int64_t>().with_device(device).verify(peer_logits);
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
    CHECK_HOST(chunk % 8 == 0 && chunk <= ll16_vp::kMaxChunk && chunk * ll16_vp::kChunks >= vocab) << "chunk " << chunk;
    CHECK_HOST(vocab % 8 == 0 && ld % 8 == 0) << "logits rows must be 16-byte vectors";
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
        .meta = static_cast<const int32_t*>(meta.data_ptr()),
        .peer_logits = static_cast<const uint64_t*>(peer_logits.data_ptr()),
        .candidates = static_cast<const int64_t*>(candidates.data_ptr()),
        .retrieve_index = static_cast<const int64_t*>(retrieve_index.data_ptr()),
        .coins = coins_ptr,
        .coins_final = coins_final_ptr,
        .predict = predict_ptr,
        .accept_index = accept_ptr,
        .num_correct = num_ptr,
        .argmax = static_cast<int64_t*>(argmax.data_ptr()),
        .ld = ld,
        .vocab = static_cast<int32_t>(vocab),
        .chunk = static_cast<int32_t>(chunk),
        .rank = static_cast<int32_t>(rank),
        .world = static_cast<int32_t>(world),
        .d = static_cast<int32_t>(d),
        .accept_width = static_cast<int32_t>(accept_width),
        .sample = static_cast<int32_t>(sample != 0),
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
