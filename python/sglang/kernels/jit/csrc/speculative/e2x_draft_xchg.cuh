// TP2 draft-step boundary in one launch (SGLANG_Q35MF_E2X_DRAFT_FUSE): the shard argmax of
// `draft_argmax_xchg_kernel`, a peer exchange of each rank's candidate with its load-time
// memo row (the embedding-half fc partial) and, from the hidden rank, the hidden-half fc
// partial, then the rank-order selection and bf16(p_embed + p_hidden) of `fc_xchg_sum_kernel`.
// Same keys and ties as the argmax exchange; the fc output equals the split fc bit for bit
// when the memo and the hidden partial come from the served norm + cuBLAS calls.
//
// Slots hold fp32 words: [memo row H][hidden partial H][meta 4]. Arrival is detected from the
// data (Lamport): every word holds kFp32Marker until the peer's store lands; partials are sent
// with NaNs canonicalized and the meta words are 16-bit halves or indices, so no data word
// equals the marker. Rows alternate between two halves by a per-row epoch flipped in lockstep.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace e2x_xchg {

inline constexpr uint32_t kMarker = 0xffffffffu;
inline constexpr uint32_t kCanonicalNaN = 0x7fffffffu;
inline constexpr uint32_t kSpinLimit = 1u << 26;

SGL_DEVICE uint32_t order_key(float f) {
  uint32_t b = __float_as_uint(f);
  b = b == 0x80000000u ? 0u : b;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

SGL_DEVICE uint32_t value_key(float f, uint32_t nan_key) {
  return f == f ? order_key(f) : nan_key;
}

SGL_DEVICE bool better(uint32_t v, int32_t i, uint32_t bv, int32_t bi) {
  return v > bv || (v == bv && i < bi);
}

SGL_DEVICE void warp_argmax(uint32_t& v, int32_t& i) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint32_t ov = __shfl_xor_sync(0xffffffffu, v, off);
    const int32_t oi = __shfl_xor_sync(0xffffffffu, i, off);
    if (better(ov, oi, v, i)) {
      v = ov;
      i = oi;
    }
  }
}

template <int kThreads>
SGL_DEVICE void block_argmax(uint32_t& v, int32_t& i, uint32_t* s_v, int32_t* s_i) {
  constexpr int kWarps = kThreads / 32;
  warp_argmax(v, i);
  const int warp = static_cast<int>(threadIdx.x / 32);
  if (device::get_lane_id() == 0) {
    s_v[warp] = v;
    s_i[warp] = i;
  }
  __syncthreads();
  v = s_v[0];
  i = s_i[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    if (better(s_v[w], s_i[w], v, i)) {
      v = s_v[w];
      i = s_i[w];
    }
  }
}

SGL_DEVICE uint32_t canonical_nan(uint32_t bits) {
  return (bits & 0x7fffffffu) > 0x7f800000u ? kCanonicalNaN : bits;
}

SGL_DEVICE uint4 canonical_nan4(uint4 v) {
  return make_uint4(canonical_nan(v.x), canonical_nan(v.y), canonical_nan(v.z), canonical_nan(v.w));
}

SGL_DEVICE bool has_marker(const uint4& v) {
  return v.x == kMarker || v.y == kMarker || v.z == kMarker || v.w == kMarker;
}

}  // namespace e2x_xchg

struct E2xDraftXchgParams {
  const bf16_t* logits;                // [M, V_local], written by the lm_head GEMM
  const float* __restrict__ memo;      // [V_local, H] embedding-half fc partial per local row
  const float* hidden_partial;         // [M, H] this rank's hidden-half partial (hidden rank), or null
  uint32_t* part_key;                  // [M, kBlocks]
  int32_t* part_idx;                   // [M, kBlocks]
  uint32_t* counter;                   // [M], zero between launches
  float* slots;                        // this rank's rows [2, max_rows, 2, kRowWords]
  float* peer_slots;                   // the peer's rows, same layout
  uint32_t* epoch;                     // [max_rows], local
  int64_t* topk_index;                 // [M]
  float* topk_p;                       // [M]
  int64_t* draft_tokens;               // [M, S] or null
  int64_t* positions;                  // [M] or null
  int64_t* mrope;                      // [3, >= M] or null
  bf16_t* fc_out;                      // [M, H], or null for a token-only exchange
  int32_t vocab;
  int32_t vocab_start;
  uint32_t nan_key;
  int32_t rank;
  int32_t hidden_rank;
  int32_t max_rows;
  int32_t draft_column;
  int32_t draft_stride;
  int32_t mrope_stride;
};

/**
 * \brief Shard argmax, peer exchange of candidate meta + memo row (+ hidden partial), selection
 * and the fc output, one launch. Grid (kBlocks, M); the last CTA of a row (atomic counter)
 * reduces the partial argmaxes and runs the exchange for that row.
 */
template <int kH, int kBlocks, int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) e2x_draft_xchg_kernel(const E2xDraftXchgParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kHVecs = kH / 4;                 // float4 vectors per H
  constexpr int kRowWords = 2 * kH + 4;          // memo | hidden | meta
  constexpr int kMetaVec = 2 * kHVecs;           // float4 index of the meta vector
  static_assert(kH % 4 == 0);
  __shared__ uint32_t s_v[kThreads / 32];
  __shared__ int32_t s_i[kThreads / 32];
  __shared__ bool s_last;
  __shared__ int s_win;
  __shared__ int32_t s_wi;

  PDLWaitPrimary<kUsePDL>();

  const int64_t m = blockIdx.y;
  const bf16_t* logits = p.logits + m * p.vocab;
  uint32_t* part_key = p.part_key + m * kBlocks;
  int32_t* part_idx = p.part_idx + m * kBlocks;
  uint32_t* counter = p.counter + m;
  uint32_t best = 0u;
  int32_t bidx = 0x7fffffff;
  const int nvec = p.vocab / 8;
  for (int v = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x); v < nvec;
       v += kBlocks * kThreads) {
    vec_t x;
    x.load(logits, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t key = e2x_xchg::value_key(cast<fp32_t>(x[j]), p.nan_key);
      const int32_t idx = v * 8 + j;
      if (e2x_xchg::better(key, idx, best, bidx)) {
        best = key;
        bidx = idx;
      }
    }
  }
  e2x_xchg::block_argmax<kThreads>(best, bidx, s_v, s_i);
  if (threadIdx.x == 0) {
    part_key[blockIdx.x] = best;
    part_idx[blockIdx.x] = bidx;
    __threadfence();
    s_last = atomicAdd(counter, 1u) == kBlocks - 1;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  uint32_t v = 0u;
  int32_t i = 0x7fffffff;
  if (threadIdx.x < kBlocks) {
    v = __ldcg(part_key + threadIdx.x);
    i = __ldcg(part_idx + threadIdx.x);
  }
  __syncthreads();
  e2x_xchg::block_argmax<kThreads>(v, i, s_v, s_i);
  // v / i: this rank's candidate (key, local row), identical in every thread.

  const bool payload = p.fc_out != nullptr;
  const bool send_hidden = payload && p.rank == p.hidden_rank;
  const bool recv_hidden = payload && p.rank != p.hidden_rank;
  const uint32_t half = p.epoch[m] & 1u;
  const int64_t base = (static_cast<int64_t>(half) * p.max_rows + m) * 2 * kRowWords;
  float* remote = p.peer_slots + base + static_cast<int64_t>(p.rank) * kRowWords;    // my row, at the peer
  float* incoming = p.slots + base + static_cast<int64_t>(1 - p.rank) * kRowWords;   // the peer's row, here
  const int32_t own_token = p.vocab_start + i;

  if (payload) {
    const uint4* mrow = reinterpret_cast<const uint4*>(p.memo + static_cast<int64_t>(i) * kH);
    for (int c = static_cast<int>(threadIdx.x); c < kHVecs; c += kThreads)
      ptx::st_relaxed_16B(e2x_xchg::canonical_nan4(mrow[c]), remote, c);
    if (send_hidden) {
      const uint4* hrow = reinterpret_cast<const uint4*>(p.hidden_partial + m * kH);
      for (int c = static_cast<int>(threadIdx.x); c < kHVecs; c += kThreads)
        ptx::st_relaxed_16B(e2x_xchg::canonical_nan4(hrow[c]), remote, kHVecs + c);
    }
  }
  if (threadIdx.x == 0) {
    // 16-bit key halves keep every meta word away from the marker.
    const uint4 meta = make_uint4(v & 0xffffu, v >> 16, static_cast<uint32_t>(own_token), 0u);
    ptx::st_relaxed_16B(meta, remote, kMetaVec);
    *counter = 0u;
  }

  // Wait for every vector the peer sends; each thread polls its own share.
  const int n_poll = payload ? (recv_hidden ? 2 * kHVecs : kHVecs) : 0;
  for (int c = static_cast<int>(threadIdx.x); c <= n_poll; c += kThreads) {
    const int vec = c == n_poll ? kMetaVec : c;
    uint4 r;
    uint32_t spins = 0;
    do {
      ptx::ld_relaxed_16B(r, incoming, vec);
      if (++spins == e2x_xchg::kSpinLimit) __trap();
    } while (e2x_xchg::has_marker(r));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    uint4 meta;
    ptx::ld_relaxed_16B(meta, incoming, kMetaVec);
    const uint32_t pv = meta.x | (meta.y << 16);
    const int32_t pt = static_cast<int32_t>(meta.z);
    // Rank order: rank 0's candidate first, as the select kernel walks the gathered rows.
    uint32_t bv;
    int32_t bt;
    int win;
    if (p.rank == 0) {
      bv = v, bt = own_token, win = 0;
      if (e2x_xchg::better(pv, pt, bv, bt)) bv = pv, bt = pt, win = 1;
    } else {
      bv = pv, bt = pt, win = 1;
      if (e2x_xchg::better(v, own_token, bv, bt)) bv = v, bt = own_token, win = 0;
    }
    s_win = win;  // 0: this rank's candidate, 1: the peer's
    s_wi = bt;
  }
  __syncthreads();
  const int win = s_win;
  const int32_t token = s_wi;

  if (payload) {
    const uint4* wrow = win == 0 ? reinterpret_cast<const uint4*>(p.memo + static_cast<int64_t>(i) * kH) : nullptr;
    const uint4* hrow = send_hidden ? reinterpret_cast<const uint4*>(p.hidden_partial + m * kH) : nullptr;
    auto* dst = reinterpret_cast<uint2*>(p.fc_out + m * kH);
    for (int c = static_cast<int>(threadIdx.x); c < kHVecs; c += kThreads) {
      uint4 a, h;
      if (win == 0) {
        a = e2x_xchg::canonical_nan4(wrow[c]);
      } else {
        ptx::ld_relaxed_16B(a, incoming, c);
      }
      if (send_hidden) {
        h = hrow[c];
      } else {
        ptx::ld_relaxed_16B(h, incoming, kHVecs + c);
      }
      // fp32 a + b == b + a: the rank-order sum of fc_xchg_sum on either rank.
      const float s0 = __uint_as_float(a.x) + __uint_as_float(h.x);
      const float s1 = __uint_as_float(a.y) + __uint_as_float(h.y);
      const float s2 = __uint_as_float(a.z) + __uint_as_float(h.z);
      const float s3 = __uint_as_float(a.w) + __uint_as_float(h.w);
      __nv_bfloat162 lo = __floats2bfloat162_rn(s0, s1);
      __nv_bfloat162 hi = __floats2bfloat162_rn(s2, s3);
      uint2 packed;
      packed.x = *reinterpret_cast<uint32_t*>(&lo);
      packed.y = *reinterpret_cast<uint32_t*>(&hi);
      dst[c] = packed;
    }
  }
  if (threadIdx.x == 0) {
    p.topk_index[m] = token;
    p.topk_p[m] = 1.0f;
    if (p.draft_tokens != nullptr) p.draft_tokens[m * p.draft_stride + p.draft_column] = token;
    if (p.positions != nullptr) p.positions[m] = p.positions[m] + 1;
    if (p.mrope != nullptr) {
#pragma unroll
      for (int d = 0; d < 3; ++d)
        p.mrope[d * p.mrope_stride + m] += 1;
    }
  }
  // Re-arm what this rank read from the peer once every thread is done, then flip the half.
  __syncthreads();
  const uint4 marker = make_uint4(e2x_xchg::kMarker, e2x_xchg::kMarker, e2x_xchg::kMarker, e2x_xchg::kMarker);
  for (int c = static_cast<int>(threadIdx.x); c <= n_poll; c += kThreads)
    ptx::st_global_16B(marker, incoming, c == n_poll ? kMetaVec : c);
  if (threadIdx.x == 0) p.epoch[m] = p.epoch[m] ^ 1u;
}

template <int kH, int kBlocks, bool kUsePDL>
struct E2xDraftXchg {
  static constexpr int kThreads = 256;
  static constexpr int kRowWords = 2 * kH + 4;
  static_assert(kBlocks <= kThreads, "partials are reduced by one CTA");
  static_assert(kH % 8 == 0);

  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView memo,
      const tvm::ffi::TensorView hidden_partial,
      const tvm::ffi::TensorView part_key,
      const tvm::ffi::TensorView part_idx,
      const tvm::ffi::TensorView counter,
      const tvm::ffi::TensorView slots,
      int64_t peer_slots,
      const tvm::ffi::TensorView epoch,
      const tvm::ffi::TensorView topk_index,
      const tvm::ffi::TensorView topk_p,
      const tvm::ffi::TensorView draft_tokens,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView mrope_positions,
      const tvm::ffi::TensorView fc_out,
      int64_t vocab_start,
      int64_t nan_key,
      int64_t rank,
      int64_t hidden_rank,
      int64_t draft_column) {
    using namespace host;
    auto M = SymbolicSize{"rows"};
    auto V = SymbolicSize{"vocab_local"};
    auto R = SymbolicSize{"max_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, V}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({V, kH}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(memo);
    TensorMatcher({M, kBlocks}).with_dtype<int32_t>().with_device(device).verify(part_key);
    TensorMatcher({M, kBlocks}).with_dtype<int32_t>().with_device(device).verify(part_idx);
    TensorMatcher({M}).with_dtype<int32_t>().with_device(device).verify(counter);
    TensorMatcher({2, R, 2, kRowWords}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(slots);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device).verify(epoch);
    TensorMatcher({M, 1}).with_dtype<int64_t>().with_device(device).verify(topk_index);
    TensorMatcher({M, 1}).with_dtype<fp32_t>().with_device(device).verify(topk_p);
    const int64_t vocab = V.unwrap();
    CHECK_HOST(vocab % 8 == 0) << "local vocab " << vocab << " must be a multiple of 8";
    CHECK_HOST(M.unwrap() <= R.unwrap()) << "more rows than exchange slots";
    CHECK_HOST(rank == 0 || rank == 1) << "two-rank exchange";
    CHECK_HOST(hidden_rank == 0 || hidden_rank == 1) << "two-rank exchange";
    CHECK_HOST(peer_slots != 0 && peer_slots % 16 == 0) << "peer slots must be a 16-byte aligned pointer";
    bf16_t* out = nullptr;
    if (fc_out.numel() > 0) {
      TensorMatcher({M, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(fc_out);
      out = static_cast<bf16_t*>(fc_out.data_ptr());
    }
    const float* hp = nullptr;
    if (out != nullptr && rank == hidden_rank) {
      TensorMatcher({M, kH}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(hidden_partial);
      hp = static_cast<const float*>(hidden_partial.data_ptr());
    }
    int64_t* pos = nullptr;
    if (positions.numel() > 0) {
      TensorMatcher({M}).with_dtype<int64_t>().with_device(device).verify(positions);
      pos = static_cast<int64_t*>(positions.data_ptr());
    }
    int64_t* mrope = nullptr;
    int32_t mrope_stride = 0;
    if (mrope_positions.numel() > 0) {
      auto N = SymbolicSize{"mrope_tokens"};
      auto S = SymbolicSize{"mrope_row_stride"};
      TensorMatcher({3, N}).with_strides({S, 1}).with_dtype<int64_t>().with_device(device).verify(mrope_positions);
      CHECK_HOST(N.unwrap() == M.unwrap()) << "mrope positions must have one column per row";
      mrope = static_cast<int64_t*>(mrope_positions.data_ptr());
      mrope_stride = static_cast<int32_t>(S.unwrap());
    }
    int64_t* tokens = nullptr;
    int32_t stride = 0;
    if (draft_tokens.numel() > 0) {
      auto S = SymbolicSize{"num_steps"};
      TensorMatcher({M, S}).with_dtype<int64_t>().with_device(device).verify(draft_tokens);
      CHECK_HOST(draft_column >= 0 && draft_column < S.unwrap()) << "draft column out of range";
      tokens = static_cast<int64_t*>(draft_tokens.data_ptr());
      stride = static_cast<int32_t>(S.unwrap());
    }
    const auto params = E2xDraftXchgParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .memo = static_cast<const float*>(memo.data_ptr()),
        .hidden_partial = hp,
        .part_key = static_cast<uint32_t*>(part_key.data_ptr()),
        .part_idx = static_cast<int32_t*>(part_idx.data_ptr()),
        .counter = static_cast<uint32_t*>(counter.data_ptr()),
        .slots = static_cast<float*>(slots.data_ptr()),
        .peer_slots = reinterpret_cast<float*>(peer_slots),
        .epoch = static_cast<uint32_t*>(epoch.data_ptr()),
        .topk_index = static_cast<int64_t*>(topk_index.data_ptr()),
        .topk_p = static_cast<float*>(topk_p.data_ptr()),
        .draft_tokens = tokens,
        .positions = pos,
        .mrope = mrope,
        .fc_out = out,
        .vocab = static_cast<int32_t>(vocab),
        .vocab_start = static_cast<int32_t>(vocab_start),
        .nan_key = static_cast<uint32_t>(nan_key),
        .rank = static_cast<int32_t>(rank),
        .hidden_rank = static_cast<int32_t>(hidden_rank),
        .max_rows = static_cast<int32_t>(R.unwrap()),
        .draft_column = static_cast<int32_t>(draft_column),
        .draft_stride = stride,
        .mrope_stride = mrope_stride,
    };
    LaunchKernel(dim3(kBlocks, static_cast<uint32_t>(M.unwrap())), kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(e2x_draft_xchg_kernel<kH, kBlocks, kThreads, kUsePDL>, params);
  }
};

}  // namespace sglang
