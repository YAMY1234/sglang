// DEP draft-step token select in one launch (SGLANG_Q35MF_DEP_DX_XCHG): replaces the K75P shard
// argmax, pair all-gather, select and topk1 finish (plus mrope + 1) with the same keys and ties.
// Slots are 16B words holding the marker until the sender's store lands (key halves and token ids
// never equal it); rows alternate between two halves by a per-row epoch every rank flips per launch.
// An owner returns only after every rank sent, i.e. after every rank's lm_head GEMM read the gathered
// rows: the guarantee K75P's row gather (no entry barrier) took from the pair all-gather.
// With the memo (SGLANG_Q35MF_DEP_DX_MEMO_FULL) the owner also copies the winner's row of its
// whole-vocab normalized-embedding memo into fc_out.

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

namespace e2x_dep {

inline constexpr uint32_t kMarker = 0xffffffffu;
inline constexpr uint32_t kSpinLimit = 1u << 26;
inline constexpr int kMaxBlocks = 32;
inline constexpr int kMaxWorld = 16;

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

SGL_DEVICE bool has_marker(const uint4& v) {
  return v.x == kMarker || v.y == kMarker || v.z == kMarker || v.w == kMarker;
}

}  // namespace e2x_dep

struct E2xDepXchgParams {
  const bf16_t* logits;          // [world * rows, ld], this rank's vocab shard of every gathered row
  uint32_t* part_key;            // [g_max, kMaxBlocks]
  int32_t* part_idx;             // [g_max, kMaxBlocks]
  uint32_t* counter;             // [g_max], zero between launches
  uint4* slots;                  // this rank's [2, g_max, world] candidates
  const uint64_t* peer_slots;    // [world] every rank's slots (symmetric memory)
  uint32_t* epoch;               // [g_max]
  int64_t* token_out;            // [rows]
  float* topk_p;                 // [rows] or null
  int64_t* draft_tokens;         // [rows, steps] or null
  int64_t* positions;            // [rows] or null
  int64_t* mrope;                // [3, >= rows] or null
  int64_t ld;
  int32_t vocab;
  int32_t vocab_start;
  uint32_t nan_key;
  int32_t rank;
  int32_t world;
  int32_t rows;
  int32_t g_max;
  int32_t draft_column;
  int32_t draft_stride;
  int32_t mrope_stride;
  const bf16_t* memo;            // [world * vocab, hidden] normalized embedding rows, or null
  bf16_t* fc_out;                // [rows, fc_stride], the winner's row goes to columns [0, hidden)
  int64_t fc_stride;
  int32_t hidden;
};

/**
 * \brief Grid (blocks, world * rows). The last CTA of gathered row g (atomic counter) reduces the
 * partial argmaxes and sends the candidate to rank g / rows; on that rank it then waits for every
 * rank's candidate and writes the token and the topk1 finish outputs of local row g % rows.
 */
template <int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) e2x_dep_xchg_kernel(const E2xDepXchgParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  __shared__ uint32_t s_v[kThreads / 32];
  __shared__ int32_t s_i[kThreads / 32];
  __shared__ bool s_last;
  __shared__ uint32_t s_key[e2x_dep::kMaxWorld];
  __shared__ int32_t s_tok[e2x_dep::kMaxWorld];
  __shared__ int32_t s_win_tok;

  PDLWaitPrimary<kUsePDL>();

  const int g = static_cast<int>(blockIdx.y);
  const int nblocks = static_cast<int>(gridDim.x);
  const bf16_t* logits = p.logits + static_cast<int64_t>(g) * p.ld;
  uint32_t best = 0u;
  int32_t bidx = 0x7fffffff;
  const int nvec = p.vocab / 8;
  for (int v = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x); v < nvec;
       v += nblocks * kThreads) {
    vec_t x;
    x.load(logits, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t key = e2x_dep::value_key(cast<fp32_t>(x[j]), p.nan_key);
      const int32_t idx = v * 8 + j;
      if (e2x_dep::better(key, idx, best, bidx)) {
        best = key;
        bidx = idx;
      }
    }
  }
  e2x_dep::block_argmax<kThreads>(best, bidx, s_v, s_i);
  uint32_t* counter = p.counter + g;
  if (threadIdx.x == 0) {
    p.part_key[g * e2x_dep::kMaxBlocks + blockIdx.x] = best;
    p.part_idx[g * e2x_dep::kMaxBlocks + blockIdx.x] = bidx;
    __threadfence();
    s_last = atomicAdd(counter, 1u) == static_cast<uint32_t>(nblocks - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  uint32_t v = 0u;
  int32_t i = 0x7fffffff;
  if (static_cast<int>(threadIdx.x) < nblocks) {
    v = __ldcg(p.part_key + g * e2x_dep::kMaxBlocks + threadIdx.x);
    i = __ldcg(p.part_idx + g * e2x_dep::kMaxBlocks + threadIdx.x);
  }
  __syncthreads();
  e2x_dep::block_argmax<kThreads>(v, i, s_v, s_i);

  const int owner = g / p.rows;
  const uint32_t half = p.epoch[g] & 1u;
  const int64_t row_slot = (static_cast<int64_t>(half) * p.g_max + g) * p.world;
  if (threadIdx.x == 0) {
    uint4* dst = reinterpret_cast<uint4*>(p.peer_slots[owner]) + row_slot + p.rank;
    // 16-bit key halves keep every word away from the marker.
    const uint4 meta = make_uint4(v & 0xffffu, v >> 16, static_cast<uint32_t>(p.vocab_start + i), 0u);
    ptx::st_relaxed_16B(meta, dst, 0);
    *counter = 0u;
  }
  if (owner != p.rank) {
    if (threadIdx.x == 0) p.epoch[g] = half ^ 1u;
    return;
  }

  uint4* incoming = p.slots + row_slot;
  if (static_cast<int>(threadIdx.x) < p.world) {
    uint4 r;
    uint32_t spins = 0;
    do {
      ptx::ld_relaxed_16B(r, incoming, threadIdx.x);
      if (++spins == e2x_dep::kSpinLimit) __trap();
    } while (e2x_dep::has_marker(r));
    s_key[threadIdx.x] = r.x | (r.y << 16);
    s_tok[threadIdx.x] = static_cast<int32_t>(r.z);
    // Re-arm the word this thread read; the senders' next store to it is two launches away.
    const uint4 marker = make_uint4(e2x_dep::kMarker, e2x_dep::kMarker, e2x_dep::kMarker, e2x_dep::kMarker);
    ptx::st_global_16B(marker, incoming, threadIdx.x);
  }
  __syncthreads();
  const int64_t j = g - p.rank * p.rows;
  if (threadIdx.x == 0) {
    // Rank order: rank 0's candidate first, as k75p_select walks the gathered pairs.
    uint32_t bv = s_key[0];
    int32_t bt = s_tok[0];
    for (int s = 1; s < p.world; ++s) {
      if (e2x_dep::better(s_key[s], s_tok[s], bv, bt)) {
        bv = s_key[s];
        bt = s_tok[s];
      }
    }
    s_win_tok = bt;
    p.token_out[j] = bt;
    if (p.topk_p != nullptr) p.topk_p[j] = 1.0f;
    if (p.draft_tokens != nullptr) p.draft_tokens[j * p.draft_stride + p.draft_column] = bt;
    if (p.positions != nullptr) p.positions[j] = p.positions[j] + 1;
    if (p.mrope != nullptr) {
#pragma unroll
      for (int d = 0; d < 3; ++d)
        p.mrope[d * p.mrope_stride + j] += 1;
    }
    p.epoch[g] = half ^ 1u;
  }
  if (p.fc_out != nullptr) {
    __syncthreads();
    const uint4* src = reinterpret_cast<const uint4*>(p.memo + static_cast<int64_t>(s_win_tok) * p.hidden);
    uint4* dst = reinterpret_cast<uint4*>(p.fc_out + j * p.fc_stride);
    for (int c = static_cast<int>(threadIdx.x); c < p.hidden / 8; c += kThreads)
      dst[c] = src[c];
  }
}

template <bool kUsePDL>
struct E2xDepXchg {
  static constexpr int kThreads = 256;
  static_assert(e2x_dep::kMaxBlocks <= kThreads, "partials are reduced by one CTA");

  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView part_key,
      const tvm::ffi::TensorView part_idx,
      const tvm::ffi::TensorView counter,
      const tvm::ffi::TensorView slots,
      const tvm::ffi::TensorView peer_slots,
      const tvm::ffi::TensorView epoch,
      const tvm::ffi::TensorView token_out,
      const tvm::ffi::TensorView topk_p,
      const tvm::ffi::TensorView draft_tokens,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView mrope_positions,
      const tvm::ffi::TensorView memo,
      const tvm::ffi::TensorView fc_out,
      int64_t vocab_start,
      int64_t nan_key,
      int64_t rank,
      int64_t rows,
      int64_t blocks,
      int64_t draft_column) {
    using namespace host;
    auto G = SymbolicSize{"gathered_rows"};
    auto V = SymbolicSize{"vocab_local"};
    auto LD = SymbolicSize{"logits_row_stride"};
    auto GM = SymbolicSize{"g_max"};
    auto WS = SymbolicSize{"world"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({G, V}).with_strides({LD, 1}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({GM, e2x_dep::kMaxBlocks}).with_dtype<int32_t>().with_device(device).verify(part_key);
    TensorMatcher({GM, e2x_dep::kMaxBlocks}).with_dtype<int32_t>().with_device(device).verify(part_idx);
    TensorMatcher({GM}).with_dtype<int32_t>().with_device(device).verify(counter);
    TensorMatcher({2, GM, WS, 4}).with_dtype<int32_t>().with_device(device).ensure_alignment(16).verify(slots);
    TensorMatcher({WS}).with_dtype<int64_t>().with_device(device).verify(peer_slots);
    TensorMatcher({GM}).with_dtype<int32_t>().with_device(device).verify(epoch);
    const int64_t world = WS.unwrap();
    const int64_t g = G.unwrap();
    CHECK_HOST(world >= 1 && world <= e2x_dep::kMaxWorld) << "world " << world;
    CHECK_HOST(rows >= 1 && g == world * rows) << "gathered rows " << g << " != world " << world << " x rows " << rows;
    CHECK_HOST(g <= GM.unwrap()) << "more gathered rows than exchange slots";
    CHECK_HOST(rank >= 0 && rank < world) << "rank " << rank;
    CHECK_HOST(blocks >= 1 && blocks <= e2x_dep::kMaxBlocks) << "blocks " << blocks;
    CHECK_HOST(V.unwrap() % 8 == 0 && LD.unwrap() % 8 == 0) << "logits rows must be 16-byte vectors";
    auto R = SymbolicSize{"rows"};
    R.set_value(rows);
    TensorMatcher({R, 1}).with_dtype<int64_t>().with_device(device).verify(token_out);
    float* tp = nullptr;
    if (topk_p.numel() > 0) {
      TensorMatcher({R, 1}).with_dtype<fp32_t>().with_device(device).verify(topk_p);
      tp = static_cast<float*>(topk_p.data_ptr());
    }
    int64_t* pos = nullptr;
    if (positions.numel() > 0) {
      TensorMatcher({R}).with_dtype<int64_t>().with_device(device).verify(positions);
      pos = static_cast<int64_t*>(positions.data_ptr());
    }
    int64_t* mrope = nullptr;
    int32_t mrope_stride = 0;
    if (mrope_positions.numel() > 0) {
      auto S = SymbolicSize{"mrope_row_stride"};
      TensorMatcher({3, R}).with_strides({S, 1}).with_dtype<int64_t>().with_device(device).verify(mrope_positions);
      mrope = static_cast<int64_t*>(mrope_positions.data_ptr());
      mrope_stride = static_cast<int32_t>(S.unwrap());
    }
    int64_t* tokens = nullptr;
    int32_t stride = 0;
    if (draft_tokens.numel() > 0) {
      auto S = SymbolicSize{"num_steps"};
      TensorMatcher({R, S}).with_dtype<int64_t>().with_device(device).verify(draft_tokens);
      CHECK_HOST(draft_column >= 0 && draft_column < S.unwrap()) << "draft column out of range";
      tokens = static_cast<int64_t*>(draft_tokens.data_ptr());
      stride = static_cast<int32_t>(S.unwrap());
    }
    const bf16_t* memo_ptr = nullptr;
    bf16_t* fc_ptr = nullptr;
    int64_t fc_stride = 0;
    int64_t hidden = 8;
    if (memo.numel() > 0) {
      auto HD = SymbolicSize{"hidden"};
      auto FS = SymbolicSize{"fc_row_stride"};
      auto VF = SymbolicSize{"vocab"};
      VF.set_value(V.unwrap() * world);
      TensorMatcher({VF, HD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(memo);
      TensorMatcher({R, HD}).with_strides({FS, 1}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(fc_out);
      hidden = HD.unwrap();
      fc_stride = FS.unwrap();
      CHECK_HOST(hidden % 8 == 0 && fc_stride % 8 == 0) << "memo rows must be 16-byte vectors";
      memo_ptr = static_cast<const bf16_t*>(memo.data_ptr());
      fc_ptr = static_cast<bf16_t*>(fc_out.data_ptr());
    }
    const auto params = E2xDepXchgParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .part_key = static_cast<uint32_t*>(part_key.data_ptr()),
        .part_idx = static_cast<int32_t*>(part_idx.data_ptr()),
        .counter = static_cast<uint32_t*>(counter.data_ptr()),
        .slots = static_cast<uint4*>(slots.data_ptr()),
        .peer_slots = static_cast<const uint64_t*>(peer_slots.data_ptr()),
        .epoch = static_cast<uint32_t*>(epoch.data_ptr()),
        .token_out = static_cast<int64_t*>(token_out.data_ptr()),
        .topk_p = tp,
        .draft_tokens = tokens,
        .positions = pos,
        .mrope = mrope,
        .ld = LD.unwrap(),
        .vocab = static_cast<int32_t>(V.unwrap()),
        .vocab_start = static_cast<int32_t>(vocab_start),
        .nan_key = static_cast<uint32_t>(nan_key),
        .rank = static_cast<int32_t>(rank),
        .world = static_cast<int32_t>(world),
        .rows = static_cast<int32_t>(rows),
        .g_max = static_cast<int32_t>(GM.unwrap()),
        .draft_column = static_cast<int32_t>(draft_column),
        .draft_stride = stride,
        .mrope_stride = mrope_stride,
        .memo = memo_ptr,
        .fc_out = fc_ptr,
        .fc_stride = fc_stride,
        .hidden = static_cast<int32_t>(hidden),
    };
    LaunchKernel(dim3(static_cast<uint32_t>(blocks), static_cast<uint32_t>(g)), kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(e2x_dep_xchg_kernel<kThreads, kUsePDL>, params);
  }
};

}  // namespace sglang
