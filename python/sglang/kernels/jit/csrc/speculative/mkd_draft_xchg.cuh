// Greedy MTP draft step boundary in one launch: shard argmax of the hot-vocab logits, a
// multicast push of each rank's candidate (its embedding row, or its fc memo row) plus the
// rank's hidden-half fc partial, and the selection, replacing pack + all-gather + select and
// the fc partial all-gather. Selection rules are those of vocab_argmax_pack / _select
// (order key, -0.0 == +0.0, NaN key from the caller, lowest vocabulary id on ties).
//
// Transport: a symmetric buffer with a multicast mapping (multi-node NVLink). Every 32-bit
// payload word travels in a 64-bit word whose high half is the call's epoch, stored with
// multimem.st.relaxed.sys.b64 and polled with ld.relaxed.sys.u64, so arrival is read from
// the data itself (single-copy-atomic 8-byte accesses). Three slots rotate by epoch: a slot
// is rewritten only three calls later, after two more exchanges that each need every rank's
// packet. Waits are bounded and trap.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace mkd_xchg {

inline constexpr int kSlots = 3;
inline constexpr uint32_t kSpinLimit = 1u << 27;

SGL_DEVICE uint32_t order_key(float f) {
  uint32_t b = __float_as_uint(f);
  b = b == 0x80000000u ? 0u : b;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

SGL_DEVICE bool better(uint32_t v, int64_t i, uint32_t bv, int64_t bi) {
  return v > bv || (v == bv && i < bi);
}

SGL_DEVICE uint32_t clear_negative_zero(uint32_t two_bf16) {
  uint32_t lo = two_bf16 & 0xffffu;
  uint32_t hi = two_bf16 >> 16;
  lo = lo == 0x8000u ? 0u : lo;
  hi = hi == 0x8000u ? 0u : hi;
  return lo | (hi << 16);
}

SGL_DEVICE void mc_store(uint64_t* addr, uint32_t payload, uint32_t tag) {
  const uint64_t w = static_cast<uint64_t>(payload) | (static_cast<uint64_t>(tag) << 32);
  asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(addr), "l"(w) : "memory");
}

SGL_DEVICE uint32_t poll(const uint64_t* addr, uint32_t tag) {
  uint64_t w;
  uint32_t n = 0;
  while (true) {
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(w) : "l"(addr) : "memory");
    if (static_cast<uint32_t>(w >> 32) == tag) break;
    if (++n > kSpinLimit) __trap();
  }
  return static_cast<uint32_t>(w);
}

}  // namespace mkd_xchg

struct MkdXchgParams {
  const bf16_t* logits;          // [V_local] hot-vocab shard logits, written by the lm_head
  const void* __restrict__ cand; // [V_local, H] embedding rows (bf16) or memo rows (fp32)
  const int64_t* token_map;      // [V_local] vocabulary id of each local row
  const float* own_partial;      // [H] this rank's hidden-slice fc partial (hidden ranks only)
  uint64_t* mc;                  // multicast base of the exchange buffer
  const uint64_t* uc;            // this rank's own copy of the exchange buffer
  uint32_t* epoch;               // [1] call counter shared by all ranks, starts at 1
  uint32_t* done;                // [1] CTA arrival counter, zero between calls
  void* out_cand;                // [H] winner's embedding row (bf16) or memo row (fp32)
  float* out_partials;           // [world - embed, H] hidden-slice partials in rank order
  int64_t* topk_index;           // [1]
  float* topk_p;                 // [1]
  int64_t* draft_tokens;         // [1, S] or null
  int64_t* positions;            // [1] or null
  int64_t* mrope;                // [3, >= 1] or null
  int32_t vocab;                 // V_local, multiple of 8
  uint32_t nan_key;
  int32_t rank;
  int32_t draft_column;
  int32_t mrope_stride;
};

/**
 * \brief Step-boundary exchange for one token. Grid = kChunks CTAs; CTA c owns columns
 * [c * H / kChunks, ...) of the candidate and partial rows. Every CTA redoes the shard
 * argmax (a few tens of KB from L2) so no CTA waits for another before sending.
 */
template <int kH, int kWorld, bool kMemo, int kChunks, int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads) mkd_xchg_kernel(const MkdXchgParams __grid_constant__ p) {
  using namespace device;
  using vec_t = AlignedVector<bf16_t, 8>;
  constexpr int kW = 2 * kH / kWorld;
  constexpr int kEmbed = kH / kW;
  constexpr int kCols = kH / kChunks;
  constexpr int kCandWords = kMemo ? kCols : kCols / 2;
  constexpr int kPartWords = kCols;
  constexpr int kRec = 2 + kCandWords + kPartWords;
  constexpr int kWarps = kThreads / 32;
  static_assert(kH % (kChunks * 2) == 0);

  __shared__ uint32_t s_v[kWarps];
  __shared__ int64_t s_i[kWarps];
  __shared__ uint32_t s_key[kWorld];
  __shared__ int64_t s_gid[kWorld];
  __shared__ int s_win;
  __shared__ uint32_t s_tag;

  const int c = static_cast<int>(blockIdx.x);
  const int warp = static_cast<int>(threadIdx.x / 32);
  const int lane = static_cast<int>(get_lane_id());

  PDLWaitPrimary<kUsePDL>();
  if (threadIdx.x == 0) {
    uint32_t e;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(e) : "l"(p.epoch) : "memory");
    s_tag = e;
  }

  // Shard argmax, ties to the lowest local row (token_map is increasing).
  uint32_t best = 0u;
  int64_t bidx = 0x7fffffffffffffffll;
  for (int v = static_cast<int>(threadIdx.x); v < p.vocab / 8; v += kThreads) {
    vec_t x;
    x.load(p.logits, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float f = cast<fp32_t>(x[j]);
      const uint32_t key = f == f ? mkd_xchg::order_key(f) : p.nan_key;
      const int64_t idx = v * 8 + j;
      if (mkd_xchg::better(key, idx, best, bidx)) {
        best = key;
        bidx = idx;
      }
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint32_t ov = __shfl_xor_sync(0xffffffffu, best, off);
    const int64_t oi = __shfl_xor_sync(0xffffffffu, bidx, off);
    if (mkd_xchg::better(ov, oi, best, bidx)) {
      best = ov;
      bidx = oi;
    }
  }
  if (lane == 0) {
    s_v[warp] = best;
    s_i[warp] = bidx;
  }
  __syncthreads();
  best = s_v[0];
  bidx = s_i[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    if (mkd_xchg::better(s_v[w], s_i[w], best, bidx)) {
      best = s_v[w];
      bidx = s_i[w];
    }
  }
  const uint32_t tag = s_tag;
  const int slot = static_cast<int>(tag % mkd_xchg::kSlots);
  const int64_t gid = p.token_map[bidx];

  // Send this CTA's chunk of the packet to every rank (self included).
  uint64_t* dst = p.mc + ((static_cast<int64_t>(slot) * kWorld + p.rank) * kChunks + c) * kRec;
  if (threadIdx.x == 0) {
    mkd_xchg::mc_store(dst, best, tag);
    mkd_xchg::mc_store(dst + 1, static_cast<uint32_t>(gid), tag);
  }
  if constexpr (kMemo) {
    const float* row = static_cast<const float*>(p.cand) + bidx * kH + c * kCols;
    for (int w = static_cast<int>(threadIdx.x); w < kCandWords; w += kThreads) {
      mkd_xchg::mc_store(dst + 2 + w, __float_as_uint(row[w]), tag);
    }
  } else {
    const uint32_t* row = reinterpret_cast<const uint32_t*>(static_cast<const bf16_t*>(p.cand) + bidx * kH + c * kCols);
    for (int w = static_cast<int>(threadIdx.x); w < kCandWords; w += kThreads) {
      mkd_xchg::mc_store(dst + 2 + w, mkd_xchg::clear_negative_zero(row[w]), tag);
    }
  }
  if (p.rank >= kEmbed) {
    const float* part = p.own_partial + c * kCols;
    for (int w = static_cast<int>(threadIdx.x); w < kPartWords; w += kThreads) {
      mkd_xchg::mc_store(dst + 2 + kCandWords + w, __float_as_uint(part[w]), tag);
    }
  }

  // Every rank's key, then the winner.
  auto rec = [&](int s) { return p.uc + ((static_cast<int64_t>(slot) * kWorld + s) * kChunks + c) * kRec; };
  if (threadIdx.x < kWorld) {
    const uint64_t* r = rec(static_cast<int>(threadIdx.x));
    s_key[threadIdx.x] = mkd_xchg::poll(r, tag);
    s_gid[threadIdx.x] = static_cast<int64_t>(mkd_xchg::poll(r + 1, tag));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int win = 0;
    uint32_t bv = s_key[0];
    int64_t bi = s_gid[0];
#pragma unroll
    for (int s = 1; s < kWorld; ++s) {
      if (mkd_xchg::better(s_key[s], s_gid[s], bv, bi)) {
        bv = s_key[s];
        bi = s_gid[s];
        win = s;
      }
    }
    s_win = win;
    if (c == 0) {
      p.topk_index[0] = bi;
      p.topk_p[0] = 1.0f;
      if (p.draft_tokens != nullptr) p.draft_tokens[p.draft_column] = bi;
      if (p.positions != nullptr) p.positions[0] = p.positions[0] + 1;
      if (p.mrope != nullptr) {
#pragma unroll
        for (int d = 0; d < 3; ++d) p.mrope[d * p.mrope_stride] += 1;
      }
    }
  }
  __syncthreads();
  const uint64_t* wrec = rec(s_win);
  for (int w = static_cast<int>(threadIdx.x); w < kCandWords; w += kThreads) {
    const uint32_t v = mkd_xchg::poll(wrec + 2 + w, tag);
    static_cast<uint32_t*>(p.out_cand)[(kMemo ? c * kCols : c * kCols / 2) + w] = v;
  }
#pragma unroll
  for (int s = kEmbed; s < kWorld; ++s) {
    const uint64_t* r = rec(s);
    for (int w = static_cast<int>(threadIdx.x); w < kPartWords; w += kThreads) {
      p.out_partials[static_cast<int64_t>(s - kEmbed) * kH + c * kCols + w] = __uint_as_float(mkd_xchg::poll(r + 2 + kCandWords + w, tag));
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.done, 1u) == kChunks - 1) {
      *p.done = 0u;
      *p.epoch = tag + 1;
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

/** \brief Zero a range of 64-bit words (the exchange buffer before first use). */
__global__ void mkd_xchg_zero_kernel(uint64_t* p, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    p[i] = 0ull;
  }
}

template <int kH, int kWorld, bool kMemo, int kChunks, bool kUsePDL>
struct MkdXchg {
  static constexpr int kThreads = 512;
  static constexpr int kCols = kH / kChunks;
  static constexpr int kRec = 2 + (kMemo ? kCols : kCols / 2) + kCols;
  static constexpr int64_t kWords = static_cast<int64_t>(mkd_xchg::kSlots) * kWorld * kChunks * kRec;

  static void zero(int64_t uc_ptr) {
    auto* p = reinterpret_cast<uint64_t*>(uc_ptr);
    mkd_xchg_zero_kernel<<<64, 256>>>(p, kWords);
    host::RuntimeDeviceCheck(cudaGetLastError());
  }

  static void run(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView cand,
      const tvm::ffi::TensorView token_map,
      const tvm::ffi::TensorView own_partial,
      const tvm::ffi::TensorView epoch_done,
      const tvm::ffi::TensorView out_cand,
      const tvm::ffi::TensorView out_partials,
      const tvm::ffi::TensorView topk_index,
      const tvm::ffi::TensorView topk_p,
      const tvm::ffi::TensorView draft_tokens,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView mrope_positions,
      int64_t mc_ptr,
      int64_t uc_ptr,
      int64_t rank,
      int64_t draft_column,
      int64_t nan_key) {
    using namespace host;
    constexpr int kW = 2 * kH / kWorld;
    constexpr int kEmbed = kH / kW;
    auto V = SymbolicSize{"vocab_local"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, V}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    if constexpr (kMemo) {
      TensorMatcher({V, kH}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(cand);
      TensorMatcher({1, kH}).with_dtype<fp32_t>().with_device(device).verify(out_cand);
    } else {
      TensorMatcher({V, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(cand);
      TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).verify(out_cand);
    }
    TensorMatcher({V}).with_dtype<int64_t>().with_device(device).verify(token_map);
    TensorMatcher({1, kH}).with_dtype<fp32_t>().with_device(device).verify(own_partial);
    TensorMatcher({2}).with_dtype<int32_t>().with_device(device).verify(epoch_done);
    TensorMatcher({kWorld - kEmbed, kH}).with_dtype<fp32_t>().with_device(device).verify(out_partials);
    TensorMatcher({1, 1}).with_dtype<int64_t>().with_device(device).verify(topk_index);
    TensorMatcher({1, 1}).with_dtype<fp32_t>().with_device(device).verify(topk_p);
    CHECK_HOST(V.unwrap() % 8 == 0) << "local vocab must be a multiple of 8";
    int64_t* tokens = nullptr;
    if (draft_tokens.numel() > 0) {
      auto S = SymbolicSize{"num_steps"};
      TensorMatcher({1, S}).with_dtype<int64_t>().with_device(device).verify(draft_tokens);
      CHECK_HOST(draft_column >= 0 && draft_column < S.unwrap()) << "draft column out of range";
      tokens = static_cast<int64_t*>(draft_tokens.data_ptr());
    }
    int64_t* pos = nullptr;
    if (positions.numel() > 0) {
      TensorMatcher({1}).with_dtype<int64_t>().with_device(device).verify(positions);
      pos = static_cast<int64_t*>(positions.data_ptr());
    }
    int64_t* mrope = nullptr;
    int32_t mrope_stride = 0;
    if (mrope_positions.numel() > 0) {
      auto R = SymbolicSize{"mrope_row_stride"};
      TensorMatcher({3, 1}).with_strides({R, 1}).with_dtype<int64_t>().with_device(device).verify(mrope_positions);
      mrope = static_cast<int64_t*>(mrope_positions.data_ptr());
      mrope_stride = static_cast<int32_t>(R.unwrap());
    }
    auto* ed = static_cast<uint32_t*>(epoch_done.data_ptr());
    const auto params = MkdXchgParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .cand = cand.data_ptr(),
        .token_map = static_cast<const int64_t*>(token_map.data_ptr()),
        .own_partial = static_cast<const float*>(own_partial.data_ptr()),
        .mc = reinterpret_cast<uint64_t*>(mc_ptr),
        .uc = reinterpret_cast<const uint64_t*>(uc_ptr),
        .epoch = ed,
        .done = ed + 1,
        .out_cand = out_cand.data_ptr(),
        .out_partials = static_cast<float*>(out_partials.data_ptr()),
        .topk_index = static_cast<int64_t*>(topk_index.data_ptr()),
        .topk_p = static_cast<float*>(topk_p.data_ptr()),
        .draft_tokens = tokens,
        .positions = pos,
        .mrope = mrope,
        .vocab = static_cast<int32_t>(V.unwrap()),
        .nan_key = static_cast<uint32_t>(nan_key),
        .rank = static_cast<int32_t>(rank),
        .draft_column = static_cast<int32_t>(draft_column),
        .mrope_stride = mrope_stride,
    };
    LaunchKernel(kChunks, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(mkd_xchg_kernel<kH, kWorld, kMemo, kChunks, kThreads, kUsePDL>, params);
  }
};

}  // namespace sglang
