// MoE router front for 17..64 decode tokens on the trtllm-gen runner, in one launch: the router GEMM
// (512 experts x 4096 hidden, bf16) in the K order of the cuBLAS router GEMM at that token count (one
// chain over K, or two K halves summed), the NVFP4 input quantization (fp4_quantize, linear scales) and
// router_topk_tables' top-k, softmax and trtllm-gen routing tables.
//
// CTA b < 32 G (G = ceil(M / 16) token groups) computes experts 16 (b % 32) .. + 15 for tokens 16 (b / 32) .. + 15
// with m16n8k16 MMA chains; weight and token rows stream into shared memory in matching K chunks (weights from
// before the PDL wait). CTA (x, g) with x < 16 also quantizes token 16 g + x and computes that token's top-k
// from the logits of all 32 CTAs of group g. CTA 32 G builds the tables from every token's top-k; the rows
// inside one expert's range are in ascending token order. Logits and expert ids cross CTAs as words of the
// caller's workspace carrying a valid bit: the reader polls them and zeroes them again for the next launch.
//
// Built with -use_fast_math like qwen_router_tables.cuh, whose top-k key, softmax and padding code it calls;
// every other float operation is an explicit instruction.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cooperative_groups/reduce.h>
#include <cub/block/block_scan.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "qwen_router_front.cuh"
#include "qwen_router_tables.cuh"

#include <algorithm>
#include <cooperative_groups.h>
#include <cstdint>

namespace sglang {

namespace qwen_router_wide_front {

namespace cg = cooperative_groups;
namespace fr = qwen_router_front;
namespace rt = qwen_router_tables;

inline constexpr uint32_t kHidden = 4096;
inline constexpr uint32_t kExperts = 512;
inline constexpr uint32_t kTopK = 10;
inline constexpr uint32_t kMinTokens = 17;
inline constexpr uint32_t kMaxTokens = 64;
// One thread per expert in the tables.
inline constexpr uint32_t kThreads = kExperts;
inline constexpr uint32_t kGroup = 16;       // tokens per CTA: the m16 tile
inline constexpr uint32_t kCtaExperts = 16;  // experts per CTA: two n8 tiles sharing the A fragment
inline constexpr uint32_t kExpertCtas = kExperts / kCtaExperts;
inline constexpr uint32_t kMaxGroups = kMaxTokens / kGroup;
inline constexpr uint32_t kChunks = 8;  // K chunks per chain
inline constexpr uint32_t kRing = 5;    // token-row chunks resident at once
// One row's K chunk: 512 K of the chain, or 256 K of each half.
inline constexpr uint32_t kChunkBytes = 1024;
// Rows padded by 16 B so the 8 rows of an ldmatrix hit distinct banks.
inline constexpr uint32_t kXRowBytes = kChunkBytes + 16;
inline constexpr uint32_t kWRowBytes = kHidden * 2 + 16;
inline constexpr uint32_t kSlotBytes = kGroup * kXRowBytes;
// Warp roles of a GEMV CTA: 0 .. kSplit - 1 MMA chains, 2 .. 5 weight chunks, 6 .. 9 token chunks, 10 .. 15 NVFP4.
inline constexpr uint32_t kWLoadWarp = 2;
inline constexpr uint32_t kXLoadWarp = 6;
inline constexpr uint32_t kQuantWarp = 10;
inline constexpr uint32_t kLoadThreads = 128;
inline constexpr uint32_t kQuantThreads = kThreads - 32 * kQuantWarp;
inline constexpr uint32_t kBlocksPerRow = kHidden / 16;
// Workspace (uint32 words): bf16 logits [kMaxTokens][kExperts], then expert ids [kMaxTokens][kTopK], each word
// holding its 16-bit value and kValid.
inline constexpr uint32_t kIdsWord = kMaxTokens * kExperts;
inline constexpr uint32_t kWorkspaceWords = kIdsWord + kMaxTokens * kTopK;
inline constexpr uint32_t kValid = 1u << 16;
inline constexpr uint32_t kTraceCtas = kExpertCtas * kMaxGroups;
static_assert(kMaxTokens <= 64, "one bit per token in the expert masks");
static_assert(kGroup * kChunkBytes / 16 == 8 * kLoadThreads, "eight 16-byte copies per loader thread and chunk");

struct Smem {
  static constexpr uint32_t kW = 0;
  static constexpr uint32_t kX = kW + kCtaExperts * kWRowBytes;
  static constexpr uint32_t kPart = kX + kRing * kSlotBytes;  // fp32 [kGroup][kCtaExperts], K half 1
  static constexpr uint32_t kBarW = kPart + kGroup * kCtaExperts * 4;  // [kChunks]
  static constexpr uint32_t kBarFull = kBarW + 8 * kChunks;            // [kRing]
  static constexpr uint32_t kBarEmpty = kBarFull + 8 * kRing;          // [kRing]
  static constexpr uint32_t kBytes = kBarEmpty + 8 * kRing;
};

struct Params {
  const bf16_t* x;            // [M, kHidden]
  const bf16_t* weight;       // [kExperts, kHidden]
  const float* global_scale;  // [1], quantization-domain global scale
  bf16_t* logits;             // [M, kExperts]
  uint8_t* x_fp4;             // [M, kHidden / 2]
  uint8_t* x_sf;              // [M, kHidden / 16]
  bf16_t* expert_weights;     // [M, kTopK]
  int32_t* topk_ids;          // [M, kTopK]
  int32_t* expanded_idx_to_permuted_idx;  // [M * kTopK]
  int32_t* permuted_idx_to_token_idx;     // [max padded + 1]
  int32_t* cta_idx_xy_to_batch_idx;       // [max CTAs]
  int32_t* cta_idx_xy_to_mn_limit;        // [max CTAs]
  int32_t* num_non_exiting_ctas;          // [1]
  int32_t* total_num_padded_tokens;       // [1]
  int32_t* num_tokens_per_expert;         // [kExperts]
  uint32_t* workspace;                    // [kWorkspaceWords], zero between launches
  uint64_t* trace;                        // optional %globaltimer stamps (instrumented build only)
  uint32_t num_tokens;
  uint32_t tile_log2;
};

SGL_DEVICE float mul_rn(float a, float b) {
  float y;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

SGL_DEVICE float add_rn(float a, float b) {
  float y;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

SGL_DEVICE float max_f32(float a, float b) {
  float y;
  asm("max.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

SGL_DEVICE void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

// Arrives on the mbarrier once every earlier cp.async of this thread has landed.
SGL_DEVICE void cp_async_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

// Workspace words written by other CTAs of this grid: never served from L1.
SGL_DEVICE uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

SGL_DEVICE uint32_t ld_volatile(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

template <bool kTrace>
SGL_DEVICE void stamp(const Params& p, uint32_t slot) {
  if constexpr (kTrace) p.trace[blockIdx.x * 8 + slot] = fr::global_timer();
}

template <bool kTrace>
SGL_DEVICE void stamp_tables(const Params& p, uint32_t slot) {
  if constexpr (kTrace) p.trace[kTraceCtas * 8 + slot] = fr::global_timer();
}

// qwen_router_front::quantize_block with its float operations pinned (this file builds with fast math).
SGL_DEVICE void quantize_block(const uint4& a, const uint4& b, float gs, uint32_t& sf, uint2& packed) {
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  const uint32_t m01 = fr::bf16x2_abs_max(w[0], w[1]);
  const uint32_t m23 = fr::bf16x2_abs_max(w[2], w[3]);
  const uint32_t m45 = fr::bf16x2_abs_max(w[4], w[5]);
  const uint32_t m67 = fr::bf16x2_abs_max(w[6], w[7]);
  const uint32_t m = fr::bf16x2_max(fr::bf16x2_max(m01, m23), fr::bf16x2_max(m45, m67));
  const float amax = max_f32(fr::bf16_lo(m), fr::bf16_hi(m));
  const float scale = mul_rn(gs, mul_rn(amax, fr::rcp_approx_ftz(6.0f)));
  sf = fr::cvt_e4m3(scale);
  const float sf_f = fr::e4m3_to_f32(sf);
  const float out_scale = (sf_f == 0.0f) ? 0.0f : fr::rcp_approx_ftz(mul_rn(sf_f, fr::rcp_approx_ftz(gs)));
  float lo[8], hi[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lo[2 * i] = mul_rn(fr::bf16_lo(w[i]), out_scale);
    lo[2 * i + 1] = mul_rn(fr::bf16_hi(w[i]), out_scale);
    hi[2 * i] = mul_rn(fr::bf16_lo(w[4 + i]), out_scale);
    hi[2 * i + 1] = mul_rn(fr::bf16_hi(w[4 + i]), out_scale);
  }
  packed.x = fr::cvt_e2m1x8(lo);
  packed.y = fr::cvt_e2m1x8(hi);
}

// K index of byte `byte` (< kChunkBytes) of a row's chunk c: K [512 c, 512 c + 512) for one chain, or
// K [256 c, 256 c + 256) of half 0 then of half 1 for two halves.
template <uint32_t kSplit>
SGL_DEVICE uint32_t chunk_k(uint32_t c, uint32_t byte) {
  if constexpr (kSplit == 1) {
    return c * (kChunkBytes / 2) + byte / 2;
  } else {
    return (byte < kChunkBytes / 2 ? 0u : kHidden / 2 - kChunkBytes / 4) + c * (kChunkBytes / 4) + byte / 2;
  }
}

// This CTA's 16 weight rows chunk by chunk into the resident weight tile (static: issued before the PDL wait).
template <uint32_t kSplit>
SGL_DEVICE void load_weight_chunks(const Params& p, uint32_t expert0, uint8_t* smem) {
  const uint32_t q = threadIdx.x - 32 * kWLoadWarp;
  const bf16_t* w = p.weight + static_cast<size_t>(expert0) * kHidden;
  const uint32_t w_base = fr::smem_addr(smem + Smem::kW);
  const uint32_t bar = fr::smem_addr(smem + Smem::kBarW);
#pragma unroll 1
  for (uint32_t c = 0; c < kChunks; ++c) {
#pragma unroll
    for (uint32_t i = q; i < kGroup * kChunkBytes / 16; i += kLoadThreads) {
      const uint32_t row = i / (kChunkBytes / 16), k = chunk_k<kSplit>(c, (i % (kChunkBytes / 16)) * 16);
      cp_async_16(w_base + row * kWRowBytes + k * 2, w + row * kHidden + k);
    }
    cp_async_arrive(bar + 8 * c);
  }
}

// Token rows 16 g .. 16 g + rows - 1 chunk by chunk through the ring.
template <uint32_t kSplit>
SGL_DEVICE void load_token_chunks(const Params& p, uint32_t tok0, uint32_t rows, uint8_t* smem) {
  const uint32_t q = threadIdx.x - 32 * kXLoadWarp;
  const uint32_t x_base = fr::smem_addr(smem + Smem::kX);
  const uint32_t bar_full = fr::smem_addr(smem + Smem::kBarFull);
  const uint32_t bar_empty = fr::smem_addr(smem + Smem::kBarEmpty);
#pragma unroll 1
  for (uint32_t c = 0; c < kChunks; ++c) {
    const uint32_t slot = c % kRing;
    if (c >= kRing) fr::mbar_wait(bar_empty + 8 * slot, (c / kRing - 1) & 1);
#pragma unroll
    for (uint32_t i = q; i < kGroup * kChunkBytes / 16; i += kLoadThreads) {
      const uint32_t row = i / (kChunkBytes / 16), byte = (i % (kChunkBytes / 16)) * 16;
      if (row < rows) {
        cp_async_16(
            x_base + slot * kSlotBytes + row * kXRowBytes + byte,
            p.x + static_cast<size_t>(tok0 + row) * kHidden + chunk_k<kSplit>(c, byte));
      }
    }
    cp_async_arrive(bar_full + 8 * slot);
  }
}

// One K chain of this CTA's 16 x 16 tile: kSplit = 1 runs all of K, kSplit = 2 runs K half `half` (the two
// halves are summed in fp32 afterwards). Each step is one m16n8k16 per n8 tile in increasing K, the order
// of the cuBLAS router GEMM; fragments are loaded kDepth steps ahead and each chunk's ring slot is released
// once its last fragment is consumed.
template <uint32_t kSplit, bool kTrace>
SGL_DEVICE void gemv_chain(const Params& p, uint32_t half, uint32_t rows, const uint8_t* smem, float (&d)[2][4]) {
  using L = Smem;
  constexpr uint32_t kSteps = kHidden / kSplit / 16;
  constexpr uint32_t kChunkSteps = kSteps / kChunks;
  constexpr uint32_t kDepth = 4;
  static_assert(kChunkSteps % kDepth == 0, "a chunk's first fragments are loaded after its barriers");
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t j = lane >> 3, r = lane & 7;
  // A: matrix j is token rows (j & 1) * 8 + r at K offset (j >> 1) * 8; rows past M repeat row 0.
  const uint32_t tok = (j & 1) * 8 + r;
  const uint32_t a_base =
      fr::smem_addr(smem + L::kX + (tok < rows ? tok : 0) * kXRowBytes + half * (kChunkBytes / 2) + (j >> 1) * 16);
  // B: matrix j is expert rows (j >> 1) * 8 + r at K offset (j & 1) * 8, so registers 0 / 1 feed n8 tile 0
  // and registers 2 / 3 feed n8 tile 1.
  const uint32_t b_base =
      fr::smem_addr(smem + L::kW + ((j >> 1) * 8 + r) * kWRowBytes + (half * (kHidden / 2) + (j & 1) * 8) * 2);
  const uint32_t bar_w = fr::smem_addr(smem + L::kBarW);
  const uint32_t bar_full = fr::smem_addr(smem + L::kBarFull);
  const uint32_t bar_empty = fr::smem_addr(smem + L::kBarEmpty);
  uint32_t a[kDepth][4], b[kDepth][4];
  fr::mbar_wait(bar_w, 0);
  if (threadIdx.x == 0) stamp<kTrace>(p, 2);
  fr::mbar_wait(bar_full, 0);
  if (threadIdx.x == 0) stamp<kTrace>(p, 3);
#pragma unroll
  for (uint32_t i = 0; i < kDepth; ++i) {
    fr::ldmatrix_x4(a[i], a_base + i * 32);
    fr::ldmatrix_x4(b[i], b_base + i * 32);
  }
#pragma unroll
  for (uint32_t s = 0; s < kSteps; ++s) {
    const uint32_t sl = s % kDepth;
    fr::mma_bf16_16816(d[0], a[sl][0], a[sl][1], a[sl][2], a[sl][3], b[sl][0], b[sl][1]);
    fr::mma_bf16_16816(d[1], a[sl][0], a[sl][1], a[sl][2], a[sl][3], b[sl][2], b[sl][3]);
    if (s % kChunkSteps == kChunkSteps - 1) {
      __syncwarp();
      if (lane == 0) fr::mbar_arrive(bar_empty + 8 * ((s / kChunkSteps) % kRing));
    }
    const uint32_t nx = s + kDepth;
    if (nx < kSteps) {
      const uint32_t c = nx / kChunkSteps;
      if (nx % kChunkSteps == 0) {
        fr::mbar_wait(bar_w + 8 * c, 0);
        fr::mbar_wait(bar_full + 8 * (c % kRing), (c / kRing) & 1);
      }
      fr::ldmatrix_x4(a[sl], a_base + (c % kRing) * kSlotBytes + (nx % kChunkSteps) * 32);
      fr::ldmatrix_x4(b[sl], b_base + nx * 32);
    }
  }
  if (threadIdx.x == 0) stamp<kTrace>(p, 4);
}

// The K-half sum (kSplit = 2) and the bf16 logits of this CTA's tile.
template <uint32_t kSplit>
SGL_DEVICE void store_logits(
    const Params& p, uint32_t half, uint32_t expert0, uint32_t tok0, uint32_t rows, uint8_t* smem, float (&d)[2][4]) {
  const uint32_t lane = threadIdx.x % 32;
  // C fragment: d[t][0 / 1] = (token g, experts 2c, 2c + 1 of n8 tile t), d[t][2 / 3] = token g + 8.
  const uint32_t g = lane >> 2, c = lane & 3;
  if constexpr (kSplit == 2) {
    float* part = reinterpret_cast<float*>(smem + Smem::kPart);
    if (half == 1) {
#pragma unroll
      for (uint32_t t = 0; t < 2; ++t) {
        part[g * kCtaExperts + t * 8 + 2 * c] = d[t][0];
        part[g * kCtaExperts + t * 8 + 2 * c + 1] = d[t][1];
        part[(g + 8) * kCtaExperts + t * 8 + 2 * c] = d[t][2];
        part[(g + 8) * kCtaExperts + t * 8 + 2 * c + 1] = d[t][3];
      }
    }
    asm volatile("bar.sync 1, 64;" ::: "memory");
    if (half == 1) return;
#pragma unroll
    for (uint32_t t = 0; t < 2; ++t) {
      d[t][0] = add_rn(d[t][0], part[g * kCtaExperts + t * 8 + 2 * c]);
      d[t][1] = add_rn(d[t][1], part[g * kCtaExperts + t * 8 + 2 * c + 1]);
      d[t][2] = add_rn(d[t][2], part[(g + 8) * kCtaExperts + t * 8 + 2 * c]);
      d[t][3] = add_rn(d[t][3], part[(g + 8) * kCtaExperts + t * 8 + 2 * c + 1]);
    }
  }
  const size_t at = static_cast<size_t>(tok0 + g) * kExperts + expert0 + 2 * c;
#pragma unroll
  for (uint32_t t = 0; t < 2; ++t) {
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
      if (g + 8 * h >= rows) continue;
      const __nv_bfloat162 v = __floats2bfloat162_rn(d[t][2 * h], d[t][2 * h + 1]);
      const uint32_t bits = *reinterpret_cast<const uint32_t*>(&v);
      const size_t i = at + h * 8 * kExperts + t * 8;
      *reinterpret_cast<__nv_bfloat162*>(p.logits + i) = v;
      *reinterpret_cast<uint2*>(p.workspace + i) = make_uint2((bits & 0xFFFFu) | kValid, (bits >> 16) | kValid);
    }
  }
}

SGL_DEVICE void quantize_row(const Params& p, uint32_t row) {
  for (uint32_t blk = threadIdx.x - 32 * kQuantWarp; blk < kBlocksPerRow; blk += kQuantThreads) {
    const fr::U32x8 v = fr::ld_produced32(p.x + static_cast<size_t>(row) * kHidden + blk * 16);
    uint32_t sf;
    uint2 packed;
    quantize_block(v.lo, v.hi, p.global_scale[0], sf, packed);
    *reinterpret_cast<uint2*>(p.x_fp4 + static_cast<size_t>(row) * (kHidden / 2) + blk * 8) = packed;
    p.x_sf[static_cast<size_t>(row) * kBlocksPerRow + blk] = static_cast<uint8_t>(sf);
  }
}

// router_topk_tables' keys of one token row, lane l holding experts 16 l .. 16 l + 15.
SGL_DEVICE void token_keys(const uint4 (&v)[2], uint32_t lane, uint32_t (&keys)[16]) {
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t w = (&v[i / 4].x)[i % 4];
    keys[2 * i] = rt::topk_key(w & 0xFFFFu, lane * 16 + 2 * i);
    keys[2 * i + 1] = rt::topk_key(w >> 16, lane * 16 + 2 * i + 1);
  }
}

// Top-10 experts and softmax weights of token t (one warp) once its group's logits are published: each lane
// keeps the top 4 of its 16 keys and ten rounds take the warp maximum of the lane heads; when a lane runs
// out before the last round the token is redone with rt::warp_top10. The keys are unique, so both are exact.
template <bool kTrace>
SGL_DEVICE void token_topk(const Params& p, uint32_t t, uint32_t* s_cand) {
  const uint32_t lane = threadIdx.x % 32;
  // Lane l reads experts 16 l .. 16 l + 15 of token t, written by CTA (l, t / 16).
  uint4* box = reinterpret_cast<uint4*>(p.workspace + static_cast<size_t>(t) * kExperts + lane * 16);
  uint4 w[4];
  bool ready;
  do {
    ready = true;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      w[i] = ld_volatile_v4(box + i);
      ready = ready && (w[i].x & w[i].y & w[i].z & w[i].w & kValid);
    }
  } while (!__all_sync(0xFFFFFFFFu, ready));
  if (threadIdx.x == 0) stamp<kTrace>(p, 6);
#pragma unroll
  for (int i = 0; i < 4; ++i) box[i] = make_uint4(0u, 0u, 0u, 0u);
  // Back to bf16 pairs in router_topk_tables' lane layout.
  const uint4 v[2] = {
      make_uint4(
          (w[0].x & 0xFFFFu) | (w[0].y << 16),
          (w[0].z & 0xFFFFu) | (w[0].w << 16),
          (w[1].x & 0xFFFFu) | (w[1].y << 16),
          (w[1].z & 0xFFFFu) | (w[1].w << 16)),
      make_uint4(
          (w[2].x & 0xFFFFu) | (w[2].y << 16),
          (w[2].z & 0xFFFFu) | (w[2].w << 16),
          (w[3].x & 0xFFFFu) | (w[3].y << 16),
          (w[3].z & 0xFFFFu) | (w[3].w << 16))};
  uint32_t k[16];
  token_keys(v, lane, k);
#pragma unroll
  for (int i = 0; i < 4; ++i) fr::sort4_desc(k[4 * i], k[4 * i + 1], k[4 * i + 2], k[4 * i + 3]);
  uint32_t top[4] = {k[0], k[1], k[2], k[3]};
  const uint32_t q1[4] = {k[4], k[5], k[6], k[7]};
  uint32_t q2[4] = {k[8], k[9], k[10], k[11]};
  const uint32_t q3[4] = {k[12], k[13], k[14], k[15]};
  fr::merge_top4_desc(top, q1);
  fr::merge_top4_desc(q2, q3);
  fr::merge_top4_desc(top, q2);
  uint32_t mine = 0, taken = 0;
  bool bad = false;
#pragma unroll
  for (uint32_t rr = 0; rr < kTopK; ++rr) {
    const uint32_t winner = __reduce_max_sync(0xFFFFFFFFu, top[0]);
    const bool owner = top[0] == winner;
    top[0] = owner ? top[1] : top[0];
    top[1] = owner ? top[2] : top[1];
    top[2] = owner ? top[3] : top[2];
    top[3] = owner ? 0u : top[3];
    taken += owner ? 1u : 0u;
    bad = bad || (owner && taken == 4u && rr + 1 < kTopK);
    mine = lane == rr ? winner : mine;
  }
  if (__any_sync(0xFFFFFFFFu, bad)) {
    token_keys(v, lane, k);
    mine = rt::warp_top10(k, lane, s_cand);
  }
  uint32_t bits, expert;
  rt::topk_unkey(mine, bits, expert);
  const bf16_t weight = rt::softmax_topk(
      cg::tiled_partition<32>(cg::this_thread_block()), __ushort_as_bfloat16(static_cast<uint16_t>(bits)), lane);
  if (lane < kTopK) {
    p.expert_weights[t * kTopK + lane] = weight;
    p.topk_ids[t * kTopK + lane] = static_cast<int32_t>(expert);
    p.workspace[kIdsWord + t * kTopK + lane] = expert | kValid;
  }
  if (threadIdx.x == 0) stamp<kTrace>(p, 7);
}

using Scan = cub::BlockScan<int32_t, kThreads, cub::BLOCK_SCAN_WARP_SCANS>;

// router_topk_tables_small's tables for up to 64 tokens from every token's expert ids: thread e owns expert e
// (count, tile-padded offset, CTA -> expert / mn-limit tables, padding rows); a token's row in its expert's
// range is the number of earlier tokens routed to that expert.
template <bool kTrace>
SGL_DEVICE void build_tables(const Params& p, Scan::TempStorage& scan_storage, uint64_t* s_mask, int32_t* s_offset) {
  constexpr uint32_t kPerThread = (kMaxTokens * kTopK + kThreads - 1) / kThreads;
  const uint32_t M = p.num_tokens;
  uint32_t* ids = p.workspace + kIdsWord;
  s_mask[threadIdx.x] = 0ull;
  __syncthreads();
  uint32_t id[kPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kPerThread; ++j) {
    const uint32_t i = threadIdx.x + j * kThreads;
    if (i < M * kTopK) {
      uint32_t v;
      do {
        v = ld_volatile(ids + i);
      } while (!(v & kValid));
      ids[i] = 0u;
      id[j] = v & 0xFFFFu;
      atomicOr(reinterpret_cast<unsigned long long*>(&s_mask[id[j]]), 1ull << (i / kTopK));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) stamp_tables<kTrace>(p, 1);

  const uint32_t e = threadIdx.x;
  const uint64_t mask = s_mask[e];
  const int32_t count = __popcll(mask);
  const int32_t tile = 1 << p.tile_log2;
  const int32_t num_cta = (count + tile - 1) >> p.tile_log2;
  int32_t cta_offset, total_cta;
  Scan(scan_storage).ExclusiveSum(num_cta, cta_offset, total_cta);
  for (int32_t c = 0; c < num_cta; ++c) {
    const int32_t end = (cta_offset + c + 1) << p.tile_log2;
    const int32_t limit = min(end, (cta_offset << p.tile_log2) + count);
    p.cta_idx_xy_to_batch_idx[cta_offset + c] = static_cast<int32_t>(e);
    p.cta_idx_xy_to_mn_limit[cta_offset + c] = limit;
    rt::pad_rows(p.permuted_idx_to_token_idx, limit, end);
  }
  s_offset[e] = cta_offset << p.tile_log2;
  p.num_tokens_per_expert[e] = count;
  if (e == 0) {
    p.num_non_exiting_ctas[0] = total_cta;
    p.total_num_padded_tokens[0] = total_cta << p.tile_log2;
  }
  __syncthreads();

#pragma unroll
  for (uint32_t j = 0; j < kPerThread; ++j) {
    const uint32_t i = threadIdx.x + j * kThreads;
    if (i < M * kTopK) {
      const uint32_t t = i / kTopK;
      const int32_t permuted = s_offset[id[j]] + __popcll(s_mask[id[j]] & ((1ull << t) - 1ull));
      p.expanded_idx_to_permuted_idx[i] = permuted;
      p.permuted_idx_to_token_idx[permuted] = static_cast<int32_t>(t);
    }
  }
  if constexpr (kTrace) {
    __syncthreads();
    if (threadIdx.x == 0) stamp_tables<kTrace>(p, 2);
  }
}

template <bool kUsePDL, uint32_t kSplit, bool kTrace>
__global__ __launch_bounds__(kThreads, 1) void router_wide_front_kernel(const __grid_constant__ Params p) {
  using L = Smem;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Scan::TempStorage scan_storage;
  __shared__ uint64_t s_mask[kExperts];
  __shared__ int32_t s_offset[kExperts];
  __shared__ uint32_t s_cand[32];

  const uint32_t M = p.num_tokens;
  const uint32_t groups = (M + kGroup - 1) / kGroup;
  if (blockIdx.x == groups * kExpertCtas) {
    if (threadIdx.x == 0) stamp_tables<kTrace>(p, 0);
    // The dependent grid reads our outputs only after its own griddepcontrol.wait.
    device::PDLTriggerSecondary<kUsePDL>();
    device::PDLWaitPrimary<kUsePDL>();
    build_tables<kTrace>(p, scan_storage, s_mask, s_offset);
    return;
  }

  const uint32_t warp = threadIdx.x / 32;
  const uint32_t x = blockIdx.x % kExpertCtas, g = blockIdx.x / kExpertCtas;
  const uint32_t tok0 = g * kGroup;
  const uint32_t rows = min(kGroup, M - tok0);
  if (threadIdx.x == 0) {
    stamp<kTrace>(p, 0);
#pragma unroll
    for (uint32_t c = 0; c < kChunks; ++c) fr::mbar_init(fr::smem_addr(smem + L::kBarW + 8 * c), kLoadThreads);
#pragma unroll
    for (uint32_t i = 0; i < kRing; ++i) {
      fr::mbar_init(fr::smem_addr(smem + L::kBarFull + 8 * i), kLoadThreads);
      fr::mbar_init(fr::smem_addr(smem + L::kBarEmpty + 8 * i), kSplit);
    }
  }
  __syncthreads();
  // Router weights are static: copied before waiting on the producer grid.
  if (warp >= kWLoadWarp && warp < kXLoadWarp) load_weight_chunks<kSplit>(p, x * kCtaExperts, smem);
  device::PDLWaitPrimary<kUsePDL>();
  if (threadIdx.x == 0) stamp<kTrace>(p, 1);

  if (warp < kSplit) {
    float d[2][4] = {};
    gemv_chain<kSplit, kTrace>(p, warp, rows, smem, d);
    store_logits<kSplit>(p, warp, x * kCtaExperts, tok0, rows, smem, d);
    if (threadIdx.x == 0) stamp<kTrace>(p, 5);
  } else if (warp >= kXLoadWarp && warp < kQuantWarp) {
    load_token_chunks<kSplit>(p, tok0, rows, smem);
  } else if (warp >= kQuantWarp && x < rows) {
    quantize_row(p, tok0 + x);
  }

  // The dependent grid reads our outputs only after its own griddepcontrol.wait.
  device::PDLTriggerSecondary<kUsePDL>();
  if (warp == 0 && x < rows) token_topk<kTrace>(p, tok0 + x, s_cand);
}

template <bool kUsePDL, bool kTrace>
struct RouterWideFront {
  /// \brief Router logits, NVFP4 input and router_topk_tables' outputs for 17..64 tokens in one launch.
  /// \param workspace int32 [kWorkspaceWords] logits and expert-id exchange, zero before the first launch (stays zero
  /// after); launches sharing it must be stream-ordered.
  /// \param k_split 1 or 2: the cuBLAS router GEMM's K order at this token count (one chain, two K halves).
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView global_scale,
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView x_fp4,
      const tvm::ffi::TensorView x_sf,
      const tvm::ffi::TensorView expert_weights,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView expanded_idx_to_permuted_idx,
      const tvm::ffi::TensorView permuted_idx_to_token_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_batch_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_mn_limit,
      const tvm::ffi::TensorView num_non_exiting_ctas,
      const tvm::ffi::TensorView total_num_padded_tokens,
      const tvm::ffi::TensorView num_tokens_per_expert,
      const tvm::ffi::TensorView workspace,
      const tvm::ffi::TensorView trace,
      int64_t tile_log2,
      int64_t k_split) {
    using namespace host;
    auto M_ = SymbolicSize{"num_tokens"};
    auto X_ = SymbolicSize{"num_expanded"};
    auto P_ = SymbolicSize{"max_padded_plus_one"};
    auto C_ = SymbolicSize{"max_ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M_, kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(x);
    TensorMatcher({kExperts, kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(weight);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(global_scale);
    TensorMatcher({M_, kExperts}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({M_, kHidden / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(8).verify(x_fp4);
    TensorMatcher({M_, kBlocksPerRow}).with_dtype<uint8_t>().with_device(device).verify(x_sf);
    TensorMatcher({M_, kTopK}).with_dtype<bf16_t>().with_device(device).verify(expert_weights);
    TensorMatcher({M_, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({X_}).with_dtype<int32_t>().with_device(device).verify(expanded_idx_to_permuted_idx);
    TensorMatcher({P_}).with_dtype<int32_t>().with_device(device).verify(permuted_idx_to_token_idx);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(cta_idx_xy_to_batch_idx);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(cta_idx_xy_to_mn_limit);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_non_exiting_ctas);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(total_num_padded_tokens);
    TensorMatcher({kExperts}).with_dtype<int32_t>().with_device(device).verify(num_tokens_per_expert);
    TensorMatcher({kWorkspaceWords}).with_dtype<int32_t>().with_device(device).ensure_alignment(16).verify(workspace);
    if constexpr (kTrace) {
      TensorMatcher({(kTraceCtas + 1) * 8}).with_dtype<int64_t>().with_device(device).verify(trace);
    }
    const auto M = static_cast<uint32_t>(M_.unwrap());
    CHECK_HOST(M >= kMinTokens && M <= kMaxTokens)
        << "router_wide_front: num_tokens " << M << " outside " << kMinTokens << ".." << kMaxTokens;
    CHECK_HOST(k_split == 1 || k_split == 2) << "router_wide_front: k_split " << k_split;
    CHECK_HOST(static_cast<uint64_t>(X_.unwrap()) >= M * kTopK) << "router_wide_front: expanded map too small";
    CHECK_HOST(tile_log2 >= 3 && tile_log2 <= 8) << "router_wide_front: tile_log2 " << tile_log2;
    // FlashInfer getMaxNumCtasInBatchDim: one CTA per touched expert, plus full tiles of the remainder.
    const int64_t filled = std::min<int64_t>(kExperts, int64_t{M} * kTopK);
    const int64_t max_ctas = filled + (int64_t{M} * kTopK - filled) / (int64_t{1} << tile_log2);
    CHECK_HOST(C_.unwrap() >= max_ctas) << "router_wide_front: CTA tables too small";
    CHECK_HOST(P_.unwrap() >= (max_ctas << tile_log2) + 1) << "router_wide_front: permuted map too small";
    Params params{};
    params.x = static_cast<const bf16_t*>(x.data_ptr());
    params.weight = static_cast<const bf16_t*>(weight.data_ptr());
    params.global_scale = static_cast<const float*>(global_scale.data_ptr());
    params.logits = static_cast<bf16_t*>(logits.data_ptr());
    params.x_fp4 = static_cast<uint8_t*>(x_fp4.data_ptr());
    params.x_sf = static_cast<uint8_t*>(x_sf.data_ptr());
    params.expert_weights = static_cast<bf16_t*>(expert_weights.data_ptr());
    params.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    params.expanded_idx_to_permuted_idx = static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
    params.permuted_idx_to_token_idx = static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr());
    params.cta_idx_xy_to_batch_idx = static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr());
    params.cta_idx_xy_to_mn_limit = static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr());
    params.num_non_exiting_ctas = static_cast<int32_t*>(num_non_exiting_ctas.data_ptr());
    params.total_num_padded_tokens = static_cast<int32_t*>(total_num_padded_tokens.data_ptr());
    params.num_tokens_per_expert = static_cast<int32_t*>(num_tokens_per_expert.data_ptr());
    params.workspace = static_cast<uint32_t*>(workspace.data_ptr());
    params.trace = kTrace ? static_cast<uint64_t*>(trace.data_ptr()) : nullptr;
    params.num_tokens = M;
    params.tile_log2 = static_cast<uint32_t>(tile_log2);
    // One GEMV CTA per (16 experts, 16 tokens) tile, then the table CTA.
    const uint32_t grid = (M + kGroup - 1) / kGroup * kExpertCtas + 1;
    if (k_split == 1) {
      launch<1>(params, grid, device.unwrap());
    } else {
      launch<2>(params, grid, device.unwrap());
    }
  }

 private:
  template <uint32_t kSplit>
  static void launch(const Params& params, uint32_t grid, DLDevice device) {
    using namespace host;
    static const bool smem_attr_set = [] {
      CHECK_CUDA(cudaFuncSetAttribute(
          router_wide_front_kernel<kUsePDL, kSplit, kTrace>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          Smem::kBytes));
      return true;
    }();
    (void)smem_attr_set;
    LaunchKernel(grid, kThreads, device, Smem::kBytes)
        .enable_pdl(kUsePDL)(router_wide_front_kernel<kUsePDL, kSplit, kTrace>, params);
  }
};

}  // namespace qwen_router_wide_front

using qwen_router_wide_front::RouterWideFront;

}  // namespace sglang
