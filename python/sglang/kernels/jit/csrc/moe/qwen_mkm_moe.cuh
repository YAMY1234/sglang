// SPDX-License-Identifier: Apache-2.0
// Fused small-M NVFP4 MoE expert phase for Qwen3.5 TP8 decode verify (M <= 7 tokens, top-10, 512 experts,
// hidden 4096, intermediate 128 per rank): DynBlock-exact top-10 from the router logits, FC1, SwiGLU, NVFP4
// requantization and FC2 in one launch over the trtllm-gen weight layout. Writes the RT deferred-finalize rows
// (row t * 10 + k, identity permutation) and the expert weights.
// Device code generated from the MKM lane engine (source md5 1193f5e7f13b8261f987b2926714a1dc) by lanes/mkm/tools/export_cuh.py.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "qwen_mkm_sm100_ptx.cuh"

namespace sglang {

namespace qwen_mkm_moe {

using namespace q35mf;


constexpr int kH = 4096, kE = 512, kTopK = 10, kMaxTok = 7;
constexpr int kN1 = 256;            // W13 rows per expert (gate + up, TP8)
constexpr int kStageBytes = 36864;  // 32 KB weights + 4 KB scale factors
constexpr int kStages = 5;
constexpr int kFc1Stages = 8;       // 4 k-blocks each (K half = 2048)
constexpr int kFc2Stages = 4;       // 2 tiles each
constexpr int kTotStages = kFc1Stages + kFc2Stages;
constexpr int kThreads = 256;
// TMEM columns: accumulator [0,256); SFA [256,384) (x: 32 k-blocks x 4; h: 16 atoms x 4); SFB ring [384,512)
constexpr uint32_t kColSfa = 256, kColSfb = 384;
constexpr int kSfbSlots = 16;
constexpr int kPRow = 260;
#ifndef SF_UTCCP_FC1
#define SF_UTCCP_FC1 0
#endif  // partial row stride in floats: 16 B aligned, rows land on distinct banks

struct Params {
  const uint16_t* logits;    // [M, 512] bf16 router logits (router_gemm_quant output)
  uint16_t* expert_weights;  // [M, 10] bf16 out: DynBlock softmax-over-top-10 weights (CTA 0 writes)
  int32_t* topk_ids;         // [M, 10] out (CTA 0 writes)
  const uint8_t* x_sf;       // [M, 256] linear e4m3
  const uint8_t* w13_sf;     // [E, 2 row blocks x 64 k blocks x 512 B] (128x4 interleaved)
  const uint8_t* w2_sf;      // [E, 32 row blocks x 2 k blocks x 512 B]
  const float* g1_alpha;     // [E] gate dequant
  const float* g1_scale_c;   // [E] up dequant * FC2 input quant scale
  const float* g2_alpha;     // [E]
  __nv_bfloat16* out;        // [M * 10, 4096]
  long long* trace;          // optional [grid, 16] %globaltimer stamps
  int num_tokens;
};

struct __align__(1024) Smem {
  uint8_t ring[kStages][kStageBytes];
  alignas(1024) uint8_t xa[8 * 1024];  // x operand: 8 rows x K half (1024 B), SW128 (8 atoms of 8 rows x 128 B), by TMA
  float own[8 * kPRow];         // this CTA's fp32 FC1 partial [8][gate 0..127 | up 0..127 | pad]
  float peer[8 * kPRow];        // the peer CTA's fp32 FC1 partial, same layout
  alignas(1024) uint8_t ha[1024];  // h operand: 8 rows x 128 B (64 B used), SW128 (atom must be 1024 B aligned)
  uint8_t xsf[8][128];          // x scale bytes of this CTA's K half (row t = token t), by bulk copy
  uint32_t hsf[8][2];           // h scale words: token t, k-block kb (4 e4m3 bytes)
  uint64_t full[kStages], empty[kStages], sf_full[kStages];
  uint64_t x_full, sfa_ready, fc1_done, peer_bar, h_ready, fc2_done;
  uint32_t tmem;
  int expert;
  uint32_t tokmask;
  int kslot[8];
  uint32_t mask[16];
  uint32_t cand[8][32];
  int32_t ids[8][kTopK];
};

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t saddr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(saddr), "r"(rank));
  return r;
}
// smem (this CTA) -> smem (peer CTA in the cluster), completes tx bytes on the peer's mbarrier
__device__ __forceinline__ void bulk_s2s(uint32_t dst_cluster, uint32_t src, uint32_t bytes, uint32_t bar_cluster) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   dst_cluster),
               "r"(src), "r"(bytes), "r"(bar_cluster)
               : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void tmem_ld_x32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x64(uint32_t taddr, uint32_t (&r)[64]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]),
        "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]),
        "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]),
        "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]),
        "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_st_x8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(taddr), "r"(r[0]),
               "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
               : "memory");
}
__device__ __forceinline__ void tmem_st_x32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
      : "memory");
}
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t cvt_e4m3(float a) {
  uint32_t out;
  asm("{\n .reg .b16 pair;\n .reg .f32 zero;\n mov.f32 zero, 0f00000000;\n"
      " cvt.rn.satfinite.e4m3x2.f32 pair, zero, %1;\n cvt.u32.u16 %0, pair;\n}"
      : "=r"(out)
      : "f"(a));
  return out & 0xFFu;
}
__device__ __forceinline__ float e4m3_to_f32(uint32_t v) {
  float out;
  asm("{\n .reg .b16 pair;\n .reg .b32 h2;\n .reg .b16 lo, hi;\n cvt.u16.u32 pair, %1;\n"
      " cvt.rn.f16x2.e4m3x2 h2, pair;\n mov.b32 {lo, hi}, h2;\n cvt.f32.f16 %0, lo;\n}"
      : "=f"(out)
      : "r"(v));
  return out;
}
__device__ __forceinline__ uint32_t cvt_e2m1x8(const float* v) {
  uint32_t out;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      " mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}
// ---- router top-k, bit-exact with FlashInfer's single-block routing (routingCustom, RenormalizeNaive on bf16):
// key = bf16 order (cub float twiddle) above 65535 - expert, i.e. the higher score first, the smaller expert on ties.
__device__ __forceinline__ uint32_t topk_key(uint32_t bits, uint32_t expert) {
  const uint32_t twiddled = bits ^ ((bits & 0x8000u) ? 0xFFFFu : 0x8000u);
  return (twiddled << 16) | (65535u - expert);
}
__device__ __forceinline__ void topk_unkey(uint32_t key, uint32_t& bits, uint32_t& expert) {
  expert = 65535u - (key & 0xFFFFu);
  const uint32_t twiddled = key >> 16;
  bits = twiddled ^ ((twiddled & 0x8000u) ? 0x8000u : 0xFFFFu);
}
__device__ __forceinline__ uint32_t max16(const uint32_t (&k)[16]) {
  const uint32_t a = max(max(max(k[0], k[1]), max(k[2], k[3])), max(max(k[4], k[5]), max(k[6], k[7])));
  const uint32_t b = max(max(max(k[8], k[9]), max(k[10], k[11])), max(max(k[12], k[13]), max(k[14], k[15])));
  return max(a, b);
}
__device__ __forceinline__ uint32_t warp_max_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ uint32_t warp_sum_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ uint32_t warp_topk_of_lanes(uint32_t v, uint32_t lane) {
  uint32_t mine = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(v);
    v = v == top ? 0u : v;
    mine = lane == k ? top : mine;
  }
  return mine;
}
// Top-10 keys of 512 (16 per lane), lane k holding the k-th largest (keys unique and nonzero).
__device__ __forceinline__ uint32_t warp_top10(uint32_t (&keys)[16], uint32_t lane, uint32_t* s_cand) {
  uint32_t v = max16(keys);
  uint32_t bound = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    bound = warp_max_u32(v);
    v = v == bound ? 0u : v;
  }
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) count += keys[i] >= bound;
  const uint32_t total = warp_sum_u32(count);
  if (total <= 32) {
    uint32_t offset = count;
#pragma unroll
    for (uint32_t d = 1; d < 32; d <<= 1) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, offset, d);
      offset += lane >= d ? up : 0u;
    }
    offset -= count;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      if (keys[i] >= bound) s_cand[offset++] = keys[i];
    }
    __syncwarp();
    const uint32_t cand = lane < total ? s_cand[lane] : 0u;
    __syncwarp();
    return warp_topk_of_lanes(cand, lane);
  }
  uint32_t mine = 0;
#pragma unroll 1
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(max16(keys));
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) keys[i] = keys[i] == top ? 0u : keys[i];
    mine = lane == k ? top : mine;
  }
  return mine;
}
// FlashInfer calcSoftmax (bf16 scores) as built with -use_fast_math: expf -> ex2.approx.ftz of x * log2(e),
// division -> div.full.ftz; reductions are cooperative_groups::reduce like the reference.
__device__ __forceinline__ float fast_expf(float x) {
  float y;
  asm("{\n .reg .f32 t;\n mul.ftz.f32 t, %1, 0f3FB8AA3B;\n ex2.approx.ftz.f32 %0, t;\n}" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fast_div(float a, float b) {
  float y;
  asm("div.full.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ uint16_t softmax_topk(uint16_t score_bits, uint32_t lane) {
  namespace cg = cooperative_groups;
  const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const float si = __uint_as_float((uint32_t)score_bits << 16);
  float max_score = -INFINITY;
  if (lane < kTopK) max_score = si >= max_score ? si : max_score;
  max_score = cg::reduce(warp, max_score, cg::greater<float>());
  float sum_score = 0.f, new_score = 0.f;
  if (lane < kTopK) {
    new_score = fast_expf(__fsub_rn(si, max_score));
    sum_score = __fadd_rn(sum_score, new_score);
  }
  sum_score = cg::reduce(warp, sum_score, cg::plus<float>());
  uint16_t out = score_bits;
  if (lane < kTopK) {
    const __nv_bfloat16 b = __float2bfloat16_rn(fast_div(new_score, sum_score));
    out = *reinterpret_cast<const uint16_t*>(&b);
  }
  return out;
}

__device__ __forceinline__ void stamp(const Params& p, int slot) {
  if (p.trace) p.trace[blockIdx.x * 16 + slot] = (long long)gtime();
}
// Physical W13 column (accumulator column) holding reordered row q: block-32 shuffle of FlashInfer's
// get_shuffle_matrix_a_row_indices (epilogue_tile_m = 128); reordered row 2j is up_j, 2j + 1 is gate_j.
__device__ __forceinline__ int w13_col(int q) { return (q & ~31) + ((q & 31) % 4) * 8 + (q & 31) / 4; }
__host__ __device__ constexpr int w13_logical(int p) {
  return ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) % 2 ? ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) / 2
                                                     : 128 + ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) / 2;
}
template <int kC>
__device__ __forceinline__ void drain_cols(uint32_t tm, float* own, int lane) {
  uint32_t r[64];
  tmem_ld_x64(tm + (uint32_t)kC * 64, r);
  tmem_ld_wait();
  if (lane < 8) {
#pragma unroll
    for (int i = 0; i < 64; ++i) own[lane * kPRow + w13_logical(kC * 64 + i)] = __uint_as_float(r[i]);
  }
}

// FC2 drain of TMEM lane quarter q (tiles 4 q .. 4 q + 3), column half `half` (128 accumulator columns)
__device__ __forceinline__ void fc2_drain(const Params& p, Smem& s, uint32_t tm, int q, int lane, int half, int e,
                                          uint32_t rank) {
  mbar_wait(&s.fc2_done, 0);
  tc_fence_after();
  if (q == 0 && half == 0 && lane == 0) stamp(p, 7);
  const int row = q * 32 + lane, tile = row / 8, t = row % 8;
  const bool valid = t < p.num_tokens && ((s.tokmask >> t) & 1u);
  const float a2 = p.g2_alpha[e];
  __nv_bfloat16* orow = p.out + (size_t)(t * kTopK + s.kslot[t]) * kH + rank * 2048 + tile * 256 + half * 128;
#pragma unroll 1
  for (int c = 0; c < 2; ++c) {
    uint32_t r[64];
    tmem_ld_x64(tm + ((uint32_t)(q * 32) << 16) + (uint32_t)(half * 128 + c * 64), r);
    tmem_ld_wait();
    if (valid) {
#pragma unroll
      for (int b32 = 0; b32 < 2; ++b32) {
        uint32_t pk[16];
#pragma unroll
        for (int l = 0; l < 32; l += 2) {
          const float v0 = __uint_as_float(r[b32 * 32 + (l % 4) * 8 + l / 4]) * a2;
          const float v1 = __uint_as_float(r[b32 * 32 + ((l + 1) % 4) * 8 + (l + 1) / 4]) * a2;
          const __nv_bfloat162 b2 = __floats2bfloat162_rn(v0, v1);
          pk[l / 2] = *reinterpret_cast<const uint32_t*>(&b2);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
          *reinterpret_cast<uint4*>(orow + c * 64 + b32 * 32 + i * 8) =
              make_uint4(pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3]);
      }
    }
  }
}

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kThreads, 1)
s1_kernel(const __grid_constant__ CUtensorMap tm13, const __grid_constant__ CUtensorMap tm2,
          const __grid_constant__ CUtensorMap tmx, const Params p) {
  extern __shared__ __align__(1024) uint8_t raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t rank = cluster_rank();
  const int pair = blockIdx.x / 2;
  if (threadIdx.x == 0) {
    stamp(p, 0);
    for (int i = 0; i < kStages; ++i) {
      mbar_init(&s.full[i], 1);
      mbar_init(&s.empty[i], 1);
      mbar_init(&s.sf_full[i], 4);
    }
    mbar_init(&s.x_full, 1);
    mbar_init(&s.sfa_ready, 4);
    mbar_init(&s.fc1_done, 1);
    mbar_init(&s.peer_bar, 1);
    mbar_init(&s.h_ready, 4);
    mbar_init(&s.fc2_done, 1);
    mbar_expect_tx(&s.peer_bar, 8 * kPRow * 4);
    fence_mbar_init();
  }
  if (warp == 0 && lane < 16) s.mask[lane] = 0;
  if (warp == 2) { tmem_alloc(&s.tmem, 512); tmem_relinquish(); }
  if (warp == 1 && lane == 0) { prefetch_tmap(&tm13); prefetch_tmap(&tm2); prefetch_tmap(&tmx); }
  __syncthreads();
  cluster_arrive();  // waited on before the first DSMEM transfer and at exit
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (warp == 1 && lane == 0) {
    // x (8 atoms of this CTA's K half, rows >= M zero-filled) and its scale bytes; independent of the routing
    mbar_expect_tx(&s.x_full, 8 * 1024 + p.num_tokens * 128);
#pragma unroll
    for (int a = 0; a < 8; ++a) tma_load_2d(s.xa + a * 1024, &tmx, &s.x_full, (int)rank * 1024 + a * 128, 0);
    for (int t = 0; t < p.num_tokens; ++t) bulk_load(s.xsf[t], p.x_sf + (size_t)t * (kH / 16) + rank * 128, 128, &s.x_full);
  }

  // routing: warp t computes token t's top-10 (DynBlock-exact) from the bf16 logits; then the pair-th selected expert
  if (warp < p.num_tokens) {
    const uint4* src = reinterpret_cast<const uint4*>(p.logits + (size_t)warp * kE + lane * 16);
    const uint4 v0 = src[0], v1 = src[1];
    uint32_t keys[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t w = i < 4 ? (&v0.x)[i] : (&v1.x)[i - 4];
      keys[2 * i] = topk_key(w & 0xFFFFu, (uint32_t)(lane * 16 + 2 * i));
      keys[2 * i + 1] = topk_key(w >> 16, (uint32_t)(lane * 16 + 2 * i + 1));
    }
    uint32_t bits, expert;
    topk_unkey(warp_top10(keys, (uint32_t)lane, s.cand[warp]), bits, expert);
    const uint16_t wbits = softmax_topk((uint16_t)bits, (uint32_t)lane);
    if (lane < kTopK) {
      s.ids[warp][lane] = (int32_t)expert;
      atomicOr(&s.mask[expert >> 5], 1u << (expert & 31));
      if (blockIdx.x == 0) {
        p.expert_weights[warp * kTopK + lane] = wbits;
        p.topk_ids[warp * kTopK + lane] = (int32_t)expert;
      }
    }
  }
  __syncthreads();
  if (warp == 0) {
    const uint32_t word = lane < 16 ? s.mask[lane] : 0u;
    const int cnt = __popc(word);
    int incl = cnt;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int up = __shfl_up_sync(0xffffffffu, incl, d);
      if (lane >= d) incl += up;
    }
    const int excl = incl - cnt;
    int e = -1;
    if (pair >= excl && pair < incl) {
      uint32_t w = word;
      for (int q = pair - excl; q > 0; --q) w &= w - 1;
      e = lane * 32 + __ffs(w) - 1;
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) e = max(e, __shfl_xor_sync(0xffffffffu, e, d));
    uint32_t mbit = 0;
    int ks = 0;
    if (lane < p.num_tokens && e >= 0) {
#pragma unroll
      for (int k = 0; k < kTopK; ++k)
        if (s.ids[lane][k] == e) { mbit = 1u << lane; ks = k; }
    }
    const uint32_t mask = __reduce_or_sync(0xffffffffu, mbit);
    if (lane < 8) s.kslot[lane] = ks;
    if (lane == 0) { s.expert = e; s.tokmask = mask; }
  }
  __syncthreads();
  const int e = s.expert;
  const uint32_t tm = s.tmem;
  if (e < 0) {  // both CTAs of the pair agree; nothing was sent over DSMEM
    mbar_wait(&s.x_full, 0);  // no async copy may still target this CTA's smem
    cluster_wait();
    if (warp == 2) tmem_dealloc(tm, 512);
    return;
  }
  tc_fence_after();
  if (threadIdx.x == 0) stamp(p, 1);

  if (warp == 1) {
    if (lane == 0) {
      // ---------------- TMA producer: the weight stream
      for (int st = 0; st < kTotStages; ++st) {
        const int b = st % kStages;
        if (st >= kStages) mbar_wait(&s.empty[b], ((st / kStages) - 1) & 1);
        mbar_expect_tx(&s.full[b], kStageBytes);
        uint8_t* dst = s.ring[b];
        if (st < kFc1Stages) {
          const int kb0 = (int)rank * 32 + st * 4;
          tma_load_2d(dst, &tm13, &s.full[b], (int)rank * 1024 + st * 128, e * kN1);
          const uint8_t* sf = p.w13_sf + (size_t)e * (kN1 * kH / 16);
          bulk_load(dst + 32768, sf + (size_t)(0 * 64 + kb0) * 512, 2048, &s.full[b]);
          bulk_load(dst + 34816, sf + (size_t)(1 * 64 + kb0) * 512, 2048, &s.full[b]);
        } else {
          const int f = st - kFc1Stages;
#pragma unroll
          for (int tl = 0; tl < 2; ++tl) {
            const int j = f * 2 + tl;
            const int row0 = (int)rank * 2048 + j * 256;
            tma_load_2d(dst + tl * 16384, &tm2, &s.full[b], 0, e * kH + row0);
            bulk_load(dst + 32768 + tl * 2048, p.w2_sf + (size_t)e * (kH * 8) + (size_t)(row0 / 128) * 2 * 512, 2048,
                      &s.full[b]);
          }
        }
        if (st == 0) stamp(p, 2);
      }
    }
    __syncwarp();
  } else if (warp == 2) {
    if (lane == 0) {
      // ---------------- MMA issuer (tensor pipe: MMAs only)
      const uint32_t idesc = make_idesc_nvf4(128u, 256u);
      mbar_wait(&s.x_full, 0);
      mbar_wait(&s.sfa_ready, 0);
      tc_fence_after();
      int n = 0;
      for (int st = 0; st < kFc1Stages; ++st) {
        const int b = st % kStages;
        mbar_wait(&s.full[b], (st / kStages) & 1);
        if (!SF_UTCCP_FC1) mbar_wait(&s.sf_full[b], (st / kStages) & 1);
        tc_fence_after();
        if (st == 0) stamp(p, 3);
#pragma unroll
        for (int q = 0; q < 4; ++q, ++n) {
          const int kb = st * 4 + q;
#if SF_UTCCP_FC1
          {
            const uint32_t slot = kColSfb + (uint32_t)(n % kSfbSlots) * 8;
            utccp_32x128b_warpx4(tm + slot, make_sdesc(smem_u32(s.ring[b] + 32768 + q * 512), 0, 128, 0));
            utccp_32x128b_warpx4(tm + slot + 4, make_sdesc(smem_u32(s.ring[b] + 34816 + q * 512), 0, 128, 0));
          }
#endif
          umma_nvf4(tm, make_sdesc(smem_u32(s.xa + (kb / 4) * 1024) + (kb % 4) * 32, 0, 0, 2),
                    make_sdesc(smem_u32(s.ring[b]) + q * 32, 0, 1024, 2), idesc, tm + kColSfa + kb * 4,
                    tm + kColSfb + (uint32_t)(n % kSfbSlots) * 8, kb > 0);
        }
        tc_commit(&s.empty[b]);
      }
      tc_commit(&s.fc1_done);
      mbar_wait(&s.h_ready, 0);
      tc_fence_after();
      stamp(p, 6);
      for (int st = kFc1Stages; st < kTotStages; ++st) {
        const int b = st % kStages;
        mbar_wait(&s.full[b], (st / kStages) & 1);
        mbar_wait(&s.sf_full[b], (st / kStages) & 1);
        tc_fence_after();
#pragma unroll
        for (int tl = 0; tl < 2; ++tl) {
          const int j = (st - kFc1Stages) * 2 + tl;
#pragma unroll
          for (int kb = 0; kb < 2; ++kb, ++n) {
            umma_nvf4(tm, make_sdesc(smem_u32(s.ha) + kb * 32, 0, 0, 2),
                      make_sdesc(smem_u32(s.ring[b] + tl * 16384) + kb * 32, 0, 512, 4), idesc,
                      tm + kColSfa + (j * 2 + kb) * 4, tm + kColSfb + (uint32_t)(n % kSfbSlots) * 8, (j | kb) != 0);
          }
        }
        tc_commit(&s.empty[b]);
      }
      tc_commit(&s.fc2_done);
    }
    __syncwarp();
  } else if (warp == 0) {
    mbar_wait(&s.fc1_done, 0);
    tc_fence_after();
    drain_cols<2>(tm, s.own, lane);
    drain_cols<3>(tm, s.own, lane);
    tc_fence_before();
    fence_proxy_async_smem();
    named_bar(3, 64);
  } else if (warp >= 4) {
    // ---------------- TMEM-side warpgroup: q = lane quarter (TMEM lanes 32 q .. 32 q + 31)
    const int q = warp - 4;
    const uint32_t lane_addr = (uint32_t)(q * 32) << 16;
    // x scale factors: row m -> (lane m % 32, column m / 32) of each 4-column atom, every quarter holds a copy
    {
      mbar_wait(&s.x_full, 0);
      uint32_t r[32];
#pragma unroll 1
      for (int c4 = 0; c4 < 4; ++c4) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const int kb = c4 * 8 + i / 4, col = i % 4;
          uint32_t w = 0;
          if (col == 0 && lane < p.num_tokens) w = *reinterpret_cast<const uint32_t*>(&s.xsf[lane][kb * 4]);
          r[i] = w;
        }
        tmem_st_x32(tm + lane_addr + kColSfa + c4 * 32, r);
      }
      tmem_st_wait();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.sfa_ready);
    }
    // weight scale factors of every stage: k-block slot (8 columns) = [atom rows 0..127 | atom rows 128..255]
    auto stage_sf = [&](int st, int n0, int nkb, int atom_stride, int half_off) {
      const int b = st % kStages;
      mbar_wait(&s.full[b], (st / kStages) & 1);
      // the slots about to be overwritten were last read by the MMAs of stage st - 4
      if (st >= 4) mbar_wait(&s.empty[(st - 4) % kStages], ((st - 4) / kStages) & 1);
      const uint8_t* sf = s.ring[b] + 32768;
      for (int i = 0; i < nkb; ++i) {
        const int tl = i / 2, kb = i % 2;
        const uint8_t* a0 = atom_stride ? sf + tl * 2048 + kb * 512 : sf + i * 512;
        const uint8_t* a1 = a0 + half_off;
        const uint4 v0 = *reinterpret_cast<const uint4*>(a0 + lane * 16);
        const uint4 v1 = *reinterpret_cast<const uint4*>(a1 + lane * 16);
        const uint32_t r[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        tmem_st_x8(tm + lane_addr + kColSfb + (uint32_t)((n0 + i) % kSfbSlots) * 8, r);
      }
      tmem_st_wait();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.sf_full[b]);
    };
    if (!SF_UTCCP_FC1)
      for (int st = 0; st < kFc1Stages; ++st) stage_sf(st, st * 4, 4, 0, 2048);

    // FC1 partial: lanes 0..31 hold token rows; quarter 0 drains, everyone else waits for the exchange
    mbar_wait(&s.fc1_done, 0);
    tc_fence_after();
    if (q == 0) {
      if (lane == 0) stamp(p, 4);
      drain_cols<0>(tm, s.own, lane);
      drain_cols<1>(tm, s.own, lane);
      fence_proxy_async_smem();
      named_bar(3, 64);  // warp 0 drained columns [128, 256)
      cluster_wait();  // the peer's barriers are initialized (first cluster phase)
      if (lane == 0)
        bulk_s2s(mapa(smem_u32(s.peer), rank ^ 1u), smem_u32(s.own), 8 * kPRow * 4, mapa(smem_u32(&s.peer_bar), rank ^ 1u));
      __syncwarp();
    }
    tc_fence_before();
    mbar_wait(&s.peer_bar, 0);
    named_bar(2, 128);  // quarter 0's writes of the own partial are visible to the whole warpgroup
    if (threadIdx.x == 128) stamp(p, 5);
    const float* P0 = rank == 0 ? s.own : s.peer;
    const float* P1 = rank == 0 ? s.peer : s.own;
    const float ag = p.g1_alpha[e], sc = p.g1_scale_c[e];
    {
      const int it = threadIdx.x - 128;  // 0..127: (token row, 16-wide block, half)
      const int row = it / 16, blk = (it / 2) % 8, hf = it % 2;
      const int j0 = blk * 16 + hf * 8;
      const float4* p0r = reinterpret_cast<const float4*>(P0 + row * kPRow);
      const float4* p1r = reinterpret_cast<const float4*>(P1 + row * kPRow);
      const float4 ga0 = p0r[j0 / 4], gb0 = p0r[j0 / 4 + 1], ga1 = p1r[j0 / 4], gb1 = p1r[j0 / 4 + 1];
      const float4 ua0 = p0r[(128 + j0) / 4], ub0 = p0r[(128 + j0) / 4 + 1];
      const float4 ua1 = p1r[(128 + j0) / 4], ub1 = p1r[(128 + j0) / 4 + 1];
      const float g0[8] = {ga0.x, ga0.y, ga0.z, ga0.w, gb0.x, gb0.y, gb0.z, gb0.w};
      const float g1[8] = {ga1.x, ga1.y, ga1.z, ga1.w, gb1.x, gb1.y, gb1.z, gb1.w};
      const float u0[8] = {ua0.x, ua0.y, ua0.z, ua0.w, ub0.x, ub0.y, ub0.z, ub0.w};
      const float u1[8] = {ua1.x, ua1.y, ua1.z, ua1.w, ub1.x, ub1.y, ub1.z, ub1.w};
      float h[8];
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float g = (g0[i] + g1[i]) * ag;
        const float u = (u0[i] + u1[i]) * sc;
        const float sg = rcp_approx(ex2_approx(g * -1.4426950408889634f) + 1.0f);
        const float v = g * sg * u;
        h[i] = v;
        amax = fmaxf(amax, fabsf(v));
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      const uint32_t sf = cvt_e4m3(amax * rcp_approx(6.0f));
      const float sff = e4m3_to_f32(sf);
      const float osc = sff == 0.f ? 0.f : rcp_approx(sff);
      float qv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) qv[i] = h[i] * osc;
      const uint32_t packed = cvt_e2m1x8(qv);
      const int chunk = blk / 2;
      *reinterpret_cast<uint32_t*>(s.ha + row * 128 + ((chunk ^ row) * 16) + (blk % 2) * 8 + hf * 4) = packed;
      if (hf == 0) reinterpret_cast<uint8_t*>(&s.hsf[row][blk / 4])[blk % 4] = (uint8_t)sf;
    }
    named_bar(1, 128);
    // h scale factors: 16 atoms (tile jt, k-block kb) of 4 columns; row m = 32 c + lane is nonzero only in its tile
    {
      uint32_t r[32];
#pragma unroll 1
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const int a = half * 8 + i / 4, col = i % 4;  // atom a = 2 jt + kb
          const int jt = a / 2, kb = a % 2;
          const int row = 32 * col + lane;
          r[i] = (row / 8 == jt) ? s.hsf[row % 8][kb] : 0u;
        }
        tmem_st_x32(tm + lane_addr + kColSfa + half * 32, r);
      }
      tmem_st_wait();
    }
    fence_proxy_async_smem();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&s.h_ready);
    // FC2 weight scale factors (the MMA issuer waits for each stage's)
    for (int st = kFc1Stages; st < kTotStages; ++st) stage_sf(st, 32 + (st - kFc1Stages) * 4, 4, 1, 1024);

    if (q < 2) fc2_drain(p, s, tm, q, lane, 0, e, rank);
  }
  if (warp < 2) fc2_drain(p, s, tm, warp, lane, 1, e, rank);
  tc_fence_before();
  __syncthreads();
  // every warp but warp 4 still owes its wait on the first cluster phase; then exit together
  if (warp != 4) cluster_wait();
  cluster_arrive();
  cluster_wait();
  if (threadIdx.x == 0) stamp(p, 8);
  if (warp == 2) tmem_dealloc(tm, 512);
}

inline PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    CHECK_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&fn), cudaEnableDefault, &q));
    CHECK_HOST(fn != nullptr) << "qwen_mkm_moe: cuTensorMapEncodeTiled unavailable";
  }
  return fn;
}

inline CUtensorMap make_tmap(const void* base, uint64_t rows, uint64_t row_bytes, uint32_t box_bytes,
                             uint32_t box_rows, CUtensorMapSwizzle swz) {
  CUtensorMap m;
  cuuint64_t dims[2] = {row_bytes, rows};
  cuuint64_t strides[1] = {row_bytes};
  cuuint32_t box[2] = {box_bytes, box_rows};
  cuuint32_t estr[2] = {1, 1};
  const CUresult rc = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(rc == CUDA_SUCCESS) << "qwen_mkm_moe: tensor map encode failed " << static_cast<int>(rc);
  return m;
}

/// \brief Launcher of the fused expert phase; every CTA pair beyond the number of selected experts exits.
struct MkmMoe {
  /// \param logits         [M, 512] bf16 router logits (router_gemm_quant output), M <= 7.
  /// \param expert_weights [M, 10] bf16 out, DynBlock softmax-over-top-10 weights.
  /// \param topk_ids       [M, 10] int32 out, DynBlock top-10 order.
  /// \param x_fp4          [M, 2048] uint8 NVFP4 activations, x_sf [M, 256] uint8 (linear e4m3 scales).
  /// \param w13            [512, 256, 2048] uint8 (trtllm-gen shuffled), w13_sf [512, 256, 256] uint8 (e4m3).
  /// \param w2             [512, 4096, 64] uint8, w2_sf [512, 4096, 8] uint8 (e4m3).
  /// \param g1_alpha       [512] fp32 gate dequant, g1_scale_c [512] up dequant times the FC2 input scale,
  ///                        g2_alpha [512] fp32.
  /// \param out            [>= M * 10, 4096] bf16 out: row t * 10 + k is expert topk_ids[t, k] on token t.
  static void run(const tvm::ffi::TensorView logits,
                  const tvm::ffi::TensorView expert_weights,
                  const tvm::ffi::TensorView topk_ids,
                  const tvm::ffi::TensorView x_fp4,
                  const tvm::ffi::TensorView x_sf,
                  const tvm::ffi::TensorView w13,
                  const tvm::ffi::TensorView w13_sf,
                  const tvm::ffi::TensorView w2,
                  const tvm::ffi::TensorView w2_sf,
                  const tvm::ffi::TensorView g1_alpha,
                  const tvm::ffi::TensorView g1_scale_c,
                  const tvm::ffi::TensorView g2_alpha,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto M_ = SymbolicSize{"num_tokens"};
    auto R_ = SymbolicSize{"out_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M_, kE}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({M_, kTopK}).with_dtype<bf16_t>().with_device(device).verify(expert_weights);
    TensorMatcher({M_, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({M_, kH / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(x_fp4);
    TensorMatcher({M_, kH / 16}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(x_sf);
    TensorMatcher({kE, kN1, kH / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w13);
    TensorMatcher({kE, kN1, kH / 16}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w13_sf);
    TensorMatcher({kE, kH, 64}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
    TensorMatcher({kE, kH, 8}).with_dtype<uint8_t>().with_device(device).verify(w2_sf);
    // Scale-factor blocks are bulk-copied in 2 KB pieces from 512 B aligned offsets of the base.
    CHECK_HOST(reinterpret_cast<uintptr_t>(w2_sf.data_ptr()) % 16 == 0) << "qwen_mkm_moe: w2_sf base not 16 B aligned";
    TensorMatcher({kE}).with_dtype<fp32_t>().with_device(device).verify(g1_alpha).verify(g1_scale_c).verify(g2_alpha);
    TensorMatcher({R_, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out);
    const auto M = static_cast<int64_t>(M_.unwrap());
    CHECK_HOST(M >= 1 && M <= kMaxTok) << "qwen_mkm_moe: num_tokens " << M << " outside 1.." << kMaxTok;
    CHECK_HOST(R_.unwrap() >= M * kTopK) << "qwen_mkm_moe: out has " << R_.unwrap() << " rows, needs " << M * kTopK;
    const CUtensorMap m13 = make_tmap(w13.data_ptr(), static_cast<uint64_t>(kE) * kN1, kH / 2, 128, 256,
                                      CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap m2 = make_tmap(w2.data_ptr(), static_cast<uint64_t>(kE) * kH, 64, 64, 256,
                                     CU_TENSOR_MAP_SWIZZLE_64B);
    const CUtensorMap mx = make_tmap(x_fp4.data_ptr(), static_cast<uint64_t>(M), kH / 2, 128, 8,
                                     CU_TENSOR_MAP_SWIZZLE_128B);
    Params p{};
    p.logits = static_cast<const uint16_t*>(logits.data_ptr());
    p.expert_weights = static_cast<uint16_t*>(expert_weights.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.x_sf = static_cast<const uint8_t*>(x_sf.data_ptr());
    p.w13_sf = static_cast<const uint8_t*>(w13_sf.data_ptr());
    p.w2_sf = static_cast<const uint8_t*>(w2_sf.data_ptr());
    p.g1_alpha = static_cast<const float*>(g1_alpha.data_ptr());
    p.g1_scale_c = static_cast<const float*>(g1_scale_c.data_ptr());
    p.g2_alpha = static_cast<const float*>(g2_alpha.data_ptr());
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.trace = nullptr;
    p.num_tokens = static_cast<int>(M);
    constexpr size_t kSmem = sizeof(Smem) + 1024;
    static const bool smem_attr_set = [] {
      CHECK_CUDA(cudaFuncSetAttribute(s1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
      return true;
    }();
    (void)smem_attr_set;
    // Cluster dims are compiled into the kernel (__cluster_dims__); one 2-CTA cluster per possible expert of 70 pairs.
    LaunchKernel(dim3(2 * kMaxTok * kTopK), dim3(kThreads), device.unwrap(), kSmem)
        .enable_pdl(true)(s1_kernel, m13, m2, mx, p);
  }
};

}  // namespace qwen_mkm_moe

using qwen_mkm_moe::MkmMoe;

}  // namespace sglang
