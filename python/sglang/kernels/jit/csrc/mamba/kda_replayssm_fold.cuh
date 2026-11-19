// KDA ReplaySSM commit fold, row-parallel (SGLANG_OPT_KDA_REPLAYSSM_FAST_FOLD).
//
// Replays each request's accepted ring window into its fp32 checkpoint, like
// kda_replayssm_exact_fold_kernel (Triton, kda_replayssm_spec_decode.py), and
// reproduces that kernel's arithmetic bit for bit. The Triton kernel gives each
// lane four contiguous K elements of a value row, sums them in-thread and then
// butterflies over 32 lanes (xor 16 first). Here eight lanes share a row: lane j
// holds Triton lanes j, j+8, j+16 and j+24, so the xor-16 and xor-8 levels are
// in-register adds and only three shuffles remain. The l2 norm and decay of a
// token's key are computed once per CTA, not once per 8-column tile. Every float
// op is spelled in PTX (no .ftz, no contraction) to keep the Triton rounding.
//
// The fold streams ~2x the checkpoint pool per step, so it is a persistent,
// grid-stride kernel that cp.asyncs the next (request, layer, head, row tile)'s
// state and ring window into shared memory while it folds the current one.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <tuple>

namespace sglang {
namespace kda_fold {

constexpr int kD = 128;  // K == V
constexpr int kLanesPerRow = 8;

struct Params {
  float* __restrict__ state;         // [layers, slots, HV, V, K] fp32, folded in place
  const bf16_t* __restrict__ rawv;   // [layers, slots, HV, L, V]
  const bf16_t* __restrict__ rawk;   // [layers, slots, H, L, K]
  const float* __restrict__ g;       // [layers, slots, HV, L, K]
  const float* __restrict__ beta;    // [layers, slots, HV, L]
  const void* slot_idx;              // [B] int32 / int64
  const void* accept;                // [B]
  const void* track_idx;             // [B]
  const void* track_step;            // [B]
  int64_t state_layer, state_slot;
  int64_t rawv_layer, rawv_slot;
  int64_t rawk_layer, rawk_slot;
  int64_t g_layer, g_slot;
  int64_t beta_layer, beta_slot;
  int64_t slot_idx_stride, accept_stride, track_idx_stride, track_step_stride;
  int64_t null_block_id;
  int64_t num_items;
  int32_t layer_heads;  // layers * HV
  int32_t HV;
  int32_t heads_per_k;  // HV / H
  int32_t L;
  int32_t idx64_mask;  // bit 0 slot_idx, 1 accept, 2 track_idx, 3 track_step
};

SGL_DEVICE int64_t load_idx(const void* p, int64_t i, bool is64) {
  return is64 ? static_cast<const int64_t*>(p)[i] : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]);
}

SGL_DEVICE float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
SGL_DEVICE float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
SGL_DEVICE float sub_rn(float a, float b) {
  float d;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
SGL_DEVICE float ex2_approx(float a) {
  float d;
  asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
SGL_DEVICE float sqrt_approx_ftz(float a) {
  float d;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
SGL_DEVICE float div_full(float a, float b) {
  float d;
  asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

// Packed f32x2 (sm_100+): same per-lane IEEE rounding as the scalar ops.
SGL_DEVICE void mul2(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{\n\t.reg .b64 ra, rb, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\t"
      "mul.rn.f32x2 rd, ra, rb;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}
SGL_DEVICE void add2(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{\n\t.reg .b64 ra, rb, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\t"
      "add.rn.f32x2 rd, ra, rb;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}
// d = a * b + c, one rounding (the Triton update is contracted to fma.rn).
SGL_DEVICE void fma2(float& d0, float& d1, float a0, float a1, float b, float c0, float c1) {
  asm("{\n\t.reg .b64 ra, rb, rc, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %4};\n\tmov.b64 rc, {%5, %6};\n\t"
      "fma.rn.f32x2 rd, ra, rb, rc;\n\tmov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b), "f"(c0), "f"(c1));
}

// Sum of one Triton lane's four contiguous elements, in Triton's in-thread order
// ((e0 + e2) + (e1 + e3) for Triton 3.8/3.9: add.f32x2, then add).
SGL_DEVICE float lane_sum4(float e0, float e1, float e2, float e3) {
  float a, b;
  add2(a, b, e0, e1, e2, e3);
  return add_rn(a, b);
}

SGL_DEVICE void cp_async16(void* smem, const void* gmem) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(gmem) : "memory");
}
SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int kPending>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending) : "memory");
}

// One warp: normalized key and decay of token t (lane l owns elements 4l..4l+3).
SGL_DEVICE void stage_key(const bf16_t* k, const float* g, float* kn_out, float* e_out, int lane) {
  const uint2 kraw = *reinterpret_cast<const uint2*>(k + 4 * lane);
  const float4 gv = *reinterpret_cast<const float4*>(g + 4 * lane);
  const float k0 = __uint_as_float(kraw.x << 16), k1 = __uint_as_float(kraw.x & 0xffff0000u);
  const float k2 = __uint_as_float(kraw.y << 16), k3 = __uint_as_float(kraw.y & 0xffff0000u);
  float sq0, sq1, sq2, sq3;
  mul2(sq0, sq1, k0, k1, k0, k1);
  mul2(sq2, sq3, k2, k3, k2, k3);
  float s = lane_sum4(sq0, sq1, sq2, sq3);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s = add_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
  const float denom = sqrt_approx_ftz(add_rn(s, 1e-6f));
  float4 kn, e;
  kn.x = div_full(k0, denom);
  kn.y = div_full(k1, denom);
  kn.z = div_full(k2, denom);
  kn.w = div_full(k3, denom);
  constexpr float kLog2e = 1.44269502f;
  e.x = ex2_approx(mul_rn(gv.x, kLog2e));
  e.y = ex2_approx(mul_rn(gv.y, kLog2e));
  e.z = ex2_approx(mul_rn(gv.z, kLog2e));
  e.w = ex2_approx(mul_rn(gv.w, kLog2e));
  *reinterpret_cast<float4*>(kn_out + 4 * lane) = kn;
  *reinterpret_cast<float4*>(e_out + 4 * lane) = e;
}

// Fold n_commit tokens into this thread's rows (row r = r * 32 + rg of the tile).
// s_kn / s_e: [n][128]; s_v: [n][kRows] (value of each tile row); s_beta: [n].
template <int kRows, int kRowsPerPass, bool kHasTrack, typename TV>
SGL_DEVICE void fold_rows(
    float4 (&st)[kRows / kRowsPerPass][4],
    const float* s_kn,
    const float* s_e,
    const TV* s_v,
    const float* s_beta,
    int n_commit,
    int track_step,
    float* track_tile,  // the track slot's tile rows, or nullptr
    int j,
    int rg) {
  constexpr int kRowsPerThread = kRows / kRowsPerPass;
  for (int t = 0; t < n_commit; ++t) {
    float4 kn[4], e[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      kn[b] = *reinterpret_cast<const float4*>(s_kn + t * kD + b * 32 + 4 * j);
      e[b] = *reinterpret_cast<const float4*>(s_e + t * kD + b * 32 + 4 * j);
    }
    const float bt = s_beta[t];
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      float blk[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        float4& h4 = st[r][b];
        mul2(h4.x, h4.y, h4.x, h4.y, e[b].x, e[b].y);  // h *= exp(gk)
        mul2(h4.z, h4.w, h4.z, h4.w, e[b].z, e[b].w);
        float p0, p1, p2, p3;
        mul2(p0, p1, h4.x, h4.y, kn[b].x, kn[b].y);  // h * k
        mul2(p2, p3, h4.z, h4.w, kn[b].z, kn[b].w);
        blk[b] = lane_sum4(p0, p1, p2, p3);
      }
      // Triton lanes j+8b: the xor-16 level pairs blocks (0,2) and (1,3),
      // the xor-8 level adds the two results, then xor 4, 2, 1 across lanes.
      float s = add_rn(add_rn(blk[0], blk[2]), add_rn(blk[1], blk[3]));
      s = add_rn(s, __shfl_xor_sync(0xffffffffu, s, 4));
      s = add_rn(s, __shfl_xor_sync(0xffffffffu, s, 2));
      s = add_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
      const float v = static_cast<float>(s_v[t * kRows + r * kRowsPerPass + rg]);
      const float u = mul_rn(sub_rn(v, s), bt);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        float4& h4 = st[r][b];
        fma2(h4.x, h4.y, kn[b].x, kn[b].y, u, h4.x, h4.y);  // h += k * u
        fma2(h4.z, h4.w, kn[b].z, kn[b].w, u, h4.z, h4.w);
      }
    }
    if constexpr (kHasTrack) {
      // Interval-crossing snapshot: the state after step `track_step`.
      if (t == track_step && track_tile != nullptr) {
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
          float* row = track_tile + static_cast<int64_t>(r * kRowsPerPass + rg) * kD + 4 * j;
#pragma unroll
          for (int b = 0; b < 4; ++b) *reinterpret_cast<float4*>(row + b * 32) = st[r][b];
        }
      }
    }
  }
}

struct Item {
  int64_t slot;
  int64_t track_slot;
  int64_t layer;
  int32_t n_commit;  // 0: nothing to fold
  int32_t track_step;
  int32_t hv, h, tile;
};

template <int kRows, bool kHasTrack>
SGL_DEVICE Item load_item(const Params& p, int64_t item) {
  constexpr int kTiles = kD / kRows;
  Item it{};
  it.n_commit = 0;
  it.track_slot = p.null_block_id;
  it.track_step = -1;
  if (item >= p.num_items) return it;
  it.tile = static_cast<int>(item % kTiles);
  const int64_t rest = item / kTiles;
  const int lh = static_cast<int>(rest % p.layer_heads);
  const int64_t n = rest / p.layer_heads;
  it.layer = lh / p.HV;
  it.hv = lh % p.HV;
  it.h = it.hv / p.heads_per_k;
  it.slot = load_idx(p.slot_idx, n * p.slot_idx_stride, p.idx64_mask & 1);
  int n_commit = static_cast<int>(load_idx(p.accept, n * p.accept_stride, p.idx64_mask & 2));
  if (it.slot <= p.null_block_id || n_commit <= 0) return it;
  it.n_commit = n_commit < p.L ? n_commit : p.L;
  if constexpr (kHasTrack) {
    it.track_slot = load_idx(p.track_idx, n * p.track_idx_stride, p.idx64_mask & 4);
    it.track_step = static_cast<int>(load_idx(p.track_step, n * p.track_step_stride, p.idx64_mask & 8));
  }
  return it;
}

// Shared-memory stage of one item: state tile and the raw key / gate / value /
// beta of up to kMaxTok tokens (longer windows restage the rest in place).
template <int kRows, int kMaxTok>
struct StageLayout {
  static constexpr int64_t kBytes =
      static_cast<int64_t>(kRows) * kD * 4 + kMaxTok * kD * 2 + kMaxTok * kD * 4 + kMaxTok * kRows * 2 + kMaxTok * 4;
  static constexpr int64_t smem(int stages) {
    return stages * kBytes + 2 * kMaxTok * kD * 4 + stages * static_cast<int64_t>(sizeof(Item));
  }
};

template <int kRows, int kStages, int kMaxTok, int kThreads, bool kHasTrack>
__global__ __launch_bounds__(kThreads, 1) void kda_replayssm_fold_kernel(const Params __grid_constant__ p) {
  constexpr int kRowsPerPass = kThreads / kLanesPerRow;
  constexpr int kWarps = kThreads / 32;
  constexpr int kRowsPerThread = kRows / kRowsPerPass;
  constexpr int kStateChunks = kRows * kD / 4;  // 16-byte chunks
  constexpr int kVChunks = kRows / 8;
  constexpr int64_t stage_bytes = StageLayout<kRows, kMaxTok>::kBytes;
  static_assert(kRows % kRowsPerPass == 0 && kD % kRows == 0 && kStages >= 2 && kMaxTok % 4 == 0);
  static_assert(stage_bytes % 16 == 0);

  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int L = p.L;
  float* s_kn = reinterpret_cast<float*>(smem_raw + kStages * stage_bytes);  // [kMaxTok][128]
  float* s_e = s_kn + kMaxTok * kD;                                            // [kMaxTok][128]
  Item* s_item = reinterpret_cast<Item*>(s_e + kMaxTok * kD);                  // [kStages]

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int j = tid % kLanesPerRow;
  const int rg = tid / kLanesPerRow;

  auto stage_state = [&](int s) { return reinterpret_cast<float*>(smem_raw + s * stage_bytes); };
  auto stage_k = [&](int s) { return reinterpret_cast<bf16_t*>(stage_state(s) + kRows * kD); };
  auto stage_g = [&](int s) { return reinterpret_cast<float*>(stage_k(s) + kMaxTok * kD); };
  auto stage_v = [&](int s) { return reinterpret_cast<bf16_t*>(stage_g(s) + kMaxTok * kD); };
  auto stage_beta = [&](int s) { return reinterpret_cast<float*>(stage_v(s) + kMaxTok * kRows); };
  auto ring = [&](const Item& it) {
    const int64_t hv = it.hv;
    return std::make_tuple(
        p.rawk + it.layer * p.rawk_layer + it.slot * p.rawk_slot + static_cast<int64_t>(it.h) * L * kD,
        p.g + it.layer * p.g_layer + it.slot * p.g_slot + hv * L * kD,
        p.rawv + it.layer * p.rawv_layer + it.slot * p.rawv_slot + hv * L * kD + it.tile * kRows,
        p.beta + it.layer * p.beta_layer + it.slot * p.beta_slot + hv * L);
  };

  // Issue the cp.asyncs of one item into stage s (one commit group per item).
  auto issue = [&](const Item& it, int s) {
    if (tid == 0) s_item[s] = it;
    if (it.n_commit > 0) {
      const int64_t hv = it.hv;
      const float* state = p.state + it.layer * p.state_layer + it.slot * p.state_slot + hv * kD * kD +
                           static_cast<int64_t>(it.tile) * kRows * kD;
      float* dst = stage_state(s);
#pragma unroll
      for (int c = tid; c < kStateChunks; c += kThreads) cp_async16(dst + 4 * c, state + 4 * c);
      const int n = it.n_commit < kMaxTok ? it.n_commit : kMaxTok;
      const auto [k, g, v, beta] = ring(it);
      for (int c = tid; c < n * (kD / 8); c += kThreads) cp_async16(stage_k(s) + 8 * c, k + 8 * c);
      for (int c = tid; c < n * (kD / 4); c += kThreads) cp_async16(stage_g(s) + 4 * c, g + 4 * c);
      for (int c = tid; c < n * kVChunks; c += kThreads) {
        const int t = c / kVChunks, q = c % kVChunks;
        cp_async16(stage_v(s) + t * kRows + 8 * q, v + t * kD + 8 * q);
      }
      for (int c = tid; c < (n + 3) / 4; c += kThreads) cp_async16(stage_beta(s) + 4 * c, beta + 4 * c);
    }
    cp_async_commit();
  };

  const int64_t stride = gridDim.x;
  int64_t item = blockIdx.x;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) issue(load_item<kRows, kHasTrack>(p, item + s * stride), s);
  Item next = load_item<kRows, kHasTrack>(p, item + (kStages - 1) * stride);

  for (int k = 0; item < p.num_items; ++k, item += stride) {
    const int s = k % kStages;
    issue(next, (k + kStages - 1) % kStages);
    next = load_item<kRows, kHasTrack>(p, item + kStages * stride);
    cp_async_wait<kStages - 1>();
    __syncthreads();

    const Item it = s_item[s];
    if (it.n_commit > 0) {
      float4 st[kRowsPerThread][4];
      const float* src = stage_state(s);
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          st[r][b] = *reinterpret_cast<const float4*>(src + (r * kRowsPerPass + rg) * kD + b * 32 + 4 * j);
        }
      }
      const int64_t tile_off = static_cast<int64_t>(it.hv) * kD * kD + static_cast<int64_t>(it.tile) * kRows * kD;
      float* track_tile = nullptr;
      if constexpr (kHasTrack) {
        if (it.track_slot > p.null_block_id) {
          track_tile = p.state + it.layer * p.state_layer + it.track_slot * p.state_slot + tile_off;
        }
      }
      for (int t0 = 0; t0 < it.n_commit; t0 += kMaxTok) {
        const int n = it.n_commit - t0 < kMaxTok ? it.n_commit - t0 : kMaxTok;
        if (t0 > 0) {  // window longer than a stage: restage the next tokens in place
          __syncthreads();
          const auto [k, g, v, beta] = ring(it);
          for (int c = tid; c < n * (kD / 8); c += kThreads) {
            *reinterpret_cast<uint4*>(stage_k(s) + 8 * c) = *reinterpret_cast<const uint4*>(k + t0 * kD + 8 * c);
          }
          for (int c = tid; c < n * (kD / 4); c += kThreads) {
            *reinterpret_cast<float4*>(stage_g(s) + 4 * c) = *reinterpret_cast<const float4*>(g + t0 * kD + 4 * c);
          }
          for (int c = tid; c < n * kRows; c += kThreads) {
            stage_v(s)[c] = v[(t0 + c / kRows) * kD + c % kRows];
          }
          for (int c = tid; c < n; c += kThreads) stage_beta(s)[c] = beta[t0 + c];
          __syncthreads();
        }
        for (int t = warp; t < n; t += kWarps) {
          stage_key(stage_k(s) + t * kD, stage_g(s) + t * kD, s_kn + t * kD, s_e + t * kD, lane);
        }
        __syncthreads();  // keys staged
        fold_rows<kRows, kRowsPerPass, kHasTrack>(
            st, s_kn, s_e, stage_v(s), stage_beta(s), n, it.track_step - t0, track_tile, j, rg);
      }
      float* out = p.state + it.layer * p.state_layer + it.slot * p.state_slot + tile_off;
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
        float* row = out + static_cast<int64_t>(r * kRowsPerPass + rg) * kD + 4 * j;
#pragma unroll
        for (int b = 0; b < 4; ++b) __stcs(reinterpret_cast<float4*>(row + b * 32), st[r][b]);
      }
    }
    __syncthreads();  // stage s and the key buffers are free again
  }
  cp_async_wait<0>();
}

template <int kRows, int kStages, int kMaxTok, int kThreads, bool kHasTrack>
struct KdaReplayFold {
  static constexpr auto kernel = kda_replayssm_fold_kernel<kRows, kStages, kMaxTok, kThreads, kHasTrack>;

  static int64_t idx64(const tvm::ffi::TensorView t) {
    using namespace host;
    RuntimeCheck(t.ndim() == 1, "index tensors must be 1-D");
    const bool is64 = is_type<int64_t>(t.dtype());
    RuntimeCheck(is64 || is_type<int32_t>(t.dtype()), "index tensors must be int32 or int64");
    return is64 ? 1 : 0;
  }

  static void
  run(const tvm::ffi::TensorView state,
      const tvm::ffi::TensorView rawv,
      const tvm::ffi::TensorView rawk,
      const tvm::ffi::TensorView g,
      const tvm::ffi::TensorView beta,
      const tvm::ffi::TensorView slot_idx,
      const tvm::ffi::TensorView accept,
      const tvm::ffi::TensorView track_idx,
      const tvm::ffi::TensorView track_step,
      int64_t null_block_id,
      int64_t max_ctas,
      int64_t stream) {
    using namespace host;
    RuntimeCheck(state.ndim() == 5 && rawv.ndim() == 5 && rawk.ndim() == 5 && g.ndim() == 5 && beta.ndim() == 4,
                 "expected all-layer [layers, slots, ...] state and rings");
    RuntimeCheck(is_type<fp32_t>(state.dtype()) && is_type<bf16_t>(rawv.dtype()) && is_type<bf16_t>(rawk.dtype()) &&
                     is_type<fp32_t>(g.dtype()) && is_type<fp32_t>(beta.dtype()),
                 "expected fp32 state, bf16 rawv/rawk, fp32 g/beta");
    const int64_t layers = state.size(0);
    const int64_t HV = state.size(2);
    const int64_t H = rawk.size(2);
    const int64_t L = rawv.size(3);
    constexpr int64_t D = kda_fold::kD;
    RuntimeCheck(state.size(3) == D && state.size(4) == D, "expected K == V == 128");
    RuntimeCheck(state.stride(4) == 1 && state.stride(3) == D && state.stride(2) == D * D,
                 "state must be contiguous within a slot");
    RuntimeCheck(rawv.stride(4) == 1 && rawv.stride(3) == D && rawv.stride(2) == L * D,
                 "rawv must be contiguous within a slot");
    RuntimeCheck(rawk.stride(4) == 1 && rawk.stride(3) == D && rawk.stride(2) == L * D,
                 "rawk must be contiguous within a slot");
    RuntimeCheck(g.stride(4) == 1 && g.stride(3) == D && g.stride(2) == L * D, "g must be contiguous within a slot");
    RuntimeCheck(beta.stride(3) == 1 && beta.stride(2) == L, "beta must be contiguous within a slot");
    RuntimeCheck(L % 4 == 0, "ring length must be a multiple of 4");
    RuntimeCheck(H > 0 && HV % H == 0, "HV must be a multiple of H");
    const int64_t B = slot_idx.size(0);
    RuntimeCheck(accept.size(0) >= B, "accept_lens shorter than the state indices");
    if (B == 0) return;

    Params params{};
    params.state = static_cast<float*>(state.data_ptr());
    params.rawv = static_cast<const bf16_t*>(rawv.data_ptr());
    params.rawk = static_cast<const bf16_t*>(rawk.data_ptr());
    params.g = static_cast<const float*>(g.data_ptr());
    params.beta = static_cast<const float*>(beta.data_ptr());
    params.slot_idx = slot_idx.data_ptr();
    params.accept = accept.data_ptr();
    params.track_idx = track_idx.data_ptr();
    params.track_step = track_step.data_ptr();
    params.state_layer = state.stride(0);
    params.state_slot = state.stride(1);
    params.rawv_layer = rawv.stride(0);
    params.rawv_slot = rawv.stride(1);
    params.rawk_layer = rawk.stride(0);
    params.rawk_slot = rawk.stride(1);
    params.g_layer = g.stride(0);
    params.g_slot = g.stride(1);
    params.beta_layer = beta.stride(0);
    params.beta_slot = beta.stride(1);
    params.slot_idx_stride = slot_idx.stride(0);
    params.accept_stride = accept.stride(0);
    params.track_idx_stride = track_idx.stride(0);
    params.track_step_stride = track_step.stride(0);
    params.null_block_id = null_block_id;
    params.layer_heads = static_cast<int32_t>(layers * HV);
    params.HV = static_cast<int32_t>(HV);
    params.heads_per_k = static_cast<int32_t>(HV / H);
    params.L = static_cast<int32_t>(L);
    params.idx64_mask = static_cast<int32_t>(idx64(slot_idx) | (idx64(accept) << 1));
    if constexpr (kHasTrack) {
      params.idx64_mask |= static_cast<int32_t>((idx64(track_idx) << 2) | (idx64(track_step) << 3));
    }
    params.num_items = B * layers * HV * (D / kRows);

    const std::size_t smem = static_cast<std::size_t>(StageLayout<kRows, kMaxTok>::smem(kStages));
    RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int device = 0, sms = 0, per_sm = 0;
    RuntimeDeviceCheck(cudaGetDevice(&device));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    RuntimeDeviceCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, smem));
    RuntimeCheck(per_sm > 0, "fold kernel does not fit on an SM");
    int64_t grid = static_cast<int64_t>(sms) * per_sm;
    if (max_ctas > 0 && max_ctas < grid) grid = max_ctas;
    if (grid > params.num_items) grid = params.num_items;
    LaunchKernel(static_cast<uint32_t>(grid), kThreads, reinterpret_cast<cudaStream_t>(stream), smem)(kernel, params);
  }
};

}  // namespace kda_fold
}  // namespace sglang
