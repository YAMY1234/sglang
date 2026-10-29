/// \file gdn_replayssm_commit.cuh
/// \brief The ReplaySSM GDN compact commit: one CTA per layer, request and value
///        head folds the accepted ring rows into the stable and tracking states.
///
/// BF16 operands, FP32 accumulation in a different order than the Triton commit:
/// tolerance-equivalent (rtol=2^-7, atol=1e-6), not bitwise.

#pragma once

#include <sgl_kernel/utils.h>  // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, RuntimeDeviceCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace sglang {

namespace gdn_replayssm {

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
constexpr int KD = 128;
constexpr int VD = 128;
constexpr int NW = 4;
constexpr u32 FULL = 0xffffffffu;
// The ring length the Python capability gate admits.
constexpr int RING = 16;

struct alignas(16) U4 {
  u32 x, y, z, w;
};

struct Params {
  u16* h0;
  long long h0_layer, h0_slot;  // [layers, slots, HV, V, K]
  const u16* d;
  long long d_layer, d_slot;  // [layers, slots, HV, L, V]
  const u16* k;
  long long k_layer, k_slot;  // [layers, slots, H, L, K]
  const float* g;
  long long g_layer, g_slot;  // [layers, slots, HV, L]
  const u16* dr;
  long long dr_layer, dr_slot;  // low parts (layout of d), or null
  const u16* kr;
  long long kr_layer, kr_slot;  // low parts (layout of k), or null
  const void* sidx;
  const void* ridx;
  const int* write_pos;
  const int* cache_base;
  const signed char* is_flush;
  const void* acc;
  const void* tidx;
  const void* tstep;
  int sidx64, ridx64, acc64, tidx64, tstep64;
  int B;
  long long null_block;
};

// shared memory: the checkpoint [V][K], then the ring's keys, key low parts, updates, update low parts
constexpr int SH_STATE = 0;
constexpr int SH_K = VD * KD;
constexpr int SH_KR = SH_K + RING * KD;
constexpr int SH_D = SH_KR + RING * KD;
constexpr int SH_DR = SH_D + RING * VD;
constexpr int SH_END = SH_DR + RING * VD;
constexpr int kSmemBytes = 2 * SH_END;

__device__ __forceinline__ float bflo(u32 x) {
  return __int_as_float(x << 16);
}
__device__ __forceinline__ float bfhi(u32 x) {
  return __int_as_float(x & 0xffff0000u);
}
__device__ __forceinline__ float bf2f(u16 x) {
  return __int_as_float(((u32)x) << 16);
}
__device__ __forceinline__ u32 pack2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ u32 word(const U4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
__device__ __forceinline__ void mma16816(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688(float (&d)[4], u32 a0, u32 a1, u32 b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ long long ld_idx(const void* p, int i, int i64) {
  return i64 ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
}
__device__ __forceinline__ u32 smem_u32(const void* p) {
  return (u32)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, u32 bytes, u64* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
      "l"(src),
      "r"(bytes),
      "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void mbar_wait0(u64* bar) {
  u32 ready = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready)
                 : "r"(smem_u32(bar))
                 : "memory");
  } while (!ready);
}

// A fragments of bf16 hi / lo of (update[t][v] * decay[t]): rows v = vb + 16 m + r (+ 8), entries t = 2q, 2q + 1
// (a0 / a1) and 2q + 8, 2q + 9 (a2 / a3)
__device__ __forceinline__ void
a_frags(const float (&u)[2][2][4], const float (&dec)[4], u32 (&AH)[2][4], u32 (&AL)[2][4]) {
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = i & 1, jp = i >> 1;
      const float lo = u[m][h][2 * jp] * dec[2 * jp], hi = u[m][h][2 * jp + 1] * dec[2 * jp + 1];
      AH[m][i] = pack2(lo, hi);
      AL[m][i] = pack2(lo - bflo(AH[m][i]), hi - bfhi(AH[m][i]));
    }
}

// out = bf16(old e^total + K^T D) for this warp's rows, from the shared checkpoint and keys; B column n = r holds
// k = 32c + 8 (r >> 1) + 2e + (r & 1), so the C fragment of lane q is word e of its 16-byte chunk c
template <bool K16>
__device__ __forceinline__ void fold_store(
    const u16* sm,
    const int (&ph)[4],
    const u32 (&AH)[2][4],
    const u32 (&AL)[2][4],
    float etot,
    bool resid,
    u16* dst,
    int vb,
    int r,
    int q) {
  const u16* sk = sm + SH_K;
  const u16* skr = sm + SH_KR;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    U4 H[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        H[m][h] = *reinterpret_cast<const U4*>(sm + SH_STATE + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q);
    u32 out[2][2][4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int kk = 32 * c + 8 * (r >> 1) + 2 * e + (r & 1);
      const u32 b0 = (u32)sk[ph[0] * KD + kk] | ((u32)sk[ph[1] * KD + kk] << 16);
      const u32 b1 = K16 ? ((u32)sk[ph[2] * KD + kk] | ((u32)sk[ph[3] * KD + kk] << 16)) : 0u;
      u32 r0 = 0u, r1 = 0u;
      if (resid) {
        r0 = (u32)skr[ph[0] * KD + kk] | ((u32)skr[ph[1] * KD + kk] << 16);
        if (K16) r1 = (u32)skr[ph[2] * KD + kk] | ((u32)skr[ph[3] * KD + kk] << 16);
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        float a[4] = {0.f, 0.f, 0.f, 0.f};
        if (K16) {
          mma16816(a, AH[m][0], AH[m][1], AH[m][2], AH[m][3], b0, b1);
          mma16816(a, AL[m][0], AL[m][1], AL[m][2], AL[m][3], b0, b1);
          if (resid) {
            mma16816(a, AH[m][0], AH[m][1], AH[m][2], AH[m][3], r0, r1);
            mma16816(a, AL[m][0], AL[m][1], AL[m][2], AL[m][3], r0, r1);
          }
        } else {  // entries 8-15 carry no weight: the k8 half of the same products
          mma1688(a, AH[m][0], AH[m][1], b0);
          mma1688(a, AL[m][0], AL[m][1], b0);
          if (resid) {
            mma1688(a, AH[m][0], AH[m][1], r0);
            mma1688(a, AL[m][0], AL[m][1], r0);
          }
        }
        const u32 w0 = word(H[m][0], e), w1 = word(H[m][1], e);
        out[m][0][e] = pack2(fmaf(bflo(w0), etot, a[0]), fmaf(bfhi(w0), etot, a[1]));
        out[m][1][e] = pack2(fmaf(bflo(w1), etot, a[2]), fmaf(bfhi(w1), etot, a[3]));
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        U4 v4;
        v4.x = out[m][h][0];
        v4.y = out[m][h][1];
        v4.z = out[m][h][2];
        v4.w = out[m][h][3];
        *reinterpret_cast<U4*>(dst + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q) = v4;
      }
  }
}

// the window's decay: masked inclusive prefix of the log-decays (lane t holds entry t), e^{total - prefix_j}
__device__ __forceinline__ float window(float gt, int len, int lane, const int (&jj)[4], float (&dec)[4]) {
  float pf = lane < len ? gt : 0.f;
#pragma unroll
  for (int dd = 1; dd < 32; dd <<= 1) {
    const float o = __shfl_up_sync(FULL, pf, dd);
    if (lane >= dd) pf += o;
  }
  const float tot = __shfl_sync(FULL, pf, 31);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float pj = __shfl_sync(FULL, pf, jj[i]);
    dec[i] = jj[i] < len ? __expf(tot - pj) : 0.f;
  }
  return __expf(tot);
}

template <int HK, int HV, bool HAS_TRACK>
__global__ void __launch_bounds__(NW * 32) gdn_replayssm_commit_kernel(const Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __shared__ u64 bar;
  constexpr int GRP = HV / HK;
  const int cta = blockIdx.x;
  const int hv = cta % HV;
  const int n = (cta / HV) % p.B;
  const int layer = cta / (HV * p.B);
  const int hk = hv / GRP;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, q = lane & 3;
  const int vb = warp * 32;

  const long long slot = ld_idx(p.sidx, n, p.sidx64);
  const long long rid = ld_idx(p.ridx, n, p.ridx64);
  if (slot <= p.null_block) return;
  const bool resid = p.dr != nullptr && p.kr != nullptr;
  u16* sm = reinterpret_cast<u16*>(smem_raw);
  // the checkpoint and the whole ring slice need only the two indices: copies first, cursors meanwhile
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const u32 bytes = 2 * (VD * KD + (resid ? 4 : 2) * RING * KD);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)), "r"(bytes) : "memory");
    bulk_g2s(sm + SH_STATE, p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD, 2 * VD * KD, &bar);
    bulk_g2s(sm + SH_K, p.k + layer * p.k_layer + rid * p.k_slot + (long long)hk * RING * KD, 2 * RING * KD, &bar);
    bulk_g2s(sm + SH_D, p.d + layer * p.d_layer + rid * p.d_slot + (long long)hv * RING * VD, 2 * RING * VD, &bar);
    if (resid) {
      bulk_g2s(
          sm + SH_KR, p.kr + layer * p.kr_layer + rid * p.kr_slot + (long long)hk * RING * KD, 2 * RING * KD, &bar);
      bulk_g2s(
          sm + SH_DR, p.dr + layer * p.dr_layer + rid * p.dr_slot + (long long)hv * RING * VD, 2 * RING * VD, &bar);
    }
  }
  const int n_hist = p.write_pos[rid];
  const bool fold = p.is_flush[rid] != 0;
  const int base = p.cache_base[rid];
  long long tslot = p.null_block;
  int track_len = 0;
  if constexpr (HAS_TRACK) {
    tslot = ld_idx(p.tidx, n, p.tidx64);
    const long long ts = ld_idx(p.tstep, n, p.tstep64);
    if (!(tslot > p.null_block && ts >= 0)) tslot = p.null_block;
    track_len = n_hist - (int)ld_idx(p.acc, n, p.acc64) + (int)ts + 1;
  }
  const bool track = tslot > p.null_block;
  const float gt =
      lane < RING ? p.g[layer * p.g_layer + rid * p.g_slot + hv * RING + ((base + lane) & (RING - 1))] : 0.f;
  const int jj[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
  int ph[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    ph[i] = (base + jj[i]) & (RING - 1);
  __syncthreads();   // the barrier's init before anyone waits on it
  mbar_wait0(&bar);  // also before an early exit: no copy may land after the CTA is gone
  if (!fold && !track) return;
  // windows of at most 8 entries (every fold of a step of up to 8 draft rows) take the k8 products
  const bool k16 = (fold && n_hist > 8) || (track && track_len > 8);

  float u[2][2][4];  // updates (high + low, FP32): [m][half][entry jj]
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int v = vb + 16 * m + 8 * h + r;
        float x = 0.f;
        if (i < 2 || k16) {
          x = bf2f(sm[SH_D + ph[i] * VD + v]);
          if (resid) x += bf2f(sm[SH_DR + ph[i] * VD + v]);
        }
        u[m][h][i] = x;
      }
  if (fold) {
    float dec[4];
    const float etot = window(gt, n_hist, lane, jj, dec);
    u32 AH[2][4], AL[2][4];
    a_frags(u, dec, AH, AL);
    if (k16)
      fold_store<true>(
          sm,
          ph,
          AH,
          AL,
          etot,
          resid,
          p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD,
          vb,
          r,
          q);
    else
      fold_store<false>(
          sm,
          ph,
          AH,
          AL,
          etot,
          resid,
          p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD,
          vb,
          r,
          q);
  }
  if (track) {
    float dec[4];
    const float etot = window(gt, track_len, lane, jj, dec);
    u32 AH[2][4], AL[2][4];
    a_frags(u, dec, AH, AL);
    if (k16)
      fold_store<true>(
          sm,
          ph,
          AH,
          AL,
          etot,
          resid,
          p.h0 + layer * p.h0_layer + tslot * p.h0_slot + (long long)hv * VD * KD,
          vb,
          r,
          q);
    else
      fold_store<false>(
          sm,
          ph,
          AH,
          AL,
          etot,
          resid,
          p.h0 + layer * p.h0_layer + tslot * p.h0_slot + (long long)hv * VD * KD,
          vb,
          r,
          q);
  }
}

inline const void* optional_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& t) {
  return t.has_value() ? t.value().data_ptr() : nullptr;
}

inline int is_int64(const tvm::ffi::TensorView& t) {
  return t.dtype().bits == 64;
}

inline int optional_is_int64(const tvm::ffi::Optional<tvm::ffi::TensorView>& t) {
  return t.has_value() && is_int64(t.value());
}

}  // namespace gdn_replayssm

/// \brief The compact-commit part of ``commit_gdn_replayssm_circular``; the Python
///        capability gate has validated every dtype, shape and stride before the call.
template <int kHK, int kHV, bool kHasTrack>
struct GdnReplaySSMCommit {
  static void
  run(tvm::ffi::TensorView checkpoint_state,
      tvm::ffi::TensorView d_cache,
      tvm::ffi::TensorView k_cache,
      tvm::ffi::TensorView g_cache,
      tvm::ffi::Optional<tvm::ffi::TensorView> d_residual_cache,
      tvm::ffi::Optional<tvm::ffi::TensorView> k_residual_cache,
      tvm::ffi::TensorView state_batch_indices,
      tvm::ffi::TensorView replay_indices,
      tvm::ffi::TensorView write_pos,
      tvm::ffi::TensorView cache_base,
      tvm::ffi::TensorView is_flush,
      tvm::ffi::TensorView accept_lens,
      tvm::ffi::Optional<tvm::ffi::TensorView> mamba_track_indices,
      tvm::ffi::Optional<tvm::ffi::TensorView> mamba_steps_to_track,
      int64_t null_block_id) {
    using namespace gdn_replayssm;
    using namespace host;

    RuntimeCheck(mamba_track_indices.has_value() == kHasTrack, "track presence must match the instantiation");
    RuntimeCheck(d_residual_cache.has_value() == k_residual_cache.has_value(), "residual caches come in pairs");
    const bool resid = d_residual_cache.has_value();
    Params p{};
    p.h0 = static_cast<u16*>(checkpoint_state.data_ptr());
    p.h0_layer = checkpoint_state.stride(0);
    p.h0_slot = checkpoint_state.stride(1);
    p.d = static_cast<const u16*>(d_cache.data_ptr());
    p.d_layer = d_cache.stride(0);
    p.d_slot = d_cache.stride(1);
    p.k = static_cast<const u16*>(k_cache.data_ptr());
    p.k_layer = k_cache.stride(0);
    p.k_slot = k_cache.stride(1);
    p.g = static_cast<const float*>(g_cache.data_ptr());
    p.g_layer = g_cache.stride(0);
    p.g_slot = g_cache.stride(1);
    if (resid) {
      p.dr = static_cast<const u16*>(d_residual_cache.value().data_ptr());
      p.dr_layer = d_residual_cache.value().stride(0);
      p.dr_slot = d_residual_cache.value().stride(1);
      p.kr = static_cast<const u16*>(k_residual_cache.value().data_ptr());
      p.kr_layer = k_residual_cache.value().stride(0);
      p.kr_slot = k_residual_cache.value().stride(1);
    }
    p.sidx = state_batch_indices.data_ptr();
    p.ridx = replay_indices.data_ptr();
    p.write_pos = static_cast<const int*>(write_pos.data_ptr());
    p.cache_base = static_cast<const int*>(cache_base.data_ptr());
    p.is_flush = static_cast<const signed char*>(is_flush.data_ptr());
    p.acc = accept_lens.data_ptr();
    p.tidx = optional_ptr(mamba_track_indices);
    p.tstep = optional_ptr(mamba_steps_to_track);
    p.sidx64 = is_int64(state_batch_indices);
    p.ridx64 = is_int64(replay_indices);
    p.acc64 = is_int64(accept_lens);
    p.tidx64 = optional_is_int64(mamba_track_indices);
    p.tstep64 = optional_is_int64(mamba_steps_to_track);
    p.B = static_cast<int>(state_batch_indices.size(0));
    p.null_block = null_block_id;

    const auto kernel = gdn_replayssm_commit_kernel<kHK, kHV, kHasTrack>;
    const DLDevice device = checkpoint_state.device();
    // Each device context needs the opt-in once; the attribute call is idempotent.
    static bool smem_ready[64] = {};
    RuntimeCheck(device.device_id >= 0 && device.device_id < 64, "device id out of range");
    if (!smem_ready[device.device_id]) {
      RuntimeDeviceCheck(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      smem_ready[device.device_id] = true;
    }
    const int64_t layers = checkpoint_state.size(0);
    const auto grid = static_cast<uint32_t>(layers * p.B * kHV);
    LaunchKernel(grid, NW * 32, device, kSmemBytes)(kernel, p);
  }
};

}  // namespace sglang
