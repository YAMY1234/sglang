# SPDX-License-Identifier: Apache-2.0
"""The GDN target-verify core as one CUDA launch: the ReplaySSM verify on the circular ring (non-flush rows), this
step's ring entries, and the gated RMSNorm writing out_proj's static FP8 input.

Ported from the sgl_nvidia_mega plugin's ``gdn_core`` (``qwen3_5/sm103/gdn/src/core.cu``, served path only: no fold
in verify), which adapts TensorRT-LLM's ``q35_gdn_core`` to SGLang's ``gdn_replayssm_spec_circular_kernel`` and
gated RMSNorm (``layernorm_gated._layer_norm_fwd``). One CTA per (request, value head), 4 warps of 32 value columns;
the products run as BF16 m16n8k16 MMAs with FP32 accumulation, the forward substitution, the intra-step output and
the norm in FP32. The BF16 roundings of SGLang's kernels are kept (normalized q, key high / low, update high / low,
the replay output before the norm, the norm output before the FP8 scale), and the q / k norms, gates and exps use
the Triton kernels' summation order and fp32 forms, so the key and decay ring entries are bitwise the Triton
verify's; the products and the substitution sum in other orders, so updates and output are numeric, not bitwise,
against the Triton verify + ``_layer_norm_fwd(static_quant_scale=...)``. The ring
entries go where the Triton verify writes them (``(cache_base + write_pos + t) mod 16`` while ``write_pos + t < 16``)
and the cursors are only read, so ``commit_gdn_replayssm_circular`` runs unchanged.

Launched as a programmatic dependent: before ``griddepcontrol.wait`` it reads only earlier-step or static data
(indices, ring cursors, checkpoint rows, A_log / dt_bias / norm weight / FP8 scale); every activation read and every
global store follows the wait; dependents are released at exit. Compiled with NVRTC on the first eager call; a call
under CUDA-graph capture before that returns False from ``try_core`` and the caller keeps the Triton path.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_SRC = r"""
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;

#define KD 128
#define VD 128
#define RING 16
#define MT (VD / (16 * NW))
#define FULL 0xffffffffu
#ifndef CONV
#define CONV 0
#endif
#define CCH (2 * KD + VD)  // CONV: the CTA's channels: its key head's q, k, then its value head's v

struct alignas(16) U4 { u32 x, y, z, w; };

struct Params {
  const u16* x;          // convolved q | k | v rows
  long long x_stride;
  const u16* z;          // gate rows: z + row * z_stride + hv * VD
  long long z_stride;
  const u16* a;
  const u16* b;
  long long ab_stride;
  const void* A_log;
  const void* dt_bias;
  const u16* state;      // [slots, HV, V, K]
  long long state_slot;
  u16* ring_d;           // [slots, HV, RING, V]
  long long d_slot;
  u16* ring_k;           // [slots, HK, RING, K]
  long long k_slot;
  float* ring_g;         // [slots, HV, RING]
  long long g_slot;
  u16* ring_dlo;         // update low parts (layout of ring_d), or null
  long long dlo_slot;
  u16* ring_klo;         // key low parts (layout of ring_k), or null
  long long klo_slot;
  const void* state_indices;
  const void* replay_indices;
  int state_index64;     // int64 (else int32) state / replay indices
  int replay_index64;
  int alog_bf16;         // BF16 (else FP32) A_log / dt_bias
  int dtb_bf16;
  const int* write_pos;
  const int* cache_base;
  const signed char* is_flush;
  const void* norm_w;
  const float* scale;
  float eps;
  float qscale;
  u8* y;                 // FP8 out_proj input [rows, HV * V]
  long long y_stride;
  u16* o_out;            // [rows, HV, V] replay output (STORE_O, checks only)
  const u16* conv_w;     // CONV: [channels, 4] weights
  long long cw_ch;
  long long cw_tap;
  const u16* conv_state; // CONV: [slots, channels, 3] by state index (read only)
  long long cs_slot;
  long long cs_ch;
  long long cs_tap;
  u16* conv_win;         // CONV: deduplicated windows [window slots, channels, >= TT + 2]
  long long win_slot;
  long long win_ch;
  const void* inter_indices;  // CONV: window slot of each request
  int inter_index64;
};

__device__ __forceinline__ float bflo(u32 x) { return __int_as_float(x << 16); }
__device__ __forceinline__ float bfhi(u32 x) { return __int_as_float(x & 0xffff0000u); }
__device__ __forceinline__ float bf2f(u16 x) { return __int_as_float(((u32)x) << 16); }
__device__ __forceinline__ u16 f2bf(float x) {
  u16 r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float bfround(float x) { return bf2f(f2bf(x)); }
__device__ __forceinline__ u32 pack2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ u32 word(const U4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ U4 ld16(const void* p) { return *reinterpret_cast<const U4*>(p); }
__device__ __forceinline__ void st16(void* p, U4 v) { *reinterpret_cast<U4*>(p) = v; }
__device__ __forceinline__ U4 zero4() { U4 z; z.x = z.y = z.z = z.w = 0u; return z; }
// the Triton kernels' fp32 forms: tl.exp = ex2.approx of x log2(e), `/` = div.full, tl.sqrt = sqrt.approx
__device__ __forceinline__ float tl_exp(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x * 1.4426950408889634f));
  return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float sqrt_approx(float x) {
  float r;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ u8 f2e4m3(float x) {
  u16 r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(r) : "f"(x));
  return (u8)(r & 0xff);
}
__device__ __forceinline__ float pick8(const float (&a)[8], int i) {
  float r = a[0];
#pragma unroll
  for (int k = 1; k < 8; ++k) r = (i == k) ? a[k] : r;
  return r;
}
__device__ __forceinline__ void mma16816(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// K-reduction of 128-wide rows: A = rows r (rows r + 8 zero), B = 8 rows as columns. Lane (r, q) holds the
// elements [32 c + 8 q, + 8) of its row in chunk c; k-step 2c + e takes words (2e, 2e + 1) of both operands.
__device__ __forceinline__ void kred(float (&acc)[4], const U4 (&A)[4], const U4 (&B)[4]) {
#pragma unroll
  for (int c = 0; c < 4; ++c)
#pragma unroll
    for (int e = 0; e < 2; ++e)
      mma16816(acc, word(A[c], 2 * e), 0u, word(A[c], 2 * e + 1), 0u, word(B[c], 2 * e), word(B[c], 2 * e + 1));
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ long long load_index(const void* p, int i, bool i64) {
  return i64 ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
}
__device__ __forceinline__ float load_param(const void* p, int i, bool bf16) {
  return bf16 ? bf2f(reinterpret_cast<const u16*>(p)[i]) : reinterpret_cast<const float*>(p)[i];
}

#if CONV
#define CPT ((CCH + NW * 32 - 1) / (NW * 32))  // channels a thread
// global channel of the CTA's local channel lch: q | k of key head hk, then v of value head hv
__device__ __forceinline__ long long conv_channel(int lch, int hk, int hv) {
  return lch < KD ? (long long)hk * KD + lch
                  : lch < 2 * KD ? (long long)HK * KD + hk * KD + (lch - KD)
                                 : (long long)2 * HK * KD + (long long)hv * VD + (lch - 2 * KD);
}
// The causal conv update of this thread's channels, as causal_conv1d_update computes it: acc = 0; acc += bf16(seq[t + i]
// w_i) for i < 4 over seq = (old state, raw rows); out = bf16(acc / (1 + tl.exp(-acc))); with s_x, the outputs into
// s_x[t][lch]; the deduplicated window row (state 1, state 2, raw rows) of the channels this CTA owns.
__device__ __forceinline__ void conv_stage(const Params& p, long long row0, int hk, int hv, int tid, long long wslot,
                                           const u16 (&cw)[CPT][4], const u16 (&cs)[CPT][3], u16 (*s_x)[CCH + 8]) {
  constexpr int GRP = HV / HK;
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    const int lch = tid + i * NW * 32;
    if (lch >= CCH) break;
    const long long gch = conv_channel(lch, hk, hv);
    u16 xs[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) xs[t] = p.x[(row0 + t) * p.x_stride + gch];
    if (s_x) {
      float seq[TT + 3];
      seq[0] = bf2f(cs[i][0]);
      seq[1] = bf2f(cs[i][1]);
      seq[2] = bf2f(cs[i][2]);
#pragma unroll
      for (int t = 0; t < TT; ++t) seq[t + 3] = bf2f(xs[t]);
      const float w0 = bf2f(cw[i][0]), w1 = bf2f(cw[i][1]), w2 = bf2f(cw[i][2]), w3 = bf2f(cw[i][3]);
#pragma unroll
      for (int t = 0; t < TT; ++t) {
        float acc = 0.f;
        acc += bfround(seq[t] * w0);
        acc += bfround(seq[t + 1] * w1);
        acc += bfround(seq[t + 2] * w2);
        acc += bfround(seq[t + 3] * w3);
        acc = div_full(acc, 1.f + tl_exp(-acc));
        s_x[t][lch] = f2bf(acc);
      }
    }
    if (lch >= 2 * KD || hv % GRP == 0) {
      u16* wr = p.conv_win + wslot * p.win_slot + gch * p.win_ch;
      wr[0] = cs[i][1];
      wr[1] = cs[i][2];
#pragma unroll
      for (int t = 0; t < TT; ++t) wr[2 + t] = xs[t];
    }
  }
}
#endif

// MINB 3: three CTAs an SM (168 registers, no spill), for grids past two waves of two an SM
extern "C" __global__ void __launch_bounds__(NW * 32, MINB) gdn_core(const Params p) {
  constexpr int GRP = HV / HK;
  const int cta = blockIdx.x;
  const int n = cta / HV;
  const int hv = cta - n * HV;
  const int hk = hv / GRP;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int r = lane >> 2;
  const int q = lane & 3;
  const int vb = warp * 16 * MT;
  const int t0 = 2 * q;
  const int t1 = 2 * q + 1;
  const long long row0 = (long long)n * TT;

  __shared__ float s_A[8][8];    // A[s][t] = beta_s e^{G_s - G_t} k_s.k_t, s > t
  __shared__ float s_F[8][8];    // F[s][t] = e^{G_t - G_s} k_s.q_t, s <= t
  __shared__ u32 s_sc[2][2][32]; // history scores (q / k, history tile) as BF16 B fragments
  __shared__ float s_norm[NW][8];

#if TRIGGER == 2
  pdl_launch();
#endif
  // ---------------- before the wait: earlier-step and static data ----------------
  const long long slot = load_index(p.state_indices, n, p.state_index64 != 0);
  bool live = slot >= 0;
#if CONV
  // the conv weights and the old conv state of this thread's channels (earlier-step / static data)
  u16 cw[CPT][4], cs[CPT][3];
  long long wslot = 0;
  if (live) {
    wslot = load_index(p.inter_indices, n, p.inter_index64 != 0);
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
      const int lch = tid + i * NW * 32;
      if (lch < CCH) {
        const long long gch = conv_channel(lch, hk, hv);
#pragma unroll
        for (int j = 0; j < 4; ++j) cw[i][j] = p.conv_w[gch * p.cw_ch + j * p.cw_tap];
#pragma unroll
        for (int j = 0; j < 3; ++j) cs[i][j] = p.conv_state[slot * p.cs_slot + gch * p.cs_ch + j * p.cs_tap];
      }
    }
  }
  __shared__ __align__(16) u16 s_x[TT][CCH + 8];  // the step's convolved q | k | v rows of this CTA
#endif
  // the checkpoint rows first: they depend on the state index only
  const u16* sp = p.state + (live ? slot : 0) * p.state_slot + (long long)hv * VD * KD;
  U4 H[MT][2][4];  // checkpoint rows v = vb + 16 m + 8 h + r: [m-tile][half][chunk]
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int c = 0; c < 4; ++c)
        H[m][h][c] = live ? ld16(sp + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q) : zero4();
  long long rid = 0;
  int wp = 0, base = 0;
  if (live) {
    rid = load_index(p.replay_indices, n, p.replay_index64 != 0);
    wp = p.write_pos[rid];
    base = p.cache_base[rid];
    live = p.is_flush[rid] == 0;
  }
  if (!live) {
    // padding (or a flush row, which the verify never sees): zeros for out_proj
    pdl_wait();
#if CONV
    if (slot >= 0) conv_stage(p, row0, hk, hv, tid, wslot, cw, cs, nullptr);
#endif
    for (int t = 0; t < TT && tid < VD; ++t) {
      p.y[(row0 + t) * p.y_stride + hv * VD + tid] = 0;
#if STORE_O
      p.o_out[((row0 + t) * HV + hv) * VD + tid] = 0;
#endif
    }
    return;
  }
  const int hist = wp;
  const float negAexp = -tl_exp(load_param(p.A_log, hv, p.alog_bf16 != 0));
  const float dtb = load_param(p.dt_bias, hv, p.dtb_bf16 != 0);
  float nw[MT][2];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int v = vb + 16 * m + 8 * h + r;
#if NORM_F32
      nw[m][h] = reinterpret_cast<const float*>(p.norm_w)[v];
#else
      nw[m][h] = bf2f(reinterpret_cast<const u16*>(p.norm_w)[v]);
#endif
    }
  const float rscale = div_full(1.f, p.scale[0]);

  // ---------------- the wait: the step's activations from here on ----------------
  pdl_wait();

  const bool rowok = r < TT;
  const bool ok0 = t0 < TT, ok1 = t1 < TT;
#if CONV
  // z first, then the conv of this CTA's channels into s_x
  u16 zr[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const u16* zv = p.z + hv * VD + vb + 16 * m + 8 * h + r;
      zr[m][2 * h] = ok0 ? zv[(row0 + t0) * p.z_stride] : (u16)0;
      zr[m][2 * h + 1] = ok1 ? zv[(row0 + t1) * p.z_stride] : (u16)0;
    }
  conv_stage(p, row0, hk, hv, tid, wslot, cw, cs, s_x);
  __syncthreads();
  U4 QN[4], KR[4];  // normalized q (BF16) and convolved k rows; rows r >= TT zero
  U4 QR[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    QR[c] = rowok ? ld16(&s_x[r][32 * c + 8 * q]) : zero4();
    KR[c] = rowok ? ld16(&s_x[r][KD + 32 * c + 8 * q]) : zero4();
  }
  u16 vr[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int v = 2 * KD + vb + 16 * m + 8 * h + r;
      vr[m][2 * h] = ok0 ? s_x[t0][v] : (u16)0;
      vr[m][2 * h + 1] = ok1 ? s_x[t1][v] : (u16)0;
    }
#else
  const u16* xr = p.x + (row0 + (r < TT ? r : 0)) * p.x_stride;
  // every activation load is issued here, right after the wait
  U4 QN[4], KR[4];  // normalized q (BF16) and raw k rows; rows r >= TT zero
  U4 QR[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    QR[c] = rowok ? ld16(xr + hk * KD + 32 * c + 8 * q) : zero4();
    KR[c] = rowok ? ld16(xr + HK * KD + hk * KD + 32 * c + 8 * q) : zero4();
  }
  // v / z of this lane's (v, t): [m][(h, t)] = (h0, t0), (h0, t1), (h1, t0), (h1, t1)
  u16 vr[MT][4], zr[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int v = vb + 16 * m + 8 * h + r;
      const u16* xv = p.x + 2 * HK * KD + hv * VD + v;
      const u16* zv = p.z + hv * VD + v;
      vr[m][2 * h] = ok0 ? xv[(row0 + t0) * p.x_stride] : (u16)0;
      vr[m][2 * h + 1] = ok1 ? xv[(row0 + t1) * p.x_stride] : (u16)0;
      zr[m][2 * h] = ok0 ? zv[(row0 + t0) * p.z_stride] : (u16)0;
      zr[m][2 * h + 1] = ok1 ? zv[(row0 + t1) * p.z_stride] : (u16)0;
    }
#endif
  u16 ar = 0, br_ = 0;
  if (lane < TT) {
    ar = p.a[(row0 + lane) * p.ab_stride + hv];
    br_ = p.b[(row0 + lane) * p.ab_stride + hv];
  }
  float kmul;       // 1 / |k_r|
  {
    // |q|^2, |k|^2 in the Triton verify's order (bitwise its normalized q / k): per 64-wide half, 8-element blocks
    // summed in order, then combined by lane xor 4 / 2 / 1; block 4 c + q is chunk c of lane q, so xor 4 is local
    float sqs = 0.f, sks = 0.f;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      float hq[2], hk[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const U4 wq = QR[2 * hf + j], wk = KR[2 * hf + j];
        float aq = bflo(wq.x) * bflo(wq.x), ak = bflo(wk.x) * bflo(wk.x);
        aq += bfhi(wq.x) * bfhi(wq.x);
        ak += bfhi(wk.x) * bfhi(wk.x);
#pragma unroll
        for (int i = 1; i < 4; ++i) {
          const u32 xq = word(wq, i), xk = word(wk, i);
          aq += bflo(xq) * bflo(xq);
          ak += bflo(xk) * bflo(xk);
          aq += bfhi(xq) * bfhi(xq);
          ak += bfhi(xk) * bfhi(xk);
        }
        hq[j] = aq;
        hk[j] = ak;
      }
      float sq = hq[0] + hq[1], sk = hk[0] + hk[1];
      sq += __shfl_xor_sync(FULL, sq, 2);
      sk += __shfl_xor_sync(FULL, sk, 2);
      sq += __shfl_xor_sync(FULL, sq, 1);
      sk += __shfl_xor_sync(FULL, sk, 1);
      sqs += sq;
      sks += sk;
    }
    const float qmul = rowok ? div_full(1.f, sqrt_approx(sqs + 1e-6f)) * p.qscale : 0.f;
    kmul = rowok ? div_full(1.f, sqrt_approx(sks + 1e-6f)) : 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      QN[c].x = pack2(bflo(QR[c].x) * qmul, bfhi(QR[c].x) * qmul);
      QN[c].y = pack2(bflo(QR[c].y) * qmul, bfhi(QR[c].y) * qmul);
      QN[c].z = pack2(bflo(QR[c].z) * qmul, bfhi(QR[c].z) * qmul);
      QN[c].w = pack2(bflo(QR[c].w) * qmul, bfhi(QR[c].w) * qmul);
    }
  }
  const float km0 = __shfl_sync(FULL, kmul, 4 * t0), km1 = __shfl_sync(FULL, kmul, 4 * t1);
  // gates of token = lane, cumulative log-decay by an 8-lane scan
  float gl = 0.f, beta = 0.f;
  if (lane < TT) {
    const float xg = bf2f(ar) + dtb;
    const float spl = xg <= 20.f ? logf(1.f + tl_exp(xg)) : xg;  // tl.log: the precise logf
    gl = negAexp * spl;
    beta = div_full(1.f, 1.f + tl_exp(-bf2f(br_)));
  }
  // the norm's gate z * sigmoid(z), here so that z arrives while the products run
  float zg[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float zv = bf2f(zr[m][i]);
      zg[m][i] = zv * div_full(1.f, 1.f + tl_exp(-zv));
    }
  float Gv = gl;
#pragma unroll
  for (int d = 1; d < 8; d <<= 1) {
    const float o = __shfl_up_sync(FULL, Gv, d);
    if (lane >= d) Gv += o;
  }
  const float Gt0 = __shfl_sync(FULL, Gv, t0), Gt1 = __shfl_sync(FULL, Gv, t1);
  const float bt0 = __shfl_sync(FULL, beta, t0), bt1 = __shfl_sync(FULL, beta, t1);
  const float Gr = __shfl_sync(FULL, Gv, r), br = __shfl_sync(FULL, beta, r);
  const float eG0 = tl_exp(Gt0), eG1 = tl_exp(Gt1);

  // checkpoint products (all warps): rows v, columns = tokens; the key product on raw rows, scaled by 1/|k_t|
  float HQ[MT][4], HKs[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int i = 0; i < 4; ++i) HQ[m][i] = HKs[m][i] = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const u32 a0 = word(H[m][0][c], 2 * e), a1 = word(H[m][1][c], 2 * e);
        const u32 a2 = word(H[m][0][c], 2 * e + 1), a3 = word(H[m][1][c], 2 * e + 1);
        mma16816(HKs[m], a0, a1, a2, a3, word(KR[c], 2 * e), word(KR[c], 2 * e + 1));
        mma16816(HQ[m], a0, a1, a2, a3, word(QN[c], 2 * e), word(QN[c], 2 * e + 1));
      }
    HKs[m][0] *= km0;
    HKs[m][1] *= km1;
    HKs[m][2] *= km0;
    HKs[m][3] *= km1;
  }
  // this step's ring entries at (base + wp + t) mod RING, t < TT, wp + t < RING: keys (warp 2) and
  // log-decays (warp 3) now, while warps 0 / 1 finish the head-common products; updates at the end
  if (warp == 3 && lane < TT && wp + lane < RING)
    p.ring_g[rid * p.g_slot + hv * RING + ((base + wp + lane) & (RING - 1))] = gl;
  if (warp == 2 && hv % GRP == 0 && r < TT && wp + r < RING) {
    const long long off = ((long long)hk * RING + ((base + wp + r) & (RING - 1))) * KD;
    u16* kp = p.ring_k + rid * p.k_slot + off;
    u16* klp = p.ring_klo ? p.ring_klo + rid * p.klo_slot + off : nullptr;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      u32 kh[4], kl[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const u32 wk = word(KR[c], i);
        const float k0 = __fmul_rn(bflo(wk), kmul), k1 = __fmul_rn(bfhi(wk), kmul);
        kh[i] = pack2(k0, k1);
        // the low part from the exact product, as Triton's fused k * rnorm - high
        kl[i] = pack2(fmaf(bflo(wk), kmul, -bflo(kh[i])), fmaf(bfhi(wk), kmul, -bfhi(kh[i])));
      }
      U4 vh, vl;
      vh.x = kh[0]; vh.y = kh[1]; vh.z = kh[2]; vh.w = kh[3];
      vl.x = kl[0]; vl.y = kl[1]; vl.z = kl[2]; vl.w = kl[3];
      st16(kp + 32 * c + 8 * q, vh);
      if (klp) st16(klp + 32 * c + 8 * q, vl);
    }
  }

  // head-common products: warp 0 k k^T (A), warp 1 k q^T (F), warps 2 / 3 the history scores
  if (warp == 0) {
    float kk[4] = {0.f, 0.f, 0.f, 0.f};
    kred(kk, KR, KR);
    const float ks = kmul;  // row s = r
    s_A[r][t0] = (r > t0 && r < TT) ? br * tl_exp(Gr - Gt0) * (kk[0] * ks * km0) : 0.f;
    s_A[r][t1] = (r > t1 && r < TT) ? br * tl_exp(Gr - Gt1) * (kk[1] * ks * km1) : 0.f;
  } else if (warp == 1) {
    float kq[4] = {0.f, 0.f, 0.f, 0.f};
    kred(kq, KR, QN);
    s_F[r][t0] = (r <= t0 && t0 < TT) ? tl_exp(Gt0 - Gr) * (kq[0] * kmul) : 0.f;
    s_F[r][t1] = (r <= t1 && t1 < TT) ? tl_exp(Gt1 - Gr) * (kq[1] * kmul) : 0.f;
  } else if ((warp == 2 || warp == 3) && hist > 0) {
    // scores[j][t] = khist_j . (q_t | k_t), j = 8 jt + 2q + i (rows j >= wp zero), rounded to BF16
    U4 KHh[4];
#pragma unroll
    for (int jt = 0; jt < 2; ++jt) {
      const int j = 8 * jt + r;
      const u16* kh = p.ring_k + rid * p.k_slot + ((long long)hk * RING + ((base + j) & (RING - 1))) * KD;
#pragma unroll
      for (int c = 0; c < 4; ++c) KHh[c] = j < hist ? ld16(kh + 32 * c + 8 * q) : zero4();
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      if (warp == 2) {
        kred(acc, QN, KHh);
      } else {
        U4 KH[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          KH[c].x = pack2(bflo(KR[c].x) * kmul, bfhi(KR[c].x) * kmul);
          KH[c].y = pack2(bflo(KR[c].y) * kmul, bfhi(KR[c].y) * kmul);
          KH[c].z = pack2(bflo(KR[c].z) * kmul, bfhi(KR[c].z) * kmul);
          KH[c].w = pack2(bflo(KR[c].w) * kmul, bfhi(KR[c].w) * kmul);
        }
        kred(acc, KH, KHh);
      }
      s_sc[warp - 2][jt][lane] = pack2(acc[0], acc[1]);
    }
  }
  __syncthreads();

  if (hist > 0) {
    // committed history: hw = e^{g_total} hw + bf16(d_j e^{g_total - G_j}) . bf16(scores)
    float gj = 0.f;
    if (lane < RING && lane < hist) gj = p.ring_g[rid * p.g_slot + hv * RING + ((base + lane) & (RING - 1))];
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float o = __shfl_up_sync(FULL, gj, d);
      if (lane >= d) gj += o;
    }
    const float gtot = __shfl_sync(FULL, gj, 31);
    const int jj[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
    float dec[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float pj = __shfl_sync(FULL, gj, jj[i]);
      dec[i] = jj[i] < hist ? tl_exp(gtot - pj) : 0.f;
    }
    const float td = tl_exp(gtot);
    const u16* dp = p.ring_d + rid * p.d_slot + (long long)hv * RING * VD;
    const u32 bq0 = s_sc[0][0][lane], bq1 = s_sc[0][1][lane];
    const u32 bk0 = s_sc[1][0][lane], bk1 = s_sc[1][1][lane];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float e[2][4];  // [half][j index i]
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          e[h][i] = jj[i] < hist
                        ? bf2f(dp[((base + jj[i]) & (RING - 1)) * VD + vb + 16 * m + 8 * h + r]) * dec[i]
                        : 0.f;
      const u32 a0 = pack2(e[0][0], e[0][1]), a1 = pack2(e[1][0], e[1][1]);
      const u32 a2 = pack2(e[0][2], e[0][3]), a3 = pack2(e[1][2], e[1][3]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        HQ[m][i] *= td;
        HKs[m][i] *= td;
      }
      mma16816(HQ[m], a0, a1, a2, a3, bq0, bq1);
      mma16816(HKs[m], a0, a1, a2, a3, bk0, bk1);
    }
  }

  // rhs = beta (v - e^G hw_k); forward substitution (I + A) U = rhs per value column, every lane of a quad
  // solving the quad's four columns after an all-gather
  float U[2 * MT][8];
  {
    float rhs[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      rhs[m][0] = bt0 * (bf2f(vr[m][0]) - eG0 * HKs[m][0]);
      rhs[m][1] = bt1 * (bf2f(vr[m][1]) - eG1 * HKs[m][1]);
      rhs[m][2] = bt0 * (bf2f(vr[m][2]) - eG0 * HKs[m][2]);
      rhs[m][3] = bt1 * (bf2f(vr[m][3]) - eG1 * HKs[m][3]);
    }
    float R[2 * MT][8];
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) {
      const int src = (lane & ~3) | qq;
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          R[2 * m + h][2 * qq] = __shfl_sync(FULL, rhs[m][2 * h], src);
          R[2 * m + h][2 * qq + 1] = __shfl_sync(FULL, rhs[m][2 * h + 1], src);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float u[2 * MT] = {};
      if (i < TT) {
#pragma unroll
        for (int col = 0; col < 2 * MT; ++col) u[col] = R[col][i];
#pragma unroll
        for (int s = 0; s < i; ++s) {
          const float a = s_A[i][s];
#pragma unroll
          for (int col = 0; col < 2 * MT; ++col) u[col] -= a * U[col][s];
        }
      }
#pragma unroll
      for (int col = 0; col < 2 * MT; ++col) U[col][i] = u[col];
    }
  }
  // this step's update vectors (high, low parts) into the ring
  {
    u16* dp = p.ring_d + rid * p.d_slot + (long long)hv * RING * VD;
    u16* dlp = p.ring_dlo ? p.ring_dlo + rid * p.dlo_slot + (long long)hv * RING * VD : nullptr;
    const bool w0 = ok0 && wp + t0 < RING, w1 = ok1 && wp + t1 < RING;
    const int ph0 = (base + wp + t0) & (RING - 1), ph1 = (base + wp + t1) & (RING - 1);
#pragma unroll
    for (int col = 0; col < 2 * MT; ++col) {
      const int v = vb + 16 * (col >> 1) + 8 * (col & 1) + r;
      if (w0) {
        const float d = pick8(U[col], t0);
        const u16 hi = f2bf(d);
        dp[ph0 * VD + v] = hi;
        if (dlp) dlp[ph0 * VD + v] = f2bf(d - bf2f(hi));
      }
      if (w1) {
        const float d = pick8(U[col], t1);
        const u16 hi = f2bf(d);
        dp[ph1 * VD + v] = hi;
        if (dlp) dlp[ph1 * VD + v] = f2bf(d - bf2f(hi));
      }
    }
  }
  // output O[v, t] = e^{G_t} hw_q + sum_{s <= t} U[v, s] F[s][t]
  float O[MT][4];
  {
    float f0[8], f1[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      f0[s] = s_F[s][t0];
      f1[s] = s_F[s][t1];
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int s = 0; s < TT; ++s) {
          a0 += U[2 * m + h][s] * f0[s];
          a1 += U[2 * m + h][s] * f1[s];
        }
        O[m][2 * h] = eG0 * HQ[m][2 * h] + a0;
        O[m][2 * h + 1] = eG1 * HQ[m][2 * h + 1] + a1;
      }
  }

  // gated RMSNorm over the head's 128 columns of the BF16-rounded replay output, then static FP8
  float xb[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) xb[m][i] = bfround(O[m][i]);
#if STORE_O
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = (i & 1) ? t1 : t0;
      const int v = vb + 16 * m + 8 * (i >> 1) + r;
      if (t < TT) p.o_out[((row0 + t) * HV + hv) * VD + v] = f2bf(O[m][i]);
    }
#endif
  float p0 = 0.f, p1 = 0.f;
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    p0 += xb[m][0] * xb[m][0] + xb[m][2] * xb[m][2];
    p1 += xb[m][1] * xb[m][1] + xb[m][3] * xb[m][3];
  }
#pragma unroll
  for (int msk = 4; msk < 32; msk <<= 1) {
    p0 += __shfl_xor_sync(FULL, p0, msk);
    p1 += __shfl_xor_sync(FULL, p1, msk);
  }
  if (r == 0) {
    s_norm[warp][t0] = p0;
    s_norm[warp][t1] = p1;
  }
  __syncthreads();
  float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
  for (int w = 0; w < NW; ++w) {
    ss0 += s_norm[w][t0];
    ss1 += s_norm[w][t1];
  }
  const float rstd0 = rsqrtf(ss0 / (float)VD + p.eps);
  const float rstd1 = rsqrtf(ss1 / (float)VD + p.eps);
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = i >> 1;
      const bool ok = (i & 1) ? ok1 : ok0;
      if (ok) {
        const int t = (i & 1) ? t1 : t0;
        const int v = vb + 16 * m + 8 * h + r;
        float yv = xb[m][i] * ((i & 1) ? rstd1 : rstd0) * nw[m][h];
        yv *= zg[m][i];
        p.y[(row0 + t) * p.y_stride + hv * VD + v] = f2e4m3(bfround(yv) * rscale);
      }
    }
}
"""

K = V = 128
RING = 16
MAX_ROWS_PER_REQUEST = 8
# 4 warps of 32 value columns (warps 2 / 3 also write the key / decay ring entries, so at least 4)
_WARPS = 4


class _Params(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p), ("x_stride", ctypes.c_int64),
        ("z", ctypes.c_void_p), ("z_stride", ctypes.c_int64),
        ("a", ctypes.c_void_p), ("b", ctypes.c_void_p), ("ab_stride", ctypes.c_int64),
        ("A_log", ctypes.c_void_p), ("dt_bias", ctypes.c_void_p),
        ("state", ctypes.c_void_p), ("state_slot", ctypes.c_int64),
        ("ring_d", ctypes.c_void_p), ("d_slot", ctypes.c_int64),
        ("ring_k", ctypes.c_void_p), ("k_slot", ctypes.c_int64),
        ("ring_g", ctypes.c_void_p), ("g_slot", ctypes.c_int64),
        ("ring_dlo", ctypes.c_void_p), ("dlo_slot", ctypes.c_int64),
        ("ring_klo", ctypes.c_void_p), ("klo_slot", ctypes.c_int64),
        ("state_indices", ctypes.c_void_p), ("replay_indices", ctypes.c_void_p),
        ("state_index64", ctypes.c_int), ("replay_index64", ctypes.c_int),
        ("alog_bf16", ctypes.c_int), ("dtb_bf16", ctypes.c_int),
        ("write_pos", ctypes.c_void_p), ("cache_base", ctypes.c_void_p), ("is_flush", ctypes.c_void_p),
        ("norm_w", ctypes.c_void_p), ("scale", ctypes.c_void_p),
        ("eps", ctypes.c_float), ("qscale", ctypes.c_float),
        ("y", ctypes.c_void_p), ("y_stride", ctypes.c_int64),
        ("o_out", ctypes.c_void_p),
        ("conv_w", ctypes.c_void_p), ("cw_ch", ctypes.c_int64), ("cw_tap", ctypes.c_int64),
        ("conv_state", ctypes.c_void_p), ("cs_slot", ctypes.c_int64), ("cs_ch", ctypes.c_int64),
        ("cs_tap", ctypes.c_int64),
        ("conv_win", ctypes.c_void_p), ("win_slot", ctypes.c_int64), ("win_ch", ctypes.c_int64),
        ("inter_indices", ctypes.c_void_p), ("inter_index64", ctypes.c_int),
    ]


_LOCK = threading.Lock()
# key -> (module, function); the modules stay referenced because captured graphs launch their functions
_KERNELS: dict = {}
_LOGGED: set = set()


def _check(res, what):
    err = res[0]
    if int(err) != 0:
        raise RuntimeError(f"gdn_core_cuda: {what}: {err!r}")
    return res[1] if len(res) == 2 else res[1:]


def _compile(key: tuple) -> tuple:
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    device, rows, hk, hv, norm_f32, trigger, store_o, warps, conv, min_blocks = key
    defines = dict(
        TT=rows, HK=hk, HV=hv, NORM_F32=int(norm_f32), TRIGGER=trigger, STORE_O=int(store_o), NW=warps,
        MINB=min_blocks, CONV=int(conv),
    )
    src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + _SRC
    prog = _check(nvrtc.nvrtcCreateProgram(src.encode(), b"gdn_core.cu", 0, [], []), "create")
    major, minor = torch.cuda.get_device_capability(device)
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17", b"-default-device"]
    try:
        if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("gdn_core_cuda: NVRTC failed:\n" + log.decode(errors="replace"))
        cubin = b" " * _check(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
        _check(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    with torch.cuda.device(device):
        module = _check(cu.cuModuleLoadData(cubin), "module")
        fn = _check(cu.cuModuleGetFunction(module, b"gdn_core"), "function")
    return module, fn


@functools.cache
def _sms(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _min_blocks(device: int, ctas: int) -> int:
    # Three CTAs an SM (168 registers) once two an SM take more than two waves; below that the uncapped registers win.
    return 3 if ctas > 4 * _sms(device) else 1


def _key(
    *, qkv, checkpoint, norm_weight, rows_per_request, num_k_heads, min_blocks, o_out=None, warps=_WARPS,
    conv_weight=None, **_
) -> tuple:
    return (
        qkv.device.index, rows_per_request, num_k_heads, checkpoint.shape[1],
        norm_weight.dtype == torch.float32, 0, o_out is not None, warps, conv_weight is not None, min_blocks,
    )


def _ensure(key: tuple) -> None:
    with _LOCK:
        if key not in _KERNELS:
            _KERNELS[key] = _compile(key)


def _ring_ok(t: Optional[torch.Tensor], heads: int) -> bool:
    return t is None or (
        t.dtype == torch.bfloat16
        and tuple(t.shape[1:]) == (heads, RING, K)
        and t.stride()[1:] == (RING * K, K, 1)
    )


def _unsupported(
    *, qkv, z, a, b, A_log, dt_bias, checkpoint, ring_d, ring_k, ring_g, ring_dlo, ring_klo, state_indices,
    replay_indices, write_pos, cache_base, is_flush, norm_weight, input_scale, out, rows_per_request, num_k_heads,
    **_,
) -> Optional[str]:
    """Why the core cannot run these tensors, or None when it can."""
    rows, T, H = qkv.shape[0], rows_per_request, num_k_heads
    HV = checkpoint.shape[1] if checkpoint.dim() == 4 else 0
    if not (qkv.is_cuda and HV and HV % H == 0):
        return "heads"
    if (
        qkv.dtype != torch.bfloat16
        or qkv.stride(1) != 1
        or qkv.stride(0) % 8
        or qkv.data_ptr() % 16
        or qkv.shape[1] < (2 * H + HV) * K
    ):
        return "q | k | v rows"
    if (
        not 2 <= T <= MAX_ROWS_PER_REQUEST
        or rows % T
        or state_indices.numel() < rows // T
        or replay_indices.numel() < rows // T
    ):
        return "rows a request"
    if (
        checkpoint.dtype != torch.bfloat16
        or tuple(checkpoint.shape[2:]) != (V, K)
        or checkpoint.stride()[1:] != (V * K, K, 1)
        or checkpoint.stride(0) % 8
        or checkpoint.data_ptr() % 16
    ):
        return "checkpoint"
    if not (_ring_ok(ring_d, HV) and ring_d is not None and _ring_ok(ring_dlo, HV)):
        return "update ring"
    # keys move as 16-byte rows
    if not (_ring_ok(ring_k, H) and ring_k is not None and _ring_ok(ring_klo, H)) or any(
        t is not None and (t.stride(0) % 8 or t.data_ptr() % 16) for t in (ring_k, ring_klo)
    ):
        return "key ring"
    if ring_g.dtype != torch.float32 or tuple(ring_g.shape[1:]) != (HV, RING) or ring_g.stride()[1:] != (RING, 1):
        return "decay ring"
    if z.dtype != torch.bfloat16 or tuple(z.shape) != (rows, HV, V) or z.stride(2) != 1 or z.stride(1) != V:
        return "z"
    if (
        a.dtype != torch.bfloat16
        or b.dtype != torch.bfloat16
        or a.shape[0] < rows
        or a.shape[-1] < HV
        or a.stride(-1) != 1
        or b.stride(-1) != 1
        or a.stride(0) != b.stride(0)
    ):
        return "a / b"
    if (
        out.dtype != torch.float8_e4m3fn
        or tuple(out.shape) != (rows, HV * V)
        or out.stride(1) != 1
        or out.stride(0) % 16
        or out.data_ptr() % 16
    ):
        return "out"
    if write_pos.dtype != torch.int32 or cache_base.dtype != torch.int32 or is_flush.dtype != torch.int8:
        return "cursor dtypes"
    for t in (write_pos, cache_base, is_flush, state_indices, replay_indices):
        if t.dim() != 1 or t.stride(0) != 1:
            return "cursor / index layout"
    if state_indices.dtype not in (torch.int32, torch.int64) or replay_indices.dtype not in (torch.int32, torch.int64):
        return "index dtypes"
    for t in (A_log, dt_bias):
        if t.dtype not in (torch.float32, torch.bfloat16) or t.numel() < HV or not t.is_contiguous():
            return "A_log / dt_bias"
    if norm_weight.dtype not in (torch.float32, torch.bfloat16) or tuple(norm_weight.shape) != (V,):
        return "norm weight"
    if not norm_weight.is_contiguous() or input_scale.dtype != torch.float32 or input_scale.numel() != 1:
        return "norm weight / FP8 scale"
    conv_weight = _.get("conv_weight")
    if conv_weight is not None:
        cs, win, inter = _.get("conv_state"), _.get("conv_windows"), _.get("inter_indices")
        C = (2 * H + HV) * K
        st = win.stride() if win is not None else ()
        if (
            conv_weight.dtype != torch.bfloat16
            or conv_weight.dim() != 2
            or conv_weight.shape[1] != 4
            or conv_weight.shape[0] < C
            or cs is None
            or cs.dtype != torch.bfloat16
            or tuple(cs.shape[1:]) != (conv_weight.shape[0], 3)
            or win is None
            or win.dtype != torch.bfloat16
            or win.dim() != 4
            or not (st[1] == 1 and st[3] == 1 and st[2] >= T + 2)
            or inter is None
            or inter.dtype not in (torch.int32, torch.int64)
            or inter.numel() < rows // T
        ):
            return "conv operands (width-4 BF16 conv, deduplicated windows)"
    return None


def try_core(**kw) -> bool:
    """``core(**kw)`` when the tensors fit and its kernel is compiled or compilable now; else False."""
    reason = _unsupported(**kw)
    if reason is not None:
        _log_once(("shape", reason), f"SGLANG_PERFHT_GDN_CORE: unsupported {reason}, Triton verify + gated norm")
        return False
    family = _key(**{**kw, "min_blocks": 0})
    keys = [family[:-1] + (mb,) for mb in (1, 3)]  # conv flag stays in the family
    if any(k not in _KERNELS for k in keys):
        if torch.cuda.is_current_stream_capturing():
            _log_once("capture", "SGLANG_PERFHT_GDN_CORE: not compiled before graph capture, Triton verify + norm")
            return False
        for k in keys:
            _ensure(k)
    core(**kw)
    _log_once("on", "SGLANG_PERFHT_GDN_CORE: CUDA verify core + gated norm FP8 runs")
    return True


def _log_once(key, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg)


def _ptr(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else t.data_ptr()


def _slot_stride(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else t.stride(0)


def core(
    *,
    qkv: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint: torch.Tensor,
    ring_d: torch.Tensor,
    ring_k: torch.Tensor,
    ring_g: torch.Tensor,
    ring_dlo: Optional[torch.Tensor],
    ring_klo: Optional[torch.Tensor],
    state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    norm_weight: torch.Tensor,
    input_scale: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    rows_per_request: int,
    num_k_heads: int,
    o_out: Optional[torch.Tensor] = None,
    warps: int = _WARPS,
    min_blocks: Optional[int] = None,
    conv_weight: Optional[torch.Tensor] = None,
    conv_state: Optional[torch.Tensor] = None,
    conv_windows: Optional[torch.Tensor] = None,
    inter_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Write the gated RMSNorm of the verify output as static FP8 into ``out`` [rows, HV * V].

    ``qkv`` [rows, >= (2 H + HV) * 128] BF16 convolved q | k | v rows; ``z`` [rows, HV, V]; ``a`` / ``b`` [rows, HV]
    views with one row stride; ``checkpoint`` [slots, HV, V, K] BF16 keyed by ``state_indices`` (negative: padding,
    zero rows out); the ring [slots, HV | H, 16, V | K] (low parts optional) and its cursors keyed by
    ``replay_indices``; ``rows_per_request`` rows per request. ``o_out`` [rows, HV, V] BF16 also receives the replay
    output; ``warps`` (>= 4) and ``min_blocks`` override the launch (checks and timing only). Callers go through
    ``try_core`` (shape checks, compile before capture)."""
    from cuda.bindings import driver as cu

    rows, T = qkv.shape[0], rows_per_request
    n, HV = rows // T, checkpoint.shape[1]
    if n == 0:
        return out
    device = qkv.device.index
    if min_blocks is None:
        min_blocks = _min_blocks(device, n * HV)
    key = _key(
        qkv=qkv, checkpoint=checkpoint, norm_weight=norm_weight, rows_per_request=T, num_k_heads=num_k_heads,
        min_blocks=min_blocks, o_out=o_out, warps=warps, conv_weight=conv_weight,
    )
    conv = conv_weight is not None
    if key not in _KERNELS:
        _ensure(key)
    p = _Params(
        qkv.data_ptr(), qkv.stride(0),
        z.data_ptr(), z.stride(0),
        a.data_ptr(), b.data_ptr(), a.stride(0),
        A_log.data_ptr(), dt_bias.data_ptr(),
        checkpoint.data_ptr(), checkpoint.stride(0),
        ring_d.data_ptr(), ring_d.stride(0),
        ring_k.data_ptr(), ring_k.stride(0),
        ring_g.data_ptr(), ring_g.stride(0),
        _ptr(ring_dlo), _slot_stride(ring_dlo),
        _ptr(ring_klo), _slot_stride(ring_klo),
        state_indices.data_ptr(), replay_indices.data_ptr(),
        int(state_indices.dtype == torch.int64), int(replay_indices.dtype == torch.int64),
        int(A_log.dtype == torch.bfloat16), int(dt_bias.dtype == torch.bfloat16),
        write_pos.data_ptr(), cache_base.data_ptr(), is_flush.data_ptr(),
        norm_weight.data_ptr(), input_scale.data_ptr(),
        float(eps), K**-0.5,
        out.data_ptr(), out.stride(0),
        _ptr(o_out),
        _ptr(conv_weight), conv_weight.stride(0) if conv else 0, conv_weight.stride(1) if conv else 0,
        _ptr(conv_state), conv_state.stride(0) if conv else 0, conv_state.stride(1) if conv else 0,
        conv_state.stride(2) if conv else 0,
        _ptr(conv_windows), conv_windows.stride(0) if conv else 0, conv_windows.stride(2) if conv else 0,
        _ptr(inter_indices), int(conv and inter_indices.dtype == torch.int64),
    )
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = n * HV, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 32 * warps, 1, 1
    cfg.sharedMemBytes = 0
    cfg.hStream = cu.CUstream(torch.cuda.current_stream(qkv.device).cuda_stream)
    attr = cu.CUlaunchAttribute()
    attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    attr.value.programmaticStreamSerializationAllowed = 1
    cfg.attrs, cfg.numAttrs = [attr], 1
    args = (ctypes.c_void_p * 1)(ctypes.addressof(p))
    _check(cu.cuLaunchKernelEx(cfg, _KERNELS[key][1], ctypes.addressof(args), 0), "launch")
    return out
