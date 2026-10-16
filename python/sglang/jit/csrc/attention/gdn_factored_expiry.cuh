// docs/222 §11 R3-a: lean slot-expiry truncation for the factored GDN state (RMAX = 32, K = V = 128, fp16 factors).
// Same mathematics as gdn_factored._factored_expiry_truncate_kernel (MGS route): G = W W^T over the first RFULL rows;
// warm start Z0 = the R coordinate directions with the largest diag(G) (ties by index); ITERS rounds of Z <- MGS2(G Z)
// with the rank tolerance (pass 0 drops a column whose explicit residual norm is <= REL_TOL x its pre-MGS norm, or
// <= 1e-12); U[:R] = Z^T U, W[:R] = Z^T W (fp16, round to nearest), count = R.  Rounding-level differences only
// (summation order, IEEE sqrt / division).
//
// One CTA of 128 threads per (slot, head); grid (B * HV, layers) with per-layer element strides.
//   load W, U -> shared (barrier 1) | Gram, 8 outputs per thread, fp32 FMA (barrier 2) |
//   warp 0: lane = row, ITERS x (G Z + MGS2 with warp-shuffle reductions, no CTA barrier) (barrier 3) |
//   projection of the R kept rows, 2 x 16 outputs per thread, fp16 stores.
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

namespace gdn_expiry {

constexpr int RMAX = 32;
constexpr int KD = 128;   // K = V
constexpr int PAD = 8;    // halves of row padding (16 B): conflict-free 128-bit row reads
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}

template <int R, int ITERS>
__global__ void __launch_bounds__(128, 4) expiry_trunc_r32_kernel(
    __half* __restrict__ U, __half* __restrict__ W, int32_t* __restrict__ cnt, const void* __restrict__ indices,
    int idx_is64, int64_t stride_idx, int HV, int rfull, float rel_tol, int64_t s_u, int64_t s_w, int64_t s_c) {
  static_assert(R >= 1 && R <= 16, "R <= 16 (one register column per kept direction)");
  __shared__ __align__(16) __half sW[RMAX][KD + PAD];
  __shared__ __align__(16) __half sU[RMAX][KD + PAD];
  __shared__ float sG[RMAX][RMAX + 1];
  __shared__ __align__(16) float sZ[RMAX][16];

  const int pid = blockIdx.x;
  const int64_t layer = blockIdx.y;
  U += layer * s_u;
  W += layer * s_w;
  cnt += layer * s_c;
  const int i_n = pid / HV, i_hv = pid % HV;
  const int64_t sidx = idx_is64 ? reinterpret_cast<const int64_t*>(indices)[i_n * stride_idx]
                                : (int64_t) reinterpret_cast<const int32_t*>(indices)[i_n * stride_idx];
  if (sidx < 0) return;
  int32_t* pc = cnt + sidx * HV + i_hv;
  if (*pc < rfull) return;  // (>= as in the Triton kernel: an overshot slot is still cut)
  const int64_t base = (sidx * HV + i_hv) * (int64_t)(RMAX * KD);
  __half* Ut = U + base;
  __half* Wt = W + base;
  const int t = threadIdx.x;

  // ---- 1. load the first RFULL rows of W and U (rows >= RFULL are zero, as the masked Triton loads)
  for (int q = t; q < RMAX * KD / 8; q += 128) {
    const int r = q >> 4, c = (q & 15) * 8;
    uint4 vw = make_uint4(0, 0, 0, 0), vu = make_uint4(0, 0, 0, 0);
    if (r < rfull) {
      vw = *reinterpret_cast<const uint4*>(Wt + r * KD + c);
      vu = *reinterpret_cast<const uint4*>(Ut + r * KD + c);
    }
    *reinterpret_cast<uint4*>(&sW[r][c]) = vw;
    *reinterpret_cast<uint4*>(&sU[r][c]) = vu;
  }
  __syncthreads();

  // ---- 2. Gram: thread t -> row i = t / 4, columns j = (t % 4) * 8 + 0..7; k ascending, fp32 FMA
  {
    const int i = t >> 2, j0 = (t & 3) * 8;
    float acc[8];
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) acc[jj] = 0.f;
#pragma unroll 2
    for (int k = 0; k < KD; k += 8) {
      const uint4 a4 = *reinterpret_cast<const uint4*>(&sW[i][k]);
      const __half2* a = reinterpret_cast<const __half2*>(&a4);
      float af[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __half22float2(a[e]);
        af[2 * e] = f.x;
        af[2 * e + 1] = f.y;
      }
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const uint4 b4 = *reinterpret_cast<const uint4*>(&sW[j0 + jj][k]);
        const __half2* b = reinterpret_cast<const __half2*>(&b4);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __half22float2(b[e]);
          acc[jj] = __fmaf_rn(af[2 * e], f.x, acc[jj]);
          acc[jj] = __fmaf_rn(af[2 * e + 1], f.y, acc[jj]);
        }
      }
    }
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) sG[i][j0 + jj] = acc[jj];
  }
  __syncthreads();

  // ---- 3. subspace iteration in warp 0 (lane = row); columns >= R of Z are identically zero and not stored
  if (t < 32) {
    const int i = t;
    const float di = i < rfull ? sG[i][i] : -1.f;
    int rank = 0;
#pragma unroll
    for (int j = 0; j < RMAX; ++j) {
      const float dj = __shfl_sync(FULL, di, j);
      rank += (dj > di) || (dj == di && j < i);
    }
    float q[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) q[c] = (c < R && rank == c && i < rfull) ? 1.f : 0.f;
#pragma unroll 1
    for (int it = 0; it < ITERS; ++it) {
      // Y = G Z (k ascending)
#pragma unroll
      for (int c = 0; c < 16; c += 4) *reinterpret_cast<float4*>(&sZ[i][c]) = make_float4(q[c], q[c + 1], q[c + 2], q[c + 3]);
      __syncwarp();
      float y[16];
#pragma unroll
      for (int c = 0; c < 16; ++c) y[c] = 0.f;
#pragma unroll 4
      for (int k = 0; k < RMAX; ++k) {
        const float g = sG[i][k];
#pragma unroll
        for (int c = 0; c < R; c += 4) {
          const float4 z = *reinterpret_cast<const float4*>(&sZ[k][c]);
          y[c] = __fmaf_rn(g, z.x, y[c]);
          if (c + 1 < R) y[c + 1] = __fmaf_rn(g, z.y, y[c + 1]);
          if (c + 2 < R) y[c + 2] = __fmaf_rn(g, z.z, y[c + 2]);
          if (c + 3 < R) y[c + 3] = __fmaf_rn(g, z.w, y[c + 3]);
        }
      }
      __syncwarp();
      // MGS2 (the Triton `_mgs`: per pass and column, projection coefficients on the previous columns in one go,
      // subtract their combination, explicit residual norm; pass 0 applies the rank tolerance)
      float n0[16];
#pragma unroll
      for (int c = 0; c < R; ++c) n0[c] = sqrtf(warp_sum(y[c] * y[c]));
#pragma unroll
      for (int c = 0; c < 16; ++c) q[c] = y[c];
#pragma unroll 1
      for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
          const float yj = q[j];
          float s[16];
#pragma unroll
          for (int c = 0; c < j; ++c) s[c] = q[c] * yj;
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (int c = 0; c < j; ++c) s[c] += __shfl_xor_sync(FULL, s[c], o);
          }
          float acc = 0.f;
#pragma unroll
          for (int c = 0; c < j; ++c) acc = __fmaf_rn(q[c], s[c], acc);
          const float v = yj - acc;
          const float n = sqrtf(warp_sum(v * v));
          bool ok = n > 1e-12f;
          if (p == 0) ok = ok && (n > rel_tol * n0[j]);
          q[j] = ok ? __fdiv_rn(v, fmaxf(n, 1e-30f)) : 0.f;
        }
      }
    }
#pragma unroll
    for (int c = 0; c < 16; c += 4) *reinterpret_cast<float4*>(&sZ[i][c]) = make_float4(q[c], q[c + 1], q[c + 2], q[c + 3]);
  }
  __syncthreads();

  // ---- 4. projection of the R kept rows: thread t -> column t of U and of W; r ascending
  {
    float au[16], aw[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) au[c] = aw[c] = 0.f;
#pragma unroll 4
    for (int r = 0; r < RMAX; ++r) {
      const float u = __half2float(sU[r][t]);
      const float w = __half2float(sW[r][t]);
#pragma unroll
      for (int c = 0; c < R; c += 4) {
        const float4 z = *reinterpret_cast<const float4*>(&sZ[r][c]);
        const float zz[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          if (c + e < R) {
            au[c + e] = __fmaf_rn(zz[e], u, au[c + e]);
            aw[c + e] = __fmaf_rn(zz[e], w, aw[c + e]);
          }
        }
      }
    }
#pragma unroll
    for (int c = 0; c < R; ++c) {
      Ut[c * KD + t] = __float2half_rn(au[c]);
      Wt[c * KD + t] = __float2half_rn(aw[c]);
    }
    if (t == 0) *pc = R;
  }
}

}  // namespace gdn_expiry
