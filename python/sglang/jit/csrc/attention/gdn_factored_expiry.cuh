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

// TAG > 0: the same code as a distinct function, so per-function launch attributes (shared-memory carveout, opt-in
// dynamic shared memory) of the docs/222 §11.6 not-due-footprint variants never touch TAG 0 (variant 0).
template <int R, int ITERS, int TAG = 0>
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


// ======================================================================================================================
// docs/222 §11.3 R3-a tuning (one iteration, lead 10-16 02:5x): the MGS of the served Triton kernel reproduced
// operation by operation, so counts and kept / dropped directions match and the result can be bitwise equal.
// Triton's MGS tensors use #blocked{sizePerThread [2,2], threadsPerWarp [2,16], warpsPerCTA [4,1]} (cached TTGIR of
// the served build): thread (w, tr, tc) holds rows {a, a+1, a+16, a+17} (a = 4w + 2tr) x columns {2tc, 2tc+1}.
//   axis-0 sum (over rows): within-thread chain over its 4 rows (first term a rounded product, later terms fused into it
//     when FUSE), + shfl over tr, + across warps (XW 0: butterfly (A0+A2)+(A1+A3); XW 1: sequential ((A0+A1)+A2)+A3)
//   axis-1 sum (over the 32 columns): within-thread pair (2tc, 2tc+1), then lane butterfly over tc with offsets 8,4,2,1
//   sqrt = sqrt.approx.ftz.f32, division = div.full.f32 (the served PTX)
// Here warp 0 lane L holds Triton's row group g = L & 7 (w = g >> 1, tr = g & 1; rows 2g, 2g+1, 2g+16, 2g+17), all 16
// kept columns; the 4 lanes with the same g (L >> 3 = 0..3) compute identically (the G Z product is split over them).
// ROWORD picks the within-thread row order: 0 = (a, a+1, a+16, a+17), 1 = (a, a+16, a+1, a+17).
__device__ __forceinline__ float sqrt_approx_ftz(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

template <int ROWORD, int FUSE, int XW>
__device__ __forceinline__ float col_sum(const float (&a)[4], const float (&b)[4], int lane) {
  constexpr int o1 = ROWORD ? 2 : 1, o2 = ROWORD ? 1 : 2;  // local rows: 0 = a, 1 = a+1, 2 = a+16, 3 = a+17
  float acc = __fmul_rn(a[0], b[0]);
  if (FUSE) {
    acc = __fmaf_rn(a[o1], b[o1], acc);
    acc = __fmaf_rn(a[o2], b[o2], acc);
    acc = __fmaf_rn(a[3], b[3], acc);
  } else {
    acc = __fadd_rn(acc, __fmul_rn(a[o1], b[o1]));
    acc = __fadd_rn(acc, __fmul_rn(a[o2], b[o2]));
    acc = __fadd_rn(acc, __fmul_rn(a[3], b[3]));
  }
  acc = __fadd_rn(acc, __shfl_xor_sync(FULL, acc, 1));  // tr
  if (XW == 0) {
    acc = __fadd_rn(acc, __shfl_xor_sync(FULL, acc, 4));  // w ^ 2
    acc = __fadd_rn(acc, __shfl_xor_sync(FULL, acc, 2));  // w ^ 1
  } else {
    const int base = lane & ~7;
    const float a0 = __shfl_sync(FULL, acc, base + 0), a1 = __shfl_sync(FULL, acc, base + 2);
    const float a2 = __shfl_sync(FULL, acc, base + 4), a3 = __shfl_sync(FULL, acc, base + 6);
    acc = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
  }
  return acc;
}

template <int R, int ITERS, int ROWORD, int FUSE, int XW>
__global__ void __launch_bounds__(128, 4) expiry_trunc_r32_exact_kernel(
    __half* __restrict__ U, __half* __restrict__ W, int32_t* __restrict__ cnt, const void* __restrict__ indices,
    int idx_is64, int64_t stride_idx, int HV, int rfull, float rel_tol, int64_t s_u, int64_t s_w, int64_t s_c) {
  static_assert(R == 16, "exact variant: R = 16");
  __shared__ __align__(16) __half sW[RMAX][KD + PAD];
  __shared__ __align__(16) __half sU[RMAX][KD + PAD];
  __shared__ float sG[RMAX][RMAX + 1];
  __shared__ __align__(16) float sZ[RMAX][16];
  __shared__ __align__(16) float sY[RMAX][16];
  __shared__ float sN0[16];

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
  if (*pc < rfull) return;
  const int64_t base = (sidx * HV + i_hv) * (int64_t)(RMAX * KD);
  __half* Ut = U + base;
  __half* Wt = W + base;
  const int t = threadIdx.x;

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

  {  // Gram (tl.dot FMA: k ascending from a zero accumulator)
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

  if (t < 32) {
    const int lane = t, g = lane & 7, h = lane >> 3;
    int rows[4] = {2 * g, 2 * g + 1, 2 * g + 16, 2 * g + 17};
    float q[4][16];
    // warm start: Z0[i, rank_i] = 1 for rank_i < R (ties by index), rows < RFULL
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = rows[e];
      const float di = i < rfull ? sG[i][i] : -1.f;
      int rank = 0;
      for (int j = 0; j < RMAX; ++j) {
        const float dj = j < rfull ? sG[j][j] : -1.f;
        rank += (dj > di) || (dj == di && j < i);
      }
#pragma unroll
      for (int c = 0; c < 16; ++c) q[e][c] = (rank == c && i < rfull) ? 1.f : 0.f;
    }
#pragma unroll 1
    for (int it = 0; it < ITERS; ++it) {
      if (h == 0) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
          for (int c = 0; c < 16; c += 4)
            *reinterpret_cast<float4*>(&sZ[rows[e]][c]) = make_float4(q[e][c], q[e][c + 1], q[e][c + 2], q[e][c + 3]);
      }
      __syncwarp();
      {  // Y = G Z: this lane's 4 rows x columns 4h..4h+3, k ascending from a zero accumulator
        float y[4][4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
          for (int c = 0; c < 4; ++c) y[e][c] = 0.f;
#pragma unroll 4
        for (int k = 0; k < RMAX; ++k) {
          const float4 z = *reinterpret_cast<const float4*>(&sZ[k][4 * h]);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float gk = sG[rows[e]][k];
            y[e][0] = __fmaf_rn(gk, z.x, y[e][0]);
            y[e][1] = __fmaf_rn(gk, z.y, y[e][1]);
            y[e][2] = __fmaf_rn(gk, z.z, y[e][2]);
            y[e][3] = __fmaf_rn(gk, z.w, y[e][3]);
          }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e)
          *reinterpret_cast<float4*>(&sY[rows[e]][4 * h]) = make_float4(y[e][0], y[e][1], y[e][2], y[e][3]);
      }
      __syncwarp();
#pragma unroll
      for (int e = 0; e < 4; ++e)
#pragma unroll
        for (int c = 0; c < 16; c += 4) {
          const float4 v = *reinterpret_cast<const float4*>(&sY[rows[e]][c]);
          q[e][c] = v.x;
          q[e][c + 1] = v.y;
          q[e][c + 2] = v.z;
          q[e][c + 3] = v.w;
        }
      // n0 = sqrt(sum(Y * Y, axis=0)) for the kept columns
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        float col[4] = {q[0][c], q[1][c], q[2][c], q[3][c]};
        const float n0c = sqrt_approx_ftz(col_sum<ROWORD, FUSE, XW>(col, col, lane));
        if (lane == 0) sN0[c] = n0c;
      }
      __syncwarp();
#pragma unroll 1
      for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          float yj[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) yj[e] = __fadd_rn(q[e][j], 0.f);  // sum(where(col j, Q, 0), axis=1)
          float s[16];
#pragma unroll
          for (int c = 0; c < 16; ++c) {
            if (c < j) {
              float col[4] = {q[0][c], q[1][c], q[2][c], q[3][c]};
              s[c] = col_sum<ROWORD, FUSE, XW>(col, yj, lane);
            } else {
              s[c] = 0.f;  // tl.where(offs_c < j, ..., 0.0)
            }
          }
          float v[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            // sum(Q * proj[None, :], axis=1): pairs (2t, 2t+1); columns >= 16 are +0 (pairs t + 8)
            float a[8];
#pragma unroll
            for (int tt = 0; tt < 8; ++tt) {
              if (2 * tt < j) {
                const float x0 = __fmul_rn(q[e][2 * tt], s[2 * tt]);
                const float pr = FUSE ? __fmaf_rn(q[e][2 * tt + 1], s[2 * tt + 1], x0)
                                      : __fadd_rn(x0, __fmul_rn(q[e][2 * tt + 1], s[2 * tt + 1]));
                a[tt] = __fadd_rn(pr, 0.f);
              } else {
                a[tt] = 0.f;
              }
            }
            const float b0 = __fadd_rn(a[0], a[4]), b1 = __fadd_rn(a[1], a[5]);
            const float b2 = __fadd_rn(a[2], a[6]), b3 = __fadd_rn(a[3], a[7]);
            const float c0 = __fadd_rn(b0, b2), c1 = __fadd_rn(b1, b3);
            v[e] = __fsub_rn(yj[e], __fadd_rn(c0, c1));
          }
          const float n = sqrt_approx_ftz(col_sum<ROWORD, FUSE, XW>(v, v, lane));
          bool ok = n > 1e-12f;
          if (p == 0) ok = ok && (n > __fmul_rn(rel_tol, __fadd_rn(sN0[j], 0.f)));
          const float den = fmaxf(n, 1e-30f);
#pragma unroll
          for (int e = 0; e < 4; ++e) q[e][j] = ok ? div_full(v[e], den) : 0.f;
        }
      }
    }
    if (h == 0) {
#pragma unroll
      for (int e = 0; e < 4; ++e)
#pragma unroll
        for (int c = 0; c < 16; c += 4)
          *reinterpret_cast<float4*>(&sZ[rows[e]][c]) = make_float4(q[e][c], q[e][c + 1], q[e][c + 2], q[e][c + 3]);
    }
  }
  __syncthreads();

  {  // projection of the R kept rows (tl.dot FMA: r ascending from zero)
    float au[16], aw[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) au[c] = aw[c] = 0.f;
#pragma unroll 4
    for (int r = 0; r < RMAX; ++r) {
      const float u = __half2float(sU[r][t]);
      const float w = __half2float(sW[r][t]);
#pragma unroll
      for (int c = 0; c < 16; c += 4) {
        const float4 z = *reinterpret_cast<const float4*>(&sZ[r][c]);
        au[c] = __fmaf_rn(z.x, u, au[c]);
        au[c + 1] = __fmaf_rn(z.y, u, au[c + 1]);
        au[c + 2] = __fmaf_rn(z.z, u, au[c + 2]);
        au[c + 3] = __fmaf_rn(z.w, u, au[c + 3]);
        aw[c] = __fmaf_rn(z.x, w, aw[c]);
        aw[c + 1] = __fmaf_rn(z.y, w, aw[c + 1]);
        aw[c + 2] = __fmaf_rn(z.z, w, aw[c + 2]);
        aw[c + 3] = __fmaf_rn(z.w, w, aw[c + 3]);
      }
    }
#pragma unroll
    for (int c = 0; c < 16; ++c) {
      Ut[c * KD + t] = __float2half_rn(au[c]);
      Wt[c * KD + t] = __float2half_rn(aw[c]);
    }
    if (t == 0) *pc = R;
  }
}

}  // namespace gdn_expiry
