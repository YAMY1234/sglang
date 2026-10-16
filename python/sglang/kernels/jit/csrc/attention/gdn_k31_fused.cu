// Fused prompt-end k31 factorization (docs/223 §9.4–9.7, round-2 R1, CUDA after the Triton version hit the kill
// line; layout round per lead 00:3x: C panel re-read from L2 instead of resident in shared memory, fp64 products on
// DMMA).
//
// Same algorithm as gdn_prefill_reference.factorize_prefill_k31 on the served recipe (mixed CholeskyQR + mixed
// eigh); the Python side (kernels/gdn_k31_cuda.py) computes `a` with the reference torch ops and launches the
// admitted Triton _fp32_jacobi between the two kernels here.
//
// One CTA of 128 threads per (row, layer, head) matrix; thread t owns row t of every 128 x 24 panel, so the
// triangular solves are per-thread forward substitutions in registers.
//   k31_phase1  : y = C^T omega with C = S - vbar a^T recomputed from S on every pass (L2-resident), three mixed
//                 CholeskyQR (shifted fp32 pass, plain fp32 pass, per-matrix fallback to the fp32 input + shared
//                 fallback counter, fp64 final pass with 1e-7 jitter; fp64 Gram on DMMA), one power step, small
//                 Gram + jitter + pad_below_spectrum + amax normalization -> Q, G64, GN.
//   k31_finalize: gdn_k31_eigh_mixed.correct_once on (G64, Z) on the dense blocks of the padded Gram's structure
//                 (24 real + 8 pad; exact, one 32 x 32 block when the structure is absent), products on DMMA, stable
//                 ascending order, U = (q wr)^T and W = (C q wr)^T stored, rows >= r zero.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include <type_traits>

namespace {

constexpr int D = 128;                   // V = K
constexpr int M = 24;                    // r + oversample
constexpr int MP = 32;                   // padded small dimension (eigh N)
constexpr int R = 16;                    // kept rank
constexpr int NT = 128;                  // threads per CTA = rows per panel
constexpr int NW = NT / 32;              // warps per CTA
constexpr int NPAIR = M * (M + 1) / 2;   // unique fp32 Gram entries

// gdn_prefill_reference._cholqr_fp32 jitters, as torch evaluates them: python double constant -> fp32 scalar.
constexpr float kShift = 40392.0f * 5.9604644775390625e-8f;                  // 11 (m n + n (n + 1)) 2^-24, exact
constexpr float kPlain = (float)(11.313708498984760390 * 5.9604644775390625e-8);  // sqrt(128) 2^-24
constexpr double kSqrtEps32 = 3.4526698300124390840e-4;                      // finfo(float32).eps ** .5

__device__ __forceinline__ float fmaT(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ double fmaT(double a, double b, double c) { return __fma_rn(a, b, c); }

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <typename T>
__device__ __forceinline__ T warp_max(T v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// D(8x8) += A(8x4) B(4x8) in fp64. Lane l: a = A(l / 4, l % 4), b = B(l % 4, l / 4),
// c0 / c1 = D(l / 4, 2 (l % 4) + 0 / 1).
__device__ __forceinline__ void dmma(double& c0, double& c1, double a, double b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
               : "+d"(c0), "+d"(c1)
               : "d"(a), "d"(b));
}

__device__ __forceinline__ void pair_index(int e, int& i, int& j) {
  int ii = (int)((sqrtf(8.f * e + 1.f) - 1.f) * 0.5f);
  while ((ii + 1) * (ii + 2) / 2 <= e) ++ii;
  while (ii * (ii + 1) / 2 > e) --ii;
  i = ii;
  j = e - ii * (ii + 1) / 2;
}

// fp32 G[i][j] = sum_t Y[t][i] Y[t][j] on the leading M x M block (both triangles); Y row-major [D][M].
__device__ void gram32(const float* __restrict__ Y, float* __restrict__ G) {
  for (int e = threadIdx.x; e < NPAIR; e += NT) {
    int i, j;
    pair_index(e, i, j);
    float acc = 0.f;
#pragma unroll 8
    for (int t = 0; t < D; ++t) acc = __fmaf_rn(Y[t * M + i], Y[t * M + j], acc);
    G[i * MP + j] = acc;
    G[j * MP + i] = acc;
  }
}

// fp64 G = Y^T Y on DMMA: the 6 lower 8 x 8 tiles of the 24 x 24 Gram over K = 128 rows, mirrored.
__device__ void gram64(const double* __restrict__ Y, double* __restrict__ G) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gi = lane >> 2, tq = lane & 3;
  for (int tile = warp; tile < 6; tile += NW) {
    const int ti = tile < 1 ? 0 : (tile < 3 ? 1 : 2);
    const int tj = tile - ti * (ti + 1) / 2;
    const int i0 = ti * 8, j0 = tj * 8;
    double c0 = 0.0, c1 = 0.0;
#pragma unroll 4
    for (int k0 = 0; k0 < D; k0 += 4) {
      const int k = k0 + tq;
      dmma(c0, c1, Y[k * M + i0 + gi], Y[k * M + j0 + gi]);
    }
    const int r = i0 + gi;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int c = j0 + 2 * tq + h;
      const double v = h ? c1 : c0;
      if (ti != tj || r >= c) {
        G[r * MP + c] = v;
        G[c * MP + r] = v;
      }
    }
  }
}

// Lower Cholesky of the leading M x M block in place (lower triangle read), one warp; lane i owns row i.
// Returns nonzero when a pivot is not finite and > 0 (LAPACK / cuSOLVER report info > 0 there).
template <typename T>
__device__ int cholesky_warp(T* A) {
  const int lane = threadIdx.x & 31;
  int bad = 0;
  for (int k = 0; k < M; ++k) {
    const T d = A[k * MP + k];
    bad |= !(d > T(0) && isfinite(d));
    const T piv = sqrt(d);
    if (lane > k && lane < M) A[lane * MP + k] = A[lane * MP + k] / piv;
    __syncwarp();
    if (lane == k) A[k * MP + k] = piv;
    if (lane > k && lane < M) {
      const T lik = A[lane * MP + k];
      for (int j = k + 1; j <= lane; ++j) A[lane * MP + j] = A[lane * MP + j] - lik * A[j * MP + k];
    }
    __syncwarp();
  }
  return bad;
}

// Row t of Q with Q L^T = Y (forward substitution, as solve_triangular(L, Y^T)).
template <typename T>
__device__ __forceinline__ void solve_row(T (&y)[M], const T* __restrict__ L) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
    T acc = y[i];
#pragma unroll
    for (int j = 0; j < i; ++j) acc = fmaT(-y[j], L[i * MP + j], acc);
    y[i] = acc / L[i * MP + i];
  }
}

// reference: x = (s - vb[..., None] * a[..., None, :]).transpose  (product, then difference: two roundings)
__device__ __forceinline__ float content(float s, float vb, float a) { return __fsub_rn(s, __fmul_rn(vb, a)); }

// y[j] = sum_v C[v][t] B[v][j]   (x @ B with x = C^T); column t of C from global (coalesced across threads)
__device__ __forceinline__ void gemm_ct(const float* __restrict__ s, const float* __restrict__ svb, float at,
                                        const float* __restrict__ Bs, float (&y)[M]) {
  const int t = threadIdx.x;
#pragma unroll
  for (int j = 0; j < M; ++j) y[j] = 0.f;
#pragma unroll 4
  for (int v = 0; v < D; ++v) {
    const float c = content(__ldg(s + v * D + t), svb[v], at);
    const float4* b = reinterpret_cast<const float4*>(Bs + v * M);
#pragma unroll
    for (int q = 0; q < M / 4; ++q) {
      const float4 bb = b[q];
      y[4 * q] = __fmaf_rn(c, bb.x, y[4 * q]);
      y[4 * q + 1] = __fmaf_rn(c, bb.y, y[4 * q + 1]);
      y[4 * q + 2] = __fmaf_rn(c, bb.z, y[4 * q + 2]);
      y[4 * q + 3] = __fmaf_rn(c, bb.w, y[4 * q + 3]);
    }
  }
}

// y[j] = sum_k C[t][k] B[k][j]   (C @ B, NB columns); row t of C from global as float4
template <int NB>
__device__ __forceinline__ void gemm_c(const float* __restrict__ s, float vbt, const float* __restrict__ sa,
                                       const float* __restrict__ Bs, float (&y)[NB]) {
  const int t = threadIdx.x;
  const float4* row = reinterpret_cast<const float4*>(s + t * D);
#pragma unroll
  for (int j = 0; j < NB; ++j) y[j] = 0.f;
#pragma unroll 2
  for (int k4 = 0; k4 < D / 4; ++k4) {
    const float4 sv = __ldg(row + k4);
    const float cs[4] = {content(sv.x, vbt, sa[4 * k4]), content(sv.y, vbt, sa[4 * k4 + 1]),
                         content(sv.z, vbt, sa[4 * k4 + 2]), content(sv.w, vbt, sa[4 * k4 + 3])};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float c = cs[u];
      const float4* b = reinterpret_cast<const float4*>(Bs + (4 * k4 + u) * NB);
#pragma unroll
      for (int q = 0; q < NB / 4; ++q) {
        const float4 bb = b[q];
        y[4 * q] = __fmaf_rn(c, bb.x, y[4 * q]);
        y[4 * q + 1] = __fmaf_rn(c, bb.y, y[4 * q + 1]);
        y[4 * q + 2] = __fmaf_rn(c, bb.z, y[4 * q + 2]);
        y[4 * q + 3] = __fmaf_rn(c, bb.w, y[4 * q + 3]);
      }
    }
  }
}

// gdn_prefill_reference._orth_cholqr2(y, mixed=True) on the 128 x 24 panel whose row t is y (in place).
// Yw: D*M doubles of scratch (also used as floats); Gd: MP*MP doubles (also used as floats); flag: one int.
__device__ void orth_mixed(float (&y)[M], double* __restrict__ Yw, double* __restrict__ Gd, int* flag,
                           unsigned long long* __restrict__ cnt) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  float* Yf = reinterpret_cast<float*>(Yw);
  float* Gf = reinterpret_cast<float*>(Gd);
  float q[M];
#pragma unroll
  for (int j = 0; j < M; ++j) q[j] = y[j];
  int nonfinite = 0;
  if (t == 0) *flag = 0;
#pragma unroll 1
  for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
    for (int j = 0; j < M; ++j) Yf[t * M + j] = q[j];
    __syncthreads();
    gram32(Yf, Gf);
    __syncthreads();
    if (warp == 0) {
      const float dg = lane < M ? Gf[lane * MP + lane] : 0.f;
      const float trace = warp_sum(dg);
      const float jitter = __fadd_rn(__fmul_rn(pass == 0 ? kShift : kPlain, trace), 1e-30f);
      __syncwarp();
      if (lane < M) Gf[lane * MP + lane] = __fadd_rn(dg, jitter);
      __syncwarp();
      const int bad = cholesky_warp<float>(Gf);
      if (lane == 0 && bad) *flag = 1;
    }
    __syncthreads();
    solve_row<float>(q, Gf);
#pragma unroll
    for (int j = 0; j < M; ++j) nonfinite |= !isfinite(q[j]);
    __syncthreads();
  }
  const bool ok = __syncthreads_or(nonfinite | *flag) == 0;
  if (!ok && t == 0) atomicAdd(cnt, 1ULL);
  double yd[M];
#pragma unroll
  for (int j = 0; j < M; ++j) yd[j] = ok ? (double)q[j] : (double)y[j];
#pragma unroll
  for (int j = 0; j < M; ++j) Yw[t * M + j] = yd[j];
  __syncthreads();
  gram64(Yw, Gd);
  __syncthreads();
  if (warp == 0) {
    const double dg = lane < M ? Gd[lane * MP + lane] : 0.0;
    const double mean = __dmul_rn(warp_sum(dg), 1.0 / M);
    const double jitter = __dadd_rn(__dmul_rn(1e-7, mean), 1e-30);
    __syncwarp();
    if (lane < M) Gd[lane * MP + lane] = __dadd_rn(dg, jitter);
    __syncwarp();
    cholesky_warp<double>(Gd);
  }
  __syncthreads();
  solve_row<double>(yd, Gd);
#pragma unroll
  for (int j = 0; j < M; ++j) y[j] = __double2float_rn(yd[j]);
  __syncthreads();
}

template <bool DEBUG>
__device__ __forceinline__ void store_stage(float* dbg, long long pid, int stage, const float (&y)[M]) {
  if constexpr (DEBUG) {
    float* p = dbg + ((pid * 5 + stage) * D + threadIdx.x) * MP;
#pragma unroll
    for (int j = 0; j < MP; ++j) p[j] = j < M ? y[j] : 0.f;
  }
}

// shared memory: panel scratch (D*M doubles; also the GEMM right operand as floats) + fp64 Gram + reduction
// scratch + vbar and a of this matrix
constexpr size_t kPhase1Smem = (size_t)D * M * 8 + (size_t)MP * MP * 8 + 64 * 8 + 2 * D * 4;

template <bool DEBUG>
__global__ void __launch_bounds__(NT, 4)
    k31_phase1(const float* __restrict__ S, const float* __restrict__ VB, const float* __restrict__ A,
               const float* __restrict__ OM, float* __restrict__ Q, double* __restrict__ G64,
               float* __restrict__ GN, unsigned long long* __restrict__ cnt, float* __restrict__ dbg, int LH,
               int H, long long om_sb, long long om_sh) {
  extern __shared__ __align__(16) unsigned char smem[];
  double* Yw = reinterpret_cast<double*>(smem);                        // [D][M] doubles
  float* Bs = reinterpret_cast<float*>(Yw);                            // GEMM right operand [D][M] (alias)
  double* Gd = Yw + D * M;                                             // [MP][MP]
  double* red = Gd + MP * MP;                                          // 64 doubles of reduction scratch
  int* flag = reinterpret_cast<int*>(red + 48);
  float* svb = reinterpret_cast<float*>(red + 64);                     // [D]
  float* sa = svb + D;                                                 // [D]
  const long long pid = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int lh = (int)(pid % LH), head = lh % H;
  const long long row = pid / LH;
  const float* s = S + pid * D * D;
  svb[t] = VB[(long long)lh * D + t];
  sa[t] = A[pid * D + t];
  const float* om = OM + row * om_sb + (long long)head * om_sh;
  for (int idx = t; idx < D * M; idx += NT) Bs[idx] = om[idx];
  __syncthreads();
  const float at = sa[t], vbt = svb[t];
  float y[M];
  gemm_ct(s, svb, at, Bs, y);                           // x @ omega
  store_stage<DEBUG>(dbg, pid, 0, y);
  __syncthreads();
  orth_mixed(y, Yw, Gd, flag, cnt);
  store_stage<DEBUG>(dbg, pid, 1, y);
#pragma unroll
  for (int j = 0; j < M; ++j) Bs[t * M + j] = y[j];
  __syncthreads();
  gemm_c<M>(s, vbt, sa, Bs, y);                         // x^T @ orth(y)
  store_stage<DEBUG>(dbg, pid, 2, y);
  __syncthreads();
  orth_mixed(y, Yw, Gd, flag, cnt);
  store_stage<DEBUG>(dbg, pid, 3, y);
#pragma unroll
  for (int j = 0; j < M; ++j) Bs[t * M + j] = y[j];
  __syncthreads();
  gemm_ct(s, svb, at, Bs, y);                           // power step: x @ orth(x^T orth(y))
  store_stage<DEBUG>(dbg, pid, 4, y);
  __syncthreads();
  orth_mixed(y, Yw, Gd, flag, cnt);                     // q
  float* qo = Q + (pid * D + t) * MP;
#pragma unroll
  for (int j = 0; j < MP; ++j) qo[j] = j < M ? y[j] : 0.f;
#pragma unroll
  for (int j = 0; j < M; ++j) Bs[t * M + j] = y[j];
  __syncthreads();
  gemm_c<M>(s, vbt, sa, Bs, y);                         // bm^T = C q
  __syncthreads();
  float* Yf = reinterpret_cast<float*>(Yw);
  float* Gf = reinterpret_cast<float*>(Gd);
#pragma unroll
  for (int j = 0; j < M; ++j) Yf[t * M + j] = y[j];
  __syncthreads();
  gram32(Yf, Gf);                                       // bm bm^T in fp32
  __syncthreads();
  double* Gp = Yw;                                      // padded fp64 Gram [MP][MP] (reuses the panel scratch)
  for (int idx = t; idx < MP * MP; idx += NT) {
    const int i = idx / MP, j = idx % MP;
    Gp[idx] = (i < M && j < M) ? (double)Gf[i * MP + j] : 0.0;
  }
  __syncthreads();
  if (warp == 0) {                                      // _small_eigh_fp64 jitter: 1e-7 mean(diag) + 1e-30
    const double dg = lane < M ? Gp[lane * MP + lane] : 0.0;
    const double mean = __dmul_rn(warp_sum(dg), 1.0 / M);
    const double jitter = __dadd_rn(__dmul_rn(1e-7, mean), 1e-30);
    if (lane < M) Gp[lane * MP + lane] = __dadd_rn(dg, jitter);
  }
  __syncthreads();
  if (t < M) {                                          // pad_below_spectrum: Gershgorin row sums
    double rs = 0.0;
    for (int j = 0; j < M; ++j) rs += fabs(Gp[t * MP + j]);
    red[t] = rs;
  }
  __syncthreads();
  if (warp == 0) {
    const double bound = warp_max(lane < M ? red[lane] : 0.0);
    const double pad = -(__dadd_rn(__dmul_rn(2.0, bound), 1.0));
    if (lane >= M) Gp[lane * MP + lane] = pad;
  }
  __syncthreads();
  double amax = 0.0;                                    // fp32_vectors: amax normalization before the cast
  for (int idx = t; idx < MP * MP; idx += NT) amax = fmax(amax, fabs(Gp[idx]));
  amax = warp_max(amax);
  if (lane == 0) red[32 + warp] = amax;
  __syncthreads();
  const double scale = fmax(fmax(fmax(red[32], red[33]), fmax(red[34], red[35])), 1e-300);
  double* g64 = G64 + pid * MP * MP;
  float* gn = GN + pid * MP * MP;
  for (int idx = t; idx < MP * MP; idx += NT) {
    g64[idx] = Gp[idx];
    gn[idx] = __double2float_rn(Gp[idx] / scale);
  }
}

// ---------------------------------------------------------------- finalize: Newton (correct_once) + factors
// X = coordinates of the padded Gram (rows of Z), E = eigen-columns (columns of Z). When the padded Gram is block
// diagonal (24 real + 8 pad coordinates) and every Z column lives in one block, all products below vanish across
// blocks; Newton then runs on the dense real (24) and pad (8) blocks, otherwise on one dense 32 x 32 block.
// C(n x n) = op(A) op(B) on DMMA, n a multiple of 8, row-major with leading dimension n.
template <bool TA, bool TB>
__device__ void dmm(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int n) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gi = lane >> 2, tq = lane & 3;
  const int tiles = n / 8;
  for (int tile = warp; tile < tiles * tiles; tile += NW) {
    const int i0 = (tile / tiles) * 8, j0 = (tile % tiles) * 8;
    double c0 = 0.0, c1 = 0.0;
    for (int k0 = 0; k0 < n; k0 += 4) {
      const int k = k0 + tq;
      const double a = TA ? A[k * n + i0 + gi] : A[(i0 + gi) * n + k];
      const double b = TB ? B[(j0 + gi) * n + k] : B[k * n + j0 + gi];
      dmma(c0, c1, a, b);
    }
    C[(i0 + gi) * n + j0 + 2 * tq] = c0;
    C[(i0 + gi) * n + j0 + 2 * tq + 1] = c1;
  }
}

struct NewtonBlock {
  int n;                         // block size (24 / 8, or 32)
  double *g, *q, *t1, *t2, *t3, *t4, *vals;
};

// correct_once, first half: gq, metric, projected, initial values (scale needs every block's values)
__device__ void newton_values(const NewtonBlock& b) {
  dmm<false, false>(b.g, b.q, b.t1, b.n);    // gq
  dmm<true, false>(b.q, b.q, b.t2, b.n);     // metric
  __syncthreads();
  dmm<true, false>(b.q, b.t1, b.t3, b.n);    // projected
  __syncthreads();
  for (int c = threadIdx.x; c < b.n; c += NT) b.vals[c] = b.t3[c * b.n + c] / b.t2[c * b.n + c];
  __syncthreads();
}

// correct_once, second half; leaves the corrected, normalized vectors in t3 and the new values in vals
__device__ void newton_update(const NewtonBlock& b, double scale) {
  const int n = b.n;
  for (int idx = threadIdx.x; idx < n * n; idx += NT) {   // update into t4; error = metric - I
    const int i = idx / n, j = idx % n;
    const double err = b.t2[idx] - (i == j ? 1.0 : 0.0);
    const double gap = b.vals[j] - b.vals[i];
    const bool safe = fabs(gap) > kSqrtEps32 * scale;
    const double residual = __dsub_rn(b.t3[idx], __dmul_rn(err, b.vals[j]));
    b.t4[idx] = safe ? residual / gap : -0.5 * err;
  }
  __syncthreads();
  dmm<false, false>(b.q, b.t4, b.t1, n);     // q update
  __syncthreads();
  for (int idx = threadIdx.x; idx < n * n; idx += NT) b.t1[idx] = b.q[idx] + b.t1[idx];   // corrected
  __syncthreads();
  dmm<true, false>(b.t1, b.t1, b.t2, n);     // ctc
  __syncthreads();
  for (int idx = threadIdx.x; idx < n * n; idx += NT) {
    const int i = idx / n, j = idx % n;
    b.t2[idx] = 0.5 * ((i == j ? 3.0 : 0.0) - b.t2[idx]);
  }
  __syncthreads();
  dmm<false, false>(b.t1, b.t2, b.t3, n);    // corrected (.5 (3 I - ctc))
  __syncthreads();
  for (int c = threadIdx.x; c < n; c += NT) {
    double ss = 0.0;
    for (int x = 0; x < n; ++x) ss = __fma_rn(b.t3[x * n + c], b.t3[x * n + c], ss);
    b.t4[c] = fmax(sqrt(ss), 1e-300);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < n * n; idx += NT) b.t3[idx] = b.t3[idx] / b.t4[idx % n];
  __syncthreads();
  dmm<false, false>(b.g, b.t3, b.t4, n);     // g corrected
  __syncthreads();
  for (int c = threadIdx.x; c < n; c += NT) {
    double v = 0.0;
    for (int x = 0; x < n; ++x) v = __fma_rn(b.t3[x * n + c], b.t4[x * n + c], v);
    b.vals[c] = v;
  }
  __syncthreads();
}

// raw g / q (2 x 32 x 32) + 6 dense buffers per block (24 + 8 structured, or one 32 x 32), then the U panel [D][R]
constexpr int kBlockDoubles = 6 * MP * MP > 6 * (M * M + 8 * 8) ? 6 * MP * MP : 6 * (M * M + 8 * 8);
constexpr size_t kFinalizeSmem = (size_t)2 * MP * MP * 8 + (size_t)kBlockDoubles * 8 + (size_t)D * R * 4 +
                                 (size_t)M * R * 4 + 2 * D * 4;

template <typename OutT, bool DEBUG>
__global__ void __launch_bounds__(NT, 3)
    k31_finalize(const float* __restrict__ S, const float* __restrict__ VB, const float* __restrict__ A,
                 const float* __restrict__ Q, const double* __restrict__ G64, const float* __restrict__ Z,
                 OutT* __restrict__ U, OutT* __restrict__ W, double* __restrict__ dbg, int LH, int rmax,
                 int force_full) {
  extern __shared__ __align__(16) unsigned char smem[];
  double* g = reinterpret_cast<double*>(smem);          // raw padded Gram [MP][MP]
  double* q = g + MP * MP;                               // raw Z (fp64) [MP][MP]
  double* blk = q + MP * MP;                             // dense block buffers (also one 32 x 32 block)
  float* Us = reinterpret_cast<float*>(blk + kBlockDoubles);   // [D][R]
  float* wr = Us + D * R;                                       // [M][R]
  float* sa = wr + M * R;                                       // [D]
  float* svb = sa + D;                                          // [D]
  __shared__ int xb[MP], eb[MP], xl[2][MP], el[2][MP], xn[2], en[2];
  __shared__ int order[MP], structured;
  __shared__ double vals[MP], scale_s;
  const long long pid = blockIdx.x;
  const int t = threadIdx.x;
  const int lh = (int)(pid % LH);
  for (int idx = t; idx < MP * MP; idx += NT) {
    g[idx] = G64[pid * MP * MP + idx];
    q[idx] = (double)Z[pid * MP * MP + idx];
  }
  sa[t] = A[pid * D + t];
  svb[t] = VB[(long long)lh * D + t];
  if (t == 0) structured = force_full ? 0 : 1;
  __syncthreads();
  if (!force_full) {     // Gram cross block zero; each Z column on the real or the pad rows only; 24 + 8
    int broken = 0;
    for (int idx = t; idx < MP * MP; idx += NT) {
      const int i = idx / MP, j = idx % MP;
      if ((i < M) != (j < M) && g[idx] != 0.0) broken = 1;
    }
    if (t < MP) {
      int realnz = 0, padnz = 0;
      for (int x = 0; x < MP; ++x) {
        if (q[x * MP + t] != 0.0) (x < M ? realnz : padnz) = 1;
      }
      eb[t] = padnz && !realnz ? 1 : 0;
      if (realnz && padnz) broken = 1;
    }
    broken = __syncthreads_or(broken);
    if (t == 0) {
      int npad = 0;
      for (int c = 0; c < MP; ++c) npad += eb[c];
      if (broken || npad != MP - M) structured = 0;
    }
    __syncthreads();
  }
  if (t == 0) {
    xn[0] = xn[1] = en[0] = en[1] = 0;
    for (int x = 0; x < MP; ++x) {
      const int b = structured && x >= M ? 1 : 0;
      xb[x] = b;
      xl[b][xn[b]++] = x;
    }
    for (int c = 0; c < MP; ++c) {
      const int b = structured ? eb[c] : 0;
      eb[c] = b;
      el[b][en[b]++] = c;
    }
  }
  __syncthreads();
  const int nblocks = structured ? 2 : 1;
  NewtonBlock nb[2];
  {
    double* p = blk;
    for (int b = 0; b < nblocks; ++b) {
      const int n = structured ? (b ? MP - M : M) : MP;
      nb[b].n = n;
      nb[b].g = p; p += n * n;
      nb[b].q = p; p += n * n;
      nb[b].t1 = p; p += n * n;
      nb[b].t2 = p; p += n * n;
      nb[b].t3 = p; p += n * n;
      nb[b].t4 = p; p += n * n;
      nb[b].vals = vals + (b ? M : 0);
    }
  }
  for (int b = 0; b < nblocks; ++b) {    // gather the dense blocks
    const int n = nb[b].n;
    for (int idx = t; idx < n * n; idx += NT) {
      const int i = idx / n, j = idx % n;
      nb[b].g[idx] = g[xl[b][i] * MP + xl[b][j]];
      nb[b].q[idx] = q[xl[b][i] * MP + el[b][j]];
    }
  }
  __syncthreads();
  for (int b = 0; b < nblocks; ++b) newton_values(nb[b]);
  if (t < 32) {
    const double sc = warp_max(fabs(vals[t]));
    if (t == 0) scale_s = fmax(sc, 1e-300);
  }
  __syncthreads();
  const double scale = scale_s;
  for (int b = 0; b < nblocks; ++b) newton_update(nb[b], scale);
  // values per E index, and the stable ascending order over all 32 columns (ties by column index)
  __shared__ double ev[MP];
  __shared__ int eblk[MP], epos[MP];
  if (t < MP) {
    const int b = eb[t];
    int p = 0;
    while (el[b][p] != t) ++p;
    eblk[t] = b;
    epos[t] = p;
    ev[t] = nb[b].vals[p];
  }
  __syncthreads();
  if (t < MP) {
    const double v = ev[t];
    int rank = 0;
    for (int c = 0; c < MP; ++c) rank += (ev[c] < v) || (ev[c] == v && c < t);
    order[rank] = t;
  }
  __syncthreads();
  // vector of column e at coordinate x (zero off its block)
  auto vec = [&](int x, int e) -> double {
    const int b = eblk[e];
    if (xb[x] != b) return 0.0;
    int i = 0;
    while (xl[b][i] != x) ++i;
    return nb[b].t3[i * nb[b].n + epos[e]];
  };
  if constexpr (DEBUG) {
    double* p = dbg + pid * (MP + 1) * MP;
    for (int idx = t; idx < MP * MP; idx += NT) p[idx] = vec(idx / MP, order[idx % MP]);
    if (t < MP) p[MP * MP + t] = ev[order[t]];
  }
  // small_eigh unpad (rows < M) and the top R columns, fp32 as the reference's .to(float32)
  for (int idx = t; idx < M * R; idx += NT) {
    const int x = idx / R, r = idx % R;
    wr[idx] = __double2float_rn(vec(x, order[MP - R + r]));
  }
  __syncthreads();
  // u_ref = q wr (row t), then W^T = C u_ref with C re-read from S
  const float* qf = Q + (pid * D + t) * MP;
  float u[R];
#pragma unroll
  for (int r = 0; r < R; ++r) u[r] = 0.f;
#pragma unroll 4
  for (int j = 0; j < M; ++j) {
    const float qq = qf[j];
#pragma unroll
    for (int r = 0; r < R; ++r) u[r] = __fmaf_rn(qq, wr[j * R + r], u[r]);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) Us[t * R + r] = u[r];
  __syncthreads();
  float w[R];
  gemm_c<R>(S + pid * D * D, svb[t], sa, Us, w);
  OutT* uo = U + pid * rmax * D;
  OutT* wo = W + pid * rmax * D;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if constexpr (std::is_same<OutT, at::Half>::value) {
      uo[r * D + t] = __float2half_rn(u[r]);
      wo[r * D + t] = __float2half_rn(w[r]);
    } else {
      uo[r * D + t] = u[r];
      wo[r * D + t] = w[r];
    }
  }
  for (int r = R; r < rmax; ++r) {
    uo[r * D + t] = OutT(0.f);
    wo[r * D + t] = OutT(0.f);
  }
}

// One configuration flag per kernel instantiation (the debug and plain kernels share a function-pointer type, so a
// static inside a generic lambda would configure only the first one launched).
template <bool DEBUG>
void launch_phase1(int n, cudaStream_t stream, torch::Tensor& s, torch::Tensor& vb, torch::Tensor& a,
                   torch::Tensor& om, torch::Tensor& q, torch::Tensor& g64, torch::Tensor& gn, torch::Tensor& cnt,
                   torch::Tensor& dbg, int64_t lh, int64_t heads, int64_t om_sb, int64_t om_sh) {
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k31_phase1<DEBUG>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)kPhase1Smem));
    configured = true;
  }
  k31_phase1<DEBUG><<<n, NT, kPhase1Smem, stream>>>(
      s.data_ptr<float>(), vb.data_ptr<float>(), a.data_ptr<float>(), om.data_ptr<float>(), q.data_ptr<float>(),
      g64.data_ptr<double>(), gn.data_ptr<float>(), reinterpret_cast<unsigned long long*>(cnt.data_ptr<int64_t>()),
      dbg.data_ptr<float>(), (int)lh, (int)heads, (long long)om_sb, (long long)om_sh);
}

template <typename OutT, bool DEBUG>
void launch_finalize(int n, cudaStream_t stream, torch::Tensor& s, torch::Tensor& vb, torch::Tensor& a,
                     torch::Tensor& q, torch::Tensor& g64, torch::Tensor& z, OutT* u, OutT* w, torch::Tensor& dbg,
                     int64_t lh, int64_t rmax, bool force_full) {
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k31_finalize<OutT, DEBUG>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)kFinalizeSmem));
    configured = true;
  }
  k31_finalize<OutT, DEBUG><<<n, NT, kFinalizeSmem, stream>>>(
      s.data_ptr<float>(), vb.data_ptr<float>(), a.data_ptr<float>(), q.data_ptr<float>(), g64.data_ptr<double>(),
      z.data_ptr<float>(), u, w, dbg.data_ptr<double>(), (int)lh, (int)rmax, force_full ? 1 : 0);
}

}  // namespace

void phase1(torch::Tensor s, torch::Tensor vb, torch::Tensor a, torch::Tensor om, torch::Tensor q,
            torch::Tensor g64, torch::Tensor gn, torch::Tensor cnt, torch::Tensor dbg, int64_t lh, int64_t heads,
            int64_t om_sb, int64_t om_sh, bool debug) {
  const int n = (int)(s.size(0) * s.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (debug)
    launch_phase1<true>(n, stream, s, vb, a, om, q, g64, gn, cnt, dbg, lh, heads, om_sb, om_sh);
  else
    launch_phase1<false>(n, stream, s, vb, a, om, q, g64, gn, cnt, dbg, lh, heads, om_sb, om_sh);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize(torch::Tensor s, torch::Tensor vb, torch::Tensor a, torch::Tensor q, torch::Tensor g64,
              torch::Tensor z, torch::Tensor u, torch::Tensor w, torch::Tensor dbg, int64_t lh, int64_t rmax,
              bool force_full, bool debug) {
  const int n = (int)(s.size(0) * s.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (u.scalar_type() == at::kHalf) {
    auto* up = u.data_ptr<at::Half>();
    auto* wp = w.data_ptr<at::Half>();
    if (debug)
      launch_finalize<at::Half, true>(n, stream, s, vb, a, q, g64, z, up, wp, dbg, lh, rmax, force_full);
    else
      launch_finalize<at::Half, false>(n, stream, s, vb, a, q, g64, z, up, wp, dbg, lh, rmax, force_full);
  } else {
    auto* up = u.data_ptr<float>();
    auto* wp = w.data_ptr<float>();
    if (debug)
      launch_finalize<float, true>(n, stream, s, vb, a, q, g64, z, up, wp, dbg, lh, rmax, force_full);
    else
      launch_finalize<float, false>(n, stream, s, vb, a, q, g64, z, up, wp, dbg, lh, rmax, force_full);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("phase1", &phase1, "k31 fused phase 1");
  m.def("finalize", &finalize, "k31 fused finalize");
}
