// Fused prompt-end k31 factorization (docs/223 §9.4, round-2 R1, CUDA after the Triton version hit the kill line).
//
// Same algorithm as gdn_prefill_reference.factorize_prefill_k31 on the served recipe (mixed CholeskyQR + mixed
// eigh); the Python side (kernels/gdn_k31_cuda.py) computes `a` with the reference torch ops and launches the
// admitted Triton _fp32_jacobi between the two kernels here.
//
// One CTA of 128 threads per (row, layer, head) matrix; thread t owns row t of every 128 x 24 panel, so the
// triangular solves are per-thread forward substitutions in registers and every Gram is one pass over the rows.
//   k31_phase1  : C = S - vbar a^T (shared memory), y = C^T omega, three mixed CholeskyQR (shifted fp32 pass,
//                 plain fp32 pass, per-matrix fallback to the fp32 input + shared fallback counter, fp64 final
//                 pass with 1e-7 jitter), one power step, small Gram + jitter + pad_below_spectrum + amax
//                 normalization -> Q, G64, GN.
//   k31_finalize: gdn_k31_eigh_mixed.correct_once on (G64, Z) restricted to the block structure of the padded
//                 Gram (24 real + 8 pad coordinates; exact, falls back to one 32 x 32 block when the structure is
//                 absent), stable ascending order, U = (q wr)^T and W = (C q wr)^T stored, rows >= r zero.
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
constexpr int CS = D + 1;                // shared-memory row stride of C (bank-conflict free both ways)
constexpr int NPAIR = M * (M + 1) / 2;   // unique Gram entries

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

__device__ __forceinline__ void pair_index(int e, int& i, int& j) {
  int ii = (int)((sqrtf(8.f * e + 1.f) - 1.f) * 0.5f);
  while ((ii + 1) * (ii + 2) / 2 <= e) ++ii;
  while (ii * (ii + 1) / 2 > e) --ii;
  i = ii;
  j = e - ii * (ii + 1) / 2;
}

// G[i][j] = sum_t Y[t][i] Y[t][j] on the leading M x M block (both triangles); Y row-major [D][M].
template <typename T>
__device__ void gram(const T* __restrict__ Y, T* __restrict__ G) {
  for (int e = threadIdx.x; e < NPAIR; e += NT) {
    int i, j;
    pair_index(e, i, j);
    T acc = T(0);
#pragma unroll 8
    for (int t = 0; t < D; ++t) acc = fmaT(Y[t * M + i], Y[t * M + j], acc);
    G[i * MP + j] = acc;
    G[j * MP + i] = acc;
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

// y[j] = sum_v C[v][t] B[v][j]   (x @ B with x = C^T)
__device__ __forceinline__ void gemm_ct(const float* __restrict__ Cs, const float* __restrict__ Bs, float (&y)[M]) {
  const int t = threadIdx.x;
#pragma unroll
  for (int j = 0; j < M; ++j) y[j] = 0.f;
#pragma unroll 4
  for (int v = 0; v < D; ++v) {
    const float c = Cs[v * CS + t];
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

// y[j] = sum_k C[t][k] B[k][j]   (x^T @ B = C @ B)
__device__ __forceinline__ void gemm_c(const float* __restrict__ Cs, const float* __restrict__ Bs, float (&y)[M]) {
  const int t = threadIdx.x;
#pragma unroll
  for (int j = 0; j < M; ++j) y[j] = 0.f;
#pragma unroll 4
  for (int k = 0; k < D; ++k) {
    const float c = Cs[t * CS + k];
    const float4* b = reinterpret_cast<const float4*>(Bs + k * M);
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
    gram<float>(Yf, Gf);
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
  gram<double>(Yw, Gd);
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

constexpr size_t kPhase1Smem = (size_t)D * CS * 4 + (size_t)D * M * 8 + (size_t)MP * MP * 8 + 64 * 8;

template <bool DEBUG>
__global__ void __launch_bounds__(NT, 2)
    k31_phase1(const float* __restrict__ S, const float* __restrict__ VB, const float* __restrict__ A,
               const float* __restrict__ OM, float* __restrict__ Q, double* __restrict__ G64,
               float* __restrict__ GN, unsigned long long* __restrict__ cnt, float* __restrict__ dbg, int LH,
               int H, long long om_sb, long long om_sh) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* Cs = reinterpret_cast<float*>(smem);                          // [D][CS]
  double* Yw = reinterpret_cast<double*>(smem + (size_t)D * CS * 4);   // [D][M] doubles
  float* Bs = reinterpret_cast<float*>(Yw);                            // GEMM right operand [D][M] (alias)
  double* Gd = Yw + D * M;                                             // [MP][MP]
  double* red = Gd + MP * MP;                                          // 64 doubles of reduction scratch
  int* flag = reinterpret_cast<int*>(red + 48);
  const long long pid = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int lh = (int)(pid % LH), head = lh % H;
  const long long row = pid / LH;
  const float* s = S + pid * D * D;
  const float* vb = VB + (long long)lh * D;
  const float* a = A + pid * D;
  // reference: x = (s - vb[..., None] * a[..., None, :]).transpose  (product, then difference: two roundings)
  for (int idx = t; idx < D * D; idx += NT) {
    const int v = idx / D, k = idx % D;
    Cs[v * CS + k] = __fsub_rn(s[idx], __fmul_rn(vb[v], a[k]));
  }
  const float* om = OM + row * om_sb + (long long)head * om_sh;
  for (int idx = t; idx < D * M; idx += NT) Bs[idx] = om[idx];
  __syncthreads();
  float y[M];
  gemm_ct(Cs, Bs, y);                                   // x @ omega
  store_stage<DEBUG>(dbg, pid, 0, y);
  __syncthreads();
  orth_mixed(y, Yw, Gd, flag, cnt);
  store_stage<DEBUG>(dbg, pid, 1, y);
#pragma unroll
  for (int j = 0; j < M; ++j) Bs[t * M + j] = y[j];
  __syncthreads();
  gemm_c(Cs, Bs, y);                                    // x^T @ orth(y)
  store_stage<DEBUG>(dbg, pid, 2, y);
  __syncthreads();
  orth_mixed(y, Yw, Gd, flag, cnt);
  store_stage<DEBUG>(dbg, pid, 3, y);
#pragma unroll
  for (int j = 0; j < M; ++j) Bs[t * M + j] = y[j];
  __syncthreads();
  gemm_ct(Cs, Bs, y);                                   // power step: x @ orth(x^T orth(y))
  store_stage<DEBUG>(dbg, pid, 4, y);
  __syncthreads();
  orth_mixed(y, Yw, Gd, flag, cnt);                     // q
  float* qo = Q + (pid * D + t) * MP;
#pragma unroll
  for (int j = 0; j < MP; ++j) qo[j] = j < M ? y[j] : 0.f;
#pragma unroll
  for (int j = 0; j < M; ++j) Bs[t * M + j] = y[j];
  __syncthreads();
  gemm_c(Cs, Bs, y);                                    // bm^T = C q
  __syncthreads();
  float* Yf = reinterpret_cast<float*>(Yw);
  float* Gf = reinterpret_cast<float*>(Gd);
#pragma unroll
  for (int j = 0; j < M; ++j) Yf[t * M + j] = y[j];
  __syncthreads();
  gram<float>(Yf, Gf);                                  // bm bm^T in fp32
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
// Index spaces: X = coordinates of the padded Gram (rows of Z), E = eigen-columns (columns of Z). Each index has a
// block id; with the padded Gram block diagonal (real 24 + pad 8) and Z block structured, every product below is
// exactly zero across blocks, so sums run over the shared block only. Without that structure everything is one
// block (plain 32 x 32).
struct Blocks {
  int xb[MP], eb[MP];           // block id per X / E index
  int xl[2][MP], el[2][MP];     // index lists per block (ascending)
  int xn[2], en[2];
};

// out[i][j] = sum_{k in block(i)} A(i, k) B(k, j) for block(i) == block(j), else 0.
// A(i, k) = TA ? A[k][i] : A[i][k]. kspace / ispace / jspace: 0 = X, 1 = E.
template <bool TA>
__device__ void bmm(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ out,
                    const Blocks& bl, int ispace, int kspace, int jspace) {
  const int* ib = ispace ? bl.eb : bl.xb;
  const int* jb = jspace ? bl.eb : bl.xb;
  for (int idx = threadIdx.x; idx < MP * MP; idx += NT) {
    const int i = idx / MP, j = idx % MP;
    const int b = ib[i];
    double acc = 0.0;
    if (b == jb[j]) {
      const int* kl = kspace ? bl.el[b] : bl.xl[b];
      const int kn = kspace ? bl.en[b] : bl.xn[b];
      for (int p = 0; p < kn; ++p) {
        const int k = kl[p];
        acc = __fma_rn(TA ? A[k * MP + i] : A[i * MP + k], B[k * MP + j], acc);
      }
    }
    out[idx] = acc;
  }
}

constexpr size_t kFinalizeSmem = (size_t)D * CS * 4 + (size_t)D * R * 4 + (size_t)M * R * 4 + 2 * MP * 8 + 1024;

template <typename OutT, bool DEBUG>
__global__ void __launch_bounds__(NT, 2)
    k31_finalize(const float* __restrict__ S, const float* __restrict__ VB, const float* __restrict__ A,
                 const float* __restrict__ Q, const double* __restrict__ G64, const float* __restrict__ Z,
                 OutT* __restrict__ U, OutT* __restrict__ W, double* __restrict__ dbg, int LH, int rmax,
                 int force_full) {
  extern __shared__ __align__(16) unsigned char smem[];
  // Newton buffers (6 x 32 x 32 doubles = 48 KB) alias the C panel, which is loaded after the eigen step.
  double* g = reinterpret_cast<double*>(smem);
  double* q = g + MP * MP;
  double* T1 = q + MP * MP;
  double* T2 = T1 + MP * MP;
  double* T3 = T2 + MP * MP;
  double* T4 = T3 + MP * MP;
  float* Cs = reinterpret_cast<float*>(smem);                                    // [D][CS] (after Newton)
  float* Us = reinterpret_cast<float*>(smem + (size_t)D * CS * 4);               // [D][R]
  float* wr = Us + D * R;                                                        // [M][R]
  double* vals = reinterpret_cast<double*>(wr + M * R);                          // [MP]
  double* red = vals + MP;                                                       // [MP]
  __shared__ Blocks bl;
  __shared__ int order[MP];
  __shared__ int structured;
  const long long pid = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int lh = (int)(pid % LH);
  for (int idx = t; idx < MP * MP; idx += NT) {
    g[idx] = G64[pid * MP * MP + idx];
    q[idx] = (double)Z[pid * MP * MP + idx];
  }
  if (t == 0) structured = force_full ? 0 : 1;
  __syncthreads();
  // structure check: Gram cross block zero; each Z column supported on the real or the pad rows only, 24 + 8
  if (!force_full) {
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
      bl.eb[t] = padnz && !realnz ? 1 : 0;
      if (realnz && padnz) broken = 1;
    }
    broken = __syncthreads_or(broken);
    if (t == 0) {
      int npad = 0;
      for (int c = 0; c < MP; ++c) npad += bl.eb[c];
      if (broken || npad != MP - M) structured = 0;
    }
    __syncthreads();
  }
  if (t == 0) {
    bl.xn[0] = bl.xn[1] = bl.en[0] = bl.en[1] = 0;
    for (int x = 0; x < MP; ++x) {
      const int b = structured && x >= M ? 1 : 0;
      bl.xb[x] = b;
      bl.xl[b][bl.xn[b]++] = x;
    }
    for (int c = 0; c < MP; ++c) {
      const int b = structured ? bl.eb[c] : 0;
      bl.eb[c] = b;
      bl.el[b][bl.en[b]++] = c;
    }
  }
  __syncthreads();
  // correct_once(g, q)
  bmm<false>(g, q, T1, bl, 0, 0, 1);       // gq = g q            (X x E)
  bmm<true>(q, q, T2, bl, 1, 0, 1);        // metric = q^T q      (E x E)
  __syncthreads();
  bmm<true>(q, T1, T3, bl, 1, 0, 1);       // projected = q^T gq  (E x E)
  __syncthreads();
  if (t < MP) vals[t] = T3[t * MP + t] / T2[t * MP + t];
  __syncthreads();
  if (warp == 0) {
    const double sc = warp_max(fabs(vals[lane]));
    if (lane == 0) red[0] = fmax(sc, 1e-300);
  }
  __syncthreads();
  const double scale = red[0];
  for (int idx = t; idx < MP * MP; idx += NT) {   // update (E x E) into T4; error = metric - I
    const int i = idx / MP, j = idx % MP;
    double u = 0.0;
    if (bl.eb[i] == bl.eb[j]) {
      const double err = T2[idx] - (i == j ? 1.0 : 0.0);
      const double gap = vals[j] - vals[i];
      const bool safe = fabs(gap) > kSqrtEps32 * scale;
      const double residual = __dsub_rn(T3[idx], __dmul_rn(err, vals[j]));
      u = safe ? residual / gap : -0.5 * err;
    }
    T4[idx] = u;
  }
  __syncthreads();
  bmm<false>(q, T4, T1, bl, 0, 1, 1);      // q update            (X x E)
  __syncthreads();
  for (int idx = t; idx < MP * MP; idx += NT) T1[idx] = q[idx] + T1[idx];   // corrected
  __syncthreads();
  bmm<true>(T1, T1, T2, bl, 1, 0, 1);      // ctc                 (E x E)
  __syncthreads();
  for (int idx = t; idx < MP * MP; idx += NT) {
    const int i = idx / MP, j = idx % MP;
    T2[idx] = bl.eb[i] == bl.eb[j] ? 0.5 * ((i == j ? 3.0 : 0.0) - T2[idx]) : 0.0;
  }
  __syncthreads();
  bmm<false>(T1, T2, T3, bl, 0, 1, 1);     // corrected (.5 (3 I - ctc))
  __syncthreads();
  if (t < MP) {
    double ss = 0.0;
    for (int x = 0; x < MP; ++x) ss = __fma_rn(T3[x * MP + t], T3[x * MP + t], ss);
    red[t] = fmax(sqrt(ss), 1e-300);
  }
  __syncthreads();
  for (int idx = t; idx < MP * MP; idx += NT) T3[idx] = T3[idx] / red[idx % MP];
  __syncthreads();
  bmm<false>(g, T3, T4, bl, 0, 0, 1);      // g corrected
  __syncthreads();
  if (t < MP) {
    double v = 0.0;
    for (int x = 0; x < MP; ++x) v = __fma_rn(T3[x * MP + t], T4[x * MP + t], v);
    vals[t] = v;
  }
  __syncthreads();
  if (t < MP) {                            // stable ascending order (ties by column index)
    const double v = vals[t];
    int rank = 0;
    for (int c = 0; c < MP; ++c) rank += (vals[c] < v) || (vals[c] == v && c < t);
    order[rank] = t;
  }
  __syncthreads();
  if constexpr (DEBUG) {
    double* p = dbg + pid * (MP + 1) * MP;
    for (int idx = t; idx < MP * MP; idx += NT) p[idx] = T3[(idx / MP) * MP + order[idx % MP]];
    if (t < MP) p[MP * MP + t] = vals[order[t]];
  }
  // small_eigh unpad (rows < M) and the top R columns, fp32 as the reference's .to(float32)
  for (int idx = t; idx < M * R; idx += NT) {
    const int x = idx / R, r = idx % R;
    wr[idx] = __double2float_rn(T3[x * MP + order[MP - R + r]]);
  }
  __syncthreads();
  // u_ref = q wr (row t), C panel, W^T = C u_ref
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
  const float* s = S + pid * D * D;
  const float* vb = VB + (long long)lh * D;
  const float* a = A + pid * D;
  for (int idx = t; idx < D * D; idx += NT) {
    const int v = idx / D, k = idx % D;
    Cs[v * CS + k] = __fsub_rn(s[idx], __fmul_rn(vb[v], a[k]));
  }
#pragma unroll
  for (int r = 0; r < R; ++r) Us[t * R + r] = u[r];
  __syncthreads();
  float w[R];
#pragma unroll
  for (int r = 0; r < R; ++r) w[r] = 0.f;
#pragma unroll 4
  for (int k = 0; k < D; ++k) {
    const float c = Cs[t * CS + k];
#pragma unroll
    for (int r = 0; r < R; ++r) w[r] = __fmaf_rn(c, Us[k * R + r], w[r]);
  }
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

template <typename K>
void ensure_smem(K kernel, size_t bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
}

}  // namespace

void phase1(torch::Tensor s, torch::Tensor vb, torch::Tensor a, torch::Tensor om, torch::Tensor q,
            torch::Tensor g64, torch::Tensor gn, torch::Tensor cnt, torch::Tensor dbg, int64_t lh, int64_t heads,
            int64_t om_sb, int64_t om_sh, bool debug) {
  const int n = (int)(s.size(0) * s.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    static bool configured = false;
    if (!configured) {
      ensure_smem(kernel, kPhase1Smem);
      configured = true;
    }
    kernel<<<n, NT, kPhase1Smem, stream>>>(
        s.data_ptr<float>(), vb.data_ptr<float>(), a.data_ptr<float>(), om.data_ptr<float>(), q.data_ptr<float>(),
        g64.data_ptr<double>(), gn.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(cnt.data_ptr<int64_t>()), dbg.data_ptr<float>(), (int)lh,
        (int)heads, (long long)om_sb, (long long)om_sh);
  };
  if (debug) launch(k31_phase1<true>); else launch(k31_phase1<false>);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize(torch::Tensor s, torch::Tensor vb, torch::Tensor a, torch::Tensor q, torch::Tensor g64,
              torch::Tensor z, torch::Tensor u, torch::Tensor w, torch::Tensor dbg, int64_t lh, int64_t rmax,
              bool force_full, bool debug) {
  const int n = (int)(s.size(0) * s.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto run = [&](auto kernel, auto* uptr, auto* wptr) {
    static bool configured = false;
    if (!configured) {
      ensure_smem(kernel, kFinalizeSmem);
      configured = true;
    }
    kernel<<<n, NT, kFinalizeSmem, stream>>>(s.data_ptr<float>(), vb.data_ptr<float>(), a.data_ptr<float>(),
                                             q.data_ptr<float>(), g64.data_ptr<double>(), z.data_ptr<float>(),
                                             uptr, wptr, dbg.data_ptr<double>(), (int)lh, (int)rmax,
                                             force_full ? 1 : 0);
  };
  if (u.scalar_type() == at::kHalf) {
    auto* up = u.data_ptr<at::Half>();
    auto* wp = w.data_ptr<at::Half>();
    if (debug) run(k31_finalize<at::Half, true>, up, wp); else run(k31_finalize<at::Half, false>, up, wp);
  } else {
    auto* up = u.data_ptr<float>();
    auto* wp = w.data_ptr<float>();
    if (debug) run(k31_finalize<float, true>, up, wp); else run(k31_finalize<float, false>, up, wp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("phase1", &phase1, "k31 fused phase 1");
  m.def("finalize", &finalize, "k31 fused finalize");
}
