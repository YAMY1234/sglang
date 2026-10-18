// Stand-alone decode hop launches built on sgl_kernel/distributed/q35_hop.cuh:
//   publish       push this rank's bf16 partial [M, 4096] (the all-reduce input)
//   publish_fin   finalize the routed MoE rows (top-k weighted, plus the gated shared output)
//                 and push the result
//   consume       pull + rank-order sum + residual + RMSNorm (+ static FP8), re-arm, advance
// Outputs are bit-identical to the FlashInfer MNNVL LL kernels plus the static FP8 quant.
// Built as a shared library and driven through ctypes (see srt/layers/q35_hop.py).

#include <cuda_runtime.h>
#include <stdint.h>

#include "sgl_kernel/distributed/q35_hop.cuh"

namespace {

using namespace q35hop;

struct Args {
  Site site;
  const uint16_t* partial;      // [M, H] bf16 (publish)
  const uint16_t* routed;       // [M*K, H] bf16 (publish_fin), rows indexed by permuted
  const uint16_t* expert_w;     // [M, K] bf16
  const int32_t* permuted;      // [M, K]; -1 skips a slot
  const uint16_t* shared;       // [M, H] bf16 gated shared expert output, or null
  const uint16_t* residual_in;  // [M, H] bf16
  const uint16_t* gamma;        // [H] bf16, the applied multiplier
  const float* fp8_scale;       // [1], or null
  uint16_t* residual_out;       // [M, H] bf16
  uint16_t* norm_out;           // [M, H] bf16, or null
  uint8_t* fp8_out;             // [M, H] e4m3, or null
  int m, top_k;
  float eps;
};

constexpr int kMaxTopK = 16;

__device__ __forceinline__ uint4 ld_v4(const void* p) { return *static_cast<const uint4*>(p); }

__device__ __forceinline__ size_t row_elem(int token, int frag) {
  return static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
}

// sum_k w_k * routed[perm_k] + shared, in k order with separate multiply and add, as the
// reference finalize does; every row load is issued before the sum.
__device__ __forceinline__ uint4 finalize_frag(const Args& a, int token, int frag, const int32_t* rows,
                                               const float* wts) {
  uint4 src[kMaxTopK];
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k)
    if (k < a.top_k && rows[k] >= 0) src[k] = ld_v4(a.routed + row_elem(rows[k], frag));
  uint4 sh = make_uint4(0, 0, 0, 0);
  if (a.shared != nullptr) sh = ld_v4(a.shared + row_elem(token, frag));
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k) {
    if (k < a.top_k && rows[k] >= 0) {
      float f[8];
      unpack8(src[k], f);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], __fmul_rn(f[e], wts[k]));
    }
  }
  if (a.shared != nullptr) {
    float f[8];
    unpack8(sh, f);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
  }
  return pack8(acc);
}

// Finalize-publish with 4 bf16 per thread (8 CTAs x 128 threads per token): twice the loads in
// flight of the fragment-wide layout; the per-element sum order is unchanged.
constexpr int kQuadCtas = 8;

__global__ void __launch_bounds__(kRowThreads) publish_fin_quad_kernel(Args a) {
  __shared__ int32_t rows[kMaxTopK];
  __shared__ float wts[kMaxTopK];
  const int token = blockIdx.x;
  const int quad = blockIdx.y * kRowThreads + threadIdx.x;
  if (threadIdx.x < a.top_k) {
    const int i = token * a.top_k + threadIdx.x;
    const int r = a.permuted[i];
    rows[threadIdx.x] = r;
    wts[threadIdx.x] = r < 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(a.expert_w[i]) << 16);
  }
  __syncthreads();
  pdl_wait();
  const uint32_t stage = stage_peek(a.site);
  const size_t col = static_cast<size_t>(quad) * 4;
  uint2 src[kMaxTopK];
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k)
    if (k < a.top_k && rows[k] >= 0) src[k] = *reinterpret_cast<const uint2*>(a.routed + static_cast<size_t>(rows[k]) * kHidden + col);
  uint2 sh = make_uint2(0, 0);
  if (a.shared != nullptr) sh = *reinterpret_cast<const uint2*>(a.shared + static_cast<size_t>(token) * kHidden + col);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k) {
    if (k < a.top_k && rows[k] >= 0) {
      const float f[4] = {bf_lo(src[k].x), bf_hi(src[k].x), bf_lo(src[k].y), bf_hi(src[k].y)};
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fadd_rn(acc[e], __fmul_rn(f[e], wts[k]));
    }
  }
  if (a.shared != nullptr) {
    const float f[4] = {bf_lo(sh.x), bf_hi(sh.x), bf_lo(sh.y), bf_hi(sh.y)};
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
  }
  push_quad(a.site, stage, token, quad, make_uint2(pack_bf16x2(acc[0], acc[1]), pack_bf16x2(acc[2], acc[3])));
  pdl_trigger();
}

template <bool FIN>
__global__ void __launch_bounds__(kRowThreads) publish_kernel(Args a) {
  __shared__ int32_t rows[kMaxTopK];
  __shared__ float wts[kMaxTopK];
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kRowThreads + threadIdx.x;
  if (FIN) {
    // Routing metadata comes from the router, several launches upstream; it is staged before
    // the PDL wait, as the reference finalize does.
    if (threadIdx.x < a.top_k) {
      const int i = token * a.top_k + threadIdx.x;
      const int r = a.permuted[i];
      rows[threadIdx.x] = r;
      wts[threadIdx.x] = r < 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(a.expert_w[i]) << 16);
    }
    __syncthreads();
  }
  pdl_wait();
  const uint32_t stage = stage_peek(a.site);
  const uint4 mine = FIN ? finalize_frag(a, token, frag, rows, wts) : ld_v4(a.partial + row_elem(token, frag));
  push_frag(a.site, stage, token, frag, mine);
  pdl_trigger();
}

template <int W>
__global__ void __launch_bounds__(kRowThreads) consume_kernel(Args a) {
  __shared__ RowSmem rs;
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kRowThreads + threadIdx.x;
  pdl_trigger();
  float gam[8];
  unpack8(ld_v4(a.gamma + static_cast<size_t>(frag) * kVec), gam);
  const float inv_s = a.fp8_out != nullptr ? inv_scale_of(*a.fp8_scale) : 1.f;
  row_init(rs);
  pdl_wait();
  const Hop h = hop_enter(a.site);
  const uint4 res = ld_v4(a.residual_in + row_elem(token, frag));
  float acc[8];
  const bool ok = pull_frag<W>(a.site, h.stage, token, frag, acc);
  float pre[8];
  const uint4 res_out = residual_add(acc, res, pre);
  const float total = row_sumsq_fold(sumsq8(pre), rs);
  uint4 norm = norm_apply(pre, inv_rms(total, a.eps), gam);
  if (!ok) norm = make_uint4(kNanPair, kNanPair, kNanPair, kNanPair);
  *reinterpret_cast<uint4*>(a.residual_out + row_elem(token, frag)) = res_out;
  if (a.norm_out != nullptr) *reinterpret_cast<uint4*>(a.norm_out + row_elem(token, frag)) = norm;
  if (a.fp8_out != nullptr) *reinterpret_cast<uint2*>(a.fp8_out + row_elem(token, frag)) = quant_fp8(norm, inv_s);
  rearm_frag<W>(a.site, h.stage, token, frag);
  hop_leave(a.site, h, gridDim.x * gridDim.y);
}

template <typename K>
int launch(K kernel, const Args& a, cudaStream_t stream, bool cluster, int ctas = kRowCtas) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.m, ctas, 1);
  cfg.blockDim = dim3(kRowThreads, 1, 1);
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  int n = 0;
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  if (cluster) {
    attrs[n].id = cudaLaunchAttributeClusterDimension;
    attrs[n].val.clusterDim.x = 1;
    attrs[n].val.clusterDim.y = kRowCtas;
    attrs[n].val.clusterDim.z = 1;
    ++n;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = n;
  return static_cast<int>(cudaLaunchKernelEx(&cfg, kernel, a));
}

}  // namespace

// which: 0 publish, 1 publish_fin, 2 consume.
extern "C" int q35hop_launch(int which, const Args* args, void* stream) {
  const Args& a = *args;
  if (a.m < 1 || a.m > a.site.m_cap || a.top_k > kMaxTopK) return -2;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (which) {
    case 0: return launch(publish_kernel<false>, a, s, false);
    case 1: return launch(publish_fin_quad_kernel, a, s, false, kQuadCtas);
    case 2:
      switch (a.site.world) {
        case 2: return launch(consume_kernel<2>, a, s, true);
        case 4: return launch(consume_kernel<4>, a, s, true);
        case 8: return launch(consume_kernel<8>, a, s, true);
        default: return -3;
      }
    default: return -1;
  }
}

extern "C" int q35hop_args_size() { return static_cast<int>(sizeof(Args)); }
