// Single-GPU timing of two FP8 GEMV cores for the Qwen3.5 GatedDeltaNet out_proj
// (static per-tensor e4m3, M <= 8 tokens, N = 4096, K = 1024 at TP8 or 2048 at TP4).
//   rows32:  CTA c owns output rows [32c, 32c + 32) over the full K, warps split K (the core of
//            the production q35hop out_proj kernel).
//   cluster: 16 units x 8-CTA clusters; a unit owns 256 rows, cluster ranks split K and send
//            their partials to the token's owner rank with st.async (the fused attention o_proj core).
// Both load their weight fragments before griddepcontrol.wait. PUSH = 1 also writes each token's
// result as 16-B fragments to a separate buffer, standing in for the hop mailbox store.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kHidden = 4096;
constexpr int kCluster = 8;

struct T1Args {
  const uint4* w;
  const uint8_t* x;  // [M, K] e4m3
  const float* sx;
  const float* sw;
  __nv_bfloat16* out;  // [M, 4096] bf16, or null
  uint4* box;          // [M, 512] 16-B fragments, or null
  int m;
};

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg_nc_v4(const uint4* q) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(q));
  return r;
}

__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint4 pack8(const float (&v)[8]) {
  return make_uint4(bf2(v[0], v[1]), bf2(v[2], v[3]), bf2(v[4], v[5]), bf2(v[6], v[7]));
}

__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}" ::"r"(
          smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void st_async_v4(uint32_t dst, float a, float b, float c, float d, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
               "f"(a), "f"(b), "f"(c), "f"(d), "r"(bar)
               : "memory");
}

// Inbox slot s of a 256-row tile -> row: warp w = s/32 holds rows 32w + {g, g+8, g+16, g+24}.
__device__ __forceinline__ int slot_row(int s) { return 32 * (s >> 5) + ((s >> 2) & 7) + 8 * (s & 3); }

template <int S, int PUSH>
__global__ void __launch_bounds__(256, 1) rows32_kernel(T1Args o) {
  constexpr int kK = 8 * S * 32;
  __shared__ __align__(16) float red[8][32][8];
  const int cta = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  uint4 a[S][2];
  {
    const uint4* wb = o.w + (static_cast<size_t>(cta) * 8 + warp) * (S * 2 * 32) + lane;
#pragma unroll
    for (int st = 0; st < S; ++st)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[st][b] = ldg_nc_v4(wb + (st * 2 + b) * 32);
  }
  const float alpha = __fmul_rn(*o.sx, *o.sw);
  pdl_wait();
  const int m = o.m;
  const bool tok_ok = g < m;
  const uint8_t* xr = o.x + static_cast<size_t>(g) * kK + warp * S * 32 + 4 * t;
  float ch[2][2][4] = {};
#pragma unroll
  for (int st = 0; st < S; ++st) {
    const uint32_t b0 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + st * 32) : 0u;
    const uint32_t b1 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + st * 32 + 16) : 0u;
    mma_e4m3(ch[0][st & 1], a[st][0], b0, b1);
    mma_e4m3(ch[1][st & 1], a[st][1], b0, b1);
  }
  {
    float4* dst = reinterpret_cast<float4*>(&red[warp][lane][0]);
    dst[0] = make_float4(__fadd_rn(ch[0][0][0], ch[0][1][0]), __fadd_rn(ch[0][0][1], ch[0][1][1]),
                         __fadd_rn(ch[0][0][2], ch[0][1][2]), __fadd_rn(ch[0][0][3], ch[0][1][3]));
    dst[1] = make_float4(__fadd_rn(ch[1][0][0], ch[1][1][0]), __fadd_rn(ch[1][0][1], ch[1][1][1]),
                         __fadd_rn(ch[1][0][2], ch[1][1][2]), __fadd_rn(ch[1][0][3], ch[1][1][3]));
  }
  __syncthreads();
  const int tok = tid >> 2, j = tid & 3;
  if (tok < m) {
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int r = 8 * j + q;
      const int l = (r % 8) * 4 + tok / 2;
      const int e = (r / 16) * 4 + ((r % 16) / 8) * 2 + (tok % 2);
      float x = red[0][l][e];
#pragma unroll
      for (int w = 1; w < 8; ++w) x = __fadd_rn(x, red[w][l][e]);
      v[q] = __fmul_rn(x, alpha);
    }
    const uint4 frag = pack8(v);
    const int fr = cta * 4 + j;
    if (PUSH) o.box[tok * (kHidden / 8) + fr] = frag;
    if (o.out != nullptr) *reinterpret_cast<uint4*>(o.out + tok * kHidden + fr * 8) = frag;
  }
  pdl_trigger();
}

// grid = 16 units x 8 cluster ranks; NQ = K / 256 (k-steps of 32 per rank).
template <int NQ, int PUSH>
__global__ void __launch_bounds__(256, 1) cluster_kernel(T1Args p) {
  constexpr int kSteps = NQ;
  constexpr int kK = NQ * 256;
  constexpr int kWarpU4 = kSteps * 2 * 32;
  __shared__ __align__(16) float inbox[kCluster][256];
  __shared__ __align__(16) __nv_bfloat16 tile[256];
  __shared__ __align__(8) uint64_t ibar;
  const int rank = static_cast<int>(cluster_ctarank());
  const int unit = blockIdx.x / kCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  if (tid == 0) {
    mbar_init(&ibar, 1);
    mbar_init_fence();
  }
  __syncthreads();
  cluster_arrive_relaxed();
  uint4 a[kSteps][2];
  {
    const uint4* wb = p.w + static_cast<size_t>(unit * kCluster + rank) * (8 * kWarpU4) + warp * kWarpU4 + lane;
#pragma unroll
    for (int s = 0; s < kSteps; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b) a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);
  pdl_wait();
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  const bool tok_ok = g < m;
  const uint8_t* xr = p.x + static_cast<size_t>(g) * kK + rank * (kSteps * 32) + 4 * t;
  constexpr int kCh = kSteps < 4 ? kSteps : 4;
  float ch[2][kCh][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int c = 0; c < kCh; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) ch[b][c][i] = 0.f;
#pragma unroll
  for (int s = 0; s < kSteps; ++s) {
    const uint32_t b0 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + s * 32) : 0u;
    const uint32_t b1 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + s * 32 + 16) : 0u;
    mma_e4m3(ch[0][s % kCh], a[s][0], b0, b1);
    mma_e4m3(ch[1][s % kCh], a[s][1], b0, b1);
  }
  float acc[2][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float x = ch[b][0][i];
#pragma unroll
      for (int c = 1; c < kCh; ++c) x = __fadd_rn(x, ch[b][c][i]);
      acc[b][i] = x;
    }
  const bool owner = rank < m;
  if (tid == 0 && owner) mbar_expect_tx(&ibar, kCluster * 256 * 4);
  cluster_wait();
  {
    const uint32_t src = smem_u32(&inbox[rank][warp * 32 + g * 4]);
    const uint32_t bar = smem_u32(&ibar);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tok = 2 * t + j;
      if (tok < m) st_async_v4(mapa(src, tok), acc[0][j], acc[0][2 + j], acc[1][j], acc[1][2 + j], mapa(bar, tok));
    }
  }
  if (!owner) {
    pdl_trigger();
    return;
  }
  mbar_wait(&ibar, 0);
  float v = 0.f;
#pragma unroll
  for (int r = 0; r < kCluster; ++r) v = __fadd_rn(v, inbox[r][tid]);
  const __nv_bfloat16 b = __float2bfloat16_rn(__fmul_rn(v, alpha));
  const int row = slot_row(tid);
  if (p.out != nullptr) p.out[static_cast<size_t>(rank) * kHidden + unit * 256 + row] = b;
  if (PUSH) {
    tile[row] = b;
    __syncthreads();
    if (tid < 32) p.box[rank * (kHidden / 8) + unit * 32 + tid] = reinterpret_cast<const uint4*>(tile)[tid];
  }
  pdl_trigger();
}

template <typename K>
int launch(K kernel, int grid, int cluster, const T1Args& a, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  int n = 0;
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  if (cluster > 1) {
    attrs[n].id = cudaLaunchAttributeClusterDimension;
    attrs[n].val.clusterDim.x = cluster;
    attrs[n].val.clusterDim.y = 1;
    attrs[n].val.clusterDim.z = 1;
    ++n;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = n;
  return static_cast<int>(cudaLaunchKernelEx(&cfg, kernel, a));
}

}  // namespace

// core: 0 rows32, 1 cluster; k_steps = K / 256.
extern "C" int t1_oproj(int core, int k_steps, int push, const void* args, void* stream) {
  const T1Args& a = *static_cast<const T1Args*>(args);
  if (a.m < 1 || a.m > 8) return -2;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int key = core * 100 + k_steps * 10 + (push ? 1 : 0);
  switch (key) {
    case 40: return launch(rows32_kernel<4, 0>, kHidden / 32, 1, a, s);
    case 41: return launch(rows32_kernel<4, 1>, kHidden / 32, 1, a, s);
    case 80: return launch(rows32_kernel<8, 0>, kHidden / 32, 1, a, s);
    case 81: return launch(rows32_kernel<8, 1>, kHidden / 32, 1, a, s);
    case 140: return launch(cluster_kernel<4, 0>, 16 * kCluster, kCluster, a, s);
    case 141: return launch(cluster_kernel<4, 1>, 16 * kCluster, kCluster, a, s);
    case 180: return launch(cluster_kernel<8, 0>, 16 * kCluster, kCluster, a, s);
    case 181: return launch(cluster_kernel<8, 1>, 16 * kCluster, kCluster, a, s);
    default: return -3;
  }
}

extern "C" int t1_args_size() { return static_cast<int>(sizeof(T1Args)); }
