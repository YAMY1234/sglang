// SPDX-License-Identifier: Apache-2.0
// FN H2560 NVFP4 prototype: TP1 storage640 or TP4 storage192. The first version uses synchronous stage
// ownership to establish numerical parity before introducing the weight ring.
#include "q38_mk_moe_primitives.cuh"

namespace q38_mk {
#ifndef Q38_MK_I
#define Q38_MK_I 640
#endif
constexpr int H = 2560, I = Q38_MK_I, E = 512, K = 10, Rows = 16, Threads = 256;
static_assert(I == 640 || I == 192, "supported native storage widths");
constexpr int TileI = I == 640 ? 128 : 64;
constexpr int ClusterSize = I / TileI, N1 = 2 * TileI, Nk = TileI / 64;
constexpr int StageBytes = N1 * 144, SfOffset = N1 * 128;
constexpr int Sw2 = TileI == 128 ? 4 : 6; // SW64 / SW32
constexpr int W2Sbo = TileI * 4;

struct Params {
  const uint8_t *x, *xsf, *sf13, *sf2;
  const float *g1, *g1c, *g2;
  const int *ids, *unique_ids, *num_unique;
  __nv_bfloat16* out;  // [M,10,H], native BF16 expert rows before finalize
  int m;
};
struct __align__(1024) Smem {
  alignas(1024) uint8_t stage[StageBytes];
  alignas(1024) uint8_t x[10][2048]; // SW128, 16 rows of 128 bytes per K256
  alignas(1024) uint8_t h[ClusterSize + 7][2048];
  // TCGEN05 M128 reads all128 operand rows even though SFA zeros mask rows16+.
  // Each useful chunk has16 rows; seven trailing chunks back the last dead-row
  // reads. They are zeroed with h and do not participate in DSMEM exchange.
  uint32_t xsf[Rows][40], hsf[Rows][I / 64];
  float acc[Rows][N1 + 4];
  uint64_t loaded, done;
  uint32_t tmem;
  int slots[Rows];
};
constexpr int SmemBytes = sizeof(Smem) + 1024;
static_assert(SmemBytes <= 233472, "FN SMEM bound");
#ifdef Q38_MK_SMEM_BYTES
static_assert(SmemBytes == Q38_MK_SMEM_BYTES, "FN launcher layout");
#endif

__device__ __forceinline__ void cluster_sync() { cluster_arrive(); cluster_wait(); }
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ void stage_scales(Smem& s, uint32_t tm, int nk,
                                             int sf_half_offset, bool single = false) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp < 4) {
    const uint32_t la = (uint32_t)(warp * 32) << 16;
    for (int k = 0; k < nk; ++k) {
      const uint8_t* a = s.stage + SfOffset + k * 512 + lane * 16;
      const uint4 u = *reinterpret_cast<const uint4*>(a);
      if (single) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(tm + la + 384 + k * 4), "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w) : "memory");
      } else {
        const uint4 v = *reinterpret_cast<const uint4*>(a + sf_half_offset);
        const uint32_t vals[8] = {u.x,u.y,u.z,u.w,v.x,v.y,v.z,v.w};
        tmem_st_x8(tm + la + 384 + k * 8, vals);
      }
    }
    tmem_st_wait();
    tc_fence_before();
  }
}
__device__ __forceinline__ void stage_x_scales(Smem& s, uint32_t tm, int st) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp < 4) {
    uint32_t vals[16];
    for (int i = 0; i < 16; ++i)
      vals[i] = lane < Rows && i % 4 == 0 ? s.xsf[lane][st * 4 + i / 4] : 0u;
    tmem_st_x16(tm + ((uint32_t)(warp * 32) << 16) + 256, vals);
    tmem_st_wait();
    tc_fence_before();
  }
}
__device__ __forceinline__ void stage_h_scales(Smem& s, uint32_t tm, int chunk) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp < 4) {
    uint32_t vals[8] = {};
    if (lane < Rows) { vals[0] = s.hsf[lane][chunk * Nk]; if (Nk == 2) vals[4] = s.hsf[lane][chunk * Nk + 1]; }
    tmem_st_x8(tm + ((uint32_t)(warp * 32) << 16) + 256, vals);
    tmem_st_wait();
    tc_fence_before();
  }
}

extern "C" __global__ __cluster_dims__(ClusterSize,1,1) __launch_bounds__(Threads,1)
void q38_mk_moe(const __grid_constant__ CUtensorMap w13,
                const __grid_constant__ CUtensorMap w2, const Params p) {
  extern __shared__ __align__(1024) uint8_t raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int rank = cluster_rank(), cluster = blockIdx.x / ClusterSize;
  if (tid == 0) { prefetch_tmap(&w13); prefetch_tmap(&w2); }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (cluster >= *p.num_unique) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }
  // Relinquishing is permanent for this CTA: allocate once, reuse across items.
  if (warp == 4) { tmem_alloc(&s.tmem,512); tmem_relinquish(); tc_fence_before(); }
  __syncthreads();
  // Cluster CTAs take identical loop bounds. Only local barriers exist.
  for (int item = cluster; item < *p.num_unique; item += gridDim.x / ClusterSize) {
    const int expert = p.unique_ids[item];
    if (tid == 0) {
      mbar_init(&s.loaded,1); mbar_init(&s.done,1); fence_mbar_init();
    }
    for (int j = tid; j < Rows * 10 * 128 / 16; j += Threads) {
      const int row = j / 80, rem = j % 80, st = rem / 8, q = rem % 8;
      uint4 v = make_uint4(0,0,0,0);
      if (row < p.m) v = *reinterpret_cast<const uint4*>(p.x + row * 1280 + st * 128 + q * 16);
      *reinterpret_cast<uint4*>(s.x[st] + (row / 8) * 1024 + (row % 8) * 128 + ((q ^ (row % 8)) * 16)) = v;
    }
    for (int j = tid; j < Rows * 40; j += Threads) {
      const int row = j / 40, kb = j % 40;
      s.xsf[row][kb] = row < p.m ? *reinterpret_cast<const uint32_t*>(p.xsf + row * 160 + kb * 4) : 0;
    }
    if (tid < Rows) {
      int slot = -1;
      if (tid < p.m) for (int k = 0; k < K; ++k) if (p.ids[tid * K + k] == expert) slot = k;
      s.slots[tid] = slot;
    }
    for (int j = tid; j < sizeof(s.h) / 16; j += Threads)
      reinterpret_cast<uint4*>(s.h)[j] = make_uint4(0,0,0,0);
    __syncthreads();
    cluster_sync();
    fence_proxy_async_smem();
    const uint32_t tm = s.tmem, idesc = make_idesc_nvf4(128,N1), idesc2 = make_idesc_nvf4(128,256);
    // Increasing K256 stages, each four K64 MMAs, preserves native summation.
    for (int st = 0; st < 10; ++st) {
      if (tid == 0) {
        mbar_expect_tx(&s.loaded,StageBytes);
        const int row = expert * (2 * I) + rank * N1;
        tma_load_2d(s.stage,&w13,&s.loaded,st * 128,row);
        if (N1 == 256) tma_load_2d(s.stage + 16384,&w13,&s.loaded,st * 128,row + 128);
        const uint8_t* sf = p.sf13 + (size_t)expert * (2 * I) * 160;
        bulk_load(s.stage + SfOffset,sf + ((rank * (N1 / 128)) * 40 + st * 4) * 512,2048,&s.loaded);
        if (N1 == 256) bulk_load(s.stage + SfOffset + 2048,sf + ((rank * 2 + 1) * 40 + st * 4) * 512,2048,&s.loaded);
      }
      mbar_wait(&s.loaded, st & 1);
      stage_scales(s,tm,4,2048,N1 == 128); stage_x_scales(s,tm,st);
      __syncthreads();
      tc_fence_after();
      if (tid == 0) {
        for (int k = 0; k < 4; ++k)
          umma_nvf4(tm,make_sdesc(smem_u32(s.x[st]) + k * 32,0,1024,2),
                    make_sdesc(smem_u32(s.stage) + k * 32,0,1024,2),idesc,
                    tm + 256 + k * 4,tm + 384 + k * (N1 / 32),(st | k) != 0);
        tc_commit(&s.done);
      }
      mbar_wait(&s.done,st & 1); tc_fence_after(); __syncthreads();
    }
    if (warp == 0) {
      for (int c = 0; c < N1 / 16; ++c) {
        uint32_t vals[16]; tmem_ld_32x32b_x16(tm + c * 16,vals); tmem_ld_wait();
        if (lane < Rows) for (int j = 0; j < 16; ++j) s.acc[lane][c * 16 + j] = __uint_as_float(vals[j]);
      }
    }
    __syncthreads();
    const float ag = p.g1[expert], sc = p.g1c[expert], a2 = p.g2[expert];
    // One adjacent lane pair owns a quantization block of sixteen intermediates.
    for (int task = tid; task < Rows * (TileI / 8); task += Threads) {
      const int row = task / (TileI / 8), blk = (task / 2) % (TileI / 16), hf = task % 2;
      const float4* ar = reinterpret_cast<const float4*>(s.acc[row] + 32 * blk + 4 * hf);
      const float4 ue = ar[0], ge = ar[2], uo = ar[4], go = ar[6];
      const float ga[8] = {ge.x,go.x,ge.y,go.y,ge.z,go.z,ge.w,go.w};
      const float ua[8] = {ue.x,uo.x,ue.y,uo.y,ue.z,uo.z,ue.w,uo.w};
      float h[8], amax = 0; const float al2e = mul_ftz(ag,1.4426950216293334961f);
      for (int j = 0; j < 8; ++j) {
        const float sg = rcp_approx(ex2_approx(-(ga[j] * al2e)) + 1.f);
        h[j] = fmaf(ua[j],ag,0.f) * (ga[j] * sg); amax = fmaxf(amax,fabsf(h[j]));
      }
      amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,1));
      const uint32_t sf = cvt_e4m3(fmaxf(0.f,fabsf(amax * sc)) * 0.16666667163372039795f);
      const float sff = e4m3_to_f32(sf), osc = sff == 0.f ? 0.f : rcp_approx(sff);
      float q[8]; for (int j = 0; j < 8; ++j) q[j] = mul_ftz(h[j],osc) * sc;
      const int off = (row / 8) * 1024 + (row % 8) * 128 + (((blk / 2) ^ (row % 8)) * 16) + (blk % 2) * 8 + hf * 4;
      *reinterpret_cast<uint32_t*>(s.h[rank] + off) = cvt_e2m1x8(q);
      if (hf == 0) reinterpret_cast<uint8_t*>(&s.hsf[row][rank * Nk + blk / 4])[blk % 4] = sf;
    }
    __syncthreads(); cluster_sync();
    // Each peer reads a disjoint slice. Barriers keep DSMEM sources alive.
    for (int peer = 0; peer < ClusterSize; ++peer) if (peer != rank) {
      // Convert shared-cluster addresses explicitly; generic pointers must retain
      // the shared address-space tag rather than treating mapa as global memory.
      for (int j = tid; j < 2048 / 16; j += Threads) {
        uint4 v;
        const uint32_t addr = mapa(smem_u32(s.h[peer] + j * 16),peer);
        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(addr) : "memory");
        reinterpret_cast<uint4*>(s.h[peer])[j] = v;
      }
      for (int j = tid; j < Rows * Nk; j += Threads) {
        uint32_t v; const int row = j / Nk, k = peer * Nk + j % Nk;
        const uint32_t addr = mapa(smem_u32(&s.hsf[row][k]),peer);
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
        s.hsf[row][k] = v;
      }
    }
    __syncthreads(); cluster_sync(); fence_proxy_async_smem();
    // Output tiles are striped across the cluster; consume every stored K in order.
    for (int tile = rank; tile < H / 256; tile += ClusterSize) {
      for (int chunk = 0; chunk < ClusterSize; ++chunk) {
        const int step = 10 + ((tile - rank) / ClusterSize) * ClusterSize + chunk;
        if (tid == 0) {
          mbar_expect_tx(&s.loaded,256 * (TileI / 2) + 1024 * Nk);
          const int row = tile * 256;
          tma_load_2d(s.stage,&w2,&s.loaded,chunk * (TileI / 2),expert * H + row);
          const uint8_t* sf = p.sf2 + (size_t)expert * H * (I / 16);
          bulk_load(s.stage + SfOffset,sf + ((row / 128) * (I / 64) + chunk * Nk) * 512,512 * Nk,&s.loaded);
          bulk_load(s.stage + SfOffset + 512 * Nk,sf + ((row / 128 + 1) * (I / 64) + chunk * Nk) * 512,512 * Nk,&s.loaded);
        }
        mbar_wait(&s.loaded,step & 1);
        stage_scales(s,tm,Nk,512 * Nk); stage_h_scales(s,tm,chunk);
        __syncthreads(); tc_fence_after();
        if (tid == 0) {
          for (int k = 0; k < Nk; ++k)
            umma_nvf4(tm,make_sdesc(smem_u32(s.h[chunk]) + k * 32,0,1024,2),
                      make_sdesc(smem_u32(s.stage) + k * 32,0,W2Sbo,Sw2),idesc2,
                      tm + 256 + k * 4,tm + 384 + k * 8,(chunk | k) != 0);
          tc_commit(&s.done);
        }
        mbar_wait(&s.done,step & 1); tc_fence_after(); __syncthreads();
      }
      if (warp == 0) {
        for (int c = 0; c < 16; ++c) {
          uint32_t vals[16]; tmem_ld_32x32b_x16(tm + c * 16,vals); tmem_ld_wait();
          if (lane < p.m && s.slots[lane] >= 0) {
            for (int j = 0; j < 16; ++j) {
              const int physical = c * 16 + j;
              const int logical = (physical & ~31) + (physical % 8) * 4 + (physical & 31) / 8;
              const int col = tile * 256 + logical;
              p.out[((size_t)lane * K + s.slots[lane]) * H + col] = __float2bfloat16_rn(__uint_as_float(vals[j]) * a2);
            }
          }
        }
      }
      __syncthreads();
    }
    tc_fence_before(); cluster_sync();
    __syncthreads(); cluster_sync();
  }
  tc_fence_before();
  if (warp == 4) tmem_dealloc(s.tmem,512);
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
} // namespace q38_mk
