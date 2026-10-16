// WS-1 K1: one CTA owns one (batch row, value head), four row-partitioned warps.
// Every cross-warp exchange has its own slab. No slab is reused in an invocation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace {
constexpr int NW = 4, NR = 4, D = 128;

__device__ __forceinline__ uint4 load128(const void* p) {
  uint4 x;
  // Volatile inline PTX keeps the read independent of the subsequently loaded
  // count. The only control dependency surrounding U/W reads is valid state_idx.
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(p) : "memory");
  return x;
}

__device__ __forceinline__ float ex(float x) {
  float y;
  x = __fmul_rn(x, 0x1.715476p+0f);
  asm("ex2.approx.f32 %0,%1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float x;
  asm("div.full.f32 %0,%1,%2;" : "=f"(x) : "f"(a), "f"(b));
  return x;
}
__device__ __forceinline__ float sq(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float half_at(uint4 p, int i) {
  unsigned z = i < 2 ? p.x : i < 4 ? p.y : i < 6 ? p.z : p.w;
  return __half2float(__ushort_as_half(static_cast<unsigned short>(z >> (16 * (i & 1)))));
}
__device__ __forceinline__ void bf8(uint4 p, float (&v)[8]) {
  unsigned z[4] = {p.x, p.y, p.z, p.w};
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    v[i] = __uint_as_float((z[i / 2] >> (16 * (i & 1))) << 16);
}
__device__ __forceinline__ uint4 pack_half(const float (&v)[8]) {
  unsigned z[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    z[i] = unsigned(__half_as_ushort(__float2half_rn(v[2*i]))) |
           (unsigned(__half_as_ushort(__float2half_rn(v[2*i+1]))) << 16);
  return make_uint4(z[0], z[1], z[2], z[3]);
}
__device__ __forceinline__ uint4 pack_bf(const float (&v)[8]) {
  unsigned z[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    z[i] = unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(v[2*i]))) |
           (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(v[2*i+1]))) << 16);
  return make_uint4(z[0], z[1], z[2], z[3]);
}
template<int Width>
__device__ __forceinline__ float sx(float x, int delta) {
  float y;
  // Every call has a full active warp (including the warp-0-only branch).
  // Keep shfl.sync's participant join, without NVCC's extra wrapper collectives.
  asm volatile("shfl.sync.bfly.b32 %0,%1,%2,%3,0xffffffff;"
               : "=f"(y) : "f"(x), "r"(delta), "n"(Width == 16 ? 0x100f : 0x1f));
  return y;
}
__device__ __forceinline__ float sum16(float x) {
  #pragma unroll
  for (int d = 8; d; d >>= 1) x += sx<16>(x,d);
  return x;
}
__device__ __forceinline__ float dot(const float (&a)[8], const float (&b)[8]) {
  float p[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) p[i] = __fmul_rn(a[i], b[i]);
  return sum16(((p[0]+p[1])+(p[2]+p[3]))+((p[4]+p[5])+(p[6]+p[7])));
}
__device__ __forceinline__ void gate(float a, float b, float alog, float bias, float& gt, float& beta) {
  float x = a + bias;
  // Triton tl.log lowers to __nv_logf in the qualified service image. Do not
  // replace logf with __logf or compile with --use_fast_math.
  float softplus = x <= 20.f ? logf(1.f + ex(x)) : x;
  gt = ex(__fmul_rn(-ex(alog), softplus));
  beta = __bfloat162float(__float2bfloat16_rn(div_full(1.f, 1.f + ex(-b))));
}

template<bool INVALIDATE>
__global__ __launch_bounds__(128, 4) void factored_step_r32_kernel(
    const __nv_bfloat16* mixed, const __nv_bfloat16* ag, const __nv_bfloat16* bg,
    const float* alog, const float* bias, const float* vb,
    float* fa, __half* fu, __half* fw, int* fc, int* stale,
    const void* idx, bool idx64, __nv_bfloat16* out, int* pv,
    int H, int HV, int64_t sm, int64_t sa, int64_t sb, int64_t si, float scale, float eps) {
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int col = (lane & 15) * 8, parity = lane >> 4;
  int n = blockIdx.x / HV, hv = blockIdx.x % HV, h = hv / (HV / H);
  int64_t slot = idx64 ? static_cast<const int64_t*>(idx)[n * si] :
                        static_cast<const int*>(idx)[n * si];
  if constexpr (INVALIDATE) {
    if (tid == 0 && hv == 0) pv[slot < 0 ? 0 : slot] = 0;
  }
  if (slot < 0) {
    if (tid < 16) reinterpret_cast<uint4*>(out + (int64_t(n) * HV + hv) * D)[tid] = make_uint4(0,0,0,0);
    // Only the first 16 uint4 vectors belong to this 128-element output.
    return;
  }
  int64_t sh = slot * HV + hv;
  const __nv_bfloat16* inp = mixed + n * sm;
  // All physical factor rows are loaded before count-dependent selections.
  uint4 uraw[NR], wraw[NR];
  #pragma unroll
  for (int j = 0; j < NR; ++j) {
    int row = 2 * (NW * j + warp) + parity;
    uraw[j] = load128(fu + sh * 32 * D + row * D + col);
    wraw[j] = load128(fw + sh * 32 * D + row * D + col);
  }
  uint4 qr = {}, kr = {}, vr = {};
  float4 ar0 = {}, ar1 = {}, vb0 = {}, vb1 = {};
  float av = 0.f, bv = 0.f, al = 0.f, db = 0.f;
  // Warp 0 owns input normalization, gate and sink. Publication below uses
  // a distinct slab and one CTA join; all factor reads remain unconditional.
  if (warp == 0) {
    qr = load128(inp + h * D + col);
    kr = load128(inp + H * D + h * D + col);
    vr = load128(inp + 2 * H * D + hv * D + col);
    ar0 = *reinterpret_cast<const float4*>(fa + sh * D + col);
    ar1 = *reinterpret_cast<const float4*>(fa + sh * D + col + 4);
    vb0 = *reinterpret_cast<const float4*>(vb + hv * D + col);
    vb1 = *reinterpret_cast<const float4*>(vb + hv * D + col + 4);
    av = __bfloat162float(ag[n * sa + hv]);
    bv = __bfloat162float(bg[n * sb + hv]);
    al = alog[hv]; db = bias[hv];
  }
  int cnt = fc[sh];
  float U[NR][8];
  #pragma unroll
  for (int j = 0; j < NR; ++j) {
    int row = 2 * (NW * j + warp) + parity;
    #pragma unroll
    for (int k = 0; k < 8; ++k) U[j][k] = row < cnt ? half_at(uraw[j], k) : 0.f;
  }
  __shared__ float q_s[D], k_s[D], v_s[D], vb_s[D], gb_s[3];
  float kn[8], qn[8], gt = 0.f, beta = 0.f, sink = 0.f;
  if (warp == 0) {
    float q[8];
    bf8(qr, q); bf8(kr, kn);
    float qnorm = sq(dot(q,q) + 1.e-6f), knorm = sq(dot(kn,kn) + 1.e-6f);
    #pragma unroll
    for (int k = 0; k < 8; ++k) { qn[k] = div_full(q[k], qnorm) * scale; kn[k] = div_full(kn[k], knorm); }
    gate(av, bv, al, db, gt, beta);
    float avec[8] = {ar0.x,ar0.y,ar0.z,ar0.w,ar1.x,ar1.y,ar1.z,ar1.w};
    float ka = dot(kn, avec);
    #pragma unroll
    for (int k = 0; k < 8; ++k) avec[k] = gt * (avec[k] - beta * kn[k] * ka) + beta * kn[k];
    sink = dot(avec, qn);
    if (lane < 16) {
      *reinterpret_cast<float4*>(fa + sh * D + col) = make_float4(avec[0],avec[1],avec[2],avec[3]);
      *reinterpret_cast<float4*>(fa + sh * D + col + 4) = make_float4(avec[4],avec[5],avec[6],avec[7]);
    }
    float vpub[8]; bf8(vr,vpub);
    float vbpub[8] = {vb0.x,vb0.y,vb0.z,vb0.w,vb1.x,vb1.y,vb1.z,vb1.w};
    if (lane < 16) {
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        q_s[col+k]=qn[k]; k_s[col+k]=kn[k]; v_s[col+k]=vpub[k]; vb_s[col+k]=vbpub[k];
      }
    }
    if (lane == 0) { gb_s[0]=gt; gb_s[1]=beta; gb_s[2]=sink; }
  }
  __syncthreads();
  #pragma unroll
  for (int k = 0; k < 8; ++k) { qn[k]=q_s[col+k]; kn[k]=k_s[col+k]; }
  gt=gb_s[0]; beta=gb_s[1]; sink=gb_s[2];
  float c[NR], cq[NR], partial[8] = {};
  #pragma unroll
  for (int j = 0; j < NR; ++j) {
    c[j] = dot(U[j], kn);
    cq[j] = dot(U[j], qn);
    #pragma unroll
    for (int k = 0; k < 8; ++k) partial[k] += U[j][k] * c[j];
  }
  __shared__ float kp0[NW][D], kp1[NW][D], mv_s[NW][D], wq_s[NW][D], cd_s[NW];
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    float x = partial[k] + sx<32>(partial[k],16);
    if (lane < 16) kp0[warp][col+k] = x;
  }
  __syncthreads();
  float kp[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k)
    kp[k] = kn[k] - ((kp0[0][col+k]+kp0[1][col+k])+(kp0[2][col+k]+kp0[3][col+k]));
  float nrm2 = dot(kp,kp);
  if (nrm2 < .25f) {
    float p2[8] = {};
    #pragma unroll
    for (int j = 0; j < NR; ++j) {
      float c2 = dot(U[j],kp); c[j] += c2;
      #pragma unroll
      for (int k = 0; k < 8; ++k) p2[k] += U[j][k] * c2;
    }
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      float x = p2[k] + sx<32>(p2[k],16);
      if (lane < 16) kp1[warp][col+k] = x;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 8; ++k)
      kp[k] -= (kp1[0][col+k]+kp1[1][col+k])+(kp1[2][col+k]+kp1[3][col+k]);
    nrm2 = dot(kp,kp);
  }
  float nrm = sq(nrm2), khat[8];
  bool keep = nrm > eps;
  #pragma unroll
  for (int k = 0; k < 8; ++k) khat[k] = keep ? div_full(kp[k], fmaxf(nrm,eps)) : 0.f;
  float clast = keep ? nrm : 0.f, kq = dot(khat,qn);
  float mv[8] = {}, wq[8] = {}, cf[NR], cd = 0.f;
  #pragma unroll
  for (int j = 0; j < NR; ++j) {
    int row = 2 * (NW * j + warp) + parity;
    cf[j] = row == cnt ? clast : c[j];
    cq[j] += row == cnt ? kq : 0.f;
    cd += cf[j] * cq[j];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      float w = row < cnt ? half_at(wraw[j],k) : 0.f;
      mv[k] += w * c[j]; wq[k] += w * cq[j];
    }
  }
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    float m = mv[k] + sx<32>(mv[k],16);
    float w = wq[k] + sx<32>(wq[k],16);
    if (lane < 16) { mv_s[warp][col+k]=m; wq_s[warp][col+k]=w; }
  }
  cd += sx<32>(cd,16);
  if (lane == 0) cd_s[warp]=cd;
  __syncthreads();
  float cd_all = (cd_s[0]+cd_s[1])+(cd_s[2]+cd_s[3]);
  float delta[8], oval[8];
  float vbv[8], v[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) { vbv[k]=vb_s[col+k]; v[k]=v_s[col+k]; }
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    float m = (mv_s[0][col+k]+mv_s[1][col+k])+(mv_s[2][col+k]+mv_s[3][col+k]);
    float w = (wq_s[0][col+k]+wq_s[1][col+k])+(wq_s[2][col+k]+wq_s[3][col+k]);
    delta[k] = beta * ((v[k]-vbv[k])-gt*m);
    oval[k] = (vbv[k]*sink+gt*w)+delta[k]*cd_all;
  }
  #pragma unroll
  for (int j = 0; j < NR; ++j) {
    int row = 2 * (NW * j + warp) + parity;
    if (row <= cnt) {
      float nw[8];
      #pragma unroll
      for (int k = 0; k < 8; ++k) nw[k] = gt * (row < cnt ? half_at(wraw[j],k) : 0.f) + cf[j]*delta[k];
      *reinterpret_cast<uint4*>(fw + sh*32*D + row*D + col) = pack_half(nw);
    }
    if (row == cnt && cnt < 32)
      *reinterpret_cast<uint4*>(fu + sh*32*D + row*D + col) = pack_half(khat);
  }
  if (warp == 0 && lane < 16)
    *reinterpret_cast<uint4*>(out + (int64_t(n)*HV+hv)*D + col) = pack_bf(oval);
  if (tid == 0) { fc[sh]=cnt+1; stale[slot]=1; }
}

__global__ void gates_kernel(const __nv_bfloat16* a,const __nv_bfloat16* b,
                             const float* al,const float* db,float* out,int n,int hv) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) gate(__bfloat162float(a[i]),__bfloat162float(b[i]),al[i%hv],db[i%hv],out[2*i],out[2*i+1]);
}
} // namespace

void factored_step_r32(torch::Tensor mixed,torch::Tensor ag,torch::Tensor bg,
    torch::Tensor alog,torch::Tensor bias,torch::Tensor vb,torch::Tensor fa,
    torch::Tensor fu,torch::Tensor fw,torch::Tensor fc,torch::Tensor stale,
    torch::Tensor idx,torch::Tensor out,int H,double scale,double eps,
    c10::optional<torch::Tensor> pv,bool invalidate) {
  const c10::cuda::CUDAGuard guard(mixed.device());
  int HV=fu.size(1), B=mixed.size(0);
  TORCH_CHECK(mixed.scalar_type()==at::kBFloat16 && ag.scalar_type()==at::kBFloat16 && bg.scalar_type()==at::kBFloat16);
  TORCH_CHECK(fu.scalar_type()==at::kHalf && fw.scalar_type()==at::kHalf);
  TORCH_CHECK(fa.scalar_type()==at::kFloat && vb.scalar_type()==at::kFloat && alog.scalar_type()==at::kFloat && bias.scalar_type()==at::kFloat);
  TORCH_CHECK(fc.scalar_type()==at::kInt && stale.scalar_type()==at::kInt && out.scalar_type()==at::kBFloat16);
  TORCH_CHECK(idx.scalar_type()==at::kInt || idx.scalar_type()==at::kLong);
  TORCH_CHECK(fu.size(2)==32 && fu.size(3)==128 && fw.size(3)==128 && HV%H==0);
  TORCH_CHECK(!invalidate || (pv.has_value() && pv->scalar_type()==at::kInt && pv->numel()>=fu.size(0)));
  auto stream=at::cuda::getCurrentCUDAStream();
  auto launch=[&](auto flag) {
    factored_step_r32_kernel<decltype(flag)::value><<<B*HV,128,0,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(mixed.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(ag.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(bg.data_ptr()),alog.data_ptr<float>(),bias.data_ptr<float>(),vb.data_ptr<float>(),
      fa.data_ptr<float>(),reinterpret_cast<__half*>(fu.data_ptr()),reinterpret_cast<__half*>(fw.data_ptr()),
      fc.data_ptr<int>(),stale.data_ptr<int>(),idx.data_ptr(),idx.scalar_type()==at::kLong,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),invalidate?pv->data_ptr<int>():nullptr,
      H,HV,mixed.stride(0),ag.stride(0),bg.stride(0),idx.stride(0),float(scale),float(eps));
  };
  if(invalidate) launch(std::true_type{}); else launch(std::false_type{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor gates(torch::Tensor a,torch::Tensor b,torch::Tensor al,torch::Tensor db) {
  auto out=torch::empty({a.numel(),2},al.options());
  gates_kernel<<<(a.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
    al.data_ptr<float>(),db.data_ptr<float>(),out.data_ptr<float>(),a.numel(),a.size(-1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("factored_step_r32",&factored_step_r32); m.def("gates",&gates); }
