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
#include <cmath>

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
__device__ __forceinline__ float sum16(float x) {
  #pragma unroll
  for (int d = 8; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d, 16);
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

// Match Triton's tl.load(dt_bias + i_hv).to(tl.float32): convert the
// stored element before any gate arithmetic, without a parameter copy.
__device__ __forceinline__ float bias_float(float x) { return x; }
__device__ __forceinline__ float bias_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float bias_float(__half x) { return __half2float(x); }

// P4: only warp 0 enters; both half warps hold the same 128 outputs.
// Preserve the materialized K1 bf16 rounding before the RMS calculation.
template<bool SHARED = false>
__device__ __forceinline__ void norm_gate(float (&o)[8], int col,
    const __nv_bfloat16* z, const __nv_bfloat16* weight, float eps) {
  bf8(pack_bf(o), o);
  // Qualified Triton ROWS_PER_BLOCK=1: four consecutive values per
  // lane, square accumulation p1^2 -> fma(p0) -> fma(p2) -> fma(p3),
  // then xor 16/8/4/2/1. Remap K1's eight values per half-warp lane.
  float lo = __fmaf_rn(o[3], o[3], __fmaf_rn(o[2], o[2],
                 __fmaf_rn(o[0], o[0], __fmul_rn(o[1], o[1]))));
  float hi = __fmaf_rn(o[7], o[7], __fmaf_rn(o[6], o[6],
                 __fmaf_rn(o[4], o[4], __fmul_rn(o[5], o[5]))));
  int lane = threadIdx.x & 31;
  // Permute Triton's lane bits instead of moving its partials: the low
  // half warp owns even four-element groups, the high half owns odd groups.
  float ss = lane < 16 ? lo : hi;
  #pragma unroll
  for (int d = 8; d; d >>= 1)
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffff, ss, d));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffff, ss, 16));
  float v = __fadd_rn(div_full(ss, 128.f), eps), rstd;
  asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(rstd) : "f"(v));
  float zv[8], wv[8];
  if constexpr (SHARED) {
    bf8(*reinterpret_cast<const uint4*>(z + col), zv);
    bf8(*reinterpret_cast<const uint4*>(weight + col), wv);
  } else {
    bf8(load128(z + col), zv); bf8(load128(weight + col), wv);
  }
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    float sig = div_full(1.f, 1.f + ex(-zv[k]));
    o[k] = __fmul_rn(__fmul_rn(__fmul_rn(o[k], rstd), wv[k]), sig);
  }
}

template<typename BiasT, bool INVALIDATE, bool NORM>
__global__ __launch_bounds__(128, 4) void factored_step_r32_kernel(
    const __nv_bfloat16* mixed, const __nv_bfloat16* ag, const __nv_bfloat16* bg,
    const float* alog, const BiasT* bias, const float* vb,
    float* fa, __half* fu, __half* fw, int* fc, int* stale,
    const void* idx, bool idx64, __nv_bfloat16* out, int* pv,
    int H, int HV, int64_t sm, int64_t sa, int64_t sb, int64_t si, float scale, float eps,
    const __nv_bfloat16* z, const __nv_bfloat16* norm_weight, float norm_eps) {
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int col = (lane & 15) * 8, parity = lane >> 4;
  int n = blockIdx.x / HV, hv = blockIdx.x % HV, h = hv / (HV / H);
  int64_t slot = idx64 ? static_cast<const int64_t*>(idx)[n * si] :
                        static_cast<const int*>(idx)[n * si];
  if constexpr (INVALIDATE) {
    if (tid == 0 && hv == 0) pv[slot < 0 ? 0 : slot] = 0;
  }
  if (slot < 0) {
    if constexpr (NORM) {
      if (warp == 0) {
        float zero[8] = {};
        norm_gate(zero, col, z + (int64_t(n)*HV+hv)*D, norm_weight, norm_eps);
        if (lane < 16)
          *reinterpret_cast<uint4*>(out + (int64_t(n)*HV+hv)*D + col) = pack_bf(zero);
      }
    } else {
      if (tid < 16) reinterpret_cast<uint4*>(out + (int64_t(n) * HV + hv) * D)[tid] = make_uint4(0,0,0,0);
    }
    // Only the first 16 uint4 vectors belong to this 128-element output.
    return;
  }
  // Immutable z/weight are staged by warp 0 while K1 performs its recurrence.
  // Only warp 0 reads these slabs. wait_group + syncwarp below publishes all
  // sixteen writers to both half-warp readers, with no new CTA barrier.
  __shared__ __align__(16) __nv_bfloat16 norm_z[D], norm_w[D];
  if constexpr (NORM) {
    if (warp == 0) {
      if (lane < 16) {
        unsigned zs = __cvta_generic_to_shared(norm_z + col);
        unsigned ws = __cvta_generic_to_shared(norm_w + col);
        const __nv_bfloat16* zg = z + (int64_t(n)*HV+hv)*D + col;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(zs), "l"(zg) : "memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(ws), "l"(norm_weight + col) : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
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
  uint4 qr = load128(inp + h * D + col);
  uint4 kr = load128(inp + H * D + h * D + col);
  uint4 vr = load128(inp + 2 * H * D + hv * D + col);
  float4 ar0 = {}, ar1 = {};
  // Only warp 0 reads/writes the sink. Its shuffle reductions join all sink
  // readers before the store; other warps cannot race this in-place update.
  if (warp == 0) {
    ar0 = *reinterpret_cast<const float4*>(fa + sh * D + col);
    ar1 = *reinterpret_cast<const float4*>(fa + sh * D + col + 4);
  }
  float4 vb0 = *reinterpret_cast<const float4*>(vb + hv * D + col);
  float4 vb1 = *reinterpret_cast<const float4*>(vb + hv * D + col + 4);
  float av = __bfloat162float(ag[n * sa + hv]);
  float bv = __bfloat162float(bg[n * sb + hv]);
  float al = alog[hv], db = bias_float(bias[hv]);
  int cnt = fc[sh];
  float U[NR][8];
  #pragma unroll
  for (int j = 0; j < NR; ++j) {
    int row = 2 * (NW * j + warp) + parity;
    #pragma unroll
    for (int k = 0; k < 8; ++k) U[j][k] = row < cnt ? half_at(uraw[j], k) : 0.f;
  }
  float q[8], kn[8], qn[8];
  bf8(qr, q); bf8(kr, kn);
  float qnorm = sq(dot(q,q) + 1.e-6f), knorm = sq(dot(kn,kn) + 1.e-6f);
  #pragma unroll
  for (int k = 0; k < 8; ++k) { qn[k] = div_full(q[k], qnorm) * scale; kn[k] = div_full(kn[k], knorm); }
  float gt, beta;
  gate(av, bv, al, db, gt, beta);
  float sink = 0.f;
  if (warp == 0) {
    float avec[8] = {ar0.x,ar0.y,ar0.z,ar0.w,ar1.x,ar1.y,ar1.z,ar1.w};
    float ka = dot(kn, avec);
    #pragma unroll
    for (int k = 0; k < 8; ++k) avec[k] = gt * (avec[k] - beta * kn[k] * ka) + beta * kn[k];
    sink = dot(avec, qn);
    if (lane < 16) {
      *reinterpret_cast<float4*>(fa + sh * D + col) = make_float4(avec[0],avec[1],avec[2],avec[3]);
      *reinterpret_cast<float4*>(fa + sh * D + col + 4) = make_float4(avec[4],avec[5],avec[6],avec[7]);
    }
  }
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
    float x = partial[k] + __shfl_xor_sync(0xffffffff, partial[k], 16);
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
      float x = p2[k] + __shfl_xor_sync(0xffffffff,p2[k],16);
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
    float m = mv[k] + __shfl_xor_sync(0xffffffff,mv[k],16);
    float w = wq[k] + __shfl_xor_sync(0xffffffff,wq[k],16);
    if (lane < 16) { mv_s[warp][col+k]=m; wq_s[warp][col+k]=w; }
  }
  cd += __shfl_xor_sync(0xffffffff,cd,16);
  if (lane == 0) cd_s[warp]=cd;
  __syncthreads();
  float cd_all = (cd_s[0]+cd_s[1])+(cd_s[2]+cd_s[3]);
  float delta[8], oval[8];
  float vbv[8] = {vb0.x,vb0.y,vb0.z,vb0.w,vb1.x,vb1.y,vb1.z,vb1.w};
  float v[8]; bf8(vr,v);
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
  if constexpr (NORM) {
    if (warp == 0) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncwarp();
      norm_gate<true>(oval, col, norm_z, norm_w, norm_eps);
    }
  }
  if (warp == 0 && lane < 16)
    *reinterpret_cast<uint4*>(out + (int64_t(n)*HV+hv)*D + col) = pack_bf(oval);
  if (tid == 0) { fc[sh]=cnt+1; stale[slot]=1; }
}

template<typename BiasT>
__global__ void gates_kernel(const __nv_bfloat16* a,const __nv_bfloat16* b,
                             const float* al,const BiasT* db,float* out,int n,int hv) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) gate(__bfloat162float(a[i]),__bfloat162float(b[i]),al[i%hv],bias_float(db[i%hv]),out[2*i],out[2*i+1]);
}
} // namespace

void factored_step_r32(torch::Tensor mixed,torch::Tensor ag,torch::Tensor bg,
    torch::Tensor alog,torch::Tensor bias,torch::Tensor vb,torch::Tensor fa,
    torch::Tensor fu,torch::Tensor fw,torch::Tensor fc,torch::Tensor stale,
    torch::Tensor idx,torch::Tensor out,int H,double scale,double eps,
    c10::optional<torch::Tensor> pv,bool invalidate,
    c10::optional<torch::Tensor> z,c10::optional<torch::Tensor> norm_weight,double norm_eps) {
  const c10::cuda::CUDAGuard guard(mixed.device());
  int HV=fu.size(1), B=mixed.size(0);
  TORCH_CHECK(mixed.scalar_type()==at::kBFloat16 && ag.scalar_type()==at::kBFloat16 && bg.scalar_type()==at::kBFloat16);
  TORCH_CHECK(fu.scalar_type()==at::kHalf && fw.scalar_type()==at::kHalf);
  TORCH_CHECK(fa.scalar_type()==at::kFloat && vb.scalar_type()==at::kFloat && alog.scalar_type()==at::kFloat);
  TORCH_CHECK(bias.scalar_type()==at::kFloat || bias.scalar_type()==at::kBFloat16 || bias.scalar_type()==at::kHalf);
  TORCH_CHECK(fc.scalar_type()==at::kInt && stale.scalar_type()==at::kInt && out.scalar_type()==at::kBFloat16);
  TORCH_CHECK(idx.scalar_type()==at::kInt || idx.scalar_type()==at::kLong);
  TORCH_CHECK(fu.size(2)==32 && fu.size(3)==128 && fw.size(3)==128 && HV%H==0);
  TORCH_CHECK(!invalidate || (pv.has_value() && pv->scalar_type()==at::kInt && pv->numel()>=fu.size(0)));
  bool norm = norm_weight.has_value();
  if (norm) {
    TORCH_CHECK(z.has_value() && z->is_cuda() && norm_weight->is_cuda());
    TORCH_CHECK(z->device()==mixed.device() && norm_weight->device()==mixed.device());
    TORCH_CHECK(z->scalar_type()==at::kBFloat16 && norm_weight->scalar_type()==at::kBFloat16);
    TORCH_CHECK(z->is_contiguous() && z->dim()==3 && z->size(0)==B && z->size(1)==HV && z->size(2)==D);
    TORCH_CHECK(norm_weight->is_contiguous() && norm_weight->dim()==1 && norm_weight->numel()==D);
    TORCH_CHECK(norm_eps>0 && std::isfinite(norm_eps));
  }
  auto stream=at::cuda::getCurrentCUDAStream();
  auto launch=[&](auto flag, auto norm_flag, auto bias_value) {
    using BiasT = decltype(bias_value);
    factored_step_r32_kernel<BiasT, decltype(flag)::value, decltype(norm_flag)::value><<<B*HV,128,0,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(mixed.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(ag.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(bg.data_ptr()),alog.data_ptr<float>(),reinterpret_cast<const BiasT*>(bias.data_ptr()),vb.data_ptr<float>(),
      fa.data_ptr<float>(),reinterpret_cast<__half*>(fu.data_ptr()),reinterpret_cast<__half*>(fw.data_ptr()),
      fc.data_ptr<int>(),stale.data_ptr<int>(),idx.data_ptr(),idx.scalar_type()==at::kLong,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),invalidate?pv->data_ptr<int>():nullptr,
      H,HV,mixed.stride(0),ag.stride(0),bg.stride(0),idx.stride(0),float(scale),float(eps),
      norm?reinterpret_cast<const __nv_bfloat16*>(z->data_ptr()):nullptr,
      norm?reinterpret_cast<const __nv_bfloat16*>(norm_weight->data_ptr()):nullptr,float(norm_eps));
  };
  auto dispatch=[&](auto bias_value) {
    if(norm) {
      if(invalidate) launch(std::true_type{}, std::true_type{}, bias_value);
      else launch(std::false_type{}, std::true_type{}, bias_value);
    } else {
      if(invalidate) launch(std::true_type{}, std::false_type{}, bias_value);
      else launch(std::false_type{}, std::false_type{}, bias_value);
    }
  };
  if(bias.scalar_type()==at::kFloat) dispatch(float{});
  else if(bias.scalar_type()==at::kBFloat16) dispatch(__nv_bfloat16{});
  else dispatch(__half{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor gates(torch::Tensor a,torch::Tensor b,torch::Tensor al,torch::Tensor db) {
  TORCH_CHECK(al.scalar_type()==at::kFloat);
  TORCH_CHECK(db.scalar_type()==at::kFloat || db.scalar_type()==at::kBFloat16 || db.scalar_type()==at::kHalf);
  auto out=torch::empty({a.numel(),2},al.options());
  auto launch=[&](auto bias_value) {
    using BiasT = decltype(bias_value);
    gates_kernel<BiasT><<<(a.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
    al.data_ptr<float>(),reinterpret_cast<const BiasT*>(db.data_ptr()),out.data_ptr<float>(),a.numel(),a.size(-1));
  };
  if(db.scalar_type()==at::kFloat) launch(float{});
  else if(db.scalar_type()==at::kBFloat16) launch(__nv_bfloat16{});
  else launch(__half{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  namespace py = pybind11;
  m.def("factored_step_r32", &factored_step_r32,
    py::arg("mixed"), py::arg("ag"), py::arg("bg"), py::arg("alog"), py::arg("bias"),
    py::arg("vb"), py::arg("fa"), py::arg("fu"), py::arg("fw"), py::arg("fc"), py::arg("stale"),
    py::arg("idx"), py::arg("out"), py::arg("H"), py::arg("scale"), py::arg("eps"),
    py::arg("pv"), py::arg("invalidate"), py::arg("z")=c10::nullopt,
    py::arg("norm_weight")=c10::nullopt, py::arg("norm_eps")=1e-6);
  m.def("gates", &gates);
}
