// WS-1 K1: one CTA owns one (batch row, value head), four row-partitioned warps.
// Immutable factor snapshots and three CTA publications; warp 0 preserves the qualified reduction order.
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
  // Qualified Triton row reduction: product 1, FMA 0,2,...,7, then
  // butterfly offsets 8,4,2,1 across the 16 column lanes.
  float x = __fmul_rn(a[1], b[1]);
  x = __fmaf_rn(a[0], b[0], x);
  #pragma unroll
  for (int i = 2; i < 8; ++i) x = __fmaf_rn(a[i], b[i], x);
  return sum16(x);
}
__device__ __forceinline__ void gate(float a, float b, float alog, float bias, float& gt, float& beta) {
  float x = a + bias;
  // Triton tl.log lowers to __nv_logf in the qualified service image. Do not
  // replace logf with __logf or compile with --use_fast_math.
  float softplus = x <= 20.f ? logf(1.f + ex(x)) : x;
  gt = ex(__fmul_rn(-ex(alog), softplus));
  beta = __bfloat162float(__float2bfloat16_rn(div_full(1.f, 1.f + ex(-b))));
}

__device__ __forceinline__ uint2 load64(const void* p) {
  uint2 x;
  asm volatile("ld.global.v2.u32 {%0,%1},[%2];" : "=r"(x.x),"=r"(x.y) : "l"(p) : "memory");
  return x;
}
__device__ __forceinline__ void bf4(uint2 p,float (&x)[4]) {
  x[0]=__uint_as_float(p.x<<16); x[1]=__uint_as_float(p.x&0xffff0000);
  x[2]=__uint_as_float(p.y<<16); x[3]=__uint_as_float(p.y&0xffff0000);
}
__device__ __forceinline__ float dot4(const float (&a)[4],const float (&b)[4]) {
  float x=__fmul_rn(a[1],b[1]); x=__fmaf_rn(a[0],b[0],x);
  x=__fmaf_rn(a[2],b[2],x); x=__fmaf_rn(a[3],b[3],x);
  #pragma unroll
  for(int d=16;d;d>>=1) x=__fadd_rn(x,sx<32>(x,d));
  return x;
}
__device__ __forceinline__ void project_rows(const __half* mat,const float* c,
                                            int col,int parity,int cnt,float (&out)[8]) {
  #pragma unroll
  for(int t=0;t<16;++t) {
    int j=t==0?1:t==1?0:t, row=2*j+parity;
    float coeff=c[row];
    #pragma unroll
    for(int k=0;k<8;++k) {
      float u=row<cnt?__half2float(mat[row*D+col+k]):0.f;
      if(t==0) out[k]=__fmul_rn(u,coeff);
      else out[k]=__fmaf_rn(u,coeff,out[k]);
    }
  }
  #pragma unroll
  for(int k=0;k<8;++k) out[k]=__fadd_rn(out[k],sx<32>(out[k],16));
}

template<bool INVALIDATE>
__global__ __launch_bounds__(128, 4) void factored_step_r32_kernel(
    const __nv_bfloat16* mixed, const __nv_bfloat16* ag, const __nv_bfloat16* bg,
    const float* alog, const float* bias, const float* vb,
    float* fa, __half* fu, __half* fw, int* fc, int* stale,
    const void* idx, bool idx64, __nv_bfloat16* out, int* pv,
    int H, int HV, int nsm, int64_t sm, int64_t sa, int64_t sb, int64_t si, float scale, float eps) {
  int tid=threadIdx.x, lane=tid&31, col=(lane&15)*8, parity=lane>>4;
  // Logical warp zero rotates across physical warp slots at successive SM-sized
  // grid bands. Correctness is independent of placement; balance is measured.
  int leader=(blockIdx.x/nsm)%NW, warp=((tid>>5)+NW-leader)%NW;
  int n=blockIdx.x/HV, hv=blockIdx.x%HV, h=hv/(HV/H);
  int64_t slot=idx64?static_cast<const int64_t*>(idx)[n*si]:static_cast<const int*>(idx)[n*si];
  if constexpr(INVALIDATE) { if(tid==0 && hv==0) pv[slot<0?0:slot]=0; }
  if(slot<0) {
    if(tid<16) reinterpret_cast<uint4*>(out+(int64_t(n)*HV+hv)*D)[tid]=make_uint4(0,0,0,0);
    return;
  }
  int64_t sh=slot*HV+hv;
  __shared__ __align__(16) __half us[32][D], ws[32][D];
  __shared__ float qs[D],ks[D],vs[D],vbs[D],gb[3],cs[32],cqs[32],c2s[32],cfs[32],ds[D],khs[D];
  uint4 wraw[NR];
  #pragma unroll
  for(int j=0;j<NR;++j) {
    int row=2*(NW*j+warp)+parity;
    uint4 u=load128(fu+sh*32*D+row*D+col);
    wraw[j]=load128(fw+sh*32*D+row*D+col);
    *reinterpret_cast<uint4*>(&us[row][col])=u;
    *reinterpret_cast<uint4*>(&ws[row][col])=wraw[j];
  }
  // Warp 0 uses the exact packed-input layout: four values per physical lane.
  // Factor reads above are unconditional with respect to count.
  uint2 qr={},kr={},vr={}; float4 ar={},vbr={}; float av=0,bv=0,al=0,db=0;
  int nc=lane*4;
  if(warp==0) {
    const auto* inp=mixed+n*sm;
    qr=load64(inp+h*D+nc); kr=load64(inp+H*D+h*D+nc); vr=load64(inp+2*H*D+hv*D+nc);
    ar=*reinterpret_cast<const float4*>(fa+sh*D+nc);
    vbr=*reinterpret_cast<const float4*>(vb+hv*D+nc);
    av=__bfloat162float(ag[n*sa+hv]); bv=__bfloat162float(bg[n*sb+hv]); al=alog[hv]; db=bias[hv];
  }
  int cnt=fc[sh];
  if(warp==0) {
    float q[4],k[4],v[4]; bf4(qr,q);bf4(kr,k);bf4(vr,v);
    float nq=sq(dot4(q,q)+1.e-6f),nk=sq(dot4(k,k)+1.e-6f);
    #pragma unroll
    for(int i=0;i<4;++i) {q[i]=__fmul_rn(div_full(q[i],nq),scale);k[i]=div_full(k[i],nk);}
    float gt,beta;gate(av,bv,al,db,gt,beta);
    float a[4]={ar.x,ar.y,ar.z,ar.w}; float ka=dot4(k,a);
    #pragma unroll
    for(int i=0;i<4;++i) {float bk=__fmul_rn(beta,k[i]);a[i]=__fmaf_rn(gt,__fmaf_rn(-bk,ka,a[i]),bk);}
    float sink=dot4(a,q);
    *reinterpret_cast<float4*>(fa+sh*D+nc)=make_float4(a[0],a[1],a[2],a[3]);
    float vbar[4]={vbr.x,vbr.y,vbr.z,vbr.w};
    #pragma unroll
    for(int i=0;i<4;++i){qs[nc+i]=q[i];ks[nc+i]=k[i];vs[nc+i]=v[i];vbs[nc+i]=vbar[i];}
    if(lane==0){gb[0]=gt;gb[1]=beta;gb[2]=sink;}
  }
  __syncthreads(); // publish immutable U/W/input snapshots
  float kn[8],qn[8];
  #pragma unroll
  for(int k=0;k<8;++k){kn[k]=ks[col+k];qn[k]=qs[col+k];}
  #pragma unroll
  for(int j=0;j<NR;++j){
    int row=2*(NW*j+warp)+parity;float u[8];
    #pragma unroll
    for(int k=0;k<8;++k)u[k]=row<cnt?__half2float(us[row][col+k]):0.f;
    float c=dot(u,kn),cq=dot(u,qn);
    if((lane&15)==0){cs[row]=c;cqs[row]=cq;}
  }
  __syncthreads(); // publish split row dot products
  if(warp==0) {
    // One warp preserves Triton's 16-row in-thread FMA order. Streaming from
    // shared memory avoids the original full-tile register spills and HBM chain.
    float proj[8],kp[8];project_rows(&us[0][0],cs,col,parity,cnt,proj);
    #pragma unroll
    for(int k=0;k<8;++k)kp[k]=__fsub_rn(kn[k],proj[k]);
    float nrm2=dot(kp,kp);
    if(nrm2<.25f){
      #pragma unroll
      for(int j=0;j<16;++j){
        int row=2*j+parity;float u[8];
        #pragma unroll
        for(int k=0;k<8;++k)u[k]=row<cnt?__half2float(us[row][col+k]):0.f;
        float c2=dot(u,kp);if((lane&15)==0)c2s[row]=c2;
      }
      asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
      project_rows(&us[0][0],c2s,col,parity,cnt,proj);
      #pragma unroll
      for(int k=0;k<8;++k)kp[k]=__fsub_rn(kp[k],proj[k]);
      if((lane&15)==0){
        #pragma unroll
        for(int j=0;j<16;++j){int row=2*j+parity;cs[row]=__fadd_rn(cs[row],c2s[row]);}
      }
      asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
      nrm2=dot(kp,kp);
    }
    float nrm=sq(nrm2),khat[8];bool keep=nrm>eps;
    #pragma unroll
    for(int k=0;k<8;++k)khat[k]=keep?div_full(kp[k],fmaxf(nrm,eps)):0.f;
    float clast=keep?nrm:0.f,kq=dot(khat,qn),mv[8],wq[8],cd=0.f;
    #pragma unroll
    for(int t=0;t<16;++t){
      int j=t==0?1:t==1?0:t,row=2*j+parity;float c=cs[row];
      float cq=__fadd_rn(cqs[row],row==cnt?kq:0.f),cf=row==cnt?clast:c;
      if((lane&15)==0)cfs[row]=cf;
      if(t==0)cd=__fmul_rn(cf,cq);else cd=__fmaf_rn(cf,cq,cd);
      #pragma unroll
      for(int k=0;k<8;++k){
        float w=row<cnt?__half2float(ws[row][col+k]):0.f;
        if(t==0){mv[k]=__fmul_rn(w,c);wq[k]=__fmul_rn(w,cq);}
        else {mv[k]=__fmaf_rn(w,c,mv[k]);wq[k]=__fmaf_rn(w,cq,wq[k]);}
      }
    }
    cd=__fadd_rn(cd,sx<32>(cd,16));
    float ov[8],gt=gb[0],beta=gb[1],sink=gb[2];
    #pragma unroll
    for(int k=0;k<8;++k){
      mv[k]=__fadd_rn(mv[k],sx<32>(mv[k],16));wq[k]=__fadd_rn(wq[k],sx<32>(wq[k],16));
      float delta=__fmul_rn(beta,__fmaf_rn(-gt,mv[k],__fsub_rn(vs[col+k],vbs[col+k])));
      ov[k]=__fmaf_rn(delta,cd,__fmaf_rn(gt,wq[k],__fmul_rn(vbs[col+k],sink)));
      if(lane<16){ds[col+k]=delta;khs[col+k]=khat[k];}
    }
    if(lane<16)*reinterpret_cast<uint4*>(out+(int64_t(n)*HV+hv)*D+col)=pack_bf(ov);
  }
  __syncthreads(); // publish exact delta/khat/cfull before split writeback
  float gt=gb[0],delta[8],khat[8];
  #pragma unroll
  for(int k=0;k<8;++k){delta[k]=ds[col+k];khat[k]=khs[col+k];}
  #pragma unroll
  for(int j=0;j<NR;++j){
    int row=2*(NW*j+warp)+parity;
    if(row<=cnt){
      float nw[8],cf=cfs[row];
      #pragma unroll
      for(int k=0;k<8;++k)nw[k]=__fmaf_rn(gt,row<cnt?half_at(wraw[j],k):0.f,__fmul_rn(cf,delta[k]));
      *reinterpret_cast<uint4*>(fw+sh*32*D+row*D+col)=pack_half(nw);
    }
    if(row==cnt && cnt<32)*reinterpret_cast<uint4*>(fu+sh*32*D+row*D+col)=pack_half(khat);
  }
  if(tid==0){fc[sh]=cnt+1;stale[slot]=1;}
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
      H,HV,at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
      mixed.stride(0),ag.stride(0),bg.stride(0),idx.stride(0),float(scale),float(eps));
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
