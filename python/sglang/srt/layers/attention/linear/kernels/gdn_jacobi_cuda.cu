#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cusolverDn.h>
#include <mma.h>

__device__ __forceinline__ float to_tf32(float x) {
  unsigned bits; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(x)); return __uint_as_float(bits);
}

// A warp owns one matrix. Each lane owns a column; rows reside in registers.
// Turn is a template argument so all row permutations become register moves,
// avoiding local-memory arrays addressed with a runtime partner index.
template<int N, int Turn>
__device__ __forceinline__ void rotate(float (&a)[N], float (&v)[N], int lane) {
  constexpr unsigned mask = N == 32 ? 0xffffffffu : 0xffffu;
  int partner = lane == N-1 ? Turn : (lane == Turn ? N-1 : (2*Turn-lane+2*(N-1))%(N-1));
  float diag=0, off=0;
  #pragma unroll
  for(int i=0;i<N;++i) { if(lane==i) diag=a[i]; if(partner==i) off=a[i]; }
  float other=__shfl_sync(mask,diag,partner,N);
  off=0.5f*(off+__shfl_sync(mask,off,partner,N));
  float tau=(diag-other)/(fabsf(off)>1.e-30f?2*off:1.f);
  float t=copysignf(1.f,tau)/(fabsf(tau)+sqrtf(1.f+tau*tau));
  if(diag==other) t=lane<partner?1.f:-1.f;
  if(fabsf(off)<=1.e-12f*sqrtf(fabsf(diag*other))) t=0;
  float c=rsqrtf(1+t*t), s=t*c;
  #pragma unroll
  for(int i=0;i<N;++i) {
    a[i]=c*a[i]+s*__shfl_sync(mask,a[i],partner,N);
    v[i]=c*v[i]+s*__shfl_sync(mask,v[i],partner,N);
  }
  float tmp[N];
  #pragma unroll
  for(int i=0;i<N;++i) {
    int p=i==N-1?Turn:(i==Turn?N-1:(2*Turn-i+2*(N-1))%(N-1));
    float ci=__shfl_sync(mask,c,i,N), si=__shfl_sync(mask,s,i,N);
    tmp[i]=ci*a[i]+si*a[p];
  }
  #pragma unroll
  for(int i=0;i<N;++i) a[i]=tmp[i];
}

template<int N,int Turn=0>
__device__ __forceinline__ void sweep(float (&a)[N],float (&v)[N],int lane) {
  rotate<N,Turn>(a,v,lane);
  if constexpr(Turn+1<N-1) sweep<N,Turn+1>(a,v,lane);
}

template<int N>
__global__ void jacobi(const float* __restrict__ gram,float* __restrict__ z,
                       const int* __restrict__ active,int batch,int r,int sweeps) {
  int lane=threadIdx.x%32, head=blockIdx.x*4+threadIdx.x/32;
  if(head>=batch || !active[head] || lane>=N) return;
  constexpr unsigned mask=N==32?0xffffffffu:0xffffu;
  float a[N],v[N];
  #pragma unroll
  for(int i=0;i<N;++i) {a[i]=gram[head*N*N+i*N+lane];v[i]=(i==lane);}
  float norm=0;
  #pragma unroll
  for(int i=0;i<N;++i) norm=fmaxf(norm,fabsf(a[i]));
  #pragma unroll
  for(int delta=N/2;delta;delta/=2) norm=fmaxf(norm,__shfl_xor_sync(mask,norm,delta,N));
  norm=fmaxf(norm,1.e-30f);
  #pragma unroll
  for(int i=0;i<N;++i) a[i]/=norm;
  for(int k=0;k<sweeps;++k) sweep<N>(a,v,lane);
  float diag=0;
  #pragma unroll
  for(int i=0;i<N;++i) if(lane==i) diag=a[i];
  int rank=0;
  #pragma unroll
  for(int j=0;j<N;++j) {float d=__shfl_sync(mask,diag,j,N);rank+=(d>diag || (d==diag && j<lane));}
  #pragma unroll
  for(int i=0;i<N;++i) z[head*N*N+i*N+rank]=rank<r?v[i]:0.f;
}

void eig(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t sweeps) {
  TORCH_CHECK(gram.is_cuda() && gram.is_contiguous() && gram.scalar_type()==torch::kFloat32);
  int n=gram.size(-1),batch=gram.size(0);
  auto stream=at::cuda::getCurrentCUDAStream();
  if(n==32) jacobi<32><<<(batch+3)/4,128,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,sweeps);
  else if(n==16) jacobi<16><<<(batch+3)/4,128,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,sweeps);
  else TORCH_CHECK(false,"N must be 16 or 32");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int N>
__device__ __forceinline__ float local_dot(const float (&a)[N], const float (&b)[N]) {
  float s0=0,s1=0,s2=0,s3=0;
  #pragma unroll
  for(int i=0;i<N;i+=4) {s0=fmaf(a[i],b[i],s0);s1=fmaf(a[i+1],b[i+1],s1);s2=fmaf(a[i+2],b[i+2],s2);s3=fmaf(a[i+3],b[i+3],s3);}
  return (s0+s1)+(s2+s3);
}

template<int N>
__global__ void subspace(const float* __restrict__ gram,float* __restrict__ z,
                         const int* __restrict__ active,int batch,int r,int iters,int passes) {
  int lane=threadIdx.x%32,head=blockIdx.x*4+threadIdx.x/32;
  if(head>=batch || !active[head] || lane>=N) return;
  constexpr unsigned mask=N==32?0xffffffffu:0xffffu;
  float g[N],v[N];
  float diag=0;
  #pragma unroll
  for(int i=0;i<N;++i) {g[i]=gram[head*N*N+i*N+lane]; if(lane==i) diag=g[i];}
  int rank=0;
  #pragma unroll
  for(int j=0;j<N;++j) {float d=__shfl_sync(mask,diag,j,N);rank+=(d>diag || (d==diag && j<lane));}
  #pragma unroll
  for(int i=0;i<N;++i) {int rr=__shfl_sync(mask,rank,i,N);v[i]=float(rr==lane && lane<r);}
  for(int it=0;it<iters;++it) {
    float y[N];
    #pragma unroll
    for(int i=0;i<N;++i) {
      float sum=0;
      #pragma unroll
      for(int k=0;k<N;++k) sum=fmaf(__shfl_sync(mask,g[i],k,N),v[k],sum);
      y[i]=sum;
    }
    float norm0=local_dot<N>(y,y);
    #pragma unroll
    for(int i=0;i<N;++i) v[i]=y[i];
    for(int pass=0;pass<passes;++pass) {
      for(int j=0;j<r;++j) {
        float norm=local_dot<N>(v,v);
        float inv=norm>1.e-24f && (pass || norm>norm0*1.e-8f)?rsqrtf(norm):0.f;
        if(lane==j) {
          #pragma unroll
          for(int i=0;i<N;++i) v[i]*=inv;
        }
        // Warp-uniform: a dropped direction is exactly zero, so its projection
        // is a no-op. Real low-rank heads should not pay the full rank-16 chain.
        if(__shfl_sync(mask,inv,j,N)>0.f) {
          float q[N];
          #pragma unroll
          for(int i=0;i<N;++i) q[i]=__shfl_sync(mask,v[i],j,N);
          float proj=local_dot<N>(v,q);
          if(lane>j && lane<r) {
            #pragma unroll
            for(int i=0;i<N;++i) v[i]=fmaf(-proj,q[i],v[i]);
          }
        }
      }
    }
  }
  #pragma unroll
  for(int i=0;i<N;++i) z[head*N*N+i*N+lane]=lane<r?v[i]:0.f;
}

template<int N>
__global__ void tensor_subspace(const float* __restrict__ gram,float* __restrict__ z,
                         const int* __restrict__ active,int batch,int r,int iters,int passes) {
  int lane=threadIdx.x%32,head=blockIdx.x*4+threadIdx.x/32;
  if(head>=batch || !active[head] || lane>=N) return;
  constexpr unsigned mask=N==32?0xffffffffu:0xffffu;
  __shared__ float shared[4][3][N*N];
  float* gs=shared[threadIdx.x/32][0];
  float* zs=shared[threadIdx.x/32][1];
  float* ys=shared[threadIdx.x/32][2];
  float g[N],v[N];
  float diag=0;
  #pragma unroll
  for(int i=0;i<N;++i) {g[i]=gram[head*N*N+i*N+lane]; if(lane==i) diag=g[i];}
  int rank=0;
  #pragma unroll
  for(int j=0;j<N;++j) {float d=__shfl_sync(mask,diag,j,N);rank+=(d>diag || (d==diag && j<lane));}
  #pragma unroll
  for(int i=0;i<N;++i) {int rr=__shfl_sync(mask,rank,i,N);v[i]=float(rr==lane && lane<r);}
  #pragma unroll
  for(int i=0;i<N;++i) gs[i*N+lane]=g[i];
  for(int it=0;it<iters;++it) {
    #pragma unroll
    for(int i=0;i<N;++i) zs[i*N+lane]=v[i];
    __syncwarp(mask);
    using namespace nvcuda;
    #pragma unroll
    for(int tile=0;tile<N/16;++tile) {
      wmma::fragment<wmma::accumulator,16,16,8,float> acc;
      wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<N;k+=8) {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
        wmma::load_matrix_sync(ah,gs+tile*16*N+k,N);
        wmma::load_matrix_sync(bh,zs+k*N,N);
        #pragma unroll
        for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
        #pragma unroll
        for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
        wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
      }
      wmma::store_matrix_sync(ys+tile*16*N,acc,N,wmma::mem_row_major);
    }
    __syncwarp(mask);
    float y[N];
    #pragma unroll
    for(int i=0;i<N;++i) y[i]=lane<r?ys[i*N+lane]:0.f;
    float norm0=local_dot<N>(y,y);
    #pragma unroll
    for(int i=0;i<N;++i) v[i]=y[i];
    for(int pass=0;pass<passes;++pass) {
      for(int j=0;j<r;++j) {
        float norm=local_dot<N>(v,v);
        float inv=norm>1.e-24f && (pass || norm>norm0*1.e-8f)?rsqrtf(norm):0.f;
        if(lane==j) {
          #pragma unroll
          for(int i=0;i<N;++i) v[i]*=inv;
        }
        // Warp-uniform: a dropped direction is exactly zero, so its projection
        // is a no-op. Real low-rank heads should not pay the full rank-16 chain.
        if(__shfl_sync(mask,inv,j,N)>0.f) {
          float q[N];
          #pragma unroll
          for(int i=0;i<N;++i) q[i]=__shfl_sync(mask,v[i],j,N);
          float proj=local_dot<N>(v,q);
          if(lane>j && lane<r) {
            #pragma unroll
            for(int i=0;i<N;++i) v[i]=fmaf(-proj,q[i],v[i]);
          }
        }
      }
    }
  }
  #pragma unroll
  for(int i=0;i<N;++i) z[head*N*N+i*N+lane]=lane<r?v[i]:0.f;
}

template<int N>
__device__ __forceinline__ void polar_correct(float (&v)[N],float* gs,float* zs,float* ys,int lane,int r) {
  using namespace nvcuda;
  #pragma unroll
  for(int i=0;i<N;++i) zs[i*N+lane]=lane<r?v[i]:0.f;
  __syncwarp();
  wmma::fragment<wmma::accumulator,16,16,8,float> gram;
  wmma::fill_fragment(gram,0.f);
  #pragma unroll
  for(int k=0;k<N;k+=8) {
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> ah,al;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
    wmma::load_matrix_sync(ah,zs+k*N,N);wmma::load_matrix_sync(bh,zs+k*N,N);
    #pragma unroll
    for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
    #pragma unroll
    for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
    wmma::mma_sync(gram,al,bh,gram);wmma::mma_sync(gram,ah,bl,gram);wmma::mma_sync(gram,ah,bh,gram);
  }
  wmma::store_matrix_sync(gs,gram,N,wmma::mem_row_major);
  __syncwarp();
  #pragma unroll
  for(int i=0;i<16;++i) gs[i*N+lane]=lane<16?1.5f*(lane==i)-0.5f*gs[i*N+lane]:0.f;
  __syncwarp();
  #pragma unroll
  for(int tile=0;tile<N/16;++tile) {
    wmma::fragment<wmma::accumulator,16,16,8,float> acc;
    wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int k=0;k<16;k+=8) {
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
      wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
      wmma::load_matrix_sync(ah,zs+tile*16*N+k,N);wmma::load_matrix_sync(bh,gs+k*N,N);
      #pragma unroll
      for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
      #pragma unroll
      for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
      wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
    }
    wmma::store_matrix_sync(ys+tile*16*N,acc,N,wmma::mem_row_major);
  }
  __syncwarp();
  #pragma unroll
  for(int i=0;i<N;++i) v[i]=lane<r?ys[i*N+lane]:0.f;
}

template<int N,typename scalar_t,int GROUP=4,int POWER=1>
__global__ void tensor_project(const float* __restrict__ gram,float* __restrict__ z,
                         const int* __restrict__ active,int batch,int r,int iters,int passes,
                         scalar_t* u,scalar_t* w,int* count,const void* indices,
                         bool idx64,int64_t stride,int h,int full) {
  int lane=threadIdx.x%32,head=blockIdx.x*GROUP+threadIdx.x/32;
  if(head>=batch || !active[head] || lane>=N) return;
  constexpr unsigned mask=N==32?0xffffffffu:0xffffu;
  __shared__ float shared[GROUP][3][N*N];
  float* gs=shared[threadIdx.x/32][0];
  float* zs=shared[threadIdx.x/32][1];
  float* ys=shared[threadIdx.x/32][2];
  float g[N],v[N];
  float diag=0;
  #pragma unroll
  for(int i=0;i<N;++i) {g[i]=gram[head*N*N+i*N+lane]; if(lane==i) diag=g[i];}
  int rank=0;
  #pragma unroll
  for(int j=0;j<N;++j) {float d=__shfl_sync(mask,diag,j,N);rank+=(d>diag || (d==diag && j<lane));}
  #pragma unroll
  for(int i=0;i<N;++i) {int rr=__shfl_sync(mask,rank,i,N);v[i]=float(rr==lane && lane<r);}
  #pragma unroll
  for(int i=0;i<N;++i) gs[i*N+lane]=g[i];
  for(int it=0;it<iters;++it) {
    for(int power=0;power<POWER;++power) {

    #pragma unroll
    for(int i=0;i<N;++i) zs[i*N+lane]=v[i];
    __syncwarp(mask);
    using namespace nvcuda;
    #pragma unroll
    for(int tile=0;tile<N/16;++tile) {
      wmma::fragment<wmma::accumulator,16,16,8,float> acc;
      wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<N;k+=8) {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
        wmma::load_matrix_sync(ah,gs+tile*16*N+k,N);
        wmma::load_matrix_sync(bh,zs+k*N,N);
        #pragma unroll
        for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
        #pragma unroll
        for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
        wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
      }
      wmma::store_matrix_sync(ys+tile*16*N,acc,N,wmma::mem_row_major);
    }
    __syncwarp(mask);
    float y[N];
    #pragma unroll
    for(int i=0;i<N;++i) y[i]=lane<r?ys[i*N+lane]:0.f;

    #pragma unroll
    for(int i=0;i<N;++i) v[i]=y[i];
    }
    float norm0=local_dot<N>(v,v);
    for(int pass=0;pass<(passes<0?1:passes);++pass) {
      for(int j=0;j<r;++j) {
        float norm=local_dot<N>(v,v);
        float inv=norm>1.e-24f && (pass || norm>norm0*1.e-8f)?rsqrtf(norm):0.f;
        if(lane==j) {
          #pragma unroll
          for(int i=0;i<N;++i) v[i]*=inv;
        }
        // Warp-uniform: a dropped direction is exactly zero, so its projection
        // is a no-op. Real low-rank heads should not pay the full rank-16 chain.
        if(__shfl_sync(mask,inv,j,N)>0.f) {
          float q[N];
          #pragma unroll
          for(int i=0;i<N;++i) q[i]=__shfl_sync(mask,v[i],j,N);
          float proj=local_dot<N>(v,q);
          if(lane>j && lane<r) {
            #pragma unroll
            for(int i=0;i<N;++i) v[i]=fmaf(-proj,q[i],v[i]);
          }
        }
      }
    }
  }
  if(passes<0) polar_correct<N>(v,gs,zs,ys,lane,r);
  #pragma unroll
  for(int i=0;i<N;++i) zs[i*N+lane]=lane<r?v[i]:0.f;
  __syncwarp(mask);
  int64_t slot=idx64?static_cast<const int64_t*>(indices)[(head/h)*stride]
                    :static_cast<const int*>(indices)[(head/h)*stride];
  int64_t sh=slot*h+head%h;
  using namespace nvcuda;
  for(int factor=0;factor<2;++factor) {
    scalar_t* ptr=factor?w:u;
    for(int d=0;d<128;d+=16) {
      #pragma unroll
      for(int j=0;j<N*16/32;++j) {
        int off=j*32+lane,row=off/16,col=off%16;
        gs[off]=row<full?float(ptr[sh*N*128+row*128+d+col]):0.f;
      }
      __syncwarp(mask);
      wmma::fragment<wmma::accumulator,16,16,8,float> acc;
      wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<N;k+=8) {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> ah,al;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
        wmma::load_matrix_sync(ah,zs+k*N,N);wmma::load_matrix_sync(bh,gs+k*16,16);
        #pragma unroll
        for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
        #pragma unroll
        for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
        wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
      }
      wmma::store_matrix_sync(ys,acc,16,wmma::mem_row_major);
      __syncwarp(mask);
      #pragma unroll
      for(int j=0;j<8;++j) {
        int off=j*32+lane,row=off/16,col=off%16;
        if(row<r) ptr[sh*N*128+row*128+d+col]=scalar_t(ys[off]);
      }
      __syncwarp(mask);
    }
  }
  if(lane==0) count[sh]=r;
}

void mgs(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes) {
  int n=gram.size(-1),batch=gram.size(0);auto stream=at::cuda::getCurrentCUDAStream();
  if(n==32) subspace<32><<<(batch+3)/4,128,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes);
  else subspace<16><<<(batch+3)/4,128,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}



struct Solver {
  cusolverDnHandle_t handle;
  syevjInfo_t params;
  Solver() {cusolverDnCreate(&handle);cusolverDnCreateSyevjInfo(&params);
            cusolverDnXsyevjSetTolerance(params,1.e-7);cusolverDnXsyevjSetMaxSweeps(params,100);}
  ~Solver(){cusolverDnDestroySyevjInfo(params);cusolverDnDestroy(handle);}
};
static Solver& solver(){static thread_local Solver s;return s;}
int64_t workspace(torch::Tensor gram,torch::Tensor vals) {
  auto& s=solver();int n=gram.size(-1),batch=gram.size(0),lw=0;
  auto st=cusolverDnSsyevjBatched_bufferSize(s.handle,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
           n,gram.data_ptr<float>(),n,vals.data_ptr<float>(),&lw,s.params,batch);
  TORCH_CHECK(st==CUSOLVER_STATUS_SUCCESS,"bufferSize status ",int(st));return lw;
}
void eiglib(torch::Tensor gram,torch::Tensor vals,torch::Tensor work,torch::Tensor info) {
  auto& s=solver();int n=gram.size(-1),batch=gram.size(0);
  cusolverDnSetStream(s.handle,at::cuda::getCurrentCUDAStream());
  auto st=cusolverDnSsyevjBatched(s.handle,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,
       gram.data_ptr<float>(),n,vals.data_ptr<float>(),work.data_ptr<float>(),work.numel(),info.data_ptr<int>(),s.params,batch);
  TORCH_CHECK(st==CUSOLVER_STATUS_SUCCESS,"syevjBatched status ",int(st));
}

void tensormgs(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes) {
  TORCH_CHECK(gram.size(-1)==32,"tensor MGS is the RMAX32 candidate");
  int batch=gram.size(0);auto stream=at::cuda::getCurrentCUDAStream();
  tensor_subspace<32><<<(batch+3)/4,128,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tensorproject(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16><<<(batch+3)/4,128,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float><<<(batch+3)/4,128,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tensorproject1(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16,1><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float,1><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tensorprojectp2(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16,1,2><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float,1,2><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tensorprojectp3(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16,1,3><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float,1,3><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("eig", &eig);m.def("mgs", &mgs);m.def("tensormgs", &tensormgs);m.def("tensorproject", &tensorproject);m.def("tensorproject1", &tensorproject1);m.def("tensorprojectp2", &tensorprojectp2);m.def("tensorprojectp3", &tensorprojectp3);m.def("workspace", &workspace);m.def("eiglib", &eiglib);}
