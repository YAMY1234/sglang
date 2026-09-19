#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cusolverDn.h>
#include <mma.h>
#ifndef K3_FAST_GZ
#define K3_FAST_GZ 0
#endif
#ifndef K3_FAST_PROJECT
#define K3_FAST_PROJECT 0
#endif
#ifndef K3_RANK_TOL_SQ
#define K3_RANK_TOL_SQ 1.e-8f
#endif

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
  for(int i=0;i<N;i+=4) {s0=fmaf(a[i],b[i],s0);if(i+1<N)s1=fmaf(a[i+1],b[i+1],s1);if(i+2<N)s2=fmaf(a[i+2],b[i+2],s2);if(i+3<N)s3=fmaf(a[i+3],b[i+3],s3);}
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
        float inv=norm>1.e-24f && (pass || norm>norm0*K3_RANK_TOL_SQ)?rsqrtf(norm):0.f;
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
        float inv=norm>1.e-24f && (pass || norm>norm0*K3_RANK_TOL_SQ)?rsqrtf(norm):0.f;
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

template<int N,int LD=N>
__device__ __forceinline__ void polar_correct(float (&v)[N],float* gs,float* zs,float* ys,int lane,int r) {
  using namespace nvcuda;
  #pragma unroll
  for(int i=0;i<N;++i) zs[i*LD+lane]=lane<r?v[i]:0.f;
  __syncwarp();
  wmma::fragment<wmma::accumulator,16,16,8,float> gram;
  wmma::fill_fragment(gram,0.f);
  #pragma unroll
  for(int k=0;k<N;k+=8) {
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> ah,al;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
    wmma::load_matrix_sync(ah,zs+k*LD,LD);wmma::load_matrix_sync(bh,zs+k*LD,LD);
    #pragma unroll
    for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
    #pragma unroll
    for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
    wmma::mma_sync(gram,al,bh,gram);wmma::mma_sync(gram,ah,bl,gram);wmma::mma_sync(gram,ah,bh,gram);
  }
  wmma::store_matrix_sync(gs,gram,LD,wmma::mem_row_major);
  __syncwarp();
  #pragma unroll
  for(int i=0;i<16;++i) gs[i*LD+lane]=lane<16?1.5f*(lane==i)-0.5f*gs[i*LD+lane]:0.f;
  __syncwarp();
  #pragma unroll
  for(int tile=0;tile<N/16;++tile) {
    wmma::fragment<wmma::accumulator,16,16,8,float> acc;
    wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int k=0;k<16;k+=8) {
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
      wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
      wmma::load_matrix_sync(ah,zs+tile*16*LD+k,LD);wmma::load_matrix_sync(bh,gs+k*LD,LD);
      #pragma unroll
      for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
      #pragma unroll
      for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
      wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
    }
    wmma::store_matrix_sync(ys+tile*16*LD,acc,LD,wmma::mem_row_major);
  }
  __syncwarp();
  #pragma unroll
  for(int i=0;i<N;++i) v[i]=lane<r?ys[i*LD+lane]:0.f;
}

template<int K,typename T>
__device__ __forceinline__ void chol_factor_reg(T (&g)[16],T original,int lane) {
  constexpr unsigned mask=0x0000ffffu;
  T pivot=__shfl_sync(mask,g[K],K,16),n0=__shfl_sync(mask,original,K,16);
  T inv=pivot>T(1.e-24) && pivot>n0*T(K3_RANK_TOL_SQ)?rsqrt(pivot):T(0);
  T lj=lane>=K?g[K]*inv:T(0);
  #pragma unroll
  for(int i=0;i<16;++i) {
    T li=__shfl_sync(mask,lj,i,16);
    if(i>K && lane>K) g[i]=fma(-li,lj,g[i]);
  }
  g[K]=lj;
  if constexpr(K<15) chol_factor_reg<K+1>(g,original,lane);
}

template<int I,typename T>
__device__ __forceinline__ void chol_inverse_reg(const T (&g)[16],T (&t)[16],int lane) {
  constexpr unsigned mask=0x0000ffffu;
  T rhs=T(I==lane);
  #pragma unroll
  for(int k=0;k<16;++k) if(k>I) rhs=fma(-__shfl_sync(mask,g[I],k,16),t[k],rhs);
  T diag=__shfl_sync(mask,g[I],I,16);
  t[I]=diag>T(0)?rhs/diag:T(0);
  if constexpr(I>0) chol_inverse_reg<I-1>(g,t,lane);
}

template<int N,int LD=N>
__device__ __forceinline__ void chol_final(float (&v)[N],float* gs,float* zs,float* ys,int lane,int r) {
  using namespace nvcuda;
  #pragma unroll
  for(int i=0;i<N;++i) zs[i*LD+lane]=lane<r?v[i]:0.f;
  __syncwarp();
  wmma::fragment<wmma::accumulator,16,16,8,float> gram;
  wmma::fill_fragment(gram,0.f);
  #pragma unroll
  for(int k=0;k<N;k+=8) {
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> ah,al;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
    wmma::load_matrix_sync(ah,zs+k*LD,LD);wmma::load_matrix_sync(bh,zs+k*LD,LD);
    #pragma unroll
    for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
    #pragma unroll
    for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
    wmma::mma_sync(gram,al,bh,gram);wmma::mma_sync(gram,ah,bl,gram);wmma::mma_sync(gram,ah,bh,gram);
  }
  wmma::store_matrix_sync(gs,gram,LD,wmma::mem_row_major);
  __syncwarp();
  float inverse[16]={};
  if(lane<16) {
    float g[16],original=0.f;
    #pragma unroll
    for(int i=0;i<16;++i) {g[i]=gs[i*LD+lane];if(i==lane) original=g[i];}
    chol_factor_reg<0>(g,original,lane);
    chol_inverse_reg<15>(g,inverse,lane);
  }
  #pragma unroll
  for(int i=0;i<16;++i) gs[i*LD+lane]=lane<16?inverse[i]:0.f;
  __syncwarp();
  #pragma unroll
  for(int tile=0;tile<N/16;++tile) {
    wmma::fragment<wmma::accumulator,16,16,8,float> acc;
    wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int k=0;k<16;k+=8) {
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
      wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
      wmma::load_matrix_sync(ah,zs+tile*16*LD+k,LD);wmma::load_matrix_sync(bh,gs+k*LD,LD);
      #pragma unroll
      for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
      #pragma unroll
      for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
      wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
    }
    wmma::store_matrix_sync(ys+tile*16*LD,acc,LD,wmma::mem_row_major);
  }
  __syncwarp();
  #pragma unroll
  for(int i=0;i<N;++i) v[i]=lane<r?ys[i*LD+lane]:0.f;
}

__device__ __forceinline__ float warp_sum(float x) {
  #pragma unroll
  for(int d=16;d;d/=2) x+=__shfl_xor_sync(0xffffffffu,x,d);
  return x;
}

// Transpose ownership only for QR: one lane per row, 16 columns in registers.
// Each dot is a five-shuffle tree rather than a 32-element serial local sum.
template<int N,int LD>
__device__ __forceinline__ void row_qr(float* ys,float* zs,float (&v)[N],int lane,int r,int passes) {
  float a[16],n0[16];
  #pragma unroll
  for(int c=0;c<16;++c) {a[c]=c<r?ys[lane*LD+c]:0.f;n0[c]=warp_sum(a[c]*a[c]);}
  for(int pass=0;pass<(passes<0?1:passes);++pass) {
    #pragma unroll
    for(int j=0;j<16;++j) {
      float norm=warp_sum(a[j]*a[j]);
      float inv=norm>1.e-24f && (pass || norm>n0[j]*K3_RANK_TOL_SQ)?rsqrtf(norm):0.f;
      a[j]*=inv;
      #pragma unroll
      for(int c=j+1;c<16;++c) {
        float dot=warp_sum(a[j]*a[c]);a[c]=fmaf(-dot,a[j],a[c]);
      }
    }
  }
  #pragma unroll
  for(int c=0;c<16;++c) zs[lane*LD+c]=a[c];
  __syncwarp();
  #pragma unroll
  for(int i=0;i<N;++i) v[i]=lane<r?zs[i*LD+lane]:0.f;
}

template<int L>
__device__ __forceinline__ float group_sum(float x) {
  unsigned mask=0xffffffffu;
  if constexpr (L<32) mask=((1u<<L)-1u)<<((threadIdx.x%32/L)*L);
  #pragma unroll
  for(int d=1;d<L;d*=2) x+=__shfl_xor_sync(mask,x,d,L);
  return x;
}

template<int N,int LD,int L>
__device__ __forceinline__ void group_qr(float* ys,float (&v)[N],int lane,int r,int passes) {
  constexpr int C=32/L,PARTS=16/C,NR=N/L;
  int col=lane/L,part=lane%L;
  float a[PARTS][NR],n0[PARTS];
  #pragma unroll
  for(int c=0;c<PARTS;++c) {
    #pragma unroll
    for(int i=0;i<NR;++i) a[c][i]=(col+c*C)<r?ys[(i*L+part)*LD+col+c*C]:0.f;
    n0[c]=group_sum<L>(local_dot<NR>(a[c],a[c]));
  }
  for(int pass=0;pass<(passes<0?1:passes);++pass) {
    #pragma unroll
    for(int j=0;j<16;++j) {
      float q[NR],original=0;
      #pragma unroll
      for(int i=0;i<NR;++i) q[i]=0.f;
      #pragma unroll
      for(int c=0;c<PARTS;++c) if(j/C==c) {
        original=n0[c];
        #pragma unroll
        for(int i=0;i<NR;++i) q[i]=a[c][i];
      }
      int source=(j%C)*L+part;
      original=__shfl_sync(0xffffffffu,original,source);
      #pragma unroll
      for(int i=0;i<NR;++i) q[i]=__shfl_sync(0xffffffffu,q[i],source);
      float norm=group_sum<L>(local_dot<NR>(q,q));
      float inv=norm>1.e-24f && (pass || norm>original*K3_RANK_TOL_SQ)?rsqrtf(norm):0.f;
      #pragma unroll
      for(int i=0;i<NR;++i) q[i]*=inv;
      #pragma unroll
      for(int c=0;c<PARTS;++c) {
        float dot=0.f;
        if(inv>0.f) dot=group_sum<L>(local_dot<NR>(a[c],q));
        if(col+c*C==j) {
          #pragma unroll
          for(int i=0;i<NR;++i) a[c][i]=q[i];
        } else if(inv>0.f && col+c*C>j) {
          #pragma unroll
          for(int i=0;i<NR;++i) a[c][i]=fmaf(-dot,q[i],a[c][i]);
        }
      }
    }
  }
  #pragma unroll
  for(int c=0;c<PARTS;++c) {
    #pragma unroll
    for(int i=0;i<NR;++i) ys[(i*L+part)*LD+col+c*C]=a[c][i];
  }
  __syncwarp();
  #pragma unroll
  for(int i=0;i<N;++i) v[i]=lane<r?ys[i*LD+lane]:0.f;
}

// Partial-pivot LU is a cheaper intermediate range normalization. It keeps
// the same column span without requiring orthogonal columns; the final round
// still uses QR before U/W projection. Explicitly zero deflated columns.
template<int N,int LD>
__device__ __forceinline__ void row_lu(float* ys,float* zs,float (&v)[N],int lane,int r) {
  float a[16],n0[16];
  #pragma unroll
  for(int c=0;c<16;++c) {
    a[c]=c<r?ys[lane*LD+c]:0.f;
    float mx=fabsf(a[c]);
    #pragma unroll
    for(int d=16;d;d/=2) mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
    n0[c]=mx;
  }
  bool used=false;
  #pragma unroll
  for(int j=0;j<16;++j) {
    float mx=used?0.f:fabsf(a[j]);
    #pragma unroll
    for(int d=16;d;d/=2) mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));
    unsigned candidates=__ballot_sync(0xffffffffu,!used && fabsf(a[j])==mx);
    int pivot=candidates?__ffs(candidates)-1:0;
    float value=__shfl_sync(0xffffffffu,a[j],pivot);
    bool keep=mx>1.e-12f && mx*mx>n0[j]*n0[j]*K3_RANK_TOL_SQ;
    float inv=keep?1.f/value:0.f;
    a[j]=used?0.f:a[j]*inv;
    #pragma unroll
    for(int c=j+1;c<16;++c) {
      float p=__shfl_sync(0xffffffffu,a[c],pivot);
      a[c]=fmaf(-a[j],p,a[c]);
    }
    used=used || (keep && lane==pivot);
  }
  #pragma unroll
  for(int c=0;c<16;++c) zs[lane*LD+c]=a[c];
  __syncwarp();
  #pragma unroll
  for(int i=0;i<N;++i) v[i]=lane<r?zs[i*LD+lane]:0.f;
}

// Rank-aware Cholesky QR in fp64. Unlike the K2 library probe, a zero or
// negative numerical pivot is explicitly deflated. Both the Gram and the
// triangular solve use double precision before storing the orthogonal basis.
template<int N>
__device__ __forceinline__ void chol_qr(float (&v)[N],int lane,int r) {
  if(lane<16) {
    constexpr unsigned mask=0x0000ffffu;
    double g[16],t[16];
    #pragma unroll
    for(int j=0;j<16;++j) {
      double a=0,b=0,c=0,d=0;
      #pragma unroll
      for(int i=0;i<N;i+=4) {
        a=fma(double(v[i]),double(__shfl_sync(mask,v[i],j,16)),a);
        b=fma(double(v[i+1]),double(__shfl_sync(mask,v[i+1],j,16)),b);
        c=fma(double(v[i+2]),double(__shfl_sync(mask,v[i+2],j,16)),c);
        d=fma(double(v[i+3]),double(__shfl_sync(mask,v[i+3],j,16)),d);
      }
      g[j]=(a+b)+(c+d);
    }
    double original=0;
    #pragma unroll
    for(int i=0;i<16;++i) if(lane==i) original=g[i];
    #pragma unroll
    for(int k=0;k<16;++k) {
      double pivot=__shfl_sync(mask,g[k],k,16);
      double n0=__shfl_sync(mask,original,k,16);
      double inv=pivot>1.e-24 && pivot>n0*double(K3_RANK_TOL_SQ)?rsqrt(pivot):0.0;
      double lj=lane>=k?g[k]*inv:0.0;
      #pragma unroll
      for(int i=0;i<16;++i) {
        double li=__shfl_sync(mask,lj,i,16);
        if(i>k && lane>k) g[i]=fma(-li,lj,g[i]);
      }
      g[k]=lj;
    }
    #pragma unroll
    for(int i=15;i>=0;--i) {
      double rhs=double(i==lane);
      #pragma unroll
      for(int k=0;k<16;++k) if(k>i) rhs=fma(-__shfl_sync(mask,g[i],k,16),t[k],rhs);
      double diag=__shfl_sync(mask,g[i],i,16);
      t[i]=diag>0?rhs/diag:0.0;
    }
    #pragma unroll
    for(int i=0;i<N;++i) {
      double out=0;
      #pragma unroll
      for(int k=0;k<16;++k) out=fma(double(__shfl_sync(mask,v[i],k,16)),t[k],out);
      v[i]=lane<r?float(out):0.f;
    }
  }
  __syncwarp();
}

template<int N,typename scalar_t,int GROUP=4,int POWER=1,bool PROJECT=true,int ROWS=0>
__global__ void tensor_project(const float* __restrict__ gram,float* __restrict__ z,
                         const int* __restrict__ active,int batch,int r,int iters,int passes,
                         scalar_t* u,scalar_t* w,int* count,const void* indices,
                         bool idx64,int64_t stride,int h,int full) {
  int lane=threadIdx.x%32,head=blockIdx.x*GROUP+threadIdx.x/32;
  if(head>=batch || !active[head] || lane>=N) return;
  constexpr unsigned mask=N==32?0xffffffffu:0xffffu;
  constexpr int LD=GROUP==1?N+4:N;
  constexpr int BD=GROUP==1?20:16;
  __shared__ float shared[GROUP][3][N*LD];
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
  for(int i=0;i<N;++i) gs[i*LD+lane]=g[i];
  for(int it=0;it<iters;++it) {
    for(int power=0;power<POWER;++power) {

    #pragma unroll
    for(int i=0;i<N;++i) zs[i*LD+lane]=v[i];
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
        wmma::load_matrix_sync(ah,gs+tile*16*LD+k,LD);
        wmma::load_matrix_sync(bh,zs+k*LD,LD);
        #pragma unroll
        for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
        #pragma unroll
        for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
        wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
      }
      wmma::store_matrix_sync(ys+tile*16*LD,acc,LD,wmma::mem_row_major);
    }
    __syncwarp(mask);
    float y[N];
    #pragma unroll
    for(int i=0;i<N;++i) y[i]=lane<r?ys[i*LD+lane]:0.f;

    #pragma unroll
    for(int i=0;i<N;++i) v[i]=y[i];
    }
    if constexpr (ROWS == 1) {row_qr<N,LD>(ys,zs,v,lane,r,passes);}
    else if constexpr (ROWS == 256) {row_lu<N,LD>(ys,zs,v,lane,r);}
    else if constexpr (ROWS == 128) {if(it+1<iters) row_lu<N,LD>(ys,zs,v,lane,r); else group_qr<N,LD,2>(ys,v,lane,r,passes);}
    else if constexpr (ROWS == 64) {chol_qr<N>(v,lane,r);}
    else if constexpr (ROWS > 1) {group_qr<N,LD,ROWS>(ys,v,lane,r,passes);}
    else {
    float norm0=local_dot<N>(v,v);
    for(int pass=0;pass<(passes<0?1:passes);++pass) {
      for(int j=0;j<r;++j) {
        float norm=local_dot<N>(v,v);
        float inv=norm>1.e-24f && (pass || norm>norm0*K3_RANK_TOL_SQ)?rsqrtf(norm):0.f;
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
  }
  if constexpr(ROWS==256) chol_final<N,LD>(v,gs,zs,ys,lane,r);
  if(passes<0) polar_correct<N,LD>(v,gs,zs,ys,lane,r);
  if constexpr (!PROJECT) {
    #pragma unroll
    for(int i=0;i<N;++i) z[head*N*N+i*N+lane]=lane<r?v[i]:0.f;
    return;
  }
  #pragma unroll
  for(int i=0;i<N;++i) zs[i*LD+lane]=lane<r?v[i]:0.f;
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
        gs[row*BD+col]=row<full?float(ptr[sh*N*128+row*128+d+col]):0.f;
      }
      __syncwarp(mask);
      wmma::fragment<wmma::accumulator,16,16,8,float> acc;
      wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<N;k+=8) {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> ah,al;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
        wmma::load_matrix_sync(ah,zs+k*LD,LD);wmma::load_matrix_sync(bh,gs+k*BD,BD);
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
void tensorvectors(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16,1,1,false><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float,1,1,false><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tensorrows(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16,1,1,false,true><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float,1,1,false,true><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tensorlanes2(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16,1,1,false,2><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float,1,1,false,2><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tensorlanes4(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16,1,1,false,4><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float,1,1,false,4><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void tensorlanes8(torch::Tensor gram,torch::Tensor z,torch::Tensor active,int64_t r,int64_t iters,int64_t passes,
                   torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t full) {
  int batch=gram.size(0),h=u.size(1);auto stream=at::cuda::getCurrentCUDAStream();
  bool idx64=indices.scalar_type()==torch::kInt64;
  if(u.scalar_type()==torch::kBFloat16)
    tensor_project<32,c10::BFloat16,1,1,false,8><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
      u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,full);
  else
    tensor_project<32,float,1,1,false,8><<<batch,32,0,stream>>>(gram.data_ptr<float>(),z.data_ptr<float>(),active.data_ptr<int>(),batch,r,iters,passes,
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

template<typename scalar_t,int ROWS=2,int POWER=1,bool DEBUG=false>
__global__ __launch_bounds__(128) void whole_tensor(scalar_t* u,scalar_t* w,int* count,const void* indices,
                            bool idx64,int64_t stride,int h,int r,int full,int iters,int passes,unsigned long long* stats=nullptr,int64_t layer_stride=0) {
  constexpr int N=32,LD=36;
  int head=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
  int64_t slot=idx64?static_cast<const int64_t*>(indices)[(head/h)*stride]:static_cast<const int*>(indices)[(head/h)*stride];
  if(slot<0) return;
  int64_t sh=(int64_t(blockIdx.y)*layer_stride+slot)*h+head%h;
  if(count[sh]<full) return;
  if(DEBUG && stats && threadIdx.x==0) stats[head*8]=clock64();
  constexpr unsigned mask=0xffffffffu;
  // Gram input and projection tiles have disjoint lifetimes. Reuse the
  // same shared arena to stay below the static 48 KiB limit, including fp32.
  __shared__ __align__(32) unsigned char arena[4*2*N*20*sizeof(float)];
  scalar_t* wm=reinterpret_cast<scalar_t*>(arena);
  float (*proj)[2][N*20]=reinterpret_cast<float (*)[2][N*20]>(arena);
  __shared__ __align__(32) float mats[3][N*LD];
  float* gs=mats[0];float* zs=mats[1];float* ys=mats[2];
  constexpr int PACK=16/sizeof(scalar_t);
  #pragma unroll
  for(int k=0;k<N*128/PACK/128;++k) {
    int vec=k*128+threadIdx.x;
    reinterpret_cast<uint4*>(wm)[vec]=vec*PACK/128<full?
      reinterpret_cast<const uint4*>(w+sh*N*128)[vec]:make_uint4(0,0,0,0);
  }
  __syncthreads();
  using namespace nvcuda;
  int row=warp/2*16,col=warp%2*16;
  if constexpr (std::is_same<scalar_t,c10::BFloat16>::value) {
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int k=0;k<128;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
      wmma::load_matrix_sync(a,reinterpret_cast<const __nv_bfloat16*>(wm)+row*128+k,128);
      wmma::load_matrix_sync(b,reinterpret_cast<const __nv_bfloat16*>(wm)+col*128+k,128);
      wmma::mma_sync(acc,a,b,acc);
    }
    wmma::store_matrix_sync(gs+row*LD+col,acc,LD,wmma::mem_row_major);
  } else {
    wmma::fragment<wmma::accumulator,16,16,8,float> acc;wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int k=0;k<128;k+=8) {
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
      wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> bh,bl;
      wmma::load_matrix_sync(ah,wm+row*128+k,128);wmma::load_matrix_sync(bh,wm+col*128+k,128);
      #pragma unroll
      for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
      #pragma unroll
      for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
      wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
    }
    wmma::store_matrix_sync(gs+row*LD+col,acc,LD,wmma::mem_row_major);
  }
  __syncthreads();
  if(DEBUG && stats && threadIdx.x==0) stats[head*8+1]=clock64();
  if(warp==0) {
  float g[N],v[N];
  float diag=0;
  #pragma unroll
  for(int i=0;i<N;++i) {g[i]=gs[i*LD+lane]; if(lane==i) diag=g[i];}
  int rank=0;
  #pragma unroll
  for(int j=0;j<N;++j) {float d=__shfl_sync(mask,diag,j,N);rank+=(d>diag || (d==diag && j<lane));}
  #pragma unroll
  for(int i=0;i<N;++i) {int rr=__shfl_sync(mask,rank,i,N);v[i]=float(rr==lane && lane<r);}
  #pragma unroll
  for(int i=0;i<N;++i) gs[i*LD+lane]=g[i];
  for(int it=0;it<iters;++it) {
    for(int power=0;power<POWER;++power) {

    #pragma unroll
    for(int i=0;i<N;++i) zs[i*LD+lane]=v[i];
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
        wmma::load_matrix_sync(ah,gs+tile*16*LD+k,LD);
        wmma::load_matrix_sync(bh,zs+k*LD,LD);
        #pragma unroll
        for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
        #pragma unroll
        for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
        if constexpr(!K3_FAST_GZ) {wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);}
        wmma::mma_sync(acc,ah,bh,acc);
      }
      wmma::store_matrix_sync(ys+tile*16*LD,acc,LD,wmma::mem_row_major);
    }
    __syncwarp(mask);
    float y[N];
    #pragma unroll
    for(int i=0;i<N;++i) y[i]=lane<r?ys[i*LD+lane]:0.f;

    #pragma unroll
    for(int i=0;i<N;++i) v[i]=y[i];
    }
    if constexpr (ROWS == 1) {row_qr<N,LD>(ys,zs,v,lane,r,passes);}
    else if constexpr (ROWS == 256) {row_lu<N,LD>(ys,zs,v,lane,r);}
    else if constexpr (ROWS == 128) {if(it+1<iters) row_lu<N,LD>(ys,zs,v,lane,r); else group_qr<N,LD,2>(ys,v,lane,r,passes);}
    else if constexpr (ROWS == 64) {chol_qr<N>(v,lane,r);}
    else if constexpr (ROWS > 1) {group_qr<N,LD,ROWS>(ys,v,lane,r,passes);}
    else {
    float norm0=local_dot<N>(v,v);
    for(int pass=0;pass<(passes<0?1:passes);++pass) {
      for(int j=0;j<r;++j) {
        float norm=local_dot<N>(v,v);
        float inv=norm>1.e-24f && (pass || norm>norm0*K3_RANK_TOL_SQ)?rsqrtf(norm):0.f;
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
  }
  if(DEBUG && stats && lane==0) stats[head*8+5]=clock64();
  if constexpr(ROWS==256) chol_final<N,LD>(v,gs,zs,ys,lane,r);
  if(passes<0) polar_correct<N,LD>(v,gs,zs,ys,lane,r);

    if(DEBUG && stats && lane==0) stats[head*8+6]=clock64();
    #pragma unroll
    for(int i=0;i<N;++i) zs[i*LD+lane]=lane<r?v[i]:0.f;
  }
  __syncthreads();
  auto* zbf=reinterpret_cast<__nv_bfloat16*>(ys);
  if constexpr(K3_FAST_PROJECT && std::is_same<scalar_t,c10::BFloat16>::value) {
    #pragma unroll
    for(int i=0;i<N*16/128;++i) {
      int off=i*128+threadIdx.x;float z=zs[(off/16)*LD+off%16];
      zbf[off]=__float2bfloat16(z);
      if constexpr(K3_FAST_PROJECT>1) zbf[N*16+off]=__float2bfloat16(z-__bfloat162float(zbf[off]));
    }
    __syncthreads();
  }
  float* ps=proj[warp][0];float* result=proj[warp][1];
  for(int factor=0;factor<2;++factor) {
    scalar_t* ptr=factor?w:u;
    for(int d=warp*16;d<128;d+=64) {
      if constexpr(K3_FAST_PROJECT && std::is_same<scalar_t,c10::BFloat16>::value) {
        auto* pb=reinterpret_cast<__nv_bfloat16*>(ps);
        #pragma unroll
        for(int j=0;j<2;++j) {
          int vec=j*32+lane,rr=vec/2,cc=(vec%2)*8;
          reinterpret_cast<uint4*>(pb)[vec]=rr<full?
            *reinterpret_cast<const uint4*>(ptr+sh*N*128+rr*128+d+cc):make_uint4(0,0,0,0);
        }
        __syncwarp();
        wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
        #pragma unroll
        for(int k=0;k<N;k+=16) {
          wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> qa;
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> vb;
          wmma::load_matrix_sync(vb,pb+k*16,16);
          if constexpr(K3_FAST_PROJECT>1) {
            wmma::load_matrix_sync(qa,zbf+N*16+k*16,16);
            wmma::mma_sync(acc,qa,vb,acc);
          }
          wmma::load_matrix_sync(qa,zbf+k*16,16);
          wmma::mma_sync(acc,qa,vb,acc);
        }
        wmma::store_matrix_sync(result,acc,16,wmma::mem_row_major);
      } else {
      #pragma unroll
      for(int j=0;j<N*16/32;++j) {
        int off=j*32+lane,rr=off/16,cc=off%16;
        ps[rr*20+cc]=rr<full?float(ptr[sh*N*128+rr*128+d+cc]):0.f;
      }
      __syncwarp();
      wmma::fragment<wmma::accumulator,16,16,8,float> acc;wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<N;k+=8) {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> ah,al;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
        wmma::load_matrix_sync(ah,zs+k*LD,LD);wmma::load_matrix_sync(bh,ps+k*20,20);
        #pragma unroll
        for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
        #pragma unroll
        for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
        wmma::mma_sync(acc,al,bh,acc);
        if constexpr (!std::is_same<scalar_t,c10::BFloat16>::value) wmma::mma_sync(acc,ah,bl,acc);
        wmma::mma_sync(acc,ah,bh,acc);
      }
      wmma::store_matrix_sync(result,acc,16,wmma::mem_row_major);
      }
      __syncwarp();

      #pragma unroll
      for(int j=0;j<8;++j) {int off=j*32+lane,rr=off/16,cc=off%16;if(rr<r) ptr[sh*N*128+rr*128+d+cc]=scalar_t(result[off]);}
      __syncwarp();
    }
  }
  if(threadIdx.x==0) count[sh]=r;
  if constexpr(DEBUG) __syncthreads();
  if(DEBUG && stats && threadIdx.x==0) stats[head*8+7]=clock64();
}

// Batch independent layer expiries at the last GDN layer, before any tracking
// copy or next-token state read. Same per-head math as wholelu, larger grid.
void layers(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  TORCH_CHECK(u.dim()==5 && w.sizes()==u.sizes() && count.dim()==3 && u.size(3)==32 && u.size(4)==128);
  TORCH_CHECK(u.is_contiguous() && w.is_contiguous() && count.is_contiguous());
  int h=u.size(2),num_layers=u.size(0);int64_t slots=u.size(1);
  dim3 grid(indices.numel()*h,num_layers);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) whole_tensor<c10::BFloat16,128,1><<<grid,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes,nullptr,slots);
  else whole_tensor<float,128,1><<<grid,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes,nullptr,slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wholestages(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes,torch::Tensor stats,int64_t which) {
  TORCH_CHECK(u.scalar_type()==torch::kBFloat16 && stats.scalar_type()==torch::kInt64);
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  auto run=[&](auto rows) {
    whole_tensor<c10::BFloat16,decltype(rows)::value,1,true><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes,reinterpret_cast<unsigned long long*>(stats.data_ptr<int64_t>()));
  };
  if(which==128) run(std::integral_constant<int,128>{});
  else if(which==64) run(std::integral_constant<int,64>{});
  else run(std::integral_constant<int,2>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void whole(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) whole_tensor<c10::BFloat16><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else whole_tensor<float><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void wholeluchol(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) whole_tensor<c10::BFloat16,256,1><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else whole_tensor<float,256,1><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wholelucholp2(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) whole_tensor<c10::BFloat16,256,2><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else whole_tensor<float,256,2><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wholelu(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) whole_tensor<c10::BFloat16,128,1><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else whole_tensor<float,128,1><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wholelup2(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) whole_tensor<c10::BFloat16,128,2><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else whole_tensor<float,128,2><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wholechol(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) whole_tensor<c10::BFloat16,64,1><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else whole_tensor<float,64,1><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void wholecholp2(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) whole_tensor<c10::BFloat16,64,2><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else whole_tensor<float,64,2><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// One warp per retained column. Ping-pong broadcast buffers need only one
// block barrier per QR column: the next writer never overwrites the preceding
// column while another warp is still reading it.
template<typename scalar_t,int L=32,bool PANEL=false>
__global__ __launch_bounds__(16*L,1) void parallel_tensor(scalar_t* u,scalar_t* w,int* count,const void* indices,
                             bool idx64,int64_t stride,int h,int r,int full,int iters,int passes) {
  constexpr int N=32,LD=36,NR=32/L,NW=L/2;
  int head=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
  int col_id=threadIdx.x/L,part=threadIdx.x%L;
  int64_t slot=idx64?static_cast<const int64_t*>(indices)[(head/h)*stride]:static_cast<const int*>(indices)[(head/h)*stride];
  if(slot<0) return;
  int64_t sh=slot*h+head%h;
  if(count[sh]<full) return;
  __shared__ __align__(32) float matrix[N*128];
  scalar_t* wm=reinterpret_cast<scalar_t*>(matrix);
  __shared__ __align__(32) float basis[N*LD];
  __shared__ __align__(32) float scratch[2*N*LD];
  __shared__ float broadcast[2][32];
  __shared__ float panel_q[2][4][32];
  __shared__ float inv_shared[2];
  __shared__ __align__(32) __nv_bfloat16 zbf[N*16*(K3_FAST_PROJECT>1?2:1)];
  float* gs=scratch;float* ys=scratch+N*LD;float* zs=basis;
  constexpr int PACK=16/sizeof(scalar_t),NT=16*L;
  #pragma unroll
  for(int k=0;k<N*128/PACK/NT;++k) {
    int vec=k*NT+threadIdx.x;
    reinterpret_cast<uint4*>(wm)[vec]=vec*PACK/128<full?
      reinterpret_cast<const uint4*>(w+sh*N*128)[vec]:make_uint4(0,0,0,0);
  }
  __syncthreads();
  using namespace nvcuda;
  for(int tile=warp;tile<4;tile+=NW) {
  int row=tile/2*16,col=tile%2*16;
  if constexpr (std::is_same<scalar_t,c10::BFloat16>::value) {
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int k=0;k<128;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
      wmma::load_matrix_sync(a,reinterpret_cast<const __nv_bfloat16*>(wm)+row*128+k,128);
      wmma::load_matrix_sync(b,reinterpret_cast<const __nv_bfloat16*>(wm)+col*128+k,128);
      wmma::mma_sync(acc,a,b,acc);
    }
    wmma::store_matrix_sync(gs+row*LD+col,acc,LD,wmma::mem_row_major);
  } else {
    wmma::fragment<wmma::accumulator,16,16,8,float> acc;wmma::fill_fragment(acc,0.f);
    #pragma unroll
    for(int k=0;k<128;k+=8) {
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
      wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> bh,bl;
      wmma::load_matrix_sync(ah,wm+row*128+k,128);wmma::load_matrix_sync(bh,wm+col*128+k,128);
      #pragma unroll
      for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
      #pragma unroll
      for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
      wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
    }
    wmma::store_matrix_sync(gs+row*LD+col,acc,LD,wmma::mem_row_major);
  }

  }

  __syncthreads();
  float diag=gs[lane*LD+lane];int rank=0;
  #pragma unroll
  for(int j=0;j<N;++j) {float d=__shfl_sync(0xffffffffu,diag,j);rank+=(d>diag || (d==diag && j<lane));}
  // Column-major basis, one 32-row column per warp.
  #pragma unroll
  for(int i=0;i<NR;++i) {int row=i*L+part;int rr=__shfl_sync(0xffffffffu,rank,row);basis[col_id*N+row]=(rr==col_id && col_id<r)?1.f:0.f;}
  __syncthreads();
  for(int it=0;it<iters;++it) {
    for(int tile=warp;tile<2;tile+=NW) {
      wmma::fragment<wmma::accumulator,16,16,8,float> acc;wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<N;k+=8) {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> bh,bl;
        wmma::load_matrix_sync(ah,gs+tile*16*LD+k,LD);wmma::load_matrix_sync(bh,basis+k,N);
        #pragma unroll
        for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
        #pragma unroll
        for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
        wmma::mma_sync(acc,al,bh,acc);wmma::mma_sync(acc,ah,bl,acc);wmma::mma_sync(acc,ah,bh,acc);
      }
      wmma::store_matrix_sync(ys+tile*16,acc,N,wmma::mem_col_major);
    }
    __syncthreads();
    float x[NR];
    #pragma unroll
    for(int i=0;i<NR;++i) x[i]=col_id<r?ys[col_id*N+i*L+part]:0.f;
    float n0=group_sum<L>(local_dot<NR>(x,x));
    if constexpr(PANEL) {
      static_assert(L==8,"four columns per warp");
      for(int pass=0;pass<(passes<0?1:passes);++pass) {
        #pragma unroll
        for(int panel=0;panel<4;++panel) {
          if(warp==panel) {
            #pragma unroll
            for(int j=0;j<4;++j) {
              float q[NR];
              #pragma unroll
              for(int i=0;i<NR;++i) q[i]=__shfl_sync(0xffffffffu,x[i],j*L+part);
              float original=__shfl_sync(0xffffffffu,n0,j*L+part);
              float norm=group_sum<L>(local_dot<NR>(q,q));
              float inv=norm>1.e-24f && (pass || norm>original*K3_RANK_TOL_SQ)?rsqrtf(norm):0.f;
              #pragma unroll
              for(int i=0;i<NR;++i) q[i]*=inv;
              float dot=group_sum<L>(local_dot<NR>(q,x));
              #pragma unroll
              for(int i=0;i<NR;++i) {
                if(col_id%4==j) x[i]=q[i];
                else if(col_id%4>j) x[i]=fmaf(-dot,q[i],x[i]);
              }
            }
            #pragma unroll
            for(int i=0;i<NR;++i) panel_q[panel%2][col_id%4][i*L+part]=x[i];
          }
          // Alternate panel buffers: the next producer cannot overwrite a
          // panel still being consumed, so only one CTA barrier per panel.
          __syncthreads();
          if(warp>panel) {
            #pragma unroll
            for(int j=0;j<4;++j) {
              float q[NR];
              #pragma unroll
              for(int i=0;i<NR;++i) q[i]=panel_q[panel%2][j][i*L+part];
              float dot=group_sum<L>(local_dot<NR>(q,x));
              #pragma unroll
              for(int i=0;i<NR;++i) x[i]=fmaf(-dot,q[i],x[i]);
            }
          }
        }
      }
    } else {
    for(int pass=0;pass<(passes<0?1:passes);++pass) {
      #pragma unroll
      for(int j=0;j<16;++j) {
        if(col_id==j) {
          float norm=group_sum<L>(local_dot<NR>(x,x));
          float inv=norm>1.e-24f && (pass || norm>n0*K3_RANK_TOL_SQ)?rsqrtf(norm):0.f;
          #pragma unroll
          for(int i=0;i<NR;++i) {x[i]*=inv;broadcast[j%2][i*L+part]=x[i];}
          if(part==0) inv_shared[j%2]=inv;
        }
        __syncthreads();
        if(col_id>j && inv_shared[j%2]>0.f) {
          float q[NR];
          #pragma unroll
          for(int i=0;i<NR;++i) q[i]=broadcast[j%2][i*L+part];
          float dot=group_sum<L>(local_dot<NR>(q,x));
          #pragma unroll
          for(int i=0;i<NR;++i) x[i]=fmaf(-dot,q[i],x[i]);
        }
      }
    }
    }
    #pragma unroll
    for(int i=0;i<NR;++i) basis[col_id*N+i*L+part]=x[i];
    __syncthreads();
  }
  // Convert to the common row-major projection basis. scratch may now replace G.
  float x[NR];
  #pragma unroll
  for(int i=0;i<NR;++i) x[i]=basis[col_id*N+i*L+part];
  __syncthreads();
  #pragma unroll
  for(int i=0;i<NR;++i) basis[(i*L+part)*LD+col_id]=x[i];
  for(int off=threadIdx.x;off<N*16;off+=blockDim.x) basis[(off/16)*LD+off%16+16]=0.f;
  __syncthreads();
  if(passes<0 && warp==0) {
    float v[N];
    #pragma unroll
    for(int i=0;i<N;++i) v[i]=lane<r?basis[i*LD+lane]:0.f;
    polar_correct<N,LD>(v,gs,zs,ys,lane,r);
    #pragma unroll
    for(int i=0;i<N;++i) basis[i*LD+lane]=lane<r?v[i]:0.f;
  }
  __syncthreads();
  if constexpr (K3_FAST_PROJECT && std::is_same<scalar_t,c10::BFloat16>::value) {
    for(int off=threadIdx.x;off<N*16;off+=blockDim.x) {
      float z=basis[(off/16)*LD+off%16];zbf[off]=__float2bfloat16_rn(z);
      if constexpr(K3_FAST_PROJECT>1) zbf[N*16+off]=__float2bfloat16_rn(z-__bfloat162float(zbf[off]));
    }
    __syncthreads();
  }
  for(int factor=0;factor<2;++factor) {
    scalar_t* ptr=factor?w:u;
    if constexpr (K3_FAST_PROJECT && std::is_same<scalar_t,c10::BFloat16>::value) {
      #pragma unroll
      for(int k=0;k<N*128/PACK/NT;++k) {
        int vec=k*NT+threadIdx.x;
        reinterpret_cast<uint4*>(wm)[vec]=vec*PACK/128<full?
          reinterpret_cast<const uint4*>(ptr+sh*N*128)[vec]:make_uint4(0,0,0,0);
      }
    } else {
      for(int off=threadIdx.x;off<N*128;off+=blockDim.x) matrix[off]=off/128<full?float(ptr[sh*N*128+off]):0.f;
    }
    __syncthreads();
    for(int tile=warp;tile<8;tile+=NW) {
      int d=tile*16;
      if constexpr (K3_FAST_PROJECT && std::is_same<scalar_t,c10::BFloat16>::value) {
        wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
        #pragma unroll
        for(int k=0;k<N;k+=16) {
          wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> qa;
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> vb;
          wmma::load_matrix_sync(vb,reinterpret_cast<const __nv_bfloat16*>(wm)+k*128+d,128);
          if constexpr(K3_FAST_PROJECT>1) {
            wmma::load_matrix_sync(qa,zbf+N*16+k*16,16);
            wmma::mma_sync(acc,qa,vb,acc);
          }
          wmma::load_matrix_sync(qa,zbf+k*16,16);
          wmma::mma_sync(acc,qa,vb,acc);
        }
        wmma::store_matrix_sync(scratch+d,acc,128,wmma::mem_row_major);
      } else {
      wmma::fragment<wmma::accumulator,16,16,8,float> acc;wmma::fill_fragment(acc,0.f);
      #pragma unroll
      for(int k=0;k<N;k+=8) {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> ah,al;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
        wmma::load_matrix_sync(ah,basis+k*LD,LD);
        wmma::load_matrix_sync(bh,matrix+k*128+d,128);
        #pragma unroll
        for(int j=0;j<ah.num_elements;++j) {float f=ah.x[j];ah.x[j]=to_tf32(f);al.x[j]=to_tf32(f-ah.x[j]);}
        #pragma unroll
        for(int j=0;j<bh.num_elements;++j) {float f=bh.x[j];bh.x[j]=to_tf32(f);bl.x[j]=to_tf32(f-bh.x[j]);}
        wmma::mma_sync(acc,al,bh,acc);
        if constexpr (!std::is_same<scalar_t,c10::BFloat16>::value) wmma::mma_sync(acc,ah,bl,acc);
        wmma::mma_sync(acc,ah,bh,acc);
      }
      wmma::store_matrix_sync(scratch+d,acc,128,wmma::mem_row_major);
      }
    }
    __syncthreads();
    for(int off=threadIdx.x;off<16*128;off+=blockDim.x) if(off/128<r) ptr[sh*N*128+off]=scalar_t(scratch[off]);
    __syncthreads();
  }
  if(threadIdx.x==0) count[sh]=r;
}

void panel(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) parallel_tensor<c10::BFloat16,8,true><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else parallel_tensor<float,8,true><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void parallel(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) parallel_tensor<c10::BFloat16><<<batch,512,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else parallel_tensor<float><<<batch,512,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void parallel4(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) parallel_tensor<c10::BFloat16,4><<<batch,64,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else parallel_tensor<float,4><<<batch,64,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void parallel8(torch::Tensor u,torch::Tensor w,torch::Tensor count,torch::Tensor indices,int64_t r,int64_t full,int64_t iters,int64_t passes) {
  int batch=indices.numel()*u.size(1),h=u.size(1);bool idx64=indices.scalar_type()==torch::kInt64;
  auto stream=at::cuda::getCurrentCUDAStream();
  if(u.scalar_type()==torch::kBFloat16) parallel_tensor<c10::BFloat16,8><<<batch,128,0,stream>>>(u.data_ptr<c10::BFloat16>(),w.data_ptr<c10::BFloat16>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  else parallel_tensor<float,8><<<batch,128,0,stream>>>(u.data_ptr<float>(),w.data_ptr<float>(),count.data_ptr<int>(),indices.data_ptr(),idx64,indices.stride(0),h,r,full,iters,passes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("eig", &eig);m.def("wholestages", &wholestages);m.def("layers", &layers);m.def("whole", &whole);m.def("wholeluchol", &wholeluchol);m.def("wholelucholp2", &wholelucholp2);m.def("wholelu", &wholelu);m.def("wholelup2", &wholelup2);m.def("wholechol", &wholechol);m.def("wholecholp2", &wholecholp2);m.def("panel", &panel);m.def("parallel", &parallel);m.def("parallel4", &parallel4);m.def("parallel8", &parallel8);m.def("mgs", &mgs);m.def("tensormgs", &tensormgs);m.def("tensorproject", &tensorproject);m.def("tensorproject1", &tensorproject1);m.def("tensorvectors", &tensorvectors);m.def("tensorrows", &tensorrows);m.def("tensorlanes2", &tensorlanes2);m.def("tensorlanes4", &tensorlanes4);m.def("tensorlanes8", &tensorlanes8);m.def("tensorprojectp2", &tensorprojectp2);m.def("tensorprojectp3", &tensorprojectp3);m.def("workspace", &workspace);m.def("eiglib", &eiglib);}
