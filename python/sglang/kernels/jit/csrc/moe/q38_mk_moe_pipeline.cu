// SPDX-License-Identifier: Apache-2.0
// FN persistent expert slices, using the Q35 TMA/MMA/epilogue warp split.
// FC2 accumulates this CTA's intermediate slice; its cluster combines FP32
// partials in slice order before native BF16 expert-row rounding.
#include "q38_mk_moe_primitives.cuh"
namespace q38_mk_pipe {
using namespace q38_mk;
#ifndef Q38_MK_I
#define Q38_MK_I 192
#endif
constexpr int H=2560,I=Q38_MK_I,K=10,Rows=16,Threads=256;
constexpr int TileI=I==640?128:64, C=I/TileI, N1=2*TileI, Nk=TileI/64;
constexpr int Stages=I==640?2:5, StageBytes=N1*144, SfOffset=N1*128;
constexpr int SfStride=N1==128?16:32, PartStride=260;
constexpr int Sw2=TileI==128?4:6, Sbo2=TileI*4;
struct Params {
 const uint8_t *x,*xsf,*sf13,*sf2;
 const float *g1,*g1c,*g2;
 const int *ids,*unique_ids,*num_unique;
 __nv_bfloat16* out;
 int m;
};
struct __align__(1024) Smem {
 alignas(1024) uint8_t ring[Stages][StageBytes];
 alignas(1024) uint8_t x[10][2048];
 alignas(1024) uint8_t h[16384]; // backing for all128 TCGEN operand rows
 uint32_t xsf[Rows][40],hsf[Rows][Nk];
 float acc[Rows][N1+4];
 alignas(1024) float partial[C][Rows][PartStride];
 uint64_t full[Stages],empty[Stages],sf_ready[Stages];
 uint64_t fc1_done,h_ready,tile_done,tile_free;
 uint64_t reduce_ready[10],reduce_ack[10];
 uint32_t tmem;
 int slots[Rows];
};
static_assert(sizeof(Smem)+1024<=233472,"FN pipeline SMEM");
#ifdef Q38_MK_SMEM_BYTES
static_assert(sizeof(Smem)+1024==Q38_MK_SMEM_BYTES,"FN pipeline launcher SMEM");
#endif
__device__ __forceinline__ void wg_sync(){asm volatile("bar.sync 1,128;":::"memory");}
__device__ __forceinline__ void cluster_sync(){cluster_arrive();cluster_wait();}
__device__ __forceinline__ void st4(uint32_t t,const uint4& u){
 asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
  ::"r"(t),"r"(u.x),"r"(u.y),"r"(u.z),"r"(u.w):"memory");
}
__device__ __forceinline__ void copy_peer(uint32_t dst,uint32_t src,uint32_t bytes,uint32_t bar){
 asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
  ::"r"(dst),"r"(src),"r"(bytes),"r"(bar):"memory");
}
__device__ __forceinline__ void arrive_peer(uint32_t bar){
 asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(bar):"memory");
}
__device__ __forceinline__ void ld32(uint32_t addr,uint32_t (&r)[32]){
 asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
  : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ld64(uint32_t addr,uint32_t (&r)[64]){
 asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
  : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]) : "r"(addr) : "memory");
}
template<int N> __device__ __forceinline__ void drain(uint32_t addr,uint32_t (&r)[N]){
 if constexpr(N==32)ld32(addr,r);else ld64(addr,r);
}
extern "C" __global__ __cluster_dims__(C,1,1) __launch_bounds__(Threads,1)
void q38_mk_moe_pipeline(const __grid_constant__ CUtensorMap w13,
                       const __grid_constant__ CUtensorMap w2,const Params p){
 extern __shared__ __align__(1024) uint8_t raw[];
 Smem& s=*reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(raw)+1023)&~uintptr_t(1023));
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32,rank=cluster_rank(),cluster=blockIdx.x/C;
 const uint32_t partial_bytes=p.m*PartStride*4;
 if(tid==0){prefetch_tmap(&w13);prefetch_tmap(&w2);}
 asm volatile("griddepcontrol.wait;":::"memory");
 if(cluster>=*p.num_unique){asm volatile("griddepcontrol.launch_dependents;":::"memory");return;}
 if(warp==1){tmem_alloc(&s.tmem,512);tmem_relinquish();tc_fence_before();}
 __syncthreads();
 const uint32_t tm=s.tmem,idesc1=make_idesc_nvf4(128,N1),idesc2=make_idesc_nvf4(128,256);
 for(int item=cluster;item<*p.num_unique;item+=gridDim.x/C){
  const int e=p.unique_ids[item];
  if(tid==0){
   for(int b=0;b<Stages;++b){mbar_init(&s.full[b],1);mbar_init(&s.empty[b],1);mbar_init(&s.sf_ready[b],4);}
   mbar_init(&s.fc1_done,1);mbar_init(&s.h_ready,4);mbar_init(&s.tile_done,1);mbar_init(&s.tile_free,4);
   for(int tile=0;tile<10;++tile){
    mbar_init(&s.reduce_ready[tile],1);mbar_init(&s.reduce_ack[tile],1);
    if(tile%C==rank)mbar_expect_tx(&s.reduce_ready[tile],(C-1)*partial_bytes);
   }
   fence_mbar_init();
  }
  for(int j=tid;j<Rows*80;j+=Threads){
   int row=j/80,rem=j%80,st=rem/8,q=rem%8;uint4 v=make_uint4(0,0,0,0);
   if(row<p.m)v=*reinterpret_cast<const uint4*>(p.x+row*1280+st*128+q*16);
   *reinterpret_cast<uint4*>(s.x[st]+(row/8)*1024+(row%8)*128+((q^(row%8))*16))=v;
  }
  for(int j=tid;j<Rows*40;j+=Threads){int r=j/40,k=j%40;s.xsf[r][k]=r<p.m?*reinterpret_cast<const uint32_t*>(p.xsf+r*160+k*4):0;}
  for(int j=tid;j<sizeof(s.h)/16;j+=Threads)reinterpret_cast<uint4*>(s.h)[j]=make_uint4(0,0,0,0);
  if(tid<Rows){int slot=-1;if(tid<p.m)for(int k=0;k<K;++k)if(p.ids[tid*K+k]==e)slot=k;s.slots[tid]=slot;}
  __syncthreads();cluster_sync();fence_proxy_async_smem();
  const float ag=p.g1[e],sc=p.g1c[e],a2=p.g2[e];
  // Warp0: stream every FC1 stage and FC2 output tile through the TMA ring.
  if(warp==0){
   if(lane==0)for(int g=0;g<20;++g){
    const int b=g%Stages;
    if(g>=Stages)mbar_wait(&s.empty[b],((g/Stages)-1)&1);
    uint8_t* d=s.ring[b];
    if(g<10){
     mbar_expect_tx(&s.full[b],StageBytes);
     const int row=e*(2*I)+rank*N1;
     tma_load_2d(d,&w13,&s.full[b],g*128,row);
     if(N1==256)tma_load_2d(d+16384,&w13,&s.full[b],g*128,row+128);
     const uint8_t* sf=p.sf13+(size_t)e*(2*I)*160;
     bulk_load(d+SfOffset,sf+((rank*(N1/128))*40+g*4)*512,2048,&s.full[b]);
     if(N1==256)bulk_load(d+SfOffset+2048,sf+((rank*2+1)*40+g*4)*512,2048,&s.full[b]);
    }else{
     const int tile=g-10,row=tile*256;
     mbar_expect_tx(&s.full[b],256*(TileI/2)+1024*Nk);
     tma_load_2d(d,&w2,&s.full[b],rank*(TileI/2),e*H+row);
     const uint8_t* sf=p.sf2+(size_t)e*H*(I/16);
     bulk_load(d+SfOffset,sf+((row/128)*(I/64)+rank*Nk)*512,512*Nk,&s.full[b]);
     bulk_load(d+SfOffset+512*Nk,sf+((row/128+1)*(I/64)+rank*Nk)*512,512*Nk,&s.full[b]);
    }
   }
   __syncwarp();
  }else if(warp==1){
   // Warp1: MMA issue and ownership handoffs; never reuse the accumulator early.
   if(lane==0){
    for(int g=0;g<10;++g){
     const int b=g%Stages;
     mbar_wait(&s.sf_ready[b],(g/Stages)&1);tc_fence_after();
     for(int k=0;k<4;++k)umma_nvf4(tm,
      make_sdesc(smem_u32(s.x[g])+k*32,0,1024,2),make_sdesc(smem_u32(s.ring[b])+k*32,0,1024,2),idesc1,
      tm+256+b*16+k*4,tm+384+b*SfStride+k*(N1/32),(g|k)!=0);
     tc_commit(&s.empty[b]);
    }
    tc_commit(&s.fc1_done);mbar_wait(&s.h_ready,0);tc_fence_after();
    for(int tile=0;tile<10;++tile){
     if(tile)mbar_wait(&s.tile_free,(tile-1)&1);
     const int g=10+tile,b=g%Stages;
     mbar_wait(&s.sf_ready[b],(g/Stages)&1);tc_fence_after();
     for(int k=0;k<Nk;++k)umma_nvf4(tm,
      make_sdesc(smem_u32(s.h)+k*32,0,1024,2),make_sdesc(smem_u32(s.ring[b])+k*32,0,Sbo2,Sw2),idesc2,
      tm+256+b*16+k*4,tm+384+b*SfStride+k*8,k!=0);
     tc_commit(&s.empty[b]);tc_commit(&s.tile_done);
    }
   }
   __syncwarp();
  }else if(warp>=4){
   // Four epilogue warps stage scale factors, drain accumulators and exchange
   // FP32 FC2 partials. All SF writes precede their matching sf_ready arrival.
   const int q=warp-4;const uint32_t la=(uint32_t)(q*32)<<16;
   auto stage_sf=[&](int g){
    const int b=g%Stages,nk=g<10?4:Nk;
    mbar_wait(&s.full[b],(g/Stages)&1);
    for(int k=0;k<nk;++k){
     const uint8_t* src=s.ring[b]+SfOffset+k*512+lane*16;
     const uint4 u=*reinterpret_cast<const uint4*>(src);
     if(g<10&&N1==128)st4(tm+la+384+b*SfStride+k*4,u);
     else{
      const uint4 v=*reinterpret_cast<const uint4*>(src+(g<10?2048:Nk*512));
      const uint32_t vals[8]={u.x,u.y,u.z,u.w,v.x,v.y,v.z,v.w};
      tmem_st_x8(tm+la+384+b*SfStride+k*8,vals);
     }
    }
    uint32_t xa[16]={};
    for(int k=0;k<(g<10?4:Nk);++k)if(lane<Rows)xa[k*4]=g<10?s.xsf[lane][g*4+k]:s.hsf[lane][k];
    tmem_st_x16(tm+la+256+b*16,xa);tmem_st_wait();tc_fence_before();__syncwarp();
    if(lane==0)mbar_arrive(&s.sf_ready[b]);
   };
   for(int g=0;g<10;++g)stage_sf(g);
   mbar_wait(&s.fc1_done,0);tc_fence_after();
   // Only quarter0 can read token rows0..15. Other warps cannot use a row0
   // address to access quarter0: TMEM load access is restricted by warp id.
   if(q==0)for(int col=0;col<N1;col+=64){
    uint32_t v[64];ld64(tm+col,v);tmem_ld_wait();
    if(lane<Rows)for(int j=0;j<64;j+=4)
     *reinterpret_cast<uint4*>(&s.acc[lane][col+j])=make_uint4(v[j],v[j+1],v[j+2],v[j+3]);
   }
   wg_sync();
   const int et=q*32+lane;
   for(int task=et;task<Rows*(TileI/8);task+=128){
    const int row=task/(TileI/8),blk=(task/2)%(TileI/16),hf=task%2;
    const float4* ar=reinterpret_cast<const float4*>(s.acc[row]+32*blk+4*hf);
    const float4 ue=ar[0],ge=ar[2],uo=ar[4],go=ar[6];
    const float ga[8]={ge.x,go.x,ge.y,go.y,ge.z,go.z,ge.w,go.w};
    const float ua[8]={ue.x,uo.x,ue.y,uo.y,ue.z,uo.z,ue.w,uo.w};
    float h[8],amax=0;const float al2e=mul_ftz(ag,1.4426950216293334961f);
    for(int j=0;j<8;++j){const float sg=rcp_approx(ex2_approx(-(ga[j]*al2e))+1.f);h[j]=fmaf(ua[j],ag,0.f)*(ga[j]*sg);amax=fmaxf(amax,fabsf(h[j]));}
    amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,1));
    const uint32_t sf=cvt_e4m3(fmaxf(0.f,fabsf(amax*sc))*0.16666667163372039795f);
    const float sff=e4m3_to_f32(sf),osc=sff==0.f?0.f:rcp_approx(sff);
    float v[8];for(int j=0;j<8;++j)v[j]=mul_ftz(h[j],osc)*sc;
    const int off=(row/8)*1024+(row%8)*128+(((blk/2)^(row%8))*16)+(blk%2)*8+hf*4;
    *reinterpret_cast<uint32_t*>(s.h+off)=cvt_e2m1x8(v);
    if(hf==0)reinterpret_cast<uint8_t*>(&s.hsf[row][blk/4])[blk%4]=sf;
   }
   wg_sync();fence_proxy_async_smem();tc_fence_before();__syncwarp();
   if(lane==0)mbar_arrive(&s.h_ready);
   stage_sf(10);
   for(int tile=0;tile<10;++tile){
    mbar_wait(&s.tile_done,tile&1);tc_fence_after();
    if(q==0)for(int col=0;col<256;col+=64){
     uint32_t v[64];ld64(tm+col,v);tmem_ld_wait();
     if(lane<p.m)for(int j=0;j<64;j+=4)
      *reinterpret_cast<uint4*>(&s.partial[rank][lane][col+j])=make_uint4(v[j],v[j+1],v[j+2],v[j+3]);
    }
    tc_fence_before();__syncwarp();if(lane==0)mbar_arrive(&s.tile_free);
    wg_sync();
    // The next MMA can run during this tile's DSMEM transfer/reduction. Its
    // accumulator is not drained until the current partial buffer is ACKed.
    if(tile+1<10)stage_sf(11+tile);
    const int owner=tile%C;
    if(q==0)fence_proxy_async_smem();
    if(q==0&&lane==0&&rank!=owner)copy_peer(
     mapa(smem_u32(s.partial[rank]),owner),smem_u32(s.partial[rank]),partial_bytes,
     mapa(smem_u32(&s.reduce_ready[tile]),owner));
    if(rank==owner){
     mbar_wait(&s.reduce_ready[tile],0);
     for(int task=et;task<p.m*256;task+=128){
      const int row=task/256,col=task%256;
      if(s.slots[row]>=0){
       float v=s.partial[0][row][col];
       for(int peer=1;peer<C;++peer)v=__fadd_rn(v,s.partial[peer][row][col]);
       const int logical=(col&~31)+(col%8)*4+(col&31)/8;
       p.out[((size_t)row*K+s.slots[row])*H+tile*256+logical]=__float2bfloat16_rn(v*a2);
      }
     }
     wg_sync();
     if(q==0&&lane==0)for(int peer=0;peer<C;++peer)arrive_peer(mapa(smem_u32(&s.reduce_ack[tile]),peer));
    }
    mbar_wait(&s.reduce_ack[tile],0);wg_sync();
   }
  }
  // Warps2/3 have no cross-warp spin work. They join only at the expert boundary.
  tc_fence_before();__syncthreads();cluster_sync();
 }
 tc_fence_before();if(warp==1)tmem_dealloc(tm,512);__syncthreads();
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
}
} // namespace q38_mk_pipe
