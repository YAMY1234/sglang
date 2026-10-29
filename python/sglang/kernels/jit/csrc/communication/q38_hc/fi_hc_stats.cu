// Experimental installed-FI MNNVL epilogue adapter. Not model-integrated.
// Uses the installed FI Lamport workspace and helpers as a single protocol;
// no CA/Q35 mailbox, flag, stage, or counter is introduced here.
#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"
#include "tvm_ffi_utils.h"
#include "hc_math.cuh"

namespace hc2_fi_stats_v1 {
using namespace flashinfer::trtllm_mnnvl_allreduce;
using T = __nv_bfloat16;

// Same arithmetic is exercised by the single-GPU local arithmetic fixture.
// Its rank inputs are resident copies; it makes no transport timing claim.
__device__ __forceinline__ void hc_epilogue(
    PackedVec<float4,T> packedAccum, int token, int tokenDim, int packedIdx,
    const T* residual, const float* gates, T* combined, float* stats) {
  using PackedType = float4;
  constexpr int kElts = 8;
  // Polling may diverge lanes. Reconverge using the launch's fixed mask;
  // __activemask after polling is not a valid data-exchange contract.
  const unsigned warpBase = (threadIdx.x / 32) * 32;
  const unsigned live = min(32u, static_cast<unsigned>(blockDim.x) - warpBase);
  const unsigned mask = live == 32 ? 0xffffffffu : ((1u << live) - 1u);
  __syncwarp(mask);
  const unsigned lane = threadIdx.x % 32;
  // Compute four gates together once/warp; every warp has at least16 lanes.
  const float ownGate = lane < 4 ? sglang::device::hc4::inject_gate(gates, token, lane) : 0.f;
#pragma unroll
  for (int branch = 0; branch < 4; ++branch) {
    PackedVec<PackedType, T> res, out;
    const int offset = (token * 4 + branch) * tokenDim + packedIdx * kElts;
    res.packed = loadPacked<PackedType>(residual + offset);
    const float gate = __shfl_sync(mask, ownGate, branch);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kElts; ++i) {
      out.elements[i] = sglang::device::hc4::combine(
          packedAccum.elements[i], res.elements[i], gate);
      const float x = __bfloat162float(out.elements[i]);
      sum = sglang::device::hc4::add_rn(sum, sglang::device::hc4::mul_rn(x, x));
    }
    *reinterpret_cast<PackedType*>(combined + offset) = out.packed;
#pragma unroll
    for (int delta = 4; delta > 0; delta /= 2)
      sum = sglang::device::hc4::add_rn(sum, __shfl_down_sync(mask, sum, delta, 8));
    if (packedIdx % 8 == 0)
      stats[(token * 4 + branch) * 40 + packedIdx / 8] = sum;
  }
}

__global__ __launch_bounds__(1024) void oneshot_hc_stats(
    AllReduceKernelParams<T> params, const T* residual, const float* gates,
    T* combined, float* stats) {
  using PackedType = float4;
  constexpr int kElts = 8, kWorld = 4;
  auto cluster = cooperative_groups::this_cluster();
  const int packedIdx = cluster.thread_rank();
  const int token = blockIdx.x;
  const int tokenDim = params.tokenDim;
  const int threadOffset = token * tokenDim + packedIdx * kElts;
  cudaGridDependencySynchronize();
  LamportFlags<PackedType, true> flag(params.bufferFlags, 1);
  T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, 0));
  T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], 0));
  if (packedIdx * kElts >= tokenDim) {
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);
    return;
  }
  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
  sanitizeLamportPayload<PackedType, T>(val);
  reinterpret_cast<PackedType*>(
      &stagePtrMcast[token * tokenDim * kWorld + params.rank * tokenDim])[packedIdx] = val.packed;
  flag.ctaArrive();
  flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);
  PackedVec<PackedType, T> packedAccum;
#define HC2_FI_RANK(R)                                                        \
  case R: {                                                                  \
    PackedVec<PackedType, T> remoteValues[kWorld];                             \
    utils::waitOneshotRemoteRanks<kWorld, R, T, PackedType, kElts>(              \
        remoteValues, stagePtrLocal, token, tokenDim, packedIdx);              \
    packedAccum = utils::reduceOneshotDeterministic<kWorld, R, T, PackedType,    \
                                                   kElts>(remoteValues, val); \
    break;                                                                   \
  }
  switch (params.rank) {
    HC2_FI_RANK(0);
    HC2_FI_RANK(1);
    HC2_FI_RANK(2);
    HC2_FI_RANK(3);
  }
#undef HC2_FI_RANK
  // Same point as installed FI: peer contributions have been consumed. The
  // dependent may prefetch weights; it must wait before reading HC outputs.
  cudaTriggerProgrammaticLaunchCompletion();
  reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = packedAccum.packed;
  hc_epilogue(packedAccum, token, tokenDim, packedIdx, residual, gates, combined, stats);
  // This is the original native finalization, after ALL HC stores.
  flag.waitAndUpdate(
      {static_cast<uint32_t>(params.numTokens * tokenDim * kWorld * sizeof(T)), 0, 0, 0});
}

// Local four-input reduction uses exactly FI's arithmetic helper. This is a
// CPU-prepared, single-GPU K fixture: no Lamport traffic or TP4 transport.
template<bool FuseHC>
__global__ void local_four_input_reduce(const T* ranks, int rows, T* reduced,
    const T* residual, const float* gates, T* combined, float* stats) {
  constexpr int H=2560;
  const int token=blockIdx.x, packedIdx=blockIdx.y*blockDim.x+threadIdx.x;
  cudaGridDependencySynchronize();
  PackedVec<float4,T> values[4];
  #pragma unroll
  for (int rank=0;rank<4;++rank) {
    values[rank].packed=loadPacked<float4>(ranks+(rank*rows+token)*H+packedIdx*8);
    sanitizeLamportPayload<float4,T>(values[rank]);
  }
  auto accum=utils::reduceOneshotDeterministic<4,0,T,float4,8>(values,values[0]);
  cudaTriggerProgrammaticLaunchCompletion();
  *reinterpret_cast<float4*>(reduced+token*H+packedIdx*8)=accum.packed;
  if constexpr(FuseHC)
    hc_epilogue(accum,token,H,packedIdx,residual,gates,combined,stats);
}

__global__ void norm_from_stats(const T* combined,const float* stats,
    const T* gamma,T* normalized,float eps) {
  const int row=blockIdx.x, branch=blockIdx.y;
  __shared__ float inverse;
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  if(threadIdx.x==0) inverse=sglang::device::hc4::inv_rms(stats,row,branch,eps);
  __syncthreads();
  for(int h=threadIdx.x;h<2560;h+=blockDim.x) {
    const int offset=(row*4+branch)*2560+h;
    normalized[offset]=sglang::device::hc4::normalize(
        combined[offset],gamma[branch*2560+h],inverse);
  }
}
}  // namespace hc2_fi_stats_v1

void hc2_fi_stats_v1_oneshot(TensorView input, int64_t multicast_ptr, int64_t peer_ptrs,
                   int64_t local_ptr, TensorView flags, int64_t rank,
                   TensorView residual, TensorView gates, TensorView reduced,
                   TensorView combined, TensorView stats) {
  using namespace hc2_fi_stats_v1;
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  TVM_FFI_ICHECK(input.ndim() == 2 && input.size(1) == 2560);
  const int m = input.size(0);
  TVM_FFI_ICHECK(m == 1 || m == 4 || m == 8 || m == 12);
  TVM_FFI_ICHECK(rank >= 0 && rank < 4);
  auto [blockSize, clusterSize, loads] = utils::adjustGridConfig(m, 2560, 8, true);
  // Every64H statistic belongs entirely to one CTA, including partial warps.
  TVM_FFI_ICHECK(loads == 1 && blockSize % 8 == 0 && blockSize * clusterSize == 320);
  AllReduceKernelParams<T> params{};
  params.outputPtr = static_cast<T*>(reduced.data_ptr());
  params.shardPtr = static_cast<const T*>(input.data_ptr());
  params.inputPtrs = reinterpret_cast<T**>(peer_ptrs);
  params.mcastPtr = reinterpret_cast<T*>(multicast_ptr);
  params.localBufferPtr = reinterpret_cast<T*>(local_ptr);
  params.nRanks = 4;
  params.rank = rank;
  params.numTokens = m;
  params.tokenDim = 2560;
  params.bufferFlags = static_cast<uint32_t*>(flags.data_ptr());
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim = {1, static_cast<unsigned>(clusterSize), 1};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(m, clusterSize, 1);
  config.blockDim = dim3(blockSize);
  config.stream = get_stream(input.device());
  config.attrs = attrs;
  config.numAttrs = 2;
  auto status = cudaLaunchKernelEx(&config, &oneshot_hc_stats, params,
      static_cast<const T*>(residual.data_ptr()), static_cast<const float*>(gates.data_ptr()),
      static_cast<T*>(combined.data_ptr()), static_cast<float*>(stats.data_ptr()));
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(hc2_fi_stats_v1_oneshot, hc2_fi_stats_v1_oneshot);

void hc_local_reduce(TensorView inputs, TensorView reduced,TensorView residual,
    TensorView gates,TensorView combined,TensorView stats,bool fused) {
  using namespace hc2_fi_stats_v1;
  ffi::CUDADeviceGuard guard(inputs.device().device_id);
  TVM_FFI_ICHECK(inputs.ndim()==3 && inputs.size(0)==4 && inputs.size(2)==2560);
  const int rows=inputs.size(1);
  TVM_FFI_ICHECK(rows==1 || rows==4);
  cudaLaunchAttribute attr{};
  attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(rows,4,1);config.blockDim=dim3(80);
  config.attrs=&attr;config.numAttrs=1;config.stream=get_stream(inputs.device());
  auto launch=[&]<bool FuseHC>() {
    return cudaLaunchKernelEx(&config,&local_four_input_reduce<FuseHC>,
      static_cast<const T*>(inputs.data_ptr()),rows,static_cast<T*>(reduced.data_ptr()),
      static_cast<const T*>(residual.data_ptr()),static_cast<const float*>(gates.data_ptr()),
      static_cast<T*>(combined.data_ptr()),static_cast<float*>(stats.data_ptr()));
  };
  auto status=fused?launch.template operator()<true>():launch.template operator()<false>();
  TVM_FFI_ICHECK(status==cudaSuccess)<<cudaGetErrorString(status);
}
void hc_norm_stats(TensorView combined,TensorView stats,TensorView gamma,
    TensorView normalized,double eps) {
  using namespace hc2_fi_stats_v1;
  ffi::CUDADeviceGuard guard(combined.device().device_id);
  const int rows=combined.size(0);
  cudaLaunchAttribute attr{};
  attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(rows,4,1);config.blockDim=dim3(256);
  config.attrs=&attr;config.numAttrs=1;config.stream=get_stream(combined.device());
  auto status=cudaLaunchKernelEx(&config,&norm_from_stats,
    static_cast<const T*>(combined.data_ptr()),static_cast<const float*>(stats.data_ptr()),
    static_cast<const T*>(gamma.data_ptr()),static_cast<T*>(normalized.data_ptr()),float(eps));
  TVM_FFI_ICHECK(status==cudaSuccess)<<cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(hc_local_reduce,hc_local_reduce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(hc_norm_stats,hc_norm_stats);
