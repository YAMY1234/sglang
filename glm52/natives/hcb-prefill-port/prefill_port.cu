// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SG baseline radix vs TRT selector/launch port. All outputs are original SG page slots.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include "legacy_radix_topk.cuh"
#include "trt_selector.cuh"

namespace {
constexpr int K=2048;
template<int Threads, bool Port, bool Transform=true>
__global__ __launch_bounds__(Threads) void topk_page_kernel(
 const float* score, const int* lengths, const int* starts, int* out,
 const int* pages, const int* cuq, int64_t score_stride, int64_t page_stride, int bs) {
 const int row=blockIdx.x, t=threadIdx.x, n=lengths[row];
 __shared__ const int* src;
 if constexpr(Transform) for(int i=t;i<bs;i+=Threads) if(row>=cuq[i] && row<cuq[i+1]) src=pages+i*page_stride;
 __syncthreads();
 int* dst=out+int64_t(row)*K;
 if(n<=K){for(int i=t;i<K;i+=Threads){if constexpr(Transform)dst[i]=i<n?src[i]:-1;else dst[i]=i<n?i:-1;}return;}
 const int start=starts?starts[row]:0;
 if constexpr(Port){
  // TRT indices are relative to the supplied row window, matching SG's radix API.
  hcb_trt::topKPerRowJob<Threads,2048,true>(nullptr,score+int64_t(row)*score_stride+start,
                                         0,n,dst,nullptr,1,K);
  __syncthreads();
  if constexpr(Transform) for(int i=t;i<K;i+=Threads){const int pos=dst[i];dst[i]=pos<0?-1:src[pos];}
 }else{
  __shared__ int indices[K];
  sglang::device::legacy_radix_topk::select<Threads,K,4096>(
    score+int64_t(row)*score_stride,indices,start,n,K);
  __syncthreads();
  for(int i=t;i<K;i+=Threads){const int pos=indices[i];dst[i]=pos<0?-1:src[pos];}
 }
}

void topk(const at::Tensor& score,const at::Tensor& lengths,const at::Tensor& starts,
          const at::Tensor& pages,const at::Tensor& cuq,at::Tensor out,int64_t variant){
 TORCH_CHECK(score.is_cuda() && score.scalar_type()==at::kFloat && score.dim()==2 && score.stride(1)==1);
 TORCH_CHECK(lengths.is_cuda() && lengths.scalar_type()==at::kInt && lengths.is_contiguous() && lengths.numel()==score.size(0));
 TORCH_CHECK(starts.is_cuda() && starts.scalar_type()==at::kInt && starts.is_contiguous() && starts.numel()==score.size(0));
 TORCH_CHECK(pages.is_cuda() && pages.scalar_type()==at::kInt && pages.dim()==2 && pages.stride(1)==1);
 TORCH_CHECK(cuq.is_cuda() && cuq.scalar_type()==at::kInt && cuq.is_contiguous() && cuq.numel()==pages.size(0)+1);
 TORCH_CHECK(out.is_cuda() && out.scalar_type()==at::kInt && out.is_contiguous() && out.size(0)==score.size(0) && out.size(1)==K);
 c10::cuda::CUDAGuard guard(score.device());
 auto stream=c10::cuda::getCurrentCUDAStream().stream();
 if(variant==0){
  topk_page_kernel<1024,false><<<score.size(0),1024,32768,stream>>>(score.data_ptr<float>(),lengths.data_ptr<int>(),starts.data_ptr<int>(),out.data_ptr<int>(),pages.data_ptr<int>(),cuq.data_ptr<int>(),score.stride(0),pages.stride(0),pages.size(0));
 }else if(variant==1){
  topk_page_kernel<512,true><<<score.size(0),512,K*sizeof(int),stream>>>(score.data_ptr<float>(),lengths.data_ptr<int>(),starts.data_ptr<int>(),out.data_ptr<int>(),pages.data_ptr<int>(),cuq.data_ptr<int>(),score.stride(0),pages.stride(0),pages.size(0));
 }else if(variant==2){
  topk_page_kernel<512,true,false><<<score.size(0),512,K*sizeof(int),stream>>>(score.data_ptr<float>(),lengths.data_ptr<int>(),starts.data_ptr<int>(),out.data_ptr<int>(),pages.data_ptr<int>(),cuq.data_ptr<int>(),score.stride(0),pages.stride(0),pages.size(0));
 }else{
  TORCH_CHECK(false,"variant must be 0 (SG), 1 (TRT512 mapped), or 2 (TRT512 relative indices)");
 }
 C10_CUDA_CHECK(cudaGetLastError());
}
} // namespace
TORCH_LIBRARY(hcb_prefill,m){m.def("topk(Tensor score, Tensor lengths, Tensor starts, Tensor pages, Tensor cuq, Tensor(a!) out, int variant) -> ()");}
TORCH_LIBRARY_IMPL(hcb_prefill,CUDA,m){m.impl("topk",&topk);}

namespace {
// TRT unified gather: each lane moves one uint32 K word, lane 0 also moves the scale.
// SG packs 64 keys followed by 64 scales in a page; no quantization or format change.
template<int Warps>
__global__ void index_k_gather(const uint8_t* cache,const int* pages,const int* lens,
 uint8_t* k,uint8_t* scale,int rows,int max_len,int64_t page_stride,int64_t table_stride,int page_size){
 const int token=blockIdx.x*Warps+threadIdx.x/32, lane=threadIdx.x%32, req=blockIdx.y;
 if(token>=lens[req])return;
 int before=0;for(int i=0;i<req;++i)before+=lens[i];
 const int page=pages[int64_t(req)*table_stride+token/page_size];
 const int off=token%page_size;
 const uint8_t* src=cache+int64_t(page)*page_stride;
 *reinterpret_cast<uint32_t*>(k+int64_t(before+token)*128+lane*4)=
  *reinterpret_cast<const uint32_t*>(src+off*128+lane*4);
 if(lane==0)*reinterpret_cast<uint32_t*>(scale+int64_t(before+token)*4)=
  *reinterpret_cast<const uint32_t*>(src+page_size*128+off*4);
}
void gather(const at::Tensor& cache,const at::Tensor& pages,const at::Tensor& lens,
 at::Tensor k,at::Tensor scale,int64_t max_len,int64_t page_size,int64_t warps){
 TORCH_CHECK(cache.is_cuda() && cache.scalar_type()==at::kByte && cache.dim()==2 && cache.stride(1)==1 && cache.stride(0)%4==0);
 TORCH_CHECK(page_size>0 && cache.size(1)==page_size*132);
 TORCH_CHECK(pages.is_cuda() && pages.scalar_type()==at::kInt && pages.dim()==2 && pages.stride(1)==1);
 TORCH_CHECK(lens.is_cuda() && lens.scalar_type()==at::kInt && lens.is_contiguous() && lens.numel()==pages.size(0));
 TORCH_CHECK(k.is_cuda() && k.scalar_type()==at::kByte && k.is_contiguous() && k.dim()==2 && k.size(1)==128);
 TORCH_CHECK(scale.is_cuda() && scale.scalar_type()==at::kByte && scale.is_contiguous() && scale.size(0)==k.size(0) && scale.size(1)==4);
 c10::cuda::CUDAGuard guard(cache.device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
 if(max_len==0 || lens.numel()==0)return;
 if(warps==1)index_k_gather<1><<<dim3(max_len,lens.numel()),32,0,stream>>>(cache.data_ptr<uint8_t>(),pages.data_ptr<int>(),lens.data_ptr<int>(),k.data_ptr<uint8_t>(),scale.data_ptr<uint8_t>(),lens.numel(),max_len,cache.stride(0),pages.stride(0),page_size);
 else if(warps==4)index_k_gather<4><<<dim3((max_len+3)/4,lens.numel()),128,0,stream>>>(cache.data_ptr<uint8_t>(),pages.data_ptr<int>(),lens.data_ptr<int>(),k.data_ptr<uint8_t>(),scale.data_ptr<uint8_t>(),lens.numel(),max_len,cache.stride(0),pages.stride(0),page_size);
 else TORCH_CHECK(false,"warps must be 1 or 4");
 C10_CUDA_CHECK(cudaGetLastError());
}
}
TORCH_LIBRARY_FRAGMENT(hcb_prefill,m){m.def("gather(Tensor cache, Tensor pages, Tensor lens, Tensor(a!) k, Tensor(b!) scale, int max_len, int page_size, int warps) -> ()");}
TORCH_LIBRARY_IMPL(hcb_prefill,CUDA,m){m.impl("gather",&gather);}

namespace {
// TRT's selected-only remap phase adapted to SG page-size-1 tables. The selector
// itself is the port above, not the previously tested SG rowmap selector.
__global__ void remap_selected(const int* selected,const int* pages,const int64_t* req,int* out,
 int64_t count,int64_t stride){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<count){int idx=selected[i];out[i]=idx<0?-1:pages[req[i/K]*stride+idx];}
}
void remap(const at::Tensor& selected,const at::Tensor& pages,const at::Tensor& request,at::Tensor out){
 TORCH_CHECK(selected.is_cuda() && selected.scalar_type()==at::kInt && selected.is_contiguous() && selected.dim()==2 && selected.size(1)==K);
 TORCH_CHECK(pages.is_cuda() && pages.scalar_type()==at::kInt && pages.dim()==2 && pages.stride(1)==1);
 TORCH_CHECK(request.is_cuda() && request.scalar_type()==at::kLong && request.is_contiguous() && request.numel()==selected.size(0));
 TORCH_CHECK(out.is_cuda() && out.scalar_type()==at::kInt && out.is_contiguous() && out.sizes()==selected.sizes());
 c10::cuda::CUDAGuard guard(selected.device());auto stream=c10::cuda::getCurrentCUDAStream().stream();
 if(!selected.numel())return;
 remap_selected<<<(selected.numel()+255)/256,256,0,stream>>>(selected.data_ptr<int>(),pages.data_ptr<int>(),request.data_ptr<int64_t>(),out.data_ptr<int>(),selected.numel(),pages.stride(0));
 C10_CUDA_CHECK(cudaGetLastError());
}
}
TORCH_LIBRARY_FRAGMENT(hcb_prefill,m){m.def("remap(Tensor selected, Tensor pages, Tensor request, Tensor(a!) out) -> ()");}
TORCH_LIBRARY_IMPL(hcb_prefill,CUDA,m){m.impl("remap",&remap);}
