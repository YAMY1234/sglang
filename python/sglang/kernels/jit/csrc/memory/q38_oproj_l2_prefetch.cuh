#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

// MICRO's single-issuer, 64 KiB bulk hints; no activation or weight writes.
__global__ void q38_l2pf_oproj_8cta(const char* weight) {
  constexpr uint32_t bytes = 31457280;
  constexpr uint32_t chunk = 65536;
  if (threadIdx.x == 0) {
    for (uint32_t off = blockIdx.x * chunk; off < bytes; off += gridDim.x * chunk) {
      const uint32_t size = bytes - off < chunk ? bytes - off : chunk;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                   : : "l"(weight + off), "r"(size) : "memory");
    }
  }
}

struct Q38OProjL2Prefetch {
  static int64_t begin(tvm::ffi::TensorView weight) {
    const auto stream = host::LaunchKernel::resolve_device(weight.device());
    cudaStreamCaptureStatus status;
    cudaGraph_t graph;
    const cudaGraphNode_t* frontier;
    const cudaGraphEdgeData* edge_data;
    size_t count;
    host::RuntimeDeviceCheck(cudaStreamGetCaptureInfo(
        stream, &status, nullptr, &graph, &frontier, &edge_data, &count));
    if (status == cudaStreamCaptureStatusNone) {
      // Load the device function during the ordinary capture warmup.
      cudaFuncAttributes attributes;
      host::RuntimeDeviceCheck(cudaFuncGetAttributes(&attributes, q38_l2pf_oproj_8cta));
      return 0;
    }
    CHECK_HOST(status == cudaStreamCaptureStatusActive);
    const char* ptr = static_cast<const char*>(weight.data_ptr());
    void* args[] = {&ptr};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(q38_l2pf_oproj_8cta);
    params.gridDim = dim3(8);
    params.blockDim = dim3(32);
    params.kernelParams = args;
    cudaGraphNode_t node;
    // Only the new side node receives full-completion dependencies. The main
    // stream's frontier and its existing PDL edge data are never replaced.
    host::RuntimeDeviceCheck(cudaGraphAddKernelNode(&node, graph, frontier, count, &params));
    return reinterpret_cast<int64_t>(node);
  }

  static void join(tvm::ffi::TensorView weight, int64_t handle) {
    const auto stream = host::LaunchKernel::resolve_device(weight.device());
    cudaStreamCaptureStatus status;
    cudaGraph_t graph;
    const cudaGraphNode_t* frontier;
    const cudaGraphEdgeData* edge_data;
    size_t count;
    host::RuntimeDeviceCheck(cudaStreamGetCaptureInfo(
        stream, &status, nullptr, &graph, &frontier, &edge_data, &count));
    CHECK_HOST(status == cudaStreamCaptureStatusActive && count > 0);
    const auto side = reinterpret_cast<cudaGraphNode_t>(handle);
    for (size_t i = 0; i < count; ++i) {
      // Attach to existing layer-end nodes, without a new event, kernel,
      // stream barrier, or a dependency on the output GEMV's activation read.
      host::RuntimeDeviceCheck(cudaGraphAddDependencies(graph, &side, &frontier[i], nullptr, 1));
    }
  }
};

}  // namespace sglang
