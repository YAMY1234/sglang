// Diagnostic bridge: attach an already captured arithmetic graph to a device IF.
// No numerical operations are implemented here. The caller retains body storage.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void set_condition(cudaGraphConditionalHandle handle, const int* flag) {
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, *flag != 0);
}

extern "C" int attach_if(uintptr_t stream_ptr, uintptr_t body_ptr, uintptr_t flag_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto body = reinterpret_cast<cudaGraph_t>(body_ptr);
  auto flag = reinterpret_cast<const int*>(flag_ptr);
  cudaStreamCaptureStatus status;
  cudaGraph_t graph;
  const cudaGraphNode_t* dependencies;
  const cudaGraphEdgeData* edges;
  size_t count;
  unsigned long long id;
#define TRY(call) do { auto e = (call); if (e != cudaSuccess) return int(e); } while (0)
  TRY(cudaStreamGetCaptureInfo(stream, &status, &id, &graph, &dependencies, &edges, &count));
  if (status != cudaStreamCaptureStatusActive) return int(cudaErrorStreamCaptureUnsupported);
  cudaGraphConditionalHandle handle;
  TRY(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
  set_condition<<<1, 1, 0, stream>>>(handle, flag);
  TRY(cudaGetLastError());
  TRY(cudaStreamGetCaptureInfo(stream, &status, &id, &graph, &dependencies, &edges, &count));
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;
  cudaGraphNode_t conditional, child;
  TRY(cudaGraphAddNode(&conditional, graph, dependencies, edges, count, &params));
  TRY(cudaGraphAddChildGraphNode(&child, params.conditional.phGraph_out[0], nullptr, 0, body));
  TRY(cudaStreamUpdateCaptureDependencies(stream, &conditional, nullptr, 1, cudaStreamSetCaptureDependencies));
  return int(cudaSuccess);
}
