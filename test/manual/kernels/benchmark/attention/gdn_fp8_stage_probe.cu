// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gdn_fp8_stage.cuh"

static void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    std::exit(2);
  }
}

static void check_driver(CUresult error, const char* operation) {
  if (error != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(error, &message);
    std::fprintf(stderr, "%s: %s\n", operation, message ? message : "error");
    std::exit(2);
  }
}

static CUtensorMap make_map(void* base, int k, int rows, int tile_k,
                           int tile_rows) {
  CUtensorMap map;
  const cuuint64_t dimensions[2] = {static_cast<cuuint64_t>(k),
                                   static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(k)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(tile_k),
                            static_cast<cuuint32_t>(tile_rows)};
  const cuuint32_t element_strides[2] = {1, 1};
  check_driver(cuTensorMapEncodeTiled(
                   &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dimensions,
                   strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_NONE,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "encode tensor map");
  return map;
}

template <int K, int TileK>
__global__ __launch_bounds__(256, 2) void projection_wrapper(
    const __grid_constant__ CUtensorMap weight_map,
    const __grid_constant__ CUtensorMap activation_map,
    __nv_bfloat16* output, int n, float alpha, bool prefetched,
    gdn_fp8::WaitControl control, bool omit_prefix) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& ring = *reinterpret_cast<gdn_fp8::ProjectionRing<64, TileK, 2>*>(storage);
  gdn_fp8::initialize(ring);
  gdn_fp8::begin_roles<32, 128>();
  if (prefetched && !omit_prefix && threadIdx.x < 128) {
    gdn_fp8::prefetch_weight_prefix<K>(&weight_map, blockIdx.x * 64, ring);
  }
  gdn_fp8::project<7, K>(&weight_map, &activation_map, blockIdx.x * 64, alpha,
                        output, n, ring, control, prefetched || omit_prefix);
}

template <int K, int N, int TileK>
static int run(int extra_shared, bool prefetched, bool validation_only,
               bool omit_prefix) {
  constexpr int m = 7;
  constexpr float alpha = 0.03125f;
  std::vector<__nv_fp8_e4m3> weights(N * K);
  std::vector<__nv_fp8_e4m3> activation(m * K);
  std::vector<__nv_bfloat16> output(m * N);
  uint32_t state = 0x9e3779b9;
  auto sample = [&]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int>((state >> 24) & 127) - 63) /
           128.0f;
  };
  for (auto& value : weights) value = __nv_fp8_e4m3(sample());
  for (auto& value : activation) value = __nv_fp8_e4m3(sample());

  void* device_weights = nullptr;
  void* device_activation = nullptr;
  __nv_bfloat16* device_output = nullptr;
  unsigned int* device_error = nullptr;
  check_cuda(cudaMalloc(&device_weights, weights.size()), "allocate weights");
  check_cuda(cudaMalloc(&device_activation, activation.size()),
             "allocate activation");
  check_cuda(cudaMalloc(&device_output, output.size() * sizeof(output[0])),
             "allocate output");
  check_cuda(cudaMalloc(&device_error, sizeof(unsigned int)), "allocate error");
  check_cuda(cudaMemset(device_error, 0, sizeof(unsigned int)), "clear error");
  check_cuda(cudaMemcpy(device_weights, weights.data(), weights.size(),
                        cudaMemcpyHostToDevice),
             "copy weights");
  check_cuda(cudaMemcpy(device_activation, activation.data(), activation.size(),
                        cudaMemcpyHostToDevice),
             "copy activation");
  const CUtensorMap weight_map = make_map(device_weights, K, N, TileK, 64);
  const CUtensorMap activation_map = make_map(device_activation, K, m, TileK, 8);
  const int shared_bytes = sizeof(gdn_fp8::ProjectionRing<64, TileK, 2>) +
                           extra_shared;
  check_cuda(cudaFuncSetAttribute(projection_wrapper<K, TileK>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_bytes),
             "set shared allocation");
  cudaFuncAttributes attributes;
  check_cuda(cudaFuncGetAttributes(&attributes, projection_wrapper<K, TileK>),
             "query function");
  // The initial CTA pool must cover both role budgets.  An upper compiler
  // bound alone does not establish how many registers the launch owns.
  constexpr int required_pool = 128 * 32 + 128 * 128;
  if (attributes.numRegs < 32 || attributes.numRegs > 128 ||
      attributes.numRegs * 256 < required_pool) {
    std::fprintf(stderr,
                 "register pool refused: initial=%d total=%d required=%d\n",
                 attributes.numRegs, attributes.numRegs * 256, required_pool);
    return 3;
  }
  int device = 0;
  int clock_khz = 0;
  check_cuda(cudaGetDevice(&device), "query device");
  check_cuda(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device),
             "query clock");
  const gdn_fp8::WaitControl control = {
      static_cast<uint64_t>(clock_khz) * 1000u, device_error};
  int active_blocks = 0;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &active_blocks, projection_wrapper<K, TileK>, 256, shared_bytes),
             "query occupancy");

  projection_wrapper<K, TileK><<<N / 64, 256, shared_bytes>>>(
      weight_map, activation_map, device_output, N, alpha, prefetched, control,
      omit_prefix);
  check_cuda(cudaGetLastError(), "projection launch");
  check_cuda(cudaDeviceSynchronize(), "projection completion");
  unsigned int error = 0;
  check_cuda(cudaMemcpy(&error, device_error, sizeof(error),
                        cudaMemcpyDeviceToHost),
             "copy error");
  if (error) {
    std::fprintf(stderr, "projection poison: %u\n", error);
    return 4;
  }
  check_cuda(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(output[0]),
                        cudaMemcpyDeviceToHost),
             "copy output");

  double max_absolute = 0.0;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  int nonfinite = 0;
  int differing = 0;
  for (int token = 0; token < m; ++token) {
    for (int channel = 0; channel < N; ++channel) {
      double accumulator = 0.0;
      for (int inner = 0; inner < K; ++inner) {
        accumulator += static_cast<float>(activation[token * K + inner]) *
                       static_cast<float>(weights[channel * K + inner]);
      }
      const float reference = static_cast<float>(
          __float2bfloat16_rn(static_cast<float>(accumulator) * alpha));
      const float actual = static_cast<float>(output[token * N + channel]);
      const double error = static_cast<double>(actual) - reference;
      nonfinite += !std::isfinite(actual);
      differing += actual != reference;
      max_absolute = std::max(max_absolute, std::abs(error));
      squared_error += error * error;
      squared_reference += static_cast<double>(reference) * reference;
    }
  }

  if (validation_only) {
    std::printf(
        "{\"m\":7,\"n\":%d,\"k\":%d,\"tile_k\":%d,"
        "\"validation_only\":true,\"prefetched\":%s,"
        "\"shared_bytes\":%d,\"num_regs\":%d,\"local_bytes\":%zu,"
        "\"max_absolute_error\":%.9g,\"relative_l2_error\":%.9g,"
        "\"different_bf16\":%d,\"elements\":%d,\"nonfinite\":%d}\n",
        N, K, TileK, prefetched ? "true" : "false", shared_bytes,
        attributes.numRegs, attributes.localSizeBytes, max_absolute,
        std::sqrt(squared_error / squared_reference), differing, m * N,
        nonfinite);
    check_cuda(cudaFree(device_error), "free error");
    check_cuda(cudaFree(device_weights), "free weights");
    check_cuda(cudaFree(device_activation), "free activation");
    check_cuda(cudaFree(device_output), "free output");
    return nonfinite || max_absolute > 0.01 ? 1 : 0;
  }

  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  check_cuda(cudaStreamCreate(&stream), "create stream");
  check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
             "begin graph");
  for (int repeat = 0; repeat < 20; ++repeat) {
    projection_wrapper<K, TileK><<<N / 64, 256, shared_bytes, stream>>>(
        weight_map, activation_map, device_output, N, alpha, prefetched, control,
        omit_prefix);
  }
  check_cuda(cudaStreamEndCapture(stream, &graph), "end graph");
  check_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
             "instantiate graph");
  for (int repeat = 0; repeat < 5; ++repeat) {
    check_cuda(cudaGraphLaunch(executable, stream), "warm graph");
  }
  check_cuda(cudaStreamSynchronize(stream), "warm completion");
  cudaEvent_t begin, end;
  check_cuda(cudaEventCreate(&begin), "create event");
  check_cuda(cudaEventCreate(&end), "create event");
  std::vector<float> samples;
  for (int sample = 0; sample < 9; ++sample) {
    check_cuda(cudaEventRecord(begin, stream), "start event");
    for (int repeat = 0; repeat < 20; ++repeat) {
      check_cuda(cudaGraphLaunch(executable, stream), "timed graph");
    }
    check_cuda(cudaEventRecord(end, stream), "end event");
    check_cuda(cudaEventSynchronize(end), "timing completion");
    float milliseconds;
    check_cuda(cudaEventElapsedTime(&milliseconds, begin, end), "elapsed time");
    samples.push_back(milliseconds * 1000.0f / 400.0f);
  }
  std::sort(samples.begin(), samples.end());
  check_cuda(cudaMemcpy(&error, device_error, sizeof(error),
                        cudaMemcpyDeviceToHost),
             "copy timed error");
  if (error) {
    std::fprintf(stderr, "timed projection poison: %u\n", error);
    return 4;
  }
  std::printf(
      "{\"m\":7,\"n\":%d,\"k\":%d,\"tile_n\":64,\"tile_k\":%d,"
      "\"prefetched\":%s,\"shared_bytes\":%d,\"num_regs\":%d,"
      "\"local_bytes\":%zu,\"active_blocks\":%d,"
      "\"numeric_class\":\"fp32-reduction-order\","
      "\"max_absolute_error\":%.9g,\"relative_l2_error\":%.9g,"
      "\"different_bf16\":%d,\"elements\":%d,\"nonfinite\":%d,"
      "\"graph_us_median\":%.6f,\"graph_us_min\":%.6f,"
      "\"graph_us_max\":%.6f,\"samples\":9}\n",
      N, K, TileK, prefetched ? "true" : "false", shared_bytes,
      attributes.numRegs, attributes.localSizeBytes, active_blocks, max_absolute,
      std::sqrt(squared_error / squared_reference), differing, m * N, nonfinite,
      samples[4], samples.front(), samples.back());
  check_cuda(cudaEventDestroy(begin), "destroy event");
  check_cuda(cudaEventDestroy(end), "destroy event");
  check_cuda(cudaGraphExecDestroy(executable), "destroy executable");
  check_cuda(cudaGraphDestroy(graph), "destroy graph");
  check_cuda(cudaStreamDestroy(stream), "destroy stream");
  check_cuda(cudaFree(device_error), "free error");
  check_cuda(cudaFree(device_weights), "free weights");
  check_cuda(cudaFree(device_activation), "free activation");
  check_cuda(cudaFree(device_output), "free output");
  return nonfinite || max_absolute > 0.01 ? 1 : 0;
}

int main(int argc, char** argv) {
  const int extra_shared = argc > 1 ? std::atoi(argv[1]) : 0;
  const bool prefetched = argc > 2 && std::atoi(argv[2]) != 0;
  const int tile_k = argc > 3 ? std::atoi(argv[3]) : 128;
  const bool validation_only = argc > 4 && std::atoi(argv[4]) != 0;
  const bool omit_prefix = argc > 5 && std::atoi(argv[5]) != 0;
  check_cuda(cudaFree(nullptr), "initialize runtime");
  if (tile_k == 128) {
    return run<4096, 2560, 128>(extra_shared, prefetched, validation_only,
                               omit_prefix) |
           run<1024, 4096, 128>(extra_shared, prefetched, validation_only,
                               omit_prefix);
  }
  if (tile_k == 256) {
    return run<4096, 2560, 256>(extra_shared, prefetched, validation_only,
                               omit_prefix) |
           run<1024, 4096, 256>(extra_shared, prefetched, validation_only,
                               omit_prefix);
  }
  std::fprintf(stderr, "tile_k must be 128 or 256\n");
  return 2;
}
