// SPDX-License-Identifier: Apache-2.0
// Modified by NVIDIA: add standalone and consumer-role CUDA wrappers around
// one ReplaySSM device stage, with optional execution counters and byte checks.

#include "gdn_replayssm_stage.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef GDN_REPLAY_PROBE_MAX_REGS
#define GDN_REPLAY_PROBE_MAX_REGS 128
#endif

#ifndef GDN_REPLAY_PROBE_BV
#define GDN_REPLAY_PROBE_BV 32
#endif

namespace {

constexpr int kBV = GDN_REPLAY_PROBE_BV;
using Scratch = gdn_stage::ReplayScratch<kBV>;
using Params = gdn_stage::ReplayParams;

// A null status pointer removes atomics from the executed timing path. A
// non-null pointer is a two-element uint64 device allocation, zeroed by caller.
struct ProbeStatus {
  unsigned long long entered;
  unsigned long long completed;
};

// phase is a device-resident runtime selector. Zero executes the real stage;
// another value is a correctness negative control. Both entry points retain
// the complete device stage regardless of the selected phase.
__device__ __forceinline__ void run_stage(
    const Params& p, const int32_t* phase, ProbeStatus* status, int local_tid,
    Scratch& scratch) {
  if (phase != nullptr && *phase != 0) {
    return;
  }
  if (status != nullptr && local_tid == 0) {
    atomicAdd(&status->entered, 1ull);
  }
  gdn_stage::replay_verify_stage<kBV, 64, 1>(
      p, blockIdx.y, blockIdx.z, blockIdx.x * kBV, local_tid, scratch);
  if (status != nullptr && local_tid == 0) {
    atomicAdd(&status->completed, 1ull);
  }
}

__global__ void __maxnreg__(GDN_REPLAY_PROBE_MAX_REGS)
replay_stage_dedicated(const __grid_constant__ Params p, const int32_t* phase,
                       ProbeStatus* status) {
  __shared__ Scratch scratch;
  run_stage(p, phase, status, threadIdx.x, scratch);
}

// This measures the consumer-role layout only. It is not a complete fused
// projection/convolution/normalization dispatcher and cannot establish the
// latter's resource or performance result.
__global__ void __maxnreg__(GDN_REPLAY_PROBE_MAX_REGS)
replay_stage_role_layout(const __grid_constant__ Params p, const int32_t* phase,
                         ProbeStatus* status) {
  __shared__ Scratch scratch;
  if (threadIdx.x < 128) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
#endif
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"
                 : : "n"(GDN_REPLAY_PROBE_MAX_REGS) : "memory");
#endif
    const int local_tid = threadIdx.x - 128;
    if (local_tid < 64) {
      run_stage(p, phase, status, local_tid, scratch);
    }
  }
}

__global__ void compare_bytes(const unsigned char* a, const unsigned char* b,
                              size_t bytes, unsigned long long* mismatch_count,
                              unsigned long long* first_mismatch) {
  unsigned long long count = 0;
  unsigned long long first = ~0ull;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    if (a[i] != b[i]) {
      ++count;
      if (i < first) {
        first = i;
      }
    }
  }
  if (count != 0) {
    atomicAdd(mismatch_count, count);
    atomicMin(first_mismatch, first);
  }
}

__global__ void evict_lines(uint4* allocation, size_t vectors) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vectors;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    uint4 value = allocation[i];
    value.x ^= 0x9e3779b9u;
    value.y ^= 0x85ebca6bu;
    value.z ^= 0xc2b2ae35u;
    value.w ^= 0x27d4eb2fu;
    allocation[i] = value;
  }
}

bool valid_params(const Params* p, int batch) {
  if (p == nullptr || batch < 1 || p->h < 1 || p->hv < 1 ||
      p->hv % p->h != 0) {
    return false;
  }
  if (p->q == nullptr || p->k == nullptr || p->v == nullptr ||
      p->a == nullptr || p->b == nullptr || p->a_log == nullptr ||
      p->dt_bias == nullptr || p->out == nullptr || p->checkpoint == nullptr ||
      p->d_cache == nullptr || p->k_cache == nullptr || p->g_cache == nullptr ||
      p->query_start_loc == nullptr || p->state_indices == nullptr ||
      p->replay_indices == nullptr || p->write_pos == nullptr ||
      p->cache_base == nullptr || p->is_flush == nullptr) {
    return false;
  }
  if (p->storage != gdn_stage::ReplayStorage::kNoResidual &&
      (p->rawv_cache == nullptr || p->rawk_cache == nullptr)) {
    return false;
  }
  return p->storage != gdn_stage::ReplayStorage::kExactRaw ||
         p->beta_cache != nullptr;
}

}  // namespace

// C ABI permits a small ctypes harness or an independent C++ caller. Parameter
// storage is the header's ReplayParams; no Torch allocation or stream sync is
// performed here. role_layout=0 uses 64 threads; role_layout=1 uses 256 threads.
extern "C" cudaError_t gdn_replay_probe_launch(
    const gdn_stage::ReplayParams* params, int batch, int role_layout,
    const int32_t* device_phase, void* device_status, cudaStream_t stream) {
  if (!valid_params(params, batch) || (role_layout != 0 && role_layout != 1)) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid((128 + kBV - 1) / kBV, batch, params->hv);
  ProbeStatus* status = static_cast<ProbeStatus*>(device_status);
  if (role_layout == 0) {
    replay_stage_dedicated<<<grid, 64, 0, stream>>>(*params, device_phase, status);
  } else {
    cudaFuncAttributes attributes{};
    const cudaError_t query = cudaFuncGetAttributes(&attributes, replay_stage_role_layout);
    if (query != cudaSuccess) {
      return query;
    }
    // The compiled CTA allocation must cover both role allocations. A register
    // ceiling alone does not establish that the donated pool is sufficient.
    constexpr int minimum_regs = (32 + GDN_REPLAY_PROBE_MAX_REGS) / 2;
    if (attributes.numRegs < minimum_regs ||
        attributes.numRegs > GDN_REPLAY_PROBE_MAX_REGS) {
      return cudaErrorLaunchOutOfResources;
    }
    replay_stage_role_layout<<<grid, 256, 0, stream>>>(*params, device_phase, status);
  }
  return cudaPeekAtLastError();
}

extern "C" size_t gdn_replay_probe_params_bytes() {
  return sizeof(gdn_stage::ReplayParams);
}

extern "C" void gdn_replay_probe_params_init(gdn_stage::ReplayParams* params) {
  *params = gdn_stage::ReplayParams{};
}

// Ordered exactly as the header fields. A ctypes caller checks every offset
// and total sizeof before passing its first device pointer.
extern "C" size_t gdn_replay_probe_param_offsets(uint64_t* values, size_t count) {
  using P = gdn_stage::ReplayParams;
#define REPLAY_OFFSET(field) offsetof(P, field)
  constexpr size_t offsets[] = {
      REPLAY_OFFSET(q), REPLAY_OFFSET(k), REPLAY_OFFSET(v),
      REPLAY_OFFSET(a), REPLAY_OFFSET(b), REPLAY_OFFSET(a_log),
      REPLAY_OFFSET(dt_bias), REPLAY_OFFSET(out), REPLAY_OFFSET(checkpoint),
      REPLAY_OFFSET(d_cache), REPLAY_OFFSET(k_cache), REPLAY_OFFSET(g_cache),
      REPLAY_OFFSET(rawv_cache), REPLAY_OFFSET(rawk_cache), REPLAY_OFFSET(beta_cache),
      REPLAY_OFFSET(query_start_loc), REPLAY_OFFSET(state_indices),
      REPLAY_OFFSET(replay_indices), REPLAY_OFFSET(write_pos), REPLAY_OFFSET(cache_base),
      REPLAY_OFFSET(is_flush), REPLAY_OFFSET(h), REPLAY_OFFSET(hv),
      REPLAY_OFFSET(null_block_id), REPLAY_OFFSET(gates_fp32),
      REPLAY_OFFSET(dt_bias_fp32), REPLAY_OFFSET(state_indices_int64),
      REPLAY_OFFSET(replay_indices_int64), REPLAY_OFFSET(storage), REPLAY_OFFSET(scale),
      REPLAY_OFFSET(stride_q_t), REPLAY_OFFSET(stride_k_t), REPLAY_OFFSET(stride_v_t),
      REPLAY_OFFSET(stride_a_t), REPLAY_OFFSET(stride_b_t), REPLAY_OFFSET(stride_o_t),
      REPLAY_OFFSET(stride_state_slot), REPLAY_OFFSET(stride_d_slot),
      REPLAY_OFFSET(stride_k_slot), REPLAY_OFFSET(stride_g_slot),
      REPLAY_OFFSET(stride_rawv_slot), REPLAY_OFFSET(stride_rawk_slot),
      REPLAY_OFFSET(stride_beta_slot), REPLAY_OFFSET(stride_qsl),
      REPLAY_OFFSET(stride_state_indices), REPLAY_OFFSET(stride_replay_indices),
  };
#undef REPLAY_OFFSET
  constexpr size_t fields = sizeof(offsets) / sizeof(offsets[0]);
  if (values != nullptr && count >= fields) {
    for (size_t i = 0; i < fields; ++i) {
      values[i] = offsets[i];
    }
  }
  return fields;
}

extern "C" size_t gdn_replay_probe_scratch_bytes() {
  return sizeof(Scratch);
}

// Values are actual compiled function attributes, not a budget estimate:
// [registers/thread, local bytes/thread, static shared bytes, max threads,
//  binary version, PTX version, parameter bytes, scratch bytes].
extern "C" cudaError_t gdn_replay_probe_attributes(int role_layout,
                                                   uint64_t* values) {
  if (values == nullptr || (role_layout != 0 && role_layout != 1)) {
    return cudaErrorInvalidValue;
  }
  cudaFuncAttributes attributes{};
  const cudaError_t error = role_layout == 0
      ? cudaFuncGetAttributes(&attributes, replay_stage_dedicated)
      : cudaFuncGetAttributes(&attributes, replay_stage_role_layout);
  if (error != cudaSuccess) {
    return error;
  }
  values[0] = attributes.numRegs;
  values[1] = attributes.localSizeBytes;
  values[2] = attributes.sharedSizeBytes;
  values[3] = attributes.maxThreadsPerBlock;
  values[4] = attributes.binaryVersion;
  values[5] = attributes.ptxVersion;
  values[6] = sizeof(gdn_stage::ReplayParams);
  values[7] = sizeof(Scratch);
  return cudaSuccess;
}

// Compare output and every cache buffer, including untouched bytes, with a
// separate invocation for each range. Counts is a two-uint64 device allocation:
// [number of unequal bytes, first unequal byte offset or UINT64_MAX].
extern "C" cudaError_t gdn_replay_probe_compare_bytes(
    const void* a, const void* b, size_t bytes, uint64_t* counts,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || counts == nullptr || bytes == 0) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = cudaMemsetAsync(counts, 0, sizeof(uint64_t), stream);
  if (error != cudaSuccess) {
    return error;
  }
  error = cudaMemsetAsync(counts + 1, 0xff, sizeof(uint64_t), stream);
  if (error != cudaSuccess) {
    return error;
  }
  const size_t needed = (bytes + 255) / 256;
  const unsigned int blocks = static_cast<unsigned int>(needed < 256 ? needed : 256);
  compare_bytes<<<blocks, 256, 0, stream>>>(
      static_cast<const unsigned char*>(a), static_cast<const unsigned char*>(b),
      bytes, reinterpret_cast<unsigned long long*>(counts),
      reinterpret_cast<unsigned long long*>(counts + 1));
  return cudaPeekAtLastError();
}

// This auxiliary kernel is captured before each timing start event, not inside
// the measured stage. Allocation is caller-owned and at least 256 MiB.
extern "C" cudaError_t gdn_replay_probe_evict(void* allocation, size_t bytes,
                                            cudaStream_t stream) {
  if (allocation == nullptr || bytes < (256ull << 20) || bytes % sizeof(uint4)) {
    return cudaErrorInvalidValue;
  }
  evict_lines<<<1024, 256, 0, stream>>>(static_cast<uint4*>(allocation),
                                      bytes / sizeof(uint4));
  return cudaPeekAtLastError();
}

extern "C" cudaError_t gdn_replay_probe_l2_bytes(uint64_t* bytes) {
  if (bytes == nullptr) {
    return cudaErrorInvalidValue;
  }
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return error;
  }
  int size = 0;
  error = cudaDeviceGetAttribute(&size, cudaDevAttrL2CacheSize, device);
  if (error == cudaSuccess) {
    *bytes = size;
  }
  return error;
}
