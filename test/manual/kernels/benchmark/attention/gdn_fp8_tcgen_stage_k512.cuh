// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gdn_fp8_tcgen_stage.cuh"

namespace gdn_fp8_tcgen {

__device__ __forceinline__ void copy_tensor_3d(
    void* destination, const void* tensor_map, int row, int k_chunk,
    uint64_t* barrier) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {0, %2, %3}], [%4];"
      :
      : "r"(gdn_fp8::shared_address(destination)), "l"(tensor_map), "r"(row),
        "r"(k_chunk), "r"(gdn_fp8::shared_address(barrier))
      : "memory");
}

// Rank-three tensor maps use UINT8, dimensions [128, rows, K/128],
// byte strides [K,128], boxes [128,64,4] or [128,8,4], and SWIZZLE_128B.
struct alignas(1024) ProjectionRing512 {
  alignas(1024) uint8_t weight[2][4][64][128];
  alignas(1024) uint8_t activation[2][4][8][128];
  alignas(16) uint64_t full[2];
  alignas(16) uint64_t empty[2];
  alignas(16) uint32_t tensor_memory;
  uint32_t generation[2];
  uint32_t issued_arrivals[2];
};

__device__ __forceinline__ void initialize(ProjectionRing512& ring) {
  if (threadIdx.x == 0) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      gdn_fp8::barrier_init(&ring.full[slot], 2);
      gdn_fp8::barrier_init(&ring.empty[slot], 1);
      ring.generation[slot] = 0;
      ring.issued_arrivals[slot] = 0;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (threadIdx.x >= 128 && threadIdx.x < 160) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
        :
        : "r"(gdn_fp8::shared_address(&ring.tensor_memory))
        : "memory");
  }
  __syncthreads();
}

template <int K>
__device__ __forceinline__ void prefetch_weight_prefix(
    const void* weight_map, int first_row, ProjectionRing512& ring) {
  static_assert(K % 512 == 0 && K >= 1024);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      ring.generation[slot] = 0;
      ring.issued_arrivals[slot] = 0;
      gdn_fp8::expect_bytes(&ring.full[slot], 64 * 512);
      copy_tensor_3d(ring.weight[slot], weight_map, first_row, slot * 4,
                      &ring.full[slot]);
      ring.issued_arrivals[slot] = 1;
    }
  }
}

__device__ __forceinline__ void drain_tma_leader(
    ProjectionRing512& ring, const gdn_fp8::WaitControl& control) {
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    const unsigned int issued = ring.issued_arrivals[slot];
    if (!issued) continue;
    // Supply a missing role arrival without issuing an activation read.
    if (!(issued & 1u)) gdn_fp8::expect_bytes(&ring.full[slot], 0);
    if (!(issued & 2u)) gdn_fp8::expect_bytes(&ring.full[slot], 0);
    drain_phase(&ring.full[slot], ring.generation[slot] & 1u, control);
    ring.issued_arrivals[slot] = 0;
  }
}

// All threads call if input readiness fails after a weight prefix was issued.
// Complete both role arrivals and all async copies before the CTA may exit.
__device__ __forceinline__ void abort_prefetch(
    ProjectionRing512& ring, const gdn_fp8::WaitControl& control) {
  __syncthreads();
  if (threadIdx.x == 0) drain_tma_leader(ring, control);
  __syncthreads();
}

// All threads call; roles have been set with gdn_fp8::begin_roles().
// FP8 E4M3 operands accumulate in FP32 and round to BF16 after alpha.
template <int M, int K>
__device__ __forceinline__ bool project(
    const void* weight_map, const void* activation_map, int first_row,
    float alpha, __nv_bfloat16* output, int output_stride, ProjectionRing512& ring,
    const gdn_fp8::WaitControl& control, bool prefetched = false) {
  static_assert(M >= 1 && M <= 8 && K % 512 == 0);
  constexpr int tiles = K / 512;
  if (threadIdx.x == 0) {
    gdn_fp8::activation_proxy_fence();
    for (int tile = 0; tile < tiles; ++tile) {
      const int slot = tile & 1;
      if (tile >= 2 &&
          !gdn_fp8::wait_phase(&ring.empty[slot], ((tile / 2) - 1) & 1,
                               control, 0x300u + slot)) {
        break;
      }
      if (!prefetched || tile >= 2) {
        ring.generation[slot] = tile / 2;
        ring.issued_arrivals[slot] = 0;
        gdn_fp8::expect_bytes(&ring.full[slot], 64 * 512);
        copy_tensor_3d(ring.weight[slot], weight_map, first_row, tile * 4,
                        &ring.full[slot]);
        ring.issued_arrivals[slot] = 1;
      }
      gdn_fp8::expect_bytes(&ring.full[slot], 8 * 512);
      copy_tensor_3d(ring.activation[slot], activation_map, 0, tile * 4,
                      &ring.full[slot]);
      ring.issued_arrivals[slot] |= 2;
    }
  } else if (threadIdx.x >= 128 && threadIdx.x < 160) {
    using Element = cutlass::float_e4m3_t;
    using Mma = cute::SM100_MMA_F8F6F4_SS;
    const uint64_t instruction =
        cute::UMMA::make_runtime_instr_desc<
            Element, Element, float, 64, 8, cute::UMMA::Major::K,
            cute::UMMA::Major::K>();
    int last_issued = -1;
    for (int tile = 0; tile < tiles; ++tile) {
      const int slot = tile & 1;
      int ready = 0;
      if (threadIdx.x == 128) {
        ready = gdn_fp8::wait_phase(&ring.full[slot], (tile / 2) & 1, control,
                                    0x400u + slot);
      }
      ready = __shfl_sync(0xffffffffu, ready, 0);
      if (!ready) {
        break;
      }
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk) {
#pragma unroll
        for (int offset = 0; offset < 128; offset += 32) {
          Mma::fma(descriptor<64>(&ring.weight[slot][chunk][0][0], offset),
                   descriptor<8>(&ring.activation[slot][chunk][0][0], offset),
                   ring.tensor_memory,
                   (tile == 0 && chunk == 0 && offset == 0) ? 0u : 1u,
                   instruction);
        }
      }
      if (threadIdx.x == 128) release_tile(&ring.empty[slot]);
      last_issued = tile;
    }
    if (threadIdx.x == 128 && last_issued >= 0) {
      drain_phase(&ring.empty[last_issued & 1], (last_issued / 2) & 1,
                   control);
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (threadIdx.x == 0 && gdn_fp8::error_acquire(control)) {
    drain_tma_leader(ring, control);
  }
  __syncthreads();

  if (threadIdx.x >= 128) {
    const int warp = (threadIdx.x - 128) / 32;
    const int lane = threadIdx.x & 31;
    int failed = lane == 0 ? gdn_fp8::error_acquire(control) != 0 : 0;
    failed = __shfl_sync(0xffffffffu, failed, 0);
    float values[8];
    if (!failed) {
      const uint32_t address = ring.tensor_memory + ((warp * 32u) << 16);
      asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
          : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
            "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
            "=f"(values[6]), "=f"(values[7])
          : "r"(address));
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    }
    if (lane < 16) {
#pragma unroll
      for (int token = 0; token < M; ++token) {
        output[token * output_stride + first_row + warp * 16 + lane] =
            __float2bfloat16_rn(failed ? __int_as_float(0x7fc00000)
                                      : alpha * values[token]);
      }
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  return gdn_fp8::error_acquire(control) == 0;
}

// All threads call after project() or abort_prefetch() has drained async work.
__device__ __forceinline__ void release_allocation(ProjectionRing512& ring) {
  __syncthreads();
  if (threadIdx.x >= 128 && threadIdx.x < 160) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                 :
                 : "r"(ring.tensor_memory)
                 : "memory");
  }
}

}  // namespace gdn_fp8_tcgen
