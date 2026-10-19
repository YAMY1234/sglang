// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/float8.h>

#include "gdn_fp8_stage.cuh"

namespace gdn_fp8_tcgen {

// Tensor maps use UINT8, a [128,64] weight box or [128,8] activation
// box, and CU_TENSOR_MAP_SWIZZLE_128B. Each ring is used for one projection.
struct alignas(1024) ProjectionRing {
  alignas(1024) uint8_t weight[2][64][128];
  alignas(1024) uint8_t activation[2][8][128];
  alignas(16) uint64_t full[2];
  alignas(16) uint64_t empty[2];
  alignas(16) uint32_t tensor_memory;
  uint32_t generation[2];
  uint32_t issued_arrivals[2];
};

__device__ __forceinline__ void initialize(ProjectionRing& ring) {
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
    const void* weight_map, int first_row, ProjectionRing& ring) {
  static_assert(K % 128 == 0 && K >= 256);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      ring.generation[slot] = 0;
      ring.issued_arrivals[slot] = 0;
      gdn_fp8::expect_bytes(&ring.full[slot], 64 * 128);
      gdn_fp8::copy_tensor_2d(ring.weight[slot], weight_map, slot * 128,
                              first_row, &ring.full[slot]);
      ring.issued_arrivals[slot] = 1;
    }
  }
}

template <int Rows>
__device__ __forceinline__ uint64_t descriptor(uint8_t* address, int k_offset) {
  using Element = cutlass::float_e4m3_t;
  auto layout = cute::tile_to_shape(
      cute::UMMA::Layout_K_SW128_Atom<Element>{},
      cute::Shape<cute::Int<Rows>, cute::_128>{});
  auto tensor = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<Element*>(address)), layout);
  auto result = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tensor);
  result.start_address_ += k_offset / 16;
  return static_cast<uint64_t>(result);
}

__device__ __forceinline__ void release_tile(uint64_t* barrier) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];"
      :
      : "r"(gdn_fp8::shared_address(barrier))
      : "memory");
}

// Drain already-issued MMA work even when another CTA has set the poison.
// An unrecoverable drain timeout aborts the launch before freeing TMEM.
__device__ __forceinline__ void drain_phase(
    uint64_t* barrier, int phase, const gdn_fp8::WaitControl& control) {
  const uint64_t start = clock64();
  uint32_t ready;
  do {
    asm volatile(
        "{ .reg .pred p; "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }"
        : "=r"(ready)
        : "r"(gdn_fp8::shared_address(barrier)), "r"(phase)
        : "memory");
    if (!ready && clock64() - start >= control.timeout_cycles) {
      atomicCAS(control.sticky_error, 0u, 0x500u);
      printf("projection drain timeout block=%u phase=%d\n", blockIdx.x,
             phase);
      asm volatile("trap;");
    }
  } while (!ready);
}

__device__ __forceinline__ void drain_tma_leader(
    ProjectionRing& ring, const gdn_fp8::WaitControl& control) {
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
    ProjectionRing& ring, const gdn_fp8::WaitControl& control) {
  __syncthreads();
  if (threadIdx.x == 0) drain_tma_leader(ring, control);
  __syncthreads();
}

// All threads call; roles have been set with gdn_fp8::begin_roles().
// FP8 E4M3 operands accumulate in FP32 and round to BF16 after alpha.
template <int M, int K>
__device__ __forceinline__ bool project(
    const void* weight_map, const void* activation_map, int first_row,
    float alpha, __nv_bfloat16* output, int output_stride, ProjectionRing& ring,
    const gdn_fp8::WaitControl& control, bool prefetched = false) {
  static_assert(M >= 1 && M <= 8 && K % 128 == 0);
  constexpr int tiles = K / 128;
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
        gdn_fp8::expect_bytes(&ring.full[slot], 64 * 128);
        gdn_fp8::copy_tensor_2d(ring.weight[slot], weight_map, tile * 128,
                                first_row, &ring.full[slot]);
        ring.issued_arrivals[slot] = 1;
      }
      gdn_fp8::expect_bytes(&ring.full[slot], 8 * 128);
      gdn_fp8::copy_tensor_2d(ring.activation[slot], activation_map, tile * 128,
                              0, &ring.full[slot]);
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
      for (int offset = 0; offset < 128; offset += 32) {
        Mma::fma(descriptor<64>(&ring.weight[slot][0][0], offset),
                 descriptor<8>(&ring.activation[slot][0][0], offset),
                 ring.tensor_memory, (tile == 0 && offset == 0) ? 0u : 1u,
                 instruction);
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
__device__ __forceinline__ void release_allocation(ProjectionRing& ring) {
  __syncthreads();
  if (threadIdx.x >= 128 && threadIdx.x < 160) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                 :
                 : "r"(ring.tensor_memory)
                 : "memory");
  }
}

// Call once, after the kernel has made all its tensor-memory allocations.
__device__ __forceinline__ void release_allocation_permit() {
  if (threadIdx.x >= 128 && threadIdx.x < 160) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 ::: "memory");
  }
}

}  // namespace gdn_fp8_tcgen
