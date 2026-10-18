// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>

namespace gdn_fp8 {

// The caller supplies 256 threads and keeps each 128-thread role uniform.
// Register transfers are local to this CTA (PTX ISA, setmaxnreg).
template <int ProducerRegisters = 32, int ConsumerRegisters = 128>
__device__ __forceinline__ void begin_roles() {
  static_assert(ProducerRegisters >= 24 && ProducerRegisters % 8 == 0);
  static_assert(ConsumerRegisters >= ProducerRegisters &&
                ConsumerRegisters <= 256 && ConsumerRegisters % 8 == 0);
  if (threadIdx.x < 128) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"
                 :
                 : "n"(ProducerRegisters));
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"
                 :
                 : "n"(ConsumerRegisters));
  }
}

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier, int arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(shared_address(barrier)), "r"(arrivals)
               : "memory");
}

__device__ __forceinline__ void expect_bytes(uint64_t* barrier, int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
      :
      : "r"(shared_address(barrier)), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void arrive(uint64_t* barrier) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               :
               : "r"(shared_address(barrier))
               : "memory");
}

struct WaitControl {
  uint64_t timeout_cycles;
  unsigned int* sticky_error;
};

__device__ __forceinline__ unsigned int error_acquire(
    const WaitControl& control) {
  unsigned int error;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(error)
               : "l"(control.sticky_error)
               : "memory");
  return error;
}

// After the caller acquires its activation epoch and joins its writers, the
// thread issuing TMA executes this cross-proxy fence before any activation copy.
__device__ __forceinline__ void activation_proxy_fence() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ __forceinline__ bool wait_phase(uint64_t* barrier, int phase,
                                          const WaitControl& control,
                                          unsigned int error_code) {
  const uint32_t address = shared_address(barrier);
  const uint64_t start = clock64();
  uint32_t ready;
  unsigned int polls = 0;
  do {
    asm volatile(
        "{ .reg .pred p; "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }"
        : "=r"(ready)
        : "r"(address), "r"(phase)
        : "memory");
    if (!ready && (++polls & 255u) == 0) {
      if (error_acquire(control)) return false;
      if (clock64() - start >= control.timeout_cycles) {
        if (atomicCAS(control.sticky_error, 0u, error_code) == 0u) {
          printf("projection barrier timeout block=%u phase=%d code=%u\n",
                 blockIdx.x, phase, error_code);
        }
        return false;
      }
    }
  } while (!ready);
  return true;
}

// Each tensor map has byte elements, K as its contiguous dimension, no
// interleave or swizzle, and a box of [TileK, TileN] or [TileK, 8].
__device__ __forceinline__ void copy_tensor_2d(void* destination,
                                             const void* tensor_map,
                                             int k,
                                             int row,
                                             uint64_t* barrier) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(shared_address(destination)), "l"(tensor_map), "r"(k), "r"(row),
        "r"(shared_address(barrier))
      : "memory");
}

template <int TileN = 64, int TileK = 128, int Stages = 2>
struct alignas(128) ProjectionRing {
  static_assert(TileN >= 64 && TileN % 64 == 0);
  static_assert(TileK >= 32 && TileK % 32 == 0 && TileK <= 256);
  static_assert(Stages >= 2);

  alignas(128) uint8_t weight[Stages][TileN][TileK];
  alignas(128) uint8_t activation[Stages][8][TileK];
  alignas(16) uint64_t full[Stages];
  alignas(16) uint64_t empty[Stages];
};

template <int TileN, int TileK, int Stages>
__device__ __forceinline__ void initialize(
    ProjectionRing<TileN, TileK, Stages>& ring) {
  if (threadIdx.x == 0) {
#pragma unroll
    for (int slot = 0; slot < Stages; ++slot) {
      // Weight and activation producers each arrive once.  All consumer
      // threads arrive before the slot may be overwritten.
      barrier_init(&ring.full[slot], 2);
      barrier_init(&ring.empty[slot], 128);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
}

// This function is producer-only; other threads may execute an independent
// stage. The first bounded weight tiles are retained until their activations
// arrive. Calling project() with prefetched=true consumes this prefix.
template <int K, int TileN, int TileK, int Stages>
__device__ __forceinline__ void prefetch_weight_prefix(
    const void* weight_map,
    int first_row,
    ProjectionRing<TileN, TileK, Stages>& ring) {
  static_assert(K % TileK == 0 && K / TileK >= Stages);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int slot = 0; slot < Stages; ++slot) {
      expect_bytes(&ring.full[slot], TileN * TileK);
      copy_tensor_2d(ring.weight[slot], weight_map, slot * TileK, first_row,
                     &ring.full[slot]);
    }
  }
}

__device__ __forceinline__ void mma_fp8(float (&accumulator)[4],
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint32_t b0,
                                      uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
        "+f"(accumulator[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Computes output[M,N] = BF16(alpha * activation[M,K] * weight[N,K]^T).
// FP8 E4M3 inputs accumulate in FP32.  The reduction order is numeric-class.
// All 256 threads call this stage; producer and consumer roles must have
// already been assigned.  Ring state is used for one projection per kernel.
template <int M, int K, int TileN = 64, int TileK = 128, int Stages = 2>
__device__ __forceinline__ bool project(
    const void* weight_map,
    const void* activation_map,
    int first_row,
    float alpha,
    __nv_bfloat16* output,
    int output_stride,
    ProjectionRing<TileN, TileK, Stages>& ring,
    const WaitControl& control,
    bool prefetched = false) {
  static_assert(M >= 1 && M <= 8 && K % TileK == 0);
  constexpr int kTiles = K / TileK;
  constexpr int kAtomsPerWarp = TileN / 64;

  if (threadIdx.x < 128) {
    if (threadIdx.x == 0) {
      activation_proxy_fence();
      for (int tile = 0; tile < kTiles; ++tile) {
        const int slot = tile % Stages;
        if (tile >= Stages) {
          if (!wait_phase(&ring.empty[slot], ((tile / Stages) - 1) & 1,
                          control, 0x100u + slot)) {
            break;
          }
        }
        if (!prefetched || tile >= Stages) {
          expect_bytes(&ring.full[slot], TileN * TileK);
          copy_tensor_2d(ring.weight[slot], weight_map, tile * TileK, first_row,
                         &ring.full[slot]);
        }
        expect_bytes(&ring.full[slot], 8 * TileK);
        copy_tensor_2d(ring.activation[slot], activation_map, tile * TileK, 0,
                       &ring.full[slot]);
      }
    }
  } else {
    const int lane = threadIdx.x & 31;
    const int warp = (threadIdx.x - 128) >> 5;
    const int group = lane >> 2;
    const int k_lane = (lane & 3) * 4;
    float accumulator[kAtomsPerWarp][4] = {};

    for (int tile = 0; tile < kTiles; ++tile) {
      const int slot = tile % Stages;
      if (!wait_phase(&ring.full[slot], (tile / Stages) & 1, control,
                      0x200u + slot)) {
        break;
      }
#pragma unroll
      for (int offset = 0; offset < TileK; offset += 32) {
        const auto* activation = ring.activation[slot][group] + offset + k_lane;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(activation);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(activation + 16);
#pragma unroll
        for (int atom = 0; atom < kAtomsPerWarp; ++atom) {
          const int row = warp * (16 * kAtomsPerWarp) + atom * 16 + group;
          const auto* weight = ring.weight[slot][row] + offset + k_lane;
          const uint32_t a0 = *reinterpret_cast<const uint32_t*>(weight);
          const uint32_t a1 =
              *reinterpret_cast<const uint32_t*>(weight + 8 * TileK);
          const uint32_t a2 = *reinterpret_cast<const uint32_t*>(weight + 16);
          const uint32_t a3 =
              *reinterpret_cast<const uint32_t*>(weight + 8 * TileK + 16);
          mma_fp8(accumulator[atom], a0, a1, a2, a3, b0, b1);
        }
      }
      arrive(&ring.empty[slot]);
    }

    const bool failed = error_acquire(control) != 0;
#pragma unroll
    for (int atom = 0; atom < kAtomsPerWarp; ++atom) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        const int token = (lane & 3) * 2 + (value & 1);
        const int channel = first_row + warp * (16 * kAtomsPerWarp) + atom * 16 +
                            group + ((value >= 2) ? 8 : 0);
        if (token < M) {
          output[token * output_stride + channel] =
              __float2bfloat16_rn(failed
                                    ? __int_as_float(0x7fc00000)
                                    : accumulator[atom][value] * alpha);
        }
      }
    }
  }
  __syncthreads();
  // A caller must propagate this poison through its global waits and must not
  // publish an output-ready epoch when this stage returns false.
  return error_acquire(control) == 0;
}

}  // namespace gdn_fp8
