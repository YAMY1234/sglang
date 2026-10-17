// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Developed at NVIDIA. The host side is
// python/sglang/kernels/ops/speculative/dspark/local_argmax.py.
//
// Row argmax of vocab-sharded logits across tensor-parallel ranks in one launch
// per rank, so no rank gathers the full logits. Template argument LogitT is the
// element type of this rank's [rows, vocab_local] logits shard (bf16_t or fp32_t).
//
// Each row splits into `splits` chunks of `chunk` columns; work item (row, chunk)
// runs on one CTA, which stores its best key, tagged with the launch epoch, into
// slot [parity][item][source rank] of every rank's mailbox (symmetric memory, one
// writer per slot and launch). Then each warp takes rows, waits for the
// world * splits records of its row in the local mailbox, resets them to zero and
// writes the argmax index. The grid must be co-resident (the host launches it
// cooperatively) and every CTA posts all its items before it waits. Every wait is
// bounded by `timeout_ns`; past it the kernel reports and traps.
//
// Key = canonical order-preserving bits of the fp32 value << 17 | (2^17 - 1 -
// global index), so the maximum key is torch.argmax's choice: ties go to the
// smaller index, -0 equals +0, and a NaN beats every number. Record = key << 15 |
// tag. Launch L and L + 1 use different parities; slot reuse at L + 2 is safe
// because a rank starts L + 2 only after every peer posted in L + 1, i.e. finished
// L. Readers reset what they read, so a slot no launch rewrote holds no old record.
// Columns at or past `valid_local` (vocab padding of this shard) take no part.

#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cstdio>

namespace sglang {

namespace dspark_local_argmax {

using u32 = unsigned int;
using u64 = unsigned long long;
using i32 = int;
using i64 = long long;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRanks = 32;  // one posting lane per rank
constexpr u32 kTagBits = 15;
constexpr u32 kTagMask = (1u << kTagBits) - 1;
constexpr u32 kIndexBits = 17;
constexpr u32 kIndexMask = (1u << kIndexBits) - 1;
// Ordered bits of +0 and of every NaN.
constexpr u32 kOrderedZero = 0x80000000u;
constexpr u32 kOrderedNan = 0xFFFFFFFFu;

__device__ __forceinline__ float ToFloat(float v) {
  return v;
}
__device__ __forceinline__ float ToFloat(bf16_t v) {
  return __bfloat162float(v);
}

// Integer tests only, so fast-math flags cannot fold the NaN and signed-zero cases.
__device__ __forceinline__ u32 OrderedBits(float value) {
  const u32 bits = __float_as_uint(value);
  const u32 magnitude = bits & 0x7FFFFFFFu;
  if (magnitude > 0x7F800000u) return kOrderedNan;
  if (magnitude == 0) return kOrderedZero;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ u64 WarpMaxU64(u64 value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const u64 other = __shfl_xor_sync(0xffffffffu, value, offset);
    value = other > value ? other : value;
  }
  return value;
}

// Never zero; distinct for the two launches that share a parity slot.
__device__ __forceinline__ u32 EpochTag(u32 count) {
  return 1u + count % 32767u;
}

__device__ __forceinline__ void StoreRecord(u64* address, u64 value) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ u64 LoadRecord(const u64* address) {
  u64 value;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ u64 GlobalTimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Kept out of line so the report's arguments cost the hot path no registers.
__device__ __noinline__ void ReportTimeout(int row, i64 timeout_ns, int source, u32 tag, u32 seen) {
  printf(
      "DSpark local argmax: row %d waited %lld ns for rank %d (tag %u, saw %u); "
      "a peer rank stopped or ran a different launch sequence\n",
      row,
      static_cast<long long>(timeout_ns),
      source,
      tag,
      seen);
  __trap();
}

// Spin until the record at `address` carries `tag`; return its key. A wait past
// `timeout_ns` reports and traps instead of hanging.
__device__ __forceinline__ u64 AwaitRecord(const u64* address, u32 tag, i64 timeout_ns, int row, int source) {
  u64 value = LoadRecord(address);
  if ((value & kTagMask) == tag) return value >> kTagBits;
  const u64 start = GlobalTimer();
  u32 spins = 0;
  while ((value & kTagMask) != tag) {
    value = LoadRecord(address);
    if ((++spins & 1023u) == 0 && GlobalTimer() - start > static_cast<u64>(timeout_ns)) {
      ReportTimeout(row, timeout_ns, source, tag, static_cast<u32>(value & kTagMask));
    }
  }
  return value >> kTagBits;
}

template <typename LogitT>
__global__ void __launch_bounds__(kThreads) local_argmax_kernel(
    const LogitT* __restrict__ logits,  // [rows, vocab_local], 16-byte aligned rows
    i64 row_stride,
    int rows,
    int vocab_local,
    int valid_local,
    int splits,
    int chunk,
    const u64* __restrict__ peers,  // [world] mailbox bases
    i64* ids,                       // [rows] global argmax index
    u32* counters,                  // [gridDim.x] per-CTA launch counts
    int rank,
    int world,
    int max_items,
    i64 timeout_ns) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(LogitT));
  __shared__ u64 warp_best[kWarps];
  __shared__ u32 s_count;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (threadIdx.x == 0) s_count = *reinterpret_cast<volatile u32*>(&counters[blockIdx.x]);
  __syncthreads();
  const u32 tag = EpochTag(s_count);
  const i64 parity_base = static_cast<i64>(s_count & 1u) * max_items * world;
  const int vocab_offset = rank * vocab_local;
  const int items = rows * splits;

  for (int item = static_cast<int>(blockIdx.x); item < items; item += static_cast<int>(gridDim.x)) {
    const int row = item / splits;
    const int begin = (item - row * splits) * chunk;
    const int end = min(begin + chunk, valid_local);
    const LogitT* const base = logits + row * row_stride;
    // Columns visited in increasing order, so a strict > keeps the first maximum.
    u32 best = 0;
    int best_index = -1;
#pragma unroll 4
    for (int i = begin + static_cast<int>(threadIdx.x) * kVec; i < end; i += kThreads * kVec) {
      const uint4 raw = *reinterpret_cast<const uint4*>(base + i);
      const LogitT* const v = reinterpret_cast<const LogitT*>(&raw);
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        const u32 bits = OrderedBits(ToFloat(v[e]));
        if (i + e < end && (best_index < 0 || bits > best)) {
          best = bits;
          best_index = i + e;
        }
      }
    }
    u64 key = best_index < 0
                  ? 0ull
                  : (static_cast<u64>(best) << kIndexBits) | (kIndexMask - static_cast<u32>(vocab_offset + best_index));
    key = WarpMaxU64(key);
    if (lane == 0) warp_best[warp] = key;
    __syncthreads();
    if (warp == 0) {
      key = lane < kWarps ? warp_best[lane] : 0ull;
      key = WarpMaxU64(key);
      if (lane < world) {
        u64* const box = reinterpret_cast<u64*>(peers[lane]);
        StoreRecord(box + parity_base + static_cast<i64>(item) * world + rank, (key << kTagBits) | tag);
      }
    }
    __syncthreads();
  }

  // Rows over all warps of the grid; a row's records are world * splits contiguous slots.
  u64* const mine = reinterpret_cast<u64*>(peers[rank]);
  const int records = world * splits;
  for (int row = static_cast<int>(blockIdx.x) * kWarps + warp; row < rows;
       row += static_cast<int>(gridDim.x) * kWarps) {
    u64* const slot = mine + parity_base + static_cast<i64>(row) * records;
    u64 best = 0ull;
    for (int j = lane; j < records; j += 32) {
      const u64 key = AwaitRecord(slot + j, tag, timeout_ns, row, j % world);
      best = key > best ? key : best;
      StoreRecord(slot + j, 0ull);
    }
    best = WarpMaxU64(best);
    if (lane == 0) ids[row] = static_cast<i64>(kIndexMask - static_cast<u32>(best & kIndexMask));
  }
  // This thread read the counter at launch start; the next launch is stream-ordered.
  if (threadIdx.x == 0) counters[blockIdx.x] = s_count + 1u;
}

}  // namespace dspark_local_argmax

/// \brief Host entries of the vocab-sharded greedy argmax exchange.
/// \tparam LogitT Element type of this rank's logits shard (bf16_t or fp32_t).
template <typename LogitT>
struct DSparkLocalArgmax {
  static constexpr auto kernel() {
    return dspark_local_argmax::local_argmax_kernel<LogitT>;
  }

  /// \brief CTAs of this kernel that can be co-resident per SM on the current device.
  static int64_t max_ctas_per_sm() {
    int per_sm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel(), dspark_local_argmax::kThreads, 0));
    return static_cast<int64_t>(per_sm);
  }

  /// \brief Cooperative launch of `ctas` CTAs on the current stream; the host
  /// (local_argmax.py) owns the shape, layout and mailbox contract.
  /// \param logits [rows, vocab_local] shard, unit column stride, 16-byte aligned rows.
  /// \param peers int64 [world] mailbox base of every rank.
  /// \param ids int64 [>= rows] global argmax index of each row.
  /// \param counters int32 [ctas] per-CTA launch counts.
  static void
  run(tvm::ffi::TensorView logits,
      tvm::ffi::TensorView peers,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView counters,
      int64_t valid_local,
      int64_t splits,
      int64_t chunk,
      int64_t rank,
      int64_t world,
      int64_t max_items,
      int64_t timeout_ns,
      int64_t ctas) {
    using namespace dspark_local_argmax;
    host::RuntimeCheck(timeout_ns > 0, "every cross-rank wait needs a positive deadline");
    host::RuntimeCheck(world >= 1 && world <= kMaxRanks, "unsupported world size");
    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeCooperative;
    attribute.val.cooperative = 1;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(static_cast<uint32_t>(ctas));
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = 0;
    config.stream = host::LaunchKernel::resolve_device(logits.device());
    config.attrs = &attribute;
    config.numAttrs = 1;
    host::RuntimeDeviceCheck(
        ::cudaLaunchKernelEx(
            &config,
            kernel(),
            static_cast<const LogitT*>(logits.data_ptr()),
            static_cast<i64>(logits.stride(0)),
            static_cast<int>(logits.size(0)),
            static_cast<int>(logits.size(1)),
            static_cast<int>(valid_local),
            static_cast<int>(splits),
            static_cast<int>(chunk),
            static_cast<const u64*>(peers.data_ptr()),
            static_cast<i64*>(ids.data_ptr()),
            static_cast<u32*>(counters.data_ptr()),
            static_cast<int>(rank),
            static_cast<int>(world),
            static_cast<int>(max_items),
            static_cast<i64>(timeout_ns)));
  }
};

}  // namespace sglang
