// DSA decode indexer logits over a paged FP8 index-K cache, one Q block per request.
//
// Adapted from DeepGEMM (sgl-deep-gemm 0.2.8, https://github.com/sgl-project/DeepGEMM, a fork of
// https://github.com/deepseek-ai/DeepGEMM):
//   - deep_gemm/include/deep_gemm/legacy/impls/sm100_mqa_logits.cuh
//       (sm100_mqa_logits_core_impl and the sm100_paged_mqa_logits entry)
//   - deep_gemm/include/deep_gemm/legacy/scheduler/sm100_paged_mqa_logits.cuh
//       (RequestInfo and SM100PagedMQALogitsScheduler)
//   - csrc/jit_kernels/impls/legacy/sm100_mqa_logits.hpp (host TMA descriptors and launch)
//
// MIT License
//
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Changes against DeepGEMM's FP8 paged path (32 heads x 128, 64-key pages, 2-D context lengths):
//   - BLOCK_Q is next_n instead of 128 / heads = 4, so every request is one Q block and its index K
//     is streamed from memory once (DeepGEMM splits next_n 5..8 into two Q blocks that each stream
//     the request's KV). UMMA_N = next_n * 32.
//   - TMEM accumulator stages, Q stages, KV stages and the weights' home (registers or shared memory)
//     are template parameters, so N up to 256 fits the 512 TMEM columns and the 227 KB of shared memory.
//   - Only the FP8 / FP32-reduce / non-varlen paths are kept.
// Unchanged, so every logit is bitwise DeepGEMM's: the schedule metadata (DeepGEMM's own
// get_paged_mqa_logits_metadata output), the per-SM task walk, the UMMA (kind::f8f6f4, M 128,
// K 32 x 4 in the same order, A = 128 keys, B = the block's rows x heads), and the epilogue
// arithmetic (relu via (a + |a|) * w, two FFMA2 chains over head pairs, /2, x kv scale), the columns
// it writes (whole 256-key splits of the request's last-row length) and the ones it leaves unwritten.

#pragma once

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/legacy/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include <cstdint>

// Top-level namespace: sglang::device / sglang::host must not shadow DeepGEMM names.
namespace sgl_mqa1b {

using namespace deep_gemm;

constexpr uint32_t kNumHeads = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kPageKV = 64;
constexpr uint32_t kSplitKV = 256;
constexpr uint32_t kNumSpecializedThreads = 128;
constexpr uint32_t kNumMathThreads = 256;
constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
constexpr uint32_t kMaxDynamicSmem = 232448;

template <uint32_t kNextN, uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages>
using SharedStorage = legacy::layout::MQALogitsSharedStorage<
    kNumHeads,
    kHeadDim,
    false,
    kNextN,
    kSplitKV,
    kNumQStages,
    kNumKVStages,
    kNumTmemStages,
    cutlass::float_e4m3_t,
    float>;

// DeepGEMM's SM100PagedMQALogitsScheduler for 2-D context lengths, without varlen, with
// BLOCK_Q = kNextN: a request is one Q block, so a task covers all of the request's splits this SM
// owns. Metadata: per-SM start (request's first q token, split), as DeepGEMM's metadata kernel writes.
template <uint32_t kNextN>
struct OneBlockScheduler {
  static constexpr uint32_t kNumPagesPerSplit = kSplitKV / kPageKV;

  const uint32_t* context_lens;
  const uint32_t* block_table;
  uint32_t block_table_stride;
  uint32_t num_q_tokens_total;

  // Walk state: the current request (first q token, splits, pages) and the next split.
  uint32_t cur_q_token_start;
  uint32_t cur_num_kv_splits;
  uint32_t cur_num_kv_pages;
  uint32_t cur_kv_split_base;
  uint32_t end_q_token_idx, end_kv_split_idx;
  bool done;

  // Geometry stashed by `next_q_block` for the accessors below.
  uint32_t cur_block_table_row;
  uint32_t cur_q_block_token_base;
  uint32_t cur_request_num_kv_pages;

  CUTLASS_DEVICE void set_request(const uint32_t& q_token_idx) {
    // 2-D lengths: the request's last row bounds its keys.
    const uint32_t context_len = context_lens[q_token_idx + kNextN - 1];
    cur_q_token_start = q_token_idx;
    cur_num_kv_splits = math::ceil_div(context_len, kSplitKV);
    cur_num_kv_pages = math::ceil_div(context_len, kPageKV);
  }

  CUTLASS_DEVICE OneBlockScheduler(
      const uint32_t& sm_idx,
      const uint32_t* context_lens,
      const uint32_t* schedule_meta,
      const uint32_t* block_table,
      const uint32_t& block_table_stride,
      const uint32_t& num_q_tokens_total) {
    this->context_lens = context_lens;
    this->block_table = block_table;
    this->block_table_stride = block_table_stride;
    this->num_q_tokens_total = num_q_tokens_total;

    const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
    const auto end = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
    end_q_token_idx = end.x;
    end_kv_split_idx = end.y;

    cur_kv_split_base = start.y;
    done = (start.x >= num_q_tokens_total) or (start.x == end_q_token_idx and start.y >= end_kv_split_idx);
    cur_q_token_start = 0;
    cur_num_kv_splits = 0;
    cur_num_kv_pages = 0;
    if (not done) set_request(start.x);

    cur_block_table_row = 0;
    cur_q_block_token_base = 0;
    cur_request_num_kv_pages = 0;
  }

  // Emit the next (request, split range) task and stash its addressing geometry.
  CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_split_base, uint32_t& num_kv_splits) {
    q_block_idx = 0;  // addressing uses stashed state
    if (done) return false;

    // Exclusive split bound for the current request, clamped at the next SM's start.
    const uint32_t upper = (cur_q_token_start == end_q_token_idx) ? end_kv_split_idx : cur_num_kv_splits;
    cur_block_table_row = cur_q_token_start / kNextN;
    cur_q_block_token_base = cur_q_token_start;
    cur_request_num_kv_pages = cur_num_kv_pages;
    kv_split_base = cur_kv_split_base;
    num_kv_splits = upper - cur_kv_split_base;

    // Advance to the next request (owned from split 0), or stop at the next SM's start.
    if (cur_q_token_start == end_q_token_idx) {
      done = true;
    } else {
      const uint32_t next_q_token = cur_q_token_start + kNextN;
      cur_kv_split_base = 0;
      if (next_q_token >= num_q_tokens_total) {
        done = true;
      } else {
        set_request(next_q_token);
        // The new request may already be this SM's end.
        if (cur_q_token_start == end_q_token_idx and end_kv_split_idx == 0) done = true;
      }
    }
    return true;
  }

  CUTLASS_DEVICE uint32_t get_q_tma_token_base() const {
    return cur_q_block_token_base;
  }

  CUTLASS_DEVICE uint32_t get_kv_page_coord_by_page_offset(const uint32_t& page_offset) const {
    if (page_offset >= cur_request_num_kv_pages) return 0;
    const auto block_table_offset = cur_block_table_row * static_cast<uint64_t>(block_table_stride);
    return block_table[block_table_offset + page_offset];
  }

  CUTLASS_DEVICE uint32_t get_logits_row(const uint32_t& token_idx) const {
    return cur_q_block_token_base + token_idx;
  }

  CUTLASS_DEVICE static uint32_t
  get_logits_col(const uint32_t& kv_split_base, const uint32_t& kv_split_idx, const uint32_t& math_thread_idx) {
    return (kv_split_base + kv_split_idx) * kSplitKV + math_thread_idx;
  }
};

// sm100_mqa_logits_core_impl (FP8, FP32 reduce, paged, no clean / compressed logits) with
// BLOCK_Q = kNextN. kWeightsInSmem reads each row's 32 weights from the Q stage per use instead of
// holding BLOCK_Q x 32 of them in registers (same values, same arithmetic).
template <uint32_t kNextN, uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages, bool kWeightsInSmem, bool kRowSplit>
__global__ __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1) void paged_mqa_logits_1blk_kernel(
    const uint32_t num_q_tokens_total,
    const uint32_t logits_stride,
    const uint32_t block_table_stride,
    const uint32_t* context_lens,
    float* logits,
    const uint32_t* block_table,
    const uint32_t* schedule_meta,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using qk_dtype_t = cutlass::float_e4m3_t;
  using Scheduler = OneBlockScheduler<kNextN>;
  constexpr uint32_t BLOCK_Q = kNextN;
  constexpr uint32_t SPLIT_KV = kSplitKV;

  const auto sm_idx = blockIdx.x;
  const auto make_scheduler = [&]() {
    return Scheduler(sm_idx, context_lens, schedule_meta, block_table, block_table_stride, num_q_tokens_total);
  };
  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  static constexpr uint32_t UMMA_M = 128;
  static constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
  static constexpr uint32_t UMMA_N = math::constexpr_align(BLOCK_QH, 8u);
  static constexpr uint32_t UMMA_K = 32;
  static constexpr uint32_t kNumQKBytesPerToken = kHeadDim;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_QH * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = SPLIT_KV * sizeof(float);
  static constexpr uint32_t kNumWeightBytesPerRow =
      math::constexpr_align(kNumHeads * static_cast<uint32_t>(sizeof(float)), 16u);
  static constexpr uint32_t kNumWeightElementsPerRow = kNumWeightBytesPerRow / static_cast<uint32_t>(sizeof(float));
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumWeightBytesPerRow;

  static_assert(SPLIT_KV == kNumMathWarpGroups * UMMA_M, "Invalid `SPLIT_KV`");
  static_assert(UMMA_N % 16 == 0 and UMMA_N <= 256, "Invalid UMMA_N for M 128");
  static_assert(kNumMathWarpGroups <= kNumTmemStages, "Math warp groups exceed TMEM stages");

  using Storage = SharedStorage<kNextN, kNumQStages, kNumKVStages, kNumTmemStages>;
  extern __shared__ __align__(Storage::kSwizzleAlignment) uint8_t smem_buffer[];
  auto& smem = *reinterpret_cast<Storage*>(smem_buffer);

  constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
  constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();
  static_assert(kNumTmemCols <= 512, "Too many tensor memory");

  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      smem.full_q_barriers[i].init(1);
      smem.empty_q_barriers[i].init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      smem.full_kv_barriers[i].init(1);
      smem.empty_kv_barriers[i].init(kNumMathThreads);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumTmemStages; ++i) {
      smem.full_tmem_barriers[i].init(1);
      smem.empty_tmem_barriers[i].init(kRowSplit ? kNumMathThreads : 128);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();

  if (warp_idx == kSpecWarpStart + 2) cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
  __syncthreads();

  RingPipeline<kNumQStages> q_pipeline;
  RingPipeline<kNumKVStages> kv_pipeline;
  RingPipeline<kNumTmemStages> tmem_pipeline;

  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = 224;
  static_assert(kNumMathRegisters * kNumMathWarpGroups + kNumSpecializedRegisters <= 512);

  cudaGridDependencySynchronize();

  if (warp_idx == kSpecWarpStart) {
    // Q producer: the block's next_n rows x 32 heads and their weights.
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      auto scheduler = make_scheduler();
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

        const uint32_t q_token_base = scheduler.get_q_tma_token_base();
        tma::copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
            &tensor_map_q, &smem.full_q_barriers[q_stage_idx], smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
        tma::copy<kNumWeightElementsPerRow, BLOCK_Q, 0>(
            &tensor_map_weights,
            &smem.full_q_barriers[q_stage_idx],
            smem.smem_weights[q_stage_idx],
            0,
            q_token_base);
        smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 1) {
    // KV producer: 4 pages (one 256-key split) and their scales per stage.
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    auto scheduler = make_scheduler();
    uint32_t cached_kv_page_base = 0;
    uint32_t cached_kv_page_coord = 0;
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
#pragma unroll 1
      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
        constexpr uint32_t kNumPagesPerSplit = Scheduler::kNumPagesPerSplit;
        static_assert(kNumPagesPerSplit <= 32, "Split spans more pages than a warp can cache");

        const uint32_t kv_page_base = (kv_base + kv_split_idx) * kNumPagesPerSplit;
        if (kv_page_base < cached_kv_page_base or kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
          cached_kv_page_base = (kv_page_base / 32) * 32;
          cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(cached_kv_page_base + lane_idx);
        }

        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        if (cute::elect_one_sync()) smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
        __syncwarp();

        int page_coords[kNumPagesPerSplit];
#pragma unroll
        for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++page_idx) {
          const auto src_lane = static_cast<int>(kv_page_base - cached_kv_page_base + page_idx);
          page_coords[page_idx] = __shfl_sync(0xffffffff, cached_kv_page_coord, src_lane);
        }

        if (cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++page_idx) {
            tma::copy<kHeadDim, kPageKV, 0, qk_dtype_t, true>(
                &tensor_map_kv,
                &smem.full_kv_barriers[kv_stage_idx],
                smem.smem_kv[kv_stage_idx] + page_idx * kPageKV * kNumQKBytesPerToken,
                0,
                0,
                1,
                page_coords[page_idx]);
            tma::copy<kPageKV, 1, 0>(
                &tensor_map_sf_kv,
                &smem.full_kv_barriers[kv_stage_idx],
                smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                0,
                page_coords[page_idx]);
          }
          smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
        }
        __syncwarp();
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    // UMMA issuer: per split, one M128 x N (next_n x 32) x K128 product per math warpgroup.
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

    auto scheduler = make_scheduler();
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);

      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

        if (cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

            smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
            ptx::tcgen05_after_thread_sync();

            auto instr_desc = cute::UMMA::make_instr_desc<
                cutlass::float_e4m3_t,
                cutlass::float_e4m3_t,
                float,
                UMMA_M,
                UMMA_N,
                cute::UMMA::Major::K,
                cute::UMMA::Major::K>();
            auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
#pragma unroll
            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
              auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                  smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
              auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                  smem.smem_q[q_stage_idx], 0, k * UMMA_K);
              ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc);
            }

            ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
          }
        }
        __syncwarp();
      }
      smem.empty_q_barriers[q_stage_idx].arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    if constexpr (kRowSplit) {
      // Math warpgroup g scores rows [g * kRowsPerGroup, ...) of both 128-key halves of every split, so each
      // thread keeps its rows' weights in registers; warp w reads keys 32 (w % 4) + lane of each half. Same
      // per-(key, row) arithmetic as below; only the assignment of (key, row) pairs to threads differs.
      constexpr uint32_t kRowsPerGroup = (BLOCK_Q + 1) / 2;
      const auto math_warpgroup_idx = warp_idx / 4;
      const uint32_t row_begin = math_warpgroup_idx * kRowsPerGroup;
      const uint32_t num_rows = math_warpgroup_idx == 0 ? kRowsPerGroup : BLOCK_Q - kRowsPerGroup;
      const uint32_t key_in_half = (warp_idx % 4) * 32 + lane_idx;

      float weights[kRowsPerGroup][kNumHeads];
      float accum[kNumHeads];

      auto scheduler = make_scheduler();
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.full_q_barriers[q_stage_idx].wait(q_phase);

#pragma unroll
        for (uint32_t r = 0; r < kRowsPerGroup; ++r) {
          if (r < num_rows) {
            const auto smem_weights_row = smem.smem_weights[q_stage_idx] + (row_begin + r) * kNumWeightElementsPerRow;
#pragma unroll
            for (uint32_t j = 0; j < kNumHeads; ++j) weights[r][j] = ptx::ld_shared(smem_weights_row + j);
          }
        }

        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
          // The KV stage holds the split's per-key scales; released once both halves' UMMAs committed.
          float scale_kv[kNumMathWarpGroups];
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
#pragma unroll
          for (uint32_t h = 0; h < kNumMathWarpGroups; ++h)
            scale_kv[h] = ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + h * UMMA_M + key_in_half);

#pragma unroll
          for (uint32_t h = 0; h < kNumMathWarpGroups; ++h) {
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
            ptx::tcgen05_after_thread_sync();
            if (h + 1 == kNumMathWarpGroups) {
              // Release KV smem only after UMMA commits TMEM; earlier release races TMA overwrite
              cutlass::arch::fence_view_async_shared();
              smem.empty_kv_barriers[kv_stage_idx].arrive();
            }
            const auto kv_offset = (kv_base + kv_split_idx) * kSplitKV + h * UMMA_M + key_in_half;

#pragma unroll
            for (uint32_t r = 0; r < kRowsPerGroup; ++r) {
              if (r < num_rows) {
                ptx::tmem_load_32dp32b<kNumHeads>(
                    tmem_stage_idx * UMMA_N + (row_begin + r) * kNumHeads, reinterpret_cast<uint32_t*>(accum));
                cutlass::arch::fence_view_async_tmem_load();
              }
              // one arrival per thread and stage, after its last TMEM read (a group without rows arrives at once)
              if (num_rows == 0 ? r == 0 : r + 1 == num_rows) {
                ptx::tcgen05_before_thread_sync();
                smem.empty_tmem_barriers[tmem_stage_idx].arrive();
              }
              if (r < num_rows) {
                float reduced;
                {
                  auto sum_0 = make_float2(0, 0);
                  auto sum_1 = make_float2(0, 0);
                  const auto transform = [&](const uint32_t& j, const float2& sum) {
                    auto a_0 = make_float2(accum[j], accum[j + 1]);
                    auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
                    auto b = make_float2(weights[r][j], weights[r][j + 1]);
                    return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
                  };

#pragma unroll
                  for (uint32_t j = 0; j < kNumHeads; j += 4) {
                    sum_0 = transform(j, sum_0);
                    sum_1 = transform(j + 2, sum_1);
                  }

                  auto sum = __fadd2_rn(sum_0, sum_1);
                  reduced = (sum.x + sum.y) / 2;
                }
                const auto q_offset = scheduler.get_logits_row(row_begin + r) * static_cast<uint64_t>(logits_stride);
                logits[q_offset + kv_offset] = reduced * scale_kv[h];
              }
            }
          }
        }

        cutlass::arch::fence_view_async_shared();
        smem.empty_q_barriers[q_stage_idx].arrive();
      }
    } else {
      // Math warpgroup g reads keys [128 g, 128 g + 128) of every split: one key per thread.
      const auto math_warpgroup_idx = warp_idx / 4;
      const auto math_thread_idx = warp_idx * 32 + lane_idx;
      tmem_pipeline.advance(math_warpgroup_idx);

      float weights[kWeightsInSmem ? 1 : BLOCK_Q][kNumHeads];
      float accum[kNumHeads];

      auto scheduler = make_scheduler();
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.full_q_barriers[q_stage_idx].wait(q_phase);

        if constexpr (not kWeightsInSmem) {
  #pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow;
  #pragma unroll
            for (uint32_t j = 0; j < kNumHeads; ++j)
              weights[i][j] = ptx::ld_shared(smem_weights_row + j);
          }
        }

        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
          auto kv_offset = scheduler.get_logits_col(kv_base, kv_split_idx, math_thread_idx);

          // FP8 consumes the KV pipeline directly to read per-KV scales
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
          const float scale_kv = ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + math_thread_idx);

          CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
          smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
          ptx::tcgen05_after_thread_sync();

          // Release KV smem only after UMMA commits TMEM; earlier release races TMA overwrite
          cutlass::arch::fence_view_async_shared();
          smem.empty_kv_barriers[kv_stage_idx].arrive();

  #pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
            ptx::tmem_load_32dp32b<kNumHeads / 2>(tmem_addr, reinterpret_cast<uint32_t*>(accum));
            cutlass::arch::fence_view_async_tmem_load();
            ptx::tmem_load_32dp32b<kNumHeads / 2>(
                tmem_addr + kNumHeads / 2, reinterpret_cast<uint32_t*>(accum + kNumHeads / 2));
            cutlass::arch::fence_view_async_tmem_load();

            if (i == BLOCK_Q - 1) {
              ptx::tcgen05_before_thread_sync();
              smem.empty_tmem_barriers[tmem_stage_idx].arrive();
            }

            float row_weights[kWeightsInSmem ? kNumHeads : 1];
            if constexpr (kWeightsInSmem) {
              const auto smem_weights_row =
                  reinterpret_cast<const float4*>(smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow);
  #pragma unroll
              for (uint32_t j = 0; j < kNumHeads / 4; ++j) {
                const float4 w4 = ptx::ld_shared(smem_weights_row + j);
                row_weights[j * 4 + 0] = w4.x;
                row_weights[j * 4 + 1] = w4.y;
                row_weights[j * 4 + 2] = w4.z;
                row_weights[j * 4 + 3] = w4.w;
              }
            }
            const auto weight = [&](const uint32_t& j) -> float {
              if constexpr (kWeightsInSmem)
                return row_weights[j];
              else
                return weights[i][j];
            };

            float reduced;
            {
              auto sum_0 = make_float2(0, 0);
              auto sum_1 = make_float2(0, 0);
              const auto transform = [&](const uint32_t& j, const float2& sum) {
                auto a_0 = make_float2(accum[j], accum[j + 1]);
                auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
                auto b = make_float2(weight(j), weight(j + 1));
                return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
              };

  #pragma unroll
              for (uint32_t j = 0; j < kNumHeads; j += 4) {
                sum_0 = transform(j, sum_0);
                sum_1 = transform(j + 2, sum_1);
              }

              auto sum = __fadd2_rn(sum_0, sum_1);
              reduced = (sum.x + sum.y) / 2;
            }
            auto result = static_cast<float>(reduced * scale_kv);
            const auto q_offset = scheduler.get_logits_row(i) * static_cast<uint64_t>(logits_stride);
            logits[q_offset + kv_offset] = result;
          }
        }

        cutlass::arch::fence_view_async_shared();
        smem.empty_q_barriers[q_stage_idx].arrive();
      }

    }

    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
#endif
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

// DeepGEMM's make_tma_{2,3}d_desc: no interleave, L2 promotion 256 B, no OOB fill.
inline CUtensorMap make_map(
    const void* base,
    CUtensorMapDataType dtype,
    uint32_t rank,
    const uint64_t* dims,
    const uint64_t* strides_bytes,
    const uint32_t* box,
    CUtensorMapSwizzle swizzle) {
  CUtensorMap map{};
  const uint32_t elem_strides[3] = {1, 1, 1};
  const CUresult rc = cuTensorMapEncodeTiled(
      &map,
      dtype,
      rank,
      const_cast<void*>(base),
      dims,
      strides_bytes,
      box,
      elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  sglang::host::RuntimeCheck(rc == CUDA_SUCCESS, "mqa1b: cuTensorMapEncodeTiled failed with ", static_cast<int>(rc));
  return map;
}

template <uint32_t kNextN, uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTmemStages, bool kWeightsInSmem, bool kRowSplit>
struct PagedMQALogits1Blk {
  static_assert(kNextN >= 1 and kNextN <= 8, "next_n must be 1..8");
  static constexpr size_t kSmem = sizeof(SharedStorage<kNextN, kNumQStages, kNumKVStages, kNumTmemStages>);
  static_assert(kSmem <= kMaxDynamicSmem, "shared memory exceeds 227 KB: fewer Q or KV stages");

  // q [B, next_n, 32, 128] e4m3; kv_cache [blocks, 64, 1, 132] uint8 (128 FP8 bytes per key, then
  // 64 fp32 scales per page); weights [B * next_n, 32] fp32; context_lens [B, next_n] int32;
  // block_table [B, >= pages] int32; schedule_meta [num_sms + 1, 2] int32 from DeepGEMM's
  // get_paged_mqa_logits_metadata; logits [B * next_n, >= 256 x splits] fp32, row stride % 256 == 0.
  static void
  run(tvm::ffi::TensorView q,
      tvm::ffi::TensorView kv_cache,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView context_lens,
      tvm::ffi::TensorView block_table,
      tvm::ffi::TensorView schedule_meta,
      tvm::ffi::TensorView logits) {
    using namespace sglang::host;
    auto B = SymbolicSize{"batch_size"};
    auto R = SymbolicSize{"rows"};
    auto NB = SymbolicSize{"num_kv_blocks"};
    auto S = SymbolicSize{"num_sms_plus_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, kNextN, kNumHeads, kHeadDim}).with_dtype<sglang::fp8_e4m3_t>().with_device(device).verify(q);
    TensorMatcher({NB, kPageKV, 1, kHeadDim + 4})
        .with_strides({-1, kHeadDim + 4, -1, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(kv_cache);
    TensorMatcher({R, kNumHeads}).with_strides({-1, 1}).with_dtype<float>().with_device(device).verify(weights);
    TensorMatcher({B, kNextN}).with_dtype<int32_t>().with_device(device).verify(context_lens);
    TensorMatcher({B, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device).verify(block_table);
    TensorMatcher({S, 2}).with_dtype<int32_t>().with_device(device).verify(schedule_meta);
    TensorMatcher({R, -1}).with_strides({-1, 1}).with_dtype<float>().with_device(device).verify(logits);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_rows = static_cast<uint32_t>(R.unwrap());
    RuntimeCheck(num_rows == batch_size * kNextN, "mqa1b: weights / logits must have B * next_n rows");
    const auto num_sms = static_cast<uint32_t>(S.unwrap() - 1);
    RuntimeCheck(num_sms > 0, "mqa1b: empty schedule metadata");
    const auto logits_stride = static_cast<uint32_t>(logits.stride(0));
    RuntimeCheck(logits_stride % kSplitKV == 0, "mqa1b: logits row stride must be a multiple of 256");
    if (batch_size == 0) return;

    const auto num_kv_blocks = static_cast<uint64_t>(NB.unwrap());
    const auto kv_stride_bytes = static_cast<uint64_t>(kv_cache.stride(0));
    const auto weights_stride_bytes = static_cast<uint64_t>(weights.stride(0)) * sizeof(float);
    auto* kv_base = static_cast<uint8_t*>(kv_cache.data_ptr());
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 and reinterpret_cast<uintptr_t>(kv_base) % 16 == 0 and
            reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0 and kv_stride_bytes % 16 == 0 and
            weights_stride_bytes % 16 == 0,
        "mqa1b: TMA needs 16-byte aligned bases and row strides");

    // q: [rows x 32, 128] bytes, box 128 x (next_n x 32), 128 B swizzle.
    const uint64_t q_dims[2] = {kHeadDim, uint64_t(num_rows) * kNumHeads};
    const uint64_t q_strides[1] = {kHeadDim};
    const uint32_t q_box[2] = {kHeadDim, kNextN * kNumHeads};
    const auto map_q =
        make_map(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, q_dims, q_strides, q_box, CU_TENSOR_MAP_SWIZZLE_128B);
    // kv: [blocks, 64, 128] bytes (row stride 128 B, page stride kv_stride_bytes), box one page.
    const uint64_t kv_dims[3] = {kHeadDim, kPageKV, num_kv_blocks};
    const uint64_t kv_strides[2] = {kHeadDim, kv_stride_bytes};
    const uint32_t kv_box[3] = {kHeadDim, kPageKV, 1};
    const auto map_kv =
        make_map(kv_base, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, kv_dims, kv_strides, kv_box, CU_TENSOR_MAP_SWIZZLE_128B);
    // sf_kv: the 64 fp32 scales after each page's FP8 bytes, box one page.
    const uint64_t sf_dims[2] = {kPageKV, num_kv_blocks};
    const uint64_t sf_strides[1] = {kv_stride_bytes};
    const uint32_t sf_box[2] = {kPageKV, 1};
    const auto map_sf_kv = make_map(
        kv_base + kPageKV * kHeadDim,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        sf_dims,
        sf_strides,
        sf_box,
        CU_TENSOR_MAP_SWIZZLE_NONE);
    // weights: [rows, 32] fp32, box 32 x next_n.
    const uint64_t w_dims[2] = {kNumHeads, num_rows};
    const uint64_t w_strides[1] = {weights_stride_bytes};
    const uint32_t w_box[2] = {kNumHeads, kNextN};
    const auto map_w = make_map(
        weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, w_dims, w_strides, w_box, CU_TENSOR_MAP_SWIZZLE_NONE);

    constexpr auto kernel = paged_mqa_logits_1blk_kernel<kNextN, kNumQStages, kNumKVStages, kNumTmemStages, kWeightsInSmem, kRowSplit>;
    const DLDevice dl_device = q.device();
    runtime::init_per_device(dl_device, [=] { runtime::set_max_dynamic_smem(kernel, kSmem); });
    LaunchKernel(num_sms, kNumSpecializedThreads + kNumMathThreads, dl_device, kSmem)
        .enable_pdl(false)(
            kernel,
            batch_size * kNextN,
            logits_stride,
            static_cast<uint32_t>(block_table.stride(0)),
            static_cast<const uint32_t*>(context_lens.data_ptr()),
            static_cast<float*>(logits.data_ptr()),
            static_cast<const uint32_t*>(block_table.data_ptr()),
            static_cast<const uint32_t*>(schedule_meta.data_ptr()),
            map_q,
            map_kv,
            map_sf_kv,
            map_w);
  }
};

}  // namespace sgl_mqa1b
