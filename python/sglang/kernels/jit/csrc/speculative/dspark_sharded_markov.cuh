// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Developed at NVIDIA; reformatted for SGLang. The host side is
// python/sglang/kernels/ops/speculative/dspark/sharded_markov.py.
//
// DSpark's sequential Markov sampling for up to 64 request rows, with the Markov
// head sharded by vocabulary across the tensor-parallel ranks. Template arguments:
//   kSteps   number of draft positions (1..8)
//   BaseT    element type of the base logits (bf16_t or fp32_t)
//   CacheT   element type of the draft-logits cache (bf16_t or fp32_t)
//
// For request r, step i needs r's previous token p (its anchor for i = 0):
//     logits_ri = cache_round(base_ri + bf16(markov_head . markov_embed[p]))
//     d_r,i+1   = argmax(logits_ri)                     (temperature 0)
//               = argmax(logits_ri / T_r + gumbel_ri)   (otherwise)
// Each rank owns rows [rank * V/world, (rank + 1) * V/world) of the head. Each
// warp keeps one 16-row tile of that shard in registers for all steps, so the
// head is read once per launch; the m16n8k16 MMA's eight columns carry eight
// request rows' embeddings, so every row's bias comes from the same MMAs. A step
// is those MMAs, a CTA reduction and an exchange of 64-bit candidate keys per
// request row through a mailbox in every rank's symmetric-memory peer buffer.
// The exchange is the grid barrier and the cross-rank argmax at once, so the
// grid must be co-resident (the host launches it cooperatively).
//
// Two exchanges. Rank-local fold: the CTAs of a rank fold their best key into a
// rank-local maximum, the last CTA to arrive stores that key, tagged with the
// launch epoch, into every rank's mailbox, and every CTA waits for the `world`
// records of each row in its own mailbox. Direct: CTA c of rank s stores its
// best key of row r into slot [parity][step][s][c][r] of every rank's mailbox,
// and every CTA waits for the world * ctas slots of each row. Every wait is
// bounded by `timeout_ns`; past it the kernel reports and traps.
//
// The base logits may hold the full vocabulary or only this rank's shard (the
// draft LM head's output before its all-gather): `base_column` is the column of
// this rank's first vocabulary entry in them, rank * V/world or 0.
//
// Rows whose request-state index is negative (CUDA-graph padding) are inert:
// their tokens are drawn greedily and their logits are not cached. A row at
// temperature 0 caches no logits either (no accept reads them); a launch where
// no row keeps logits skips the system fences and the epilogue.
//
// The draft-logits cache must hold the full vocabulary on every rank. Each rank
// writes its shard into its own cache and into a staging buffer of every peer;
// after the last exchange each rank copies the peers' shards from its staging
// buffer into its cache. Staging and mailbox records alternate between two
// parities so that a launch never overwrites what a peer may still be reading
// from the previous launch. Each CTA counts its own launches (every launch runs
// every CTA once), so the epoch needs no end-of-launch atomics. Each exchange and
// the wide kernel keep their own counters, records and staging logits, so the
// host may alternate them between launches on one peer buffer.

#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cstdio>

namespace sglang {

namespace dspark_markov {

using u32 = unsigned int;
using u64 = unsigned long long;
using i32 = int;
using i64 = long long;

constexpr int kRankDim = 256;
constexpr int kChunks = kRankDim / 32;  // 8
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kMaxSteps = 8;
constexpr int kMaxRanks = 8;
constexpr int kMaxRows = 8;  // MMA columns
// Peer buffer layout (bytes): fold records u64 [parity 2][step 8][row 8][source
// rank 8] (8 KiB), then the staging logits CacheT [exchange 2][parity
// 2][max_rows][kSteps][vocab] at kStagingOffset; the direct-exchange mailbox and
// the wide kernel's records and staging follow at host-given offsets.
constexpr int kRecordWords = 2 * kMaxSteps * kMaxRows * kMaxRanks;
constexpr i64 kStagingOffset = 16384;
// Local control words (uint32 view): combine maxima u64 [step 8][row 8] at words
// 0..127, arrivals u32 [step 8] at 128..135, per-CTA launch counters of the fold
// at 256 + cta and of the direct exchange at 512 + cta.
constexpr int kArrivalWord = 128;
constexpr int kFoldCountWord = 256;
constexpr int kDirectCountWord = 512;

// Wide kernel: 9 to 64 request rows, its own control words: combine maxima u64
// [step 8][row 64] at words 0..1023, arrivals [step 8] at 1024..1031, per-CTA
// launch counters at 1280 + cta. Records u64 [parity 2][step 8][row 64][source
// rank 8] at the mailbox offset; the staging logits [parity 2][max_rows][kSteps][vocab]
// follow.
constexpr int kWideRows = 64;
constexpr int kWideArrivalWord = 1024;
constexpr int kWideCountWord = 1280;
constexpr i64 kWideRecordBytes = 2LL * kMaxSteps * kWideRows * kMaxRanks * 8;

constexpr u32 kTagBits = 15;
constexpr u32 kTagMask = (1u << kTagBits) - 1;
constexpr u32 kIndexBits = 17;
constexpr u32 kIndexMask = (1u << kIndexBits) - 1;

static_assert(kRecordWords * 8 <= kStagingOffset, "mailbox overlaps staging");
static_assert(kThreads == kMaxSteps * kWideRows, "the wide kernel clears row_best with one store per thread");

// A 16-row tile of logits is this many 16-byte vectors of this many elements.
template <typename CacheT>
constexpr int kVectorElements = 16 / static_cast<int>(sizeof(CacheT));
template <typename CacheT>
constexpr int kTileVectors = 16 / kVectorElements<CacheT>;

// Pull one embedding row (512 bytes) into L2.
__device__ __forceinline__ void PrefetchRowL2(const bf16_t* row) {
#pragma unroll
  for (int line = 0; line < kRankDim * 2 / 128; ++line) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(row) + line * 128));
  }
}

__device__ __forceinline__ float ToFloat(float v) {
  return v;
}
__device__ __forceinline__ float ToFloat(bf16_t v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ void FromFloat(float v, float& out) {
  out = v;
}
__device__ __forceinline__ void FromFloat(float v, bf16_t& out) {
  out = __float2bfloat16_rn(v);
}

// Triton lowers an fp32 `/` to div.full.f32; the stock sampler divides that way.
__device__ __forceinline__ float DivFull(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// fp32 add without contraction into a neighbouring multiply.
__device__ __forceinline__ float AddRn(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ u32 OrderedBits(float value) {
  const u32 bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// [48:17] order-preserving score bits, [16:0] inverted index (ties prefer the
// smaller index, as torch.argmax does). Larger is better; 0 means "no key".
__device__ __forceinline__ u64 CandidateKey(float score, i32 index) {
  return (static_cast<u64>(OrderedBits(score)) << kIndexBits) | static_cast<u64>(kIndexMask - static_cast<u32>(index));
}

__device__ __forceinline__ i32 KeyIndex(u64 key) {
  return static_cast<i32>(kIndexMask - static_cast<u32>(key & kIndexMask));
}

// Never zero; distinct for the two launches that share a parity slot.
__device__ __forceinline__ u32 EpochTag(u32 count) {
  return 1u + count % 32767u;
}

__device__ __forceinline__ u64 WarpMaxU64(u64 value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const u64 other = __shfl_xor_sync(0xffffffffu, value, offset);
    value = other > value ? other : value;
  }
  return value;
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
__device__ __noinline__ void ReportTimeout(int step, i64 timeout_ns, int source, u32 tag, u32 seen) {
  printf(
      "DSpark sharded Markov chain: step %d waited %lld ns for rank %d (tag %u, saw %u); "
      "a peer rank stopped or ran a different launch sequence\n",
      step,
      static_cast<long long>(timeout_ns),
      source,
      tag,
      seen);
  __trap();
}

// Spin until the record at `address` carries `tag`; return its key. A wait past
// `timeout_ns` reports and traps instead of hanging.
__device__ __forceinline__ u64 AwaitRecord(const u64* address, u32 tag, i64 timeout_ns, int step, int source) {
  u64 value = LoadRecord(address);
  if ((value & kTagMask) == tag) return value >> kTagBits;
  const u64 start = GlobalTimer();
  u32 spins = 0;
  while ((value & kTagMask) != tag) {
    value = LoadRecord(address);
    if ((++spins & 1023u) == 0 && GlobalTimer() - start > static_cast<u64>(timeout_ns)) {
      ReportTimeout(step, timeout_ns, source, tag, static_cast<u32>(value & kTagMask));
    }
  }
  return value >> kTagBits;
}

// Re-read the `pending` rows of one mailbox block until each carries `tag`; all
// pending loads of a pass are in flight together.
__device__ __forceinline__ void
AwaitBlock(const u64* slot, u64 (&got)[kMaxRows], u32 pending, u32 tag, i64 timeout_ns, int step, int source) {
  const u64 start = GlobalTimer();
  u32 spins = 0;
  while (pending != 0u) {
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
      if ((pending >> r) & 1u) got[r] = LoadRecord(slot + r);
    }
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
      if (((pending >> r) & 1u) && (got[r] & kTagMask) == tag) pending &= ~(1u << r);
    }
    if ((++spins & 1023u) == 0 && GlobalTimer() - start > static_cast<u64>(timeout_ns)) {
      u32 seen = 0u;
#pragma unroll
      for (int r = kMaxRows - 1; r >= 0; --r) {
        if ((pending >> r) & 1u) seen = static_cast<u32>(got[r] & kTagMask);
      }
      ReportTimeout(step, timeout_ns, source, tag, seen);
    }
  }
}

__device__ __forceinline__ char* PeerBase(const u64* peers, int rank) {
  return reinterpret_cast<char*>(static_cast<unsigned long long>(peers[rank]));
}

// Whether request-row step `i` keeps its logits (a sampled row of the draft).
__device__ __forceinline__ bool KeepsLogits(const i32* idx_mapping, const float* temperature, int i) {
  const int req_state = idx_mapping[i];
  return req_state >= 0 && temperature[req_state] != 0.0F;
}

// mma.m16n8k16 sums over k, so any bijection between fragment columns and
// physical k positions gives the same dot products when A and B share it. Lane
// (g = lane / 4, t = lane % 4) loads 16 contiguous bytes of weight rows g and
// g + 8 at k = base + 8 t and the matching 8 embedding values; values 0-3 feed
// the first MMA and 4-7 the second.
__device__ __forceinline__ void Mma16816(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void MmaChunk(float (&c)[4], const uint4& lo, const uint4& hi, const uint4& x) {
  Mma16816(c, lo.x, hi.x, lo.y, hi.y, x.x, x.y);
  Mma16816(c, lo.z, hi.z, lo.w, hi.w, x.z, x.w);
}

__device__ __forceinline__ void
LoadTile(uint4 (&lo)[kChunks], uint4 (&hi)[kChunks], const bf16_t* head, int tile, int lane) {
  const bf16_t* row_lo = head + static_cast<i64>(tile * 16 + (lane >> 2)) * kRankDim + (lane & 3) * 8;
  const bf16_t* row_hi = row_lo + 8 * kRankDim;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    lo[chunk] = *reinterpret_cast<const uint4*>(row_lo + chunk * 32);
    hi[chunk] = *reinterpret_cast<const uint4*>(row_hi + chunk * 32);
  }
}

// Inputs written by earlier graph nodes are deliberately not const __restrict__,
// so their loads stay coherent across replays.
template <int kSteps, typename BaseT, typename CacheT, bool kDirect>
__global__ void __launch_bounds__(kThreads, 1) sharded_markov_kernel(
    const BaseT* base_logits,                // [rows, kSteps, vocab or vocab_local]
    i64 base_row_stride,                     // elements
    i64 base_step_stride,                    // elements
    i64 base_column,                         // column of this rank's shard
    const bf16_t* __restrict__ embed_table,  // [vocab, 256]
    const bf16_t* __restrict__ head,         // this rank's [vocab_local, 256]
    const i32* input_ids,                    // row r's anchor at [r * query_per_row]
    int query_per_row,                       //
    const i32* idx_mapping,                  // [rows * kSteps] request-state rows
    const float* temperature,                // [max_num_reqs]
    const float* noise,                      // [rows * kSteps, vocab_local]
    CacheT* logits_cache,                    // [max_num_reqs, steps, vocab]
    i64 cache_stride0,                       //
    i64 cache_stride1,                       //
    i64* draft_tokens,                       // [rows, kSteps]
    i64 draft_stride,                        // row stride
    const u64* __restrict__ peers,           // [world] peer buffer bases
    u32* control,                            // local control words
    int rank,                                //
    int world,                               //
    int vocab_local,                         //
    int rows,                                //
    int max_rows,                            //
    i64 timeout_ns,                          //
    i64 mailbox_offset) {                    // bytes from a peer buffer's base to the direct mailbox
  static_assert(kSteps >= 1 && kSteps <= kMaxSteps, "unsupported kSteps");
  constexpr int kCountWord = kDirect ? kDirectCountWord : kFoldCountWord;
  constexpr int kVector = kVectorElements<CacheT>;
  constexpr int kTileVecs = kTileVectors<CacheT>;
  __shared__ __align__(16) bf16_t embedding[kMaxRows][kRankDim];
  __shared__ unsigned long long warp_best[kWarps][kMaxRows];
  __shared__ unsigned long long row_best[kMaxSteps][kMaxRows];  // direct exchange
  __shared__ unsigned long long winner[kMaxRows];               // rank-local fold
  __shared__ u32 s_count;
  __shared__ __align__(16) CacheT tile_logits[kWarps][kMaxRows][16];

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int tiles = vocab_local / 16;
  const int stride = static_cast<int>(gridDim.x) * kWarps;
  const int first_tile = static_cast<int>(blockIdx.x) * kWarps + warp;
  const int ctas = static_cast<int>(gridDim.x);
  const int vocab_offset = rank * vocab_local;
  const i64 vocab = static_cast<i64>(vocab_local) * world;
  // MMA column 2t + col of this lane's accumulators is request row 2t + col.
  const bool active[2] = {2 * t < rows, 2 * t + 1 < rows};

  uint4 lo[kChunks], hi[kChunks];
  if (first_tile < tiles) LoadTile(lo, hi, head, first_tile, lane);

  // Base logits and noise of this lane's resident tile rows and request rows for
  // the current step; the next step's are loaded while the exchange is in flight.
  const bool resident = first_tile < tiles;
  float base[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
  float gumbel[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
  auto load_inputs = [&](int step) {
    if (!resident) return;
#pragma unroll
    for (int col = 0; col < 2; ++col) {
      if (!active[col]) continue;
      const int r = 2 * t + col;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = first_tile * 16 + g + 8 * half;
        base[col][half] = ToFloat(base_logits[r * base_row_stride + step * base_step_stride + base_column + row]);
        gumbel[col][half] = noise[static_cast<i64>(r * kSteps + step) * vocab_local + row];
      }
    }
  };
  load_inputs(0);

  if (threadIdx.x == 0) {
    s_count = *reinterpret_cast<volatile u32*>(&control[kCountWord + blockIdx.x]);
  }
  if constexpr (kDirect) {
    if (threadIdx.x < kMaxSteps * kMaxRows) (&row_best[0][0])[threadIdx.x] = 0ull;
  }
  if (warp < kMaxRows) {
    uint4 value = make_uint4(0u, 0u, 0u, 0u);
    if (warp < rows) {
      const int anchor = input_ids[warp * query_per_row];
      value = reinterpret_cast<const uint4*>(embed_table + static_cast<i64>(anchor) * kRankDim)[lane];
    }
    reinterpret_cast<uint4*>(embedding[warp])[lane] = value;
  }
  // Uniform over the grid and the ranks (the inputs are replicated).
  bool keeps = false;
  for (int i = lane; i < rows * kSteps; i += 32)
    keeps |= KeepsLogits(idx_mapping, temperature, i);
  const bool launch_keeps = __any_sync(0xffffffffu, keeps);
  __syncthreads();
  const u32 tag = EpochTag(s_count);
  const int parity = static_cast<int>(s_count & 1u);
  u64* const records = reinterpret_cast<u64*>(PeerBase(peers, rank));
  unsigned long long* const combine = reinterpret_cast<unsigned long long*>(control);
  u32* const arrivals = control + kArrivalWord;
  // Staging logits [exchange 2][parity 2][max_rows][kSteps][vocab].
  const i64 staging_parity = kStagingOffset / static_cast<i64>(sizeof(CacheT)) +
                             static_cast<i64>(2 * kDirect + parity) * max_rows * kSteps * vocab;

#pragma unroll 1
  for (int step = 0; step < kSteps; ++step) {
    const bool last_step = step + 1 == kSteps;
    float temp[2];
#pragma unroll
    for (int col = 0; col < 2; ++col) {
      const int req_state = active[col] ? idx_mapping[(2 * t + col) * kSteps + step] : -1;
      temp[col] = req_state >= 0 ? temperature[req_state] : 0.0F;
    }
    // Bit r: request row r keeps this step's logits.
    const u32 keep_rows =
        __ballot_sync(0xffffffffu, lane < rows && KeepsLogits(idx_mapping, temperature, lane * kSteps + step));
    u64 best[2] = {0ull, 0ull};
    for (int tile = first_tile; tile < tiles; tile += stride) {
      float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
      if (tile == first_tile) {
#pragma unroll
        for (int chunk = 0; chunk < kChunks; ++chunk) {
          const uint4 e = reinterpret_cast<const uint4*>(embedding[g])[chunk * 4 + t];
          MmaChunk(c, lo[chunk], hi[chunk], e);
        }
      } else {
        // Tiles beyond one per warp (small grids) stream from L2.
        const bf16_t* row_lo = head + static_cast<i64>(tile * 16 + g) * kRankDim + t * 8;
        for (int chunk = 0; chunk < kChunks; ++chunk) {
          const uint4 e = reinterpret_cast<const uint4*>(embedding[g])[chunk * 4 + t];
          const uint4 a = *reinterpret_cast<const uint4*>(row_lo + chunk * 32);
          const uint4 b = *reinterpret_cast<const uint4*>(row_lo + 8 * kRankDim + chunk * 32);
          MmaChunk(c, a, b, e);
        }
      }
#pragma unroll
      for (int col = 0; col < 2; ++col) {
        if (!active[col]) continue;
        const int r = 2 * t + col;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int row = tile * 16 + g + 8 * half;
          const int index = vocab_offset + row;
          const float base_value =
              tile == first_tile
                  ? base[col][half]
                  : ToFloat(base_logits[r * base_row_stride + step * base_step_stride + base_column + row]);
          const float noise_value =
              tile == first_tile ? gumbel[col][half] : noise[static_cast<i64>(r * kSteps + step) * vocab_local + row];
          // The bias GEMM's BF16 output, then the eager add's fp32 sum rounded
          // once to the cache dtype.
          const float bias = __bfloat162float(__float2bfloat16_rn(c[2 * half + col]));
          CacheT logit;
          FromFloat(AddRn(base_value, bias), logit);
          tile_logits[warp][r][g + 8 * half] = logit;
          const float value = ToFloat(logit);
          const float score = temp[col] != 0.0F ? AddRn(DivFull(value, temp[col]), noise_value) : value;
          const u64 key = CandidateKey(score, index);
          best[col] = key > best[col] ? key : best[col];
        }
      }
      // The tile's logits leave as 16-byte vectors: one per lane and request row
      // for the local cache and for each peer's staging buffer.
      if (keep_rows != 0u) {
        __syncwarp();
        const int r = lane / kTileVecs;
        if (lane < rows * kTileVecs && ((keep_rows >> r) & 1u)) {
          const int vector = lane - r * kTileVecs;
          const int req_state = idx_mapping[r * kSteps + step];
          const uint4 v = reinterpret_cast<const uint4*>(tile_logits[warp][r])[vector];
          const i64 element = vocab_offset + tile * 16 + vector * kVector;
          *reinterpret_cast<uint4*>(logits_cache + req_state * cache_stride0 + step * cache_stride1 + element) = v;
          const i64 staged = staging_parity + static_cast<i64>(r * kSteps + step) * vocab + element;
          for (int peer = 0; peer < world; ++peer) {
            if (peer == rank) continue;
            *reinterpret_cast<uint4*>(reinterpret_cast<CacheT*>(PeerBase(peers, peer)) + staged) = v;
          }
        }
        __syncwarp();
      }
    }
    if (step + 1 < kSteps) load_inputs(step + 1);

    // Each request row's best key over the warp (lanes sharing t), then the CTA.
#pragma unroll
    for (int col = 0; col < 2; ++col) {
#pragma unroll
      for (int offset = 4; offset < 32; offset <<= 1) {
        const u64 other = __shfl_xor_sync(0xffffffffu, best[col], offset);
        best[col] = other > best[col] ? other : best[col];
      }
    }
    if (g == 0) {
      warp_best[warp][2 * t] = best[0];
      warp_best[warp][2 * t + 1] = best[1];
    }
    __syncthreads();
    // Only the last step's records publish logits stored to peers.
    const bool publish = last_step && launch_keeps;
    if constexpr (kDirect) {
      // Mailbox blocks [parity][step][source rank][cta], each max_rows records.
      const i64 step_blocks = (static_cast<i64>(parity) * kSteps + step) * world * ctas;
      if (warp == 0) {
        u64 key = 0ull;
        if (lane < rows) {
#pragma unroll
          for (int w = 0; w < kWarps; ++w)
            key = warp_best[w][lane] > key ? warp_best[w][lane] : key;
        }
        // Lane L posts row L % max_rows to rank L / max_rows. Rows past `rows` carry the
        // zero key so that every slot is rewritten on every launch of its parity.
        const int dest = lane / max_rows;
        const int r = lane - dest * max_rows;
        const u64 mine = __shfl_sync(0xffffffffu, key, r);
        if (lane < world * max_rows) {
          if (publish) __threadfence_system();
          u64* const box = reinterpret_cast<u64*>(PeerBase(peers, dest) + mailbox_offset);
          StoreRecord(
              box + (step_blocks + rank * ctas + static_cast<int>(blockIdx.x)) * max_rows + r,
              (mine << kTagBits) | tag);
        }
      }
      {
        const u64* const box =
            reinterpret_cast<const u64*>(PeerBase(peers, rank) + mailbox_offset) + step_blocks * max_rows;
        const int blocks = world * ctas;
        for (int first = 0; first < blocks; first += kThreads) {
          const int block = first + static_cast<int>(threadIdx.x);
          const bool polls = block < blocks;
          const u64* const slot = box + static_cast<i64>(block) * max_rows;
          u64 got[kMaxRows];
          u32 pending = 0u;
#pragma unroll
          for (int r = 0; r < kMaxRows; ++r)
            got[r] = polls && r < rows ? LoadRecord(slot + r) : 0ull;
#pragma unroll
          for (int r = 0; r < kMaxRows; ++r) {
            if (polls && r < rows && (got[r] & kTagMask) != tag) pending |= 1u << r;
          }
          if (pending != 0u) AwaitBlock(slot, got, pending, tag, timeout_ns, step, block / ctas);
          if (publish && polls) __threadfence_system();
#pragma unroll
          for (int r = 0; r < kMaxRows; ++r) {
            if (r >= rows) break;
            const u64 key = WarpMaxU64(got[r] >> kTagBits);
            if (lane == 0) {
              atomicMax(&row_best[step][r], static_cast<unsigned long long>(key));
              // The CTA's candidate may be the winner: start its embedding row's HBM read.
              if (!last_step && key != 0ull) {
                PrefetchRowL2(embed_table + static_cast<i64>(KeyIndex(key)) * kRankDim);
              }
            }
          }
        }
      }
      __syncthreads();
      if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < rows) {
        draft_tokens[threadIdx.x * draft_stride + step] = KeyIndex(row_best[step][threadIdx.x]);
      }
      if (step + 1 < kSteps) {
        if (warp < rows) {
          reinterpret_cast<uint4*>(embedding[warp])[lane] = reinterpret_cast<const uint4*>(
              embed_table + static_cast<i64>(KeyIndex(row_best[step][warp])) * kRankDim)[lane];
        }
        __syncthreads();
      }
    } else {
      if (warp == 0) {
        if (lane < rows) {
          u64 key = 0ull;
#pragma unroll
          for (int w = 0; w < kWarps; ++w)
            key = warp_best[w][lane] > key ? warp_best[w][lane] : key;
          atomicMax(combine + step * kMaxRows + lane, static_cast<unsigned long long>(key));
        }
        __syncwarp();
        int last = 0;
        if (lane == 0) {
          // The last step's fence also publishes this CTA's peer stores of every
          // step before its arrival is counted.
          if (publish) {
            __threadfence_system();
          } else {
            __threadfence();
          }
          last = atomicAdd(arrivals + step, 1u) == static_cast<u32>(ctas - 1);
        }
        last = __shfl_sync(0xffffffffu, last, 0);
        // Lane L handles request row L / world and rank L % world.
        const int r = lane / world;
        const int other_rank = lane - r * world;
        const bool exchanging = lane < rows * world;
        // Post every row slot up to max_rows (rows past `rows` carry the zero key)
        // so each slot is rewritten on every launch of its parity. Graphs of
        // different row counts share the launch counter; a slot left from a
        // larger launch could otherwise carry a tag that matches a later one.
        const bool posting = lane < max_rows * world;
        const i64 record = ((static_cast<i64>(parity) * kMaxSteps + step) * kMaxRows + r) * kMaxRanks;
        if (last) {
          if (lane == 0) __threadfence();
          __syncwarp();
          u64 rank_key = 0ull;
          if (lane < rows) rank_key = atomicExch(combine + step * kMaxRows + lane, 0ull);
          if (lane == 0) atomicExch(arrivals + step, 0u);
          const u64 mine = __shfl_sync(0xffffffffu, rank_key, r);
          if (posting) {
            // Only the last step's record must publish (system scope) the logits
            // this rank pushed to its peers; a system fence on the other steps
            // would wait for those stores to drain.
            if (publish) __threadfence_system();
            StoreRecord(reinterpret_cast<u64*>(PeerBase(peers, other_rank)) + record + rank, (mine << kTagBits) | tag);
          }
        }
        u64 global = 0ull;
        if (exchanging) {
          global = AwaitRecord(records + record + other_rank, tag, timeout_ns, step, other_rank);
          if (publish) {
            __threadfence_system();
          } else if (!last_step && global != 0ull) {
            // Each rank's candidate may be the winner: start its embedding row's
            // HBM read while the other ranks' records are still in flight.
            PrefetchRowL2(embed_table + static_cast<i64>(KeyIndex(global)) * kRankDim);
          }
        }
        for (int offset = 1; offset < world; offset <<= 1) {
          const u64 other = __shfl_xor_sync(0xffffffffu, global, offset);
          global = other > global ? other : global;
        }
        if (exchanging && other_rank == 0) winner[r] = global;
      }
      __syncthreads();
      if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < rows) {
        draft_tokens[threadIdx.x * draft_stride + step] = KeyIndex(winner[threadIdx.x]);
      }
      if (step + 1 < kSteps && warp < rows) {
        reinterpret_cast<uint4*>(embedding[warp])[lane] =
            reinterpret_cast<const uint4*>(embed_table + static_cast<i64>(KeyIndex(winner[warp])) * kRankDim)[lane];
      }
      __syncthreads();
    }
  }

  // The peers' shards are in this rank's staging buffer: copy them into the cache.
  if (launch_keeps) {
    const int vectors = vocab_local * static_cast<int>(sizeof(CacheT)) / 16;
    const int per_row_step = (world - 1) * vectors;
    const int total = rows * kSteps * per_row_step;
    for (int task = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); task < total;
         task += static_cast<int>(gridDim.x * blockDim.x)) {
      const int row_step = task / per_row_step;
      const int within = task - row_step * per_row_step;
      int peer = within / vectors;
      const int vector = within - peer * vectors;
      peer += peer >= rank ? 1 : 0;
      if (!KeepsLogits(idx_mapping, temperature, row_step)) continue;
      const int req_state = idx_mapping[row_step];
      const int step = row_step % kSteps;
      const i64 element = static_cast<i64>(peer) * vocab_local + static_cast<i64>(vector) * kVector;
      const CacheT* staged = reinterpret_cast<const CacheT*>(PeerBase(peers, rank)) + staging_parity +
                             static_cast<i64>(row_step) * vocab + element;
      const uint4 v = __ldcg(reinterpret_cast<const uint4*>(staged));
      *reinterpret_cast<uint4*>(logits_cache + req_state * cache_stride0 + step * cache_stride1 + element) = v;
    }
  }
  // This thread read the counter at launch start; the next launch is stream-ordered.
  if (threadIdx.x == 0) control[kCountWord + blockIdx.x] = s_count + 1u;
}

// 9 to 64 request rows, rank-local fold only. Each warp keeps its one 16-row W2
// tile in registers and runs the MMAs once per group of 8 request rows; a group's
// best keys go into the CTA maximum by shared-memory atomicMax. The host launches
// it only when every warp owns at most one tile.
template <int kSteps, typename BaseT, typename CacheT>
__global__ void __launch_bounds__(kThreads, 1) sharded_markov_wide_kernel(
    const BaseT* base_logits,
    i64 base_row_stride,
    i64 base_step_stride,
    i64 base_column,
    const bf16_t* __restrict__ embed_table,
    const bf16_t* __restrict__ head,
    const i32* input_ids,
    int query_per_row,
    const i32* idx_mapping,
    const float* temperature,
    const float* noise,
    CacheT* logits_cache,
    i64 cache_stride0,
    i64 cache_stride1,
    i64* draft_tokens,
    i64 draft_stride,
    const u64* __restrict__ peers,
    u32* control,
    int rank,
    int world,
    int vocab_local,
    int rows,
    int max_rows,
    i64 timeout_ns,
    i64 mailbox_offset) {  // bytes from a peer buffer's base to this kernel's records
  static_assert(kSteps >= 1 && kSteps <= kMaxSteps, "unsupported kSteps");
  constexpr int kVector = kVectorElements<CacheT>;
  constexpr int kTileVecs = kTileVectors<CacheT>;
  __shared__ __align__(16) bf16_t embedding[kWideRows][kRankDim];
  __shared__ unsigned long long row_best[kMaxSteps][kWideRows];
  __shared__ unsigned long long rank_key[kWideRows];
  __shared__ unsigned long long winner[kWideRows];
  __shared__ u32 s_count;
  __shared__ __align__(16) CacheT tile_logits[kWarps][kMaxRows][16];

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int tiles = vocab_local / 16;
  const int tile = static_cast<int>(blockIdx.x) * kWarps + warp;
  const bool resident = tile < tiles;
  const int ctas = static_cast<int>(gridDim.x);
  const int vocab_offset = rank * vocab_local;
  const i64 vocab = static_cast<i64>(vocab_local) * world;
  const int groups = (rows + kMaxRows - 1) / kMaxRows;

  uint4 lo[kChunks], hi[kChunks];
  if (resident) LoadTile(lo, hi, head, tile, lane);

  // This lane's tile rows of request rows 8 group + 2 t + col at one step.
  auto load_group = [&](int step, int group, float (&b)[2][2], float (&n)[2][2]) {
    if (!resident) return;
#pragma unroll
    for (int col = 0; col < 2; ++col) {
      const int r = group * kMaxRows + 2 * t + col;
      if (r >= rows) continue;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = tile * 16 + g + 8 * half;
        b[col][half] = ToFloat(base_logits[r * base_row_stride + step * base_step_stride + base_column + row]);
        n[col][half] = noise[static_cast<i64>(r * kSteps + step) * vocab_local + row];
      }
    }
  };
  float base[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
  float gumbel[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
  load_group(0, 0, base, gumbel);

  if (threadIdx.x == 0) {
    s_count = *reinterpret_cast<volatile u32*>(&control[kWideCountWord + blockIdx.x]);
  }
  (&row_best[0][0])[threadIdx.x] = 0ull;
  for (int r = warp; r < kWideRows; r += kWarps) {
    uint4 value = make_uint4(0u, 0u, 0u, 0u);
    if (r < rows) {
      const int anchor = input_ids[r * query_per_row];
      value = reinterpret_cast<const uint4*>(embed_table + static_cast<i64>(anchor) * kRankDim)[lane];
    }
    reinterpret_cast<uint4*>(embedding[r])[lane] = value;
  }
  bool keeps = false;
  for (int i = lane; i < rows * kSteps; i += 32)
    keeps |= KeepsLogits(idx_mapping, temperature, i);
  const bool launch_keeps = __any_sync(0xffffffffu, keeps);
  __syncthreads();
  const u32 tag = EpochTag(s_count);
  const int parity = static_cast<int>(s_count & 1u);
  u64* const records = reinterpret_cast<u64*>(PeerBase(peers, rank) + mailbox_offset);
  unsigned long long* const combine = reinterpret_cast<unsigned long long*>(control);
  u32* const arrivals = control + kWideArrivalWord;
  const i64 staging_parity = (mailbox_offset + kWideRecordBytes) / static_cast<i64>(sizeof(CacheT)) +
                             static_cast<i64>(parity) * max_rows * kSteps * vocab;

#pragma unroll 1
  for (int step = 0; step < kSteps; ++step) {
    const bool last_step = step + 1 == kSteps;
    // Bit r of keep_lo / keep_hi: request row r / 32 + r keeps this step's logits.
    const u32 keep_lo =
        __ballot_sync(0xffffffffu, lane < rows && KeepsLogits(idx_mapping, temperature, lane * kSteps + step));
    const u32 keep_hi = __ballot_sync(
        0xffffffffu, lane + 32 < rows && KeepsLogits(idx_mapping, temperature, (lane + 32) * kSteps + step));
#pragma unroll 1
    for (int group = 0; group < groups; ++group) {
      // The next group's inputs (or the next step's first) load under this group's MMAs.
      float next_base[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
      float next_gumbel[2][2] = {{0.0F, 0.0F}, {0.0F, 0.0F}};
      if (group + 1 < groups) {
        load_group(step, group + 1, next_base, next_gumbel);
      } else if (step + 1 < kSteps) {
        load_group(step + 1, 0, next_base, next_gumbel);
      }
      u64 best[2] = {0ull, 0ull};
      if (resident) {
        float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
        for (int chunk = 0; chunk < kChunks; ++chunk) {
          const uint4 e = reinterpret_cast<const uint4*>(embedding[group * kMaxRows + g])[chunk * 4 + t];
          MmaChunk(c, lo[chunk], hi[chunk], e);
        }
#pragma unroll
        for (int col = 0; col < 2; ++col) {
          const int r = group * kMaxRows + 2 * t + col;
          if (r >= rows) continue;
          const int req_state = idx_mapping[r * kSteps + step];
          const float temp = req_state >= 0 ? temperature[req_state] : 0.0F;
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int index = vocab_offset + tile * 16 + g + 8 * half;
            const float bias = __bfloat162float(__float2bfloat16_rn(c[2 * half + col]));
            CacheT logit;
            FromFloat(AddRn(base[col][half], bias), logit);
            tile_logits[warp][2 * t + col][g + 8 * half] = logit;
            const float value = ToFloat(logit);
            const float score = temp != 0.0F ? AddRn(DivFull(value, temp), gumbel[col][half]) : value;
            const u64 key = CandidateKey(score, index);
            best[col] = key > best[col] ? key : best[col];
          }
        }
        const u32 keep_group = ((group < 4 ? keep_lo : keep_hi) >> ((group & 3) * kMaxRows)) & 0xFFu;
        if (keep_group != 0u) {
          __syncwarp();
          const int r8 = lane / kTileVecs;
          if (lane < kMaxRows * kTileVecs && ((keep_group >> r8) & 1u)) {
            const int r = group * kMaxRows + r8;
            const int vector = lane - r8 * kTileVecs;
            const int req_state = idx_mapping[r * kSteps + step];
            const uint4 v = reinterpret_cast<const uint4*>(tile_logits[warp][r8])[vector];
            const i64 element = vocab_offset + tile * 16 + vector * kVector;
            *reinterpret_cast<uint4*>(logits_cache + req_state * cache_stride0 + step * cache_stride1 + element) = v;
            const i64 staged = staging_parity + static_cast<i64>(r * kSteps + step) * vocab + element;
            for (int peer = 0; peer < world; ++peer) {
              if (peer == rank) continue;
              *reinterpret_cast<uint4*>(reinterpret_cast<CacheT*>(PeerBase(peers, peer)) + staged) = v;
            }
          }
          __syncwarp();
        }
      }
      // The group's best keys over the warp (lanes sharing t), then into the CTA maximum.
#pragma unroll
      for (int col = 0; col < 2; ++col) {
#pragma unroll
        for (int offset = 4; offset < 32; offset <<= 1) {
          const u64 other = __shfl_xor_sync(0xffffffffu, best[col], offset);
          best[col] = other > best[col] ? other : best[col];
        }
        const int r = group * kMaxRows + 2 * t + col;
        if (g == 0 && r < rows && best[col] != 0ull) {
          atomicMax(&row_best[step][r], static_cast<unsigned long long>(best[col]));
        }
      }
#pragma unroll
      for (int col = 0; col < 2; ++col) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          base[col][half] = next_base[col][half];
          gumbel[col][half] = next_gumbel[col][half];
        }
      }
    }
    __syncthreads();

    // Only the last step's records publish logits stored to peers.
    const bool publish = last_step && launch_keeps;
    const i64 step_records = (static_cast<i64>(parity) * kMaxSteps + step) * kWideRows * kMaxRanks;
    if (warp == 0) {
      for (int r = lane; r < rows; r += 32) {
        atomicMax(combine + step * kWideRows + r, row_best[step][r]);
      }
      __syncwarp();
      int last = 0;
      if (lane == 0) {
        // The last step's fence also publishes this CTA's peer stores of every step.
        if (publish) {
          __threadfence_system();
        } else {
          __threadfence();
        }
        last = atomicAdd(arrivals + step, 1u) == static_cast<u32>(ctas - 1);
      }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        if (lane == 0) __threadfence();
        __syncwarp();
        for (int r = lane; r < max_rows; r += 32) {
          rank_key[r] = r < rows ? atomicExch(combine + step * kWideRows + r, 0ull) : 0ull;
        }
        if (lane == 0) atomicExch(arrivals + step, 0u);
        __syncwarp();
        if (publish) __threadfence_system();
        // Every row slot up to max_rows, so each slot is rewritten on every launch of
        // its parity (graphs of different row counts share the launch counters).
        for (int i = lane; i < max_rows * world; i += 32) {
          const int r = i / world;
          const int dest = i - r * world;
          StoreRecord(
              reinterpret_cast<u64*>(PeerBase(peers, dest) + mailbox_offset) + step_records + r * kMaxRanks + rank,
              (rank_key[r] << kTagBits) | tag);
        }
      }
    } else {
      // Warps 1.. wait for (row, source rank) records i = 32 (warp - 1) + lane, ...
      const int pairs = rows * world;
      for (int first = (warp - 1) * 32; first < pairs; first += (kWarps - 1) * 32) {
        const int i = first + lane;
        const int r = i / world;
        const int source = i - r * world;
        u64 global = 0ull;
        if (i < pairs) {
          global = AwaitRecord(records + step_records + r * kMaxRanks + source, tag, timeout_ns, step, source);
          if (publish) {
            __threadfence_system();
          } else if (!last_step && global != 0ull) {
            PrefetchRowL2(embed_table + static_cast<i64>(KeyIndex(global)) * kRankDim);
          }
        }
        // world divides 32, so a row's records sit in aligned lanes of one warp.
        for (int offset = 1; offset < world; offset <<= 1) {
          const u64 other = __shfl_xor_sync(0xffffffffu, global, offset);
          global = other > global ? other : global;
        }
        if (i < pairs && source == 0) winner[r] = global;
      }
    }
    __syncthreads();
    if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < rows) {
      draft_tokens[threadIdx.x * draft_stride + step] = KeyIndex(winner[threadIdx.x]);
    }
    if (step + 1 < kSteps) {
      for (int r = warp; r < rows; r += kWarps) {
        reinterpret_cast<uint4*>(embedding[r])[lane] =
            reinterpret_cast<const uint4*>(embed_table + static_cast<i64>(KeyIndex(winner[r])) * kRankDim)[lane];
      }
      __syncthreads();
    }
  }

  if (launch_keeps) {
    const int vectors = vocab_local * static_cast<int>(sizeof(CacheT)) / 16;
    const int per_row_step = (world - 1) * vectors;
    const int total = rows * kSteps * per_row_step;
    for (int task = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); task < total;
         task += static_cast<int>(gridDim.x * blockDim.x)) {
      const int row_step = task / per_row_step;
      const int within = task - row_step * per_row_step;
      int peer = within / vectors;
      const int vector = within - peer * vectors;
      peer += peer >= rank ? 1 : 0;
      if (!KeepsLogits(idx_mapping, temperature, row_step)) continue;
      const int req_state = idx_mapping[row_step];
      const int step = row_step % kSteps;
      const i64 element = static_cast<i64>(peer) * vocab_local + static_cast<i64>(vector) * kVector;
      const CacheT* staged = reinterpret_cast<const CacheT*>(PeerBase(peers, rank)) + staging_parity +
                             static_cast<i64>(row_step) * vocab + element;
      const uint4 v = __ldcg(reinterpret_cast<const uint4*>(staged));
      *reinterpret_cast<uint4*>(logits_cache + req_state * cache_stride0 + step * cache_stride1 + element) = v;
    }
  }
  // This thread read the counter at launch start; the next launch is stream-ordered.
  if (threadIdx.x == 0) control[kWideCountWord + blockIdx.x] = s_count + 1u;
}

}  // namespace dspark_markov

/// \brief Host entries of one sharded Markov chain kernel.
/// \tparam kSteps Draft positions per launch (1..8).
/// \tparam BaseT Base-logits element type (bf16_t or fp32_t).
/// \tparam CacheT Draft-logits cache element type (bf16_t or fp32_t).
/// \tparam kDirect Direct exchange (otherwise the rank-local fold); narrow kernel only.
/// \tparam kWide The 9-64-row kernel instead of the narrow one (up to 8 rows).
template <int kSteps, typename BaseT, typename CacheT, bool kDirect, bool kWide>
struct DSparkShardedMarkov {
  static_assert(!(kDirect && kWide), "the wide kernel has the rank-local fold only");

  static constexpr auto kernel() {
    if constexpr (kWide) {
      return dspark_markov::sharded_markov_wide_kernel<kSteps, BaseT, CacheT>;
    } else {
      return dspark_markov::sharded_markov_kernel<kSteps, BaseT, CacheT, kDirect>;
    }
  }

  /// \brief CTAs of this kernel that can be co-resident on the current device.
  static int64_t max_ctas() {
    int device = 0;
    int sms = 0;
    int per_sm = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel(), dspark_markov::kThreads, 0));
    return static_cast<int64_t>(per_sm) * sms;
  }

  /// \brief Cooperative launch of `ctas` CTAs on the current stream; the host
  /// (sharded_markov.py) owns the shape, layout and peer-buffer contract.
  static void
  run(tvm::ffi::TensorView base_logits,
      tvm::ffi::TensorView embed_table,
      tvm::ffi::TensorView head,
      tvm::ffi::TensorView anchor,
      tvm::ffi::TensorView idx_mapping,
      tvm::ffi::TensorView temperature,
      tvm::ffi::TensorView noise,
      tvm::ffi::TensorView logits_cache,
      int64_t cache_stride0,
      int64_t cache_stride1,
      tvm::ffi::TensorView draft_tokens,
      tvm::ffi::TensorView peers,
      tvm::ffi::TensorView control,
      int64_t base_column,
      int64_t rank,
      int64_t world,
      int64_t vocab_local,
      int64_t rows,
      int64_t max_rows,
      int64_t timeout_ns,
      int64_t mailbox_offset,
      int64_t ctas) {
    using namespace dspark_markov;
    host::RuntimeCheck(timeout_ns > 0, "every cross-rank wait needs a positive deadline");
    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeCooperative;
    attribute.val.cooperative = 1;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(static_cast<uint32_t>(ctas));
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = 0;
    config.stream = host::LaunchKernel::resolve_device(base_logits.device());
    config.attrs = &attribute;
    config.numAttrs = 1;
    host::RuntimeDeviceCheck(
        ::cudaLaunchKernelEx(
            &config,
            kernel(),
            static_cast<const BaseT*>(base_logits.data_ptr()),
            static_cast<i64>(base_logits.stride(0)),
            static_cast<i64>(base_logits.stride(1)),
            static_cast<i64>(base_column),
            static_cast<const bf16_t*>(embed_table.data_ptr()),
            static_cast<const bf16_t*>(head.data_ptr()),
            static_cast<const i32*>(anchor.data_ptr()),
            1,
            static_cast<const i32*>(idx_mapping.data_ptr()),
            static_cast<const float*>(temperature.data_ptr()),
            static_cast<const float*>(noise.data_ptr()),
            static_cast<CacheT*>(logits_cache.data_ptr()),
            static_cast<i64>(cache_stride0),
            static_cast<i64>(cache_stride1),
            static_cast<i64*>(draft_tokens.data_ptr()),
            static_cast<i64>(draft_tokens.stride(0)),
            static_cast<const u64*>(peers.data_ptr()),
            static_cast<u32*>(control.data_ptr()),
            static_cast<int>(rank),
            static_cast<int>(world),
            static_cast<int>(vocab_local),
            static_cast<int>(rows),
            static_cast<int>(max_rows),
            static_cast<i64>(timeout_ns),
            static_cast<i64>(mailbox_offset)));
  }
};

}  // namespace sglang
