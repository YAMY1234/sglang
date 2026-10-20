// Cute-DSL grouped-GEMM routing tables for the experts of one EP rank, from the top-k ids an A2A dispatch
// received (global expert ids; padding rows carry an out-of-range id), in one single-CTA launch.
// Same table layout as flashinfer.fused_moe.cute_dsl moe_sort: tiles in ascending local-expert order,
// mn-limit = tile start + rows of the expert in that tile, -1 for non-local entries; like moe_sort, the unused
// rows of a tile are left unwritten (the GEMMs mask them by mn-limit). Rows inside an expert are ranked by
// shared-memory atomics, as FlashInfer's routing does.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <cub/block/block_scan.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

namespace qwen_ep_tables {

inline constexpr uint32_t kThreads = 1024;
inline constexpr uint32_t kMaxLocalExperts = 256;
// Entries per thread: 6 covers DEP4 decode (448 rows x top-10); 11 covers DEP16 decode at 10 requests per rank
// (1120 rows x top-10).
inline constexpr uint32_t kSmallPerThread = 6;
inline constexpr uint32_t kMaxPerThread = 11;

struct Params {
  const int32_t* topk_ids;               // [R * top_k], global expert ids
  int32_t* tile_idx_to_expert_idx;       // [max tiles]
  int32_t* tile_idx_to_mn_limit;         // [max tiles]
  int32_t* expanded_idx_to_permuted_idx; // [R * top_k]
  int32_t* permuted_idx_to_expanded_idx; // [max permuted]
  int32_t* total_num_padded_tokens;      // [1]
  int32_t* num_non_exiting_tiles;        // [1]
  uint32_t expanded;                     // R * top_k
  int32_t local_offset;
  uint32_t num_local;
  uint32_t tile_log2;
};

template <bool kUsePDL, uint32_t kPerThread>
__global__ __launch_bounds__(kThreads, 1) void ep_tables_kernel(const __grid_constant__ Params p) {
  using Scan = cub::BlockScan<int32_t, kThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ int32_t s_count[kMaxLocalExperts];
  __shared__ int32_t s_offset[kMaxLocalExperts];

  if (threadIdx.x < kMaxLocalExperts) s_count[threadIdx.x] = 0;
  device::PDLWaitPrimary<kUsePDL>();
  __syncthreads();

  int32_t local[kPerThread], rank[kPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kPerThread; ++j) {
    const uint32_t i = threadIdx.x + j * kThreads;
    local[j] = -1;
    rank[j] = 0;
    if (i < p.expanded) {
      const int32_t e = p.topk_ids[i] - p.local_offset;
      if (e >= 0 && e < static_cast<int32_t>(p.num_local)) {
        local[j] = e;
        rank[j] = atomicAdd(&s_count[e], 1);
      }
    }
  }
  __syncthreads();

  // Thread e owns local expert e: tile count, tile-padded offset, tile tables and the unused rows of its tiles.
  const uint32_t e = threadIdx.x;
  const int32_t count = e < p.num_local ? s_count[e] : 0;
  const int32_t tile = 1 << p.tile_log2;
  const int32_t num_tiles = (count + tile - 1) >> p.tile_log2;
  int32_t tile_offset, total_tiles;
  Scan(scan_storage).ExclusiveSum(num_tiles, tile_offset, total_tiles);
  for (int32_t c = 0; c < num_tiles; ++c) {
    const int32_t end = (tile_offset + c + 1) << p.tile_log2;
    const int32_t limit = min(end, (tile_offset << p.tile_log2) + count);
    p.tile_idx_to_expert_idx[tile_offset + c] = static_cast<int32_t>(e);
    p.tile_idx_to_mn_limit[tile_offset + c] = limit;
  }
  if (e < kMaxLocalExperts) s_offset[e] = tile_offset << p.tile_log2;
  if (e == 0) {
    p.num_non_exiting_tiles[0] = total_tiles;
    p.total_num_padded_tokens[0] = total_tiles << p.tile_log2;
  }
  __syncthreads();

#pragma unroll
  for (uint32_t j = 0; j < kPerThread; ++j) {
    const uint32_t i = threadIdx.x + j * kThreads;
    if (i < p.expanded) {
      const int32_t permuted = local[j] >= 0 ? s_offset[local[j]] + rank[j] : -1;
      p.expanded_idx_to_permuted_idx[i] = permuted;
      if (permuted >= 0) p.permuted_idx_to_expanded_idx[permuted] = static_cast<int32_t>(i);
    }
  }
  // The FC1 launch reads every table written above.
  __syncthreads();
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct EpTables {
  static void
  run(const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView tile_idx_to_expert_idx,
      const tvm::ffi::TensorView tile_idx_to_mn_limit,
      const tvm::ffi::TensorView expanded_idx_to_permuted_idx,
      const tvm::ffi::TensorView permuted_idx_to_expanded_idx,
      const tvm::ffi::TensorView total_num_padded_tokens,
      const tvm::ffi::TensorView num_non_exiting_tiles,
      int64_t local_offset,
      int64_t num_local,
      int64_t tile_log2) {
    using namespace host;
    auto R_ = SymbolicSize{"rows"};
    auto K_ = SymbolicSize{"top_k"};
    auto T_ = SymbolicSize{"max_tiles"};
    auto P_ = SymbolicSize{"max_permuted"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, K_}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({T_}).with_dtype<int32_t>().with_device(device).verify(tile_idx_to_expert_idx);
    TensorMatcher({T_}).with_dtype<int32_t>().with_device(device).verify(tile_idx_to_mn_limit);
    TensorMatcher({R_, K_}).with_dtype<int32_t>().with_device(device).verify(expanded_idx_to_permuted_idx);
    TensorMatcher({P_}).with_dtype<int32_t>().with_device(device).verify(permuted_idx_to_expanded_idx);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(total_num_padded_tokens);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_non_exiting_tiles);
    const auto expanded = static_cast<uint64_t>(R_.unwrap()) * static_cast<uint64_t>(K_.unwrap());
    CHECK_HOST(expanded <= kThreads * kMaxPerThread) << "ep_tables: " << expanded << " expanded entries";
    CHECK_HOST(num_local > 0 && num_local <= kMaxLocalExperts) << "ep_tables: num_local " << num_local;
    CHECK_HOST(tile_log2 >= 3 && tile_log2 <= 10) << "ep_tables: tile_log2 " << tile_log2;
    if (expanded == 0) return;
    // Every local expert with rows owns ceil(rows / tile) tiles.
    const int64_t filled = std::min<int64_t>(num_local, static_cast<int64_t>(expanded));
    const int64_t max_tiles = filled + (static_cast<int64_t>(expanded) - filled) / (int64_t{1} << tile_log2);
    CHECK_HOST(T_.unwrap() >= max_tiles) << "ep_tables: tile tables too small";
    CHECK_HOST(P_.unwrap() >= (max_tiles << tile_log2)) << "ep_tables: permuted map too small";
    Params params{};
    params.topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr());
    params.tile_idx_to_expert_idx = static_cast<int32_t*>(tile_idx_to_expert_idx.data_ptr());
    params.tile_idx_to_mn_limit = static_cast<int32_t*>(tile_idx_to_mn_limit.data_ptr());
    params.expanded_idx_to_permuted_idx = static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
    params.permuted_idx_to_expanded_idx = static_cast<int32_t*>(permuted_idx_to_expanded_idx.data_ptr());
    params.total_num_padded_tokens = static_cast<int32_t*>(total_num_padded_tokens.data_ptr());
    params.num_non_exiting_tiles = static_cast<int32_t*>(num_non_exiting_tiles.data_ptr());
    params.expanded = static_cast<uint32_t>(expanded);
    params.local_offset = static_cast<int32_t>(local_offset);
    params.num_local = static_cast<uint32_t>(num_local);
    params.tile_log2 = static_cast<uint32_t>(tile_log2);
    if (expanded <= kThreads * kSmallPerThread) {
      LaunchKernel(1, kThreads, device.unwrap()).enable_pdl(kUsePDL)(ep_tables_kernel<kUsePDL, kSmallPerThread>, params);
    } else {
      LaunchKernel(1, kThreads, device.unwrap()).enable_pdl(kUsePDL)(ep_tables_kernel<kUsePDL, kMaxPerThread>, params);
    }
  }
};

}  // namespace qwen_ep_tables

using qwen_ep_tables::EpTables;

}  // namespace sglang
