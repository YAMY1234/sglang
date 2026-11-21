#pragma once

#include <sgl_kernel/runtime.cuh>

#include "hicache.cuh"
#include <dlpack/dlpack.h>
#include <algorithm>
#include <cstdint>

namespace sglang {

constexpr int kBlockSize = 1024;
constexpr int kBlockQuotaBackup = 2;
constexpr int kBlockQuotaLoad = 2;

constexpr int kBytesPerThreadPerStep = 16;
constexpr int kBytesPerBlockPerStep = kBlockSize * kBytesPerThreadPerStep;

struct MambaTransferParams {
  const char* __restrict__ src_base;
  char* __restrict__ dst_base;

  const uintptr_t* __restrict__ layer_ptrs;
  const int64_t* __restrict__ src_indices;
  const int64_t* __restrict__ dst_indices;
  int64_t item_size;
  int64_t src_layout_dim;
  int64_t dst_layout_dim;
  int64_t layer_id;
  int64_t num_items;
  int64_t num_layers;
};

__global__
__launch_bounds__(kBlockSize, 1) void transfer_mamba_load_kernel(const __grid_constant__ MambaTransferParams params) {
  const int tid = threadIdx.x;

  for (int64_t item_id = static_cast<int64_t>(blockIdx.x); item_id < params.num_items; item_id += gridDim.x) {
    const int64_t src_page = params.src_indices[item_id];
    const int64_t dst_page = params.dst_indices[item_id];

    const char* src = params.src_base + src_page * params.src_layout_dim + params.layer_id * params.item_size;

    char* dst = params.dst_base + dst_page * params.item_size;

    const int64_t base = static_cast<int64_t>(tid) * kBytesPerThreadPerStep;
    if (base < params.item_size) {
      uint4 v_cur = device::details::load_nc(reinterpret_cast<const uint4*>(src + base));
      int64_t off;
      for (off = base + kBytesPerBlockPerStep; off < params.item_size; off += kBytesPerBlockPerStep) {
        uint4 v_next = device::details::load_nc(reinterpret_cast<const uint4*>(src + off));
        device::details::store_nc(reinterpret_cast<uint4*>(dst + off - kBytesPerBlockPerStep), v_cur);
        v_cur = v_next;
      }
      device::details::store_nc(reinterpret_cast<uint4*>(dst + off - kBytesPerBlockPerStep), v_cur);
    }
  }
}

__global__
__launch_bounds__(kBlockSize, 1) void transfer_mamba_backup_kernel(const __grid_constant__ MambaTransferParams params) {
  const int tid = threadIdx.x;
  const int64_t total_work = params.num_items * params.num_layers;

  for (int64_t work_id = static_cast<int64_t>(blockIdx.x); work_id < total_work; work_id += gridDim.x) {
    const int64_t layer_id = work_id % params.num_layers;
    const int64_t item_id = work_id / params.num_layers;

    const int64_t src_page = params.src_indices[item_id];
    const int64_t dst_page = params.dst_indices[item_id];

    const char* src = reinterpret_cast<const char*>(params.layer_ptrs[layer_id]) + src_page * params.item_size;

    char* dst = params.dst_base + dst_page * params.dst_layout_dim + layer_id * params.item_size;

    const int64_t base = static_cast<int64_t>(tid) * kBytesPerThreadPerStep;
    if (base < params.item_size) {
      uint4 v_cur = device::details::load_nc(reinterpret_cast<const uint4*>(src + base));
      int64_t off;
      for (off = base + kBytesPerBlockPerStep; off < params.item_size; off += kBytesPerBlockPerStep) {
        uint4 v_next = device::details::load_nc(reinterpret_cast<const uint4*>(src + off));
        device::details::store_nc(reinterpret_cast<uint4*>(dst + off - kBytesPerBlockPerStep), v_cur);
        v_cur = v_next;
      }
      device::details::store_nc(reinterpret_cast<uint4*>(dst + off - kBytesPerBlockPerStep), v_cur);
    }
  }
}

// Chunked variant with the grid sized to the transfer: a block copies one chunk
// of one item (one layer's state of one slot) at a time, so a few large states
// (a KDA SSM state is ~0.8 MB per layer) spread over many SMs instead of two.
// Each thread keeps kChunkUnroll 16-byte loads in flight: host reads over the
// C2C link are latency-bound, so bytes in flight set the bandwidth.
constexpr int kChunkThreads = 256;
constexpr int kChunkUnroll = 4;
constexpr int64_t kChunkBytes = 32 * 1024;

struct MambaChunkedParams {
  MambaTransferParams base;
  int64_t chunks_per_item;
  int64_t num_units;
};

template <bool kBackup>
__global__ __launch_bounds__(kChunkThreads) void transfer_mamba_chunked_kernel(
    const __grid_constant__ MambaChunkedParams params) {
  const MambaTransferParams& p = params.base;
  constexpr int64_t kStep = int64_t{kChunkThreads} * kChunkUnroll * kBytesPerThreadPerStep;

  for (int64_t unit = blockIdx.x; unit < params.num_units; unit += gridDim.x) {
    const int64_t chunk = unit % params.chunks_per_item;
    const int64_t work = unit / params.chunks_per_item;
    const char* src;
    char* dst;
    if constexpr (kBackup) {
      // layer_first device -> page_first host; work = item * num_layers + layer
      const int64_t layer_id = work % p.num_layers;
      const int64_t item_id = work / p.num_layers;
      src = reinterpret_cast<const char*>(p.layer_ptrs[layer_id]) + p.src_indices[item_id] * p.item_size;
      dst = p.dst_base + p.dst_indices[item_id] * p.dst_layout_dim + layer_id * p.item_size;
    } else {
      // page_first host -> layer_first device, one layer; work = item
      src = p.src_base + p.src_indices[work] * p.src_layout_dim + p.layer_id * p.item_size;
      dst = p.dst_base + p.dst_indices[work] * p.item_size;
    }
    const int64_t begin = chunk * kChunkBytes;
    const int64_t end = begin + kChunkBytes < p.item_size ? begin + kChunkBytes : p.item_size;
    for (int64_t off = begin + int64_t{threadIdx.x} * kBytesPerThreadPerStep; off < end; off += kStep) {
      uint4 v[kChunkUnroll];
#pragma unroll
      for (int k = 0; k < kChunkUnroll; ++k) {
        const int64_t o = off + int64_t{k} * kChunkThreads * kBytesPerThreadPerStep;
        if (o < end) v[k] = device::details::load_nc(reinterpret_cast<const uint4*>(src + o));
      }
#pragma unroll
      for (int k = 0; k < kChunkUnroll; ++k) {
        const int64_t o = off + int64_t{k} * kChunkThreads * kBytesPerThreadPerStep;
        if (o < end) device::details::store_nc(reinterpret_cast<uint4*>(dst + o), v[k]);
      }
    }
  }
}

struct TransferMambaKernel {
  // Load: page_first -> layer_first (single layer at a time)
  static void run_pf_lf(
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t layer_id,
      const int64_t item_size,
      const int64_t src_layout_dim) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);

    RuntimeCheck(item_size > 0, "transfer_mamba: item_size must be positive");
    RuntimeCheck(item_size % 16 == 0, "transfer_mamba: item_size must be 16-byte aligned (uint4)");
    const auto num_items = L.unwrap();
    if (num_items == 0) return;

    const auto device = device_.unwrap();
    const int grid_x = static_cast<int>(std::min(static_cast<int64_t>(kBlockQuotaLoad), num_items));
    dim3 grid(grid_x);

    const auto params = MambaTransferParams{
        .src_base = static_cast<const char*>(runtime::get_device_accessible_ptr(src)),
        .dst_base = static_cast<char*>(runtime::get_device_accessible_ptr(dst)),
        .layer_ptrs = nullptr,
        .src_indices = static_cast<const int64_t*>(src_indices.data_ptr()),
        .dst_indices = static_cast<const int64_t*>(dst_indices.data_ptr()),
        .item_size = item_size,
        .src_layout_dim = src_layout_dim,
        .dst_layout_dim = 0,
        .layer_id = layer_id,
        .num_items = num_items,
        .num_layers = 1,
    };

    LaunchKernel(grid, kBlockSize, device)(transfer_mamba_load_kernel, params);
  }

  // Backup: layer_first -> page_first (all layers at once)
  static void run_lf_pf(
      const tvm::ffi::TensorView src_ptrs,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t item_size,
      const int64_t dst_layout_dim,
      const int64_t num_layers) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);
    // src_ptrs is a 1D tensor of device pointers (uint64) on CUDA
    TensorMatcher({static_cast<int64_t>(num_layers)})  //
        .with_dtype<uint64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_ptrs);

    RuntimeCheck(item_size > 0, "transfer_mamba: item_size must be positive");
    RuntimeCheck(item_size % 16 == 0, "transfer_mamba: item_size must be 16-byte aligned (uint4)");
    RuntimeCheck(num_layers > 0, "transfer_mamba: num_layers must be positive");
    const auto num_items = L.unwrap();
    if (num_items == 0) return;

    const auto device = device_.unwrap();
    const int64_t total_work = num_items * num_layers;
    const int grid_x = static_cast<int>(std::min(static_cast<int64_t>(kBlockQuotaBackup), total_work));
    dim3 grid(grid_x);

    const auto params = MambaTransferParams{
        .src_base = nullptr,
        .dst_base = static_cast<char*>(runtime::get_device_accessible_ptr(dst)),
        .layer_ptrs = static_cast<const uintptr_t*>(src_ptrs.data_ptr()),
        .src_indices = static_cast<const int64_t*>(src_indices.data_ptr()),
        .dst_indices = static_cast<const int64_t*>(dst_indices.data_ptr()),
        .item_size = item_size,
        .src_layout_dim = 0,
        .dst_layout_dim = dst_layout_dim,
        .layer_id = 0,
        .num_items = num_items,
        .num_layers = num_layers,
    };

    LaunchKernel(grid, kBlockSize, device)(transfer_mamba_backup_kernel, params);
  }

  static void launch_chunked(
      const MambaTransferParams& base, int64_t num_work, int64_t block_quota, bool backup, DLDevice device) {
    using namespace host;
    RuntimeCheck(block_quota > 0, "transfer_mamba: block_quota must be positive");
    const int64_t chunks_per_item = (base.item_size + kChunkBytes - 1) / kChunkBytes;
    const auto params = MambaChunkedParams{
        .base = base,
        .chunks_per_item = chunks_per_item,
        .num_units = num_work * chunks_per_item,
    };
    const dim3 grid(static_cast<unsigned>(std::min(block_quota, params.num_units)));
    if (backup) {
      LaunchKernel(grid, kChunkThreads, device)(transfer_mamba_chunked_kernel<true>, params);
    } else {
      LaunchKernel(grid, kChunkThreads, device)(transfer_mamba_chunked_kernel<false>, params);
    }
  }

  // run_pf_lf with the grid capped at block_quota instead of kBlockQuotaLoad.
  static void run_pf_lf_chunked(
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t layer_id,
      const int64_t item_size,
      const int64_t src_layout_dim,
      const int64_t block_quota) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);

    RuntimeCheck(item_size > 0, "transfer_mamba: item_size must be positive");
    RuntimeCheck(item_size % 16 == 0, "transfer_mamba: item_size must be 16-byte aligned (uint4)");
    const auto num_items = L.unwrap();
    if (num_items == 0) return;

    const auto base = MambaTransferParams{
        .src_base = static_cast<const char*>(runtime::get_device_accessible_ptr(src)),
        .dst_base = static_cast<char*>(runtime::get_device_accessible_ptr(dst)),
        .layer_ptrs = nullptr,
        .src_indices = static_cast<const int64_t*>(src_indices.data_ptr()),
        .dst_indices = static_cast<const int64_t*>(dst_indices.data_ptr()),
        .item_size = item_size,
        .src_layout_dim = src_layout_dim,
        .dst_layout_dim = 0,
        .layer_id = layer_id,
        .num_items = num_items,
        .num_layers = 1,
    };
    launch_chunked(base, num_items, block_quota, /*backup=*/false, device_.unwrap());
  }

  // run_lf_pf with the grid capped at block_quota instead of kBlockQuotaBackup.
  static void run_lf_pf_chunked(
      const tvm::ffi::TensorView src_ptrs,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t item_size,
      const int64_t dst_layout_dim,
      const int64_t num_layers,
      const int64_t block_quota) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);
    TensorMatcher({static_cast<int64_t>(num_layers)})  //
        .with_dtype<uint64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_ptrs);

    RuntimeCheck(item_size > 0, "transfer_mamba: item_size must be positive");
    RuntimeCheck(item_size % 16 == 0, "transfer_mamba: item_size must be 16-byte aligned (uint4)");
    RuntimeCheck(num_layers > 0, "transfer_mamba: num_layers must be positive");
    const auto num_items = L.unwrap();
    if (num_items == 0) return;

    const auto base = MambaTransferParams{
        .src_base = nullptr,
        .dst_base = static_cast<char*>(runtime::get_device_accessible_ptr(dst)),
        .layer_ptrs = static_cast<const uintptr_t*>(src_ptrs.data_ptr()),
        .src_indices = static_cast<const int64_t*>(src_indices.data_ptr()),
        .dst_indices = static_cast<const int64_t*>(dst_indices.data_ptr()),
        .item_size = item_size,
        .src_layout_dim = 0,
        .dst_layout_dim = dst_layout_dim,
        .layer_id = 0,
        .num_items = num_items,
        .num_layers = num_layers,
    };
    launch_chunked(base, num_items * num_layers, block_quota, /*backup=*/true, device_.unwrap());
  }
};

}  // namespace sglang
