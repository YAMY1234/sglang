// Fence-free MoE all-to-all for attention-DP + EP decode. dispatch: NCCL-LL lines (8 B payload + the call epoch
// twice per 16-B store) into fixed [source][token] slots; combine: BF16 rows over a 0xFFFF Lamport sentinel (pushers
// map 0xFFFF to 0x7FFF, which no fp32 sum or BF16 output can tell apart), summed in FlashInfer's TOP_K = 10 order.
// Every call writes a header to every target, so no rank starts call c + 2 before all peers finished call c.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {

namespace qwen_dep_ll_a2a {

inline constexpr uint32_t kThreads = 256;
inline constexpr uint32_t kMaxRanks = 8;
inline constexpr uint32_t kMaxTokens = 128;
inline constexpr uint32_t kDispatchStages = 2;
inline constexpr uint32_t kCombineStages = 3;
inline constexpr uint32_t kSpinMax = 1u << 26;

/// \brief Per-call geometry of the NVFP4 dispatch line format and the BF16 combine rows.
template <uint32_t kRanks, uint32_t kTopK, uint32_t kHidden>
struct Geometry {
  static_assert(kRanks >= 2 && kRanks <= kMaxRanks);
  static_assert(kTopK == 10, "the combine reproduces FlashInfer's TOP_K = 10 reduction order");
  static_assert(kHidden % 128 == 0);
  static constexpr uint32_t kQBytes = kHidden / 2;
  static constexpr uint32_t kSBytes = kHidden / 16;
  static constexpr uint32_t kIdLines = kTopK / 2;
  static constexpr uint32_t kWtLines = kTopK / 2;
  static constexpr uint32_t kQLines = kQBytes / 8;
  static constexpr uint32_t kSLines = kSBytes / 8;
  static constexpr uint32_t kLines = kIdLines + kWtLines + kQLines + kSLines;
  static constexpr uint32_t kChunks = kHidden / 8;

  SGL_DEVICE_HOST static constexpr uint64_t header_offset(uint32_t stage, uint32_t src) {
    return (static_cast<uint64_t>(stage) * kRanks + src) * 16;
  }
  SGL_DEVICE_HOST static constexpr uint64_t slot_offset(uint32_t stage, uint32_t src, uint32_t t, uint32_t slots) {
    return static_cast<uint64_t>(kDispatchStages) * kRanks * 16 +
           ((static_cast<uint64_t>(stage) * kRanks + src) * slots + t) * kLines * 16;
  }
  SGL_DEVICE_HOST static constexpr uint64_t dispatch_bytes(uint32_t slots) {
    return slot_offset(kDispatchStages, 0, 0, slots);
  }
  SGL_DEVICE_HOST static constexpr uint64_t combine_offset(uint32_t stage, uint32_t src, uint32_t t, uint32_t slots) {
    return ((static_cast<uint64_t>(stage) * kRanks + src) * slots + t) * kChunks * 16;
  }
  SGL_DEVICE_HOST static constexpr uint64_t combine_bytes(uint32_t slots) {
    return combine_offset(kCombineStages, 0, 0, slots);
  }
};

struct Peers {
  uint8_t* base[kMaxRanks];
};

struct DispatchParams {
  Peers peers;  // each rank's dispatch region
  const uint8_t* __restrict__ x;
  const uint8_t* __restrict__ sf;
  const int32_t* __restrict__ ids;
  const float* __restrict__ wts;
  int32_t* ctr;
  uint8_t* __restrict__ out_x;
  uint8_t* __restrict__ out_sf;
  int32_t* __restrict__ out_ids;
  float* __restrict__ out_wts;
  int32_t* __restrict__ recv_tokens;
  int32_t* err;
  uint32_t tokens;
  uint32_t tmax;
  uint32_t slots;
  uint32_t rank;
  uint32_t experts_per_rank;
  int32_t invalid_id;
};

struct CombineParams {
  Peers peers;  // each rank's combine region
  const bf16_t* __restrict__ partial;
  const int32_t* __restrict__ rids;
  const int32_t* __restrict__ recv_tokens;
  const int32_t* __restrict__ ids;
  bf16_t* __restrict__ out;
  int32_t* ctr;
  int32_t* err;
  uint32_t tokens;
  uint32_t tmax;
  uint32_t slots;
  uint32_t rank;
  uint32_t experts_per_rank;
  uint32_t num_experts;
};

SGL_DEVICE void st_line(void* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(x), "r"(y), "r"(z), "r"(w)
               : "memory");
}

SGL_DEVICE uint4 ld_line(const void* addr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

/// \brief Count one more poll; past kSpinMax record `code` and give up, so a broken peer cannot hang the GPU.
SGL_DEVICE bool give_up(uint32_t& spins, int32_t* err, int32_t code) {
  if (++spins > kSpinMax) {
    atomicExch(err, code);
    return true;
  }
  return false;
}

SGL_DEVICE uint32_t canonical_nan2(uint32_t x) {
  const uint32_t m = __vcmpeq2(x, 0xFFFFFFFFu);
  return (x & ~m) | (0x7FFF7FFFu & m);
}

SGL_DEVICE bool has_sentinel(const uint4& v) {
  return (__vcmpeq2(v.x, 0xFFFFFFFFu) | __vcmpeq2(v.y, 0xFFFFFFFFu) | __vcmpeq2(v.z, 0xFFFFFFFFu) |
          __vcmpeq2(v.w, 0xFFFFFFFFu)) != 0u;
}

template <uint32_t kRanks, uint32_t kTopK, uint32_t kHidden, bool kUsePDL>
__global__ __launch_bounds__(kThreads) void dispatch_kernel(const __grid_constant__ DispatchParams p) {
  using G = Geometry<kRanks, kTopK, kHidden>;
  __shared__ uint32_t s_mask[kMaxTokens];
  __shared__ uint32_t s_tokens[kRanks];
  const uint32_t b = blockIdx.x, grid = gridDim.x, tid = threadIdx.x;
  // The counter was written by an earlier call of this kernel; only the wait orders us after it.
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  const uint32_t call = static_cast<uint32_t>(p.ctr[b]), epoch = call + 1;
  const uint32_t stage = call & 1u;
  if (tid < p.tokens) {
    uint32_t m = 0;
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k)
      m |= 1u << (static_cast<uint32_t>(p.ids[tid * kTopK + k]) / p.experts_per_rank);
    s_mask[tid] = m;
  }
  if (b == 0 && tid < kRanks)
    st_line(p.peers.base[tid] + G::header_offset(stage, p.rank), p.tokens, epoch, p.tokens, epoch);
  __syncthreads();

  const uint32_t sends = p.tokens * kRanks * G::kLines;
  for (uint32_t task = b * kThreads + tid; task < sends; task += grid * kThreads) {
    const uint32_t i = task % G::kLines, j = task / G::kLines, r = j % kRanks, t = j / kRanks;
    uint2 d;
    if (i < G::kIdLines) {
      d = (s_mask[t] >> r & 1u) ? *reinterpret_cast<const uint2*>(p.ids + t * kTopK + 2 * i)
                                : make_uint2(static_cast<uint32_t>(p.invalid_id), static_cast<uint32_t>(p.invalid_id));
    } else if (i < G::kIdLines + G::kWtLines) {
      d = *reinterpret_cast<const uint2*>(p.wts + t * kTopK + 2 * (i - G::kIdLines));
    } else if (i < G::kIdLines + G::kWtLines + G::kQLines) {
      d = *reinterpret_cast<const uint2*>(
          p.x + static_cast<uint64_t>(t) * G::kQBytes + 8 * (i - G::kIdLines - G::kWtLines));
    } else {
      d = *reinterpret_cast<const uint2*>(
          p.sf + static_cast<uint64_t>(t) * G::kSBytes + 8 * (i - G::kIdLines - G::kWtLines - G::kQLines));
    }
    st_line(p.peers.base[r] + G::slot_offset(stage, p.rank, t, p.slots) + i * 16, d.x, epoch, d.y, epoch);
  }

  const uint8_t* mine = p.peers.base[p.rank];
  if (tid < kRanks) {
    const uint8_t* h = mine + G::header_offset(stage, tid);
    uint4 v;
    uint32_t spins = 0;
    do {
      v = ld_line(h);
    } while ((v.y != epoch || v.w != epoch) && !give_up(spins, p.err, 1));
    s_tokens[tid] = v.x;
    if (b == 0) p.recv_tokens[tid] = static_cast<int32_t>(v.x);
  }
  __syncthreads();

  // Padding rows (t >= tokens of that source) get invalid ids, as FlashInfer's sanitize does.
  for (uint32_t task = b * kThreads + tid; task < kRanks * p.tmax * kTopK; task += grid * kThreads) {
    const uint32_t s = task / (p.tmax * kTopK), t = task / kTopK % p.tmax;
    if (t >= s_tokens[s]) p.out_ids[task] = p.invalid_id;
  }
  uint32_t max_tokens = 0;
#pragma unroll
  for (uint32_t s = 0; s < kRanks; ++s)
    max_tokens = max(max_tokens, s_tokens[s]);
  const uint32_t recvs = max_tokens * kRanks * G::kLines;
  for (uint32_t task = b * kThreads + tid; task < recvs; task += grid * kThreads) {
    const uint32_t i = task % G::kLines, j = task / G::kLines, s = j % kRanks, t = j / kRanks;
    if (t >= s_tokens[s]) continue;
    const uint8_t* line = mine + G::slot_offset(stage, s, t, p.slots) + i * 16;
    uint4 v;
    uint32_t spins = 0;
    do {
      v = ld_line(line);
    } while ((v.y != epoch || v.w != epoch) && !give_up(spins, p.err, 2));
    const uint2 d = make_uint2(v.x, v.z);
    const uint64_t row = static_cast<uint64_t>(s) * p.tmax + t;
    if (i < G::kIdLines) {
      *reinterpret_cast<uint2*>(p.out_ids + row * kTopK + 2 * i) = d;
    } else if (i < G::kIdLines + G::kWtLines) {
      *reinterpret_cast<uint2*>(p.out_wts + row * kTopK + 2 * (i - G::kIdLines)) = d;
    } else if (i < G::kIdLines + G::kWtLines + G::kQLines) {
      *reinterpret_cast<uint2*>(p.out_x + row * G::kQBytes + 8 * (i - G::kIdLines - G::kWtLines)) = d;
    } else {
      *reinterpret_cast<uint2*>(p.out_sf + row * G::kSBytes + 8 * (i - G::kIdLines - G::kWtLines - G::kQLines)) = d;
    }
  }
  __syncthreads();
  if (tid == 0) p.ctr[b] = static_cast<int32_t>(epoch);
}

template <uint32_t kRanks, uint32_t kTopK, uint32_t kHidden, bool kUsePDL>
__global__ __launch_bounds__(kThreads) void combine_kernel(const __grid_constant__ CombineParams p) {
  using G = Geometry<kRanks, kTopK, kHidden>;
  __shared__ uint32_t s_tokens[kRanks];
  const uint32_t b = blockIdx.x, grid = gridDim.x, tid = threadIdx.x;
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  const uint32_t call = static_cast<uint32_t>(p.ctr[b]);
  const uint32_t stage = call % kCombineStages;
  if (tid < kRanks) s_tokens[tid] = static_cast<uint32_t>(p.recv_tokens[tid]);
  __syncthreads();
  uint32_t max_tokens = 0;
#pragma unroll
  for (uint32_t s = 0; s < kRanks; ++s)
    max_tokens = max(max_tokens, s_tokens[s]);

  // Push this rank's row of every routed (source, token) to the source.
  const uint32_t pushes = max_tokens * kRanks * G::kChunks;
  for (uint32_t task = b * kThreads + tid; task < pushes; task += grid * kThreads) {
    const uint32_t i = task % G::kChunks, j = task / G::kChunks, s = j % kRanks, t = j / kRanks;
    if (t >= s_tokens[s]) continue;
    const uint64_t row = static_cast<uint64_t>(s) * p.tmax + t;
    if (static_cast<uint32_t>(p.rids[row * kTopK]) >= p.num_experts) continue;
    const uint4 v = reinterpret_cast<const uint4*>(p.partial + row * kHidden)[i];
    st_line(
        p.peers.base[s] + G::combine_offset(stage, p.rank, t, p.slots) + i * 16,
        canonical_nan2(v.x),
        canonical_nan2(v.y),
        canonical_nan2(v.z),
        canonical_nan2(v.w));
  }

  // Sum the first-route contributions of every own token, chunk by chunk.
  uint8_t* mine = p.peers.base[p.rank];
  for (uint32_t task = b * kThreads + tid; task < p.tokens * G::kChunks; task += grid * kThreads) {
    const uint32_t i = task % G::kChunks, t = task / G::kChunks;
    int32_t src[kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      const uint32_t r = static_cast<uint32_t>(p.ids[t * kTopK + k]) / p.experts_per_rank;
      bool first = true;
#pragma unroll
      for (uint32_t q = 0; q < kTopK; ++q) {
        if (q < k && static_cast<uint32_t>(p.ids[t * kTopK + q]) / p.experts_per_rank == r) first = false;
      }
      src[k] = first ? static_cast<int32_t>(r) : -1;
    }
    uint4 v[kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      v[k] =
          src[k] >= 0 ? ld_line(mine + G::combine_offset(stage, src[k], t, p.slots) + i * 16) : make_uint4(0, 0, 0, 0);
    }
    uint32_t spins = 0;
    bool pending = true;
    while (pending) {
      pending = false;
#pragma unroll
      for (uint32_t k = 0; k < kTopK; ++k) {
        if (src[k] >= 0 && has_sentinel(v[k])) {
          v[k] = ld_line(mine + G::combine_offset(stage, src[k], t, p.slots) + i * 16);
          pending = true;
        }
      }
      if (pending && give_up(spins, p.err, 3)) break;
    }
    float acc[8][kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      const uint32_t w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
      for (uint32_t e = 0; e < 8; ++e) {
        acc[e][k] = src[k] >= 0 ? __uint_as_float((e & 1) ? (w[e >> 1] & 0xFFFF0000u) : (w[e >> 1] << 16)) : 0.0f;
      }
      if (src[k] >= 0) {
        st_line(
            mine + G::combine_offset(stage, src[k], t, p.slots) + i * 16,
            0xFFFFFFFFu,
            0xFFFFFFFFu,
            0xFFFFFFFFu,
            0xFFFFFFFFu);
      }
    }
    alignas(16) bf16_t o[8];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
      float* c = acc[e];
      // FlashInfer moeAlltoAllCombine.cu reduce_contributions<10>.
      c[0] += c[1];
      c[2] += c[3];
      c[4] += c[5];
      c[6] += c[7];
      c[8] += c[9];
      c[0] += c[2];
      c[4] += c[6];
      c[0] += c[4];
      c[0] += c[8];
      o[e] = __float2bfloat16(c[0]);
    }
    reinterpret_cast<uint4*>(p.out + static_cast<uint64_t>(t) * kHidden)[i] = *reinterpret_cast<const uint4*>(o);
  }
  __syncthreads();
  if (tid == 0) p.ctr[b] = static_cast<int32_t>(call + 1);
}

/// \brief Host entry points. Peer region bases come as a CPU int64 tensor [ranks].
/// \tparam kRanks EP ranks. \tparam kTopK experts per token. \tparam kHidden hidden size. \tparam kUsePDL PDL launch.
template <uint32_t kRanks, uint32_t kTopK, uint32_t kHidden, bool kUsePDL>
struct LlA2A {
  using G = Geometry<kRanks, kTopK, kHidden>;

  static Peers peers_from(const tvm::ffi::TensorView peers) {
    using namespace host;
    TensorMatcher({static_cast<int64_t>(kRanks)}).with_dtype<int64_t>().with_device<kDLCPU>().verify(peers);
    Peers out{};
    const int64_t* base = static_cast<const int64_t*>(peers.data_ptr());
    for (uint32_t r = 0; r < kRanks; ++r)
      out.base[r] = reinterpret_cast<uint8_t*>(base[r]);
    return out;
  }

  static void dispatch(
      const tvm::ffi::TensorView peers,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView sf,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView wts,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView out_x,
      const tvm::ffi::TensorView out_sf,
      const tvm::ffi::TensorView out_ids,
      const tvm::ffi::TensorView out_wts,
      const tvm::ffi::TensorView recv_tokens,
      const tvm::ffi::TensorView err,
      int64_t tmax,
      int64_t slots,
      int64_t rank,
      int64_t experts_per_rank,
      int64_t invalid_id) {
    using namespace host;
    auto T_ = SymbolicSize{"tokens"};
    auto R_ = SymbolicSize{"rows"};
    auto C_ = SymbolicSize{"ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T_, static_cast<int64_t>(G::kQBytes)}).with_dtype<uint8_t>().with_device(device).verify(x);
    TensorMatcher({T_, static_cast<int64_t>(G::kSBytes)}).with_dtype<uint8_t>().with_device(device).verify(sf);
    TensorMatcher({T_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({T_, static_cast<int64_t>(kTopK)}).with_dtype<fp32_t>().with_device(device).verify(wts);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(ctr);
    TensorMatcher({R_, static_cast<int64_t>(G::kQBytes)}).with_dtype<uint8_t>().with_device(device).verify(out_x);
    TensorMatcher({R_, static_cast<int64_t>(G::kSBytes)}).with_dtype<uint8_t>().with_device(device).verify(out_sf);
    TensorMatcher({R_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(out_ids);
    TensorMatcher({R_, static_cast<int64_t>(kTopK)}).with_dtype<fp32_t>().with_device(device).verify(out_wts);
    TensorMatcher({static_cast<int64_t>(kRanks)}).with_dtype<int32_t>().with_device(device).verify(recv_tokens);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(err);
    const int64_t tokens = T_.unwrap();
    CHECK_HOST(tokens <= tmax && tmax <= slots && slots <= kMaxTokens)
        << "ll_a2a dispatch: tokens " << tokens << " tmax " << tmax << " slots " << slots;
    CHECK_HOST(R_.unwrap() == kRanks * tmax) << "ll_a2a dispatch: " << R_.unwrap() << " output rows, tmax " << tmax;
    CHECK_HOST(rank >= 0 && rank < kRanks && experts_per_rank > 0);
    DispatchParams params{};
    params.peers = peers_from(peers);
    params.x = static_cast<const uint8_t*>(x.data_ptr());
    params.sf = static_cast<const uint8_t*>(sf.data_ptr());
    params.ids = static_cast<const int32_t*>(ids.data_ptr());
    params.wts = static_cast<const float*>(wts.data_ptr());
    params.ctr = static_cast<int32_t*>(ctr.data_ptr());
    params.out_x = static_cast<uint8_t*>(out_x.data_ptr());
    params.out_sf = static_cast<uint8_t*>(out_sf.data_ptr());
    params.out_ids = static_cast<int32_t*>(out_ids.data_ptr());
    params.out_wts = static_cast<float*>(out_wts.data_ptr());
    params.recv_tokens = static_cast<int32_t*>(recv_tokens.data_ptr());
    params.err = static_cast<int32_t*>(err.data_ptr());
    params.tokens = static_cast<uint32_t>(tokens);
    params.tmax = static_cast<uint32_t>(tmax);
    params.slots = static_cast<uint32_t>(slots);
    params.rank = static_cast<uint32_t>(rank);
    params.experts_per_rank = static_cast<uint32_t>(experts_per_rank);
    params.invalid_id = static_cast<int32_t>(invalid_id);
    LaunchKernel(static_cast<uint32_t>(C_.unwrap()), kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(dispatch_kernel<kRanks, kTopK, kHidden, kUsePDL>, params);
  }

  static void combine(
      const tvm::ffi::TensorView peers,
      const tvm::ffi::TensorView partial,
      const tvm::ffi::TensorView rids,
      const tvm::ffi::TensorView recv_tokens,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView err,
      int64_t tmax,
      int64_t slots,
      int64_t rank,
      int64_t experts_per_rank,
      int64_t num_experts) {
    using namespace host;
    auto T_ = SymbolicSize{"tokens"};
    auto R_ = SymbolicSize{"rows"};
    auto C_ = SymbolicSize{"ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, static_cast<int64_t>(kHidden)})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(partial);
    TensorMatcher({R_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(rids);
    TensorMatcher({static_cast<int64_t>(kRanks)}).with_dtype<int32_t>().with_device(device).verify(recv_tokens);
    TensorMatcher({T_, static_cast<int64_t>(kTopK)}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({T_, static_cast<int64_t>(kHidden)})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(out);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(ctr);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(err);
    const int64_t tokens = T_.unwrap();
    CHECK_HOST(tokens <= tmax && tmax <= slots && slots <= kMaxTokens)
        << "ll_a2a combine: tokens " << tokens << " tmax " << tmax << " slots " << slots;
    CHECK_HOST(R_.unwrap() == kRanks * tmax) << "ll_a2a combine: " << R_.unwrap() << " partial rows, tmax " << tmax;
    CHECK_HOST(rank >= 0 && rank < kRanks && experts_per_rank > 0 && num_experts == kRanks * experts_per_rank);
    CombineParams params{};
    params.peers = peers_from(peers);
    params.partial = static_cast<const bf16_t*>(partial.data_ptr());
    params.rids = static_cast<const int32_t*>(rids.data_ptr());
    params.recv_tokens = static_cast<const int32_t*>(recv_tokens.data_ptr());
    params.ids = static_cast<const int32_t*>(ids.data_ptr());
    params.out = static_cast<bf16_t*>(out.data_ptr());
    params.ctr = static_cast<int32_t*>(ctr.data_ptr());
    params.err = static_cast<int32_t*>(err.data_ptr());
    params.tokens = static_cast<uint32_t>(tokens);
    params.tmax = static_cast<uint32_t>(tmax);
    params.slots = static_cast<uint32_t>(slots);
    params.rank = static_cast<uint32_t>(rank);
    params.experts_per_rank = static_cast<uint32_t>(experts_per_rank);
    params.num_experts = static_cast<uint32_t>(num_experts);
    LaunchKernel(static_cast<uint32_t>(C_.unwrap()), kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(combine_kernel<kRanks, kTopK, kHidden, kUsePDL>, params);
  }
};

}  // namespace qwen_dep_ll_a2a

using qwen_dep_ll_a2a::LlA2A;

}  // namespace sglang
