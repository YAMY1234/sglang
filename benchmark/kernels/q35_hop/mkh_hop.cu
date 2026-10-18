// Decode hop kernels (TP all-reduce + residual + RMSNorm [+ FP8 quant]) for the q35_hop
// micro-benchmark. Built as a shared library and driven from mkh_hop.py through ctypes.
//
// Transports:
//   LL   one-shot multicast push (multimem.st) of bf16 fragments, -0.0 sentinel, 3 stages,
//        reader re-arms its own copy after use.
//   TU   one-shot multicast push of 64-bit words {bf16x2 payload, 32-bit tag}; the tag is the
//        hop sequence number, so no re-arm is needed.
//   LDR1 every rank writes its partial into its own copy; readers multimem.ld_reduce the row,
//        NaN marks a copy that is not there yet.
//   LDR2 ld_reduce of this rank's 1/W slice, then multimem.st of the reduced slice (-0.0
//        sentinel) to every rank.
// Every spin is bounded by %globaltimer; on timeout a sticky error bit is set and the output
// is poisoned with NaN.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace mkh {

constexpr int kStages = 3;
constexpr uint32_t kNegZeroPair = 0x80008000u;
constexpr uint32_t kNanPair = 0x7FC07FC0u;
constexpr int kCtas = 4;       // CTAs per token (one cluster)
constexpr int kThreads = 128;  // threads per CTA; one 16-byte fragment of 8 bf16 per thread
constexpr int kVec = 8;        // bf16 per fragment
constexpr int kHidden = kCtas * kThreads * kVec;  // 4096
constexpr int kFrags = kHidden / kVec;            // fragments per token row

enum ErrBits : uint32_t { kErrTimeout = 1u, kErrBadShape = 2u };

struct Ctrl {
  uint32_t stage;   // Lamport stage of the next hop
  uint32_t arrive;  // CTA tickets of the running hop
  uint32_t seq;     // hops completed so far (tag source, stamp index)
  uint32_t err;     // sticky error bits
};

struct Params {
  // inputs
  const uint16_t* partial;      // [M, H] bf16, this rank's contribution (non-finalize)
  const uint16_t* routed;       // [M*K, H] bf16 (finalize)
  const uint16_t* expert_w;     // [M, K] bf16 (finalize)
  const int32_t* permuted;      // [M, K] (finalize)
  const uint16_t* shared;       // [M, H] bf16 gated shared expert output, may be null
  const uint16_t* residual_in;  // [M, H] bf16
  const uint16_t* gamma;        // [H] bf16, applied multiplier
  const float* fp8_scale;       // [1]
  const uint16_t* prod_src;     // [M, H] bf16 producer input (synthetic producer)
  // outputs
  uint16_t* residual_out;       // [M, H] bf16
  uint16_t* norm_out;           // [M, H] bf16, may be null
  uint8_t* fp8_out;             // [M, H] e4m3, may be null
  uint16_t* prod_out;           // [M, H] bf16 synthetic producer output when not pushing
  // transport
  void* mb_mc;                  // multicast base of the mailbox
  void* mb_uc;                  // this rank's unicast view of the mailbox
  void* aux_mc;                 // second buffer (two-shot all-gather), multicast
  void* aux_uc;
  Ctrl* ctrl;
  unsigned long long* stamps;   // optional [kStampCalls][grid][8]
  // shape and scalars
  int rank, world, m, m_cap, top_k;
  int stamp_calls;
  float eps, prod_coeff;
  unsigned long long timeout_ns;
  unsigned long long peer_uc[8];  // every rank's unicast mailbox base (unicast push)
};

// ---------------------------------------------------------------- PTX helpers

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint4 ld_v4(const void* p) {
  // Plain load: the compiler may batch it, but not move it above an asm memory clobber.
  return *static_cast<const uint4*>(p);
}
__device__ __forceinline__ uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_v4(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void st_relaxed_sys_v4(void* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void mc_st_v4(void* mc, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(mc),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void mc_st_b64(void* mc, unsigned long long v) {
  asm volatile("multimem.st.relaxed.sys.global.f64 [%0], %1;" ::"l"(mc),
               "d"(__longlong_as_double((long long)v))
               : "memory");
}
__device__ __forceinline__ unsigned long long ld_relaxed_sys_b64(const void* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 mc_ld_reduce_v4(const void* mc) {
  uint4 v;
  asm volatile(
      "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(mc)
      : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::
                   : "memory");
}
__device__ __forceinline__ void st_cluster_f32(float* local_slot, uint32_t peer, float v) {
  uint32_t local = static_cast<uint32_t>(__cvta_generic_to_shared(local_slot));
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(peer));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(v) : "memory");
}
__device__ __forceinline__ void mbar_init_expect(uint64_t* mbar, uint32_t bytes) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(a) : "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void cluster_sync_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// Send one f32 into the same smem slot of CTA `peer`, completing bytes on its mbarrier.
__device__ __forceinline__ void st_async_f32(float* local_slot, uint64_t* local_mbar, uint32_t peer, float v) {
  const uint32_t slot = static_cast<uint32_t>(__cvta_generic_to_shared(local_slot));
  const uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(local_mbar));
  uint32_t rslot, rmbar;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rslot) : "r"(slot), "r"(peer));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmbar) : "r"(mbar), "r"(peer));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(rslot),
               "f"(v), "r"(rmbar)
               : "memory");
}
__device__ __forceinline__ void mbar_wait_parity0(uint64_t* mbar) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "{\n\t.reg .pred P;\n\tWAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n\t"
      "@!P bra WAIT;\n\t}" ::"r"(a)
      : "memory");
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// Two f32 to an e4m3 pair; lo lands in the low byte.
__device__ __forceinline__ uint16_t cvt_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// ---------------------------------------------------------------- bf16 helpers

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void unpack8(uint4 v, float f[8]) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}
__device__ __forceinline__ uint4 pack8(const float f[8]) {
  return make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]), pack_bf16x2(f[4], f[5]),
                    pack_bf16x2(f[6], f[7]));
}
__device__ __forceinline__ bool word_has_negzero(uint32_t w) {
  return (w & 0xFFFFu) == 0x8000u || (w >> 16) == 0x8000u;
}
__device__ __forceinline__ bool frag_has_negzero(uint4 v) {
  return word_has_negzero(v.x) | word_has_negzero(v.y) | word_has_negzero(v.z) |
         word_has_negzero(v.w);
}
__device__ __forceinline__ uint32_t sanitize_word(uint32_t w) {
  if ((w & 0xFFFFu) == 0x8000u) w &= 0xFFFF0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000FFFFu;
  return w;
}
__device__ __forceinline__ uint4 sanitize(uint4 v) {
  return make_uint4(sanitize_word(v.x), sanitize_word(v.y), sanitize_word(v.z),
                    sanitize_word(v.w));
}
__device__ __forceinline__ bool word_has_nan(uint32_t w) {
  return ((w & 0x7F80u) == 0x7F80u && (w & 0x007Fu)) ||
         ((w & 0x7F800000u) == 0x7F800000u && (w & 0x007F0000u));
}
__device__ __forceinline__ bool frag_has_nan(uint4 v) {
  return word_has_nan(v.x) | word_has_nan(v.y) | word_has_nan(v.z) | word_has_nan(v.w);
}

// ---------------------------------------------------------------- layout

// Lamport / ld_reduce mailbox element offset (bf16 units): [stage][src][m_cap][H].
__device__ __forceinline__ size_t mb_off(const Params& p, uint32_t stage, int src, int token,
                                         int frag) {
  return ((static_cast<size_t>(stage) * p.world + src) * p.m_cap + token) * kHidden +
         static_cast<size_t>(frag) * kVec;
}
// Row buffer without a source dimension: [stage][m_cap][H].
__device__ __forceinline__ size_t row_off(const Params& p, uint32_t stage, int token, int frag) {
  return (static_cast<size_t>(stage) * p.m_cap + token) * kHidden +
         static_cast<size_t>(frag) * kVec;
}

struct HopState {
  uint32_t stage;
  uint32_t seq;
  uint32_t ticket;  // thread 0 only: arrival order of this CTA
};

// Every thread reads the stage after the PDL wait; thread 0 takes a ticket whose value is
// only consumed in hop_end, so the atomic's latency hides under the hop. The last CTA to
// arrive advances the stage there; no CTA of this hop can observe the advanced value.
__device__ __forceinline__ HopState hop_begin(const Params& p) {
  HopState hs;
  hs.stage = ld_volatile_u32(&p.ctrl->stage);
  hs.seq = ld_volatile_u32(&p.ctrl->seq);
  hs.ticket = 0;
  if (threadIdx.x == 0) hs.ticket = atomicAdd(&p.ctrl->arrive, 1u);
  return hs;
}

__device__ __forceinline__ void hop_end(const Params& p, const HopState& hs) {
  if (threadIdx.x == 0 && hs.ticket == gridDim.x * gridDim.y - 1) {
    p.ctrl->arrive = 0;
    p.ctrl->stage = (hs.stage + 1) % kStages;
    p.ctrl->seq = hs.seq + 1;
  }
}

__device__ __forceinline__ void fail(const Params& p, uint32_t bits) {
  atomicOr(&p.ctrl->err, bits);
}

// ---------------------------------------------------------------- push side

template <int W, bool UNI>
__device__ __forceinline__ void push_ll(const Params& p, uint32_t stage, int token, int frag, uint4 v) {
  v = sanitize(v);
  if (UNI) {
#pragma unroll
    for (int r = 0; r < W; ++r)
      st_relaxed_sys_v4(reinterpret_cast<uint16_t*>(p.peer_uc[r]) + mb_off(p, stage, p.rank, token, frag), v);
  } else {
    mc_st_v4(static_cast<uint16_t*>(p.mb_mc) + mb_off(p, stage, p.rank, token, frag), v);
  }
}

// ---------------------------------------------------------------- pull side

// Poll W sources of one fragment until none holds the -0.0 sentinel; sum in rank order (fp32).
template <int W>
__device__ __forceinline__ bool pull_ll(const Params& p, uint32_t stage, int token, int frag,
                                        float acc[8], unsigned long long* arrival = nullptr) {
  const uint16_t* uc = static_cast<const uint16_t*>(p.mb_uc);
  uint4 v[W];
#pragma unroll
  for (int r = 0; r < W; ++r) v[r] = ld_volatile_v4(uc + mb_off(p, stage, r, token, frag));
  uint32_t pending = 0;
#pragma unroll
  for (int r = 0; r < W; ++r) pending |= frag_has_negzero(v[r]) ? (1u << r) : 0u;
  if (arrival != nullptr) {
    const unsigned long long now = gtimer();
#pragma unroll
    for (int r = 0; r < W; ++r) if (!(pending & (1u << r))) arrival[r] = now;
  }
  unsigned long long t0 = 0;
  uint32_t spins = 0;
  while (pending) {
#pragma unroll
    for (int r = 0; r < W; ++r)
      if (pending & (1u << r)) v[r] = ld_volatile_v4(uc + mb_off(p, stage, r, token, frag));
    uint32_t still = 0;
#pragma unroll
    for (int r = 0; r < W; ++r)
      if ((pending & (1u << r)) && frag_has_negzero(v[r])) still |= 1u << r;
    if (arrival != nullptr && still != pending) {
      const unsigned long long now = gtimer();
#pragma unroll
      for (int r = 0; r < W; ++r) if ((pending & ~still) & (1u << r)) arrival[r] = now;
    }
    pending = still;
    if (pending && ((++spins & 255u) == 0)) {
      unsigned long long now = gtimer();
      if (t0 == 0) t0 = now;
      else if (now - t0 > p.timeout_ns) { fail(p, kErrTimeout); return false; }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
  for (int r = 0; r < W; ++r) {
    float f[8];
    unpack8(v[r], f);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = acc[e] + f[e];
  }
  return true;
}

template <int W>
__device__ __forceinline__ void rearm_ll(const Params& p, uint32_t stage, int token, int frag) {
  uint16_t* uc = static_cast<uint16_t*>(p.mb_uc);
  const uint4 s = make_uint4(kNegZeroPair, kNegZeroPair, kNegZeroPair, kNegZeroPair);
#pragma unroll
  for (int r = 0; r < W; ++r) st_v4(uc + mb_off(p, stage, r, token, frag), s);
}

// Tagged words: mailbox element is a 64-bit word {bf16x2, tag}; 4 words per fragment.
__device__ __forceinline__ size_t tu_off(const Params& p, uint32_t stage, int src, int token,
                                         int frag) {
  return (((static_cast<size_t>(stage) * p.world + src) * p.m_cap + token) * kFrags + frag) * 4;
}

template <int W>
__device__ __forceinline__ bool pull_tu(const Params& p, uint32_t stage, uint32_t tag, int token,
                                        int frag, float acc[8]) {
  const unsigned long long* uc = static_cast<const unsigned long long*>(p.mb_uc);
  unsigned long long w[W][4];
#pragma unroll
  for (int r = 0; r < W; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) w[r][j] = ld_relaxed_sys_b64(uc + tu_off(p, stage, r, token, frag) + j);
  unsigned long long t0 = 0;
  uint32_t spins = 0;
  while (true) {
    uint32_t pending = 0;
#pragma unroll
    for (int r = 0; r < W; ++r)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (static_cast<uint32_t>(w[r][j] >> 32) != tag) pending |= 1u << (r * 4 + j);
    if (!pending) break;
#pragma unroll
    for (int r = 0; r < W; ++r)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (pending & (1u << (r * 4 + j)))
          w[r][j] = ld_relaxed_sys_b64(uc + tu_off(p, stage, r, token, frag) + j);
    if ((++spins & 255u) == 0) {
      unsigned long long now = gtimer();
      if (t0 == 0) t0 = now;
      else if (now - t0 > p.timeout_ns) { fail(p, kErrTimeout); return false; }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
  for (int r = 0; r < W; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t lo = static_cast<uint32_t>(w[r][j]);
      acc[2 * j] = acc[2 * j] + bf_lo(lo);
      acc[2 * j + 1] = acc[2 * j + 1] + bf_hi(lo);
    }
  return true;
}

// ---------------------------------------------------------------- norm epilogue

// RMS order variants (to match the reference kernel bit for bit).
// kSqSeq: ((0 + x0^2) + x1^2) + ...; kSqTree: pairwise; kSqFma: fma chain from 0.
enum SqOrder { kSqSeq = 0, kSqTree = 1, kSqFma = 2 };
// Cross-warp reduction in warp 0: kWrUp xor 1,2,4,8,16; kWrDown xor 16,8,4,2,1.
enum WarpOrder { kWrUp = 0, kWrDown = 1 };

template <int SQ>
__device__ __forceinline__ float sum_sq8(const float x[8]) {
  if (SQ == kSqTree) {
    float s0 = __fadd_rn(__fmul_rn(x[0], x[0]), __fmul_rn(x[1], x[1]));
    float s1 = __fadd_rn(__fmul_rn(x[2], x[2]), __fmul_rn(x[3], x[3]));
    float s2 = __fadd_rn(__fmul_rn(x[4], x[4]), __fmul_rn(x[5], x[5]));
    float s3 = __fadd_rn(__fmul_rn(x[6], x[6]), __fmul_rn(x[7], x[7]));
    return __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
  } else if (SQ == kSqFma) {
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) s = __fmaf_rn(x[e], x[e], s);
    return s;
  } else {
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) s = __fadd_rn(s, __fmul_rn(x[e], x[e]));
    return s;
  }
}

constexpr int kWarps = kThreads / 32;
static_assert(kWarps == 4, "the reference fold below assumes four warps per CTA");

// Cluster-wide sum with one cluster barrier: lane butterfly per warp, then every warp leader
// writes its warp sum into each CTA of the cluster; each thread folds the kCtas x kWarps sums
// as the reference does (warp-0 fold per CTA, then CTAs in rank order from 0).
// Cluster-wide sum: lane butterfly per warp, then lanes 0..kCtas-1 of each warp send the warp
// sum to every CTA of the cluster with st.async; each CTA waits on its own mbarrier (no
// release fence, so the wait does not cover this thread's earlier remote stores) and folds the
// kCtas x kWarps sums as the reference does (warp-0 fold per CTA, then CTAs in rank order).
struct ClusterSmem {
  float xs[kCtas][kWarps];
  uint64_t mbar;
};

// Kernel prologue (before the PDL wait): arm the exchange, then make every CTA's barrier
// visible to its peers before any st.async can target it.
__device__ __forceinline__ void cluster_exchange_init(ClusterSmem& cs) {
  if (threadIdx.x == 0) mbar_init_expect(&cs.mbar, kCtas * kWarps * sizeof(float));
  cluster_sync_relaxed();
}

template <int WR>
__device__ __forceinline__ float cluster_sum(float v, ClusterSmem& cs) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  if (lane < kCtas) st_async_f32(&cs.xs[cluster_ctarank()][warp], &cs.mbar, lane, v);
  mbar_wait_parity0(&cs.mbar);
  float total = 0.f;
#pragma unroll
  for (int c = 0; c < kCtas; ++c) {
    const float cta = WR == kWrUp ? __fadd_rn(__fadd_rn(cs.xs[c][0], cs.xs[c][1]), __fadd_rn(cs.xs[c][2], cs.xs[c][3]))
                                  : __fadd_rn(__fadd_rn(cs.xs[c][0], cs.xs[c][2]), __fadd_rn(cs.xs[c][1], cs.xs[c][3]));
    total = __fadd_rn(total, cta);
  }
  return total;
}

template <int W>
__device__ __forceinline__ void rearm_ll(const Params& p, uint32_t stage, int token, int frag);

// residual + RMSNorm + optional FP8 from the fp32 rank sum; every global store (outputs and
// the mailbox re-arm) is issued after the cluster barrier so the barrier does not wait on them.
template <int SQ, int WR, int REARM_W>
__device__ __forceinline__ void norm_epilogue(const Params& p, int token, int frag, const float acc[8],
                                              const float gam[8], const uint4 res_in, ClusterSmem& cs,
                                              bool ok, uint32_t rearm_stage, float inv_s,
                                              unsigned long long* t_rms) {
  const size_t row = static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
  float res[8], pre[8];
  unpack8(res_in, res);
#pragma unroll
  for (int e = 0; e < 8; ++e) pre[e] = __fadd_rn(acc[e], res[e]);
  const uint4 pre_bf = pack8(pre);
  unpack8(pre_bf, pre);  // the bf16-rounded values feed the norm, as in the reference
  const float total = cluster_sum<WR>(sum_sq8<SQ>(pre), cs);
  if (t_rms != nullptr) *t_rms = gtimer();
  const float inv = rsqrtf(__fadd_rn(total / static_cast<float>(kHidden), p.eps));
  float out[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) out[e] = __fmul_rn(__fmul_rn(pre[e], inv), gam[e]);
  uint4 out_bf = pack8(out);
  if (!ok) out_bf = make_uint4(kNanPair, kNanPair, kNanPair, kNanPair);
  st_v4(p.residual_out + row, pre_bf);
  if (p.norm_out != nullptr) st_v4(p.norm_out + row, out_bf);
  if (p.fp8_out != nullptr) {
    float q[8];
    unpack8(out_bf, q);
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a = fminf(fmaxf(__fmul_rn(q[2 * j], inv_s), -448.f), 448.f);
      const float b = fminf(fmaxf(__fmul_rn(q[2 * j + 1], inv_s), -448.f), 448.f);
      w[j] = cvt_e4m3x2(a, b);
    }
    *reinterpret_cast<uint2*>(p.fp8_out + row) = make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
  }
  if (REARM_W > 0) rearm_ll<REARM_W>(p, rearm_stage, token, frag);
}

__device__ __forceinline__ void load_gamma(const Params& p, int frag, float gam[8]) {
  uint4 g = ld_v4(p.gamma + static_cast<size_t>(frag) * kVec);
  unpack8(g, gam);
}

// Finalize of this rank's MoE output for one fragment: sum_k w_k * routed[perm_k] + shared,
// accumulated in k order as the reference does; all row loads are issued before the sum.
template <bool FMA>
__device__ __forceinline__ uint4 finalize_frag(const Params& p, int token, int frag,
                                               const int32_t* rows, const float* wts) {
  constexpr int kMaxK = 16;
  uint4 src[kMaxK];
#pragma unroll
  for (int k = 0; k < kMaxK; ++k)
    if (k < p.top_k && rows[k] >= 0)
      src[k] = ld_v4(p.routed + static_cast<size_t>(rows[k]) * kHidden + static_cast<size_t>(frag) * kVec);
  uint4 sh = make_uint4(0, 0, 0, 0);
  if (p.shared != nullptr)
    sh = ld_v4(p.shared + static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec);
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxK; ++k) {
    if (k < p.top_k && rows[k] >= 0) {
      float f[8];
      unpack8(src[k], f);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[e] = FMA ? __fmaf_rn(f[e], wts[k], acc[e]) : __fadd_rn(acc[e], __fmul_rn(f[e], wts[k]));
    }
  }
  if (p.shared != nullptr) {
    float f[8];
    unpack8(sh, f);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = __fadd_rn(acc[e], f[e]);
  }
  return pack8(acc);
}

// Stamp points (thread 0 of each CTA): 0 entry, 1 after the PDL wait, 2 push issued,
// 3 all sources arrived, 4 cluster RMS done, 5 last store issued.

// M1: one kernel = push + pull + residual + RMSNorm (+ FP8). FIN adds the MoE finalize.
template <int W, bool FIN, bool FIN_FMA, int SQ, int WR, bool TU, bool UNI>
__global__ void __launch_bounds__(kThreads) hop_one_kernel(Params p) {
  __shared__ ClusterSmem cs;
  __shared__ int32_t meta_rows[16];
  __shared__ float meta_wts[16];
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kThreads + threadIdx.x;
  const unsigned long long t_entry = gtimer();
  unsigned long long arrival[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  pdl_trigger();
  float gam[8];
  load_gamma(p, frag, gam);
  const float inv_s = p.fp8_out != nullptr ? div_full(1.f, *p.fp8_scale) : 1.f;
  if (FIN) {
    // Routing metadata comes from the router, several kernels upstream; it is staged before
    // the PDL wait, as the stock finalize does.
    if (threadIdx.x < p.top_k) {
      const int i = token * p.top_k + threadIdx.x;
      const int r = p.permuted[i];
      meta_rows[threadIdx.x] = r;
      meta_wts[threadIdx.x] = r < 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(p.expert_w[i]) << 16);
    }
    __syncthreads();
  }
  cluster_exchange_init(cs);
  pdl_wait();
  const unsigned long long t_wait = gtimer();
  const HopState hs = hop_begin(p);
  const size_t row = static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
  const uint4 res_in = ld_v4(p.residual_in + row);
  uint4 mine;
  if (FIN) {
    mine = finalize_frag<FIN_FMA>(p, token, frag, meta_rows, meta_wts);
  } else {
    mine = ld_v4(p.partial + row);
  }
  float acc[8];
  bool ok;
  unsigned long long t_push;
  if (TU) {
    const uint32_t tag = hs.seq + 1;
    unsigned long long* mc = static_cast<unsigned long long*>(p.mb_mc) + tu_off(p, hs.stage, p.rank, token, frag);
    mc_st_b64(mc + 0, (static_cast<unsigned long long>(tag) << 32) | mine.x);
    mc_st_b64(mc + 1, (static_cast<unsigned long long>(tag) << 32) | mine.y);
    mc_st_b64(mc + 2, (static_cast<unsigned long long>(tag) << 32) | mine.z);
    mc_st_b64(mc + 3, (static_cast<unsigned long long>(tag) << 32) | mine.w);
    t_push = gtimer();
    ok = pull_tu<W>(p, hs.stage, tag, token, frag, acc);
  } else {
    push_ll<W, UNI>(p, hs.stage, token, frag, mine);
    t_push = gtimer();
    ok = pull_ll<W>(p, hs.stage, token, frag, acc, threadIdx.x == 0 && p.stamps != nullptr ? arrival : nullptr);
  }
  const unsigned long long t_pull = gtimer();
  unsigned long long t_rms = 0;
  if (TU) norm_epilogue<SQ, WR, 0>(p, token, frag, acc, gam, res_in, cs, ok, 0, inv_s, &t_rms);
  else norm_epilogue<SQ, WR, W>(p, token, frag, acc, gam, res_in, cs, ok, hs.stage, inv_s, &t_rms);
  hop_end(p, hs);
  if (p.stamps != nullptr && p.stamp_calls > 0 && threadIdx.x == 0) {
    const size_t call = hs.seq % static_cast<uint32_t>(p.stamp_calls);
    unsigned long long* st = p.stamps + (call * gridDim.x * gridDim.y + blockIdx.x * gridDim.y + blockIdx.y) * 16;
    st[0] = t_entry; st[1] = t_wait; st[2] = t_push; st[3] = t_pull; st[4] = t_rms; st[5] = gtimer();
    st[6] = hs.seq;
#pragma unroll
    for (int r = 0; r < 8; ++r) st[8 + r] = arrival[r];
  }
}

// M2 producer stand-in: out = bf16(src * coeff); pushes it (PUSH) or writes it to prod_out.
template <int W, bool PUSH, bool UNI>
__global__ void __launch_bounds__(kThreads) prod_kernel(Params p) {
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kThreads + threadIdx.x;
  pdl_trigger();
  pdl_wait();
  const size_t row = static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
  uint32_t stage = 0;
  if (PUSH) stage = ld_volatile_u32(&p.ctrl->stage);
  float f[8];
  unpack8(ld_v4(p.prod_src + row), f);
#pragma unroll
  for (int e = 0; e < 8; ++e) f[e] = __fmul_rn(f[e], p.prod_coeff);
  const uint4 out = pack8(f);
  if (PUSH) {
    push_ll<W, UNI>(p, stage, token, frag, out);
  } else {
    st_v4(p.prod_out + row, out);
  }
}

// M2 consumer: pull + residual + RMSNorm (+ FP8); its producer already pushed.
template <int W, int SQ, int WR>
__global__ void __launch_bounds__(kThreads) consume_kernel(Params p) {
  __shared__ ClusterSmem cs;
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kThreads + threadIdx.x;
  const unsigned long long t_entry = gtimer();
  pdl_trigger();
  float gam[8];
  load_gamma(p, frag, gam);
  const float inv_s = p.fp8_out != nullptr ? div_full(1.f, *p.fp8_scale) : 1.f;
  cluster_exchange_init(cs);
  pdl_wait();
  const unsigned long long t_wait = gtimer();
  const HopState hs = hop_begin(p);
  const size_t row = static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
  const uint4 res_in = ld_v4(p.residual_in + row);
  float acc[8];
  const bool ok = pull_ll<W>(p, hs.stage, token, frag, acc);
  const unsigned long long t_pull = gtimer();
  unsigned long long t_rms = 0;
  norm_epilogue<SQ, WR, W>(p, token, frag, acc, gam, res_in, cs, ok, hs.stage, inv_s, &t_rms);
  hop_end(p, hs);
  if (p.stamps != nullptr && p.stamp_calls > 0 && threadIdx.x == 0) {
    const size_t call = hs.seq % static_cast<uint32_t>(p.stamp_calls);
    unsigned long long* st = p.stamps + (call * gridDim.x * gridDim.y + blockIdx.x * gridDim.y + blockIdx.y) * 16;
    st[0] = t_entry; st[1] = t_wait; st[2] = t_wait; st[3] = t_pull; st[4] = t_rms; st[5] = gtimer();
    st[6] = hs.seq;
  }
}

// M3 one-shot: own copy holds this rank's partial; ld_reduce the row until no NaN.
template <int SQ, int WR>
__global__ void __launch_bounds__(kThreads) ldr1_kernel(Params p) {
  __shared__ ClusterSmem cs;
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kThreads + threadIdx.x;
  pdl_trigger();
  float gam[8];
  load_gamma(p, frag, gam);
  const float inv_s = p.fp8_out != nullptr ? div_full(1.f, *p.fp8_scale) : 1.f;
  cluster_exchange_init(cs);
  pdl_wait();
  const HopState hs = hop_begin(p);
  const size_t row = static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
  const uint4 mine = ld_v4(p.partial + row);
  const uint4 res_in = ld_v4(p.residual_in + row);
  uint16_t* uc = static_cast<uint16_t*>(p.mb_uc);
  st_relaxed_sys_v4(uc + row_off(p, hs.stage, token, frag), mine);
  const uint16_t* mc = static_cast<const uint16_t*>(p.mb_mc) + row_off(p, hs.stage, token, frag);
  uint4 v = mc_ld_reduce_v4(mc);
  bool ok = true;
  unsigned long long t0 = 0;
  uint32_t spins = 0;
  while (frag_has_nan(v)) {
    v = mc_ld_reduce_v4(mc);
    if ((++spins & 63u) == 0) {
      unsigned long long now = gtimer();
      if (t0 == 0) t0 = now;
      else if (now - t0 > p.timeout_ns) { fail(p, kErrTimeout); ok = false; break; }
    }
  }
  // Every rank finished reading the previous stage before it wrote this one.
  const uint4 nan4 = make_uint4(kNanPair, kNanPair, kNanPair, kNanPair);
  st_relaxed_sys_v4(uc + row_off(p, (hs.stage + kStages - 1) % kStages, token, frag), nan4);
  float acc[8];
  unpack8(v, acc);
  norm_epilogue<SQ, WR, 0>(p, token, frag, acc, gam, res_in, cs, ok, 0, inv_s, nullptr);
  hop_end(p, hs);
}

// M3 two-shot: ld_reduce this rank's 1/W slice, multimem.st it, then pull the whole row.
template <int W, int SQ, int WR>
__global__ void __launch_bounds__(kThreads) ldr2_kernel(Params p) {
  __shared__ ClusterSmem cs;
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kThreads + threadIdx.x;
  pdl_trigger();
  float gam[8];
  load_gamma(p, frag, gam);
  const float inv_s = p.fp8_out != nullptr ? div_full(1.f, *p.fp8_scale) : 1.f;
  cluster_exchange_init(cs);
  pdl_wait();
  const HopState hs = hop_begin(p);
  const size_t row = static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
  const uint4 mine = ld_v4(p.partial + row);
  const uint4 res_in = ld_v4(p.residual_in + row);
  uint16_t* uc = static_cast<uint16_t*>(p.mb_uc);
  st_relaxed_sys_v4(uc + row_off(p, hs.stage, token, frag), mine);
  const uint4 nan4 = make_uint4(kNanPair, kNanPair, kNanPair, kNanPair);
  bool ok = true;
  constexpr int kSlice = kFrags / W;
  if (threadIdx.x + blockIdx.y * kThreads < kSlice) {
    const int sfrag = p.rank * kSlice + threadIdx.x + blockIdx.y * kThreads;
    const uint16_t* mc = static_cast<const uint16_t*>(p.mb_mc) + row_off(p, hs.stage, token, sfrag);
    uint4 v = mc_ld_reduce_v4(mc);
    unsigned long long t0 = 0;
    uint32_t spins = 0;
    while (frag_has_nan(v)) {
      v = mc_ld_reduce_v4(mc);
      if ((++spins & 63u) == 0) {
        unsigned long long now = gtimer();
        if (t0 == 0) t0 = now;
        else if (now - t0 > p.timeout_ns) { fail(p, kErrTimeout); ok = false; break; }
      }
    }
    mc_st_v4(static_cast<uint16_t*>(p.aux_mc) + row_off(p, hs.stage, token, sfrag), sanitize(v));
  }
  const uint16_t* auc = static_cast<const uint16_t*>(p.aux_uc) + row_off(p, hs.stage, token, frag);
  uint4 v = ld_volatile_v4(auc);
  unsigned long long t0 = 0;
  uint32_t spins = 0;
  while (frag_has_negzero(v)) {
    v = ld_volatile_v4(auc);
    if ((++spins & 255u) == 0) {
      unsigned long long now = gtimer();
      if (t0 == 0) t0 = now;
      else if (now - t0 > p.timeout_ns) { fail(p, kErrTimeout); ok = false; break; }
    }
  }
  st_v4(static_cast<uint16_t*>(p.aux_uc) + row_off(p, hs.stage, token, frag),
        make_uint4(kNegZeroPair, kNegZeroPair, kNegZeroPair, kNegZeroPair));
  // A complete all-gather fragment means its owner reduced this hop, so every rank wrote this
  // stage, hence finished reading the previous one.
  st_relaxed_sys_v4(uc + row_off(p, (hs.stage + kStages - 1) % kStages, token, frag), nan4);
  float acc[8];
  unpack8(v, acc);
  norm_epilogue<SQ, WR, 0>(p, token, frag, acc, gam, res_in, cs, ok, 0, inv_s, nullptr);
  hop_end(p, hs);
}

// Latency probe: rank 0 and rank `peer` bounce a 16-byte tagged fragment `iters` times through
// the mailbox (multicast store or unicast store into the peer's copy); one thread per rank.
template <bool UNI>
__global__ void pingpong_kernel(Params p, int peer, int iters, unsigned long long* out, unsigned long long gap_ns) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const bool a = p.rank == 0, b = p.rank == peer && peer != 0;
  if (!a && !b) return;
  if (peer == 0) {  // self loop: rank 0 multicasts and polls its own copy, idling gap_ns between rounds
    uint16_t* own = static_cast<uint16_t*>(p.mb_uc) + mb_off(p, 2, 0, 0, 0);
    unsigned long long busy = 0;
    for (int i = 1; i <= iters; ++i) {
      const unsigned long long g0 = gtimer();
      while (gtimer() - g0 < gap_ns) {}
      const unsigned long long t0 = gtimer();
      const uint4 v = make_uint4(i, i, i, i);
      if (UNI) st_relaxed_sys_v4(own, v);
      else mc_st_v4(static_cast<uint16_t*>(p.mb_mc) + mb_off(p, 2, 0, 0, 0), v);
      while (ld_volatile_v4(own).x != static_cast<uint32_t>(i)) {
        if (gtimer() - t0 > p.timeout_ns) { fail(p, kErrTimeout); return; }
      }
      busy += gtimer() - t0;
    }
    out[0] = busy * 2;  // reported as a one-way: busy / iters
    return;
  }
  const int other = a ? peer : 0;
  // slot [stage 2][src][token 0][frag 0]; the tag is stored in all four words
  uint16_t* mine_uc = static_cast<uint16_t*>(p.mb_uc) + mb_off(p, 2, other, 0, 0);
  unsigned long long t0 = gtimer();
  for (int i = 1; i <= iters; ++i) {
    const uint32_t tag = static_cast<uint32_t>(i) | (b ? 0x40000000u : 0u);
    const uint32_t want = static_cast<uint32_t>(i) | (a ? 0x40000000u : 0u);
    if (a) {
      if (gap_ns) { const unsigned long long g0 = gtimer(); while (gtimer() - g0 < gap_ns) {} }
      const uint4 v = make_uint4(tag, tag, tag, tag);
      if (UNI) st_relaxed_sys_v4(reinterpret_cast<uint16_t*>(p.peer_uc[other]) + mb_off(p, 2, p.rank, 0, 0), v);
      else mc_st_v4(static_cast<uint16_t*>(p.mb_mc) + mb_off(p, 2, p.rank, 0, 0), v);
    }
    unsigned long long w0 = gtimer();
    while (ld_volatile_v4(mine_uc).x != want) {
      if (gtimer() - w0 > p.timeout_ns) { fail(p, kErrTimeout); return; }
    }
    if (b) {
      const uint4 v = make_uint4(tag, tag, tag, tag);
      if (UNI) st_relaxed_sys_v4(reinterpret_cast<uint16_t*>(p.peer_uc[other]) + mb_off(p, 2, p.rank, 0, 0), v);
      else mc_st_v4(static_cast<uint16_t*>(p.mb_mc) + mb_off(p, 2, p.rank, 0, 0), v);
    }
  }
  if (a) out[0] = gtimer() - t0 - static_cast<unsigned long long>(iters) * gap_ns;
}

}  // namespace mkh

// ---------------------------------------------------------------- host entry points

using namespace mkh;

enum Variant : int {
  kOneLL = 1,       // M1, all-reduce, multicast push
  kOneLLFin = 2,    // M1, finalize + all-reduce
  kProdPush = 3,    // M2 producer, multicast push
  kProdPlain = 4,   // M2 producer, writes prod_out
  kConsume = 5,     // M2 consumer
  kLdr1 = 6,        // M3 one-shot
  kLdr2 = 7,        // M3 two-shot
  kOneTU = 8,       // M4 tagged words, all-reduce
  kOneLLFinFma = 9,
  kOneLLUni = 10,   // M1, unicast push to every peer
  kProdPushUni = 11 // M2 producer, unicast push
};

template <typename K>
static int launch(K kernel, const Params& p, cudaStream_t stream, bool cluster) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.m, kCtas, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  int n = 0;
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  if (cluster) {
    attrs[n].id = cudaLaunchAttributeClusterDimension;
    attrs[n].val.clusterDim.x = 1;
    attrs[n].val.clusterDim.y = kCtas;
    attrs[n].val.clusterDim.z = 1;
    ++n;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = n;
  return static_cast<int>(cudaLaunchKernelEx(&cfg, kernel, p));
}

// order = sq * 2 + wr selects the RMS fold (see SqOrder / WarpOrder).
#define MKH_BY_ORDER(EXPR_SQ_WR)                                      \
  switch (order) {                                                    \
    case 0: { constexpr int SQ = kSqSeq, WR = kWrUp; EXPR_SQ_WR; }    \
    case 1: { constexpr int SQ = kSqSeq, WR = kWrDown; EXPR_SQ_WR; }  \
    case 2: { constexpr int SQ = kSqTree, WR = kWrUp; EXPR_SQ_WR; }   \
    case 3: { constexpr int SQ = kSqTree, WR = kWrDown; EXPR_SQ_WR; } \
    case 4: { constexpr int SQ = kSqFma, WR = kWrUp; EXPR_SQ_WR; }    \
    default: { constexpr int SQ = kSqFma, WR = kWrDown; EXPR_SQ_WR; } \
  }

template <int W>
static int dispatch(int variant, int order, const Params& p, cudaStream_t s) {
  switch (variant) {
    case kOneLL: MKH_BY_ORDER(return launch(hop_one_kernel<W, false, false, SQ, WR, false, false>, p, s, true))
    case kOneLLUni: MKH_BY_ORDER(return launch(hop_one_kernel<W, false, false, SQ, WR, false, true>, p, s, true))
    case kOneLLFin: MKH_BY_ORDER(return launch(hop_one_kernel<W, true, false, SQ, WR, false, false>, p, s, true))
    case kOneLLFinFma: MKH_BY_ORDER(return launch(hop_one_kernel<W, true, true, SQ, WR, false, false>, p, s, true))
    case kOneTU: MKH_BY_ORDER(return launch(hop_one_kernel<W, false, false, SQ, WR, true, false>, p, s, true))
    case kProdPush: return launch(prod_kernel<W, true, false>, p, s, false);
    case kProdPushUni: return launch(prod_kernel<W, true, true>, p, s, false);
    case kProdPlain: return launch(prod_kernel<W, false, false>, p, s, false);
    case kConsume: MKH_BY_ORDER(return launch(consume_kernel<W, SQ, WR>, p, s, true))
    case kLdr1: MKH_BY_ORDER(return launch(ldr1_kernel<SQ, WR>, p, s, true))
    case kLdr2: MKH_BY_ORDER(return launch(ldr2_kernel<W, SQ, WR>, p, s, true))
    default: return -1;
  }
}
#undef MKH_BY_ORDER

extern "C" int mkh_launch(int variant, int order, const Params* params, void* stream) {
  const Params& p = *params;
  if (p.m < 1 || p.m > p.m_cap) return -2;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (p.world) {
    case 2: return dispatch<2>(variant, order, p, s);
    case 4: return dispatch<4>(variant, order, p, s);
    case 8: return dispatch<8>(variant, order, p, s);
    default: return -3;
  }
}

extern "C" int mkh_pingpong(const Params* params, int peer, int iters, int unicast,
                            unsigned long long* out, void* stream, unsigned long long gap_ns) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (unicast) pingpong_kernel<true><<<1, 32, 0, s>>>(*params, peer, iters, out, gap_ns);
  else pingpong_kernel<false><<<1, 32, 0, s>>>(*params, peer, iters, out, gap_ns);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int mkh_params_size() { return static_cast<int>(sizeof(Params)); }
extern "C" int mkh_hidden() { return kHidden; }
