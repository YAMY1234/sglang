// docs/222 §11 R3-a torch binding: expiry_truncate(U, W, count, indices, r, rfull, iters, rel_tol).
// U, W: [S, HV, 32, 128] or a layer slice [L, S, HV, 32, 128] (fp16, inner dims contiguous); count: [S, HV] or
// [L, S, HV] int32; indices: [B] int32 / int64 (stride allowed).  Launches on the current stream (graph-capturable).
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "gdn_factored_expiry.cuh"

// docs/222 §11.6: variants 9..12 = variant 0 with a different not-due CTA footprint / carveout (diagnosis of the served
// non-due bucket-4 offset).  9: preferred carveout = max shared; 10: carveout = max L1; 11: + dynamic shared so the CTA
// reserves the Triton kernel's 32 KB; 12: + dynamic shared to 96 KB so at most 2 CTAs fit per SM (the Triton kernel's
// register-limited residency).  Variant 0 is never touched.
constexpr int kPadTriton = 32768 - 23680;
constexpr int kPadTwoPerSm = 98304 - 23680;

void prepare() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_expiry::expiry_trunc_r32_kernel<16, 3, 1>,
                                      cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_expiry::expiry_trunc_r32_kernel<16, 3, 2>,
                                      cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
  C10_CUDA_CHECK(cudaFuncSetAttribute(gdn_expiry::expiry_trunc_r32_kernel<16, 3, 4>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kPadTwoPerSm));
}

void expiry_truncate(torch::Tensor U, torch::Tensor W, torch::Tensor cnt, torch::Tensor idx, int64_t r, int64_t rfull,
                     int64_t iters, double rel_tol, int64_t variant) {
  TORCH_CHECK(U.scalar_type() == at::kHalf && W.scalar_type() == at::kHalf, "fp16 factors");
  TORCH_CHECK(cnt.scalar_type() == at::kInt, "int32 counts");
  TORCH_CHECK(idx.scalar_type() == at::kInt || idx.scalar_type() == at::kLong, "int32/int64 indices");
  TORCH_CHECK(r == 16 && iters == 3, "compiled for R = 16, ITERS = 3");
  const bool stacked = U.dim() == 5;
  TORCH_CHECK(U.dim() == (stacked ? 5 : 4) && W.dim() == U.dim() && cnt.dim() == U.dim() - 2, "shapes");
  const int64_t L = stacked ? U.size(0) : 1;
  const int64_t HV = U.size(-3);
  TORCH_CHECK(U.size(-2) == 32 && U.size(-1) == 128 && W.size(-2) == 32 && W.size(-1) == 128, "RMAX 32, K = V = 128");
  TORCH_CHECK(rfull <= 32 && rfull > r, "rfull");
  // per-layer view must be contiguous over (S, HV, 32, 128) / (S, HV)
  TORCH_CHECK(U.stride(-1) == 1 && U.stride(-2) == 128 && U.stride(-3) == 32 * 128 && U.stride(-4) == HV * 32 * 128, "U layout");
  TORCH_CHECK(W.stride(-1) == 1 && W.stride(-2) == 128 && W.stride(-3) == 32 * 128 && W.stride(-4) == HV * 32 * 128, "W layout");
  TORCH_CHECK(cnt.stride(-1) == 1 && cnt.stride(-2) == HV, "count layout");
  const int64_t n = idx.numel();
  if (n == 0 || L == 0) return;
  dim3 grid((unsigned)(n * HV), (unsigned)L);
  auto stream = c10::cuda::getCurrentCUDAStream(U.device().index());
  // variant 0 = R3-a (§11.1); 1..8 = Triton-order MGS (§11.3): 1 + ROWORD + 2 * FUSE + 4 * XW
  __half* u = reinterpret_cast<__half*>(U.data_ptr<at::Half>());
  __half* w = reinterpret_cast<__half*>(W.data_ptr<at::Half>());
  int32_t* c = cnt.data_ptr<int32_t>();
  const void* ip = idx.data_ptr();
  const int i64 = idx.scalar_type() == at::kLong ? 1 : 0;
  const int64_t su = stacked ? U.stride(0) : 0, sw = stacked ? W.stride(0) : 0, sc = stacked ? cnt.stride(0) : 0;
#define GDN_EXPIRY_ARGS u, w, c, ip, i64, idx.stride(0), (int)HV, (int)rfull, (float)rel_tol, su, sw, sc
  switch (variant) {
    case 0: gdn_expiry::expiry_trunc_r32_kernel<16, 3><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 1: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 0, 0, 0><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 2: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 1, 0, 0><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 3: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 0, 1, 0><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 4: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 1, 1, 0><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 5: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 0, 0, 1><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 6: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 1, 0, 1><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 7: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 0, 1, 1><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 8: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 1, 1, 1><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    // docs/222 §11.6 not-due footprint probes / fixes: variant 0's code, its own function, attributes set in prepare()
    case 9: gdn_expiry::expiry_trunc_r32_kernel<16, 3, 1><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 10: gdn_expiry::expiry_trunc_r32_kernel<16, 3, 2><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
    case 11: gdn_expiry::expiry_trunc_r32_kernel<16, 3, 3><<<grid, 128, kPadTriton, stream>>>(GDN_EXPIRY_ARGS); break;
    case 12: gdn_expiry::expiry_trunc_r32_kernel<16, 3, 4><<<grid, 128, kPadTwoPerSm, stream>>>(GDN_EXPIRY_ARGS); break;
    default: TORCH_CHECK(false, "variant 0..12");
  }
#undef GDN_EXPIRY_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------------------------------
// docs/222 §12 R8: per-layer IF conditional nodes around the side-stream expiry, built during CUDA-graph stream capture
// (the CUDA-graph conditional-node recipe that PyTorch's CUDAGraph::begin_capture_to_if_node uses; here one setter
// kernel sets all layers' handles).  All host work happens at capture; replay is device-only.
static void capture_info(cudaStream_t s, cudaGraph_t* g, const cudaGraphNode_t** deps, const cudaGraphEdgeData** ed,
                         size_t* n) {
  cudaStreamCaptureStatus st;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(s, &st, nullptr, g, deps, ed, n));
  TORCH_CHECK(st == cudaStreamCaptureStatusActive, "R8: the current stream is not capturing a CUDA graph");
}

std::vector<int64_t> cond_handles(int64_t n) {
  TORCH_CHECK(n > 0 && n <= gdn_expiry::kMaxCondLayers, "R8: 1..64 handles");
  cudaGraph_t g;
  capture_info(c10::cuda::getCurrentCUDAStream(), &g, nullptr, nullptr, nullptr);
  std::vector<int64_t> out;
  for (int64_t i = 0; i < n; ++i) {
    cudaGraphConditionalHandle h;
    // default 0, re-applied at every launch: a body never runs unless the setter of this launch says so
    C10_CUDA_CHECK(cudaGraphConditionalHandleCreate(&h, g, 0, cudaGraphCondAssignDefault));
    out.push_back(static_cast<int64_t>(h));
  }
  return out;
}

void cond_set(torch::Tensor cnt, torch::Tensor idx, int64_t rfull, std::vector<int64_t> handles) {
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.dim() == 3 && cnt.stride(2) == 1 && cnt.stride(1) == cnt.size(2),
              "R8: count [L, S, HV] int32 with contiguous (S, HV)");
  TORCH_CHECK(idx.scalar_type() == at::kInt || idx.scalar_type() == at::kLong, "R8: int32/int64 indices");
  const int64_t L = cnt.size(0);
  TORCH_CHECK((int64_t)handles.size() == L && L <= gdn_expiry::kMaxCondLayers, "R8: one handle per layer");
  gdn_expiry::CondHandles hs{};
  for (int64_t l = 0; l < L; ++l) hs.h[l] = static_cast<unsigned long long>(handles[l]);
  const int threads = 32 * (int)std::min<int64_t>(L, 32);
  gdn_expiry::expiry_due_setcond_kernel<<<1, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      cnt.data_ptr<int32_t>(), cnt.stride(0), idx.data_ptr(), idx.scalar_type() == at::kLong ? 1 : 0, idx.stride(0),
      (int)idx.numel(), (int)cnt.size(2), (int)rfull, (int)L, hs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static cudaStream_t g_body_stream = nullptr;

int64_t cond_body_stream() {
  // the stream every IF body is captured on (created once, before any capture)
  if (g_body_stream == nullptr) C10_CUDA_CHECK(cudaStreamCreateWithFlags(&g_body_stream, cudaStreamNonBlocking));
  return reinterpret_cast<int64_t>(g_body_stream);
}

void cond_begin(int64_t handle) {
  TORCH_CHECK(g_body_stream != nullptr, "R8: cond_body_stream() must be created before capture");
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  cudaGraph_t g;
  const cudaGraphNode_t* deps;
  const cudaGraphEdgeData* ed;
  size_t n;
  capture_info(s, &g, &deps, &ed, &n);
  cudaGraphNodeParams p{};
  p.type = cudaGraphNodeTypeConditional;
  p.conditional.handle = static_cast<cudaGraphConditionalHandle>(handle);
  p.conditional.type = cudaGraphCondTypeIf;
  p.conditional.size = 1;
  cudaGraphNode_t node;
  C10_CUDA_CHECK(cudaGraphAddNode(&node, g, deps, ed, n, &p));
  // later work captured on this stream depends on the IF node (and so on its body)
  C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
  C10_CUDA_CHECK(cudaStreamBeginCaptureToGraph(g_body_stream, p.conditional.phGraph_out[0], nullptr, nullptr, 0,
                                               cudaStreamCaptureModeThreadLocal));
}

void cond_end() {
  cudaGraph_t body;
  C10_CUDA_CHECK(cudaStreamEndCapture(g_body_stream, &body));
}

void count_trap(torch::Tensor cnt, torch::Tensor idx, int64_t limit) {
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.dim() == 2 && cnt.is_contiguous(), "trap: count [S, HV] int32");
  gdn_expiry::count_trap_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      cnt.data_ptr<int32_t>(), idx.data_ptr(), idx.scalar_type() == at::kLong ? 1 : 0, idx.stride(0), (int)idx.numel(),
      (int)cnt.size(1), (int)limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expiry_truncate", &expiry_truncate, "docs/222 R3-a expiry");
  m.def("prepare", &prepare, "docs/222 §11.6 variant 9..12 function attributes");
  m.def("cond_handles", &cond_handles, "docs/222 §12 R8: n conditional handles of the capturing graph");
  m.def("cond_set", &cond_set, "docs/222 §12 R8: setter kernel (all layers' handles)");
  m.def("cond_body_stream", &cond_body_stream, "docs/222 §12 R8: the IF-body capture stream");
  m.def("cond_begin", &cond_begin, "docs/222 §12 R8: add an IF node on the current stream, start capturing its body");
  m.def("cond_end", &cond_end, "docs/222 §12 R8: end the IF body capture");
  m.def("count_trap", &count_trap, "docs/222 §12 debug trap: count >= limit at a step's entry");
}
