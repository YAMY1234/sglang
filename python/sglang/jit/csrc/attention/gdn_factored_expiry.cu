// docs/222 §11 R3-a torch binding: expiry_truncate(U, W, count, indices, r, rfull, iters, rel_tol).
// U, W: [S, HV, 32, 128] or a layer slice [L, S, HV, 32, 128] (fp16 or fp32, the same for both; inner dims contiguous); count: [S, HV] or
// [L, S, HV] int32; indices: [B] int32 / int64 (stride allowed).  Launches on the current stream (graph-capturable).
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "gdn_factored_expiry.cuh"

void expiry_truncate(torch::Tensor U, torch::Tensor W, torch::Tensor cnt, torch::Tensor idx, int64_t r, int64_t rfull,
                     int64_t iters, double rel_tol, int64_t variant) {
  TORCH_CHECK(U.scalar_type() == W.scalar_type() && (U.scalar_type() == at::kHalf || U.scalar_type() == at::kFloat),
              "fp16 or fp32 factors");
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
  int32_t* c = cnt.data_ptr<int32_t>();
  const void* ip = idx.data_ptr();
  const int i64 = idx.scalar_type() == at::kLong ? 1 : 0;
  const int64_t su = stacked ? U.stride(0) : 0, sw = stacked ? W.stride(0) : 0, sc = stacked ? cnt.stride(0) : 0;
  auto launch = [&](auto factor_value) {
    using FT = decltype(factor_value);
    FT* u = reinterpret_cast<FT*>(U.data_ptr());
    FT* w = reinterpret_cast<FT*>(W.data_ptr());
#define GDN_EXPIRY_ARGS u, w, c, ip, i64, idx.stride(0), (int)HV, (int)rfull, (float)rel_tol, su, sw, sc
    switch (variant) {
      case 0: gdn_expiry::expiry_trunc_r32_kernel<16, 3, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      case 1: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 0, 0, 0, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      case 2: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 1, 0, 0, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      case 3: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 0, 1, 0, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      case 4: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 1, 1, 0, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      case 5: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 0, 0, 1, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      case 6: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 1, 0, 1, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      case 7: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 0, 1, 1, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      case 8: gdn_expiry::expiry_trunc_r32_exact_kernel<16, 3, 1, 1, 1, FT><<<grid, 128, 0, stream>>>(GDN_EXPIRY_ARGS); break;
      default: TORCH_CHECK(false, "variant 0..8");
    }
#undef GDN_EXPIRY_ARGS
  };
  // variant 0 = R3-a (§11.1); 1..8 = Triton-order MGS (§11.3): 1 + ROWORD + 2 * FUSE + 4 * XW
  if (U.scalar_type() == at::kFloat) launch(float{});
  else launch(__half{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("expiry_truncate", &expiry_truncate, "docs/222 R3-a expiry"); }
