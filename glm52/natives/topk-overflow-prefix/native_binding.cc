// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#include <ATen/core/TensorBody.h>
#include <torch/all.h>
#include <torch/library.h>
#include <optional>

namespace topk_overflow_prefix_serving_v2 {
void fast_topk_transform_interface(
    const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
    const at::Tensor&, std::optional<at::Tensor>);
}
// Registers only the new prefill operator; the image keeps every existing operator and its binary.
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("fast_topk_transform_fused_overflow_prefix(Tensor score, Tensor lengths, Tensor dst_page_table, "
        "Tensor src_page_table, Tensor cu_seqlens_q, Tensor? row_starts) -> ()");
  m.impl("fast_topk_transform_fused_overflow_prefix", torch::kCUDA,
         &topk_overflow_prefix_serving_v2::fast_topk_transform_interface);
}
