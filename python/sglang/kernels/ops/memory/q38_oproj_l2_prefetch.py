"""Capture-only, read-only L2 hints on an FN layer's output weight."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_module():
    return load_jit(
        "q38_oproj_l2_prefetch",
        cuda_files=["memory/q38_oproj_l2_prefetch.cuh"],
        cuda_wrappers=[
            ("begin", "Q38OProjL2Prefetch::begin"),
            ("join", "Q38OProjL2Prefetch::join"),
        ],
    )


class OProjL2Prefetch:
    def __init__(self, projection):
        self.projection = projection
        self.weight = None
        self.module = None
        self.node = 0
        self.supported = None

    def begin(self, hidden_states, forward_batch):
        self.node = 0
        if (
            not forward_batch.forward_mode.is_target_verify()
            or hidden_states.shape[0] != 4
        ):
            return
        if self.supported is None:
            # First eligible warmup is after load/repack. Keep the final tensor
            # alive along with the graph rather than reconstructing a layout.
            weight = self.projection.weight
            self.supported = (
                weight.is_cuda
                and weight.dtype == torch.bfloat16
                and tuple(weight.shape) == (2560, 6144)
                and weight.is_contiguous()
                and torch.cuda.get_device_capability(weight.device) == (10, 3)
            )
            if self.supported:
                self.weight = weight
                self.module = _jit_module()
        if self.supported:
            self.node = self.module.begin(self.weight)

    def join(self):
        if self.node:
            self.module.join(self.weight, self.node)
            self.node = 0
