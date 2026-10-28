# SPDX-License-Identifier: Apache-2.0
"""Fixed K2 FP32 projection partials. AOT only; conv is owned by the core."""
from functools import cache
import os
from pathlib import Path

import cuda.bindings.driver as cuda
import torch
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.jit.cute_aot_cache import _load_object


@cache
def _front(device, rows):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('prewarm MK_GDN2 AOT modules before capture')
    directory = Path(os.environ['SGLANG_Q38C_MK_GDN2_AOT_DIR'])
    return _load_object(directory / f'front-m{rows}.o', 'q38_mkgdn2_front', False)


def project_splitk(x, weight):
    rows = x.shape[0]
    if rows not in (4, 8, 12) or x.shape[1] != 2560 or weight.shape != (4120, 2560):
        raise ValueError('MK_GDN2 projection requires TP4 M4/M8/M12')
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError('MK_GDN2 preserves BF16 projection weights/inputs')
    if x.stride() != (2560, 1) or weight.stride() != (2560, 1):
        raise ValueError('MK_GDN2 projection requires native contiguous rows')
    partial = torch.empty((2, rows, 4120), device=x.device, dtype=torch.float32)
    # L is the two K halves, not two weight copies. All layouts are static AOT.
    a = weight.as_strided((4120, 1280, 2), (2560, 1, 1280))
    b = x.as_strided((rows, 1280, 2), (2560, 1, 1280))
    c = partial.permute(2, 1, 0)
    # fuse_conv=False eliminates these inherited unused arguments entirely.
    tensors = (a, b, c, c, a, a, a, a, a, a, a)
    args = tuple(from_dlpack(t.detach(), assumed_align=16 if i < 4 else 2)
                 for i, t in enumerate(tensors))
    _front(x.device.index, rows)(*args, cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    return partial
