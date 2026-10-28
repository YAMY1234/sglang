# SPDX-License-Identifier: Apache-2.0
"""Original Q35 TP4 scheduling, FN shape adaptation; precompiled cubin only."""
import ctypes
from pathlib import Path

import torch

from sglang.kernels.ops.moe import q38_router_jit as jit
from sglang.kernels.ops.moe.q38_mk_moe import tensor_map

H, I, E, TOPK, MAX_M = 2560, 192, 512, 10, 7
GRID, CLUSTER, THREADS, SHARED = 140, 2, 256, 216064


class Params(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        'logits', 'expert_weights', 'topk_ids', 'x_sf', 'w13_sf', 'w2_sf',
        'g1_alpha', 'g1_scale_c', 'g2_alpha', 'out', 'trace',
    )] + [('num_tokens', ctypes.c_int)]


def load(cubin):
    return jit.load(Path(cubin).read_bytes(), 'q35_mkm_moe_tp4', (THREADS, 1, 1), SHARED)


def maps(w13, w2, xq):
    sw = jit.driver().CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    assert tuple(w13.shape) == (E, 2 * I, H // 2)
    assert tuple(w2.shape) == (E, H, I // 2)
    # Keep Q35's 128-byte SW128 shared-memory box. TMA zero-fills the
    # 32-byte K tail beyond each native I192 W2 row; storage remains I192.
    return (tensor_map(w13, E * 2 * I, H // 2, 128, 128, sw),
            tensor_map(w2, E * H, I // 2, 128, 256, sw),
            tensor_map(xq, xq.shape[0], H // 2, 128, 8, sw))


def run(kernel, tensor_maps, logits, weights, ids, xsf, layer, out):
    m = logits.shape[0]
    assert 1 <= m <= MAX_M
    tensors = (logits, weights, ids, xsf, layer['gemm1_weights_scale'],
               layer['gemm2_weights_scale'], layer['output1_scale_gate_scalar'],
               layer['output1_scale_scalar'], layer['output2_scale_scalar'], out)
    params = Params(*(t.data_ptr() for t in tensors), 0, m)
    jit.launch(kernel, (GRID, 1, 1), [*tensor_maps, params],
               torch.cuda.current_stream().cuda_stream, cluster=(CLUSTER, 1, 1))
    return out
