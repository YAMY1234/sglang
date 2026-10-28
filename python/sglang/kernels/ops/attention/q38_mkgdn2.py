# SPDX-License-Identifier: Apache-2.0
"""AOT K2 reduction + width4 conv + full-head 8-warp ReplaySSM/norm."""
import ctypes
from functools import cache
import os
from pathlib import Path

import torch

from sglang.kernels.ops.attention.q38_mkgdn import Params, _check


class Params2(ctypes.Structure):
    _fields_ = Params._fields_ + [
        ('partial', ctypes.c_void_p),
        ('part_stride', ctypes.c_int64), ('packed_stride', ctypes.c_int64),
        ('conv_state', ctypes.c_void_p),
        ('conv_slot', ctypes.c_int64), ('conv_channel', ctypes.c_int64), ('conv_element', ctypes.c_int64),
        ('conv_weight', ctypes.c_void_p),
        ('conv_weight_channel', ctypes.c_int64), ('conv_weight_element', ctypes.c_int64),
        ('conv_bias', ctypes.c_void_p), ('conv_bias_stride', ctypes.c_int64),
        ('conv_window', ctypes.c_void_p),
        ('window_slot', ctypes.c_int64), ('window_step', ctypes.c_int64),
        ('window_channel', ctypes.c_int64), ('window_element', ctypes.c_int64),
        ('intermediate_indices', ctypes.c_void_p), ('intermediate_index64', ctypes.c_int),
    ]


@cache
def _kernel(device, dt_bf16, norm_f32):
    from cuda.bindings import driver as cu

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('prewarm MK_GDN2 AOT modules before capture')
    directory = Path(os.environ['SGLANG_Q38C_MK_GDN2_AOT_DIR'])
    cubin = (directory / f'core-dt{int(dt_bf16)}-norm{int(norm_f32)}.cubin').read_bytes()
    module = _check(cu.cuModuleLoadData(cubin), 'MK_GDN2 module')
    fn = _check(cu.cuModuleGetFunction(module, b'q38_mkgdn2_core'), 'MK_GDN2 function')
    return module, fn


def core(partial, conv_weight, conv_bias, conv_state, window, intermediate_indices,
         A_log, dt_bias, checkpoint, ring_d, ring_k, ring_g, ring_dlo, ring_klo,
         state_indices, replay_indices, write_pos, cache_base, is_flush,
         norm_weight, *, eps=1e-6):
    from cuda.bindings import driver as cu

    rows = partial.shape[1]
    if partial.shape != (2, rows, 4120) or rows not in (4, 8, 12) or partial.dtype != torch.float32:
        raise ValueError('MK_GDN2 requires fixed TP4 projection partials')
    if checkpoint.shape[1:] != (12, 128, 128) or checkpoint.dtype != torch.bfloat16:
        raise ValueError('MK_GDN2 requires BF16 HK4/HV12 circular ReplaySSM')
    if ring_k.shape[1:] != (4, 16, 128) or ring_d.shape[1:] != (12, 16, 128):
        raise ValueError('MK_GDN2 requires the native 16-entry ring')
    if conv_state.shape[1:] != (2560, 3) or window.shape[1:] != (4, 2560, 3):
        raise ValueError('MK_GDN2 requires native width4 conv state/windows')
    if conv_weight.shape != (2560, 4) or conv_weight.dtype != torch.bfloat16:
        raise ValueError('MK_GDN2 preserves native BF16 convolution')
    if conv_state.dtype != torch.bfloat16 or window.dtype != torch.bfloat16:
        raise ValueError('MK_GDN2 requires BF16 conv history/window')
    if conv_bias is not None and conv_bias.dtype != torch.bfloat16:
        raise ValueError('MK_GDN2 requires native BF16 conv bias')
    out = torch.empty((rows, 12, 128), device=partial.device, dtype=torch.bfloat16)
    ptr = lambda t: 0 if t is None else t.data_ptr()
    stride = lambda t: 0 if t is None else t.stride(0)
    p = Params2(A_log=ptr(A_log), dt_bias=ptr(dt_bias),
        state=ptr(checkpoint), state_slot=stride(checkpoint),
        ring_d=ptr(ring_d), d_slot=stride(ring_d), ring_k=ptr(ring_k), k_slot=stride(ring_k),
        ring_g=ptr(ring_g), g_slot=stride(ring_g), ring_dlo=ptr(ring_dlo), dlo_slot=stride(ring_dlo),
        ring_klo=ptr(ring_klo), klo_slot=stride(ring_klo), state_indices=ptr(state_indices),
        replay_indices=ptr(replay_indices), state_index64=int(state_indices.dtype == torch.int64),
        replay_index64=int(replay_indices.dtype == torch.int64), write_pos=ptr(write_pos),
        cache_base=ptr(cache_base), is_flush=ptr(is_flush), norm_w=ptr(norm_weight),
        eps=eps, qscale=128**-0.5, y=ptr(out), y_stride=out.stride(0),
        partial=ptr(partial), part_stride=partial.stride(0), packed_stride=partial.stride(1),
        conv_state=ptr(conv_state), conv_slot=conv_state.stride(0),
        conv_channel=conv_state.stride(1), conv_element=conv_state.stride(2),
        conv_weight=ptr(conv_weight), conv_weight_channel=conv_weight.stride(0),
        conv_weight_element=conv_weight.stride(1), conv_bias=ptr(conv_bias),
        conv_bias_stride=0 if conv_bias is None else conv_bias.stride(0),
        conv_window=ptr(window), window_slot=window.stride(0), window_step=window.stride(1),
        window_channel=window.stride(2), window_element=window.stride(3),
        intermediate_indices=ptr(intermediate_indices),
        intermediate_index64=int(intermediate_indices.dtype == torch.int64))
    _, fn = _kernel(partial.device.index, dt_bias.dtype == torch.bfloat16, norm_weight.dtype == torch.float32)
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = rows // 4 * 12, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 256, 1, 1
    cfg.sharedMemBytes = 0
    cfg.hStream = cu.CUstream(torch.cuda.current_stream(partial.device).cuda_stream)
    attr = cu.CUlaunchAttribute()
    attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    attr.value.programmaticStreamSerializationAllowed = 1
    cfg.attrs, cfg.numAttrs = [attr], 1
    args = (ctypes.c_void_p * 1)(ctypes.addressof(p))
    _check(cu.cuLaunchKernelEx(cfg, fn, ctypes.addressof(args), 0), 'MK_GDN2 launch')
    return out
