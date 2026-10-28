# SPDX-License-Identifier: Apache-2.0
"""Original Q35 TP4 scheduling, FN shape adaptation; precompiled cubin only."""
import ctypes
from functools import cache
import hashlib
import json
import logging
import os
from pathlib import Path

import torch
from flashinfer.tllm_enums import ActivationType

from sglang.kernels.ops.moe import q38_router_jit as jit
from sglang.kernels.ops.moe.q38_mk_moe import tensor_map

H, I, E, TOPK, MAX_M = 2560, 192, 512, 10, 7
GRID, CLUSTER, THREADS, SHARED = 140, 2, 256, 216064
logger = logging.getLogger(__name__)
_seen = set()
_MOE_PDL = os.environ.get('SGLANG_Q38C_MOE_PDL', '0') == '1'


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


@cache
def _bundle(device_index, use_pdl=False):
    """Load CPU-built artifacts once; content hashes reject stale source/binaries."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Q38 MK MoE AOT bundle must load during eager graph warmup')
    root = Path(os.environ['SGLANG_Q38C_MOE_PDL_AOT'] if use_pdl
                else os.environ['SGLANG_Q38C_MK_MOE_AOT'])
    manifest = json.loads((root / 'manifest.json').read_text())
    source = Path(__file__).resolve().parents[2] / 'jit/csrc/moe/q38_mk_moe_q35_tp4.cu'
    finalize_source = Path(__file__).with_name('q38_mk_finalize.py')
    assert manifest['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest['finalize_source_sha256'] == hashlib.sha256(finalize_source.read_bytes()).hexdigest()
    if use_pdl:
        pdl_source = Path(__file__).with_name('q38_mk_finalize_pdl.py')
        assert manifest['finalize_pdl_source_sha256'] == hashlib.sha256(pdl_source.read_bytes()).hexdigest()
        assert manifest['pdl_trigger'] == 1 and manifest['finalize']['use_pdl'] is True
    else:
        assert manifest.get('pdl_trigger', 0) == 0
    assert manifest['shape'] == [H, I, E, TOPK, GRID, CLUSTER, SHARED]
    assert torch.cuda.get_device_capability(device_index) == (10, 3)
    for name in ('q35-port.cubin', 'finalize.cubin'):
        assert manifest['sha256'][name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    expert = load(root / 'q35-port.cubin')
    final = manifest['finalize']
    finalize_kernel = jit.load((root / 'finalize.cubin').read_bytes(), final['symbol'],
                               (128, 1, 1), final['shared'])
    return expert, finalize_kernel, final


def _finalize(kernel, metadata, rows, weights, output):
    # Ordinary and PDL cubins preserve the same MTP ordered arithmetic.
    cu = jit.driver()
    values = [ctypes.c_void_p(t.data_ptr()) for t in (rows, weights, output)]
    values += [ctypes.c_void_p(0) for _ in range(metadata['params'] - 3)]
    pointers = (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = 10, rows.shape[0], 1
    config.blockDimX, config.blockDimY, config.blockDimZ = 128, 1, 1
    config.sharedMemBytes = metadata['shared']
    config.hStream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
    if metadata.get('use_pdl', False):
        attr = cu.CUlaunchAttribute()
        attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attr.value.programmaticStreamSerializationAllowed = 1
        config.attrs, config.numAttrs = [attr], 1
    else:
        config.attrs, config.numAttrs = [], 0
    jit.check(cu.cuLaunchKernelEx(config, kernel.function, ctypes.addressof(pointers), 0),
              'Q38 ordered local MoE finalize')


@cache
def _identity(device_index, m):
    # Prepare before the producer, once in eager warmup. Deferred consumers
    # retain FC2 as their last-grid dependency and share this immutable mapping.
    return torch.arange(m * TOPK, dtype=torch.int32, device=f'cuda:{device_index}')


def maybe_run_service(kwargs):
    """Native FI-compatible local output; native shared, AR and HC remain outside.

    Deferred mode exposes complete BF16 expert rows with identity permutation to
    the existing consumer, retaining PDL_TRIGGER=0 in deferred mode. The optional
    ordinary-finalize pair uses trigger1 with its matching waiting consumer.
    Fresh outputs follow the original Q35 ownership through capture and replay.
    """
    xq = kwargs['hidden_states']
    m = xq.shape[0]
    if not 1 <= m <= MAX_M:
        return None
    if (kwargs['intermediate_size'] != I or xq.shape[-1] != H // 2
            or kwargs['num_experts'] != E or kwargs['local_num_experts'] != E
            or kwargs['local_expert_offset'] != 0 or kwargs['top_k'] != TOPK
            or int(kwargs['routing_method_type']) != 4
            or kwargs['activation_type'] != ActivationType.Swiglu
            or kwargs['routing_bias'] is not None or kwargs['per_token_scale'] is not None
            or kwargs['routed_scaling_factor'] not in (None, 1.0)
            or kwargs['gemm1_alpha'] is not None or kwargs['gemm1_beta'] is not None
            or kwargs['gemm1_clamp_limit'] is not None):
        return None
    # Deferred consumers retain the qualified trigger0 kernel and original ABI.
    # Only the paired ordinary finalize has the wait required by an early MKM.
    use_pdl = _MOE_PDL and bool(kwargs['do_finalize'])
    kernel, finalize_kernel, final_metadata = _bundle(xq.device.index, use_pdl)
    layer = dict(gemm1_weights=kwargs['gemm1_weights'], gemm2_weights=kwargs['gemm2_weights'],
                 gemm1_weights_scale=kwargs['gemm1_weights_scale'], gemm2_weights_scale=kwargs['gemm2_weights_scale'],
                 output1_scale_gate_scalar=kwargs['output1_scale_gate_scalar'],
                 output1_scale_scalar=kwargs['output1_scale_scalar'], output2_scale_scalar=kwargs['output2_scale_scalar'])
    rows = torch.empty((m, TOPK, H), dtype=torch.bfloat16, device=xq.device)
    weights = torch.empty((m, TOPK), dtype=torch.bfloat16, device=xq.device)
    ids = torch.empty((m, TOPK), dtype=torch.int32, device=xq.device)
    permutation = _identity(xq.device.index, m) if not kwargs['do_finalize'] else None
    run(kernel, maps(layer['gemm1_weights'], layer['gemm2_weights'], xq),
        kwargs['routing_logits'], weights, ids, kwargs['hidden_states_scale'], layer, rows)
    capturing = torch.cuda.is_current_stream_capturing()
    signature = m, capturing, kwargs['do_finalize']
    if signature not in _seen:
        _seen.add(signature)
        logger.info('Q38C_MK_MOE_ACTIVE M=%d I=192 capture=%s finalize=%s grid=140 cluster=2',
                    m, capturing, kwargs['do_finalize'])
    if not kwargs['do_finalize']:
        return rows.view(m * TOPK, H), weights, permutation
    output = kwargs.get('output')
    if output is None:
        output = torch.empty((m, H), dtype=torch.bfloat16, device=xq.device)
    _finalize(finalize_kernel, final_metadata, rows, weights, output)
    if use_pdl and ('pdl', m, capturing) not in _seen:
        _seen.add(('pdl', m, capturing))
        logger.info('Q38C_MOE_PDL_ACTIVE M=%d capture=%s trigger=1 finalize_wait=1', m, capturing)
    return (output,)
