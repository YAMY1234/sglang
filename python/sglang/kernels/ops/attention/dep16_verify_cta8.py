# SPDX-License-Identifier: Apache-2.0
"""Opt-in native FlashInfer q7 launcher with eight KV CTAs on SM103.

The vendored host headers change launch geometry and scratch layout only. The
device kernels, scales, reduction implementation and counter lifecycle remain
FlashInfer's. Prepare the module before CUDA-graph capture.
"""

import hashlib
from dataclasses import replace
from pathlib import Path

import flashinfer
import flashinfer.decode as fd
import torch
from flashinfer.jit import env as jit_env

from sglang.kernels.jit.utils import cache_once
from sglang.srt.utils.common import print_info_once

_STOCK_HEADER_SHA256 = (
    "7f2ea5116d4929cfb5350614671c7524da29df49db969f668d2e672bc5c0369c"
)


def supports_dep16_verify_cta8(device: torch.device) -> bool:
    """The host adapter is bound to the qualified FlashInfer header ABI."""
    header = (
        jit_env.FLASHINFER_CSRC_DIR.parent
        / "include/flashinfer/trtllm/fmha/fmhaKernels.cuh"
    )
    return (
        flashinfer.__version__ == "0.7.1"
        and torch.cuda.get_device_capability(device) == (10, 3)
        and hashlib.sha256(header.read_bytes()).hexdigest() == _STOCK_HEADER_SHA256
    )


class _Dep16VerifyModule(fd.TrtllmGenDecodeModule):
    def __init__(self, device: torch.device):
        self._sm_count = fd.get_device_sm_count(device)
        includes = (
            Path(__file__).parents[2] / "jit/csrc/attention/dep16_verify_cta8/include"
        )
        stock = fd.gen_trtllm_gen_fmha_module()
        digest = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(includes.rglob("*.cuh")))
        ).hexdigest()
        # -I directories precede the original -isystem FlashInfer include root.
        # The original host CUDA sources and cubin inventory are reused intact.
        self._mod = replace(
            stock,
            name=f"sglang_dep16_verify_cta8_{digest[:16]}",
            extra_include_dirs=[
                includes,
                jit_env.FLASHINFER_CSRC_DIR.parent / "include/flashinfer/trtllm/fmha",
                *(stock.extra_include_dirs or []),
            ],
        )
        self._op = self._mod.build_and_load()
        for library_path in self._mod.get_library_paths():
            fd.setup_cubin_loader(library_path)


def prepare_dep16_verify_cta8(device: torch.device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _prepare_dep16_verify_cta8(index)


@cache_once
def _prepare_dep16_verify_cta8(index: int):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare DEP16 verify CTA8 before CUDA-graph capture")
    device = torch.device("cuda", index)
    if not supports_dep16_verify_cta8(device):
        raise RuntimeError(
            "DEP16 verify CTA8 requires the qualified SM103/FlashInfer0.7.1 ABI"
        )
    return _Dep16VerifyModule(device)


def trtllm_dep16_verify_cta8(
    *,
    query: torch.Tensor,
    kv_cache,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale=1.0,
    bmm2_scale=1.0,
    window_left=-1,
    sinks=None,
    out=None,
    out_dtype=None,
    kv_cache_sf=None,
    multi_ctas_kv_counter_buffer=None,
    q_len_per_req=1,
    mask=None,
    skip_softmax_threshold_scale_factor=None,
):
    """Run the qualified q7 shape; unsupported shapes keep the original API."""
    resolved_out_dtype = out_dtype or (out.dtype if out is not None else query.dtype)
    batch = seq_lens.shape[0]
    eligible = (
        q_len_per_req == 7
        and query.ndim == 3
        and query.shape == (batch * 7, 32, 256)
        and 10 <= batch <= 16
        and query.dtype == torch.float8_e4m3fn
        and isinstance(kv_cache, tuple)
        and len(kv_cache) == 2
        and all(
            k.ndim == 4
            and k.shape[1:] == (2, 64, 256)
            and k.dtype == torch.float8_e4m3fn
            for k in kv_cache
        )
        and resolved_out_dtype == torch.bfloat16
        and window_left == -1
        and mask is None
        and kv_cache_sf is None
        and max_seq_len <= 262144
        and workspace_buffer.numel() * workspace_buffer.element_size() >= 512 * 1024**2
        and multi_ctas_kv_counter_buffer is not None
    )
    if not eligible:
        return fd.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=window_left,
            sinks=sinks,
            out=out,
            out_dtype=out_dtype,
            kv_cache_sf=kv_cache_sf,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
            q_len_per_req=q_len_per_req,
            mask=mask,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )
    module = prepare_dep16_verify_cta8(query.device)
    print_info_once(f"E2X_VFMHA_CTA8 engaged: target-verify batch={batch}")
    if out is None:
        out = torch.empty_like(query, dtype=torch.bfloat16)
    fd.check_shape_dtype_device(out, query.shape, torch.bfloat16, query.device, "out")
    fd._check_block_tables_shape(
        block_tables, True, block_sparse=False, num_kv_heads=2, batch_size=batch
    )
    multi_ctas_kv_counter_buffer = fd._resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
        multi_ctas_kv_counter_buffer, batch, 32, module._sm_count, query.device
    )
    return module._paged_run(
        query.unflatten(0, (batch, 7)),
        kv_cache[0],
        kv_cache[1],
        workspace_buffer,
        multi_ctas_kv_counter_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        bmm1_scale,
        bmm2_scale,
        workspace_buffer.numel() * workspace_buffer.element_size(),
        window_left=window_left,
        enable_pdl=fd.device_support_pdl(query.device),
        out=out,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
    )
