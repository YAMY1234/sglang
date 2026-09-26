"""CPU checks for NVFP4 KV on the QSA sparse-attention path.

Run in the serving image with CUDA hidden and ``TRITON_INTERPRET=1`` so the Triton
gathers execute on CPU. Prints one JSON report and exits non-zero on any failure.

Bitwise references are torch elementwise: the E2M1 table times the e4m3 block
scale times the fp32 global scale, in that order, as in
``NVFP4KVQuantizeUtil.dequantize``'s elementwise path, which is also compared.

Rounding: with the production global scale 6.0 (and 1.0) every product is exact in
bf16 (at most 2 + 4 + 2 significant bits), so no rounding happens and the gathers
must match the round-to-nearest-even reference bit for bit. For other global
scales the Triton interpreter converts fp32 to bf16 by truncation whatever the
requested mode, so those cases are compared with a truncating reference; the GPU
smoke compares every scale against the RNE reference.
"""

import json
import os
import sys
import traceback
import types

import torch

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
assert not torch.cuda.is_available()

from sglang.srt.layers.attention.qsa.sparse_attn import (
    nvfp4_dequant_torch,
    nvfp4_gather_dequant,
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
)
from sglang.srt.layers.quantization import fp4_kv_cache_quant_method as fp4m
from sglang.srt.layers.quantization.kvfp4_tensor import (
    E2M1_VALUES,
    NVFP4KVQuantizeUtil,
)
from sglang.srt.runtime_context import get_parallel, override_platform

REPORT = {}
GLOBAL_SCALES = (6.0, 1.0, 0.0123456, 3.14159)
EXACT_SCALES = (6.0, 1.0)
HEAD_DIM = 256


def check(fn):
    try:
        info = fn() or {}
        REPORT[fn.__name__] = dict(ok=bool(info.pop("ok", True)), **info)
    except Exception:
        REPORT[fn.__name__] = dict(ok=False, error=traceback.format_exc(limit=8))
    return fn


MISMATCHES = []


def same_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    view = {2: torch.int16, 4: torch.int32, 1: torch.uint8}[a.element_size()]
    equal = a.view(view) == b.view(view)
    if not bool(equal.all()) and len(MISMATCHES) < 12:
        where = (~equal).nonzero()[:4].tolist()
        MISMATCHES.append(dict(count=int((~equal).sum()), of=equal.numel(), first=[
            dict(index=i, got=float(a[tuple(i)]), want=float(b[tuple(i)])) for i in where]))
    return bool(equal.all())


def truncate_bf16(x):
    """fp32 -> bf16 rounding toward zero (the Triton interpreter's conversion)."""
    bits = x.float().contiguous().view(torch.int32) & ~0xFFFF
    return bits.view(torch.float32).to(torch.bfloat16)


def reference_for(data, scales, global_scale):
    """RNE reference for exact scales, truncating one otherwise (see module doc)."""
    if global_scale in EXACT_SCALES:
        return elementwise_reference(data, scales, global_scale)
    return truncate_bf16(elementwise_reference(data, scales, global_scale, torch.float32))


def elementwise_reference(data, scales, global_scale, dtype=torch.bfloat16):
    """Per-element: E2M1[nibble] * float(e4m3 scale) * global, nibble 0 low."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    rows, heads, half = data.shape
    dim = half * 2
    codes = torch.empty(rows, heads, dim, dtype=torch.long)
    codes[..., 0::2] = (data & 0x0F).long()
    codes[..., 1::2] = (data >> 4).long()
    block = torch.arange(dim) // 16
    scale = scales.view(torch.float8_e4m3fn).float()[..., block]
    gs = torch.as_tensor(global_scale, dtype=torch.float32).reshape(())
    return ((lut[codes] * scale) * gs).to(dtype)


def upstream_reference(data, scales, global_scale, dtype=torch.bfloat16):
    """NVFP4KVQuantizeUtil.dequantize's elementwise (non-SM100) branch."""
    rows, heads, half = data.shape
    with override_platform(is_sm90=True, is_sm100=False, is_sm120=False):
        out = NVFP4KVQuantizeUtil.dequantize(
            data.reshape(rows * heads, 1, half),
            scales.view(torch.float8_e4m3fn).reshape(rows * heads, half // 8),
            torch.tensor([global_scale], dtype=torch.float32),
            dtype,
        )
    return out.reshape(rows, heads, half * 2)


def random_layer(rows, heads, dim=HEAD_DIM, seed=0, nan_free=True):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (rows, heads, dim // 2), dtype=torch.uint8, generator=g)
    scales = torch.randint(0, 256, (rows, heads, dim // 16), dtype=torch.uint8, generator=g)
    if nan_free:
        scales = torch.where((scales & 0x7F) == 0x7F, scales - 1, scales)
    return data, scales


def e2m1_encode(x):
    """Round-to-nearest-even onto the E2M1 grid with saturation at 6."""
    grid = torch.tensor(E2M1_VALUES[:8], dtype=torch.float32)
    mag = x.abs().clamp(max=6.0)
    dist = (mag.unsqueeze(-1) - grid).abs()
    best = dist.min(-1, keepdim=True).values
    ties = dist == best
    # Ties go to the even code (mantissa bit 0).
    even = ties & (torch.arange(8) % 2 == 0)
    code = torch.where(even.any(-1), even.float().argmax(-1), ties.float().argmax(-1))
    # Keep the sign bit, including values that round to zero (as cvt does).
    return code | torch.signbit(x).long() << 3


def quantize_reference(x, global_scale):
    """Torch NVFP4 quantizer: e4m3 scale = amax / (6 * global), codes = x / (scale * global)."""
    rows, heads, dim = x.shape
    blocks = x.float().reshape(rows, heads, dim // 16, 16)
    amax = blocks.abs().amax(-1)
    scale = (amax / (6.0 * global_scale)).clamp(max=448.0).to(torch.float8_e4m3fn)
    denom = scale.float().unsqueeze(-1) * global_scale
    scaled = torch.where(denom > 0, blocks / denom, torch.zeros_like(blocks))
    codes = e2m1_encode(scaled).reshape(rows, heads, dim)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed, scale.view(torch.uint8)


@check
def dequant_references_agree():
    results = {}
    for heads in (1, 2):
        data, scales = random_layer(512, heads, seed=heads)
        for gs in GLOBAL_SCALES:
            ref = elementwise_reference(data, scales, gs)
            up = upstream_reference(data, scales, gs)
            fast = nvfp4_dequant_torch(data, scales, torch.tensor([gs]), torch.bfloat16)
            results[f"h{heads}_g{gs}"] = same_bits(ref, up) and same_bits(ref, fast)
    return dict(ok=all(results.values()), cases=results)


@check
def edge_codes():
    # Every e2m1 code in every position under zero / min-subnormal / min-normal /
    # max / sign-bit scales; plus NaN scale codes propagate NaN.
    data = torch.arange(256, dtype=torch.uint8).reshape(2, 1, 128).repeat(8, 1, 1)
    special = torch.tensor([0x00, 0x01, 0x07, 0x08, 0x38, 0x7E, 0x80, 0xFE], dtype=torch.uint8)
    scales = torch.stack([special.repeat_interleave(2).roll(r) for r in range(16)]).reshape(16, 1, 16)
    results = {}
    for gs in (6.0, 1.0):
        ref = elementwise_reference(data, scales, gs)
        out = nvfp4_gather_dequant(data, scales, torch.tensor([gs]), torch.arange(16))
        results[f"g{gs}"] = same_bits(ref, out)
    nan_scales = scales.clone()
    nan_scales[..., 3] = 0x7F
    nan_scales[..., 4] = 0xFF
    ref = elementwise_reference(data, nan_scales, 6.0)
    out = nvfp4_gather_dequant(data, nan_scales, torch.tensor([6.0]), torch.arange(16))
    results["nan_scale_positions"] = bool(torch.equal(ref.isnan(), out.isnan()))
    finite = ~ref.isnan()
    results["nan_scale_rest"] = bool(torch.equal(ref[finite].view(torch.int16), out[finite].view(torch.int16)))
    return dict(ok=all(results.values()), cases=results)


@check
def gather_kernel_bitwise():
    results = {}
    for heads in (1, 2):
        data, scales = random_layer(8192, heads, seed=10 + heads)
        g = torch.Generator().manual_seed(heads)
        locations = torch.randint(0, 8192, (3001,), generator=g)
        locations[:5] = torch.tensor([0, 8191, 8191, 17, 0])
        for gs in GLOBAL_SCALES:
            ref = reference_for(data, scales, gs)[locations]
            out = nvfp4_gather_dequant(data, scales.view(torch.float8_e4m3fn), torch.tensor([gs]), locations)
            results[f"h{heads}_g{gs}"] = same_bits(ref, out) and out.dtype == torch.bfloat16 and tuple(out.shape) == (3001, heads, HEAD_DIM)
    empty = nvfp4_gather_dequant(data, scales, torch.tensor([1.0]), torch.empty(0, dtype=torch.long))
    results["empty"] = tuple(empty.shape) == (0, 2, HEAD_DIM)
    return dict(ok=all(results.values()), cases=results)


def _compact_case(k_pool, v_pool, nvfp4, expected_k, expected_v, heads, dim, zero_fill, mapped, seed):
    batch, topk, page, pool_rows = 3, 2051, 64, k_pool.shape[0]
    g = torch.Generator().manual_seed(seed)
    seq_lens = torch.tensor([733, 109, 2500], dtype=torch.int32)
    req_to_token = torch.randperm(pool_rows, generator=g)[: batch * 2600].reshape(batch, 2600).to(torch.int32)
    req_indices = torch.tensor([2, 0, 1], dtype=torch.int32)
    indices = torch.full((batch, topk), -1, dtype=torch.int32)
    for b in range(batch):
        n = min(int(seq_lens[b]), topk)
        indices[b, :n] = torch.randperm(int(seq_lens[b]), generator=g)[:n].to(torch.int32)
    page_mapping = None
    if mapped:
        page_mapping = torch.randperm(pool_rows // page, generator=g).to(torch.int32).reshape(-1, 1)
    if zero_fill:
        stride = (topk + page - 1) // page * page
        cu = torch.arange(batch + 1, dtype=torch.int32) * stride
        cols = stride
    else:
        counts = torch.empty(batch, dtype=torch.int32)
        cu = torch.empty(batch + 1, dtype=torch.int32)
        qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, counts, cu, batch, topk)
        cols = topk
    rows = int(cu[-1])
    out_k = torch.full((rows, heads, dim), float("nan"), dtype=torch.bfloat16)
    out_v = out_k.clone()
    qwen_sparse_kv_extraction_compact_triton(
        k_pool, v_pool, req_to_token, req_indices, indices, seq_lens, cu, out_k, out_v,
        batch, topk, zero_fill_cols=stride if zero_fill else 0,
        page_mapping=page_mapping, nvfp4_scales=nvfp4,
    )
    ok = True
    for b in range(batch):
        n = min(int(seq_lens[b]), topk)
        slots = req_to_token[req_indices[b].long(), indices[b, :n].long()].long()
        if mapped:
            slots = page_mapping[slots // page, 0].long() * page + slots % page
        start = int(cu[b])
        ok &= same_bits(out_k[start : start + n], expected_k[slots])
        ok &= same_bits(out_v[start : start + n], expected_v[slots])
        if zero_fill:
            tail = out_k[start + n : start + cols], out_v[start + n : start + cols]
            ok &= all(bool((t.view(torch.int16) == 0).all()) for t in tail)
    return ok


@check
def compact_kernel_nvfp4_bitwise():
    results = {}
    heads = 2
    k_data, k_sc = random_layer(8192, heads, seed=21)
    v_data, v_sc = random_layer(8192, heads, seed=22)
    for gs_k, gs_v in ((6.0, 6.0), (1.0, 0.0123456)):
        exp_k = reference_for(k_data, k_sc, gs_k)
        exp_v = reference_for(v_data, v_sc, gs_v)
        nvfp4 = (k_sc.view(torch.float8_e4m3fn), v_sc.view(torch.float8_e4m3fn), torch.tensor([gs_k]), torch.tensor([gs_v]))
        for zero_fill in (False, True):
            for mapped in (False, True):
                name = f"g{gs_k}/{gs_v}_{'strided' if zero_fill else 'compact'}_{'mapped' if mapped else 'direct'}"
                results[name] = _compact_case(k_data, v_data, nvfp4, exp_k, exp_v, heads, HEAD_DIM, zero_fill, mapped, seed=len(results))
    return dict(ok=all(results.values()), cases=results)


@check
def compact_kernel_plain_unchanged():
    # bf16 / fp8 pools still take the plain load-and-cast branch.
    results = {}
    heads = 2
    g = torch.Generator().manual_seed(5)
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        k_pool = torch.randn(8192, heads, HEAD_DIM, generator=g).to(dtype)
        v_pool = torch.randn(8192, heads, HEAD_DIM, generator=g).to(dtype)
        for zero_fill in (False, True):
            name = f"{str(dtype).split('.')[-1]}_{'strided' if zero_fill else 'compact'}"
            results[name] = _compact_case(k_pool, v_pool, None, k_pool.to(torch.bfloat16), v_pool.to(torch.bfloat16), heads, HEAD_DIM, zero_fill, False, seed=40 + len(results))
    return dict(ok=all(results.values()), cases=results)


def _layers_stub(global_ids):
    layers = []
    for i in range(48):
        layer = types.SimpleNamespace()
        if i in global_ids:
            layer.attn = types.SimpleNamespace(layer_id=i, k_scale=None, v_scale=None)
        else:
            layer.linear_attn = object()
        layers.append(layer)
    return types.SimpleNamespace(layers=layers)


FULL_IDS = list(range(3, 48, 4))


@check
def method_registry_and_buffers():
    m = fp4m.NVFP4QSAKVCacheMethod(num_layers=12, device="cpu")
    base = fp4m.NVFP4KVCacheMethod(num_layers=12, device="cpu")
    kinds = fp4m.KVCacheAttentionAccessKind
    res = {
        "name": m.name == "nvfp4_qsa",
        "no_workspace": (not m.needs_dequant_workspace()) and m.dequant_workspace_dtype() is None,
        "no_plain_dequant_read": not m.needs_plain_kv_dequant_read(),
        "storage_uint8": m.kv_storage_dtype() == torch.uint8,
        "qsa_prefill": m.resolve_attention_access("prefill", "qsa_sparse").kind == kinds.GATHER_DEQUANT,
        "qsa_decode": m.resolve_attention_access("decode", "qsa_sparse").kind == kinds.GATHER_DEQUANT,
        "qsa_not_flashinfer": m.resolve_attention_access("prefill", "flashinfer") is None,
        "qsa_not_trtllm": m.resolve_attention_access("decode", "trtllm_mha") is None,
        "base_unchanged_prefill": base.resolve_attention_access("prefill", "flashinfer").kind == kinds.DEQUANT_WORKSPACE,
        "base_unchanged_decode": base.resolve_attention_access("decode", "trtllm_mha").kind == kinds.NATIVE_FP4,
        "base_no_qsa": base.resolve_attention_access("decode", "qsa_sparse") is None,
        "base_workspace": base.needs_dequant_workspace(),
        "cli_registry_unchanged": sorted(fp4m.KV_CACHE_QUANT_REGISTRY) == ["cpu_fp8_e4m3", "fp4_mx_block16", "nvfp4"],
        "cell_size_target": m.compute_cell_size(1, HEAD_DIM, 12, 1) == 3456,
        "cell_size_base": base.compute_cell_size(1, HEAD_DIM, 12, 1) == 3456 + 512,
    }
    bufs = m.create_buffers(128, 1, HEAD_DIM, 12, "cpu")
    res["buffers"] = (
        len(bufs["k_buffer"]) == 12
        and tuple(bufs["k_buffer"][0].shape) == (128, 1, 128)
        and tuple(bufs["k_scale_buffer"][0].shape) == (128, 1, 16)
        and bufs["k_buffer"][0].dtype == torch.uint8
        and bufs["dq_k_buffer"] is None
        and bufs["dq_v_buffer"] is None
    )
    with override_platform(is_sm100=True, is_sm120=False):
        m.load_scales_from_model(_layers_stub(FULL_IDS))
    res["global_scales"] = (
        m.k_scales_gpu.shape[0] == 48
        and all(float(m.k_scales_gpu[i]) == 6.0 and float(m.v_scales_gpu[i]) == 6.0 for i in FULL_IDS)
    )
    return dict(ok=all(res.values()), cases=res)


class _Config(types.SimpleNamespace):
    pass


def _flashnext_kvc():
    text = _Config(
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128,
        indexer_budget=2048, indexer_compress_ratio=4,
        model_type="qwen4_exp_text", architectures=["Qwen4ExpForConditionalGeneration"],
    )
    hf = _Config(text_config=text, model_type="qwen4_exp", architectures=["Qwen4ExpForConditionalGeneration"])
    model_config = types.SimpleNamespace(
        hf_config=hf, head_dim=HEAD_DIM, v_head_dim=HEAD_DIM,
        get_num_kv_heads=lambda tp, dcp=1: 2 // tp,
    )
    return types.SimpleNamespace(
        # enable_hisparse only skips the GLM DSA layer-split lookup here.
        model_config=model_config, server_args=types.SimpleNamespace(enable_hisparse=True),
        use_mla_backend=False, is_draft_worker=False,
    )


@check
def pool_cell_size():
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

    from unittest.mock import patch

    kvc = _flashnext_kvc()
    res = {}
    # is_float4_e2m1fn_x2 also requires CUDA; the formula is what is under test.
    fp4 = lambda dtype: dtype == torch.float4_e2m1fn_x2
    with get_parallel().override(attn_tp_size=2, attn_dcp_size=1), patch(
        "sglang.srt.model_executor.pool_configurator.is_float4_e2m1fn_x2", fp4
    ):
        for label, dtype, dtype_str, want in (
            ("bf16", torch.bfloat16, "bfloat16", 13056),
            ("fp8", torch.float8_e4m3fn, "fp8_e4m3", 6912),
            ("nvfp4", torch.float4_e2m1fn_x2, "nvfp4", 4224),
            ("fp4_mx_block16", torch.float4_e2m1fn_x2, "fp4_mx_block16", 4736),
        ):
            conf = DefaultPoolConfigurator.__new__(DefaultPoolConfigurator)
            conf.kv_cache_dtype_str = dtype_str
            kvc.kv_cache_dtype = dtype
            got = conf._compute_cell_size(kvc, 12)
            res[label] = dict(bytes=got, want=want, ok=got == want)
            # Same expression as the EAGLE/NEXTN draft scaling in DefaultPoolConfigurator.
            res[label + "_mtp_on"] = int(got * (1 + 1 / 12))
    ok = all(v["ok"] for v in res.values() if isinstance(v, dict))
    ok &= res["nvfp4_mtp_on"] == 4576
    return dict(ok=ok, cases=res)


def _patched_quantize(tensor, global_scale):
    gs = float(global_scale.reshape(-1)[0])
    b, m, n = tensor.shape
    packed, scale = quantize_reference(tensor.reshape(b * m, 1, n), gs)
    return packed.reshape(b, m, n // 2), scale.view(torch.float8_e4m3fn).reshape(b, m, n // 16), global_scale


@check
def pool_write_and_backend_reads():
    """Real MHA pool + hybrid wrapper on CPU: write through the backend helper with
    the reference quantizer, read back with the backend's gather."""
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    res = {}
    method = fp4m.NVFP4QSAKVCacheMethod(num_layers=12, device="cpu")
    with override_platform(is_sm100=True, is_sm120=False):
        method.load_scales_from_model(_layers_stub(FULL_IDS))
    pool = HybridLinearKVPool(
        size=1024, dtype=torch.float4_e2m1fn_x2, page_size=64, head_num=1, head_dim=HEAD_DIM,
        full_attention_layer_ids=FULL_IDS, device="cpu", mamba_pool=None, quant_method=method,
    )
    res["full_pool_quantized"] = pool.full_kv_pool.is_quantized_kv_cache
    res["no_workspace"] = pool.full_kv_pool.dq_k_buffer is None
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = pool
    backend._kv_quant = QwenSparseAttnBackend._resolve_kv_quant(pool)
    res["resolved"] = backend._kv_quant is method
    g = torch.Generator().manual_seed(7)
    original = NVFP4KVQuantizeUtil.quantize
    NVFP4KVQuantizeUtil.quantize = staticmethod(_patched_quantize)
    try:
        for layer_id in (3, 47):
            layer = types.SimpleNamespace(layer_id=layer_id)
            loc = torch.randperm(1024, generator=g)[:300]
            k = (torch.randn(300, 1, HEAD_DIM, generator=g) * 3).to(torch.bfloat16)
            v = (torch.randn(300, 1, HEAD_DIM, generator=g) * 0.05).to(torch.bfloat16)
            backend._write_kv(layer, loc, k, v)
            k_buf, v_buf, nvfp4 = backend._kv_buffers(pool, layer)
            want_k = quantize_reference(k, 6.0)
            want_v = quantize_reference(v, 6.0)
            key = f"layer{layer_id}"
            res[key + "_codes"] = same_bits(k_buf[loc], want_k[0]) and same_bits(v_buf[loc], want_v[0])
            res[key + "_scales"] = same_bits(nvfp4[0].view(torch.uint8)[loc], want_k[1]) and same_bits(nvfp4[1].view(torch.uint8)[loc], want_v[1])
            untouched = torch.ones(k_buf.shape[0], dtype=torch.bool)
            untouched[loc] = False
            res[key + "_untouched_zero"] = bool((k_buf[untouched] == 0).all() and (nvfp4[0].view(torch.uint8)[untouched] == 0).all())
            res[key + "_global"] = float(nvfp4[2]) == 6.0 and float(nvfp4[3]) == 6.0
            gathered = nvfp4_gather_dequant(k_buf, nvfp4[0], nvfp4[2], loc)
            res[key + "_gather"] = same_bits(gathered, elementwise_reference(want_k[0], want_k[1], 6.0))
            dk, dv = backend._dequant_layer(k_buf, v_buf, nvfp4, torch.bfloat16)
            res[key + "_cpu_path"] = same_bits(dv[loc], elementwise_reference(want_v[0], want_v[1], 6.0))
            res[key + "_raw_shapes"] = tuple(k_buf.shape) == (1024 + 64, 1, 128) and tuple(nvfp4[0].shape) == (1024 + 64, 1, 16)
    finally:
        NVFP4KVQuantizeUtil.quantize = original
    plain = HybridLinearKVPool(
        size=256, dtype=torch.bfloat16, page_size=64, head_num=1, head_dim=HEAD_DIM,
        full_attention_layer_ids=FULL_IDS, device="cpu", mamba_pool=None,
    )
    res["plain_pool_unquantized"] = QwenSparseAttnBackend._resolve_kv_quant(plain) is None
    return dict(ok=all(res.values()), cases=res)


@check
def discarded_full_backend_not_built():
    """QSA + gather-dequant KV: no trtllm_mha backend is built only to be replaced."""
    from unittest.mock import patch

    from sglang.srt.model_executor.model_runner_components import attention_backend_setup as setup

    kvc = _flashnext_kvc()
    res = {}
    for label, method, want in (
        ("nvfp4_qsa", fp4m.NVFP4QSAKVCacheMethod(num_layers=12, device="cpu"), True),
        ("nvfp4", fp4m.NVFP4KVCacheMethod(num_layers=12, device="cpu"), False),
        ("bf16", fp4m.UnquantizedKVCacheMethod(), False),
    ):
        pool = types.SimpleNamespace(get_kv_cache_quant_method=lambda m=method: m)
        runner = types.SimpleNamespace(model_config=kvc.model_config, token_to_kv_pool=pool)
        res[label] = setup._qsa_gather_dequant_kv(runner) == want
        built = []
        with patch.object(setup, "attn_backend_wrapper", lambda r, full: ("wrapped", full)), patch.object(
            setup, "_build_full_attention_backend_from_str", lambda **kw: built.append(kw) or "trtllm"
        ):
            out = setup._build_backend_from_str(
                model_runner=runner, backend_str="trtllm_mha", init_new_workspace=False
            )
        res[label + "_wrapper_arg"] = out == ("wrapped", None if want else "trtllm") and bool(built) != want
    return dict(ok=all(res.values()), cases=res)


@check
def quantize_error_profile():
    """Informative: reference-quantizer error by global scale on K-like and V-like
    magnitudes (the gate decides; this records where 4 bit loses precision)."""
    g = torch.Generator().manual_seed(11)
    out = {}
    for label, sigma in (("k_like_sigma3", 3.0), ("v_like_sigma0.05", 0.05), ("v_like_sigma0.005", 0.005)):
        x = (torch.randn(2048, 1, HEAD_DIM, generator=g) * sigma).to(torch.bfloat16)
        for gs in (6.0, 1.0):
            packed, scale = quantize_reference(x, gs)
            y = elementwise_reference(packed, scale, gs).float()
            err = (y - x.float()).norm() / x.float().norm()
            sc = scale.view(torch.float8_e4m3fn).float()
            out[f"{label}_g{gs}"] = dict(
                rel_rms_error=round(float(err), 5),
                zero_scale_blocks=round(float((sc == 0).float().mean()), 5),
                subnormal_scale_blocks=round(float(((sc > 0) & (sc < 2**-6)).float().mean()), 5),
            )
    return dict(ok=True, cases=out)


if __name__ == "__main__":
    ok = all(entry["ok"] for entry in REPORT.values())
    print(json.dumps(dict(ok=ok, torch=torch.__version__, checks=REPORT, mismatches=MISMATCHES),
                     indent=1, sort_keys=True))
    sys.exit(0 if ok else 1)
