"""Actual C03 producer/collector/facade; CUDA eligibility and launches are doubles."""

import sys
from dataclasses import fields, replace
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from capability_fixtures import mtp_spec
from test_mtp_capture_seed_producer import CudaMetadata
from test_mtp_capture_seed_producer import context as _context_fixture
from test_mtp_capture_seed_producer import setup as seed_setup
from test_mtp_capture_seed_producer import tensor

context = _context_fixture
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpDeferredKvIO,
    NumericOffer,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.srt import runtime_context
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.kernel_plugins.nvidia import mtp_deferred_kv as policy
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.layers import radix_attention
from sglang.srt.layers.attention import qwen_sparse_attn_backend as impl
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def setup(map_dtype=torch.int32, cache_dtype=torch.bfloat16, heads=1):
    idx, model, runner, backend, state, batch, *_ = seed_setup()
    state.indices[0].copy_(tensor(torch.arange(2055).expand(8, -1)))
    state.captured_len.fill_(100)
    state._tail_offsets = tensor(torch.arange(4), torch.int64)
    table = tensor(torch.arange(8192).remainder(64).expand(7, -1).clone(), map_dtype)
    kc = tensor(torch.arange(64 * heads * 256).reshape(64, heads, 256), cache_dtype)
    vc = kc.clone()
    backend._fused_kv_pool_eligible = True
    backend.token_to_kv_pool = NS(
        layer_router=NS(read=lambda _: NS(key=lambda: kc, value=lambda: vc))
    )
    backend.req_to_token_pool = NS(req_to_token=table)
    backend._cuda_graph_max_tokens = 1
    backend._fa2_scratch = {
        (1, 256, torch.bfloat16, kc.device): (
            tensor(torch.zeros((2112, 1, 256)), torch.bfloat16),
            tensor(torch.zeros((2112, 1, 256)), torch.bfloat16),
        ),
    }
    backend._trtllm_sparse_tables = {}
    backend._trtllm_workspace = torch.zeros(1, dtype=torch.uint8)
    backend._trtllm_counters = {(1, 6, kc.device): torch.zeros(1, dtype=torch.uint8)}
    backend.compress_ratio, backend.qsa_profile = 4, NS(block_topk=512)
    positions = tensor([102])
    md = NS(
        is_cuda_graph=True,
        fa2_valid_counts=tensor([-9]),
        row_req_pool_indices=tensor([2]),
        sequence_lengths=tensor([4097]),
        indexer_metadata=NS(decode_logical_positions=positions),
    )
    backend.forward_metadata = md
    batch.forward_mode = ForwardMode.DECODE
    batch.out_cache_loc = tensor([5], torch.int64)
    runner.model_config.hf_text_config = NS(
        hidden_size=2560, hc_count=4, num_attention_heads=24
    )
    runner.draft_attn_backend = object.__new__(impl.QwenSparseMultiStepDraftBackend)
    runner.draft_attn_backend.attn_backends = [backend]
    layer = NS(layer_id=3, tp_q_head_num=6, head_dim=256, scaling=0.0625)
    q = tensor(torch.zeros((1, 1536)), torch.bfloat16)
    k = tensor(torch.full((1, 256), 3), torch.bfloat16)
    # Real singleton projection stride is retained until the backend reshape.
    v = tensor(torch.zeros((1, 3584)), torch.bfloat16)[:, -256:]
    v.fill_(7)
    return idx, model, runner, backend, state, batch, layer, q, k, v, md


@pytest.fixture
def enabled(context, monkeypatch):
    monkeypatch.setenv("SGLANG_Q38_LOOKUP_NATIVE", "1")
    monkeypatch.setenv("SGLANG_Q38_MK_MTP", "1")
    monkeypatch.setattr(
        runtime_context,
        "get_spec",
        lambda: NS(
            speculative_num_steps=3,
            speculative_eagle_topk=1,
        ),
    )
    monkeypatch.setattr(
        impl,
        "_resolve_trtllm_sparse_decode",
        lambda: lambda **kwargs: torch.zeros((1, 6, 256)),
    )
    # Only the external FlashInfer boundary is doubled; the backend function runs.
    fi, utils = ModuleType("flashinfer"), ModuleType("flashinfer.utils")
    utils.get_trtllm_gen_multi_ctas_kv_counter_bytes = lambda *_: 1
    fi.utils = utils
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setitem(sys.modules, "flashinfer.utils", utils)
    return context


def prepare(idx, model, runner, monkeypatch):
    assert numeric_binding.collect_numeric_signatures((idx,), (1,), 4) == ()
    owner = binding.MegakernelBinding(model=model)
    owner._numeric_spec = mtp_spec()
    monkeypatch.setattr(owner, "_prepare_hybrid", lambda _: None)
    (selected,) = owner.preparation_view(runner).numeric_signatures
    assert selected.operation == O.MTP_DEFERRED_KV_PREPARE
    return selected


def raw_double(io):
    """Original lookup/packer masks on CPU; no claim of Triton arithmetic testing."""
    req = int(io.lookup_requests[0])
    positions = io.indices_pool[req].clone().long()
    tail = int(io.captured_lengths[req]) + torch.arange(4)
    positions[-4:] = torch.where(tail <= int(io.lookup_positions[0]), tail, -1)
    valid = (positions >= 0) & (positions < int(io.sequence_lengths[0]))
    io.counts_out[0] = valid.sum()
    loc = int(io.locations[0])
    if loc != 0:
        io.k_cache[loc].copy_(io.new_k[0])
        io.v_cache[loc].copy_(io.new_v[0])
    io.packed_k_out.zero_()
    io.packed_v_out.zero_()
    if loc != 0:
        cols = torch.nonzero(valid).flatten()
        slots = io.req_to_token[int(io.row_requests[0]), positions[valid]].long()
        io.packed_k_out.index_copy_(0, cols, io.k_cache[slots])
        io.packed_v_out.index_copy_(0, cols, io.v_cache[slots])


@pytest.mark.parametrize("map_dtype", [torch.int32, torch.int64])
def test_actual_post_pool_graph_layout_not_guessed(enabled, monkeypatch, map_dtype):
    idx, model, runner, backend, *_ = setup(map_dtype)
    selected = prepare(idx, model, runner, monkeypatch)
    assert selected.index_pool_rows == 8 and selected.request_pool_rows == 7
    assert selected.map_columns == selected.map_row_stride == 8192
    assert selected.cache_slots == 64
    assert selected.new_k_row_stride == selected.new_v_row_stride == 256
    assert selected.metadata_dtype_mask == (17 if map_dtype == torch.int32 else 25)
    assert backend._numeric_mtp_deferred_signatures[3] == selected
    assert numeric_binding.collect_numeric_signatures(
        (NS(numeric_signatures=lambda *_: (selected,)),),
        (1,),
        4,
    ) == (selected,)


@pytest.mark.parametrize(
    "reason",
    [
        "lookupoff",
        "mkoff",
        "tp1",
        "target",
        "wide",
        "heads2",
        "fp8",
        "steps",
        "topk",
        "zero_map_stride",
    ],
)
def test_unqualified_preparation_stays_stock(enabled, monkeypatch, reason):
    idx, model, runner, backend, *_ = setup(
        cache_dtype=torch.float8_e4m3fn if reason == "fp8" else torch.bfloat16,
        heads=2 if reason == "heads2" else 1,
    )
    if reason == "lookupoff":
        monkeypatch.setenv("SGLANG_Q38_LOOKUP_NATIVE", "0")
    if reason == "mkoff":
        monkeypatch.setenv("SGLANG_Q38_MK_MTP", "0")
    if reason == "tp1":
        monkeypatch.setattr(runtime_context, "get_parallel", lambda: NS(tp_size=1))
    if reason == "target":
        runner.is_draft_worker = False
    if reason == "steps":
        monkeypatch.setattr(
            runtime_context,
            "get_spec",
            lambda: NS(speculative_num_steps=4, speculative_eagle_topk=1),
        )
    if reason == "topk":
        monkeypatch.setattr(
            runtime_context,
            "get_spec",
            lambda: NS(speculative_num_steps=3, speculative_eagle_topk=2),
        )
    if reason == "zero_map_stride":
        backend.req_to_token_pool.req_to_token = backend.req_to_token_pool.req_to_token[
            :1
        ].expand(7, -1)
    numeric_binding.collect_numeric_signatures(
        (idx,), (2,) if reason == "wide" else (1,), 4
    )
    assert idx.numeric_state_preparation_signatures(runner) == ()
    assert not getattr(backend, "_numeric_mtp_deferred_signatures", {})


@pytest.mark.parametrize("loc", [0, 5])
def test_real_lookup_facade_packer_then_attention_once(enabled, monkeypatch, loc):
    idx, model, runner, backend, state, fb, layer, q, k, v, md = setup()
    selected = prepare(idx, model, runner, monkeypatch)
    calls = []

    def provider(io):
        calls.append(io)
        raw_double(io)

    enabled.register(NumericOffer(selected, provider))
    indices = backend.lookup_mtp_sparse_indices(fb, 3)
    assert isinstance(indices, policy.DeferredMtpSparseIndices)
    assert indices.indices.data_ptr() == state.indices[0].data_ptr()
    fb.out_cache_loc.fill_(loc)
    output = backend.forward_decode(q, k, v, layer, fb, topk_indices=indices)
    assert output.shape == (1, 1536) and len(calls) == 1
    io = calls[0]
    assert isinstance(io, MtpDeferredKvIO)
    assert io.new_k.data_ptr() == k.data_ptr() and io.new_v.data_ptr() == v.data_ptr()
    assert io.new_k.stride() == io.new_v.stride() == (256, 256, 1)
    assert io.counts_out is md.fa2_valid_counts and io.counts_out[0] == 2054
    if loc == 0:
        assert torch.count_nonzero(io.packed_k_out) == 0
    else:
        assert torch.equal(io.k_cache[loc], io.new_k[0])
        assert torch.equal(io.v_cache[loc], io.new_v[0])
    assert state.captured_len[0, 2] == 100


@pytest.mark.parametrize(
    "reason", ["nooffer", "masteroff", "badmetadata", "saveoff", "badnewkv"]
)
def test_missing_or_incomplete_route_materializes_then_stock_once(
    enabled, monkeypatch, reason
):
    idx, model, runner, backend, _, fb, layer, q, k, v, md = setup()
    selected = prepare(idx, model, runner, monkeypatch)
    calls = []
    enabled.register(NumericOffer(selected, lambda io: calls.append(io)))
    indices = backend.lookup_mtp_sparse_indices(fb, 3)
    if reason == "nooffer":
        monkeypatch.setattr(numeric_binding, "numeric_registry", type(enabled)())
    if reason == "masteroff":
        monkeypatch.setattr(binding, "megakernel_enabled", lambda: False)
    if reason == "badmetadata":
        md.row_req_pool_indices = md.row_req_pool_indices.long()
    if reason == "badnewkv":
        k = k.float()
    seen = []

    def fallback(*args, **kwargs):
        seen.append(args[5] if len(args) > 4 else args[3])
        return torch.zeros((1, 1536))

    monkeypatch.setattr(backend, "_try_fused_kv_attention", fallback)
    monkeypatch.setattr(backend, "_forward_paged_attention", fallback)
    output = backend.forward_decode(
        q, k, v, layer, fb, topk_indices=indices, save_kv_cache=reason != "saveoff"
    )
    assert output.shape == (1, 1536)
    assert not calls and len(seen) == 1 and isinstance(seen[0], torch.Tensor)
    assert torch.equal(seen[0], indices.materialize())


def test_lookup_nooffer_returns_original_tensor(enabled, monkeypatch):
    idx, model, runner, backend, state, fb, *_ = setup()
    prepare(idx, model, runner, monkeypatch)
    actual = backend.lookup_mtp_sparse_indices(fb, 3)
    assert isinstance(actual, torch.Tensor)
    assert torch.equal(
        actual,
        state.lookup(
            fb.req_pool_indices,
            backend.forward_metadata.indexer_metadata.decode_logical_positions,
            3,
        ),
    )


def test_selected_write_error_never_materializes_or_retries(enabled, monkeypatch):
    idx, model, runner, backend, _, fb, layer, q, k, v, _ = setup()
    selected = prepare(idx, model, runner, monkeypatch)

    def provider(io):
        io.k_cache[5, 0, 0] = 11
        raise RuntimeError("selected KV write")

    enabled.register(NumericOffer(selected, provider))
    indices = backend.lookup_mtp_sparse_indices(fb, 3)
    monkeypatch.setattr(
        policy.DeferredMtpSparseIndices,
        "materialize",
        lambda _: pytest.fail("materialized after write"),
    )
    monkeypatch.setattr(
        backend, "_try_fused_kv_attention", lambda *_args, **_kw: pytest.fail("retried")
    )
    with pytest.raises(RuntimeError, match="selected KV write"):
        backend.forward_decode(q, k, v, layer, fb, topk_indices=indices)


def test_bool_signature_rejected_before_registry(enabled, monkeypatch):
    idx, model, runner, *_ = setup()
    selected = prepare(idx, model, runner, monkeypatch)
    with pytest.raises(TypeError, match="cannot contain bool"):
        numeric_binding.collect_numeric_signatures(
            (NS(numeric_signatures=lambda *_: (replace(selected, rows=True),)),),
            (1,),
            4,
        )


@pytest.mark.parametrize("field", [field.name for field in fields(MtpDeferredKvIO)])
def test_all_fifteen_views_checked_before_any_write(enabled, monkeypatch, field):
    idx, model, runner, backend, _, fb, layer, q, k, v, _ = setup()
    selected = prepare(idx, model, runner, monkeypatch)
    enabled.register(NumericOffer(selected, lambda _: pytest.fail("premature write")))
    indices = backend.lookup_mtp_sparse_indices(fb, 3)
    run, io, _, _ = policy.bind_prepare(backend, q, k, v, layer, fb, indices)
    assert run is not None and policy.signature(io, tail_width=4) == selected
    malformed = replace(io, **{field: getattr(io, field).float()})
    assert policy.signature(malformed, tail_width=selected.tail_width) != selected


def test_source_borrowed_lifetimes_record_all_five_views(enabled, monkeypatch):
    idx, model, runner, backend, _, fb, *_ = setup()
    selected = prepare(idx, model, runner, monkeypatch)
    enabled.register(NumericOffer(selected, lambda _: None))
    indices = backend.lookup_mtp_sparse_indices(fb, 3)
    observed = []
    stream = object()
    monkeypatch.setattr(
        CudaMetadata,
        "record_stream",
        lambda t, actual: observed.append((id(t), actual)),
    )
    indices.record_stream(stream)
    assert observed == [
        (id(t), stream)
        for t in (
            indices.indices,
            indices.captured_len,
            indices.requests,
            indices.positions,
            indices.tail_offsets,
        )
    ]


def test_actual_model_lookup_radix_backend_and_native_attention(enabled, monkeypatch):
    idx, model, runner, backend, _, fb, _, q, k, v, _ = setup()
    selected = prepare(idx, model, runner, monkeypatch)
    enabled.register(NumericOffer(selected, raw_double))
    monkeypatch.setattr(qwen4_exp, "get_attn_backend", lambda: backend)
    monkeypatch.setattr(radix_attention, "get_attn_backend", lambda: backend)
    indices = qwen4_exp.Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices_eager(
        NS(layer_id=3),
        q,
        torch.ones(1, dtype=torch.int64),
        fb,
    )
    assert isinstance(indices, policy.DeferredMtpSparseIndices)
    layer = radix_attention.RadixAttention(6, 256, 0.0625, 1, 3)
    output = layer(q, k, v, fb, topk_indices=indices)
    assert output.shape == (1, 1536)


def test_deferred_prepare_uses_attention_backend_interface(enabled, monkeypatch):
    idx, model, runner, backend, state, batch, layer, q, k, v, _ = setup()
    prepare(idx, model, runner, monkeypatch)
    monkeypatch.setattr(
        numeric_binding, "numeric_operator", lambda *args, **kwargs: lambda io: None
    )
    selected = policy.lookup(backend, batch, tensor([102]), layer.layer_id)
    sentinel = lambda **kwargs: None
    monkeypatch.setattr(backend, "sparse_decode_kernel", lambda: sentinel)
    run = policy.bind_prepare(backend, q, k, v, layer, batch, selected)
    assert run is not None and run[-1] is sentinel
    # The model-specific resolver is not imported by the generic producer.
    import inspect

    assert "qwen_sparse_attn_backend" not in inspect.getsource(policy)
    assert backend.sparse_decode_page_size == 64


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
