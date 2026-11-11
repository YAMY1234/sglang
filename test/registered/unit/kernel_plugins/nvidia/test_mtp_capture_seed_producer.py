"""Installed C02 producer; CUDA metadata and kernel work are explicit doubles."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
from capability_fixtures import mtp_spec

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpCaptureSeedIO,
    NumericOffer,
    NumericRegistry,
)
from sglang.srt import runtime_context
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding
from sglang.srt.layers.attention import qwen_sparse_attn_backend as impl
from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class CudaMetadata(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def tensor(value, dtype=torch.int32):
    return torch.as_tensor(value, dtype=dtype).as_subclass(CudaMetadata)


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setenv("SGLANG_Q38_GLUE_MERGE", "1")
    monkeypatch.setenv("SGLANG_Q38_PDL_SMALL", "1")
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: NS(tp_size=4))
    monkeypatch.setattr("sglang.srt.utils.is_sm100_supported", lambda: True)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    registry = NumericRegistry()
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
        yield registry


def setup():
    indexer = object.__new__(qsa_indexer.QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.layer_id, indexer.token_topk, indexer.compress_ratio = 3, 2048, 4
    state = impl.QSAMTPSharedSparseIndices(
        layer_ids=[3],
        num_requests=7,
        token_topk=2051,
        tail_width=4,
        device="cpu",
    )
    state.indices = tensor(torch.full_like(state.indices, -19))
    state.captured_len = tensor(torch.full_like(state.captured_len, -23))
    backend = object.__new__(impl.QwenSparseAttnBackend)
    backend._mtp_shared_sparse_indices = state
    model = torch.nn.Module()
    model.add_module("indexer", indexer)
    runner = NS(
        attn_backend=backend, is_draft_worker=True, model_config=NS(context_len=0)
    )
    batch = NS(
        batch_size=1,
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        _original_forward_mode=None,
        req_pool_indices=tensor([2], torch.int64),
        spec_info=NS(
            num_accept_tokens=tensor([3]),
            num_front_tokens=0,
            extend_seq_lens_tensor=tensor([4]),
        ),
    )
    topk = tensor(torch.arange(4 * 2051).reshape(4, 2051))
    lengths = tensor([101, 102, 103, 104])
    metadata = NS(get_seqlens_expanded=lambda: lengths, is_cuda_graph=True)
    return indexer, model, runner, backend, state, batch, topk, metadata


def prepare(indexer, model, runner, monkeypatch, widths=(1, 4, 8), tokens=4):
    assert numeric_binding.collect_numeric_signatures((indexer,), widths, tokens) == ()
    owner = binding.MegakernelBinding(model=model)
    owner._numeric_spec = mtp_spec()
    monkeypatch.setattr(owner, "_prepare_hybrid", lambda _: None)
    return owner.preparation_view(runner).numeric_signatures


def raw_double(io):
    """CPU implementation of original raw semantics, including trash/tail masks."""
    anchor = max(0, min(3, int(io.accept_counts[0]) - 1))
    row = (
        int(io.requests[0])
        if int(io.extend_lengths[0]) > 0 and int(io.requests[0]) > 0
        else 7
    )
    io.seed_out[row, :2051].copy_(io.topk_indices[anchor])
    io.captured_lengths_out[row] = io.expanded_lengths[anchor]


def test_actual_collector_post_pool_and_facade(context, monkeypatch):
    idx, model, runner, backend, state, fb, topk, md = setup()
    (selected,) = prepare(idx, model, runner, monkeypatch)
    assert (selected.rows, selected.front_tokens, selected.columns) == (4, 0, 2051)
    assert (selected.seed_rows, selected.trash_row, selected.seed_columns) == (
        8,
        7,
        2055,
    )
    assert (
        selected.extend_scalar_type,
        selected.accept_scalar_type,
        selected.request_scalar_type,
        selected.length_scalar_type,
        selected.pdl,
    ) == (5, 5, 4, 5, 1)
    assert numeric_binding.collect_numeric_signatures(
        (NS(numeric_signatures=lambda *_: (selected,)),),
        (4,),
        4,
    ) == (selected,)
    calls = []

    def provider(io):
        assert isinstance(io, MtpCaptureSeedIO)
        calls.append(io)
        raw_double(io)

    context.register(NumericOffer(selected, provider))
    backend._capture_mtp_sparse_indices_from_extend_lens(topk, fb, md, 3)
    assert len(calls) == 1
    assert calls[0].topk_indices is topk
    assert torch.equal(state.indices[0, 2, :2051], topk[2])
    assert torch.all(state.indices[0, 2, 2051:] == -19)
    assert state.captured_len[0, 2] == 103


@pytest.mark.parametrize(
    "reason",
    [
        "flag",
        "tp1",
        "arch",
        "target",
        "width",
        "tokens",
        "tail",
        "missing_layer",
        "dtype",
    ],
)
def test_unqualified_startup_declares_nothing(context, monkeypatch, reason):
    idx, model, runner, backend, state, *_ = setup()
    if reason == "flag":
        monkeypatch.setenv("SGLANG_Q38_GLUE_MERGE", "0")
    if reason == "tp1":
        monkeypatch.setattr(runtime_context, "get_parallel", lambda: NS(tp_size=1))
    if reason == "arch":
        monkeypatch.setattr("sglang.srt.utils.is_sm100_supported", lambda: False)
    if reason == "target":
        runner.is_draft_worker = False
    if reason == "tail":
        state.tail_width = 8
    if reason == "missing_layer":
        state.layer_slots = {}
    if reason == "dtype":
        state.indices = state.indices.long()
    assert (
        prepare(
            idx,
            model,
            runner,
            monkeypatch,
            widths=(8,) if reason == "width" else (4,),
            tokens=1 if reason == "tokens" else 4,
        )
        == ()
    )


@pytest.mark.parametrize(
    "reason",
    [
        "off",
        "masteroff",
        "nooffer",
        "rows",
        "front",
        "request_type",
        "topk_stride",
        "mode",
        "rewrite",
    ],
)
def test_refusal_runs_original_capture_once(context, monkeypatch, reason):
    idx, model, runner, backend, state, fb, topk, md = setup()
    (selected,) = prepare(idx, model, runner, monkeypatch)
    calls = []
    if reason != "nooffer":
        context.register(NumericOffer(selected, lambda io: calls.append(io)))
    if reason == "off":
        monkeypatch.setenv("SGLANG_Q38_GLUE_MERGE", "0")
    if reason == "masteroff":
        monkeypatch.setattr(binding, "megakernel_enabled", lambda: False)
    if reason == "rows":
        topk = topk[:3]
    if reason == "front":
        fb.spec_info.num_front_tokens = 1
    if reason == "request_type":
        fb.req_pool_indices = fb.req_pool_indices.int()
    if reason == "topk_stride":
        topk = tensor(torch.zeros((4, 4102)))[:, ::2]
    if reason == "mode":
        fb.forward_mode = ForwardMode.EXTEND
    if reason == "rewrite":
        fb._original_forward_mode = ForwardMode.EXTEND
    original_capture = state.capture
    stock = []

    def capture(*args):
        stock.append(args)
        original_capture(*args)

    state.capture = capture
    backend._capture_mtp_sparse_indices_from_extend_lens(topk, fb, md, 3)
    assert not calls and len(stock) == 1


@pytest.mark.parametrize(
    "request_id,extend,accept", [(2, 4, 1), (2, 4, 4), (0, 4, 0), (2, 0, 2), (-1, 4, 7)]
)
def test_selected_equals_unchanged_stock_and_preserves_pool(
    context, monkeypatch, request_id, extend, accept
):
    idx, model, runner, backend, state, fb, topk, md = setup()
    (selected,) = prepare(idx, model, runner, monkeypatch)
    fb.req_pool_indices.fill_(request_id)
    fb.spec_info.extend_seq_lens_tensor.fill_(extend)
    fb.spec_info.num_accept_tokens.fill_(accept)
    # First run the intact installed fallback, then the actual facade/provider.
    backend._capture_mtp_sparse_indices_from_extend_lens(topk, fb, md, 3)
    want_seed, want_lengths = state.indices.clone(), state.captured_len.clone()
    state.indices.fill_(-19)
    state.captured_len.fill_(-23)
    context.register(NumericOffer(selected, raw_double))
    backend._capture_mtp_sparse_indices_from_extend_lens(topk, fb, md, 3)
    assert torch.equal(state.indices, want_seed)
    assert torch.equal(state.captured_len, want_lengths)


def test_selected_write_exception_never_runs_stock(context, monkeypatch):
    idx, model, runner, backend, state, fb, topk, md = setup()
    (selected,) = prepare(idx, model, runner, monkeypatch)

    def fail(io):
        io.seed_out[2, 0] = 42
        raise RuntimeError("after seed write")

    context.register(NumericOffer(selected, fail))
    monkeypatch.setattr(state, "capture", lambda *_: pytest.fail("stock retried"))
    with pytest.raises(RuntimeError, match="after seed write"):
        backend._capture_mtp_sparse_indices_from_extend_lens(topk, fb, md, 3)
    assert state.indices[0, 2, 0] == 42


def test_nooffer_does_not_add_an_expanded_length_gather(context, monkeypatch):
    idx, model, runner, backend, _, fb, topk, md = setup()
    prepare(idx, model, runner, monkeypatch)
    gathered = []
    original = md.get_seqlens_expanded
    md.get_seqlens_expanded = lambda: (gathered.append(True), original())[1]
    backend._capture_mtp_sparse_indices_from_extend_lens(topk, fb, md, 3)
    assert len(gathered) == 1


def test_bool_signature_rejected_by_actual_collector(context, monkeypatch):
    idx, model, runner, *_ = setup()
    (selected,) = prepare(idx, model, runner, monkeypatch)
    with pytest.raises(TypeError, match="cannot contain bool"):
        numeric_binding.collect_numeric_signatures(
            (NS(numeric_signatures=lambda *_: (replace(selected, pdl=True),)),),
            (4,),
            4,
        )


@pytest.mark.parametrize(
    "change",
    ["missing_callback", "tp", "arch", "draft", "topk", "ratio", "tail", "tokens"],
)
def test_actual_facts_ask_provider_before_seed_declaration(
    context, monkeypatch, change
):
    idx, model, runner, backend, state, *_ = setup()
    numeric_binding.collect_numeric_signatures((idx,), (1, 4, 8), 4)
    owner = binding.MegakernelBinding(model=model)
    owner._numeric_spec = mtp_spec()
    monkeypatch.setattr(owner, "_prepare_hybrid", lambda _: None)
    if change == "missing_callback":
        owner._numeric_spec = None
    if change == "tp":
        monkeypatch.setattr(runtime_context, "get_parallel", lambda: NS(tp_size=1))
    if change == "arch":
        monkeypatch.setattr("sglang.srt.utils.is_sm100_supported", lambda: False)
    if change == "draft":
        runner.is_draft_worker = False
    if change == "topk":
        idx.token_topk = 1024
    if change == "ratio":
        idx.compress_ratio = 8
    if change == "tail":
        state.tail_width = 8
    if change == "tokens":
        idx._numeric_mtp_tokens_per_request = 8
    assert owner.preparation_view(runner).numeric_signatures == ()
    assert not hasattr(backend, "_numeric_mtp_capture_signatures")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("served", [True, False])
def test_batched_capture_runs_before_the_torch_capture(monkeypatch, served):
    from sglang.srt.layers.attention.qsa import mtp_capture

    _, _, _, backend, state, fb, topk, md = setup()
    md = NS(
        get_seqlens_expanded=md.get_seqlens_expanded,
        is_cuda_graph=True,
        sequence_lengths=tensor([0, 0, 103]),
        get_token_to_batch_idx=lambda: tensor([2, 2, 2, 2]),
    )
    calls = []

    def batched(*args):
        calls.append(args)
        return served

    monkeypatch.setattr(mtp_capture, "try_capture_metadata_batched", batched)
    with envs.SGLANG_ENABLE_QSA_MTP_CAPTURE_BATCHED.override(True):
        backend._capture_mtp_sparse_indices_from_extend_lens(topk, fb, md, 3)
    (args,) = calls
    assert args[0] is topk and args[3] == 0 and args[9] == state.trash_row
    assert args[7].data_ptr() == state.indices[0].data_ptr()
    # Served: the torch capture does not run; declined: it writes the seed as before.
    assert torch.all(state.indices[0, 2, :2051] == -19) == served
    with envs.SGLANG_ENABLE_QSA_MTP_CAPTURE_BATCHED.override(False):
        backend._capture_mtp_sparse_indices_from_extend_lens(topk, fb, md, 3)
    assert len(calls) == 1
