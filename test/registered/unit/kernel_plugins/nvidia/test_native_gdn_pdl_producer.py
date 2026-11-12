"""Installed native producer entries; CUDA metadata/launches are CPU doubles."""

from dataclasses import MISSING, fields, replace
from types import SimpleNamespace as NS
from typing import get_args

import pytest
import torch

from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as ordinary
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnSignature,
    NativeGdnPdlIO,
    NativeGdnPdlSignature,
    NumericOffer,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as O
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericRegistry,
    OperatorSignature,
)
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding, native_gdn_pdl, numeric_binding
from sglang.srt.layers.attention.linear import gdn_backend as backend_impl
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import qwen3_5 as model_impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class CudaMetadata(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda:0")

    def data_ptr(self):
        return 64 + self.storage_offset() * self.element_size()

    def new_empty(self, *size, **kwargs):
        # Model output allocation is a metadata double on this CPU runner.
        shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
        return tensor(shape, kwargs.get("dtype", self.dtype))


def tensor(shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta").as_subclass(CudaMetadata)


def setup(
    rows=32,
    heads=(16, 48),
    mode=1,
    state_dtype=torch.bfloat16,
    bias_dtype=torch.float32,
):
    h, hv = heads
    channels, batch, replay_slots = (2 * h + hv) * 128, rows // 4, rows // 4 + 1
    state = MambaPool.SpeculativeState(
        conv=[tensor((batch + 3, channels, 3))],
        temporal=tensor((batch + 3, hv, 128, 128), state_dtype),
        intermediate_ssm=None,
        intermediate_conv_window=[tensor((replay_slots, 4, channels, 3))],
        replayssm_d=tensor((replay_slots, hv, 16, 128)),
        replayssm_k=tensor((replay_slots, h, 16, 128)),
        replayssm_g=tensor((replay_slots, hv, 16), torch.float32),
        replayssm_rawv=tensor((replay_slots, hv, 16, 128)) if mode else None,
        replayssm_rawk=tensor((replay_slots, h, 16, 128)) if mode else None,
        replayssm_beta=tensor((replay_slots, hv, 16), torch.float32)
        if mode == 2
        else None,
    )
    backend = object.__new__(backend_impl.GDNAttnBackend)
    backend.mis_metadata = None
    backend.prefill_graph_metadata = None
    backend.requires_contiguous_prefill_state = False
    backend._use_strided_target_verify_qkv = True
    backend.req_to_token_pool = NS(size=batch + 3)
    backend.state_layer_cache = lambda _: state
    backend.state_pool = NS(
        replayssm_cache_base=tensor((replay_slots,), torch.int32),
        replayssm_spec_write_pos=tensor((replay_slots,), torch.int32),
        replayssm_is_flush=tensor((replay_slots,), torch.int8),
        replayssm_spec_fold=False,
        replayssm_is_kda=False,
    )
    backend.forward_metadata = NS(
        query_start_loc=tensor((batch + 1,), torch.int32),
        mamba_cache_indices=tensor((batch,), torch.int64),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    backend.verify_intermediate_state_indices = tensor((batch,), torch.int32)
    layer = NS(
        layer_id=0,
        num_k_heads=h,
        num_q_heads=h,
        num_v_heads=hv,
        head_k_dim=128,
        head_q_dim=128,
        head_v_dim=128,
        q_dim=h * 128,
        k_dim=h * 128,
        v_dim=hv * 128,
        activation="silu",
        A_log=tensor((hv,), torch.float32),
        dt_bias=tensor((hv,), bias_dtype),
        conv_weights=tensor((channels, 4)),
        bias=None,
    )
    model = object.__new__(model_impl.Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(model)
    model.attn, model.attn_tp_size = layer, 16 // h
    model.head_k_dim = model.head_v_dim = 128
    from capability_fixtures import contract_spec

    model._numeric_capability_spec = contract_spec()
    model.hidden_size = 2560
    model._fused_in_proj_weight = None
    model.norm = NS(
        weight=tensor((128,)),
        activation="sigmoid",
        bias=None,
        group_size=None,
        norm_before_gate=True,
    )
    batch_view = NS(
        spec_info=NS(draft_token_num=4),
        batch_size=batch,
        forward_mode=ForwardMode.TARGET_VERIFY,
        logical_forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=tensor((batch,), torch.int64),
    )
    return model, backend, layer, batch_view, state


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setenv("Q38_GDN_PDL", "1")
    monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "0")
    monkeypatch.setattr(backend_impl, "pp_spec_stable_rows_enabled", lambda: False)
    registry = NumericRegistry()
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
        yield registry


def prepare(model, backend, monkeypatch, widths=(4, 32, 192), tokens=4):
    assert numeric_binding.collect_numeric_signatures((model,), widths, tokens) == ()
    bound = binding.MegakernelBinding(model=model)
    bound._numeric_spec = model._numeric_capability_spec
    monkeypatch.setattr(bound, "_prepare_hybrid", lambda _: None)
    runner = NS(attn_backend=NS(linear_attn_backend=backend))
    return bound.preparation_view(runner)


@pytest.mark.parametrize("heads", [(4, 12), (16, 48)])
@pytest.mark.parametrize("storage", [0, 1, 2])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_actual_post_pool_publishes_only_real_state(
    context, monkeypatch, heads, storage, state_dtype, bias_dtype
):
    model, backend, layer, _, state = setup(
        heads=heads, mode=storage, state_dtype=state_dtype, bias_dtype=bias_dtype
    )
    view = prepare(model, backend, monkeypatch)
    assert len(view.numeric_signatures) == 3
    assert view.numeric_signatures == layer._numeric_native_pdl_signatures
    assert {s.rows for s in view.numeric_signatures} == {4, 32, 192}
    assert {s.storage_mode for s in view.numeric_signatures} == {storage}
    assert {s.state_scalar_type for s in view.numeric_signatures} == {
        1 if state_dtype == torch.bfloat16 else 3
    }
    assert {s.dt_bias_scalar_type for s in view.numeric_signatures} == {
        1 if bias_dtype == torch.bfloat16 else 3
    }
    assert {s.wait_for_gate_producer for s in view.numeric_signatures} == {0}
    assert state.temporal.shape[0] != state.replayssm_d.shape[0]


@pytest.mark.parametrize("three", [False, True])
def test_signature_carries_the_verify_stage_count(context, monkeypatch, three):
    """SGLANG_OPT_GDN_REPLAYSSM_VERIFY_3_STAGES reaches the native PDL provider as the signature's stage count."""
    from sglang.srt.environ import envs

    with envs.SGLANG_OPT_GDN_REPLAYSSM_VERIFY_3_STAGES.override(three):
        model, backend, layer, _, _ = setup()
        view = prepare(model, backend, monkeypatch)
    assert {s.num_stages for s in view.numeric_signatures} == {3 if three else 2}


@pytest.mark.parametrize(
    "reason",
    ["off", "tokens", "widths", "kda", "fold", "invalid_metadata", "partial_storage"],
)
def test_missing_source_preconditions_do_not_declare(context, monkeypatch, reason):
    model, backend, _, _, state = setup()
    if reason == "off":
        monkeypatch.setenv("Q38_GDN_PDL", "0")
    elif reason == "kda":
        backend.state_pool.replayssm_is_kda = True
    elif reason == "fold":
        backend.state_pool.replayssm_spec_fold = True
    elif reason == "invalid_metadata":
        backend.mis_metadata = object()
    elif reason == "partial_storage":
        backend.state_layer_cache = lambda _: replace(state, replayssm_rawk=None)
    assert (
        prepare(
            model,
            backend,
            monkeypatch,
            widths=(1, 3) if reason == "widths" else (32,),
            tokens=8 if reason == "tokens" else 4,
        ).numeric_signatures
        == ()
    )


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "nooffer",
        "off",
        "masteroff",
        "dtype",
        "flush",
        "tree",
        "undeclared",
        "selected_error",
    ],
)
def test_imported_forward_conv_once_then_provider_or_native_once(
    context, monkeypatch, reason
):
    model, backend, layer, fb, state = setup()
    view = prepare(model, backend, monkeypatch, widths=(32,))
    (signature,) = view.numeric_signatures
    events = []

    def provider(io):
        assert isinstance(io, NativeGdnPdlIO)
        assert io.checkpoint_state is state.temporal
        assert io.query_start_loc is backend.forward_metadata.query_start_loc
        events.append("provider")
        if reason == "selected_error":
            raise RuntimeError("selected wrote then failed")

    if reason != "nooffer":
        context.register(NumericOffer(signature, provider))
    monkeypatch.setattr(
        ordinary, "gdn_replayssm_spec_decode", lambda **kw: events.append("native")
    )

    def conv(x, *args, **kwargs):
        events.append("conv")
        return x

    monkeypatch.setattr(backend_impl, "causal_conv1d_update", conv)
    if reason == "off":
        monkeypatch.setenv("Q38_GDN_PDL", "0")
    if reason == "masteroff":
        monkeypatch.setattr(binding, "megakernel_enabled", lambda: False)
    if reason == "flush":
        backend.state_pool.replayssm_is_flush = tensor((9,), torch.int32)
    if reason == "tree":
        backend.forward_metadata.retrieve_parent_token = tensor((32,), torch.int64)
    if reason == "undeclared":
        layer._numeric_native_pdl_signatures = ()
    a = tensor((32, 48), torch.float32 if reason == "dtype" else torch.bfloat16)
    args = layer, fb, tensor((32, 10240)), a, tensor((32, 48))
    if reason == "selected_error":
        with pytest.raises(RuntimeError, match="selected wrote"):
            backend.forward_extend(*args)
        assert events == ["conv", "provider"]
    else:
        out = backend.forward_extend(*args)
        assert isinstance(out, torch.Tensor) and out.shape == (1, 32, 48, 128)
        assert events == ["conv", "provider" if reason is None else "native"]


def test_actual_collector_new_signature_bool_refused(context):
    signature = NativeGdnPdlSignature(O.GDN_NATIVE_PDL, 32, 16, 48, 128, 128, 4, 1)
    producer = NS(numeric_signatures=lambda *_: (signature,))
    assert numeric_binding.collect_numeric_signatures((producer,), (32,), 4) == (
        signature,
    )
    producer.numeric_signatures = lambda *_: (
        replace(signature, wait_for_gate_producer=True),
    )
    with pytest.raises(TypeError, match="bool"):
        numeric_binding.collect_numeric_signatures((producer,), (32,), 4)
    assert numeric_binding._IO_TYPES[O.GDN_NATIVE_PDL] is NativeGdnPdlIO


def test_actual_collector_accepts_all_declared_signature_types():
    classes = get_args(OperatorSignature)
    assert len(classes) == 10
    signatures = tuple(
        cls(
            **{
                f.name: 1
                for f in fields(cls)
                if f.default is MISSING and f.default_factory is MISSING
            }
        )
        for cls in classes
    )
    producer = NS(numeric_signatures=lambda *_: signatures)
    assert (
        numeric_binding.collect_numeric_signatures((producer,), (4,), 4) == signatures
    )


@pytest.mark.parametrize(
    "fault",
    [
        "q_stride",
        "a_dtype",
        "state_dtype",
        "ring_dtype",
        "ring_capacity",
        "missing_rawk",
        "indices_dtype",
        "query_count",
        "cursor_dtype",
        "flush_dtype",
        "missing_indices",
        "wait_not_declared",
    ],
)
def test_incomplete_actual_io_refuses_before_lookup(context, monkeypatch, fault):
    model, backend, layer, fb, state = setup()
    prepare(model, backend, monkeypatch, widths=(32,))
    h, hv = layer.num_k_heads, layer.num_v_heads
    args = dict(
        q=tensor((32, h, 128)),
        k=tensor((32, h, 128)),
        v=tensor((32, hv, 128)),
        a=tensor((32, hv)),
        b=tensor((32, hv)),
        out=tensor((32, hv, 128)),
        query_start_loc=backend.forward_metadata.query_start_loc,
        state_indices=fb.req_pool_indices,
        replay_indices=fb.req_pool_indices,
    )
    if fault == "q_stride":
        args["q"] = args["q"].transpose(1, 2)
    elif fault == "a_dtype":
        args["a"] = tensor((32, hv), torch.float32)
    elif fault == "state_dtype":
        state = replace(state, temporal=tensor(state.temporal.shape, torch.float32))
    elif fault == "ring_dtype":
        state = replace(state, replayssm_g=tensor(state.replayssm_g.shape))
    elif fault == "ring_capacity":
        state = replace(state, replayssm_k=state.replayssm_k[:1])
    elif fault == "missing_rawk":
        state = replace(state, replayssm_rawk=None)
    elif fault == "indices_dtype":
        args["state_indices"] = tensor((8,), torch.float32)
    elif fault == "query_count":
        args["query_start_loc"] = tensor((8,), torch.int32)
    elif fault == "cursor_dtype":
        backend.state_pool.replayssm_spec_write_pos = tensor((9,), torch.int64)
    elif fault == "flush_dtype":
        backend.state_pool.replayssm_is_flush = tensor((9,), torch.bool)
    elif fault == "missing_indices":
        args["state_indices"] = None
    elif fault == "wait_not_declared":
        args["wait_for_gate_producer"] = True
    monkeypatch.setattr(
        numeric_binding,
        "numeric_operator",
        lambda *a, **kw: pytest.fail("incomplete view looked up provider"),
    )
    assert (
        native_gdn_pdl.bind_native_gdn_pdl(layer, state, backend.state_pool, **args)
        is None
    )


@pytest.mark.parametrize("mk_offer", [False, True])
def test_existing_mk_core_precedes_native_pdl(context, monkeypatch, mk_offer):
    model, backend, layer, fb, _ = setup(4, (4, 12))
    (native_signature,) = prepare(
        model, backend, monkeypatch, widths=(4,)
    ).numeric_signatures
    events = []
    context.register(
        NumericOffer(native_signature, lambda io: events.append("native_pdl"))
    )
    core = GdnSignature(O.GDN_CORE, 4, 4, 12, 128, 128, 4, 1, gate_scalar_type=1)
    if mk_offer:
        layer._numeric_gdn_signatures = frozenset((core,))
        context.register(NumericOffer(core, lambda io: events.append("mk_core")))
    monkeypatch.setenv("SGLANG_Q38_MK_GDN", "1")
    monkeypatch.setattr(backend_impl, "causal_conv1d_update", lambda x, *a, **kw: x)
    monkeypatch.setattr(
        ordinary,
        "gdn_replayssm_spec_decode",
        lambda **kw: pytest.fail("unexpected ordinary retry"),
    )
    backend.forward_extend(
        layer, fb, tensor((4, 2560)), tensor((4, 12)), tensor((4, 12))
    )
    assert events == ["mk_core" if mk_offer else "native_pdl"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
