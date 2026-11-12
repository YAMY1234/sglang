"""Real imported model/backend entry regressions for original target GDN views.

CPU tensors prove layout/value/selection policy; CUDA conv/recurrence/norm launches
are explicit doubles. These cases do not assert GPU numeric or race equivalence.
"""

from types import SimpleNamespace as NS

import pytest
import torch
from test_gdn_backend_imports import setup

from sglang.srt.layers.attention.linear import gdn_backend as backend_impl
from sglang.srt.layers.attention.linear import utils as linear_utils
from sglang.srt.model_executor import forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import qwen3_5 as model_impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize(
    "mode", [ForwardMode.TARGET_VERIFY, ForwardMode.DECODE, ForwardMode.EXTEND]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_actual_model_selects_source_views_without_unpack(
    tp, rows, mode, enabled, monkeypatch
):
    monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "0")
    monkeypatch.setenv("Q38_GDN_FRONT", "1" if enabled else "0")
    monkeypatch.setattr(model_impl, "_is_cuda", True)
    monkeypatch.setattr(model_impl, "_is_cpu", False)
    monkeypatch.setattr(model_impl, "_is_xpu", False)
    # Match CUDA's import-time supported ratio set in this CPU-only process.
    monkeypatch.setattr(model_impl, "_GDN_FUSED_QKVZBA_RATIOS", (1, 2, 3, 4))
    monkeypatch.setattr(model_impl, "_gdn_decode_fused_proj_conv", False)
    monkeypatch.setattr(model_impl, "_fp8_tuple_input", lambda *args: False)
    monkeypatch.setattr(
        forward_context, "get_attn_backend", lambda: NS(supports_packed_verify=False)
    )
    h, hv = 16 // tp, 48 // tp
    c = (2 * h + hv) * 128
    qkvz = torch.randn(rows, c + hv * 128, dtype=torch.bfloat16)
    ba = torch.randn(rows, 2 * hv, dtype=torch.bfloat16)
    events = []
    actual_views = model_impl.qwen3_5_gdn_prefill_projection_views

    def views(*args):
        events.append("views")
        return actual_views(*args)

    def unpack(*args):
        events.append("unpack")
        return tuple(t.contiguous() for t in actual_views(*args))

    monkeypatch.setattr(model_impl, "qwen3_5_gdn_prefill_projection_views", views)
    monkeypatch.setattr(model_impl, "fused_qkvzba_split_reshape_cat_contiguous", unpack)
    expected = actual_views(qkvz, ba, h, hv, 128, 128)

    class Attention(NS):
        def __call__(self, batch, *, mixed_qkv, a, b):
            events.append("attention")
            for actual, reference in zip(
                (mixed_qkv, b, a), (expected[0], expected[2], expected[3])
            ):
                assert torch.equal(actual, reference)
            self.seen = mixed_qkv, a, b
            return torch.zeros(rows, hv, 128, dtype=torch.bfloat16)

    class Norm:
        def __call__(self, x, z):
            events.append("norm")
            assert torch.equal(z.reshape(rows, hv, 128), expected[1])
            self.seen = z
            return x

    m = model_impl.Qwen3_5GatedDeltaNet.__new__(model_impl.Qwen3_5GatedDeltaNet)
    torch.nn.Module.__init__(m)
    m.num_k_heads, m.num_v_heads, m.attn_tp_size = 16, 48, tp
    m.head_k_dim = m.head_v_dim = 128
    m._fused_in_proj_weight = None
    m.activation = "silu"
    m.attn = Attention(num_k_heads=h, num_v_heads=hv)
    m.norm = Norm()
    m._forward_input_proj = lambda x: (qkvz, ba)
    m.out_proj = lambda x: (x, None)
    fb = NS(forward_mode=mode, logical_forward_mode=mode, batch_size=rows // 4)
    result = m.forward(torch.empty(rows, 2560, dtype=torch.bfloat16), fb)
    use_views = mode.is_extend_without_speculative() or (
        enabled and mode.is_target_verify()
    )
    assert events == [("views" if use_views else "unpack"), "attention", "norm"]
    assert result.shape == (rows, hv * 128)
    if use_views:
        assert m.attn.seen[0].data_ptr() == qkvz.data_ptr()
        assert m.attn.seen[0].stride(0) == qkvz.stride(0)
        assert (
            m.attn.seen[1].untyped_storage().data_ptr()
            == ba.untyped_storage().data_ptr()
        )
        assert (
            m.norm.seen.untyped_storage().data_ptr()
            == qkvz.untyped_storage().data_ptr()
        )
        assert m.norm.seen.stride() == expected[1].stride()


class ComparedIndices(torch.Tensor):
    comparisons = 0

    def __ge__(self, other):
        type(self).comparisons += 1
        return super().__ge__(other)


@pytest.mark.parametrize("rows", [4, 16, 32])
@pytest.mark.parametrize("pp", [False, True])
def test_actual_backend_does_not_evaluate_unused_mask(rows, pp, monkeypatch):
    backend, layer, fb = setup(rows)
    batch = rows // 4
    backend.prefill_graph_metadata = None
    backend.requires_contiguous_prefill_state = False
    backend.req_to_token_pool = NS(size=64)
    indices = torch.arange(batch, dtype=torch.int64)
    indices[-1] = -1
    backend.forward_metadata.mamba_cache_indices = indices.as_subclass(ComparedIndices)
    backend.verify_intermediate_state_indices = (
        torch.arange(batch, dtype=torch.int32) + 17
    )
    fb.req_pool_indices = torch.arange(batch, dtype=torch.int64) + 3
    ComparedIndices.comparisons = 0
    monkeypatch.setattr(backend_impl, "pp_spec_stable_rows_enabled", lambda: pp)
    monkeypatch.setattr(linear_utils, "pp_spec_stable_rows_enabled", lambda: pp)

    # Run the actual imported selector; stop at the first CUDA-only operation.
    class AtConv(Exception):
        pass

    captured = {}

    def conv(*args, **kwargs):
        captured.update(kwargs)
        raise AtConv

    monkeypatch.setattr(backend_impl, "causal_conv1d_update", conv)
    with pytest.raises(AtConv):
        backend.forward_extend(
            layer,
            fb,
            torch.empty(rows, 2560),
            torch.empty(rows, 12),
            torch.empty(rows, 12),
        )
    assert ComparedIndices.comparisons == int(pp)
    expected = (
        torch.where(indices >= 0, fb.req_pool_indices, 64).to(torch.int32)
        if pp
        else backend.verify_intermediate_state_indices
    )
    assert torch.equal(captured["intermediate_state_indices"], expected)
    if not pp:
        assert (
            captured["intermediate_state_indices"].data_ptr()
            == backend.verify_intermediate_state_indices.data_ptr()
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
