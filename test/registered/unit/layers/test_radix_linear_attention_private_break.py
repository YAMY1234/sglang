"""A linear-attention module that is not the forward context's module for its
layer_id (a trained TwinStar emitter) must run its own weights in the break."""

from types import SimpleNamespace

import sglang.srt.layers.radix_linear_attention as rla
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_private_layers_get_their_own_break_key(monkeypatch):
    target, private = SimpleNamespace(layer_id=2), SimpleNamespace(layer_id=2)
    context = SimpleNamespace(attention_layers=[None, None, target], forward_batch="fb")
    monkeypatch.setattr(rla, "get_tc_piecewise_forward_context", lambda: context)
    monkeypatch.setattr(rla, "_PRIVATE_BREAK_LAYERS", {})
    assert rla._break_layer_key(target) == 2
    key = rla._break_layer_key(private)
    assert key >= rla._PRIVATE_BREAK_KEY_BASE and rla._break_layer_key(private) == key
    seen = {}
    monkeypatch.setattr(rla, "_linear_attention_with_output_impl",
                        lambda **kw: seen.update(layer=kw["attention_layer"]))
    rla._unified_linear_attention_with_output_impl(None, None, None, None, key)
    assert seen["layer"] is private
    rla._unified_linear_attention_with_output_impl(None, None, None, None, 2)
    assert seen["layer"] is target
