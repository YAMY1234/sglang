"""FlashInfer MegaMOE fused epilogue: which batches hand it on, how the FFN exit
carries it, and when the next attention input takes it in the fused kernel."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.layer_boundary import (
    Layout,
    OutputContract,
    SumGroup,
    UnreducedOutput,
    complete_owed,
)
from sglang.srt.layers.layer_boundary import exit as exit_policy
from sglang.srt.layers.layer_boundary.fusions import megamoe_epilogue as fusion
from sglang.srt.layers.layer_boundary.fusions.megamoe_epilogue import (
    MegaMoeEpilogueFusion,
    MegaMoeEpilogueHandoff,
)
from sglang.srt.layers.layer_boundary.ops import keep_output
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 8


@pytest.mark.parametrize(
    "mode,hands_off",
    [
        (ForwardMode.DECODE, True),
        (ForwardMode.TARGET_VERIFY, True),
        (ForwardMode.EXTEND, False),
        (ForwardMode.MIXED, False),
        (ForwardMode.IDLE, False),
        (ForwardMode.DRAFT_EXTEND_V2, False),
    ],
)
def test_hands_off_decode_and_verify_batches_of_sparse_layers(mode, hands_off):
    batch = SimpleNamespace(forward_mode=mode)
    plan = SimpleNamespace(terminal=False)
    assert (
        MegaMoeEpilogueFusion(defers_finalize=True).can_defer_finalize(plan, batch)
        is hands_off
    )
    # A dense layer, and the stack's last layer, never hand off.
    assert not MegaMoeEpilogueFusion(defers_finalize=False).can_defer_finalize(
        plan, batch
    )
    assert not MegaMoeEpilogueFusion(defers_finalize=True).can_defer_finalize(
        SimpleNamespace(terminal=True), batch
    )


def _steps(**overrides):
    """An FFN's bound path whose output is complete and already on the next
    input's rows, as under a MegaMoE all-to-all."""
    fields = dict(
        output=OutputContract(Layout(frozenset())),
        writes_at_handoff=False,
        returns_over_dp=False,
        output_move=keep_output,
        exit=SimpleNamespace(sum_in_reduce_scatter=False),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _FinalizeAllReduceFusion:
    """A provider whose handoff the next input completes with an all-reduce."""

    def can_defer_finalize(self, plan, forward_batch):
        return True


def _decide(fusions, steps):
    plan = SimpleNamespace(fusions=fusions, terminal=False)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    return exit_policy.ExitPolicy(plan)._decide(batch, steps)


def test_exit_hands_on_the_finalize_of_a_complete_output():
    decision = _decide(MegaMoeEpilogueFusion(defers_finalize=True), _steps())
    assert decision.defer_moe_finalize
    # A plain tensor (a batch the MoE does not hand off) completes here, owing no sum.
    assert decision.complete.func.__name__ == "_complete_now"
    assert decision.complete.keywords["owes_sum"] is False


def test_exit_keeps_other_providers_decisions():
    decision = _decide(_FinalizeAllReduceFusion(), _steps())
    assert not decision.defer_moe_finalize


@pytest.mark.parametrize(
    "overrides",
    [
        {"output": OutputContract(Layout(frozenset()), group=SumGroup.MOE_OUTPUT)},
        {"writes_at_handoff": True},
        {"returns_over_dp": True},
        {"output_move": lambda **kwargs: kwargs},
        {
            "output": OutputContract(
                Layout(frozenset()), transform=OutputTransform(apply=lambda x: x)
            )
        },
    ],
)
def test_exit_does_not_hand_on_an_output_that_owes_more(overrides):
    plan = SimpleNamespace(terminal=False)
    assert exit_policy._hands_on_finalize_only(_steps(), plan)
    assert not exit_policy._hands_on_finalize_only(_steps(**overrides), plan)
    assert not exit_policy._hands_on_finalize_only(
        _steps(), SimpleNamespace(terminal=True)
    )


def test_handoff_completes_with_the_unfused_shared_add():
    routed = torch.tensor([[1.0, 2.5], [-3.0, 0.125]], dtype=torch.bfloat16)
    shared = torch.tensor([[0.5, 0.5], [1.0, -0.125]], dtype=torch.bfloat16)
    expected = routed + shared
    handoff = MegaMoeEpilogueHandoff(routed=routed, shared=shared)

    assert handoff.completion_group is SumGroup.MOE_OUTPUT
    completed = complete_owed(handoff)
    assert completed is routed
    assert torch.equal(completed, expected)


@pytest.fixture
def fused_kernel(monkeypatch):
    """The CUDA kernel module, replaced: the decision is what is under test."""
    ops = types.ModuleType("sglang.kernels.ops.layernorm.megamoe_epilogue")
    ops.can_use_megamoe_shared_add_rmsnorm = Mock(return_value=True)
    ops.megamoe_shared_add_rmsnorm = Mock(return_value="normed")
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    return ops


def _jit_norm():
    norm = RMSNorm(HIDDEN, eps=1e-6).to(torch.bfloat16)
    # The CUDA JIT RMSNorm family, as selected on an SM100 GPU.
    norm.use_jit_rmsnorm = True
    return norm


def _handoff():
    return MegaMoeEpilogueHandoff(
        routed=torch.ones(2, HIDDEN, dtype=torch.bfloat16),
        shared=torch.ones(2, HIDDEN, dtype=torch.bfloat16),
    )


def test_attention_input_fuses_the_handoff(fused_kernel):
    norm = _jit_norm()
    owed = _handoff()
    residual = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    (fused,) = MegaMoeEpilogueFusion(defers_finalize=True).attn_input_fusions(
        SimpleNamespace(norm=norm)
    )

    hidden_states, out_residual = fused(owed, residual, None, None)

    assert hidden_states == "normed"
    assert out_residual is residual
    fused_kernel.megamoe_shared_add_rmsnorm.assert_called_once_with(
        owed.routed, owed.shared, residual, norm.weight, norm.variance_epsilon
    )


@pytest.mark.parametrize(
    "case",
    [
        "unreduced_output",
        "post_residual_addition",
        "gemma_norm",
        "hf_rounding",
        "fp32_residual",
        "not_jit",
        "fp32_weight",
        "batch_invariant",
        "kernel_declines",
    ],
)
def test_attention_input_leaves_other_inputs_unfused(fused_kernel, monkeypatch, case):
    norm = _jit_norm()
    owed = _handoff()
    residual = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    post_residual_addition = None
    if case == "unreduced_output":
        owed = UnreducedOutput(torch.ones(2, HIDDEN, dtype=torch.bfloat16))
    elif case == "post_residual_addition":
        post_residual_addition = torch.ones_like(residual)
    elif case == "gemma_norm":
        norm = GemmaRMSNorm(HIDDEN, eps=1e-6).to(torch.bfloat16)
    elif case == "hf_rounding":
        norm.cast_x_before_out_mul = True
    elif case == "fp32_residual":
        norm.fp32_residual = True
    elif case == "not_jit":
        norm.use_jit_rmsnorm = False
    elif case == "fp32_weight":
        norm = norm.to(torch.float32)
    elif case == "batch_invariant":
        monkeypatch.setattr(fusion, "is_batch_invariant_mode_enabled", lambda: True)
    elif case == "kernel_declines":
        fused_kernel.can_use_megamoe_shared_add_rmsnorm.return_value = False
    (fused,) = MegaMoeEpilogueFusion(defers_finalize=True).attn_input_fusions(
        SimpleNamespace(norm=norm)
    )

    assert fused(owed, residual, None, post_residual_addition) is None
    fused_kernel.megamoe_shared_add_rmsnorm.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
