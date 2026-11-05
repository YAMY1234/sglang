"""Pure host-data admission; no provider/kernel/runtime dependency."""

from types import SimpleNamespace as NS

import pytest

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericCapabilityQuery as Query
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as Op
from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ScalarType,
)
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.kernel_plugins.nvidia import numeric_binding as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


@pytest.mark.parametrize("operation", [3, True, "SPARSE_SELECT", None])
def test_query_rejects_non_enum_operation(operation):
    with pytest.raises(TypeError):
        Query(operation, ())


@pytest.mark.parametrize(
    "attributes",
    [
        [],
        (("x",),),
        (["x", 1],),
        (("", 1),),
        ((3, 1),),
        (("x", True),),
        (("x", 1.0),),
        (("x", object()),),
    ],
)
def test_query_rejects_mutable_or_nonnumeric_attributes(attributes):
    with pytest.raises(TypeError):
        Query(Op.SPARSE_SELECT, attributes)


def test_query_unique_names_and_real_enum_values():
    with pytest.raises(ValueError):
        Query(Op.SPARSE_SELECT, (("rows", 1), ("rows", 2)))
    q = Query(Op.SPARSE_SELECT, (("scalar_type", ScalarType.FP32),))
    assert q.attributes == (("scalar_type", ScalarType.FP32),)
    with pytest.raises(AttributeError):
        q.attributes = ()


@pytest.mark.parametrize(
    "enabled,master,spec",
    [(False, True, True), (True, False, True), (True, True, None), (True, True, False)],
)
def test_absent_or_disabled_support_never_calls_callback(
    monkeypatch, enabled, master, spec
):
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: enabled)
    callback = lambda _: pytest.fail("disabled source queried provider")
    selected = NS(numeric_capability=callback) if spec else spec
    if spec is False:
        selected = NS()
    with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master):
        assert not impl.numeric_capability_supported(
            Query(Op.SPARSE_SELECT, ()), spec=selected
        )


@pytest.mark.parametrize("answer", [True, False, 1, None, ()])
def test_pre_registration_pure_callback_and_exact_bool(monkeypatch, answer):
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setattr(
        impl,
        "numeric_registry",
        NS(get=lambda _: pytest.fail("circular registry lookup")),
    )
    q = Query(Op.SPARSE_SELECT, (("rows", 4),))
    seen = []
    selected = NS(numeric_capability=lambda actual: seen.append(actual) or answer)
    with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
        if type(answer) is bool:
            assert impl.numeric_capability_supported(q, spec=selected) is answer
        else:
            with pytest.raises(TypeError):
                impl.numeric_capability_supported(q, spec=selected)
    assert seen == [q]


def test_callback_exception_propagates(monkeypatch):
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)

    def fail(_):
        raise RuntimeError("provider query failure")

    with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
        with pytest.raises(RuntimeError, match="provider query failure"):
            impl.numeric_capability_supported(
                Query(Op.SPARSE_SELECT, ()), spec=NS(numeric_capability=fail)
            )


def test_postpool_uses_the_same_selected_spec_reference(monkeypatch):
    import torch

    seen = []

    class Producer(torch.nn.Module):
        def numeric_state_preparation_signatures(self, runner):
            seen.append(self._numeric_capability_spec)
            return ()

    model = Producer()
    selected = NS(numeric_capability=lambda query: False)
    owner = binding.MegakernelBinding(model=model)
    owner._numeric_spec = selected
    monkeypatch.setattr(owner, "_prepare_hybrid", lambda _: None)
    assert owner.preparation_view(NS()).numeric_signatures == ()
    assert seen == [selected] and model._numeric_capability_spec is selected


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
