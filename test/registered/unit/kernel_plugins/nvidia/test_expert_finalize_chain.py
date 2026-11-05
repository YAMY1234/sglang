"""Installed expert/sink/finalize chain and actual extracted stream caller."""

import __future__

import ast
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from test_expert_bf16_admission import TestActualAdmission as _AdmissionFixture

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import local_experts as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def exercise(reason=None):
    events = []

    class Metadata(torch.Tensor):
        @property
        def is_cuda(self):
            return True

        def record_stream(self, stream):
            events.append(("retain", stream.name))

    def tensor(shape):
        return torch.empty(shape, device="meta", dtype=torch.bfloat16).as_subclass(
            Metadata
        )

    kwargs = _AdmissionFixture().kwargs(
        4, torch.float32 if reason == "fp32" else torch.bfloat16
    )
    hidden, gate, shared = tensor((4, 2560)), tensor((1, 2560)), tensor((4, 2560))
    output = tensor((4, 2560))
    kwargs.update(do_finalize=reason != "deferred", output=output)

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait", self.name, other.name))

    current, side = Stream("main"), Stream("side")
    active = [current]

    @contextmanager
    def stream(s):
        old = active[0]
        active[0] = s
        try:
            yield
        finally:
            active[0] = old

    scope = {
        "torch": NS(cuda=NS(current_stream=lambda: active[0], stream=stream)),
        "envs": envs,
        "_SGLANG_EXPERIMENTAL_LORA_OPTI": False,
    }
    path = Path(impl.__file__).parents[2] / "models/qwen2_moe.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen2MoeSparseMoeBlock"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_normal_dual_stream"
    )
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    sink = []
    views = []
    native = []

    def choose(signature, *, prepared):
        if not prepared:
            return None
        if (
            signature.operation == NumericOperation.LOCAL_EXPERTS
            and signature.variant != 1
        ):
            return None
        if signature.operation == NumericOperation.LOCAL_EXPERTS:
            if reason == "nooffer":
                return None

            def experts(io):
                events.append(("experts", active[0].name))
                views.append(io)

            return experts
        assert signature.operation == NumericOperation.FINALIZE_SHARED

        def finalize(io):
            events.append(("finalize", active[0].name))
            views.append(io)
            assert io.output is output and io.hidden is hidden and io.shared is shared
            assert io.cols == 256 and io.use_pdl

        return finalize

    def router(_hidden, **unused):
        result = None if reason == "native" else impl.try_numeric_local_experts(kwargs)
        if result is None:
            native.append(True)
            events.append(("native", active[0].name))
            return output
        return result if reason == "deferred" else result[0]

    obj = NS(
        experts=NS(intermediate_size_per_partition=192, num_experts=512, top_k=10),
        alt_stream=side,
        shared_expert=object(),
        shared_expert_gate=NS(weight=gate),
        _shared_expert_input=lambda hidden: hidden.clone(),
        _forward_shared_experts=lambda *a, **kw: shared,
        _forward_router_experts=router,
        _gate_shared_output_out_of_place=lambda x, y: y,
    )
    flags = {
        "SGLANG_Q38_MK_MOE": "0" if reason == "off" else "1",
        "SGLANG_Q38_MK_MOE_SF_PACK": "1",
        "SGLANG_Q38_FINALIZE_SHARED": "1",
        "SGLANG_Q38_FINALIZE_SHARED_COLS": "256",
        "SGLANG_Q38_FINALIZE_SHARED_PDL": "1",
    }
    fake_context = NS(
        has_forward_context=lambda: False, get_forward_context=lambda: None
    )
    empty = torch.empty

    def allocated(*args, **kw):
        value = empty(*args, **kw)
        return value.as_subclass(Metadata) if value.dtype == torch.bfloat16 else value

    with (
        patch(
            "sglang.srt.runtime_context.get_spec",
            return_value=NS(speculative_num_draft_tokens=4),
        ),
        patch.dict(os.environ, flags),
        patch.dict(
            sys.modules,
            {
                "flashinfer.tllm_enums": NS(ActivationType=NS(Swiglu=1)),
                "sglang.srt.model_executor.forward_context": fake_context,
            },
        ),
        patch.object(impl, "numeric_operator", side_effect=choose),
        patch.object(torch, "empty", side_effect=allocated),
        patch.object(torch.cuda, "current_stream", side_effect=lambda *a: active[0]),
    ):
        routed, got_shared = scope["forward_normal_dual_stream"](
            obj,
            hidden,
            use_fused_gate=True,
            defer_finalize=reason == "deferred",
            ordered_finalize_sink=None if reason == "nosink" else sink,
        )
        assert impl._finalize_sink.get() is None
        if sink:
            assert len(sink) == 1 and routed is output
            assert (
                sink[0][0] is views[0].expert_rows_out
                and sink[0][1] is views[0].route_weights_out
            )
            assert sink[0][2] is output and views[0].final_out is None
            assert impl.finish_local_shared_finalize(
                sink, hidden, gate.squeeze(), got_shared, routed
            )
        else:
            assert (
                impl.finish_local_shared_finalize(sink, None, None, None, None) is False
            )
    return events, sink, views, native, output


def test_actual_installed_expert_sink_finish_and_real_side_join():
    events, sink, views, native, output = exercise()
    assert len(sink) == 1 and len(views) == 2 and native == []
    assert views[1].expert_rows is views[0].expert_rows_out
    assert views[1].route_weights is views[0].route_weights_out
    assert (
        events.index(("experts", "side"))
        < events.index(("wait", "main", "side"))
        < events.index(("finalize", "main"))
    )
    assert (
        events.count(("finalize", "main")) == 1
        and events.count(("retain", "main")) == 3
    )
    assert views[1].output is output


@pytest.mark.parametrize(
    "reason", ["fp32", "off", "nooffer", "native", "nosink", "deferred"]
)
def test_refused_native_no_sink_and_deferred_do_not_stage_finalize(reason):
    events, sink, views, native, _ = exercise(reason)
    assert sink == [] and not any(e[0] == "finalize" for e in events)
    if reason == "nosink":
        assert views[0].final_out is not None
    if reason == "deferred":
        assert views[0].final_out is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
