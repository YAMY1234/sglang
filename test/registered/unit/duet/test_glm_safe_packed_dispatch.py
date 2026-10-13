"""CPU route/shape tests executing the backend's actual dispatch AST."""

import ast
import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
LINEAR = ROOT / "python/sglang/srt/layers/attention/linear"
spec = importlib.util.spec_from_file_location(
    "glm_shape_guard", LINEAR / "glm_safe_packed.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class CudaMetadata:
    """Host tensor layout stand-in; no CUDA allocation/initialization."""

    device = torch.device("cuda:0")

    def __init__(self, value):
        self.value = value

    def __getattr__(self, key):
        return getattr(self.value, key)


class TritonKDAKernel:
    pass


def fixture(batch=2, heads=16):
    cuda = lambda *shape, dtype=torch.float32: CudaMetadata(
        torch.empty(*shape, dtype=dtype)
    )
    layer = SimpleNamespace(
        num_q_heads=heads,
        num_k_heads=heads,
        num_v_heads=heads,
        head_q_dim=128,
        head_k_dim=128,
        head_v_dim=128,
        q_dim=heads * 128,
        k_dim=heads * 128,
        v_dim=heads * 128,
        lower_bound=-5.0,
        A_log=cuda(heads),
        dt_bias=cuda(heads * 128),
        layer_id=0,
    )
    return (
        layer,
        cuda(batch, 3 * heads * 128),
        cuda(batch, heads * 128),
        cuda(batch, heads),
        cuda(batch + 1, heads, 128, 128),
        cuda(batch, dtype=torch.int32),
    )


def route():
    """Compile just the real recurrent dispatch block, with logging/pool stubs."""
    tree = ast.parse((LINEAR / "kda_backend.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "KDAAttnBackend"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_decode"
    )
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "glm_safe_packed"
    )
    fn = ast.parse(
        "def route(self, layer, qkv, a, b, ssm_states, cache_indices):\n    pass\n"
    ).body[0]
    fn.body = copy.deepcopy(method.body[start:])
    code = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    env = {
        "TritonKDAKernel": TritonKDAKernel,
        "glm_safe_packed_covered": guard.covered,
        "get_is_capture_mode": lambda: False,
        "rank0_log": lambda _: None,
        "replayssm_d": None,
        "replayssm_k": None,
        "replayssm_g": None,
        "replayssm_write_pos": None,
        "replayssm_force_flush": None,
        "forward_batch": None,
        "conv_states": None,
        "query_start_loc": None,
    }
    exec(compile(code, str(LINEAR / "kda_backend.py"), "exec"), env)  # noqa: S102 - trusted repo source
    return env["route"], env


class Dispatch(unittest.TestCase):
    def test_route_default_off_opt_in_and_unsupported_shape(self):
        call, _ = route()
        inputs = fixture()
        calls = []
        dispatcher = SimpleNamespace(
            supports_packed_decode=True,
            decode_kernel=TritonKDAKernel(),
            packed_decode=lambda **kwargs: calls.append(("packed", kwargs)) or "packed",
            decode=lambda **kwargs: calls.append(("generic", kwargs)) or "generic",
        )
        backend = SimpleNamespace(
            glm_safe_packed_decode=False,
            kernel_dispatcher=dispatcher,
            state_pruner=None,
            _track_mamba_state_decode=lambda *args: None,
        )
        self.assertEqual(call(backend, *inputs), "generic")
        backend.glm_safe_packed_decode = True
        self.assertEqual(call(backend, *inputs), "packed")
        self.assertTrue(calls[-1][1]["glm_safe_packed"])
        inputs[0].head_k_dim = 64
        self.assertEqual(call(backend, *inputs), "generic")
        inputs[0].head_k_dim = 128
        inputs[0].lower_bound = None
        backend.glm_safe_packed_decode = False
        self.assertEqual(call(backend, *inputs), "packed")
        self.assertFalse(calls[-1][1]["glm_safe_packed"])

    def test_tp4_guard_and_capture_without_device_value_reads(self):
        for heads in (8, 16, 32, 64):
            self.assertEqual(guard.covered(*fixture(heads=heads)), heads == 16)
        call, env = route()
        env["get_is_capture_mode"] = lambda: True
        calls = []
        dispatcher = SimpleNamespace(
            supports_packed_decode=True,
            decode_kernel=TritonKDAKernel(),
            packed_decode=lambda **kwargs: calls.append(kwargs) or "packed",
        )
        backend = SimpleNamespace(
            glm_safe_packed_decode=True,
            kernel_dispatcher=dispatcher,
            state_pruner=None,
            _track_mamba_state_decode=lambda *args: None,
        )
        self.assertEqual(call(backend, *fixture()), "packed")
        self.assertEqual(len(calls), 1)
        self.assertTrue(backend._glm_safe_packed_logged)
        self.assertFalse(torch.cuda.is_initialized())

    def test_glm_policy_does_not_enter_legacy_auxiliary_sink_recurrence(self):
        call, _ = route()
        dispatcher = SimpleNamespace(
            supports_packed_decode=True,
            decode_kernel=TritonKDAKernel(),
            packed_decode=lambda **kwargs: "packed",
        )
        backend = SimpleNamespace(
            glm_safe_packed_decode=True,
            kernel_dispatcher=dispatcher,
            state_pruner=SimpleNamespace(),  # dense GLM policy has no decode()
            _track_mamba_state_decode=lambda *args: None,
        )
        self.assertEqual(call(backend, *fixture()), "packed")

    def test_actual_glm_beta_shape_and_row_pitched_projection(self):
        values = list(fixture(batch=8))
        beta = torch.empty(8, 48)[:, :16].unsqueeze(0)
        self.assertFalse(beta.is_contiguous())
        values[3] = CudaMetadata(beta)
        values[1] = CudaMetadata(torch.empty(8, 3 * 16 * 128 + 64)[:, : 3 * 16 * 128])
        self.assertTrue(guard.covered(*values))

    def test_reject_layout_dtype_gate_device_and_non_t1(self):
        for attribute, value in (
            ("num_q_heads", 8),
            ("head_v_dim", 64),
            ("lower_bound", float("nan")),
            ("lower_bound", 0.0),
            ("lower_bound", -3.0),
            ("lower_bound", None),
        ):
            values = fixture()
            setattr(values[0], attribute, value)
            self.assertFalse(guard.covered(*values))
        values = list(fixture())
        values[1] = CudaMetadata(torch.empty(4, 3 * 16 * 128))
        self.assertFalse(guard.covered(*values))
        values = list(fixture())
        values[4] = CudaMetadata(torch.empty(3, 16, 128, 128, dtype=torch.bfloat16))
        self.assertFalse(guard.covered(*values))
        values = list(fixture())
        values[1] = values[1].value  # CPU isn't a supported production route
        self.assertFalse(guard.covered(*values))
        values = list(fixture())
        values[2] = CudaMetadata(torch.empty(2, 16 * 128, 2)[:, :, 0])
        self.assertFalse(guard.covered(*values))

    def test_no_packed_replayssm_or_other_backend(self):
        call, env = route()
        dispatcher = SimpleNamespace(
            supports_packed_decode=True,
            decode_kernel=TritonKDAKernel(),
            decode=lambda **kwargs: "generic",
        )
        backend = SimpleNamespace(
            glm_safe_packed_decode=True,
            kernel_dispatcher=dispatcher,
            state_pruner=None,
            _track_mamba_state_decode=lambda *args: None,
        )
        env["replayssm_d"] = object()
        self.assertEqual(call(backend, *fixture()), "generic")
        env["replayssm_d"] = None
        dispatcher.decode_kernel = object()
        self.assertEqual(call(backend, *fixture()), "generic")


if __name__ == "__main__":
    unittest.main()
