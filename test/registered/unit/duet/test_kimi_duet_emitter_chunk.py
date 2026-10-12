"""CPU equivalence and allocation-bound tests for the production KDA emitter."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]


def emitter_class():
    source = ROOT / "python/sglang/srt/models/kimi_linear_duet/runtime.py"
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("DuetKDAEmitter", "kda_state")
    ]
    model = ROOT / "python/sglang/srt/models/kimi_linear_duet/model.py"
    rms = next(
        node
        for node in ast.parse(model.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_rms"
    )
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "_KDAEmitter": object,
    }
    exec(
        compile(ast.Module(body=[rms, *nodes], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["DuetKDAEmitter"], namespace


def fixture(*, heads=2, dim=8, hidden=16, kernel=4, budget=128, seed=72):
    cls, namespace = emitter_class()
    emitter = cls()
    generator = torch.Generator().manual_seed(seed)
    seg = heads * dim

    def weight(*shape):
        return torch.randn(*shape, generator=generator) * 0.1

    attributes = {
        "Hl": heads,
        "Dh": dim,
        "K": kernel,
        "eps": 1e-6,
        "max_emitter_tokens": budget,
        "norm_w": torch.ones(hidden),
        "qkv_w": weight(3 * seg, hidden),
        "conv_w": weight(3 * seg, kernel),
        "f_a_w": weight(dim, hidden),
        "f_b_w": weight(seg, dim),
        "A_log": torch.linspace(-2, 0, heads).view(1, 1, heads, 1),
        "dt_bias": weight(seg),
        "b_w": weight(heads, hidden),
    }
    for name, value in attributes.items():
        setattr(emitter, name, value)
    return emitter, namespace


def cache(emitter, slots, *, conv_dtype=torch.float32):
    return SimpleNamespace(
        conv=[
            torch.zeros(
                slots,
                emitter.K - 1,
                3 * emitter.Hl * emitter.Dh,
                dtype=conv_dtype,
            )
        ],
        temporal=torch.zeros(slots, emitter.Hl, emitter.Dh, emitter.Dh),
    )


class EmitterChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mixed_lengths_full_state_and_tail_elementwise(self):
        torch.manual_seed(17)
        for dtype in (torch.float32, torch.bfloat16):
            for kernel, budget in ((4, 128), (9, 4)):
                with self.subTest(conv_dtype=dtype, kernel=kernel, budget=budget):
                    emitter, _ = fixture(kernel=kernel, budget=budget)
                    lengths = [1, 63, 65, 128, 128, 131]
                    batch = SimpleNamespace(extend_seq_lens_cpu=lengths)
                    slots = torch.tensor([6, 2, 1, 5, 4, 3])
                    hidden = torch.randn(sum(lengths), 16)
                    expected = cache(emitter, 8, conv_dtype=dtype)
                    actual = cache(emitter, 8, conv_dtype=dtype)
                    emitter._emit_reference(hidden, batch, expected, slots)
                    emitter._emit_batched(hidden, batch, actual, slots)
                    torch.testing.assert_close(
                        actual.temporal, expected.temporal, atol=1e-6, rtol=1e-5
                    )
                    torch.testing.assert_close(
                        actual.conv[0], expected.conv[0], atol=1e-6, rtol=1e-5
                    )

    def test_owned_tail_and_workspace_reused_for_all_blocks(self):
        emitter, namespace = fixture(budget=128)
        batch = SimpleNamespace(extend_seq_lens_cpu=[321, 321])
        hidden = torch.randn(642, 16)
        target = cache(emitter, 2)
        outputs, volumes, initial_states, projection_rows = [], [], [], []
        original_scan = namespace["kda_state"]
        original_mm = torch.mm

        def scan(k, v, g, beta, initial_state=None):
            volumes.append(k.shape[0] * k.shape[1])
            initial_states.append(initial_state)
            return original_scan(k, v, g, beta, initial_state)

        def mm(*args, **kwargs):
            outputs.append(kwargs["out"].data_ptr())
            projection_rows.append(args[0].shape[0])
            return original_mm(*args, **kwargs)

        with (
            patch.dict(namespace, kda_state=scan),
            patch.object(torch, "mm", side_effect=mm),
            patch.object(
                emitter, "_emitter_workspace", wraps=emitter._emitter_workspace
            ) as allocate,
        ):
            emitter._emit_batched(hidden, batch, target, torch.tensor([0, 1]))
        self.assertEqual(allocate.call_count, 1)
        self.assertEqual(volumes, [128] * 5 + [2])
        # Last two real tokens use 128 projection rows, never extra scan steps.
        self.assertEqual(projection_rows[::4], [128] * 6)
        self.assertIsNone(initial_states[0])
        self.assertTrue(all(state is not None for state in initial_states[1:]))
        for index in range(4):
            self.assertEqual(len(set(outputs[index::4])), 1)
        self.assertFalse(hasattr(emitter, "_workspace"))

    def test_new_prefix_ignores_previous_pool_state_and_empty_rows(self):
        emitter, _ = fixture(budget=64)
        batch = SimpleNamespace(extend_seq_lens_cpu=[0, 67])
        slots = torch.tensor([1, 3])
        hidden = torch.randn(67, 16)
        actual, expected = cache(emitter, 4), cache(emitter, 4)
        actual.temporal.fill_(19)
        actual.conv[0].fill_(7)
        emitter._emit_batched(hidden, batch, actual, slots)
        emitter._emit_reference(
            hidden,
            SimpleNamespace(extend_seq_lens_cpu=[67]),
            expected,
            slots[1:],
        )
        torch.testing.assert_close(actual.temporal[3], expected.temporal[3])
        torch.testing.assert_close(actual.conv[0][3], expected.conv[0][3])
        self.assertTrue((actual.temporal[1] == 19).all())
        self.assertTrue((actual.conv[0][1] == 7).all())

    def test_long_prompt_plan_and_scan_bound_with_scaled_dimensions(self):
        emitter, namespace = fixture(heads=1, dim=1, hidden=2, budget=4096)
        hidden = torch.zeros(170880, 2)
        batch = SimpleNamespace(extend_seq_lens_cpu=[170880])
        target = cache(emitter, 1)
        shapes = []

        def scan(k, v, g, beta, initial_state=None):
            shapes.append((k.shape, initial_state is None))
            return torch.zeros(1, 1, 1, 1)

        with patch.dict(namespace, kda_state=scan):
            emitter._emit_batched(hidden, batch, target, torch.tensor([0]))
        self.assertEqual(len(shapes), 42)
        self.assertEqual([shape[0][1] for shape in shapes], [4096] * 41 + [2944])
        self.assertEqual(sum(shape[0][1] for shape in shapes), 170880)
        self.assertEqual([shape[1] for shape in shapes], [True] + [False] * 41)
        plan = emitter._batch_plan(hidden, batch)
        self.assertEqual(plan[0][2].shape, (1,))
        self.assertEqual(plan[0][2].numel(), 1)  # No retained T-sized index tensor.
        self.assertIs(plan, emitter._batch_plan(hidden, batch))

    def test_more_than_64_requests_respects_total_token_budget(self):
        emitter, namespace = fixture(heads=1, dim=1, hidden=2, budget=4096)
        batch = SimpleNamespace(extend_seq_lens_cpu=[65] * 65)
        target = cache(emitter, 65)
        volumes = []

        def scan(k, v, g, beta, initial_state=None):
            volumes.append(k.shape[0] * k.shape[1])
            return torch.zeros(k.shape[0], 1, 1, 1)

        with patch.dict(namespace, kda_state=scan):
            emitter._emit_batched(
                torch.zeros(65 * 65, 2), batch, target, torch.arange(65)
            )
        self.assertEqual(volumes, [4096, 64, 65])
        self.assertLessEqual(max(volumes), 4096)

    def test_scan_continuation_bitwise_preserves_initial_state(self):
        _, namespace = fixture()
        scan = namespace["kda_state"]
        generator = torch.Generator().manual_seed(91)
        k = torch.randn(2, 137, 2, 8, generator=generator)
        k /= k.norm(dim=-1, keepdim=True)
        v = torch.randn_like(k)
        decay = -torch.rand(k.shape, generator=generator)
        beta = torch.rand(2, 137, 2, generator=generator)
        initial = torch.randn(2, 2, 8, 8, generator=generator)
        before = initial.clone()
        full = scan(k, v, decay, beta, initial)
        state = initial
        for start, stop in ((0, 64), (64, 128), (128, 137)):
            state = scan(
                k[:, start:stop],
                v[:, start:stop],
                decay[:, start:stop],
                beta[:, start:stop],
                state,
            )
        self.assertTrue(torch.equal(full, state))
        self.assertTrue(torch.equal(initial, before))

    def test_fla_receives_fp32_initial_state_without_changing_flags(self):
        _, namespace = fixture()
        scan = namespace["kda_state"]
        calls = []
        fake = ModuleType("fla.ops.kda")

        def chunk_kda(**kwargs):
            calls.append(kwargs)
            return None, kwargs["initial_state"]

        fake.chunk_kda = chunk_kda
        modules = {
            "fla": ModuleType("fla"),
            "fla.ops": ModuleType("fla.ops"),
            "fla.ops.kda": fake,
        }
        initial = torch.randn(1, 2, 8, 8)
        values = torch.randn(1, 4, 2, 8)
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch.Tensor, "is_cuda", property(lambda _: True)),
        ):
            actual = scan(values, values, -values.abs(), torch.ones(1, 4, 2), initial)
        self.assertIs(actual, initial)
        self.assertIs(calls[0]["initial_state"], initial)
        self.assertTrue(calls[0]["output_final_state"])
        self.assertFalse(calls[0]["use_qk_l2norm_in_kernel"])


if __name__ == "__main__":
    unittest.main()
