"""CPU lifecycle/numerical gate for the production KDA batched cut.

The CPU MGS helper is the shared Flash-Next implementation, loaded without
bootstrapping SGLang's serving dependencies. CUDA uses its matching Triton MGS.
"""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
sys.path.insert(0, str(ROOT / "test/registered/unit/duet"))
from lightning_duet._common import load

load("state_factor")
from kimi_linear_duet import fast_state
from test_kimi_duet_runtime import Pool, policy_class


def direct(name, relative):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_shared_mgs():
    direct(
        "sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference",
        "python/sglang/srt/layers/attention/linear/kernels/gdn_prefill_reference.py",
    )
    return direct(
        "sglang.srt.mem_cache.gdn_factored_pool",
        "python/sglang/srt/mem_cache/gdn_factored_pool.py",
    )


class FastpathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        load_shared_mgs()

    def policy(self, *, every=4, explicit=True):
        cls, _ = policy_class()
        torch.manual_seed(62)
        pool = Pool()
        pool.mamba_pool.mamba_cache.temporal.normal_()
        return cls(
            pool,
            torch.randn(3, 2, 32),
            dict(
                state_rank=8,
                state_every=every,
                state_sink="explicit" if explicit else "implicit",
            ),
            [0, 2],
            graph_safe=True,
        )

    def test_one_project_call_for_all_layers_slots_with_independent_masks(self):
        policy = self.policy()
        policy.pending_prefix[:, [1, 3]] = True
        untouched = policy.states[:, [0, 2]].clone()
        with patch.object(fast_state, "project", wraps=fast_state.project) as project:
            policy.flush(torch.tensor([1, 3, -1, 1]), prefix=True, layer_id=0)
            policy.flush(torch.tensor([1, 3]), prefix=True, layer_id=2)
        self.assertEqual(project.call_count, 1)
        self.assertEqual(project.call_args.args[0].shape, (4, 2, 32, 32))
        self.assertEqual(policy.prefix_cuts, 4)
        self.assertTrue(torch.equal(untouched, policy.states[:, [0, 2]]))
        self.assertFalse(policy.pending_prefix.any())
        self.assertEqual(set(policy.warm), {(l, s) for l in [0, 1] for s in [1, 3]})

    def test_layer_boundary_matches_reference_reads_then_cuts(self):
        fast = self.policy()
        cls, _ = policy_class()
        reference = cls(
            Pool(), torch.randn(3, 2, 32), dict(state_rank=8, state_every=4), [0, 2]
        )
        reference.vbar.copy_(fast.vbar)
        reference.states.copy_(fast.states)
        slots = torch.tensor([1, 2])
        fast.pending_prefix[:, slots] = reference.pending_prefix[:, slots] = True
        for step in range(8):
            for layer in [0, 2]:
                fast.flush(slots, prefix=True, layer_id=layer)
                reference.flush(slots, prefix=True, layer_id=layer)
                index = fast.layer_map[layer]
                torch.testing.assert_close(
                    fast.states[index, slots],
                    reference.states[index, slots],
                    atol=4e-5,
                    rtol=4e-5,
                )
                # Stand in for the same dense recurrence/output on this layer.
                fast.states[index, slots] += 0.001
                reference.states[index, slots] += 0.001
                descriptor = SimpleNamespace(layer_id=layer)
                for policy in [fast, reference]:
                    policy.decode(None, descriptor, None, None, None, slots, None)
                    policy.flush(slots, prefix=False, layer_id=layer)
        self.assertTrue(torch.equal(fast.count, reference.count))
        self.assertEqual(fast.prefix_cuts, reference.prefix_cuts)
        self.assertEqual(fast.decode_cuts, reference.decode_cuts)

    def test_mixed_fresh_warm_reset_copy_and_cpu_restore(self):
        policy = self.policy()
        slots = torch.tensor([1, 2])
        policy.pending_prefix[:, slots] = True
        policy.flush(slots, prefix=True)
        saved_basis = policy.warm[0, 1].clone()
        policy.copy_slots(torch.tensor([1]), torch.tensor([3]))
        self.assertNotEqual(policy.warm[0, 3].data_ptr(), policy.warm[0, 1].data_ptr())
        saved = policy.get_cpu_slots(torch.tensor([3]))
        policy.reset_slots(torch.tensor([3]))
        policy.load_cpu_slots(saved, torch.tensor([3]))
        self.assertTrue(torch.equal(saved_basis, policy.warm[0, 3]))
        policy.mark_prefix(0, torch.tensor([2]))
        policy.count[:, slots] = 4
        with patch.object(fast_state, "project", wraps=fast_state.project) as project:
            policy.flush(slots, prefix=False)
        self.assertEqual(project.call_args.args[4].tolist(), [True, True, False, True])

    def test_one_fp64_small_eigh_call_for_20_layers_32_heads(self):
        torch.manual_seed(19)
        state = torch.randn(20, 32, 128, 128)
        direction = torch.randn(20, 32, 128)
        with patch.object(
            fast_state, "small_eigh", wraps=fast_state.small_eigh
        ) as solve:
            fast_state.factorize(
                state,
                direction,
                32,
                torch.zeros(20, 32, 128, 32),
                torch.zeros(20, dtype=torch.bool),
                fast_state.probes(32, 128, 128, 32, "cpu"),
            )
        self.assertEqual(solve.call_count, 1)
        gram = solve.call_args.args[0]
        self.assertEqual(gram.shape, (20, 32, 40, 40))
        self.assertEqual(gram.dtype, torch.float64)

    def test_unpadded_jacobi_cpu_mirror_matches_fp64_reference(self):
        from kimi_linear_duet.small_eigh import jacobi_cpu

        torch.manual_seed(43)
        a = torch.randn(5, 40, 40, dtype=torch.float64)
        gram = a @ a.transpose(-1, -2)
        gram[0].zero_()
        diagnostics = {}
        values, vectors = jacobi_cpu(gram, diagnostics=diagnostics)
        expected, basis = torch.linalg.eigh(gram)
        torch.testing.assert_close(values, expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(
            (vectors * values[..., None, :]) @ vectors.transpose(-1, -2),
            gram,
            atol=1e-10,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            vectors[..., -32:] @ vectors[..., -32:].transpose(-1, -2),
            basis[..., -32:] @ basis[..., -32:].transpose(-1, -2),
            atol=1e-11,
            rtol=1e-11,
        )
        self.assertEqual(diagnostics["sweeps"][0], 0)
        self.assertLess(max(diagnostics["sweeps"]), 12)

    def test_arbitrary_rank_gram_dispatch_preserves_shared_fallback(self):
        from kimi_linear_duet.small_eigh import small_eigh

        for width in (9, 73):
            gram = torch.diag(torch.arange(width, dtype=torch.float64))[None]
            values, vectors = small_eigh(gram)
            torch.testing.assert_close(values, gram.diagonal(dim1=-2, dim2=-1))
            self.assertEqual(vectors.shape, gram.shape)

    def test_graph_snapshot_uses_one_read_and_preserves_cut_boundaries(self):
        fast = self.policy()
        reference = self.policy()
        reference.fastpath = False
        reference.vbar.copy_(fast.vbar)
        reference.states.copy_(fast.states)
        slots = torch.tensor([1, 2])
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
        fast.pending_prefix[:, slots] = reference.pending_prefix[:, slots] = True
        with (
            patch.object(fast, "slots", return_value=slots),
            patch.object(reference, "slots", return_value=slots),
            patch.object(fast, "_read_metadata", wraps=fast._read_metadata) as read,
        ):
            for step in range(8):
                for policy in (fast, reference):
                    policy.before_graph(batch)
                    for layer in (0, 2):
                        policy.decode(
                            None,
                            SimpleNamespace(layer_id=layer),
                            None,
                            None,
                            None,
                            slots,
                            None,
                        )
                    policy.states[:, slots] += 0.001
                    policy.after_graph(batch)
                self.assertTrue(torch.equal(fast.count, reference.count))
                self.assertEqual(fast.decode_cuts, reference.decode_cuts)
                torch.testing.assert_close(
                    fast.states, reference.states, atol=4e-5, rtol=4e-5
                )
            self.assertEqual(read.call_count, 8)

    def test_zero_and_rank_deficient_states_finite_and_exact_reconstruction(self):
        torch.manual_seed(17)
        direction = torch.randn(3, 2, 32)
        state = torch.randn(3, 2, 32, 1) @ direction[..., None, :]
        state[0] = 0
        previous = torch.zeros(3, 2, 32, 8)
        for explicit in [False, True]:
            out, basis = fast_state.project(
                state,
                direction,
                8,
                previous,
                torch.zeros(3, dtype=torch.bool),
                fast_state.probes(2, 32, 32, 8, "cpu"),
                explicit=explicit,
            )
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(basis).all())
            torch.testing.assert_close(out, state, atol=2e-5, rtol=2e-5)

    def test_warm_allocations_owned_per_slot(self):
        policy = self.policy()
        policy.pending_prefix[:, [1, 2]] = True
        policy.flush(torch.tensor([1, 2]), prefix=True)
        storage1 = policy.warm[0, 1].untyped_storage()
        storage2 = policy.warm[0, 2].untyped_storage()
        self.assertNotEqual(storage1.data_ptr(), storage2.data_ptr())
        self.assertEqual(storage1.nbytes(), 2 * 2 * 32 * 8 * 4)
        policy.reset_slots(torch.tensor([2]))
        self.assertEqual(set(policy.warm), {(0, 1), (1, 1)})

    def test_batched_emitter_mixed_lengths_preserves_recurrence_and_conv_tails(self):
        runtime = ROOT / "python/sglang/srt/models/kimi_linear_duet/runtime.py"
        nodes = [
            n
            for n in ast.parse(runtime.read_text()).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in ("DuetKDAEmitter", "kda_state")
        ]
        model = ROOT / "python/sglang/srt/models/kimi_linear_duet/model.py"
        rms = next(
            n
            for n in ast.parse(model.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_rms"
        )
        ns = dict(torch=torch, F=torch.nn.functional, _KDAEmitter=object)
        exec(
            compile(
                ast.Module(body=[rms, *nodes], type_ignores=[]), str(runtime), "exec"
            ),
            ns,
        )
        emitter = ns["DuetKDAEmitter"]()
        torch.manual_seed(84)
        heads, dim, hidden, kernel = 2, 8, 16, 4
        seg = heads * dim
        weights = dict(
            norm_w=torch.ones(hidden),
            qkv_w=torch.randn(3 * seg, hidden) * 0.1,
            conv_w=torch.randn(3 * seg, kernel) * 0.1,
            f_a_w=torch.randn(dim, hidden) * 0.1,
            f_b_w=torch.randn(seg, dim) * 0.1,
            A_log=torch.zeros(1, 1, heads, 1),
            dt_bias=torch.zeros(seg),
            b_w=torch.randn(heads, hidden) * 0.1,
        )
        for name, value in dict(
            Hl=heads, Dh=dim, K=kernel, eps=1e-6, **weights
        ).items():
            setattr(emitter, name, value)
        batch = SimpleNamespace(extend_seq_lens_cpu=[1, 4, 4, 7])
        slots = torch.tensor([4, 1, 3, 2])
        h = torch.randn(16, hidden)

        def cache():
            return SimpleNamespace(
                conv=[torch.zeros(6, kernel - 1, 3 * seg)],
                temporal=torch.zeros(6, heads, dim, dim),
            )

        reference, actual = cache(), cache()
        emitter._emit_reference(h, batch, reference, slots)
        emitter._emit_batched(h, batch, actual, slots)
        torch.testing.assert_close(
            actual.conv[0], reference.conv[0], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            actual.temporal, reference.temporal, atol=1e-6, rtol=1e-5
        )
        plan = emitter._batch_plan(h, batch)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[1][1].tolist(), [1, 2])
        self.assertEqual(plan[1][2].tolist(), [[1, 2, 3, 4], [5, 6, 7, 8]])
        emitter._batch_plan(h[:1], SimpleNamespace(extend_seq_lens_cpu=[1]))
        self.assertIs(plan, emitter._batch_plan(h, batch))

    def test_prefix_only_and_full_rank(self):
        policy = self.policy(every=0)
        policy.mark_prefix(0, torch.tensor([1]))
        policy.flush(torch.tensor([1]), prefix=True)
        state = policy.states.clone()
        policy.count.fill_(100)
        policy.flush(torch.tensor([1]), prefix=False)
        self.assertTrue(torch.equal(state, policy.states))
        policy.rank = 32
        policy.pending_prefix[:, 1] = True
        policy.flush(torch.tensor([1]), prefix=True)
        self.assertTrue(torch.equal(state, policy.states))
        self.assertFalse(policy.warm)


if __name__ == "__main__":
    unittest.main()
