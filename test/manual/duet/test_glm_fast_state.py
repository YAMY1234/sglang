# ruff: noqa: I001
# Bootstrap lightweight serving modules before importing the GLM adapter.
"""CPU routing, batching, TP probe and warm-storage tests for GLM W32 cuts."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_kimi_duet_fastpath import load_shared_mgs
from test_glm_duet import _Pool, glm_state_policy
from glm_duet import _fast_state, _state_factor
from glm_slot_fallback_receipt import evaluate as fallback_receipt


class GlmFastStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        load_shared_mgs()

    def policy(self):
        gen = torch.Generator().manual_seed(194)
        pool = _Pool(torch.randn(3, 5, 2, 48, 48, generator=gen), torch.arange(5))
        return glm_state_policy.GlmStatePolicy(
            pool,
            torch.randn(3, 2, 48, generator=gen),
            {"state_rank": 32, "state_every": 32, "state_sink": "explicit"},
            [0, 1, 2],
            omega_heads=(2, 8),
        )

    def test_prompt_unchanged_periodic_one_dispatch_and_probe_cache(self):
        p = self.policy()
        before = p.states.clone()
        with patch.object(
            _fast_state.fast_state, "project", wraps=_fast_state.fast_state.project
        ) as project:
            p._cut([1, 3], fresh=True)
            self.assertEqual(project.call_count, 0)
            p._cut([1, 3], fresh=False)
            self.assertEqual(project.call_count, 1)
            self.assertEqual(project.call_args.args[0].shape, (6, 2, 48, 48))
            omega = p._fast_omega
            p._cut([1], fresh=False)
            self.assertIs(p._fast_omega, omega)
        self.assertTrue(torch.equal(p.states[:, [0, 2, 4]], before[:, [0, 2, 4]]))
        self.assertEqual(set(p.warm), {(l, s) for l in range(3) for s in [1, 3]})

    def test_mixed_warm_rows_chunking_and_owned_storage(self):
        p = self.policy()
        p._cut([1], fresh=True)
        with (
            patch.object(_state_factor, "ROW_CHUNK", 2),
            patch.object(
                _fast_state.fast_state, "project", wraps=_fast_state.fast_state.project
            ) as project,
        ):
            p._cut([1, 3], fresh=False)
            self.assertEqual(project.call_count, 3)
            masks = [call.args[4].tolist() for call in project.call_args_list]
            self.assertEqual(masks, [[True, False]] * 3)
        values = list(p.warm.values())
        self.assertEqual(len({v.untyped_storage().data_ptr() for v in values}), 6)
        for value in values:
            self.assertEqual(
                value.untyped_storage().nbytes(), value.numel() * value.element_size()
            )

    def test_full_model_probe_draw_then_head_slice(self):
        state = torch.zeros(3, 8, 48, 48)
        full = _fast_state.probes(state, 32, None)
        part = _fast_state.probes(state[:, 2:4], 32, (2, 8))
        for got, want in zip(part, full):
            self.assertTrue(torch.equal(got, want[:, 2:4]))
        gen = torch.Generator().manual_seed(194)
        state.normal_(generator=gen)
        direction = torch.randn(3, 8, 48, generator=gen)
        whole, _bases = _fast_state.project_state_batched(
            state, direction, 32, [None] * 3, full
        )
        sliced, _ = _fast_state.project_state_batched(
            state[:, 2:4], direction[:, 2:4], 32, [None] * 3, part
        )
        torch.testing.assert_close(sliced, whole[:, 2:4], rtol=1e-6, atol=1e-6)

    def test_reset_copy_and_restore_warm_lifecycle(self):
        p = self.policy()
        p._cut([1], fresh=False)
        p.h_count[1] = 17
        saved = {key: value.clone() for key, value in p.warm.items()}
        p.copy_slots(torch.tensor([1]), torch.tensor([3]))
        self.assertEqual(p.h_count[3], 17)
        for layer in range(3):
            self.assertTrue(torch.equal(p.warm[layer, 3], saved[layer, 1]))
            self.assertNotEqual(
                p.warm[layer, 3].data_ptr(), p.warm[layer, 1].data_ptr()
            )
        exported = p.get_cpu_slots(torch.tensor([3]))
        p.reset_slots(torch.tensor([1, 3]))
        self.assertFalse(p.warm)
        p.load_cpu_slots(exported, torch.tensor([4]))
        for layer in range(3):
            self.assertTrue(torch.equal(p.warm[layer, 4], saved[layer, 1]))

    def test_host_schedule_projects_after_exactly_32_steps(self):
        p = self.policy()
        p.mark_prompt_end([1, 3], [1, 3])
        indices = torch.tensor([1, 3])
        batch = SimpleNamespace(
            req_pool_indices=indices, req_pool_indices_cpu=indices, batch_size=2
        )
        p.before_graph(batch)
        with patch.object(
            _fast_state.fast_state, "project", wraps=_fast_state.fast_state.project
        ) as project:
            for step in range(1, 65):
                p.after_graph(batch)
                self.assertEqual(project.call_count, step // 32)
                self.assertEqual(p.h_count[1], step % 32)
                self.assertEqual(p.steps[3], step)
        self.assertEqual(p.prefix_cuts, 6)
        self.assertEqual(p.decode_cuts, 12)

    def test_fallback_counts_hooks_logs_once_and_preserves_slots(self):
        p = self.policy()
        p.mark_prompt_end([1, 3], [1, 3])
        indices = torch.tensor([1, 3])
        batch = SimpleNamespace(
            req_pool_indices=indices, req_pool_indices_cpu=indices, batch_size=2
        )
        with patch.object(glm_state_policy.logger, "warning") as warning:
            self.assertEqual(p._batch_slots(batch), [1, 3])
            self.assertEqual(p.slot_fallback_count, 0)
            batch.req_pool_indices_cpu = None
            self.assertEqual(p._batch_slots(batch), [1, 3])
            batch.req_pool_indices_cpu = indices[:1]
            self.assertEqual(p._batch_slots(batch), [1, 3])
            batch.req_pool_indices_cpu = batch.req_pool_indices = torch.tensor([2])
            batch.batch_size = 1
            self.assertEqual(p._batch_slots(batch), [2])
            self.assertEqual(warning.call_count, 1)
            self.assertEqual(p.runtime_diagnostics(), {"slot_fallback_count": 3})
            p.reset_slots(torch.tensor([1]))
            self.assertEqual(p.slot_fallback_count, 3)

    def test_scheduler_internal_state_exports_live_counter_without_tensor_reads(self):
        # Execute the real endpoint method without importing CUDA-serving
        # dependencies. Its context/memory providers are CPU-only test doubles.
        root = Path(__file__).resolve().parents[3]
        tree = ast.parse((root / "python/sglang/srt/managers/scheduler.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_internal_state"
        )
        namespace = {
            "GetInternalStateReq": SimpleNamespace,
            "GetInternalStateReqOutput": SimpleNamespace,
            "get_context": lambda: SimpleNamespace(resolved_server_args_dict=dict),
            "compute_world_size": lambda **kw: 4,
            "get_parallel": lambda: SimpleNamespace(
                enable_dp_attention=False, dp_size=1, tp_size=4, pp_size=1
            ),
            "build_memory_usage": lambda **kw: kw,
            "get_exec": lambda: SimpleNamespace(
                moe=SimpleNamespace(elastic_ep_backend=None)
            ),
            "RECORD_STEP_TIME": False,
            "envs": SimpleNamespace(
                SGLANG_EXPOSE_OWN_ENV_VARS=SimpleNamespace(get=lambda: False)
            ),
            "msgspec_to_builtins": lambda value: value,
        }
        exec(  # noqa: S102 -- execute the repository endpoint, not user input
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(root / "python/sglang/srt/managers/scheduler.py"),
                "exec",
            ),
            namespace,
        )
        p = self.policy()
        p.slot_fallback_count = 3
        runner = SimpleNamespace(weight_load_mem_usage=0, duet_state_policy=p)
        scheduler = SimpleNamespace(
            metrics_reporter=SimpleNamespace(
                last_gen_throughput=0, spec_total_num_forward_ct=0
            ),
            draft_worker=None,
            tp_worker=SimpleNamespace(model_runner=runner, graph_memory_usage=0),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(mem_usage=0)
            ),
            startup_available_gpu_memory_gb=0,
            max_total_num_tokens=128,
            swa_tokens_per_layer=None,
            startup_time=0,
            max_running_requests=128,
            spec_algorithm=SimpleNamespace(
                is_none=lambda: True, is_dspark=lambda: False
            ),
        )
        result = namespace["get_internal_state"](scheduler, None).internal_state
        self.assertEqual(result["duet_state_policy"], {"slot_fallback_count": 3})
        del runner.duet_state_policy
        result = namespace["get_internal_state"](scheduler, None).internal_state
        self.assertNotIn("duet_state_policy", result)

    def test_measurement_receipt_requires_every_reported_count_zero(self):
        snapshot = {
            "internal_states": [
                {"duet_state_policy": {"slot_fallback_count": 0}},
                {"duet_state_policy": {"slot_fallback_count": 0}},
            ]
        }
        self.assertTrue(fallback_receipt([("server.json", snapshot)])["passed"])
        snapshot["internal_states"][1]["duet_state_policy"]["slot_fallback_count"] = 1
        result = fallback_receipt([("server.json", snapshot)])
        self.assertFalse(result["passed"])
        self.assertIn("!= 0", result["errors"][0])

    def test_measurement_receipt_does_not_invent_missing_zero(self):
        for info in (
            {},
            {"internal_states": []},
            {"internal_states": [{}]},
            {
                "internal_states": [
                    {"duet_state_policy": {"slot_fallback_count": False}}
                ]
            },
        ):
            self.assertFalse(fallback_receipt([("server.json", info)])["passed"])


if __name__ == "__main__":
    unittest.main()
