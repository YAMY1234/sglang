"""Every cross-rank spin wait on the decode path gets a deadline below --watchdog-timeout,
and every launcher of such a kernel hands it to the kernel; the GPU side (a missing
peer traps at the deadline) is test_cross_rank_wait_deadline under kernels/ops."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.communication import all_reduce_fusion, all_reduce_mhc
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.monokernels import guards
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORLD = 4
MHC_HIDDEN = 5120
# Arbitrary; only has to differ from every derived deadline.
SENTINEL_NS = 1234


def _run_op_body(module, name: str):
    """Route a custom op registered for the CUDA key to its Python body on CPU tensors."""
    op = getattr(torch.ops.sglang, name).default
    keys = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return patch.object(module, name, lambda *args: op.redispatch(keys, *args))


def _publish_watchdog(watchdog_timeout: float) -> None:
    reset_context()
    publish(
        ServerArgs(model_path="dummy", watchdog_timeout=watchdog_timeout), role="test"
    )


class TestWaitDeadline(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        patcher = patch.object(all_reduce_fusion, "_WAIT_DEADLINE_NS", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_deadline_is_half_the_watchdog_capped_at_a_minute(self):
        reset_context()
        # Offline use and tests have no watchdog: the cap, never unbounded.
        self.assertEqual(guards.wait_deadline_ns(), 60 * 10**9)
        _publish_watchdog(30)
        self.assertEqual(guards.wait_deadline_ns(), 15 * 10**9)
        _publish_watchdog(3600)
        self.assertEqual(guards.wait_deadline_ns(), 60 * 10**9)

    def test_push_plane_deadline_is_read_once_after_publication(self):
        reset_context()
        self.assertEqual(all_reduce_fusion.wait_deadline_ns(), 60 * 10**9)
        _publish_watchdog(30)
        with patch.object(guards, "get_device", wraps=guards.get_device) as reads:
            for _ in range(2):
                self.assertEqual(all_reduce_fusion.wait_deadline_ns(), 15 * 10**9)
        self.assertEqual(reads.call_count, 1)

    def test_every_push_plane_launcher_passes_the_deadline(self):
        def t(*shape, dtype=torch.bfloat16):
            return torch.zeros(*shape, dtype=dtype)

        idx, weights = t(1, dtype=torch.int32), t(1, 1, dtype=torch.float32)
        mhc = (t(1, 4, MHC_HIDDEN), t(1, 4, dtype=torch.float32))
        comb = t(1, 4, 4, dtype=torch.float32)
        norm = (t(1, 4, dtype=torch.float32), t(MHC_HIDDEN), 1e-6)
        comm = SimpleNamespace(world_size=WORLD)
        module = MagicMock()
        _publish_watchdog(30)
        launchers = {
            "moe_finalize_all_reduce": lambda: (
                all_reduce_fusion.moe_finalize_all_reduce(
                    t(1, 2048), idx, weights, 1, world_size=WORLD, hidden_dim=2048
                )
            ),
            "moe_finalize_shared_gate_all_reduce": lambda: (
                all_reduce_fusion.moe_finalize_shared_gate_all_reduce(
                    t(1, 2048), idx, weights, t(1, 2048), t(1), comm
                )
            ),
            "moe_finalize_all_reduce_mhc_post": lambda: (
                all_reduce_mhc.moe_finalize_all_reduce_mhc_post(
                    t(1, MHC_HIDDEN),
                    idx,
                    weights,
                    1,
                    None,
                    *mhc,
                    comb,
                    world_size=WORLD,
                )
            ),
            "moe_finalize_all_reduce_mhc_post_combine_norm": lambda: (
                all_reduce_mhc.moe_finalize_all_reduce_mhc_post_combine_norm(
                    t(1, MHC_HIDDEN),
                    idx,
                    weights,
                    1,
                    None,
                    *mhc,
                    comb,
                    *norm,
                    world_size=WORLD,
                )
            ),
            "moe_finalize_all_reduce_mhc_post_combine_norm_quant": lambda: (
                all_reduce_mhc.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
                    t(1, MHC_HIDDEN),
                    idx,
                    weights,
                    1,
                    None,
                    *mhc,
                    comb,
                    *norm,
                    world_size=WORLD,
                )
            ),
        }
        with (
            patch.dict(all_reduce_fusion._COMM_MAP, {WORLD: comm}),
            patch.object(all_reduce_fusion, "_jit_module", return_value=module),
            patch.object(
                all_reduce_fusion, "_jit_shared_gate_module", return_value=module
            ),
            patch.object(all_reduce_mhc, "_jit_mhc_module", return_value=module),
            patch.object(all_reduce_mhc, "_jit_mhc_quant_module", return_value=module),
            _run_op_body(all_reduce_fusion, "_moe_finalize_all_reduce_op"),
            _run_op_body(all_reduce_mhc, "_moe_finalize_all_reduce_mhc_post_op"),
            _run_op_body(
                all_reduce_mhc, "_moe_finalize_all_reduce_mhc_post_combine_norm_op"
            ),
            _run_op_body(
                all_reduce_mhc,
                "_moe_finalize_all_reduce_mhc_post_combine_norm_quant_op",
            ),
        ):
            for name, launch in launchers.items():
                with self.subTest(name):
                    module.reset_mock()
                    launch()
                    (call,) = module.method_calls
                    self.assertEqual(call.args[-1], 15 * 10**9)


class TestAllGatherDeadline(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        hidden = WORLD * 64
        self.state = SimpleNamespace(
            world_size=WORLD,
            rank_in_group=1,
            max_token_num=8,
            hidden_dim=hidden,
            comm_buff=torch.zeros(8, hidden, dtype=torch.bfloat16),
            symm_mem_hdl=SimpleNamespace(
                rank=1, world_size=WORLD, multicast_ptr=1, signal_pad_ptrs_dev=2
            ),
        )
        self.shard = torch.zeros(8, 64, dtype=torch.bfloat16)
        self.launches = []
        kernel = MagicMock()
        kernel.__getitem__.return_value = lambda **kw: self.launches.append(kw)
        patcher = patch.object(triton_symm_mem_ag, "_all_gather_kernel_inner", kernel)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _gather(self, **kw):
        return triton_symm_mem_ag.all_gather_inner(
            self.state, self.shard, tp_hidden_dim=WORLD * 64, **kw
        )

    def test_barrier_deadline_reaches_the_kernel(self):
        _publish_watchdog(30)
        self._gather()
        self._gather(timeout_ns=SENTINEL_NS)
        gatherer = triton_symm_mem_ag.MultimemAllGatherer(max_tokens=8, enabled=False)
        # The deadline is fixed when the gatherer is built, as a graph capture fixes it.
        _publish_watchdog(3600)
        gatherer._state = self.state
        gatherer(self.shard)
        self.assertEqual(
            [kw["timeout_ns"] for kw in self.launches],
            [15 * 10**9, SENTINEL_NS, 15 * 10**9],
        )

    def test_a_non_positive_deadline_is_refused_before_launch(self):
        for timeout_ns in (0, -1):
            with self.subTest(timeout_ns=timeout_ns), self.assertRaises(AssertionError):
                self._gather(timeout_ns=timeout_ns)
        self.assertEqual(self.launches, [])


if __name__ == "__main__":
    unittest.main()
