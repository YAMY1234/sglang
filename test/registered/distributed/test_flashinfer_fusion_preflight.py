"""Distributed tests for FlashInfer allreduce-fusion workspace preflight."""

import inspect
import multiprocessing as mp
import os
import socket
import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import get_cuda_driver_bindings, is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="stage-b", runner_config="2-gpu-large")

WORLD_SIZE = 2


class _FakeCudaDriver:
    class CUresult:
        CUDA_SUCCESS = 0

    class CUmemAllocationGranularity_flags:
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = object()

    def __init__(self, granularity):
        self.granularity = granularity
        self.allocation_granularity_calls = 0

    def cuMemGetAllocationGranularity(self, prop, flag):
        del prop, flag
        self.allocation_granularity_calls += 1
        return self.CUresult.CUDA_SUCCESS, self.granularity


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_rank(rank, world_size, port, scenario, result_q):
    held = None
    cuda_driver = None
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        torch.cuda.set_device(rank)

        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )
        cpu_group = dist.group.WORLD

        from sglang.srt.layers.flashinfer_comm_fusion import (
            _make_flashinfer_workspace_allocation_prop,
            _preflight_check_workspace_memory,
        )

        probe_kwargs = dict(
            world_size=8,
            max_token_num=2048,
            hidden_dim=12288,
            dtype=torch.bfloat16,
            cpu_group=cpu_group,
        )

        if scenario == "rank0_starved" and rank == 0:
            cuda_driver = get_cuda_driver_bindings()
            prop = _make_flashinfer_workspace_allocation_prop(cuda_driver)

            free, _total = torch.cuda.mem_get_info(rank)
            target = max(free - (1 << 30), 0)
            granularity_flag = (
                cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            )
            err, gran = cuda_driver.cuMemGetAllocationGranularity(
                prop,
                granularity_flag,
            )
            assert err == cuda_driver.CUresult.CUDA_SUCCESS, err
            aligned = (target // gran) * gran
            assert aligned > 0, "not enough free memory to starve the preflight"
            err, held = cuda_driver.cuMemCreate(aligned, prop, 0)
            assert err == cuda_driver.CUresult.CUDA_SUCCESS, (err, aligned)

        decision = _preflight_check_workspace_memory(**probe_kwargs)
        result_q.put((rank, "ok", bool(decision)))
    except Exception as e:  # pragma: no cover - debug path
        result_q.put((rank, "err", repr(e)))
    finally:
        if held is not None:
            cuda_driver.cuMemRelease(held)
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def _spawn_and_collect(scenario, world_size=WORLD_SIZE):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _get_free_port()
    procs = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_run_rank,
            args=(rank, world_size, port, scenario, q),
        )
        proc.start()
        procs.append(proc)

    try:
        results = {}
        for _ in range(world_size):
            rank, status, payload = q.get(timeout=300)
            results[rank] = (status, payload)

        for proc in procs:
            proc.join(timeout=60)
            assert proc.exitcode == 0, f"rank exited with {proc.exitcode}"
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)

    return results


class TestFlashInferPreflightSizeMath(unittest.TestCase):
    def test_bf16_sizes_match_non_multicast_flashinfer_workspace(self):
        from sglang.srt.layers.flashinfer_comm_fusion import (
            _flashinfer_trtllm_workspace_allocation_sizes,
        )

        granularity = 1 << 16
        cuda_driver = _FakeCudaDriver(granularity)
        sizes = _flashinfer_trtllm_workspace_allocation_sizes(
            cuda_driver=cuda_driver,
            prop=object(),
            world_size=4,
            max_token_num=2048,
            hidden_dim=4096,
            dtype=torch.bfloat16,
        )

        self.assertEqual(
            sizes,
            [
                (64 << 20) + granularity,
                granularity,
                (192 << 20) + granularity,
            ],
        )
        self.assertEqual(cuda_driver.allocation_granularity_calls, 3)

    def test_fp32_lamport_size_uses_four_byte_elements(self):
        from sglang.srt.layers.flashinfer_comm_fusion import (
            _flashinfer_trtllm_workspace_allocation_sizes,
        )

        granularity = 1 << 16
        cuda_driver = _FakeCudaDriver(granularity)
        sizes = _flashinfer_trtllm_workspace_allocation_sizes(
            cuda_driver=cuda_driver,
            prop=object(),
            world_size=4,
            max_token_num=2048,
            hidden_dim=4096,
            dtype=torch.float32,
        )

        self.assertEqual(
            sizes,
            [
                (64 << 20) + granularity,
                granularity,
                (384 << 20) + granularity,
            ],
        )
        self.assertEqual(cuda_driver.allocation_granularity_calls, 3)


class TestFlashInferPDLCompletionContract(unittest.TestCase):
    def test_real_wrapper_defaults_to_completion_at_end(self):
        import sglang.srt.layers.flashinfer_comm_fusion  # noqa: F401

        schema = torch.ops.sglang.flashinfer_allreduce_residual_rmsnorm.default._schema
        parameter = next(
            argument
            for argument in schema.arguments
            if argument.name == "trigger_completion_at_end"
        )
        self.assertTrue(parameter.has_default_value())
        self.assertIs(parameter.default_value, True)

    def test_fake_wrapper_matches_real_completion_default(self):
        from sglang.srt.layers.flashinfer_comm_fusion import (
            fake_flashinfer_allreduce_residual_rmsnorm,
        )

        parameter = inspect.signature(
            fake_flashinfer_allreduce_residual_rmsnorm
        ).parameters["trigger_completion_at_end"]
        self.assertIs(parameter.default, True)


class TestFlashInferWorkspaceDecisionProtocol(unittest.TestCase):
    def test_rank_local_reinitialization_is_promoted_to_group_decision(self):
        import sglang.srt.layers.flashinfer_comm_fusion as fusion

        def mark_required(flag, op, group):
            self.assertIs(op, torch.distributed.ReduceOp.MAX)
            self.assertIs(group, cpu_group)
            flag.fill_(1)

        cpu_group = object()
        with patch.object(
            torch.distributed, "get_world_size", return_value=2
        ), patch.object(torch.distributed, "all_reduce", side_effect=mark_required):
            self.assertTrue(
                fusion._sync_workspace_reinitialization_required(False, cpu_group)
            )

    def test_peer_unavailable_vote_updates_local_state(self):
        import sglang.srt.layers.flashinfer_comm_fusion as fusion

        def mark_unavailable(flag, op, group):
            self.assertIs(op, torch.distributed.ReduceOp.MAX)
            self.assertIs(group, cpu_group)
            flag.fill_(1)

        cpu_group = object()
        old_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_allreduce_unavailable = False
            with patch.object(
                torch.distributed, "get_world_size", return_value=2
            ), patch.object(
                torch.distributed, "all_reduce", side_effect=mark_unavailable
            ):
                self.assertTrue(
                    fusion._sync_allreduce_unavailable_across_group(cpu_group)
                )
                self.assertTrue(fusion._flashinfer_allreduce_unavailable)
        finally:
            fusion._flashinfer_allreduce_unavailable = old_unavailable

    def test_availability_sync_failure_is_not_swallowed(self):
        import sglang.srt.layers.flashinfer_comm_fusion as fusion

        old_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_allreduce_unavailable = False
            with patch.object(
                torch.distributed, "get_world_size", return_value=2
            ), patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=RuntimeError("gloo failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "aborting instead of allowing"
                ):
                    fusion._sync_allreduce_unavailable_across_group(object())
                self.assertTrue(fusion._flashinfer_allreduce_unavailable)
        finally:
            fusion._flashinfer_allreduce_unavailable = old_unavailable

    def test_peer_initialization_failure_disables_local_success(self):
        import sglang.srt.layers.flashinfer_comm_fusion as fusion

        class FakeCoordinator:
            device_group = object()
            cpu_group = object()

        class FakeManager:
            initialized = True
            world_size = 2
            rank = 0
            group = (FakeCoordinator.device_group, FakeCoordinator.cpu_group)
            cleaned = False

            def is_buffer_size_sufficient(self, **kwargs):
                del kwargs
                return False

            def initialize(self, **kwargs):
                del kwargs
                self.initialized = True

            def cleanup(self):
                self.cleaned = True
                self.initialized = False

        manager = FakeManager()
        old_unavailable = fusion._flashinfer_allreduce_unavailable
        old_comm = fusion._flashinfer_comm
        try:
            fusion._flashinfer_allreduce_unavailable = False
            fusion._flashinfer_comm = object()
            with patch.object(
                fusion, "is_flashinfer_available", return_value=True
            ), patch.object(
                fusion,
                "get_attn_tensor_model_parallel_world_size",
                return_value=2,
            ), patch.object(
                fusion, "get_attn_tensor_model_parallel_rank", return_value=0
            ), patch.object(
                fusion, "get_attn_tp_group", return_value=FakeCoordinator()
            ), patch.object(
                fusion, "_get_workspace_manager", return_value=manager
            ), patch.object(
                fusion,
                "_sync_workspace_reinitialization_required",
                return_value=True,
            ), patch.object(
                fusion,
                "_sync_allreduce_unavailable_across_group",
                return_value=True,
            ):
                self.assertFalse(
                    fusion.ensure_workspace_initialized(
                        synchronize_reinitialization=True
                    )
                )
        finally:
            fusion._flashinfer_allreduce_unavailable = old_unavailable
            fusion._flashinfer_comm = old_comm

        self.assertTrue(manager.cleaned)


class TestFlashInferPreflightDistributed(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
            raise unittest.SkipTest(
                f"Need {WORLD_SIZE} CUDA devices, got {torch.cuda.device_count()}"
            )
        if not is_flashinfer_available():
            raise unittest.SkipTest("FlashInfer is not available")
        try:
            from sglang.srt.layers.flashinfer_comm_fusion import (
                _make_flashinfer_workspace_allocation_prop,
            )

            cuda_driver = get_cuda_driver_bindings()
            _make_flashinfer_workspace_allocation_prop(cuda_driver)
        except Exception as e:
            raise unittest.SkipTest(
                f"FlashInfer preflight dependencies unavailable: {e}"
            )

    def test_happy_path_votes_proceed(self):
        results = _spawn_and_collect("normal")
        for rank, (status, payload) in results.items():
            self.assertEqual(status, "ok", f"rank {rank}: {payload}")
            self.assertTrue(payload, f"rank {rank} voted SKIP unexpectedly")

    def test_starved_rank_broadcasts_skip(self):
        results = _spawn_and_collect("rank0_starved")
        for rank, (status, payload) in results.items():
            self.assertEqual(status, "ok", f"rank {rank}: {payload}")
            self.assertFalse(
                payload,
                f"rank {rank} voted PROCEED but rank 0 was starved",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
