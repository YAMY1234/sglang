"""Correctness test for the symmetric-memory multimem all-gather kernel.

Compares ``all_gather_inner`` (concat-along-hidden multimem.st gather) against
NCCL all-gather for a sweep of token counts, hidden widths, and the
``safe`` / ``skip_entry_sync`` knobs, in both eager and CUDA-graph modes.

Usage::

    # Run on the default world sizes (2, 4, 8 GPUs):
    python test/registered/kernels/ops/communication/test_symm_mem_all_gather.py
    # Pick a specific world size (or comma-separated list):
    python test/registered/kernels/ops/communication/test_symm_mem_all_gather.py --num-gpu 4
    python test/registered/kernels/ops/communication/test_symm_mem_all_gather.py --num-gpu 2,4,8
    # Extra pytest args (forwarded to each torchrun worker):
    python test/registered/kernels/ops/communication/test_symm_mem_all_gather.py -k 16384
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once, get_ci_test_range
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
    all_gather_inner,
    create_state,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=38, stage="extra-b", runner_config="8-gpu-h200")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=70, stage="nightly", runner_config="8-gpu-h200")

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# Full gathered hidden width H (per-rank shard is H / world_size). Each value
# is a multiple of 8 * 8 so it stays valid for world sizes 2 / 4 / 8.
TEST_HIDDEN = [2048, 7168, 16384]
TEST_NUM_TOKENS = [1, 8, 16, 128]
TEST_LOOP = 8

TEST_HIDDEN = get_ci_test_range(TEST_HIDDEN, [7168])
TEST_NUM_TOKENS = get_ci_test_range(TEST_NUM_TOKENS, [16])

MAX_HIDDEN = max(TEST_HIDDEN)
MAX_TOKENS = max(TEST_NUM_TOKENS)

# ---------------------------------------------------------------------------
# Per-rank distributed setup (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group_once() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    get_parallel().override_permanently(world_group=ps._WORLD)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    cpu_group = ps._WORLD.cpu_group
    assert isinstance(cpu_group, dist.ProcessGroup)
    return cpu_group


@cache_once
def _init_nccl_group_once() -> dist.ProcessGroup:
    _init_cpu_group_once()
    coord = ps._WORLD
    assert coord is not None and coord.device_group is not None
    return coord.device_group


@cache_once
def _init_state_once():
    _init_cpu_group_once()
    coord = ps._WORLD
    return create_state(
        group=coord.device_group,
        rank_in_group=coord.rank_in_group,
        max_tokens=MAX_TOKENS,
        hidden_size=MAX_HIDDEN,
    )


def _nccl_all_gather(x: torch.Tensor, group: dist.ProcessGroup, world_size: int):
    """Reference gather matching ``tensor_model_parallel_all_gather(dim=-1)``:
    concat per-rank ``[T, H/W]`` shards in rank order into ``[T, H]``."""
    num_tokens, local_hidden = x.shape
    gathered = torch.empty(
        world_size * num_tokens, local_hidden, dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(gathered, x.contiguous(), group=group)
    return (
        gathered.reshape(world_size, num_tokens, local_hidden)
        .movedim(0, 1)
        .reshape(num_tokens, world_size * local_hidden)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skip_entry_sync", [False, True])
@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("hidden", TEST_HIDDEN)
@pytest.mark.parametrize("num_tokens", TEST_NUM_TOKENS)
@torch.inference_mode()
def test_symm_mem_all_gather(
    num_tokens: int,
    hidden: int,
    safe: bool,
    skip_entry_sync: bool,
) -> None:
    nccl_group = _init_nccl_group_once()
    state = _init_state_once()
    world_size = state.world_size
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    if state.symm_mem_hdl.multicast_ptr == 0:
        pytest.skip(f"multimem multicast unavailable for world_size={world_size}")

    local_hidden = hidden // world_size
    if hidden % world_size != 0 or local_hidden % 8 != 0:
        pytest.skip(f"hidden={hidden} incompatible with world_size={world_size}")

    def gather(x: torch.Tensor) -> torch.Tensor:
        return all_gather_inner(
            state,
            x,
            tp_hidden_dim=hidden,
            skip_entry_sync=skip_entry_sync,
            safe=safe,
        ).clone()

    for _ in range(TEST_LOOP):
        # Entry barrier may be skipped on the kernel side; make sure every rank's
        # input is ready and the buffer is free before the next gather.
        dist.barrier(nccl_group)
        x = torch.randn(num_tokens, local_hidden, dtype=torch.bfloat16, device=device)
        ref = _nccl_all_gather(x, nccl_group, world_size)
        out = gather(x)
        # Pure copy gather: exact bitwise equality.
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


def _tp_override_once() -> None:
    coord = ps._WORLD
    assert coord is not None
    get_parallel().override_permanently(
        tp_group=coord, tp_size=coord.world_size, nnodes=1
    )


def _reset_gatherer_registry() -> None:
    triton_symm_mem_ag._shared_needs.clear()
    triton_symm_mem_ag._shared_states.clear()
    triton_symm_mem_ag._disabled_groups.clear()


def _gather_hidden(world_size: int) -> int:
    return 2048 if 2048 % (8 * world_size) == 0 else 8 * world_size * 32


class _RendezvousRecorder:
    """Record the device syncs and the stream each rendezvous runs on."""

    def __init__(self, device: torch.device):
        self.device = device
        self.events: list[str] = []
        self._sync = torch.cuda.synchronize
        self._rendezvous = triton_symm_mem_ag.symm_mem.rendezvous

    def __enter__(self):
        def sync(dev=None):
            self.events.append("sync")
            return self._sync(dev)

        def rendezvous(tensor, group):
            on_default = torch.cuda.current_stream(
                self.device
            ) == torch.cuda.default_stream(self.device)
            self.events.append(
                "rendezvous:default" if on_default else "rendezvous:side"
            )
            return self._rendezvous(tensor, group)

        triton_symm_mem_ag.torch.cuda.synchronize = sync
        triton_symm_mem_ag.symm_mem.rendezvous = rendezvous
        return self

    def __exit__(self, *exc):
        triton_symm_mem_ag.torch.cuda.synchronize = self._sync
        triton_symm_mem_ag.symm_mem.rendezvous = self._rendezvous


@torch.inference_mode()
def test_lazy_rendezvous_runs_on_idle_default_stream() -> None:
    """A gatherer built lazily from a non-default stream with in-flight work
    (the EAGLE draft LogitsProcessor's first call inside the FlashInfer
    autotune forward) must drain the device and rendezvous on the default
    stream: a rendezvous issued on the busy forward stream handed back peer
    signal pads the all-gather kernel could not reach."""
    nccl_group = _init_nccl_group_once()
    _tp_override_once()
    _reset_gatherer_registry()
    coord = ps._WORLD
    world_size = coord.world_size
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    hidden = _gather_hidden(world_size)
    local_hidden = hidden // world_size

    gatherer = MultimemAllGatherer(16, skip_entry_sync=True)
    side = torch.cuda.Stream(device=device)
    dist.barrier(nccl_group)
    x = torch.randn(16, local_hidden, dtype=torch.bfloat16, device=device)
    ref = _nccl_all_gather(x, nccl_group, world_size)
    with _RendezvousRecorder(device) as rec, torch.cuda.stream(side):
        for _ in range(64):
            x = x + 0  # keep the side stream busy when the build triggers
        out = gatherer(x).clone()
    side.synchronize()
    if gatherer._state is None:
        pytest.skip(f"multimem multicast unavailable for world_size={world_size}")
    assert rec.events[: rec.events.index("rendezvous:default") + 1][-2:] == [
        "sync",
        "rendezvous:default",
    ], rec.events
    assert "rendezvous:side" not in rec.events, rec.events
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@torch.inference_mode()
def test_declared_width_rendezvous_at_construction_and_shares() -> None:
    """Gatherers that declare their gathered width rendezvous at construction
    (device idle, default stream), and two of them in one process -- target and
    draft LogitsProcessors -- end up on one buffer sized for the larger need."""
    nccl_group = _init_nccl_group_once()
    _tp_override_once()
    _reset_gatherer_registry()
    coord = ps._WORLD
    world_size = coord.world_size
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    hidden = _gather_hidden(world_size)
    local_hidden = hidden // world_size

    target = MultimemAllGatherer(64, skip_entry_sync=True, hidden_size=hidden)
    if target._state is None:
        pytest.skip(f"multimem multicast unavailable for world_size={world_size}")
    assert target._state is not MultimemAllGatherer._UNINIT, "built lazily"
    # Same declared need as the target (recommended_max_tokens is TP-replicated
    # and identical for both LogitsProcessors): no second rendezvous.
    draft = MultimemAllGatherer(64, skip_entry_sync=True, hidden_size=hidden)
    assert draft._state is target._state, "target and draft must share one state"
    assert sum(len(v) for v in triton_symm_mem_ag._shared_states.values()) == 1
    for gatherer, num_tokens in ((target, 16), (draft, 64)):
        dist.barrier(nccl_group)
        x = torch.randn(num_tokens, local_hidden, dtype=torch.bfloat16, device=device)
        ref = _nccl_all_gather(x, nccl_group, world_size)
        torch.testing.assert_close(gatherer(x).clone(), ref, atol=0, rtol=0)


if __name__ == "__main__":
    # multimem multicast needs world_size in {4, 6, 8} (cc9) or {6, 8} (cc10);
    # unsupported sizes self-skip via the multicast_ptr guard above.
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
