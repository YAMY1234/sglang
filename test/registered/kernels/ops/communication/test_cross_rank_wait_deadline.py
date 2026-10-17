"""A rank whose peer never arrives stops at the wait deadline instead of hanging, in
the fused MoE-finalize push all-reduce (the stage-2 push-plane poll) and in the
symmetric-memory all-gather (its signal-pad barriers). Four spawned ranks; the
last one never launches the collective, and the other three must fail after about
the deadline, not before it and not never.
"""

import os
import re
import socket
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

WORLD = 4
HIDDEN = 7168
ROWS = 2
# Short so that the ranks the absent peer leaves waiting trap quickly.
DEADLINE_S = 2.0
MB = 1024 * 1024


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_world(rank: int, port: int):
    import torch.distributed as dist

    import sglang.srt.distributed.parallel_state as ps
    from sglang.srt.runtime_context import get_parallel

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD),
    )
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(WORLD)), local_rank=rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    return device, coord.cpu_group


def _wait_or_trap(rank: int, device, launch) -> None:
    """Launch on every rank but the last; the last keeps its memory mapped while
    the others fail."""
    if rank < WORLD - 1:
        launch()
        # After the JIT build, which runs before the launch.
        start = time.monotonic()
        try:
            torch.cuda.synchronize(device)
        except RuntimeError as error:
            print(
                f"trapped after {time.monotonic() - start:.2f} s: {error}", flush=True
            )
            raise
    print(f"rank {rank} finished", flush=True)
    time.sleep(10 * DEADLINE_S)


def _push_plane_worker(rank: int, port: int) -> None:
    import torch.distributed as dist

    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    device, cpu_group = _init_world(rank, port)
    comm = CustomAllReduceV2(
        cpu_group, device, max_pull_size=1 * MB, max_push_size=2 * MB
    )
    assert not comm.disabled, "the push plane needs CustomAllReduceV2"
    all_reduce_fusion.register_comm(comm.obj)
    gemm2 = torch.ones(ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    idx = torch.arange(ROWS, dtype=torch.int32, device=device)
    weights = torch.ones(ROWS, 1, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)

    def launch():
        with patch.object(
            all_reduce_fusion, "wait_deadline_ns", return_value=int(DEADLINE_S * 1e9)
        ):
            all_reduce_fusion.moe_finalize_all_reduce(
                gemm2, idx, weights, 1, world_size=WORLD, hidden_dim=HIDDEN
            )

    _wait_or_trap(rank, device, launch)


def _all_gather_worker(rank: int, port: int) -> None:
    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
        all_gather_inner,
        create_state,
    )

    device, cpu_group = _init_world(rank, port)
    state = create_state(
        group=cpu_group, rank_in_group=rank, max_tokens=8, hidden_size=WORLD * 64
    )
    shard = torch.zeros(8, 64, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)

    def launch():
        # The entry barrier is where the present ranks wait for the absent one.
        all_gather_inner(
            state, shard, tp_hidden_dim=WORLD * 64, timeout_ns=int(DEADLINE_S * 1e9)
        )

    _wait_or_trap(rank, device, launch)


def _main(site: str) -> None:
    worker = {"push_plane": _push_plane_worker, "all_gather": _all_gather_worker}
    torch.multiprocessing.spawn(
        worker[site], args=(_free_port(),), nprocs=WORLD, join=True
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < WORLD or torch.cuda.get_device_capability() < (9, 0),
    reason=f"needs {WORLD} SM90+ GPUs",
)
@pytest.mark.parametrize("site", ["push_plane", "all_gather"])
def test_absent_peer_stops_at_the_deadline(site: str):
    module = os.path.splitext(os.path.basename(__file__))[0]
    child = subprocess.run(
        [sys.executable, "-c", f"import {module}; {module}._main({site!r})"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = child.stdout + child.stderr
    assert child.returncode != 0 and f"rank {WORLD - 1} finished" in child.stdout, (
        output
    )
    waits = [float(m) for m in re.findall(r"trapped after ([0-9.]+) s", output)]
    # A wait of about the deadline: not a fast fault, and not a hang.
    assert waits and all(0.9 * DEADLINE_S <= wait < DEADLINE_S + 5 for wait in waits), (
        output
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
