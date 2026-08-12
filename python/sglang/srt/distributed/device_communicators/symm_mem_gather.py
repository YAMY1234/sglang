"""One-sided fixed-shape gather over torch symmetric memory.

Each rank stores its row into every peer's buffer, then waits on a barrier.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# A stuck peer raises instead of spinning forever.
_BARRIER_TIMEOUT_MS = 10_000
# A peer's stores for round N+1 land before its barrier(N+1) returns, so one
# region would be overwritten while a slower rank still reads round N.
_NUM_SLOTS = 2


class SymmMemGather:
    """Allocated and rendezvoused once: a symmetric operand must keep its
    address for its whole lifetime and resolve to the same (region, offset) on
    every rank, which a per-forward pool allocation does not satisfy."""

    def __init__(
        self,
        world_size: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str,
    ):
        from torch._C._distributed_c10d import _SymmetricMemory

        # Outside inference mode on purpose: a region created inside it is an
        # inference tensor and rejects the in-place peer stores below.
        with torch.inference_mode(False):
            region = _SymmetricMemory.empty_strided_p2p(
                (_NUM_SLOTS * world_size * width,),
                [1],
                dtype,
                device,
                group_name,
            ).view(_NUM_SLOTS, world_size, width)
        self._handle = _SymmetricMemory.rendezvous(region)
        self._region = region
        self._world_size = world_size
        self._width = width
        self._slot = 0
        # Keep staging off the WAR-fenced schedule stream. Any forward ordering
        # dependency is attached directly to this stream in gather().
        self._stream = torch.cuda.Stream(device=device)
        self._staging = torch.zeros(width, dtype=dtype, device=device)
        self._host_in = torch.zeros(width, dtype=dtype).pin_memory()
        self._host_out = torch.zeros(world_size, width, dtype=dtype).pin_memory()
        rank = self._handle.rank
        # A peer row is a tensor view of that peer's memory; writing it is a
        # store that never blocks on the peer.
        self._peer_rows = [
            [
                self._handle.get_buffer(peer, (_NUM_SLOTS, world_size, width), dtype)[
                    slot
                ][rank]
                for peer in range(world_size)
            ]
            for slot in range(_NUM_SLOTS)
        ]
        logger.info(
            "Symmetric-memory DP gather active: world=%d width=%d slots=%d",
            world_size,
            width,
            _NUM_SLOTS,
        )

    def gather(
        self,
        local_row_cpu: torch.Tensor,
        dependency_event=None,
        dependency_stream=None,
    ) -> torch.Tensor:
        """Host row in, (world_size, width) host rows out."""
        slot = self._slot
        self._slot = (slot + 1) % _NUM_SLOTS
        self._host_in.copy_(local_row_cpu)
        with torch.cuda.stream(self._stream):
            if dependency_event is not None:
                self._stream.wait_event(dependency_event)
            elif dependency_stream is not None:
                self._stream.wait_stream(dependency_stream)
            self._staging.copy_(self._host_in, non_blocking=True)
            for row in self._peer_rows[slot]:
                row.copy_(self._staging)
            self._handle.barrier(0, _BARRIER_TIMEOUT_MS)
            self._host_out.copy_(self._region[slot], non_blocking=True)
        self._stream.synchronize()
        return self._host_out


def maybe_create_symm_mem_gather(
    world_size: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
) -> Optional[SymmMemGather]:
    """Build a gatherer, or return None when symmetric memory is unusable."""
    try:
        return SymmMemGather(world_size, width, dtype, device, group_name)
    except Exception as e:
        logger.warning(
            "Symmetric-memory DP gather unavailable (%s: %s); falling back.",
            type(e).__name__,
            e,
        )
        return None
