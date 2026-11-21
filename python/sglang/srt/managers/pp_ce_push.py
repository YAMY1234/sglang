"""Copy-engine push for PP activations (SGLANG_PP_CE_PUSH).

Each stage after the first owns an input ring of `slots` fixed-size slots in CUDA VMM memory exported to the previous
stage (FABRIC handles, so it also maps across an NVLink domain). Micro-batch `seq` goes to slot `seq % slots`:

  producer, on its push stream: wait free[slot] >= seq - slots + 1 (its own flag, written by the consumer),
      cuMemcpyDtoDAsync each tensor into the consumer's slot (copy engine), then write full[slot] = seq + 1;
  consumer, on its forward stream: wait full[slot] >= seq + 1, run the forward on views of the slot, and once
      nothing reads the slot any more write free[slot] = seq + 1 into the producer's flag array.

No kernel stays resident while a side waits (a stream wait-value occupies no SM), so the forward's cooperative
kernels never queue behind the hand-off, and neither host waits. The tensors' shapes travel with the host metadata
the stages already exchange; a message larger than a slot falls back to NCCL.
"""

from __future__ import annotations

import logging
from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# Host-metadata stand-in for a tensor that travels through the ring: dtype, shape and byte offset in the slot.
CESlice = namedtuple("CESlice", ["dtype", "size", "offset"])
# Metadata key carrying (slot, seq) of a ring message.
CE_KEY = "__ce__"
# Receive-side stand-in for a recv event: the forward stream waits on the slot's full flag instead.
CEWait = namedtuple("CEWait", ["slot_seq"])
_ALIGN = 256


def pack_layout(
    tensors: Sequence[Tuple[str, torch.dtype, Tuple[int, ...], int]],
) -> Tuple[List[Tuple[str, CESlice]], int]:
    """(key, dtype, shape, nbytes) in order -> (key, CESlice) with 256-byte aligned offsets, and the bytes used."""
    out, off = [], 0
    for key, dtype, shape, nbytes in tensors:
        out.append((key, CESlice(dtype, tuple(shape), off)))
        off += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    return out, off


def slot_of(seq: int, slots: int) -> int:
    return seq % slots


def free_target(seq: int, slots: int) -> int:
    """The free-flag value the producer waits for before writing `seq`: the slot's previous user, seq - slots,
    must have been released (it writes its own seq + 1)."""
    return seq - slots + 1


class _Drv:
    """Thin wrapper over cuda.bindings.driver stream memory operations."""

    def __init__(self) -> None:
        import cuda.bindings.driver as drv

        self.drv = drv

    def _ok(self, res, label: str):
        err = res[0] if isinstance(res, tuple) else res
        if err != self.drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{label}: {err}")

    def wait_geq(self, stream: torch.cuda.Stream, addr: int, value: int) -> None:
        drv = self.drv
        self._ok(
            drv.cuStreamWaitValue32(
                drv.CUstream(stream.cuda_stream),
                drv.CUdeviceptr(addr),
                value,
                drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
            ),
            "cuStreamWaitValue32",
        )

    def write(self, stream: torch.cuda.Stream, addr: int, value: int) -> None:
        # Default flags fence the stream's earlier writes (the slot copies) before the flag.
        drv = self.drv
        self._ok(
            drv.cuStreamWriteValue32(
                drv.CUstream(stream.cuda_stream), drv.CUdeviceptr(addr), value, 0
            ),
            "cuStreamWriteValue32",
        )

    def signal(self, stream: torch.cuda.Stream, remote: int, scratch: int, value: int) -> None:
        """Set a peer's flag: stream memory operations cannot target memory imported from another node
        (CUDA_ERROR_INVALID_VALUE across MNNVL), so write the value locally and copy it with the copy engine,
        ordered after the stream's earlier copies."""
        self.write(stream, scratch, value)
        self.copy(stream, remote, scratch, 4)

    def copy(self, stream: torch.cuda.Stream, dst: int, src: int, nbytes: int) -> None:
        drv = self.drv
        self._ok(
            drv.cuMemcpyDtoDAsync(
                drv.CUdeviceptr(dst),
                drv.CUdeviceptr(src),
                nbytes,
                drv.CUstream(stream.cuda_stream),
            ),
            "cuMemcpyDtoDAsync",
        )


class _EventWork:
    """P2PWork-shaped handle for a push: wait() returns once the push's flag write ran, so the sources can go."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self.event = event

    def wait(self) -> None:
        self.event.synchronize()


class _Push:
    def __init__(self, work: _EventWork, sources: List[torch.Tensor]) -> None:
        self.work = work
        self.payload = sources


def _alloc_exported(device_id: int, nbytes: int):
    from sglang.srt.utils.cuda_vmm_utils import (
        align_up,
        check_drv,
        get_allocation_granularity,
        make_device_allocation_prop,
        make_rw_access_desc,
    )

    import cuda.bindings.driver as drv

    prop = make_device_allocation_prop(
        device_id,
        handle_types=int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC),
    )
    size = align_up(nbytes, get_allocation_granularity(prop))
    handle = check_drv(drv.cuMemCreate(size, prop, 0), "cuMemCreate")
    va = int(check_drv(drv.cuMemAddressReserve(size, 0, 0, 0), "cuMemAddressReserve"))
    check_drv(drv.cuMemMap(va, size, 0, handle, 0), "cuMemMap")
    check_drv(
        drv.cuMemSetAccess(va, size, [make_rw_access_desc(device_id)], 1),
        "cuMemSetAccess",
    )
    fabric = check_drv(
        drv.cuMemExportToShareableHandle(
            handle, drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC, 0
        ),
        "cuMemExportToShareableHandle(FABRIC)",
    )
    return va, size, bytes(fabric.data)


def _import(device_id: int, fabric: bytes, size: int, peer: int) -> int:
    from sglang.srt.utils.cuda_vmm_utils import import_and_map_alloc

    return import_and_map_alloc(
        fabric, None, size, device_id, use_fabric=True, peer_rank=peer
    )


class PPCEPush:
    """One stage's side of the ring protocol: an input ring (stage > 0) and a push target (stage < last)."""

    def __init__(
        self,
        *,
        device_id: int,
        slots: int,
        slot_bytes: int,
        has_prev: bool,
        has_next: bool,
    ) -> None:
        self.device_id = device_id
        self.slots = slots
        self.slot_bytes = slot_bytes
        self.has_prev = has_prev
        self.has_next = has_next
        self.drv = _Drv()
        self.stream = torch.cuda.Stream(device=device_id) if has_next else None
        # Input ring + full flags (stage > 0); free flags for the ring I push into (stage < last).
        self.ring_va = self.full_va = self.free_va = None
        self.peer_ring_va = self.peer_full_va = self.peer_free_va = None
        self._exports: Dict[str, Tuple[bytes, int]] = {}
        self.send_seq = 0
        self.recv_seq = 0
        self.ring_bytes = slots * slot_bytes
        # Local staging words for signal(): [0] the push stream's full flags, [1] the release's free flags.
        self._scratch = torch.zeros(64, dtype=torch.int32, device=f"cuda:{device_id}")

    def allocate(self) -> Dict[str, Tuple[bytes, int]]:
        """Allocate and export this stage's memory; returns {name: (fabric handle, size)} for the neighbours."""
        if self.has_prev:
            self.ring_va, size, fab = _alloc_exported(self.device_id, self.ring_bytes)
            self._exports["ring"] = (fab, size)
            self.full_va, size, fab = _alloc_exported(self.device_id, 4 * self.slots)
            self._exports["full"] = (fab, size)
        if self.has_next:
            self.free_va, size, fab = _alloc_exported(self.device_id, 4 * self.slots)
            self._exports["free"] = (fab, size)
        for va in (self.full_va, self.free_va):
            if va is not None:
                self._zero(va, 4 * self.slots)
        torch.cuda.synchronize(self.device_id)
        return dict(self._exports)

    def _zero(self, va: int, nbytes: int) -> None:
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        tensor_from_pointer(va, nbytes, dtype=torch.uint8, device_id=self.device_id).zero_()

    def attach(
        self,
        *,
        next_exports: Optional[Dict[str, Tuple[bytes, int]]],
        prev_exports: Optional[Dict[str, Tuple[bytes, int]]],
        next_rank: int,
        prev_rank: int,
    ) -> None:
        if self.has_next:
            fab, size = next_exports["ring"]
            self.peer_ring_va = _import(self.device_id, fab, size, next_rank)
            fab, size = next_exports["full"]
            self.peer_full_va = _import(self.device_id, fab, size, next_rank)
        if self.has_prev:
            fab, size = prev_exports["free"]
            self.peer_free_va = _import(self.device_id, fab, size, prev_rank)

    # ---- producer ----

    def fits(self, tensors: Dict[str, torch.Tensor]) -> bool:
        _, used = pack_layout(
            [(k, t.dtype, tuple(t.shape), t.numel() * t.element_size()) for k, t in tensors.items()]
        )
        return used <= self.slot_bytes and all(t.is_contiguous() for t in tensors.values())

    def push(
        self, tensors: Dict[str, torch.Tensor], ready_event: Optional[torch.cuda.Event]
    ) -> Tuple[List[Tuple[str, object]], _Push]:
        """Enqueue the copies of one message on the push stream; returns its metadata entries (CESlice per
        tensor, plus CE_KEY) and a window entry that keeps the sources alive until the push completed."""
        seq = self.send_seq
        self.send_seq += 1
        slot = slot_of(seq, self.slots)
        layout, _ = pack_layout(
            [(k, t.dtype, tuple(t.shape), t.numel() * t.element_size()) for k, t in tensors.items()]
        )
        with torch.cuda.stream(self.stream):
            if ready_event is not None:
                self.stream.wait_event(ready_event)
            target = free_target(seq, self.slots)
            if target > 0:
                self.drv.wait_geq(self.stream, self.free_va + 4 * slot, target)
            base = self.peer_ring_va + slot * self.slot_bytes
            for (key, ce), t in zip(layout, tensors.values()):
                self.drv.copy(self.stream, base + ce.offset, t.data_ptr(), t.numel() * t.element_size())
            self.drv.signal(self.stream, self.peer_full_va + 4 * slot, self._scratch.data_ptr(), seq + 1)
            done = torch.cuda.Event()
            done.record(self.stream)
        meta: List[Tuple[str, object]] = list(layout)
        meta.append((CE_KEY, (slot, seq)))
        return meta, _Push(_EventWork(done), list(tensors.values()))

    # ---- consumer ----

    def views(self, meta: Dict[str, object]) -> Tuple[Dict[str, torch.Tensor], Tuple[int, int]]:
        """Slot views for one received message's metadata (CESlice values); returns them and (slot, seq)."""
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        slot, seq = meta[CE_KEY]
        assert seq == self.recv_seq, (seq, self.recv_seq)
        self.recv_seq += 1
        base = self.ring_va + slot * self.slot_bytes
        out = {}
        for key, value in meta.items():
            if isinstance(value, CESlice):
                nbytes = torch.Size(value.size).numel() * torch.empty((), dtype=value.dtype).element_size()
                out[key] = tensor_from_pointer(
                    base + value.offset,
                    nbytes,
                    shape=tuple(value.size),
                    dtype=value.dtype,
                    device_id=self.device_id,
                )
            elif key != CE_KEY:
                out[key] = value
        return out, (slot, seq)

    def wait_full(self, stream: torch.cuda.Stream, slot_seq: Tuple[int, int]) -> None:
        slot, seq = slot_seq
        self.drv.wait_geq(stream, self.full_va + 4 * slot, seq + 1)

    def release(self, stream: torch.cuda.Stream, slot_seq: Tuple[int, int]) -> None:
        # Its own staging word: the release can run on a different stream from the pushes.
        slot, seq = slot_seq
        self.drv.signal(stream, self.peer_free_va + 4 * slot, self._scratch.data_ptr() + 128, seq + 1)


def setup_ce_push(*, pp_group, device_id: int, slots: int, slot_bytes: int) -> Optional[PPCEPush]:
    """Collective over the PP group (every stage calls it once, before any relay receive thread starts):
    allocate, exchange FABRIC handles with both neighbours, map. Returns None on every stage if any stage fails,
    so all stages keep NCCL."""
    from sglang.srt.distributed.communication_tags import P2PTag
    from sglang.srt.utils.cuda_vmm_utils import all_ranks_ok

    rank, size = pp_group.rank_in_group, pp_group.world_size
    ce = PPCEPush(
        device_id=device_id,
        slots=slots,
        slot_bytes=slot_bytes,
        has_prev=rank > 0,
        has_next=rank + 1 < size,
    )
    exports: Dict[str, Tuple[bytes, int]] = {}
    try:
        exports = ce.allocate()
        ok = True
    except Exception as exc:  # FABRIC unsupported here: every stage falls back together.
        logger.warning("PP CE push: allocation failed on PP%d: %s", rank, exc)
        ok = False
    if not all_ranks_ok(pp_group.cpu_group, ok):
        return None
    tag = int(P2PTag.PP_CE_PUSH_SETUP)
    works = []
    for peer in (rank - 1, rank + 1):
        if 0 <= peer < size:
            works += pp_group.send_object(exports, dst=peer, async_send=True, tag=tag)
    got = {}
    for peer in (rank - 1, rank + 1):
        if 0 <= peer < size:
            got[peer] = pp_group.recv_object(src=peer, tag=tag)
    for w in works:
        w.work.wait()
    try:
        ce.attach(
            next_exports=got.get(rank + 1),
            prev_exports=got.get(rank - 1),
            next_rank=rank + 1,
            prev_rank=rank - 1,
        )
        ok = True
    except Exception as exc:
        logger.warning("PP CE push: mapping a neighbour failed on PP%d: %s", rank, exc)
        ok = False
    if not all_ranks_ok(pp_group.cpu_group, ok):
        return None
    logger.info(
        "PP CE push on PP%d: %d slots x %.0f MiB input ring, copy-engine pushes to PP%s",
        rank,
        slots,
        slot_bytes / (1 << 20),
        rank + 1 if rank + 1 < size else "-",
    )
    return ce
