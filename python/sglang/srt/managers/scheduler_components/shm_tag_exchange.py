"""Shared-memory all-gather of a few small integers among the ranks of one host.

The scheduler loop runs two tiny CPU collectives on every pass (the request
broadcast size and the HiCache ready counts). Over the gloo backend each one
costs a few hundred microseconds even when every rank is ready. When all ranks
of the group share a host, this exchange replaces them with a shared-memory
table.

Every stored element carries the call's sequence tag in its high bits, and
readers check each element on its own. So the protocol needs no ordering
between stores of different elements, only single-copy atomic aligned 64-bit
loads and stores. The table is double-buffered by sequence parity: a rank can
reuse a slot only after every rank has published the next call, which a rank
does only after it has finished reading the previous call.

Enabled by SGLANG_OPT_SCHEDULER_SHM_SYNC=1 (default off). Groups that span hosts
keep the stock gloo path unless SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST=1 is also
set: then each host exchanges through shared memory and only the host leaders
meet in one small gloo all_gather (HierTagExchange).

SGLANG_DP_SYNC_SHM_EXCHANGE=1 (default off) routes the DP-attention MLP sync rows
through the same tables (get_dp_sync_exchange): whole rows are stored and read as
arrays, and host leaders meet over TCP, one link for two hosts or a mesh beyond.
"""

from __future__ import annotations

import ctypes
import logging
import os
import socket
import time
from array import array
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Sequence
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_device

logger = logging.getLogger(__name__)

_TAG_BITS = 23
_VAL_BITS = 40
_TAG_MASK = (1 << _TAG_BITS) - 1
_VAL_MASK = (1 << _VAL_BITS) - 1


class ShmTagExchange:
    def __init__(
        self,
        group: dist.ProcessGroup,
        kind: str,
        width: int = 8,
        rows: Optional[int] = None,
    ):
        from sglang.srt.utils.stale_shm_cleanup import make_shm_name

        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.width = width
        # One row per rank for allgather; publish_rows / read_rows may use any row count.
        self.rows = self.world if rows is None else rows
        self.seq = 0
        self.timeout_s = float(
            os.environ.get("TORCH_DISTRIBUTED_DEFAULT_TIMEOUT", "1800")
        )
        nbytes = 2 * self.rows * width * 8
        ranks = dist.get_process_group_ranks(group)
        self._creator = self.rank == 0
        if self._creator:
            self.shm = shared_memory.SharedMemory(
                create=True, size=nbytes, name=make_shm_name(f"edgex{kind}")
            )
            self.shm.buf[:nbytes] = bytes(nbytes)
            dist.broadcast_object_list([self.shm.name], src=ranks[0], group=group)
        else:
            recv = [None]
            dist.broadcast_object_list(recv, src=ranks[0], group=group)
            # The creator owns unlinking; do not let this process's tracker unlink it.
            with patch(
                "multiprocessing.resource_tracker.register", lambda *a, **k: None
            ):
                self.shm = shared_memory.SharedMemory(name=recv[0])
        self.table = (ctypes.c_int64 * (2 * self.rows * width)).from_buffer(
            self.shm.buf
        )
        self._view = np.frombuffer(self.shm.buf, dtype=np.int64)
        dist.barrier(group=group)

    def allgather(self, values: Sequence[int]) -> List[List[int]]:
        """Every rank passes the same number of values, each in [0, 2**40)."""
        n = len(values)
        assert n <= self.width and self.rows == self.world
        self.seq += 1
        tag = self.seq & _TAG_MASK
        base = (self.seq & 1) * self.rows * self.width
        self._store(base + self.rank * self.width, values, tag)
        return self._wait_rows(base, n, tag)

    def publish_rows(self, table: Sequence[Sequence[int]]) -> None:
        """Single writer: stores every row of this call; the other ranks call read_rows."""
        self.seq += 1
        tag = self.seq & _TAG_MASK
        base = (self.seq & 1) * self.rows * self.width
        for r, row in enumerate(table):
            self._store(base + r * self.width, row, tag)

    def read_rows(self, n: int) -> List[List[int]]:
        self.seq += 1
        tag = self.seq & _TAG_MASK
        return self._wait_rows((self.seq & 1) * self.rows * self.width, n, tag)

    def allgather_array(self, values: np.ndarray) -> np.ndarray:
        """allgather of an int64 vector, each value in [0, 2**40): [world, n]."""
        assert values.shape[0] <= self.width and self.rows == self.world
        self.seq += 1
        tag = self.seq & _TAG_MASK
        table = self._table_half(self.seq, values.shape[0])
        _store_tagged(table[self.rank], values, tag)
        return self._wait_table(table, tag)

    def publish_array(self, rows: np.ndarray) -> None:
        """publish_rows for an int64 [self.rows, n] array."""
        self.seq += 1
        _store_tagged(
            self._table_half(self.seq, rows.shape[1]), rows, self.seq & _TAG_MASK
        )

    def read_array(self, n: int) -> np.ndarray:
        self.seq += 1
        return self._wait_table(self._table_half(self.seq, n), self.seq & _TAG_MASK)

    def _table_half(self, seq: int, n: int) -> np.ndarray:
        base = (seq & 1) * self.rows * self.width
        half = self._view[base : base + self.rows * self.width]
        return half.reshape(self.rows, self.width)[:, :n]

    def _wait_table(self, table: np.ndarray, tag: int) -> np.ndarray:
        deadline = None
        while True:
            # A ufunc reads each element with one 64-bit load; a memcpy-based copy
            # could tear an element across a concurrent store.
            snap = np.add(table, 0)
            if ((snap >> _VAL_BITS) == tag).all():
                return snap & _VAL_MASK
            if deadline is None:
                deadline = time.monotonic() + self.timeout_s
            elif time.monotonic() > deadline:
                raise RuntimeError(
                    f"shm exchange timed out after {self.timeout_s:.0f}s "
                    f"(seq {self.seq}); a peer scheduler is stuck or dead"
                )
            os.sched_yield()

    def _store(self, start: int, values: Sequence[int], tag: int) -> None:
        tagged = tag << _VAL_BITS
        for i, v in enumerate(values):
            if not 0 <= v <= _VAL_MASK:
                raise ValueError(f"shm exchange value out of range: {v}")
            self.table[start + i] = tagged | v

    def _wait_rows(self, base: int, n: int, tag: int) -> List[List[int]]:
        t = self.table
        out: List[List[int]] = []
        deadline = None
        for r in range(self.rows):
            row = base + r * self.width
            vals = []
            for i in range(n):
                x = t[row + i]
                while (x >> _VAL_BITS) != tag:
                    if deadline is None:
                        deadline = time.monotonic() + self.timeout_s
                    elif time.monotonic() > deadline:
                        raise RuntimeError(
                            f"shm exchange timed out after {self.timeout_s:.0f}s waiting "
                            f"for rank {r} (seq {self.seq}); a peer scheduler is stuck or dead"
                        )
                    os.sched_yield()
                    x = t[row + i]
                vals.append(x & _VAL_MASK)
            out.append(vals)
        return out

    def close(self):
        try:
            del self.table, self._view
            self.shm.close()
            if self._creator:
                self.shm.unlink()
        except Exception:
            pass


class _LeaderLink:
    """Persistent TCP connection between the two host leaders; one send and one exact
    receive per call (payloads are a few hundred bytes, so the sends never block)."""

    def __init__(
        self,
        leaders: dist.ProcessGroup,
        first: bool,
        timeout_s: float,
        spin: bool = False,
    ):
        from sglang.srt.utils.network import get_local_ip_auto

        server = None
        box: List[Optional[tuple]] = [None]
        if first:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("0.0.0.0", 0))
            server.listen(1)
            box[0] = (get_local_ip_auto(), server.getsockname()[1])
        src = dist.get_process_group_ranks(leaders)[0]
        dist.broadcast_object_list(box, src=src, group=leaders)
        if first:
            server.settimeout(timeout_s)
            self.sock, _ = server.accept()
            server.close()
        else:
            self.sock = socket.create_connection(box[0], timeout=timeout_s)
        self.sock.settimeout(timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.timeout_s = timeout_s
        # Spin on a non-blocking socket: no sleep and wakeup per receive.
        self.spin = spin
        if spin:
            self.sock.setblocking(False)

    def swap(self, payload: bytes) -> bytes:
        msg = len(payload).to_bytes(4, "little") + payload
        if self.spin:
            deadline = time.monotonic() + self.timeout_s
            _spin_send(self.sock, msg, deadline)
            size = int.from_bytes(_spin_recv(self.sock, 4, deadline), "little")
            return _spin_recv(self.sock, size, deadline)
        self.sock.sendall(msg)
        size = int.from_bytes(self._recv_exact(4), "little")
        return self._recv_exact(size)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError(
                    "Shared-memory exchange leader link closed by the peer leader"
                )
            buf += chunk
        return bytes(buf)


class _LeaderMesh:
    """Persistent TCP connections between every pair of host leaders (more than two
    hosts). A call sends this leader's rows to every peer, then spins until each peer
    has delivered its rows: one network hop, where a gloo all_gather walks a ring."""

    def __init__(self, leaders: dist.ProcessGroup, timeout_s: float):
        from sglang.srt.utils.network import get_local_ip_auto

        n = dist.get_world_size(leaders)
        me = dist.get_rank(leaders)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", 0))
        server.listen(n)
        addrs: List[Optional[tuple]] = [None] * n
        dist.all_gather_object(
            addrs, (get_local_ip_auto(), server.getsockname()[1]), group=leaders
        )
        # Leader j connects to every leader i < j and says who it is.
        self.peers: Dict[int, socket.socket] = {}
        for i in range(me):
            sock = socket.create_connection(addrs[i], timeout=timeout_s)
            sock.sendall(me.to_bytes(4, "little"))
            self.peers[i] = sock
        server.settimeout(timeout_s)
        for _ in range(n - 1 - me):
            sock, _ = server.accept()
            sock.settimeout(timeout_s)
            j = int.from_bytes(_recv_exact(sock, 4), "little")
            self.peers[j] = sock
        server.close()
        for sock in self.peers.values():
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
        self.me = me
        self.n = n
        self.timeout_s = timeout_s

    def all_gather(self, payload: bytes) -> List[bytes]:
        msg = len(payload).to_bytes(4, "little") + payload
        deadline = time.monotonic() + self.timeout_s
        for sock in self.peers.values():
            _spin_send(sock, msg, deadline)
        out: List[bytes] = []
        for i in range(self.n):
            if i == self.me:
                out.append(payload)
            else:
                sock = self.peers[i]
                size = int.from_bytes(_spin_recv(sock, 4, deadline), "little")
                out.append(_spin_recv(sock, size, deadline))
        return out


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("Shared-memory exchange leader link closed by a peer")
        buf += chunk
    return bytes(buf)


def _spin_send(sock: socket.socket, msg: bytes, deadline: float) -> None:
    view = memoryview(msg)
    while view:
        try:
            view = view[sock.send(view) :]
        except BlockingIOError:
            _spin_wait(deadline)


def _spin_recv(sock: socket.socket, n: int, deadline: float) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except BlockingIOError:
            _spin_wait(deadline)
            continue
        if not chunk:
            raise RuntimeError("Shared-memory exchange leader link closed by a peer")
        buf += chunk
    return bytes(buf)


def _spin_wait(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise RuntimeError(
            "Shared-memory exchange leader link timed out; a peer scheduler is "
            "stuck or dead"
        )
    os.sched_yield()


def _store_tagged(dst: np.ndarray, values: np.ndarray, tag: int) -> None:
    if values.size and (values.min() < 0 or values.max() > _VAL_MASK):
        raise ValueError("shm exchange value out of range")
    # A ufunc writes each element with one 64-bit store (see _wait_table).
    np.bitwise_or(values, tag << _VAL_BITS, out=dst)


class HierTagExchange:
    """ShmTagExchange.allgather for a group that spans hosts: ranks exchange through
    shared memory on their host, and only the host leaders run one gloo all_gather."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        kind: str,
        width: int = 8,
        leader_tcp: Optional[bool] = None,
    ):
        """``leader_tcp``: None follows SGLANG_OPT_SCHEDULER_SHM_SYNC_LEADER_TCP (two
        hosts); True spins on TCP between the leaders at any host count."""
        ranks = dist.get_process_group_ranks(group)
        me = dist.get_rank(group)
        hostnames: List[Optional[str]] = [None] * len(ranks)
        dist.all_gather_object(hostnames, socket.gethostname(), group=group)
        hosts = list(dict.fromkeys(hostnames))
        # Group ranks per host, in group-rank order (new_group keeps that order).
        self.members = [
            [i for i, h in enumerate(hostnames) if h == host] for host in hosts
        ]
        # new_group is collective over the default group: every rank creates every subgroup.
        local_groups = [
            dist.new_group([ranks[i] for i in m], backend="gloo") for m in self.members
        ]
        self.leaders = dist.new_group(
            [ranks[m[0]] for m in self.members], backend="gloo"
        )
        host = hosts.index(hostnames[me])
        self.is_leader = me == self.members[host][0]
        self.host = host
        self.world = len(ranks)
        self.width = width
        self.local_max = max(len(m) for m in self.members)
        self.local = ShmTagExchange(local_groups[host], kind + "l", width)
        self.full = ShmTagExchange(
            local_groups[host], kind + "f", width, rows=self.world
        )
        # Two hosts: the leaders swap rows over one TCP connection instead of a gloo all_gather.
        self.link = None
        self.mesh = None
        if leader_tcp is None:
            leader_tcp = envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_LEADER_TCP.get()
            spin = False
            mesh = False
        else:
            spin = mesh = leader_tcp
        if self.is_leader and len(self.members) == 2 and leader_tcp:
            self.link = _LeaderLink(
                self.leaders, host == 0, self.local.timeout_s, spin=spin
            )
        elif self.is_leader and len(self.members) > 2 and mesh:
            self.mesh = _LeaderMesh(self.leaders, self.local.timeout_s)

    def allgather(self, values: Sequence[int]) -> List[List[int]]:
        n = len(values)
        rows_here = self.local.allgather(values)
        if not self.is_leader:
            return self.full.read_rows(n)
        mine = [v for row in rows_here for v in row]
        if self.link is not None:
            peer = array("q")
            peer.frombytes(self.link.swap(array("q", mine).tobytes()))
            per_host = (
                [mine, peer.tolist()] if self.host == 0 else [peer.tolist(), mine]
            )
        else:
            flat = torch.zeros(self.local_max * n, dtype=torch.int64)
            flat[: len(mine)] = torch.tensor(mine, dtype=torch.int64)
            outs = [torch.empty_like(flat) for _ in self.members]
            dist.all_gather(outs, flat, group=self.leaders)
            per_host = [out.tolist() for out in outs]
        table: List[List[int]] = [[] for _ in range(self.world)]
        for m, vals in zip(self.members, per_host):
            for j, g in enumerate(m):
                table[g] = vals[j * n : (j + 1) * n]
        self.full.publish_rows(table)
        return table

    def allgather_array(self, values: np.ndarray) -> np.ndarray:
        """ShmTagExchange.allgather_array across hosts: [world, n] in group-rank order."""
        n = values.shape[0]
        rows_here = self.local.allgather_array(values)
        if not self.is_leader:
            return self.full.read_array(n)
        table = np.empty((self.world, n), dtype=np.int64)
        for members, rows in zip(self.members, self._host_rows(rows_here)):
            table[members] = rows
        self.full.publish_array(table)
        return table

    def _host_rows(self, rows_here: np.ndarray) -> List[np.ndarray]:
        """Every host's local rows, in host order, as [ranks on that host, n]."""
        n = rows_here.shape[1]
        if self.link is not None:
            peer = np.frombuffer(self.link.swap(rows_here.tobytes()), dtype=np.int64)
            peer = peer.reshape(-1, n)
            return [rows_here, peer] if self.host == 0 else [peer, rows_here]
        if self.mesh is not None:
            return [
                np.frombuffer(raw, dtype=np.int64).reshape(-1, n)
                for raw in self.mesh.all_gather(rows_here.tobytes())
            ]
        flat = torch.zeros(self.local_max * n, dtype=torch.int64)
        flat[: rows_here.size] = torch.from_numpy(rows_here.reshape(-1))
        outs = [torch.empty_like(flat) for _ in self.members]
        dist.all_gather(outs, flat, group=self.leaders)
        return [
            out.numpy()[: len(m) * n].reshape(len(m), n)
            for out, m in zip(outs, self.members)
        ]


_EXCHANGES: Dict[tuple, object] = {}


def get_shm_exchange(
    group: Optional[dist.ProcessGroup], kind: str, multihost: bool = False
):
    """Collective on first call per (group, kind): every rank of the group must call it
    at the same point. Returns None (stock path) when disabled, under --sleep-on-idle,
    or when the group spans hosts and the caller did not opt in with multihost=True."""
    if group is None:
        return None
    key = (id(group), kind)
    if key not in _EXCHANGES:
        _EXCHANGES[key] = (
            _new_exchange(
                group,
                kind,
                multihost=multihost
                and envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST.get(),
            )
            if envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.get()
            else None
        )
    return _EXCHANGES[key]


def get_dp_sync_exchange(group: dist.ProcessGroup, width: int):
    """The exchange for the DP-attention MLP sync rows (SGLANG_DP_SYNC_SHM_EXCHANGE),
    ``width`` values per rank; collective on first call, like get_shm_exchange."""
    if not envs.SGLANG_DP_SYNC_SHM_EXCHANGE.get():
        return None
    key = (id(group), "dpsync", width)
    if key not in _EXCHANGES:
        _EXCHANGES[key] = _new_exchange(
            group, "dps", width=width, multihost=True, leader_tcp=True
        )
    return _EXCHANGES[key]


def _new_exchange(
    group: dist.ProcessGroup,
    kind: str,
    *,
    multihost: bool,
    width: int = 8,
    leader_tcp: Optional[bool] = None,
):
    # an idle rank 0 sleeps in its poller; the waiting ranks would spin meanwhile
    if get_device().sleep_on_idle or dist.get_world_size(group) <= 1:
        return None
    from sglang.srt.distributed.parallel_state import in_the_same_node_as

    if all(in_the_same_node_as(group, source_rank=0)):
        logger.info(
            "Shared-memory exchange '%s' enabled for a %d-rank group",
            kind,
            dist.get_world_size(group),
        )
        return ShmTagExchange(group, kind, width)
    # Only when this group is the whole default group, so the subgroup creation
    # inside is collective over exactly these ranks.
    if not multihost or dist.get_world_size() != dist.get_world_size(group):
        return None
    exchange = HierTagExchange(group, kind, width, leader_tcp=leader_tcp)
    logger.info(
        "Hierarchical shared-memory exchange '%s' enabled for a %d-rank group on %d hosts",
        kind,
        dist.get_world_size(group),
        len(exchange.members),
    )
    return exchange


def fold_u64(x: int) -> List[int]:
    """Split an arbitrary int into two 32-bit halves of its low 64 bits."""
    x &= (1 << 64) - 1
    return [x & 0xFFFFFFFF, x >> 32]
