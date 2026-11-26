#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HiCache L3 pool service: one DRAM pool, served to prefill ranks over NIXL (UCX RDMA, memory to memory).

One service per node with spare DRAM (typically a decode node), or per NUMA node of it, holds one pinned,
NUMA-bound DRAM pool and a key -> slot index. The prefill ranks' `nixl_dram` HiCache storage backend asks it for
slots over a small ZMQ control plane and moves the KV bytes itself with one-sided NIXL READ/WRITE into or out of
the pool; this process never touches the data.

Pool: an anonymous mapping of --segment-gib GiB, MPOL_BIND to --numa (set here, read back, and mbind on the
range), prefaulted in parallel so it is resident before READY, checked in /proc/self/numa_maps, then registered
with NIXL's UCX backend in --reg-chunk-gib chunks (slots never cross one).

Index and eviction: the pool is cut into --slab-mib slabs; a slab serves one slot size (object size rounded up to
--align bytes). A full size class evicts its least recently used object (LRU on write commit and read lookup); a
class with nothing to evict takes over the coldest slab of another class, evicting that slab's objects. An object
being written (reserved, not yet committed) or read (pinned by a lookup) is never evicted; reservations and pins
expire after the client's lease.

Readiness: once it serves, the service writes its record to --registry-dir (if set), where clients discover it,
and prints `L3B_READY ... endpoint IP:PORT`. Exit (SIGTERM or SIGINT): stop serving, remove the registry record,
deregister the pool from NIXL and free it, then wait until every NUMA node's available memory (MemFree + clean file
LRU + SReclaimable) is back to >= --teardown-pct % of this service's own start baseline. Each node prints
`L3B_TEARDOWN ok|FAIL host H numa M avail_kb A base_kb B`; any FAIL exits 1, so a launcher can tell that a node did
not get its memory back.

NICs: the pool is published on --ucx-devices (UCX NET_DEVICES), always ports of ONE link layer. `auto` (the
default) takes every port 1 that is ACTIVE (state 4; ACTIVE_DEFER and the rest are not, as UCX and NCCL skip
them) with --link-layer's link layer (InfiniBand, or Ethernet = RoCE; `auto`: the one most ACTIVE ports have,
InfiniBand on a tie); a list must name ACTIVE ports of one link layer; empty: UCX picks, unchecked. Fewer selected
ports than --expect-ports fails the preflight, listing every port's state, before any allocation. A client whose
`nixl_dram_ucx_devices` is empty (the default) gives its UCX no device list, so it opens every device of its node;
it still reaches the pool only on lanes that match an address this service publishes.

Needs the standard library, pyzmq and nixl, and imports nothing from SGLang, so it runs as a plain script.
"""

import argparse
import concurrent.futures
import ctypes
import json
import os
import platform
import re
import signal
import socket
import sys
import threading
import time
import traceback
from collections import OrderedDict

PROTO = 1
# alloc/lookup codes, shared with the client (nixl_dram_client.py)
EXISTS, BUSY, NO_SPACE, TOO_BIG = -1, -2, -3, -4
GIB, MIB = 1 << 30, 1 << 20

SYSFS_NODE = "/sys/devices/system/node"
SYSFS_IB = "/sys/class/infiniband"
# IB or RoCE, never both in one list; InfiniBand wins a tie (pick_layer)
RDMA_LINKS = ("InfiniBand", "Ethernet")
BITS = 1024
NR = {
    "aarch64": {"mbind": 235, "get_mempolicy": 236, "set_mempolicy": 237},
    "x86_64": {"mbind": 237, "set_mempolicy": 238, "get_mempolicy": 239},
}
MPOL_MODES = {
    0: "default",
    1: "preferred",
    2: "bind",
    3: "interleave",
    4: "local",
    5: "preferred_many",
    6: "weighted_interleave",
}
MPOL_BIND, MPOL_MF_STRICT, MPOL_MF_MOVE = 2, 1, 2
PROT_RW, MAP_PRIVATE, MAP_ANONYMOUS, MAP_NORESERVE = 3, 0x02, 0x20, 0x4000
MADV_DONTNEED, MADV_HUGEPAGE = 4, 14

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.syscall.restype = ctypes.c_long


def say(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------- NUMA and memory


def meminfo(sysfs=SYSFS_NODE):
    """{node: {field: kB}} for every NUMA node with memory (GPU-memory nodes report MemTotal 0)."""
    nodes = {}
    for d in sorted(os.listdir(sysfs)):
        if not re.fullmatch(r"node\d+", d):
            continue
        m = {}
        with open(os.path.join(sysfs, d, "meminfo")) as f:
            for line in f:
                hit = re.match(r"Node \d+ (\S+):\s+(\d+)", line)
                if hit:
                    m[hit.group(1)] = int(hit.group(2))
        if m.get("MemTotal", 0) > 0:
            nodes[int(d[4:])] = m
    return nodes


def available_kb(m):
    """What a node can still hand out: free pages, clean file LRU and reclaimable slab (the exit check's measure)."""
    cache = (
        m.get("Active(file)", 0)
        + m.get("Inactive(file)", 0)
        - m.get("Dirty", 0)
        - m.get("Writeback", 0)
    )
    return m.get("MemFree", 0) + max(cache, 0) + m.get("SReclaimable", 0)


def _nodemask(node):
    mask = (ctypes.c_ulong * (BITS // 64))()
    mask[node // 64] |= 1 << (node % 64)
    return mask


def get_mempolicy():
    mode, mask = ctypes.c_int(0), (ctypes.c_ulong * (BITS // 64))()
    if libc.syscall(
        ctypes.c_long(NR[platform.machine()]["get_mempolicy"]),
        ctypes.byref(mode),
        mask,
        ctypes.c_ulong(BITS + 1),
        None,
        ctypes.c_ulong(0),
    ):
        raise OSError(ctypes.get_errno(), "get_mempolicy")
    return MPOL_MODES.get(mode.value, str(mode.value)), [
        i for i in range(BITS) if mask[i // 64] >> (i % 64) & 1
    ]


def bind_to(node, sysfs=SYSFS_NODE):
    """MPOL_BIND this process to node and its CPUs to the node's; returns the policy read back."""
    if libc.syscall(
        ctypes.c_long(NR[platform.machine()]["set_mempolicy"]),
        ctypes.c_int(MPOL_BIND),
        _nodemask(node),
        ctypes.c_ulong(BITS + 1),
    ):
        raise OSError(ctypes.get_errno(), "set_mempolicy")
    cpus = set()
    try:
        for part in open(f"{sysfs}/node{node}/cpulist").read().strip().split(","):
            if part:
                lo, _, hi = part.partition("-")
                cpus.update(range(int(lo), int(hi or lo) + 1))
        if cpus & os.sched_getaffinity(0):
            os.sched_setaffinity(0, cpus & os.sched_getaffinity(0))
    except OSError:
        pass  # CPU affinity is advisory; the memory policy is what holds the pool on its node
    return get_mempolicy()


def mbind(addr, length, node):
    if libc.syscall(
        ctypes.c_long(NR[platform.machine()]["mbind"]),
        ctypes.c_void_p(addr),
        ctypes.c_ulong(length),
        ctypes.c_int(MPOL_BIND),
        _nodemask(node),
        ctypes.c_ulong(BITS + 1),
        ctypes.c_uint(MPOL_MF_STRICT | MPOL_MF_MOVE),
    ):
        raise OSError(ctypes.get_errno(), "mbind")


def placement(base, length, numa_maps="/proc/self/numa_maps"):
    """{node: resident bytes} of the mappings that start inside [base, base + length)."""
    out = {}
    with open(numa_maps) as f:
        for row in f:
            start = int(row.split(" ", 1)[0], 16)
            if not base <= start < base + length:
                continue
            kb = re.search(r"kernelpagesize_kB=(\d+)", row)
            kb = int(kb.group(1)) if kb else os.sysconf("SC_PAGE_SIZE") // 1024
            for n, c in re.findall(r"\bN(\d+)=(\d+)", row):
                out[int(n)] = out.get(int(n), 0) + int(c) * kb * 1024
    return out


def _parallel_chunks(base, length, chunk, threads, fn, stop=None):
    """fn(addr, n) over [base, base + length) in chunk-sized pieces on `threads` threads (ctypes calls drop the GIL)."""
    pieces = [(base + off, min(chunk, length - off)) for off in range(0, length, chunk)]

    def run(piece):
        if stop is not None and stop.is_set():
            return
        fn(*piece)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        list(ex.map(run, pieces))


class Pool:
    """One NUMA-bound, prefaulted anonymous DRAM region."""

    def __init__(self, nbytes, node, threads, stop=None):
        self.nbytes, self.node, self.threads = nbytes, node, threads
        addr = libc.mmap(
            None, nbytes, PROT_RW, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0
        )
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f"mmap of {nbytes} bytes")
        self.base = addr
        try:
            # Best effort: THP off is not an error.
            libc.madvise(ctypes.c_void_p(addr), nbytes, MADV_HUGEPAGE)
            mbind(addr, nbytes, node)
            _parallel_chunks(
                addr, nbytes, GIB, threads, lambda a, n: ctypes.memset(a, 0, n), stop
            )
        except BaseException:
            self.free()
            raise

    def free(self):
        """Give the pages back (in parallel: MADV_DONTNEED drops them under the read lock), then unmap."""
        if self.base is None:
            return
        _parallel_chunks(
            self.base,
            self.nbytes,
            GIB,
            self.threads,
            lambda a, n: libc.madvise(ctypes.c_void_p(a), n, MADV_DONTNEED),
        )
        if libc.munmap(ctypes.c_void_p(self.base), self.nbytes):
            raise OSError(ctypes.get_errno(), "munmap")
        self.base = None


# --------------------------------------------------------------------------------------------- index


class _Entry:
    __slots__ = (
        "key",
        "slot_size",
        "slab",
        "slot",
        "size",
        "committed",
        "pins",
        "token",
        "last",
    )

    def __init__(self, key, slot_size, slab, slot, size, token, now):
        self.key = key
        self.slot_size = slot_size
        self.slab = slab
        self.slot = slot
        self.size = size
        self.committed = False
        self.pins = 0
        self.token = token
        self.last = now


class _Class:
    __slots__ = ("partial", "lru", "slabs")

    def __init__(self):
        self.partial = OrderedDict()  # slabs of this class with a free slot
        self.lru = OrderedDict()  # committed keys, least recently used first
        self.slabs = 0


class SlabIndex:
    """key -> slot in a pool of equal slabs; each slab serves one slot size. Not thread-safe: the service is one loop.

    Eviction approximates one LRU over the whole pool at slab granularity: a size class with no free slot compares
    its least recently used object with the coldest slab of any other class, and either evicts that object or takes
    the slab over (evicting every object in it, all older than the class's own oldest). Reserved (being written) and
    pinned (being read) objects are never evicted, nor is a slab holding one.
    """

    def __init__(self, base, pool_bytes, slab_bytes, align=4096, clock=time.monotonic):
        if slab_bytes <= 0 or slab_bytes % align or pool_bytes < slab_bytes:
            raise ValueError(
                f"slab {slab_bytes} B must be a positive multiple of align {align} and fit the pool"
            )
        self.base = base
        self.slab_bytes = slab_bytes
        self.align = align
        self.clock = clock
        self.nslabs = pool_bytes // slab_bytes
        self.slab_size = [0] * self.nslabs  # slot size the slab serves, 0 if unassigned
        self.slab_free = [None] * self.nslabs  # free slot numbers
        self.slab_keys = [None] * self.nslabs  # slot -> key
        self.slab_busy = [0] * self.nslabs  # entries reserved or pinned
        self.slab_last = [
            0.0
        ] * self.nslabs  # last write commit or read lookup in the slab
        self.slab_lru = OrderedDict()  # assigned slabs, least recently used first
        self.unassigned = list(range(self.nslabs - 1, -1, -1))
        self.classes = {}
        self.entries = {}
        self.used = 0  # bytes of the slots held by entries
        self.write_tokens = {}  # token -> (deadline, [key])
        self.read_tokens = {}  # token -> (deadline, [key])
        self.stats = dict(
            puts=0,
            put_bytes=0,
            exists_hits=0,
            busy=0,
            no_space=0,
            too_big=0,
            lookups=0,
            hits=0,
            misses=0,
            evicted=0,
            slab_moves=0,
            aborted=0,
            expired_writes=0,
            expired_pins=0,
        )

    # -- slots
    def addr(self, e):
        return self.base + e.slab * self.slab_bytes + e.slot * e.slot_size

    def _assign(self, slab, slot_size, now):
        c = self.classes.setdefault(slot_size, _Class())
        n = self.slab_bytes // slot_size
        self.slab_size[slab] = slot_size
        self.slab_free[slab] = list(range(n - 1, -1, -1))
        self.slab_keys[slab] = {}
        self.slab_last[slab] = now
        self.slab_lru[slab] = None
        c.partial[slab] = None
        c.slabs += 1

    def _pop_slot(self, c):
        slab = next(iter(c.partial))
        slot = self.slab_free[slab].pop()
        if not self.slab_free[slab]:
            del c.partial[slab]
        return slab, slot

    def _touch(self, e, now):
        e.last = now
        self.slab_last[e.slab] = now
        self.slab_lru.move_to_end(e.slab)

    def _drop(self, e, evicted):
        del self.entries[e.key]
        self.used -= e.slot_size
        if e.committed:
            self.classes[e.slot_size].lru.pop(e.key, None)
        self.slab_keys[e.slab].pop(e.slot, None)
        self.slab_free[e.slab].append(e.slot)
        self.classes[e.slot_size].partial[e.slab] = None
        if evicted:
            self.stats["evicted"] += 1

    def _busy(self, e, delta):
        self.slab_busy[e.slab] += delta

    def _coldest_slab(self, slot_size, older_than):
        """The least recently used slab of another class that no one is writing or reading, last used before
        older_than, or None. slab_lru is in slab_last order, so the scan stops at the first newer slab."""
        for s in self.slab_lru:
            if self.slab_last[s] >= older_than:
                return None
            if self.slab_size[s] != slot_size and self.slab_busy[s] == 0:
                return s
        return None

    def _move_slab(self, slab, slot_size, now):
        old = self.classes[self.slab_size[slab]]
        for key in list(self.slab_keys[slab].values()):
            self._drop(self.entries[key], evicted=True)
        old.partial.pop(slab, None)
        old.slabs -= 1
        del self.slab_lru[slab]
        self.slab_size[slab] = 0
        self.stats["slab_moves"] += 1
        self._assign(slab, slot_size, now)

    def _take_slot(self, slot_size, now):
        c = self.classes.get(slot_size)
        if c is not None and c.partial:
            return self._pop_slot(c)
        if self.unassigned:
            self._assign(self.unassigned.pop(), slot_size, now)
            return self._pop_slot(self.classes[slot_size])
        victim = None  # this class's least recently used object that no one is reading
        for key in c.lru if c is not None else ():
            if self.entries[key].pins == 0:
                victim = self.entries[key]
                break
        cold = self._coldest_slab(
            slot_size, float("inf") if victim is None else victim.last
        )
        if cold is not None:
            self._move_slab(cold, slot_size, now)
            return self._pop_slot(self.classes[slot_size])
        if victim is not None:
            self._drop(victim, evicted=True)
            return self._pop_slot(c)
        return None

    # -- operations
    def exists(self, keys):
        out = []
        for k in keys:
            e = self.entries.get(k)
            out.append(1 if e is not None and e.committed else 0)
        return out

    def alloc(self, keys, sizes, token, lease_s):
        """A reserved slot address per key, or EXISTS / BUSY (being written) / TOO_BIG / NO_SPACE."""
        now, out, reserved = self.clock(), [], []
        for key, size in zip(keys, sizes):
            e = self.entries.get(key)
            if e is not None:
                out.append(EXISTS if e.committed else BUSY)
                self.stats["exists_hits" if e.committed else "busy"] += 1
                continue
            if size <= 0 or size > self.slab_bytes:
                out.append(TOO_BIG)
                self.stats["too_big"] += 1
                continue
            slot_size = -(-size // self.align) * self.align
            loc = self._take_slot(slot_size, now)
            if loc is None:
                out.append(NO_SPACE)
                self.stats["no_space"] += 1
                continue
            e = _Entry(key, slot_size, loc[0], loc[1], size, token, now)
            self.entries[key] = e
            self.used += slot_size
            self.slab_keys[e.slab][e.slot] = key
            self._busy(e, 1)
            reserved.append(key)
            out.append(self.addr(e))
        if reserved:
            self.write_tokens[token] = (now + lease_s, reserved)
        return out

    def commit(self, token, keys, oks):
        """Publish each written key whose write succeeded, free the rest; returns one bool per key."""
        rec = self.write_tokens.pop(token, None)
        now, done = self.clock(), []
        mine = set(rec[1]) if rec else set()
        for key, ok in zip(keys, oks):
            e = self.entries.get(key)
            if e is None or e.committed or e.token != token or key not in mine:
                done.append(False)
                continue
            mine.discard(key)
            self._busy(e, -1)
            if ok:
                e.committed, e.token = True, None
                self.classes[e.slot_size].lru[key] = None
                self._touch(e, now)
                self.stats["puts"] += 1
                self.stats["put_bytes"] += e.size
                done.append(True)
            else:
                self._drop(e, evicted=False)
                self.stats["aborted"] += 1
                done.append(False)
        for key in mine:  # reserved under this token but not named in the commit
            e = self.entries.get(key)
            if e is not None and not e.committed and e.token == token:
                self._busy(e, -1)
                self._drop(e, evicted=False)
                self.stats["aborted"] += 1
        return done

    def lookup(self, keys, token, lease_s):
        """(addresses, sizes) of the committed keys, pinned until release(token); -1 for the others."""
        now, addrs, sizes, pinned = self.clock(), [], [], []
        self.stats["lookups"] += len(keys)
        for key in keys:
            e = self.entries.get(key)
            if e is None or not e.committed:
                addrs.append(-1)
                sizes.append(0)
                self.stats["misses"] += 1
                continue
            if e.pins == 0:
                self._busy(e, 1)
            e.pins += 1
            self.classes[e.slot_size].lru.move_to_end(key)
            self._touch(e, now)
            pinned.append(key)
            addrs.append(self.addr(e))
            sizes.append(e.size)
            self.stats["hits"] += 1
        if pinned:
            old = self.read_tokens.get(token)
            self.read_tokens[token] = (now + lease_s, (old[1] if old else []) + pinned)
        return addrs, sizes

    def _unpin(self, keys):
        n = 0
        for key in keys:
            e = self.entries.get(key)
            if e is not None and e.pins > 0:
                e.pins -= 1
                if e.pins == 0:
                    self._busy(e, -1)
                n += 1
        return n

    def release(self, token):
        rec = self.read_tokens.pop(token, None)
        return self._unpin(rec[1]) if rec else 0

    def expire(self, now=None):
        """Free reservations and drop pins whose lease ran out; returns (writes freed, pins dropped)."""
        now = self.clock() if now is None else now
        freed = dropped = 0
        for token in [t for t, (d, _) in self.write_tokens.items() if d <= now]:
            for key in self.write_tokens.pop(token)[1]:
                e = self.entries.get(key)
                if e is not None and not e.committed and e.token == token:
                    self._busy(e, -1)
                    self._drop(e, evicted=False)
                    freed += 1
        for token in [t for t, (d, _) in self.read_tokens.items() if d <= now]:
            dropped += self._unpin(self.read_tokens.pop(token)[1])
        self.stats["expired_writes"] += freed
        self.stats["expired_pins"] += dropped
        return freed, dropped

    def clear(self):
        """Drop every committed object no one is reading; returns how many."""
        n = 0
        for e in [e for e in self.entries.values() if e.committed and e.pins == 0]:
            self._drop(e, evicted=False)
            n += 1
        return n

    def check(self):
        """Internal consistency: every entry in its slab, free lists and keys disjoint, busy counts right."""
        busy = [0] * self.nslabs
        for key, e in self.entries.items():
            assert self.slab_size[e.slab] == e.slot_size, key
            assert self.slab_keys[e.slab].get(e.slot) == key, key
            assert e.slot not in self.slab_free[e.slab], key
            assert (key in self.classes[e.slot_size].lru) == e.committed, key
            busy[e.slab] += (not e.committed) or e.pins > 0
        assert busy == self.slab_busy, (busy, self.slab_busy)
        assert self.used == sum(e.slot_size for e in self.entries.values())
        lasts = [self.slab_last[s] for s in self.slab_lru]
        assert lasts == sorted(lasts), "slab_lru out of slab_last order"
        assigned = [s for s in range(self.nslabs) if self.slab_size[s]]
        assert sorted(self.slab_lru) == assigned
        for s in assigned:
            n = self.slab_bytes // self.slab_size[s]
            assert len(self.slab_keys[s]) + len(self.slab_free[s]) == n, s
            assert (s in self.classes[self.slab_size[s]].partial) == bool(
                self.slab_free[s]
            ), s
        assert (
            sum(c.slabs for c in self.classes.values()) + len(self.unassigned)
            == self.nslabs
        )


# --------------------------------------------------------------------------------------------- export (data plane)


class NixlExport:
    """The pool registered with a NIXL agent's UCX backend; clients load this agent's metadata to READ/WRITE it."""

    def __init__(self, name, pool, chunk_bytes, devices, num_workers, engine_config):
        from nixl._api import nixl_agent, nixl_agent_config

        self.agent = nixl_agent(
            name,
            nixl_agent_config(
                enable_prog_thread=True, enable_listen_thread=False, backends=[]
            ),
        )
        params = {"num_workers": str(num_workers)}
        if devices:
            # NIXL splits device_list on ", " and appends ":1" to each (UCX NET_DEVICES).
            params["device_list"] = ", ".join(devices)
        if engine_config:
            params["engine_config"] = engine_config
        self.agent.create_backend("UCX", params)
        self.params = params
        descs = [
            (pool.base + off, min(chunk_bytes, pool.nbytes - off), 0, "")
            for off in range(0, pool.nbytes, chunk_bytes)
        ]
        self.reg = self.agent.register_memory(descs, "DRAM")
        self.chunks = len(descs)
        self.metadata = self.agent.get_agent_metadata()
        self.name = name

    def hello(self):
        return {"agent_name": self.name}, self.metadata

    def close(self):
        if self.reg is not None:
            self.agent.deregister_memory(self.reg)
            self.reg = None
        self.agent = None  # the agent's destructor closes its UCX workers


# --------------------------------------------------------------------------------------------- service


class Service:
    def __init__(self, args, index, export, pool, host, ip, stop):
        import zmq

        self.zmq = zmq
        self.args = args
        self.index = index
        self.export = export
        self.pool = pool
        self.host = host
        self.ip = ip
        self.stop = stop
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.setsockopt(zmq.LINGER, 0)
        if args.port:
            self.sock.bind(f"tcp://{args.bind}:{args.port}")
            self.port = args.port
        else:
            self.port = self.sock.bind_to_random_port(f"tcp://{args.bind}")
        self.service_id = f"{host}.numa{args.numa}.p{self.port}"
        self.clients = set()
        self.registry_file = None

    def publish(self):
        if not self.args.registry_dir:
            return
        os.makedirs(self.args.registry_dir, exist_ok=True)
        rec = {
            "proto": PROTO,
            "service": self.service_id,
            "host": self.host,
            "ip": self.ip,
            "port": self.port,
            "endpoint": f"{self.ip}:{self.port}",
            "numa": self.args.numa,
            "pid": os.getpid(),
            "pool_bytes": self.pool.nbytes,
        }
        path = os.path.join(self.args.registry_dir, f"{self.service_id}.json")
        tmp = os.path.join(
            self.args.registry_dir, f".{self.service_id}.{os.getpid()}.tmp"
        )
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, path)
        self.registry_file = path

    def unpublish(self):
        path, self.registry_file = self.registry_file, None
        if not path:
            return
        # Only our own record: a restarted service on this node, NUMA node and port rewrites the same name.
        try:
            if json.load(open(path)).get("pid") == os.getpid():
                os.unlink(path)
        except (OSError, ValueError):
            pass

    def handle(self, req, payload):
        op, ix = req.get("op"), self.index
        if op == "exists":
            return {"ok": 1, "present": ix.exists(req["keys"])}, None
        if op == "alloc":
            addrs = ix.alloc(
                req["keys"], req["sizes"], req["token"], float(req["lease_s"])
            )
            return {"ok": 1, "addrs": addrs}, None
        if op == "commit":
            return {
                "ok": 1,
                "done": ix.commit(req["token"], req["keys"], req["ok"]),
            }, None
        if op == "lookup":
            addrs, sizes = ix.lookup(req["keys"], req["token"], float(req["lease_s"]))
            return {"ok": 1, "addrs": addrs, "sizes": sizes}, None
        if op == "release":
            return {"ok": 1, "released": ix.release(req["token"])}, None
        if op == "hello":
            if req.get("proto") != PROTO:
                return {
                    "ok": 0,
                    "error": f"protocol {req.get('proto')} != service protocol {PROTO}",
                }, None
            self.clients.add(req.get("client", "?"))
            extra, md = self.export.hello()
            return {
                "ok": 1,
                "proto": PROTO,
                "service": self.service_id,
                "host": self.host,
                "numa": self.args.numa,
                "pid": os.getpid(),
                "pool_base": self.pool.base,
                "pool_bytes": self.pool.nbytes,
                "slab_bytes": ix.slab_bytes,
                "align": ix.align,
                **extra,
            }, md
        if op == "clear":
            return {"ok": 1, "removed": ix.clear()}, None
        return {"ok": 0, "error": f"unknown op {op!r}"}, None

    def stats(self):
        ix = self.index
        return {
            "entries": len(ix.entries),
            "used_bytes": ix.used,
            "pool_bytes": self.pool.nbytes,
            "classes": len(ix.classes),
            "free_slabs": len(ix.unassigned),
            "clients": len(self.clients),
            **ix.stats,
        }

    def serve(self):
        zmq = self.zmq
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        next_expire = next_stats = time.monotonic()
        while not self.stop.is_set():
            if poller.poll(200):
                frames = self.sock.recv_multipart()
                # REQ envelope: identity, empty delimiter, header[, payload]
                ident, body = frames[0], frames[2:]
                try:
                    reply, data = self.handle(
                        json.loads(body[0]), body[1] if len(body) > 1 else None
                    )
                except Exception as e:  # a bad request must not take the pool down; the client sees the error
                    say(f"L3B_ERROR request failed: {e!r}")
                    reply, data = {"ok": 0, "error": repr(e)}, None
                out = [ident, b"", json.dumps(reply).encode()]
                if data is not None:
                    out.append(data)
                self.sock.send_multipart(out)
            now = time.monotonic()
            if now >= next_expire:
                freed, dropped = self.index.expire()
                if freed or dropped:
                    say(
                        f"L3B_WARN lease expired: {freed} reserved slot(s) freed, "
                        f"{dropped} read pin(s) dropped"
                    )
                next_expire = now + 1.0
            if self.args.stats_interval_s > 0 and now >= next_stats:
                s = self.stats()
                say("L3B_STATS " + " ".join(f"{k}={v}" for k, v in s.items()))
                next_stats = now + self.args.stats_interval_s

    def close(self):
        self.unpublish()
        self.sock.close(0)
        self.ctx.term()


# --------------------------------------------------------------------------------------------- teardown


def teardown_check(
    host,
    baseline,
    pct,
    timeout_s,
    sysfs=SYSFS_NODE,
    interval_s=1.0,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Poll until every baseline NUMA node's available memory is >= pct % of its start value (or the deadline);
    print one L3B_TEARDOWN line per node; 0 if all are back, else 1."""
    deadline = clock() + timeout_s
    while True:
        now = meminfo(sysfs)
        low = [
            n
            for n, b in baseline.items()
            if n in now and available_kb(now[n]) < pct / 100.0 * b
        ]
        if not low or clock() >= deadline:
            break
        sleep(interval_s)
    for n in sorted(baseline):
        a = available_kb(now[n]) if n in now else 0
        say(
            f"L3B_TEARDOWN {'FAIL' if n in low else 'ok'} host {host} numa {n} avail_kb {a} base_kb {baseline[n]}"
        )
    return 1 if low else 0


# --------------------------------------------------------------------------------------------- NICs


def node_ip():
    ip = socket.gethostbyname(socket.gethostname())
    if ip.startswith("127."):
        import subprocess

        ip = next(
            (
                a
                for a in subprocess.run(
                    ["hostname", "-I"], capture_output=True, text=True
                ).stdout.split()
                if not a.startswith("127.")
            ),
            ip,
        )
    return ip


def port1(d, ib_sysfs=SYSFS_IB):
    """(link layer, state, NUMA node) of an RDMA device's port 1."""
    p = f"{ib_sysfs}/{d}"
    link = open(f"{p}/ports/1/link_layer").read().strip()
    state = open(f"{p}/ports/1/state").read().strip()
    try:
        numa = open(f"{p}/device/numa_node").read().strip()
    except OSError:
        numa = "?"
    return link, state, numa


def usable(link, state):
    """An ACTIVE (sysfs "4: ACTIVE", not "5: ACTIVE_DEFER") InfiniBand or RoCE port."""
    return link in RDMA_LINKS and state.split(":", 1)[-1].strip() == "ACTIVE"


def discover_devices(ib_sysfs=SYSFS_IB):
    """(every RDMA device as 'name link state numaN', [(name, link)] of the ACTIVE IB/RoCE ones), in name order."""
    names = sorted(
        os.listdir(ib_sysfs) if os.path.isdir(ib_sysfs) else [],
        key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d)],
    )
    seen, active = [], []
    for d in names:
        try:
            link, state, numa = port1(d, ib_sysfs)
        except OSError as e:
            seen.append(f"{d} {e.strerror}")
            continue
        seen.append(f"{d} {link} {state} numa{numa}")
        if usable(link, state):
            active.append((d, link))
    return seen, active


def pick_layer(active, want="auto"):
    """The one link layer to use: want, else the one most ACTIVE ports have (InfiniBand on a tie); None if none."""
    if want != "auto":
        return want
    counts = [sum(link == layer for _, link in active) for layer in RDMA_LINKS]
    return RDMA_LINKS[counts.index(max(counts))] if active else None


def check_devices(devices, ib_sysfs=SYSFS_IB, link_layer="auto"):
    """[why] for each device that is not an ACTIVE InfiniBand or RoCE (Ethernet) port 1, plus one if they are not
    all of one link layer (link_layer's, unless auto)."""
    bad, links = [], {}
    for d in devices:
        try:
            link, state, _ = port1(d, ib_sysfs)
        except OSError as e:
            bad.append(f"{d}: {e.strerror}")
            continue
        links[d] = link
        if not usable(link, state):
            bad.append(f"{d}: {link} {state}")
    if len(set(links.values()) | ({link_layer} if link_layer != "auto" else set())) > 1:
        bad.append(
            f"not one link layer{'' if link_layer == 'auto' else ' (--link-layer ' + link_layer + ')'}: "
            + ", ".join(f"{d} {layer}" for d, layer in links.items())
        )
    return bad


def select_devices(spec, link_layer="auto", expect=0, ib_sysfs=SYSFS_IB):
    """(devices, seen, [why the preflight fails]) for --ucx-devices SPEC: `auto` -> every ACTIVE port of one link
    layer (pick_layer), a list -> itself, checked (check_devices); empty -> [] (UCX picks)."""
    seen, active = discover_devices(ib_sysfs)
    names = [d.strip() for d in spec.split(",") if d.strip()]
    if names == ["auto"]:
        layer = pick_layer(active, link_layer)
        devices = [d for d, link in active if link == layer]
        bad = (
            []
            if devices
            else [f"auto found no ACTIVE {layer or 'InfiniBand or RoCE'} port"]
        )
    else:
        devices = names
        bad = check_devices(names, ib_sysfs, link_layer) if names else []
    if names and not bad and len(devices) < expect:
        bad.append(
            f"{len(devices)} port(s) {','.join(devices)}, --expect-ports {expect}"
        )
    return devices, seen, bad


# --------------------------------------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--numa",
        type=int,
        required=True,
        help="NUMA node that holds the pool (MPOL_BIND)",
    )
    p.add_argument("--segment-gib", type=int, required=True, help="pool size, GiB")
    p.add_argument(
        "--port", type=int, default=18800, help="ZMQ control port; 0: any free port"
    )
    p.add_argument("--bind", default="*", help="ZMQ bind address")
    p.add_argument(
        "--advertise-host",
        default="",
        help="address clients reach this service at (default: node IP)",
    )
    p.add_argument(
        "--registry-dir",
        default="",
        help="shared directory the clients discover services in",
    )
    p.add_argument(
        "--ucx-devices",
        default="auto",
        help="RDMA devices UCX may use, comma-separated; auto: this node's ACTIVE ports of one link "
        "layer (--link-layer), found at start; empty: UCX picks",
    )
    p.add_argument(
        "--link-layer",
        choices=("auto",) + RDMA_LINKS,
        default="auto",
        help="the one link layer of --ucx-devices (Ethernet: RoCE); auto: the one most ACTIVE ports have",
    )
    p.add_argument(
        "--expect-ports",
        type=int,
        default=0,
        help="fail the preflight if fewer ports are selected",
    )
    p.add_argument(
        "--num-workers", type=int, default=1, help="UCX workers of the agent"
    )
    p.add_argument(
        "--engine-config",
        default="",
        help="extra UCX config, NIXL engine_config (K=V,K=V)",
    )
    p.add_argument("--slab-mib", type=int, default=64)
    p.add_argument("--align", type=int, default=4096)
    p.add_argument(
        "--reg-chunk-gib",
        type=int,
        default=32,
        help="NIXL registration chunk (a multiple of the slab)",
    )
    p.add_argument("--prefault-threads", type=int, default=16)
    p.add_argument(
        "--headroom-pct",
        type=float,
        default=25.0,
        help="preflight: the node must keep this %% of its MemTotal available beside the pool",
    )
    p.add_argument(
        "--min-placement-pct",
        type=float,
        default=98.0,
        help="fail if less of the resident pool is on --numa",
    )
    p.add_argument("--stats-interval-s", type=float, default=60.0)
    p.add_argument(
        "--teardown-pct",
        type=float,
        default=95.0,
        help="exit check: every NUMA node's available memory must be back to this %% of its value at start",
    )
    p.add_argument(
        "--teardown-timeout-s",
        type=float,
        default=20.0,
        help="time for release plus the exit memory check; keep it below the launcher's grace period before SIGKILL",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    host, pid = socket.gethostname(), os.getpid()
    stop = threading.Event()

    def on_signal(signum, _frame):
        if not stop.is_set():
            say(f"L3B_SIGNAL {signal.Signals(signum).name}: stopping")
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    nbytes = args.segment_gib * GIB
    say(
        f"L3B_SERVICE pid {pid} host {host} numa {args.numa} port {args.port} segment_bytes {nbytes}"
    )
    baseline = {n: available_kb(m) for n, m in meminfo().items()}
    for n in sorted(baseline):
        say(f"L3B_BASELINE host {host} numa {n} avail_kb {baseline[n]}")

    try:
        mode, nodes = bind_to(args.numa)
    except OSError as e:
        say(f"L3B_BIND FAIL numa {args.numa}: {e}")
        return 3
    if (mode, nodes) != ("bind", [args.numa]):
        say(f"L3B_BIND FAIL numa {args.numa}: memory policy is {mode} {nodes}")
        return 3
    m = meminfo().get(args.numa)
    if m is None:
        say(f"L3B_PREFLIGHT FAIL numa {args.numa} has no memory")
        return 1
    need_kb = nbytes // 1024 + int(m["MemTotal"] * args.headroom_pct / 100)
    if available_kb(m) < need_kb:
        say(
            f"L3B_PREFLIGHT FAIL numa {args.numa} avail_kb {available_kb(m)} need_kb {need_kb}"
        )
        return 1
    devices, seen, bad = select_devices(
        args.ucx_devices, args.link_layer, args.expect_ports
    )
    say(
        f"L3B_NICS host {host} seen {'; '.join(seen) or 'none'} | {args.ucx_devices} link {args.link_layer} "
        f"expect {args.expect_ports} -> {','.join(devices) or 'UCX picks'}"
    )
    if bad:
        say(
            f"L3B_PREFLIGHT FAIL devices: {'; '.join(bad)} | ports: {'; '.join(seen) or 'none'}"
        )
        return 1
    say(
        f"L3B_PREFLIGHT ok numa {args.numa} avail_kb {available_kb(m)} need_kb {need_kb} devices "
        f"{devices or 'any'}"
    )

    slab = args.slab_mib * MIB
    chunk = args.reg_chunk_gib * GIB
    if chunk % slab or nbytes % slab:
        say(
            f"L3B_ERROR the pool ({nbytes} B) and registration chunk ({chunk} B) must be multiples "
            f"of the slab ({slab} B)"
        )
        return 1
    pool = export = service = None
    rc = 0
    try:
        pool = Pool(nbytes, args.numa, args.prefault_threads, stop)
        where = placement(pool.base, nbytes)
        resident = sum(where.values())
        pct = 100.0 * where.get(args.numa, 0) / max(1, resident)
        say(
            f"L3B_POOL host {host} numa {args.numa} bytes {nbytes} base 0x{pool.base:x} "
            f"resident_gib {resident / GIB:.1f} on_node_pct {pct:.1f} by_node "
            + ",".join(f"N{n}={b / GIB:.1f}GiB" for n, b in sorted(where.items()))
        )
        if stop.is_set():
            raise SystemExit(0)
        if pct < args.min_placement_pct or resident < 0.9 * nbytes:
            say(
                f"L3B_PLACEMENT FAIL numa {args.numa}: {pct:.1f}% of the resident pool on its node "
                f"(need {args.min_placement_pct}%), resident {resident} of {nbytes} B"
            )
            raise SystemExit(3)
        export = NixlExport(
            f"nixl-dram-l3-{host}-numa{args.numa}-{pid}",
            pool,
            chunk,
            devices,
            args.num_workers,
            args.engine_config,
        )
        say(
            f"L3B_NIXL agent {export.name} backend UCX params {export.params} "
            f"registered_chunks {export.chunks} metadata_bytes {len(export.metadata)}"
        )
        index = SlabIndex(pool.base, nbytes, slab, args.align)
        ip = args.advertise_host or node_ip()
        service = Service(args, index, export, pool, host, ip, stop)
        service.publish()
        say(
            f"L3B_READY host {host} numa {args.numa} endpoint {ip}:{service.port} "
            f"pool_gib {nbytes / GIB:.2f} slab_mib {args.slab_mib} slabs {index.nslabs} pid {pid}"
        )
        service.serve()
    except SystemExit as e:
        rc = int(e.code or 0)
    except BaseException as e:  # noqa: BLE001 -- report, then always release the pool
        say(f"L3B_ERROR {e!r}\n{traceback.format_exc()}")
        rc = 1
    finally:
        t0 = time.monotonic()
        for what, fn in (
            ("service", service and service.close),
            ("export", export and export.close),
            ("pool", pool and pool.free),
        ):
            if fn:
                try:
                    fn()
                except Exception as e:
                    # Keep going: the memory check below is the verdict.
                    say(f"L3B_TEARDOWN FAIL {what}: {e!r}")
                    rc = rc or 1
        left = max(1.0, args.teardown_timeout_s - (time.monotonic() - t0))
        trc = teardown_check(host, baseline, args.teardown_pct, left)
        say(f"L3B_EXIT host {host} numa {args.numa} rc {rc or trc}")
    return rc or trc


if __name__ == "__main__":
    sys.exit(main())
