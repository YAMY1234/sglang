"""SGLANG_LINKER_HELPER: the direct linker's store I/O in a helper process.

The scheduler process keeps the relay, the consensus and the trees; per read or
write it sends a compact descriptor (page hashes and slot indices per pool).
The helper rebuilds the object keys and range arrays from per-pool tables
shipped once at start, runs the session, range-get and exist/put calls on its
own Mooncake client, and answers. Its GIL is its own, so none of that work
contends with the scheduler's forward launch (8.29.44: the linker's I/O threads
held the scheduler's GIL 6-9% of wall).

Transport: one duplex pipe (a kernel byte ring per direction; the syscalls also
order the shared GPU memory). Device memory reaches the helper as exported
allocations it maps (FABRIC handles on GPUs, shared memory in tests); pointers
in the tables are translated once at start.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Store codes worth one retry after renewing the read sessions (see
# MooncakeDirectLinker._get_layer_ranges): -600 / -707 lease, -800 transfer.
_RETRY_CODES = (-600, -707, -800)


@dataclass
class PoolTable:
    """Everything the helper needs about one pool, all picklable.

    ``ranges``: (src, base, stride, size, offset) per item of one key's read, as
    ``key_range_table``; ``puts``: (src, base, stride, size) per buffer of one
    page object, as ``page_buffer_table``; ``sources``: (page_size, row_span,
    row_count) per source column of the rows (two for a merged entry)."""

    name: str
    suffix: str
    ranges: np.ndarray
    puts: np.ndarray
    sources: list


@dataclass
class HelperInit:
    tables: dict
    prefix: str | None
    exports: list = field(default_factory=list)
    store_factory: object = None
    store_args: tuple = ()
    coalesce: bool = True
    threads: int = 4
    counted_sessions: bool = True
    # (module, class) of the put config the store's multi-buffer put takes
    put_config: tuple | None = None
    # the parent's registered buffer ranges (parent addresses); None: every span
    registered: list | None = None
    # CUDA device index to bind (None: no CUDA, as in the CPU tests)
    device: int | None = None


def pool_table(name, entry, suffix: str, num_layers: int) -> PoolTable | None:
    """The helper's table for a DevicePoolEntry / MergedPageEntry (None: unsupported)."""
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import page_buffer_table

    ranges = entry.key_range_table(num_layers)
    puts = page_buffer_table(entry)
    if ranges is None or puts is None:
        return None
    parts = [entry.first, entry.second] if hasattr(entry, "first") else [entry]
    sources = [(p.page_size, p._row_span, p._row_count) for p in parts]
    return PoolTable(str(name), suffix, ranges.copy(), puts.copy(), sources)


def rows_from_indices(indices: np.ndarray, page_size: int, row_span: int, row_count: int) -> np.ndarray:
    """``DevicePoolEntry._rows`` in numpy: page-aligned contiguous slots -> rows."""
    slots = np.asarray(indices, dtype=np.int64).reshape(-1)
    if slots.size % page_size:
        raise ValueError(f"{slots.size} indices, expected a multiple of page_size={page_size}")
    if not slots.size:
        return slots
    pages = slots.reshape(-1, page_size)
    starts = pages[:, 0]
    if (starts % page_size).any() or not np.array_equal(pages, starts[:, None] + np.arange(page_size)):
        raise ValueError("pool requires aligned contiguous pages")
    rows = starts // page_size if row_span == 1 else starts
    if rows.min() < 0 or rows.max() + row_span > row_count:
        raise ValueError(f"rows [{rows.min()}, {rows.max() + row_span}) exceed {row_count}")
    return rows


def coalesce_table(table: np.ndarray) -> np.ndarray:
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import coalesce_key_range_table

    return coalesce_key_range_table(table)


def _translate(table: np.ndarray, base_col: int, delta) -> np.ndarray:
    out = table.copy()
    out[:, base_col] = [delta(int(b)) for b in table[:, base_col]]
    return out


class _Translator:
    """Parent device address -> this process's mapping of the same allocation."""

    def __init__(self, mapped: list[tuple[int, int, int]]):
        # (parent_base, size, local_base)
        self.spans = sorted(mapped)

    def __call__(self, ptr: int) -> int:
        for base, size, local in self.spans:
            if base <= ptr < base + size:
                return local + (ptr - base)
        if not self.spans:
            return ptr
        raise ValueError(f"address {ptr:#x} is in no exported allocation")


def mooncake_client(host, metadata, segment, local_buffer, protocol, device_name, master):
    """The helper's store client: the parent's setup arguments, its own (small or
    empty) segment; it only moves data the parent's pools hold."""
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    rc = store.setup(host, metadata, segment, local_buffer, protocol, device_name, master)
    if rc:
        raise RuntimeError(f"helper Mooncake setup failed: {rc}")
    return store


def import_exports(exports: list, device: int | None = None) -> tuple[list, list]:
    """Map every exported allocation here. Returns (spans, keepalive)."""
    spans, keep = [], []
    for kind, parent_base, size, payload in exports:
        if kind == "shm":
            from multiprocessing import shared_memory

            shm = shared_memory.SharedMemory(name=payload)
            buf = np.frombuffer(shm.buf, dtype=np.uint8)
            spans.append((parent_base, size, buf.ctypes.data))
            keep += [shm, buf]
        elif kind == "fabric":
            from sglang.srt.mem_cache.storage.mooncake_store.linker_helper_vmm import import_fabric

            local, handle = import_fabric(payload, size, device)
            spans.append((parent_base, size, local))
            keep.append(handle)
        else:
            raise ValueError(f"unknown export kind {kind}")
    return spans, keep


class HelperWorker:
    """The helper's side: tables translated to local addresses, one store
    client, the read sessions, and the read / write / exists operations."""

    def __init__(self, init: HelperInit, store, spans: list):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            _RecentKeys,
            _SessionRefs,
        )

        self.store, self.prefix, self.coalesce = store, init.prefix, init.coalesce
        self.sessions = _SessionRefs(store, init.counted_sessions)
        self.recently_loaded = _RecentKeys(500_000)
        self.counts: Counter = Counter()
        tr = _Translator(spans)
        self.tables = {}
        for name, t in init.tables.items():
            # Translate before coalescing: allocations adjacent in the parent's
            # address space are mapped separately here.
            ranges = _translate(t.ranges, 1, tr)
            if self.coalesce:
                ranges = coalesce_table(ranges)
            self.tables[name] = PoolTable(name, t.suffix, ranges, _translate(t.puts, 1, tr), t.sources)
        self.put_config = None
        if init.put_config is not None:
            module, cls = init.put_config
            self.put_config = getattr(__import__(module, fromlist=[cls]), cls)()

    # ---- descriptors -> keys and ranges -------------------------------------------

    def _keys(self, name: str, hashes: list[str]) -> list[str]:
        suffix = self.tables[name].suffix
        if self.prefix is None:
            return [f"{h}{suffix}" for h in hashes]
        return [f"{self.prefix}_{h}{suffix}" for h in hashes]

    def _rows(self, name: str, indices: np.ndarray) -> np.ndarray:
        cols = [rows_from_indices(indices, *src) for src in self.tables[name].sources]
        return np.stack(cols, axis=1)

    def read_ranges(self, name: str, indices: np.ndarray):
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import RangeArrays

        table = self.tables[name].ranges
        rows = self._rows(name, indices)
        src, base, stride, size, offset = table.T
        ptrs = base[None, :] + rows[:, src] * stride[None, :]
        return RangeArrays.from_rows(ptrs, size, offset)

    def put_meta(self, name: str, indices: np.ndarray):
        table = self.tables[name].puts
        rows = self._rows(name, indices)
        src, base, stride, size = table.T
        ptrs = base[None, :] + rows[:, src] * stride[None, :]
        return ptrs.tolist(), [size.tolist()] * rows.shape[0]

    def set_prefix(self, prefix: str | None) -> None:
        self.prefix = prefix

    # ---- operations -----------------------------------------------------------------

    def _range_get(self, keys, ranges):
        method = getattr(self.store, "batch_get_into_multi_buffer_ranges_arrays", None)
        if method is not None:
            return list(method(keys, *ranges))
        return list(self.store.batch_get_into_multi_buffer_ranges(keys, *ranges.to_lists()))

    def read(self, pools: list) -> tuple:
        """``pools``: [(name, hashes, indices)]. Returns (True | reason, info)."""
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import RangeArrays

        t0 = time.monotonic()
        keys, parts = [], []
        for name, hashes, indices in pools:
            keys += self._keys(name, hashes)
            parts.append(self.read_ranges(name, indices))
        if not keys:
            return True, {"objects": 0}
        ranges = RangeArrays.concat(parts)
        self.recently_loaded.add(keys)
        started = self.sessions.start(keys)
        if started != [0] * len(keys):
            failed = Counter(v for v in started if v != 0)
            self.counts["session_failed"] += 1
            return "session_start:" + ",".join(f"{k}x{v}" for k, v in sorted(failed.items())), {}
        try:
            expected = ranges.expected()
            result = self._range_get(keys, ranges)
            retry = [i for i, (v, want) in enumerate(zip(result, expected)) if v != want]
            if retry and all(result[i] in _RETRY_CODES for i in retry):
                lease = [keys[i] for i in retry if result[i] in (-600, -707)]
                if self.sessions.renew(lease) != [0] * len(lease):
                    return "read:session renewal failed", {}
                again = self._range_get([keys[i] for i in retry], ranges.select(retry))
                for i, v in zip(retry, again):
                    result[i] = v
                self.counts["retried"] += len(retry)
            if result != expected:
                failed = Counter(v for v, want in zip(result, expected) if v != want)
                self.counts["read_failed"] += 1
                return f"read:failed by code {dict(failed)}", {}
        finally:
            self.sessions.end(keys)
        self.counts["reads"] += 1
        return True, {"objects": len(keys), "bytes": int(sum(expected)), "ms": (time.monotonic() - t0) * 1e3}

    def write(self, pools: list) -> tuple:
        """``pools``: [(name, hashes, indices)]. Puts the objects the store lacks
        (skipping keys a read just took). Returns ({name: [ok per page]}, info)."""
        keys, owner, ptrs, sizes = [], [], [], []
        for name, hashes, indices in pools:
            k = self._keys(name, hashes)
            p, s = self.put_meta(name, indices)
            keys += k
            owner += [name] * len(k)
            ptrs += p
            sizes += s
        recent = self.recently_loaded.keys
        ask = [i for i, key in enumerate(keys) if key not in recent]
        exist = [1] * len(keys)
        if ask:
            for i, state in zip(ask, self.store.batch_is_exist([keys[i] for i in ask])):
                exist[i] = state
        missing = [i for i, state in enumerate(exist) if state != 1]
        codes = [0] * len(keys)
        if missing:
            args = ([keys[i] for i in missing], [ptrs[i] for i in missing], [sizes[i] for i in missing])
            if self.put_config is not None:
                put = self.store.batch_put_from_multi_buffers(*args, self.put_config)
            else:
                put = self.store.batch_put_from_multi_buffers(*args)
            for i, code in zip(missing, put):
                codes[i] = code
        results: dict = {}
        for name, code in zip(owner, codes):
            results.setdefault(name, []).append(code == 0)
        put_objects = Counter(owner[i] for i in missing)
        self.counts["writes"] += 1
        return results, {"put_objects": dict(put_objects), "exist_skipped": len(keys) - len(ask)}

    def exists(self, keys: list[str]) -> list[int]:
        return list(self.store.batch_is_exist(keys))


# ---- process plumbing ---------------------------------------------------------------


def helper_main(conn, init: HelperInit) -> None:
    """The helper process: build the store client and the worker, then serve."""
    # The scheduler's env may force eager module loading, which hung the helper
    # smoke (8.29.39); the helper launches no kernels.
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    try:
        if init.device is not None:
            import torch

            torch.cuda.set_device(init.device)
        spans, keep = import_exports(init.exports, init.device)
        store = init.store_factory(*init.store_args)
        tr = _Translator(spans)
        if hasattr(store, "register_buffer"):
            ranges = init.registered or [(base, size) for base, size, _ in spans]
            for ptr, size in ranges:
                rc = store.register_buffer(tr(ptr), size)
                if rc not in (0, None):
                    raise RuntimeError(f"register_buffer({size} bytes) -> {rc}")
        worker = HelperWorker(init, store, spans)
    except BaseException as error:  # report and exit; the parent falls back
        conn.send(("init_failed", repr(error)))
        return
    conn.send(("ready", os.getpid()))
    send_lock = threading.Lock()
    pool = ThreadPoolExecutor(init.threads, thread_name_prefix="linker-helper")

    def run(req_id, op, args):
        try:
            if op == "read":
                out = worker.read(*args)
            elif op == "write":
                out = worker.write(*args)
            elif op == "exists":
                out = worker.exists(*args)
            elif op == "prefix":
                out = worker.set_prefix(*args)
            elif op == "stats":
                out = dict(worker.counts)
            else:
                raise ValueError(f"unknown op {op}")
            reply = (req_id, True, out)
        except BaseException as error:
            reply = (req_id, False, repr(error))
        with send_lock:
            conn.send(reply)

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        req_id, op, args = msg
        if op in ("prefix",):
            run(req_id, op, args)  # in order with the requests after it
        else:
            pool.submit(run, req_id, op, args)
    pool.shutdown(wait=True)
    del keep


class HelperClient:
    """The scheduler side: start the helper, submit operations, resolve futures
    on one completion thread (blocked in recv, so it holds no GIL while idle)."""

    def __init__(self, init: HelperInit, start_timeout_s: float = 300.0):
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=helper_main, args=(child, init), daemon=True, name="linker-helper")
        self.proc.start()
        child.close()
        if not self.conn.poll(start_timeout_s):
            self.proc.kill()
            raise RuntimeError("linker helper did not start")
        status, detail = self.conn.recv()
        if status != "ready":
            self.proc.join(5)
            raise RuntimeError(f"linker helper failed to start: {detail}")
        self.pid = detail
        self.futures: dict[int, Future] = {}
        self.next_id = 0
        self.lock = threading.Lock()
        self.closed = False
        self.reader = threading.Thread(target=self._read_loop, name="linker-helper-rx", daemon=True)
        self.reader.start()

    def submit(self, op: str, *args) -> Future:
        future: Future = Future()
        with self.lock:
            if self.closed:
                future.set_exception(RuntimeError("linker helper is closed"))
                return future
            self.next_id += 1
            req_id = self.next_id
            self.futures[req_id] = future
            self.conn.send((req_id, op, args))
        return future

    def _read_loop(self) -> None:
        while True:
            try:
                req_id, ok, value = self.conn.recv()
            except (EOFError, OSError):
                break
            future = self.futures.pop(req_id, None)
            if future is None:
                continue
            if ok:
                future.set_result(value)
            else:
                future.set_exception(RuntimeError(f"linker helper: {value}"))
        with self.lock:
            self.closed = True
            pending, self.futures = self.futures, {}
        for future in pending.values():
            future.set_exception(RuntimeError("linker helper exited"))

    def close(self, timeout_s: float = 30.0) -> None:
        with self.lock:
            if self.closed:
                return
            self.conn.send(None)
        self.proc.join(timeout_s)
        if self.proc.is_alive():
            self.proc.kill()
        self.reader.join(5)
