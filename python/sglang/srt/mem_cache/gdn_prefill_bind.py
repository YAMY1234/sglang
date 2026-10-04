"""Pack prompt-commit inputs without one CUDA launch per layer/field.

Only descriptors cross H2D. State values keep their original copy conversions;
factorization, final/tracked ownership and publication remain in their callers.
"""
from bisect import bisect_left
import logging
import threading

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
FIELDS = 20


@triton.jit
def _pack(DESC, START: tl.constexpr, BLOCK: tl.constexpr):
    p = DESC + (START + tl.program_id(0)) * 20
    src, dst = tl.load(p), tl.load(p + 1)
    valid, n = tl.load(p + 2), tl.load(p + 3)
    fill, mode = tl.load(p + 4), tl.load(p + 5)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    d1, d2, d3 = tl.load(p + 7), tl.load(p + 8), tl.load(p + 9)
    c3 = i % d3
    c2 = (i // d3) % d2
    c1 = (i // (d3 * d2)) % d1
    c0 = i // (d3 * d2 * d1)
    so = (c0 * tl.load(p + 10) + c1 * tl.load(p + 11)
          + c2 * tl.load(p + 12) + c3 * tl.load(p + 13))
    do = (c0 * tl.load(p + 14) + c1 * tl.load(p + 15)
          + c2 * tl.load(p + 16) + c3 * tl.load(p + 17))
    item = tl.load(p + 19)
    read = (i < n) & (c0 < valid) & (mode >= 0)
    # mode=-1 is an exact self-view: retain valid rows, only fill padding.
    write = (i < n) & ((mode >= 0) | (c0 >= valid))
    if item == 8:
        if mode == 3:  # int32 -> int64, the same sign extension as copy_.
            value = tl.load((src + so).to(tl.pointer_type(tl.int32)), read, other=0).to(tl.int64)
        else:
            value = tl.load((src + so).to(tl.pointer_type(tl.int64)), read, other=0)
        tl.store((dst + do).to(tl.pointer_type(tl.int64)),
                 tl.where(c0 < valid, value, fill), write)
    elif item == 4:
        if mode == 1:  # BF16 -> FP32 is an exact bit expansion, including NaNs.
            value = tl.load((src + so).to(tl.pointer_type(tl.uint16)), read, other=0).to(tl.uint32) << 16
        elif mode == 2:
            value = tl.load((src + so).to(tl.pointer_type(tl.float16)), read, other=0).to(tl.float32).to(tl.uint32, bitcast=True)
        elif mode == 4:  # int64 -> int32 retains the low bits.
            value = tl.load((src + so).to(tl.pointer_type(tl.int64)), read, other=0).to(tl.uint32)
        else:
            value = tl.load((src + so).to(tl.pointer_type(tl.uint32)), read, other=0)
        tl.store((dst + do).to(tl.pointer_type(tl.uint32)),
                 tl.where(c0 < valid, value, fill.to(tl.uint32)), write)
    elif item == 2:
        value = tl.load((src + so).to(tl.pointer_type(tl.uint16)), read, other=0)
        tl.store((dst + do).to(tl.pointer_type(tl.uint16)),
                 tl.where(c0 < valid, value, fill.to(tl.uint16)), write)
    else:
        value = tl.load((src + so).to(tl.pointer_type(tl.uint8)), read, other=0)
        tl.store((dst + do).to(tl.pointer_type(tl.uint8)),
                 tl.where(c0 < valid, value, fill.to(tl.uint8)), write)


def extent(tensor):
    return (tensor.data_ptr(), tensor.data_ptr() + tensor.element_size() * (
        1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))))


def destination(tensor):
    if not 1 <= tensor.ndim <= 4 or any(s < 0 for s in tensor.stride()):
        raise ValueError("unsupported target layout")
    shape = tuple(tensor.shape) + (1,) * (4 - tensor.ndim)
    strides = tuple(s * tensor.element_size() for s in tensor.stride()) + (0,) * (4 - tensor.ndim)
    return [0, tensor.data_ptr(), 0, tensor.numel(), 0, 0, *shape,
            0, 0, 0, 0, *strides, 0, tensor.element_size()]


class PackedBind:
    def __init__(self, buffers):
        self.buffers = buffers
        self.device = buffers.pool.a.device
        self.lock = threading.Lock()
        self.targets = [(t, destination(t)) for t in (
            *buffers.normal, *(buffers.tracked or ()), buffers.slots,
            buffers.ring_dst, buffers.track_slots, buffers.final_src,
            buffers.final_dst, buffers.required) if t is not None]
        # Ring growth is also packed, not an uncounted per-layer fill loop.
        self.ring_views = list(buffers.ring_pointers.split(1))
        self.targets.extend((t, destination(t)) for t in self.ring_views)
        self.templates = {id(t): row for t, row in self.targets}
        self.ends = []
        self.ranges = []
        for lo, hi in sorted(extent(t) for t, _ in self.targets):
            if self.ranges and lo <= self.ranges[-1][1]:
                self.ranges[-1] = (self.ranges[-1][0], max(self.ranges[-1][1], hi))
            else:
                self.ranges.append((lo, hi))
        self.ends = [hi for _, hi in self.ranges]
        size = (len(self.targets), FIELDS)
        self.host = [torch.zeros(size, dtype=torch.int64, pin_memory=self.device.type == "cuda")
                     for _ in range(3)]
        self.host_arrays = [t.numpy() for t in self.host]
        self.meta = [torch.empty(size, dtype=torch.int64, device=self.device) for _ in range(3)]
        self.events = [None] * 3
        self.bank = 0
        self.calls = self.bank_waits = 0
        self.last_launches = 0
        self.fallbacks = {}

    def fallback(self, reason):
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1
        self.last_launches = 0
        if sum(self.fallbacks.values()) == 1 or sum(self.fallbacks.values()) % 100 == 0:
            logger.info("GDN packed bind fallback: reason=%s counts=%s", reason, self.fallbacks)
        return False

    def row(self, dst, src, fill):
        if dst is None:
            if src is not None and src.numel():
                raise ValueError("unexpected tracked controls")
            return None
        row = self.templates[id(dst)].copy()
        row[4] = fill
        if src is None or not src.numel():
            return row
        if (src.device != dst.device or src.ndim != dst.ndim
                or src.shape[0] > dst.shape[0] or src.shape[1:] != dst.shape[1:]
                or any(s < 0 for s in src.stride())):
            raise ValueError("source shape/device/layout")
        mode = 0
        if src.dtype != dst.dtype:
            mode = {(torch.bfloat16, torch.float32): 1,
                    (torch.float16, torch.float32): 2,
                    (torch.int32, torch.int64): 3,
                    (torch.int64, torch.int32): 4}.get((src.dtype, dst.dtype))
            if mode is None:
                raise ValueError("source conversion")
        same = (src.data_ptr() == dst.data_ptr() and src.dtype == dst.dtype
                and src.stride() == dst.stride())
        if same:
            mode = -1
        else:
            lo, hi = extent(src)
            i = bisect_left(self.ends, lo + 1)
            if i < len(self.ranges) and self.ranges[i][0] < hi:
                raise ValueError("cross-input alias")
        row[0], row[2], row[5] = src.data_ptr(), src.shape[0], mode
        row[10:14] = [s * src.element_size() for s in src.stride()] + [0] * (4 - src.ndim)
        row[18] = src.element_size()
        return row

    def prepare(self, plan, states, track_slots, final_src, final_dst):
        b = self.buffers
        rows, sources = [], []
        for i, (normal, tracked) in enumerate(states):
            if normal is None or (b.tracked is not None and tracked is None):
                raise ValueError("missing state")
            rows.append(self.row(b.normal[i], normal, 0)); sources.append(normal)
            if b.tracked is not None:
                rows.append(self.row(b.tracked[i], tracked, 0)); sources.append(tracked)
        split = len(rows)
        for dst, src, fill in (
            (b.slots, plan.slots, -1), (b.ring_dst, plan.ring_dst, -1),
            (b.track_slots, track_slots, -1), (b.final_src, final_src, -1),
            (b.final_dst, final_dst, -1), (b.required, plan.dense_required_after_commit, 0)
        ):
            row = self.row(dst, src, fill)
            if row is not None:
                rows.append(row)
            if src is not None:
                sources.append(src)
        generation = getattr(b.pool, "ring_generation", 0)
        if generation != b.ring_generation:
            for dst, source in zip(self.ring_views, b.pool.dense_ring):
                rows.append(self.row(dst, None, source.data_ptr()))
        return rows, split, sources, generation

    def run(self, plan, states, track_slots, final_src, final_dst):
        with self.lock:
            if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                return self.fallback("capture")
            # Validation is all-or-nothing, before any upload or state write.
            try:
                rows, split, sources, generation = self.prepare(
                    plan, states, track_slots, final_src, final_dst)
            except ValueError as error:
                return self.fallback(str(error))
            bank = self.bank
            event = self.events[bank]
            if event is not None and not event.query():
                # Protect pinned CPU memory too: a CUDA wait alone is not enough.
                event.synchronize()
                self.bank_waits += 1
            array = self.host_arrays[bank]
            array[:len(rows)] = rows
            self.meta[bank].copy_(self.host[bank], non_blocking=True)
            self.last_launches = 1
            if self.device.type == "cuda":
                stream = torch.cuda.current_stream(self.device)
                # Sources are hidden behind descriptors. Preserve their allocator
                # lifetime until the copies finish, without retaining whole states.
                for tensor in sources:
                    tensor.record_stream(stream)
            for start, end in ((0, split), (split, len(rows))):
                if start != end:
                    size = max(row[3] for row in rows[start:end])
                    _pack[(end - start, triton.cdiv(size, 512))](
                        self.meta[bank], start, 512, num_warps=4)
                    self.last_launches += 1
            if self.device.type == "cuda":
                event = torch.cuda.Event()
                event.record(stream)
                self.events[bank] = event
            self.buffers.ring_generation = generation
            self.bank = (bank + 1) % len(self.meta)
            self.calls += 1
            if self.calls == 1 or self.calls % 100 == 0:
                logger.info("GDN packed bind: calls=%d descriptors=%d copy_kernel_launches=%d "
                            "bank_waits=%d implicit_d2h=0 fallbacks=%s",
                            self.calls, len(rows), self.last_launches,
                            self.bank_waits, self.fallbacks)
            return True
