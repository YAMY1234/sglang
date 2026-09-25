"""Optional batched byte copies for private prefill graph inputs and outputs."""
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_logged = False


@triton.jit
def _copies(SRCS, DSTS, SHAPES: tl.constexpr, SS: tl.constexpr, DS: tl.constexpr,
            START: tl.constexpr, SIZE: tl.constexpr, WIDTH: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0)
    for i in tl.static_range(len(SHAPES)):
        if (pid >= START[i]) & (pid < START[i+1]):
            x = (pid - START[i])*B + tl.arange(0, B)
            rest = x
            so = tl.full((B,), 0, tl.int64)
            do = tl.full((B,), 0, tl.int64)
            for j in tl.static_range(len(SHAPES[i])-1, -1, -1):
                coord = rest % SHAPES[i][j]
                rest = rest // SHAPES[i][j]
                so += coord * SS[i][j]
                do += coord * DS[i][j]
            if WIDTH[i] == 1:
                kind: tl.constexpr = tl.uint8
            elif WIDTH[i] == 2:
                kind: tl.constexpr = tl.uint16
            elif WIDTH[i] == 4:
                kind: tl.constexpr = tl.uint32
            else:
                kind: tl.constexpr = tl.uint64
            value = tl.load(SRCS[i].to(tl.pointer_type(kind)) + so, x < SIZE[i], other=0)
            tl.store(DSTS[i].to(tl.pointer_type(kind)) + do, value, x < SIZE[i])


def copy_many(sources, destinations):
    """Disjoint owned destinations; arbitrary positive strides, identical dtype."""
    sources, destinations = tuple(sources), tuple(destinations)
    if not sources or len(sources) != len(destinations):
        raise ValueError('nonempty paired graph buffers required')
    for src, dst in zip(sources, destinations, strict=True):
        if src.shape != dst.shape or src.dtype != dst.dtype or src.device != dst.device:
            raise ValueError('graph copy layout/dtype/device mismatch')
    shapes = tuple(tuple(x.shape) for x in sources)
    sizes = tuple(x.numel() for x in sources)
    starts = [0]
    for size in sizes:
        starts.append(starts[-1] + triton.cdiv(size, 512))
    if starts[-1]:
        _copies[(starts[-1],)](sources, destinations, shapes,
            tuple(x.stride() for x in sources), tuple(x.stride() for x in destinations),
            tuple(starts), sizes, tuple(x.element_size() for x in sources), 512, num_warps=4)


def enabled(tensors):
    return (os.environ.get('SGLANG_GDN_PREFILL_PACKED_COPY', '0') == '1'
            and all(x.is_cuda for x in tensors))


def bind_many(sources, destinations):
    global _logged
    sources, destinations = tuple(sources), tuple(destinations)
    if enabled(sources):
        copy_many(sources, destinations)
        if not _logged:
            logger.info('GDN prefill packed graph copies used')
            _logged = True
    else:
        for dst, src in zip(destinations, sources, strict=True):
            dst.copy_(src)


def clone_many(sources):
    sources = tuple(sources)
    if not enabled(sources):
        return tuple(x.clone() for x in sources)
    outputs = tuple(torch.empty_like(x) for x in sources)
    bind_many(sources, outputs)
    return outputs
