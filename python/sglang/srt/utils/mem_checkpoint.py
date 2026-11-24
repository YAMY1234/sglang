"""Startup GPU memory checkpoints (SGLANG_LOG_MEM_CHECKPOINTS=1, default off): live tensors,
the caching allocator's reserve and the device's free memory at each startup step, with the
change since the previous checkpoint. A free-memory drop without an allocator change is memory
outside the torch allocator (cuMem / NCCL / symmetric buffers)."""

import logging
from typing import Optional, Tuple

import torch
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GIB = float(1 << 30)
_last: Optional[Tuple[int, int, int]] = None


def mem_checkpoint(tag: str) -> None:
    global _last
    if not envs.SGLANG_LOG_MEM_CHECKPOINTS.get() or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    free, _ = torch.cuda.mem_get_info()
    delta = ""
    if _last is not None:
        a0, r0, f0 = _last
        outside = (f0 - free) - (reserved - r0)
        delta = (
            f" | d_allocated {(alloc - a0) / _GIB:+.3f} d_reserved "
            f"{(reserved - r0) / _GIB:+.3f} d_free {(free - f0) / _GIB:+.3f} "
            f"outside_torch {outside / _GIB:+.3f}"
        )
    _last = (alloc, reserved, free)
    logger.info(
        f"MEMCKPT {tag}: allocated {alloc / _GIB:.3f} reserved {reserved / _GIB:.3f} "
        f"free {free / _GIB:.3f} GiB{delta}"
    )
