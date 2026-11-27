"""H331 knockout ablation (debug only, never merge): SGLANG_H331_SKIP=g1,g2 drops these groups' kernels from the
decode graph. Numerics are wrong by design; the step time is the measurement. Groups: mhc (mix stats: prenorm GEMM +
Sinkhorn), fin (MoE-seam finalize + all-reduce + post), ar1 (attention-seam all-reduce + post), dense (MXFP8 / wo_a
projections), attn (sparse MLA decode core), samp (draft Markov sample + verify accept)."""

import logging
import os

import torch

logger = logging.getLogger(__name__)
_SKIP = frozenset(g for g in os.environ.get("SGLANG_H331_SKIP", "").split(",") if g)
_SEEN = set()
_BUFS = {}


def skip(group: str) -> bool:
    if group not in _SKIP:
        return False
    if group not in _SEEN:
        _SEEN.add(group)
        logger.info("H331 skip engaged: %s", group)
    return True


def static(key, shape, dtype, device, fill=0.0) -> torch.Tensor:
    """A filled buffer per key, allocated on first use (eager warmup), so a replay launches nothing for it."""
    k = (key, tuple(shape), dtype, str(device))
    t = _BUFS.get(k)
    if t is None:
        t = torch.full(tuple(shape), fill, dtype=dtype, device=device)
        _BUFS[k] = t
    return t
