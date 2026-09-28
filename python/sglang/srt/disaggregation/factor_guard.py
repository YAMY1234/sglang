"""Degraded factored-GDN checkpoint guard on the P side (TwinStar docs/139, opt-in SGLANG_FLASHNEXT_FACTOR_GUARD_ABORT).

When the extend plan finds a cached prefix whose factored checkpoint is invalid it records the request instead of
killing the scheduler. After the forward, before any result is used, every attention-TP rank agrees on the flagged
requests (max all-reduce, so the scheduling decision stays identical across ranks), clears the checkpoint flag of
everything the request published in that forward (its slot and ping-pong track slots), and aborts it through the
native paths: the final chunk is dropped by the batch-result loop (nothing is sent to D or cached), an in-flight
chunked request goes through process_pending_chunked_abort, which runs before the chunk is cached.
"""
import logging
from http import HTTPStatus

import torch

logger = logging.getLogger(__name__)
ABORTED = [0]  # guard aborts in this process (for the cell annotation)


def apply(scheduler, batch) -> int:
    from sglang.srt.disaggregation.utils import prepare_abort
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.mem_cache.gdn_factored_pool import pop_guard_aborts

    flagged = pop_guard_aborts()
    flags = torch.tensor([int(req.rid in flagged) for req in batch.reqs], dtype=torch.int32)
    group = getattr(scheduler, "attn_tp_cpu_group", None)
    if group is not None and torch.distributed.is_initialized() and torch.distributed.get_world_size(group) > 1:
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX, group=group)
    count = 0
    for req, bad in zip(batch.reqs, flags.tolist()):
        if not bad:
            continue
        message = flagged.get(req.rid, "cached x256 P prefix has no factored GDN checkpoint (reported by another TP rank)")
        invalidate(scheduler.req_to_token_pool, req)
        prepare_abort(req, message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        if getattr(scheduler, "chunked_req", None) is req:
            scheduler.abort_request(AbortReq(rid=req.rid))
        ABORTED[0] += 1
        count += 1
        logger.error("FACTOR_GUARD_ABORT rid=%s total=%d: %s", req.rid, ABORTED[0], message)
    return count


def invalidate(pool, req) -> None:
    """Checkpoint flag 0 on every slot this request published in the guarded forward."""
    factor = getattr(pool, "factored_gdn_pool", None)
    if factor is None or factor.prefix_valid is None:
        return
    parts = [t.reshape(-1) for t in (getattr(req.kv, "mamba_pool_idx", None),
                                     getattr(req.kv, "mamba_ping_pong_track_buffer", None)) if t is not None]
    if not parts:
        return
    slots = torch.cat([p.to(factor.prefix_valid.device) for p in parts]).long()
    slots = slots[slots >= 0]
    translate = getattr(pool, "translate_mamba_indices", None)
    factor.invalidate_prefix_dense(translate(slots) if translate is not None else slots)
