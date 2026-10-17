"""Opt-in Flash-Next boundary protocol; no factor arithmetic or transactions."""
import copy
import logging
import os

from sglang.srt.runtime_context import get_disagg

logger = logging.getLogger(__name__)
_ASYNC = None


def boundary_async():
    """SGLANG_GDN_PD_BOUNDARY_ASYNC=off|concurrent (docs/233 P6); read once."""
    global _ASYNC
    if _ASYNC is None:
        value = os.environ.get("SGLANG_GDN_PD_BOUNDARY_ASYNC", "off")
        if value not in ("off", "concurrent"):
            raise ValueError(f"SGLANG_GDN_PD_BOUNDARY_ASYNC={value!r}")
        _ASYNC = value == "concurrent"
        if _ASYNC:
            logger.info("PD_BOUNDARY_ASYNC_ACTIVE scheduler: slot-reuse events replace the PREBUILT stream fence")
    return _ASYNC


def note_preallocated(scheduler, decode_reqs):
    """Concurrent boundary: remember what the reused slots' previous occupants
    may still have in flight. Every forward that could touch a freed row was
    launched before this allocation; an event on forward_stream now covers it."""
    if not decode_reqs or not boundary_async():
        return
    import torch

    event = torch.cuda.Event()
    event.record(scheduler.forward_stream)
    for decode_req in decode_reqs:
        decode_req.req.flashnext_slot_reuse_event = event


def prebuilt_fence(scheduler, requests):
    """Stock: drain forward_stream before a prebuilt request seeds a reused row.
    Concurrent boundary: wait only for the events recorded when those rows were
    preallocated (falls back to the stock fence for any request without one)."""
    if not boundary_async():
        scheduler.schedule_stream.wait_stream(scheduler.forward_stream)
        return
    events = [getattr(req, "flashnext_slot_reuse_event", None) for req in requests]
    pending = bool(requests) and all(
        getattr(req, "flashnext_pd_boundary_pending", False) for req in requests
    )
    # The admission counter advances identically on every TP rank (same
    # scheduling decisions), so all ranks pick the same boundary path.
    use = pending and _ab_concurrent() and all(e is not None for e in events)
    for req in requests:
        req.flashnext_slot_reuse_event = None  # one admission per allocation
        req.flashnext_boundary_concurrent = use
    if use:
        for event in {id(e): e for e in events}.values():
            scheduler.schedule_stream.wait_event(event)
        return
    scheduler.schedule_stream.wait_stream(scheduler.forward_stream)


_AB = None
_admissions = 0


def _ab_concurrent():
    """SGLANG_GDN_PD_BOUNDARY_AB=alternate:<K> (docs/233 §4 A/B): boundary
    admissions alternate serial / concurrent, parity swapped after K
    admissions. Unset: every boundary admission is concurrent."""
    global _AB, _admissions
    if _AB is None:
        value = os.environ.get("SGLANG_GDN_PD_BOUNDARY_AB", "")
        _AB = int(value.split(":", 1)[1]) if value.startswith("alternate:") else -1
        if value and _AB < 0:
            raise ValueError(f"SGLANG_GDN_PD_BOUNDARY_AB={value!r}")
    n = _admissions
    _admissions += 1
    if _AB < 0:
        return True
    return (n % 2 == 0) != (n >= _AB)


def enabled(model_config):
    from sglang.srt.model_executor.fullstack_policy import fullstack_enabled

    return (get_disagg().flashnext_pd_shallow_prefill
            and fullstack_enabled(model_config))


def pending_receive(scheduler, decode_req):
    from sglang.srt.disaggregation.decode import _is_fake_transfer

    active = enabled(scheduler.model_config) and not _is_fake_transfer(decode_req.req)
    if active:
        if decode_req.is_rebootstrap:
            raise RuntimeError("shallow PD boundary does not support rebootstrap yet")
        if decode_req.req.return_sampling_mask:
            raise ValueError("shallow PD boundary does not support sampling-mask export")
        decode_req.req.flashnext_pd_boundary_pending = True
    return active


def complete_prebuilt(scheduler, batch):
    pending = batch is not None and any(
        getattr(req, 'flashnext_pd_boundary_pending', False) for req in batch.reqs
    )
    if pending and not all(getattr(req, 'flashnext_pd_boundary_pending', False) for req in batch.reqs):
        raise RuntimeError("mixed legacy and shallow PD boundary batch")

    # This entry is called by every scheduler, including ranks without a new
    # PREBUILT batch. Boundary MoE layers use the same DP collectives as decode;
    # their counts must be gathered before any rank enters the deep layers.
    dp_boundary = enabled(scheduler.model_config) and scheduler.dp_attn_adapter.ps.attn_dp_size > 1
    if dp_boundary and boundary_async():
        raise ValueError("concurrent PD boundary (docs/233) supports attention DP 1 only")
    work = batch if pending else None
    if dp_boundary:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if scheduler.enable_overlap:
            scheduler.schedule_stream.wait_stream(scheduler.forward_stream)
        if work is not None:
            work = copy.copy(work)
            # PREBUILT normally votes zero tokens. Each incoming boundary
            # instead performs one decode token; preserve the original mode.
            work.forward_mode = ForwardMode.DECODE
            work.is_extend_in_batch = False
        work = scheduler.dp_attn_adapter.prepare_mlp_sync_batch(work)
    if work is None:
        return
    model = scheduler.tp_worker.model_runner.model
    complete = getattr(model, 'complete_pd_boundary', None)
    if complete is None:
        raise RuntimeError("shallow PD receiver has no boundary implementation")
    complete(work, scheduler)
    if pending:
        batch.sampling_info = work.sampling_info


def partition_prebuilt(requests):
    """Keep fake health probes and real boundary work in adjacent batches.

    A fake transfer intentionally has no h31 and must retain its ordinary
    PREBUILT path. Select the first request's phase, preserving order within
    both groups. Deferred requests go at the front of the waiting queue so a
    health probe cannot starve behind newly arriving boundary work. Uniform
    batches (including stock/flag-off) retain the original list object.
    """
    if not requests:
        return requests, []
    phase = bool(getattr(requests[0], "flashnext_pd_boundary_pending", False))
    selected, deferred = [], []
    for req in requests:
        target = selected if bool(getattr(req, "flashnext_pd_boundary_pending", False)) == phase else deferred
        target.append(req)
    return (selected, deferred) if deferred else (requests, [])
