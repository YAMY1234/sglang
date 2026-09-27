"""Opt-in Flash-Next boundary protocol; no factor arithmetic or transactions."""
from sglang.srt.runtime_context import get_disagg


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
    if not any(getattr(req, 'flashnext_pd_boundary_pending', False) for req in batch.reqs):
        return
    if not all(getattr(req, 'flashnext_pd_boundary_pending', False) for req in batch.reqs):
        raise RuntimeError("mixed legacy and shallow PD boundary batch")
    model = scheduler.tp_worker.model_runner.model
    complete = getattr(model, 'complete_pd_boundary', None)
    if complete is None:
        raise RuntimeError("shallow PD receiver has no boundary implementation")
    complete(batch, scheduler)


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
