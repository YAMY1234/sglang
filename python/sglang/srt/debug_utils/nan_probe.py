"""SGLANG_DEBUG_NAN_PROBE: find the first non-finite row on the decode forward path.

Each probed site marks its non-finite rows in a static device table, graph-safe (no
host sync), and keeps the forward's req_pool_indices when it saw one. Mega-gate inputs
are zeroed in place after the mark, so the gate's device assert does not trap first.
After the forward, `check` reads the table on the host, logs the first site (in call
order) with bad rows, the requests behind them and their admission record, and raises.
Debug only: it adds a full device sync per decode step.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MAX_SITES = 256
_MAX_ROWS = 16384

_sites: Dict[str, int] = {}
_names: List[str] = []
_flags: Optional[torch.Tensor] = None  # bool [sites, rows]
# int64 [sites, rows]: req_pool_indices at the bad call; [sites, 2]: rows, batch size.
_rpi: Optional[torch.Tensor] = None
_shape: Optional[torch.Tensor] = None


def enabled() -> bool:
    return envs.SGLANG_DEBUG_NAN_PROBE.get()


def _tables(device) -> None:
    global _flags, _rpi, _shape
    if _flags is None:
        _flags = torch.zeros((_MAX_SITES, _MAX_ROWS), dtype=torch.bool, device=device)
        _rpi = torch.full((_MAX_SITES, _MAX_ROWS), -1, dtype=torch.int64, device=device)
        _shape = torch.zeros((_MAX_SITES, 2), dtype=torch.int64, device=device)


def record(name: str, x, forward_batch=None, sanitize: bool = False) -> None:
    """Mark the non-finite rows of x (rows = dim 0) for site `name`."""
    if not isinstance(x, torch.Tensor) or x.dim() == 0:
        return
    n = x.shape[0]
    if n == 0 or n > _MAX_ROWS:
        return
    _tables(x.device)
    site = _sites.get(name)
    if site is None:
        if len(_names) >= _MAX_SITES:
            return
        site = _sites[name] = len(_names)
        _names.append(name)
    bad = ~torch.isfinite(x.reshape(n, -1)).all(dim=-1)
    _flags[site, :n] |= bad
    rpi = getattr(forward_batch, "req_pool_indices", None)
    bs = 0
    if isinstance(rpi, torch.Tensor) and 0 < rpi.shape[0] <= _MAX_ROWS:
        bs = rpi.shape[0]
        cur = _rpi[site, :bs]
        cur.copy_(torch.where(bad.any(), rpi.to(torch.int64), cur))
    _shape[site, 0].fill_(n)
    _shape[site, 1].fill_(bs)
    if sanitize:
        x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


def note_admit(scheduler, reqs, now: float, held_since: Optional[dict]) -> None:
    """Log every admission while the probe is on, and keep it on the request."""
    for req in reqs:
        held = held_since.get(req.rid, now) if held_since else now
        req.nan_probe_admit = (scheduler.forward_ct, 1000 * (now - held))
        logger.info(
            "NaN probe admit: rid=%s req_pool_idx=%s metadata_slot=%s step=%d "
            "hold_ms=%.0f waited_ms=%.0f origin_len=%d out_len=%d",
            req.rid,
            req.kv.req_pool_idx,
            getattr(req, "nan_probe_slot", None),
            scheduler.forward_ct,
            1000 * (now - held),
            1000 * (time.monotonic() - getattr(req, "nan_probe_transfer_t", now)),
            len(req.origin_input_ids),
            len(req.output_ids),
        )


def check(scheduler, batch, result) -> None:
    """After a decode forward: raise on the first site that saw a non-finite row."""
    if _flags is None:
        return
    copy_done = getattr(result, "copy_done", None)
    if copy_done is not None:
        copy_done.synchronize()
    # No device-wide sync, so the overlap timing stays as in production; a forward
    # launched after this one may also have marked rows, and its req_pool_idx names them.
    hit = _flags[: len(_names)].any(dim=1).nonzero().flatten().tolist()
    if not hit:
        return
    by_rpi = {}
    for owner in (scheduler.running_batch.reqs, batch.reqs):
        for req in owner:
            if req.kv.req_pool_idx is not None:
                by_rpi[req.kv.req_pool_idx] = req
    pos_of = {id(req): i for i, req in enumerate(batch.reqs)}
    from sglang.srt.runtime_context import get_parallel

    rank = get_parallel().attn_dp_rank
    lines = []
    for site in hit:
        rows_n, bs = _shape[site].tolist()
        rows = _flags[site, :rows_n].nonzero().flatten().tolist()
        per = rows_n // bs if bs and rows_n % bs == 0 else 0
        desc = []
        for p in sorted({r // per for r in rows}) if per else []:
            rpi = int(_rpi[site, p])
            req = by_rpi.get(rpi)
            if req is None:
                desc.append(f"pos={p} req_pool_idx={rpi} req=?")
                continue
            step, hold = getattr(req, "nan_probe_admit", (None, None))
            desc.append(
                f"pos={p} req_pool_idx={rpi} rid={req.rid} batch_pos={pos_of.get(id(req))} "
                f"metadata_slot={getattr(req, 'nan_probe_slot', None)} admit_step={step} "
                f"hold_ms={hold} out_len={len(req.output_ids)} "
                f"seq_len={len(req.origin_input_ids) + len(req.output_ids)}"
            )
        lines.append(
            f"site#{site} {_names[site]} rows={rows} of {rows_n} bs={bs} per_req={per} "
            + "; ".join(desc)
        )
    msg = (
        f"NaN probe tripped: DP{rank} step={scheduler.forward_ct} batch={len(batch.reqs)} "
        f"waiting={len(scheduler.waiting_queue)} first={_names[hit[0]]}\n  "
        + "\n  ".join(lines)
    )
    logger.error(msg)
    raise RuntimeError(msg)
