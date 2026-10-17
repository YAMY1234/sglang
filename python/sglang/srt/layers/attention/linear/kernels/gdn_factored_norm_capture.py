"""Explicit diagnostic only: snapshot actual eager decode inputs for P4 replay.

Set DIR plus create FLAGFILE only after the service is ready and before a real
request. Never capture GPU graph capture/warmup; never enable in scored timing.
"""
import logging
import os
from pathlib import Path
import torch

_COUNTS = {}
_PRODUCED = set()


def log_produced_z(z, norm_requested):
    key = (z.shape[0], z.shape[1], tuple(z.stride()), z.storage_offset())
    if key not in _PRODUCED and z.shape[0] <= 8:
        _PRODUCED.add(key)
        logging.getLogger(__name__).warning("P4_PRODUCED_Z norm_requested=%s metadata=%s",
                                           norm_requested, z_metadata(z))



def z_metadata(z):
    return dict(shape=list(z.shape), stride=list(z.stride()), storage_offset=z.storage_offset())


def restore_z(values, metadata):
    """Rebuild the captured view outside graph/timing, preserving even its offset."""
    shape, stride, offset = metadata['shape'], metadata['stride'], metadata['storage_offset']
    assert tuple(values.shape) == tuple(shape)
    assert len(shape) == len(stride) == 3 and all(s > 0 for s in stride) and offset >= 0
    size = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    backing = torch.empty(size, dtype=values.dtype, device=values.device)
    z = backing.as_strided(shape, stride, offset)
    z.copy_(values)
    return z


def capture_step(mixed, a, b, alog, bias, vbar, fa, fu, fw, count, stale,
                 indices, prefix_valid, invalidate, z, weight, eps, H):
    dst = os.environ.get("SGLANG_GDN_P4_CAPTURE_DIR")
    flag = os.environ.get("SGLANG_GDN_P4_CAPTURE_FLAGFILE")
    if not dst or not flag or not Path(flag).is_file() or torch.cuda.is_current_stream_capturing():
        return
    B, HV = mixed.shape[0], fu.shape[1]
    key = (B, HV)
    limit = int(os.environ.get("SGLANG_GDN_P4_CAPTURE_LIMIT", "36"))
    n = _COUNTS.get(key, 0)
    if n >= limit:
        return
    _COUNTS[key] = n + 1
    original = indices.detach().cpu().long()
    selected = sorted({0, *[int(x) for x in original if x >= 0]})
    mapping = {slot: j for j, slot in enumerate(selected)}
    remapped = torch.tensor([mapping[int(x)] if x >= 0 else -1 for x in original], dtype=indices.dtype)
    pick = torch.tensor(selected, device=indices.device, dtype=torch.long)
    def cpu(x):
        return x.detach().contiguous().cpu()
    p = {k: cpu(t.index_select(0, pick))[None]
         for k, t in (("fa", fa), ("fu", fu), ("fw", fw), ("fcount", count))}
    p["stale"] = cpu(stale.index_select(0, pick))
    p["prefix_valid"] = (cpu(prefix_valid.index_select(0, pick)) if prefix_valid is not None
                           else torch.ones(len(selected), dtype=torch.int32))
    data = dict(schema="p4-service-step-v2", p=p,
                inp=dict(mixed=cpu(mixed), a=cpu(a), b=cpu(b), alog=cpu(alog)[None],
                         bias=cpu(bias)[None], vbar=cpu(vbar)[None]),
                slots=remapped, z=cpu(z), z_metadata=z_metadata(z), weight=cpu(weight), eps=eps,
                invalidate=invalidate, H=H, HV=HV, original_slots=original,
                source_pool_slots=selected)
    path = Path(dst)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(data, path / f"p4-pid{os.getpid()}-hv{HV}-b{B}-{n:03d}.pt")
