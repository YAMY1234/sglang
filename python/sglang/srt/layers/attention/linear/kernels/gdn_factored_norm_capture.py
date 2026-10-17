"""Explicit diagnostic only: snapshot actual eager decode inputs for P4 replay.

Set DIR plus create FLAGFILE only after the service is ready and before a real
request. Never capture GPU graph capture/warmup; never enable in scored timing.
"""
import os
from pathlib import Path
import torch

_COUNTS = {}


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
    data = dict(schema="p4-service-step-v1", p=p,
                inp=dict(mixed=cpu(mixed), a=cpu(a), b=cpu(b), alog=cpu(alog)[None],
                         bias=cpu(bias)[None], vbar=cpu(vbar)[None]),
                slots=remapped, z=cpu(z), weight=cpu(weight), eps=eps,
                invalidate=invalidate, H=H, HV=HV, original_slots=original,
                source_pool_slots=selected)
    path = Path(dst)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(data, path / f"p4-pid{os.getpid()}-hv{HV}-b{B}-{n:03d}.pt")
