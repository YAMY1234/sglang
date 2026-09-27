"""Diagnostic latent-row provenance audit for the shared arena (#905), opt-in.

SGLANG_FLASHNEXT_LATENT_AUDIT=1 only. Stored numerics are unchanged: the stamp
lives in the wire row's zero padding (bytes 1976..1995 of the 2,048-byte unit
pair), and decode parses only the first 1,972 bytes. Anomalies are logged as
JSON lines prefixed ``LATENT_AUDIT``; the audit never repairs or skips a row.

Three checks, one per suspected mechanism of the "malformed TP gap8 stream":
  * forward: pending arena mappings exist but the forward carries no CPU request
    indices, so ``prepare_request_mappings`` cannot publish them to the GPU map;
  * store: the GPU virtual->physical map for the pages being written differs
    from the CPU arena table (the write lands in another owner's units);
  * load: each materialized row is classified by its stamp (never written,
    another token's latent, overwritten by a non-latent write) and gap8 validity.
"""
import json
import logging
import os

import torch

ENABLED = os.environ.get("SGLANG_FLASHNEXT_LATENT_AUDIT", "0") == "1"
MAGIC = 0x3E4C4154
STAMP = slice(1976, 1996)  # 5 x int32: magic, position, virtual location, 0, store epoch
LIMIT = 40
logger = logging.getLogger(__name__)
counts = {}


def emit(kind, **fields):
    n = counts.get(kind, 0) + 1
    counts[kind] = n
    if n <= LIMIT:
        logger.error("LATENT_AUDIT " + json.dumps(dict(kind=kind, n=n, **fields), default=str))


def stamp(payload, locations, positions, epoch):
    n = payload.shape[0]
    s = torch.zeros((n, 5), dtype=torch.int32, device=payload.device)
    s[:, 0] = MAGIC
    s[:, 1] = -1 if positions is None else positions.reshape(-1).to(torch.int32)
    s[:, 2] = locations.reshape(-1).to(torch.int32)
    s[:, 4] = epoch
    payload[:, STAMP] = s.view(torch.uint8).reshape(n, 20)


def read_stamp(payload):
    return payload[:, STAMP].contiguous().view(torch.int32).reshape(-1, 5)


def forward_unflushed(pool, host_indices):
    arena = pool.arena
    if host_indices is None and (arena.pending_shared or arena.pending_deep):
        emit("forward_unflushed", pending_shared=sorted(arena.pending_shared)[:16],
             n_pending_shared=len(arena.pending_shared), n_pending_deep=len(arena.pending_deep))


def check_store_map(pool, locations):
    pages = torch.unique(locations.long() // pool.page_size).cpu().tolist()
    gpu = pool.physical_page_map[pages].cpu().tolist()
    bad = []
    for page, row in zip(pages, gpu):
        cpu = pool.arena.shared.get(page)
        if cpu is None or list(cpu) != row:
            bad.append(dict(page=page, gpu=row, cpu=cpu, pending=page in pool.arena.pending_shared))
    if bad:
        emit("store_map_mismatch", pages=len(pages), bad=len(bad), first=bad[:8])


def gap8_valid(stream, lengths, sparse, width):
    """Per-row validity of the reassembled stream (same test as the decode assert)."""
    if stream.is_cuda:
        import triton
        from sglang.srt.mem_cache.flashnext_scheme_c_kernels import _unpack_gap
        n, cap = stream.shape
        out = torch.empty((n, sparse), dtype=torch.int64, device=stream.device)
        valid = torch.empty(n, dtype=torch.bool, device=stream.device)
        if n:
            _unpack_gap[(n,)](stream, lengths, out, valid, *stream.stride(), lengths.stride(0),
                              cap, sparse, width, triton.next_power_of_2(cap))
        return valid
    from sglang.srt.mem_cache.flashnext_scheme_c import _unpack_gap8_torch
    flags = []
    for row in range(stream.shape[0]):
        try:
            _unpack_gap8_torch(stream[row:row+1], lengths[row:row+1], sparse=sparse, width=width)
            flags.append(True)
        except ValueError:
            flags.append(False)
    return torch.tensor(flags, dtype=torch.bool)


def classify(stamps, locations, positions):
    """0 ok, 1 never written (zero stamp), 2 another token's latent, 3 overwritten (no magic)."""
    magic = stamps[:, 0] == MAGIC
    zero = (stamps == 0).all(-1)
    same = (stamps[:, 2] == locations.reshape(-1).to(torch.int32))
    if positions is not None:
        same = same & (stamps[:, 1] == positions.reshape(-1).to(torch.int32))
    cls = torch.full_like(stamps[:, 0], 3)
    cls = torch.where(zero, torch.ones_like(cls), cls)
    cls = torch.where(magic & ~same, torch.full_like(cls, 2), cls)
    cls = torch.where(magic & same, torch.zeros_like(cls), cls)
    return cls


def check_load(pool, locations, positions, local_payload, stream, lengths, token_ids, *, sparse=512, width=10240):
    stamps = read_stamp(local_payload)
    cls = classify(stamps, locations, positions)
    valid = gap8_valid(stream, lengths, sparse, width)
    bad = (cls != 0) | ~valid
    if not bool(bad.any()):
        return
    rows = bad.nonzero().flatten()[:16].cpu().tolist()
    loc = locations.reshape(-1).long()
    pages = (loc // pool.page_size).cpu()
    gpu_units = pool.physical_page_map[pages.to(pool.physical_page_map.device)].cpu()
    detail = []
    for r in rows:
        page = int(pages[r])
        cpu = pool.arena.shared.get(page)
        detail.append(dict(row=r, position=None if positions is None else int(positions.reshape(-1)[r]),
                           loc=int(loc[r]), page=page, gpu_latent_units=gpu_units[r, 7:9].tolist(),
                           cpu_latent_units=None if cpu is None else list(cpu)[7:9],
                           pending=page in pool.arena.pending_shared, stamp=stamps[r].cpu().tolist(),
                           cls=int(cls[r]), gap8_valid=bool(valid[r]), length=int(lengths.reshape(-1)[r]),
                           token_id=int(token_ids.reshape(-1)[r])))
    by_cls = torch.bincount(cls.long(), minlength=4).cpu().tolist()
    emit("load_bad", rank=pool.tp_rank, rows=int(loc.numel()), bad=int(bad.sum()),
         by_class=dict(ok=by_cls[0], never_written=by_cls[1], foreign_latent=by_cls[2], overwritten=by_cls[3]),
         invalid_gap8=int((~valid).sum()), first=detail)
