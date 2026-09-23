"""Opt-in actual P/D KV digest before transfer and before first D forward.

Enabled only for transport guards. Payloads stay on the compute node; the JSON
contains hashes/shapes/bytes, not weights or prompt data. No effect when unset.
"""
import hashlib
import json
import os
from pathlib import Path


def audit_received_kv(req, request_pool, pool, role):
    out = os.getenv("SGLANG_FLASHNEXT_PD_KV_PROOF_DIR")
    if not out or req.bootstrap_room is None:
        return
    from sglang.srt.disaggregation.utils import _is_fake_transfer
    if _is_fake_transfer(req):
        return
    import torch
    from sglang.srt.distributed import get_tp_group
    n = len(req.origin_input_ids)
    rank = get_tp_group().rank_in_group
    items = {}
    def record(name, tensor):
        data = tensor.contiguous().view(torch.uint8).cpu().numpy()
        items[name] = dict(shape=list(tensor.shape), bytes=data.nbytes,
                           sha256=hashlib.sha256(data).hexdigest())
    shared = request_pool.req_to_token[req.kv.req_pool_idx, :n].long()
    for layer in range(3, 48, 4):
        part = pool
        loc = shared
        if getattr(pool, "shared_arena", False):
            if layer >= 31:
                part = pool.deep
                loc = pool.deep_req_to_token[req.kv.req_pool_idx, :n].long()
            loc = part.translate_locations(layer, loc)
        record(f"layer{layer}.k", part.get_key_buffer(layer)[loc])
        record(f"layer{layer}.v", part.get_value_buffer(layer)[loc])
        # Full groups only; raw pending group is transferred separately.
        compact = loc[:(n // 4) * 4:4] // 4
        record(f"layer{layer}.index", part.get_qsa_compressed_k_buffer(layer)[compact])
        local = part._transfer_full_attention_id(layer)
        ring = int(req.kv.req_pool_idx) * 4
        record(f"layer{layer}.pending", part.qsa_key_state_buffer_pool[local][ring:ring+4])
    record("rope_pending", pool.qsa_rope_position_buffer[ring:ring+4])
    target = Path(out) / f"{req.bootstrap_room}-{role}-rank{rank}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(room=req.bootstrap_room, role=role, rank=rank,
                                    tokens=n, entries=items), sort_keys=True))
    temp.replace(target)
