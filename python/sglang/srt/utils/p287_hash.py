"""#287 diagnostic (SGLANG_P287_BREAK_HASH=<dir>): hash the real-row inputs and
outputs of the eager cores of a prefill forward (GDN / QSA attention, QSA
indexer, PLE), so graph and eager runs can be compared op by op. Synchronizes
the device; never enabled in a timing engine."""
import hashlib
import json
import os

import torch

_state = {"n": 0, "file": None}
_LIMIT = 40000


def enabled() -> bool:
    return bool(os.environ.get("SGLANG_P287_BREAK_HASH"))


def _sha(t) -> str:
    if t is None:
        return None
    t = t.detach().contiguous()
    return hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def record(op: str, layer_id, forward_batch, rows: int, **tensors) -> None:
    if not enabled() or _state["n"] >= _LIMIT:
        return
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_extend() or mode.is_target_verify() or mode.is_draft_extend_v2():
        return
    torch.cuda.synchronize()
    if _state["file"] is None:
        os.makedirs(os.environ["SGLANG_P287_BREAK_HASH"], exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        _state["file"] = open(os.path.join(os.environ["SGLANG_P287_BREAK_HASH"], f"rank{rank}.jsonl"), "a", buffering=1)
    row = dict(i=_state["n"], op=op, layer=int(layer_id), rows=int(rows),
               **{k: _sha(v[:rows] if v is not None and v.dim() and v.shape[0] >= rows else v)
                  for k, v in tensors.items()})
    _state["file"].write(json.dumps(row) + "\n")
    _state["n"] += 1
