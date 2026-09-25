"""#287 diagnostic (SGLANG_P287_BREAK_HASH=<dir>): hash the real-row inputs and
outputs of the eager cores of a prefill forward (GDN / QSA attention, QSA
indexer, PLE) and the final hidden rows, so graph and eager runs can be
compared op by op. Only logprob-returning EXTEND forwards are recorded (the
probe requests; never capture or warmup). Token tensors are cut to the live
token count; the trailing partial 16-row block is also hashed on its own.
Synchronizes the device; never enabled in a timing engine."""
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
    """rows: the token-row count of the tensors as passed (the bucket under a
    graph); tensors with that leading size are token tensors, the rest are
    hashed whole."""
    if not enabled() or _state["n"] >= _LIMIT:
        return
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_extend() or mode.is_target_verify() or mode.is_draft_extend_v2():
        return
    if not getattr(forward_batch, "return_logprob", False):
        return
    real = getattr(forward_batch, "extend_num_tokens", None)
    real = int(rows) if real is None else min(int(rows), int(real))
    aligned = real // 16 * 16
    torch.cuda.synchronize()
    if _state["file"] is None:
        os.makedirs(os.environ["SGLANG_P287_BREAK_HASH"], exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        _state["file"] = open(os.path.join(os.environ["SGLANG_P287_BREAK_HASH"], f"rank{rank}.jsonl"), "a", buffering=1)
    row = dict(i=_state["n"], op=op, layer=int(layer_id), rows=real)
    for name, value in tensors.items():
        token = value is not None and value.dim() and value.shape[0] == rows
        if token:
            row[name] = _sha(value[:aligned])
            row[name + "@tail"] = _sha(value[aligned:real])
        else:
            row[name] = _sha(value)
    _state["file"].write(json.dumps(row) + "\n")
    _state["n"] += 1
