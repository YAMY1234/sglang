"""NaN watch capture (SGLANG_NAN_WATCH_CAPTURE_DIR): a replay bundle for a row where a stage produced a NaN.

Per extend forward, rows with a long prefix and a short extend (the shape of the origins seen so far) are copied to
pinned host memory on a side stream: the stage's received proxy rows, and at the stage's first decoder layer its input,
KDA inputs (qkv, gate, beta), the KDA layer's SSM and conv state before it reads them, its kernel output, output norm,
attention output and layer output. The NaN watch drops the copies with its record, or saves them when that row is a
"produced in this stage" hit (up to _MAX_DUMPS per process). The side-stream wait adds no host sync; a dump waits on it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

# The two origins seen (nw2): prefix 51k / 139k, extend 272 / 1038 tokens. Arbitrary bounds around them.
_MIN_PREFIX, _MAX_EXTEND, _MAX_DUMPS = 32768, 2048, 3
_stream: Optional[torch.cuda.Stream] = None
_dumps = [0]


def enabled() -> bool:
    return bool(envs.SGLANG_NAN_WATCH_CAPTURE_DIR.get())


class Capture:
    def __init__(self, rows: List[Tuple[int, int, int]]):
        self.rows = rows  # (batch row, first token, extend tokens)
        self.layer: Optional[int] = None
        self.state_done = self.params_done = False
        self.tensors: Dict[str, torch.Tensor] = {}
        self.event = None


def select(prefix_lens, extend_lens) -> Optional[Capture]:
    rows, start = [], 0
    for i, (prefix, extend) in enumerate(zip(prefix_lens, extend_lens)):
        if prefix >= _MIN_PREFIX and extend <= _MAX_EXTEND:
            rows.append((i, start, extend))
        start += extend
    return Capture(rows) if rows else None


def _copy(cap: Capture, name: str, src: torch.Tensor) -> None:
    global _stream
    if not src.is_cuda:
        cap.tensors[name] = src.detach().clone()
        return
    if _stream is None:
        _stream = torch.cuda.Stream(device=src.device)
    _stream.wait_stream(torch.cuda.current_stream(src.device))
    with torch.cuda.stream(_stream):
        dst = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
        dst.copy_(src, non_blocking=True)
    src.record_stream(_stream)
    cap.tensors[name] = dst


def rows_of(cap: Capture, name: str, t: torch.Tensor) -> None:
    """Copy each selected row's tokens of a token-major tensor."""
    for i, start, n in cap.rows:
        _copy(cap, f"r{i}.{name}", t[start : start + n])


def point(cap: Capture, layer_idx: int, name: str, t, num_tokens: int) -> None:
    """At the first decoder layer seen in this forward only."""
    if cap.layer is None:
        cap.layer = layer_idx
    if (
        layer_idx == cap.layer
        and isinstance(t, torch.Tensor)
        and t.shape[0] >= num_tokens
    ):
        rows_of(cap, name, t)


def state(cap: Capture, cache_indices: torch.Tensor, ssm, conv) -> None:
    """The first KDA layer's SSM and conv state of each selected row, before the layer reads them."""
    if cap.state_done:
        return
    cap.state_done = True
    for i, _, _ in cap.rows:
        slot = cache_indices[i : i + 1].long()
        _copy(cap, f"r{i}.ssm", ssm.index_select(0, slot))
        _copy(cap, f"r{i}.conv", conv.index_select(0, slot))


def params(cap: Capture, layer_idx: int, tensors) -> None:
    """The captured layer's parameters (None entries skipped), once per forward."""
    if cap.params_done or layer_idx != cap.layer:
        return
    cap.params_done = True
    for name, t in tensors.items():
        if isinstance(t, torch.Tensor):
            _copy(cap, f"param.{name}", t)


def finish(cap: Capture) -> None:
    if _stream is not None:
        cap.event = _stream.record_event()


def dump(cap: Capture, row: int, meta: dict) -> Optional[str]:
    """Save one row's copies with its metadata; None when the row was not selected or the budget is spent."""
    if _dumps[0] >= _MAX_DUMPS or row not in {i for i, _, _ in cap.rows}:
        return None
    _dumps[0] += 1
    if cap.event is not None:
        cap.event.synchronize()  # the side stream's copies; only on a dump
    prefix = f"r{row}."
    tensors = {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in cap.tensors.items()
        if k.startswith((prefix, "param."))
    }
    out_dir = envs.SGLANG_NAN_WATCH_CAPTURE_DIR.get()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir,
        f"pp{meta['stage']}-{meta['rid'][:8]}-p{meta['prefix']}-e{meta['extend']}.pt",
    )
    torch.save({"meta": meta, "tensors": tensors}, path)
    return path
