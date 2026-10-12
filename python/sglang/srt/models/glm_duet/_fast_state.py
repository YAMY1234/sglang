"""GLM R32 periodic-cut adapter for Kimi's asynchronous batched factorizer.

Keep full-model probe draws before slicing TP heads. Matrices remain independent
across layer/slot/head; only workspace size limits the number of rows per call.
"""

import importlib

import torch

from . import _state_factor

# Also support the lightweight, top-level glm_duet import used by CPU tests.
_prefix = __package__.rpartition(".")[0]
fast_state = importlib.import_module(
    (_prefix + "." if _prefix else "") + "kimi_linear_duet.fast_state"
)


def probes(states, rank, omega_heads):
    heads, keys, values = states.shape[-3:]
    offset, total = (0, heads) if omega_heads is None else omega_heads
    if offset < 0 or offset + heads > total:
        raise ValueError("invalid GLM TP head slice")
    cold, extra = fast_state.probes(total, values, keys, rank, states.device)
    return cold[:, offset : offset + heads], extra[:, offset : offset + heads]


def _mask(values, device):
    mask = torch.tensor(values, dtype=torch.bool)
    if device.type == "cuda":
        return mask.pin_memory().to(device, non_blocking=True)
    return mask.to(device)


def project_state_batched(states, direction, rank, prevs, omega, *, explicit=True):
    """Return dense states and independently owned warm bases, in input order.

    Mixed cold/warm rows use one device mask instead of Python-list CUDA indexing.
    No tensor value is read on the host. ROW_CHUNK retains GLM's workspace bound;
    34 resident layers × one due slot is one dispatch of 544 head matrices.
    """
    out = torch.empty_like(states)
    bases = []
    for start in range(0, states.shape[0], _state_factor.ROW_CHUNK):
        stop = min(start + _state_factor.ROW_CHUNK, states.shape[0])
        sub = states[start:stop]
        sub_direction = direction[start:stop] if direction.dim() == 3 else direction
        if sub_direction.dim() == 2:
            sub_direction = sub_direction[None].expand(stop - start, -1, -1)
        zero = sub.new_zeros(1, sub.shape[1], sub.shape[-1], rank)
        previous = prevs[start:stop]
        valid = [
            p is not None and p.shape == zero.shape and p.device == sub.device
            for p in previous
        ]
        packed = torch.cat([p if use else zero for p, use in zip(previous, valid)])
        projected, basis = fast_state.project(
            sub,
            sub_direction,
            rank,
            packed,
            _mask(valid, sub.device),
            omega,
            explicit=explicit,
        )
        out[start:stop] = projected
        # A surviving slot must not retain a whole historical cut workspace.
        bases.extend(basis[i : i + 1].clone() for i in range(stop - start))
    return out, bases
