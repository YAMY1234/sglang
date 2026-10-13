"""GLM-only SEED-v3.5 batch math; shared performance-stack state_factor is untouched."""
import os
import torch
from ._common import load as _load
_reference = _load("state_factor")
OVERSAMPLE = _reference.OVERSAMPLE
OMEGA_SEED = _reference.OMEGA_SEED
project_state = _reference.project_state
factorize_left = _reference.factorize_left

def _batched_orthonormalize(y):
    """CholeskyQR2 in fp64 (state.py L39-50); returns y's dtype."""
    yd = y.double()
    for _ in range(2):
        gram = yd.transpose(-1, -2) @ yd
        jitter = (
            1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30
        )
        gram = gram + jitter * torch.eye(
            gram.shape[-1], device=y.device, dtype=gram.dtype
        )
        lower = torch.linalg.cholesky_ex(gram).L          # same factor as cholesky; no error-code check, so no host sync
        yd = torch.linalg.solve_triangular(
            lower, yd.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
    return yd.to(y.dtype)


ROW_CHUNK = 256

class ProjectionWorkspace:
    """Per-policy probes and pinned transfers, used only by the P1 opt-in.

    Draw on the original device with the full head count BEFORE slicing. Cold
    and warm draws have different shapes and must have separate cache entries.
    Pending pinned allocations stay owned until their H2D event completes.
    """

    def __init__(self):
        self.probes = {}
        self.transfers = []

    def index(self, values, device):
        self.transfers = [(event, host) for event, host in self.transfers if not event.query()]
        host = torch.tensor(values, dtype=torch.long)
        if device.type != "cuda":
            return host.to(device)
        host = host.pin_memory()
        result = host.to(device, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        self.transfers.append((event, host))
        return result

    def probe(self, device, total, n, columns):
        key = (device, total, n, columns, OMEGA_SEED)
        if key not in self.probes:
            generator = torch.Generator(device=device).manual_seed(OMEGA_SEED)
            self.probes[key] = torch.randn(
                1, total, n, columns, generator=generator, device=device, dtype=torch.float32
            )
        return self.probes[key]


def project_state_batched(states, direction, rank, prevs, *, explicit=True, side="right", omega_heads=None, workspace=None):
    """project_state for several slots at once: states (B, H, P, N), direction (H, D) shared by the rows or (B, H, D) one
    per row (rows of different layers; D = N for side="right", P for side="left"), prevs a list of B warm bases (each
    (1, H, N, r)) or None.  Numerically the per-slot reference convention: the random directions Omega are drawn ONCE with
    the B=1 shape and seed (OMEGA_SEED) and shared by every slot, so each slot sees exactly the numbers it would see alone
    (Omega per request, B=1).  Slots with and without a warm basis form two groups.  Returns (projected
    (B, H, P, N) fp32, list of B next warm bases (1, H, N, r)).  omega_heads=(offset, total): the rows hold heads offset ..
    offset + H - 1 of a model with `total` heads (a tensor-parallel rank's head slice); Omega is drawn with the full (1, total, N, m)
    shape and sliced, so every head sees the numbers of the unsharded B=1 reference."""
    if side not in ("right", "left"):
        raise ValueError(f"unknown DUET sink side: {side!r}")
    sf = states.float()
    out = torch.empty_like(sf)
    bases = [None] * sf.shape[0]
    groups = {"fresh": [i for i, p in enumerate(prevs) if p is None], "warm": [i for i, p in enumerate(prevs) if p is not None]}
    for name, group in groups.items():
        indices = workspace.index(group, sf.device) if workspace is not None else None
        # rows are independent; chunking bounds the workspace of the batched fp64 eigh (128 prompts x 20 layers at once
        # asked cusolver for 25 GiB)
        for start in range(0, len(group), ROW_CHUNK):
            idx = group[start:start + ROW_CHUNK]
            device_idx = idx if indices is None else indices[start:start + ROW_CHUNK]
            sub = sf[device_idx]
            sub_direction = direction[device_idx] if direction.dim() == 3 else direction
            prev = torch.cat([prevs[i].float() for i in idx], 0) if name == "warm" else None
            proj, basis = _project_shared_omega(sub, sub_direction, rank, prev, explicit, side, omega_heads, workspace)
            out[device_idx] = proj
            if basis is not None:
                # a clone, not a view: the policy keeps each slot's basis until the slot is reused (SGLang clears a slot only
                # when a new request takes it), and a view would keep the whole chunk's (rows, H, N, r) tensor alive per
                # finished slot -- on the v3-c LiveCodeBench midterm (one repetition per node, no queue after the first burst)
                # 2 of 4 servers per node ran out of memory in this eigh after ~35 min of decoding (2026-10-06)
                for j, i in enumerate(idx):
                    bases[i] = basis[j:j + 1].clone() if workspace is None else basis[j:j + 1]
    return out, bases


def _project_shared_omega(sf, direction, rank, prev, explicit, side, omega_heads=None, workspace=None):
    """project_state with Omega expanded from one B=1 draw."""
    if rank <= 0:
        return sf, None
    direction = direction.to(sf.device, sf.dtype)
    if direction.dim() == 2:
        direction = direction[None].expand(sf.shape[0], -1, -1)                 # (B, H, D)
    if explicit:
        n2 = (direction * direction).sum(-1).clamp_min(1e-12)                   # (B, H)
        if side == "left":
            coeff = torch.einsum("bhpn,bhp->bhn", sf, direction) / n2[:, :, None]
            sink = direction[:, :, :, None] * coeff[:, :, None, :]
        else:
            coeff = torch.einsum("bhkv,bhv->bhk", sf, direction) / n2[:, :, None]
            sink = coeff[..., :, None] * direction[:, :, None, :]
        content = sf - sink
    else:
        sink, content = 0.0, sf
    if rank >= min(content.shape[-2:]):
        return (sink + content), None
    batch, heads, p, n = content.shape
    # SGLANG_DUET_STATE_OVERSAMPLE: diagnostic override of the reference's OVERSAMPLE (8) -- the number of fresh random
    # sketch columns beside the warm basis; W new rank-1 writes per cut are sketched by these columns plus one power step
    oversample = int(os.environ.get("SGLANG_DUET_STATE_OVERSAMPLE", OVERSAMPLE))
    m = min(rank + oversample, p, n)
    use_prev = prev is not None and prev.shape == (batch, heads, n, rank) and prev.device == sf.device
    offset, total = (0, heads) if omega_heads is None else omega_heads
    columns = m - (rank if use_prev else 0)
    if workspace is None:
        generator = torch.Generator(device=sf.device).manual_seed(OMEGA_SEED)
        omega = torch.randn(1, total, n, columns, generator=generator, device=sf.device,
                            dtype=torch.float32)
    else:
        omega = workspace.probe(sf.device, total, n, columns)
    omega = omega[:, offset:offset + heads].expand(batch, -1, -1, -1)
    if use_prev:
        omega = torch.cat([prev.float(), omega], -1)
    y = content @ omega
    y = content @ _batched_orthonormalize(content.transpose(-1, -2) @ _batched_orthonormalize(y))
    q = _batched_orthonormalize(y)
    bm = q.transpose(-1, -2) @ content
    gram = (bm @ bm.transpose(-1, -2)).double()
    jitter = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30
    gram = gram + jitter * torch.eye(gram.shape[-1], device=sf.device, dtype=gram.dtype)
    # cusolver's batched fp64 eigh: 17 ms for 5120 24x24 matrices vs 39 ms for the padded Jacobi kernel (benchmark/duet_eval/bench_state_cut.py)
    vectors = torch.linalg.eigh(gram)[1].float()
    u = q @ vectors[..., -rank:]
    uts = u.transpose(-1, -2) @ content
    return sink + u @ uts, _batched_orthonormalize(uts.transpose(-1, -2))
