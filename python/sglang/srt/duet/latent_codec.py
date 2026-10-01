"""The residual code of DUET/SEED (origin/minma/0913 twinstar/duet/latent.py, latentfmt.py), model independent.

Two entry points, moved verbatim from the lines that wrote them (no arithmetic changed):
  * `quantize` + `ResidualCode` (Kimi line, twinstar_sgl/kimi_duet_math.py): fake-quantised forward on fp32
    -- c = (h - h0) - mu; z = Q_z(E c); rec = D z; res = c - rec; top-m |res| exact, Q_value; out = mu + rec + spikes + h0;
  * `pack_nvfp4` / `unpack_nvfp4` / `pack_gap8` / `unpack_gap8` + `PackedResidualCode` / `LatentRecord` (Lightning line,
    models/lightning_duet/latent.py): the same code with the real scheme-C storage (NVFP4 codes + e4m3 block scales
    + fp32 row scale, bf16 exact values, gap8 indices).
Both are bitwise against the pinned reference on CPU (tests/test_kimi_duet_math.py; test_lightning_duet.py).
Flash-Next's TP-split Triton packing (mem_cache/flashnext_scheme_c.py) stays in its adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
NVFP4_BLOCK = 16


# ----------------------------------------------------------------------------- fake quantisation (Kimi line)
def quantize(x, fmt):
    if fmt == "fp32":
        return x
    xf = x.float()
    if fmt == "bf16":
        return xf.to(torch.bfloat16).float()
    if fmt == "fp8":
        scale = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
        return (xf / scale).to(torch.float8_e4m3fn).float() * scale
    if fmt != "nvfp4" or x.shape[-1] % 16:
        raise ValueError(f"unsupported code format/shape: {fmt}, {x.shape}")
    rows = xf.reshape(-1, x.shape[-1])
    scale = rows.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (6.0 * 448.0)
    blocks = (rows / scale).reshape(rows.shape[0], -1, 16)
    bs = (blocks.abs().amax(-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn).float().clamp_min(2.0 ** -9)
    y = (blocks / bs).clamp(-6.0, 6.0)
    edges = y.new_tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    levels = y.new_tensor((0., .5, 1., 1.5, 2., 3., 4., 6.))
    idx = torch.bucketize(y.abs(), edges, right=True)
    idx = torch.where(torch.isin(y.abs(), edges) & (idx % 2 == 1), idx - 1, idx)
    q = levels[idx] * torch.sign(y) * bs
    return (q.reshape_as(rows) * scale).reshape_as(x)


class _code_matmul_precision:
    """TF32 for the E / D matmuls only (docs/162 §3.2 --duet-code-precision; production profile).  The flag is
    scoped to the code: nothing else in the forward changes precision, and fp32 (reference) is the default."""

    def __init__(self, tf32):
        self.tf32 = bool(tf32)

    def __enter__(self):
        self._prev = torch.backends.cuda.matmul.allow_tf32
        if self.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self._prev
        return False


class ResidualCode(nn.Module):
    """E (1, r, d), D (1, d, r), mu (1, d) fp32; forward(h, base) returns the decoded residual in h's dtype."""

    def __init__(self, dim, spec, *, tf32=False):
        super().__init__()
        self.spec = spec
        self.tf32 = bool(tf32)
        if spec["latent_rank"] == 0:
            self.code = None
            return
        self.code = nn.Module()
        self.code.E = nn.Parameter(torch.empty(1, spec["latent_rank"], dim, dtype=torch.float32), requires_grad=False)
        self.code.D = nn.Parameter(torch.empty(1, dim, spec["latent_rank"], dtype=torch.float32), requires_grad=False)
        self.code.register_buffer("mu", torch.empty(1, dim, dtype=torch.float32))

    def forward(self, h, base):
        if self.code is None:
            return h
        dt = h.dtype
        x = h[..., None, :].float()
        b = base[..., None, :].float() if self.spec["latent_id_side"] else None
        if b is not None:
            x = x - b
        c = x - self.code.mu
        with _code_matmul_precision(self.tf32):
            z = torch.einsum("...gd,grd->...gr", c, self.code.E)
            z = quantize(z, self.spec["latent_z_format"])
            rec = torch.einsum("...gr,gdr->...gd", z, self.code.D)
        residual = c - rec
        if self.spec["latent_spikes"]:
            idx = residual.abs().topk(self.spec["latent_spikes"], dim=-1).indices
            vals = quantize(residual.gather(-1, idx), self.spec["latent_value_format"])
            rec = rec + torch.zeros_like(residual).scatter(-1, idx, vals)
        out = self.code.mu + rec
        if b is not None:
            out = out + b
        return out.to(dt)[..., 0, :]


# ----------------------------------------------------------------------------- real storage (Lightning line)
def pack_nvfp4(z):
    if z.shape[-1] % 16:
        raise ValueError("NVFP4 code rank must be a multiple of 16")
    z = z.float()
    rows = z.reshape(-1, z.shape[-1])
    global_scale = rows.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (6.0 * 448.0)
    blocks = (rows / global_scale).reshape(rows.shape[0], -1, 16)
    scales = (blocks.abs().amax(-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn)
    divisors = scales.float().clamp_min(2.0 ** -9)
    values = (blocks / divisors).clamp(-6.0, 6.0)
    edges = torch.tensor(EDGES, device=z.device)
    codes = torch.bucketize(values.abs(), edges, right=True)
    ties = torch.isin(values.abs(), edges) & (codes % 2 == 1)
    codes = torch.where(ties, codes - 1, codes).to(torch.uint8)
    codes = (codes | ((values < 0).to(torch.uint8) << 3)).reshape(rows.shape)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scales.squeeze(-1), global_scale.squeeze(-1)


def unpack_nvfp4(packed, scales, global_scale):
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    levels = torch.tensor(LEVELS, device=packed.device)
    values = levels[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    return (values.reshape(packed.shape[0], -1, 16)
            * scales.float().clamp_min(2.0 ** -9)[..., None]).flatten(-2) * global_scale[:, None]


def pack_gap8(indices):
    """One sorted unique row -> byte stream, first gap measured from -1."""
    values = indices.tolist()
    output = []
    previous = -1
    for index in values:
        gap = index - previous - 1
        if gap < 0 or gap > 65535:
            raise ValueError("gap8 requires sorted unique uint16 coordinates")
        output.extend([gap] if gap < 255 else [255, gap & 255, gap >> 8])
        previous = index
    return output


def unpack_gap8(stream, count):
    output, previous, cursor = [], -1, 0
    for _ in range(count):
        if cursor >= len(stream):
            raise ValueError("truncated gap8 stream")
        gap = stream[cursor]
        cursor += 1
        if gap == 255:
            if cursor + 2 > len(stream):
                raise ValueError("truncated gap8 escape")
            gap = stream[cursor] | (stream[cursor + 1] << 8)
            cursor += 2
        previous += gap + 1
        output.append(previous)
    if cursor != len(stream):
        raise ValueError("trailing gap8 bytes")
    return output


@dataclass
class LatentRecord:
    codes: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    values: torch.Tensor
    gaps: torch.Tensor
    offsets: torch.Tensor
    token_ids: torch.Tensor
    residual_dtype: torch.dtype

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.codes, self.scales, self.global_scale, self.values,
            self.gaps, self.offsets, self.token_ids,
        ))


class PackedResidualCode:
    """The same code with real scheme-C storage: encode -> LatentRecord, decode(record, embeddings) -> residual."""

    def __init__(self, encoder, decoder, mean, spikes, id_side=True, *, tf32=False):
        self.encoder = encoder.float().squeeze(0)
        self.decoder = decoder.float().squeeze(0)
        self.mean = mean.float().squeeze(0)
        self.spikes = spikes
        self.id_side = id_side
        self.tf32 = bool(tf32)  # production profile: TF32 for the E / D matmuls only

    def _project(self, residual, embeddings):
        centered = residual.float() - (embeddings.float() if self.id_side else 0) - self.mean
        # einsum shape/order deliberately mirrors the unified reference.
        with _code_matmul_precision(self.tf32):
            z = torch.einsum("...gd,grd->...gr", centered[:, None], self.encoder[None])[:, 0]
            codes, scales, global_scale = pack_nvfp4(z)
            zq = unpack_nvfp4(codes, scales, global_scale)
            reconstructed = torch.einsum("...gr,gdr->...gd", zq[:, None], self.decoder[None])[:, 0]
        return centered, reconstructed, codes, scales, global_scale

    def reconstruct(self, residual, embeddings):
        """Quantized roundtrip for consumers that do not persist a latent record.

        Keep the same E/D and NVFP4 arithmetic as encode/decode. Coordinates
        from topk are unique, so their order does not change the scatter. No
        gap8 serialization or tensor-to-host transfer is needed here.
        """
        centered, reconstructed, _, _, _ = self._project(residual, embeddings)
        error = centered - reconstructed
        indices = error.abs().topk(self.spikes, dim=-1).indices
        values = error.gather(-1, indices).to(torch.bfloat16)
        reconstructed = reconstructed + torch.zeros_like(reconstructed).scatter(
            -1, indices, values.float()
        )
        return (
            self.mean + reconstructed + (embeddings.float() if self.id_side else 0)
        ).to(residual.dtype)

    def encode(self, residual, embeddings, token_ids):
        centered, reconstructed, codes, scales, global_scale = self._project(
            residual, embeddings
        )
        error = centered - reconstructed
        indices = error.abs().topk(self.spikes, dim=-1).indices
        indices = indices.sort(-1).values
        values = error.gather(-1, indices).to(torch.bfloat16)
        stream, offsets = [], [0]
        for row in indices:
            stream.extend(pack_gap8(row))
            offsets.append(len(stream))
        return LatentRecord(
            codes, scales, global_scale, values,
            torch.tensor(stream, device=residual.device, dtype=torch.uint8),
            torch.tensor(offsets, device=residual.device, dtype=torch.int32),
            token_ids.to(torch.int32).clone(), residual.dtype,
        )

    def decode(self, record, embeddings):
        z = unpack_nvfp4(record.codes, record.scales, record.global_scale)
        with _code_matmul_precision(self.tf32):
            reconstructed = torch.einsum("...gr,gdr->...gd", z[:, None], self.decoder[None])[:, 0]
        offsets = record.offsets.tolist()
        stream = record.gaps.tolist()
        indices = [unpack_gap8(stream[a:b], self.spikes) for a, b in zip(offsets, offsets[1:])]
        index = torch.tensor(indices, device=z.device, dtype=torch.int64)
        reconstructed = reconstructed + torch.zeros_like(reconstructed).scatter(-1, index, record.values.float())
        return (self.mean + reconstructed + (embeddings.float() if self.id_side else 0)).to(record.residual_dtype)
