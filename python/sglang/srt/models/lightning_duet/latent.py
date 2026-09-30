"""Scheme C residual code with actual NVFP4/BF16/gap8 storage.

Arithmetic/rounding contract: twinstar.duet.latent{,fmt} at dd9c7bdbd955.
This module has no SGLang/CUDA import, so it is also exercised on CPU.
"""

from dataclasses import dataclass

import torch


LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


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


class ResidualCode:
    def __init__(self, encoder, decoder, mean, spikes, id_side=True):
        self.encoder = encoder.float().squeeze(0)
        self.decoder = decoder.float().squeeze(0)
        self.mean = mean.float().squeeze(0)
        self.spikes = spikes
        self.id_side = id_side

    def encode(self, residual, embeddings, token_ids):
        centered = residual.float() - (embeddings.float() if self.id_side else 0) - self.mean
        # einsum shape/order deliberately mirrors the unified reference.
        z = torch.einsum("...gd,grd->...gr", centered[:, None], self.encoder[None])[:, 0]
        codes, scales, global_scale = pack_nvfp4(z)
        zq = unpack_nvfp4(codes, scales, global_scale)
        reconstructed = torch.einsum("...gr,gdr->...gd", zq[:, None], self.decoder[None])[:, 0]
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
        reconstructed = torch.einsum("...gr,gdr->...gd", z[:, None], self.decoder[None])[:, 0]
        offsets = record.offsets.tolist()
        stream = record.gaps.tolist()
        indices = [unpack_gap8(stream[a:b], self.spikes) for a, b in zip(offsets, offsets[1:])]
        index = torch.tensor(indices, device=z.device, dtype=torch.int64)
        reconstructed = reconstructed + torch.zeros_like(reconstructed).scatter(-1, index, record.values.float())
        return (self.mean + reconstructed + (embeddings.float() if self.id_side else 0)).to(record.residual_dtype)
