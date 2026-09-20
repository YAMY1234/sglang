"""Compact prefix-code arithmetic, independent of the serving cache allocator.

This module defines the format and a correctness oracle. It is not a serving
attention kernel. Spike payloads are original input coordinates; readers must
subtract the decoded coordinate before adding a sparse correction.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QSACodeLayout:
    key_rank: int = 96
    value_rank: int = 64
    key_sparse: int = 16
    value_sparse: int = 32
    rotary_dim: int = 64
    head_dim: int = 256

    def __post_init__(self):
        if not 0 <= self.rotary_dim < self.head_dim <= 256:
            raise ValueError("QSA code coordinates must fit uint8")
        for width, rank, sparse in (
            (self.head_dim - self.rotary_dim, self.key_rank, self.key_sparse),
            (self.head_dim, self.value_rank, self.value_sparse),
        ):
            if not 0 < rank <= width or not 0 <= sparse <= width:
                raise ValueError("Invalid QSA rank or sparse coordinate count")

    def payload_bytes_per_head(self, code_dtype=torch.bfloat16):
        # Original K/V and RoPE coordinates are bf16. Indices are uint8.
        return (
            2 * self.rotary_dim
            + code_dtype.itemsize * (self.key_rank + self.value_rank)
            + 3 * (self.key_sparse + self.value_sparse)
        )


@dataclass(frozen=True)
class CodeWeights:
    encoder: torch.Tensor  # [head, rank, dim], fp32
    decoder: torch.Tensor  # [head, dim, rank], fp32
    mean: torch.Tensor  # [head, dim], fp32

    def __post_init__(self):
        if self.encoder.ndim != 3:
            raise ValueError("encoder must be [head, rank, dim]")
        h, r, d = self.encoder.shape
        if self.decoder.shape != (h, d, r) or self.mean.shape != (h, d):
            raise ValueError("encoder, decoder and mean shapes disagree")
        if any(x.dtype != torch.float32 for x in self.tensors()):
            raise ValueError("QSA code weights must stay fp32")
        if any(x.device != self.encoder.device for x in self.tensors()):
            raise ValueError("QSA code weights must share a device")

    def tensors(self):
        return self.encoder, self.decoder, self.mean


@dataclass(frozen=True)
class SparseCode:
    latent: torch.Tensor  # [token, head, rank]
    indices: torch.Tensor  # [token, head, sparse], uint8
    originals: torch.Tensor  # [token, head, sparse], input dtype

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.latent, self.indices, self.originals))


@dataclass(frozen=True)
class PrefixCode:
    rotary: torch.Tensor  # [token, head, rotary_dim], input dtype
    key: SparseCode
    value: SparseCode

    @property
    def nbytes(self):
        return self.rotary.nbytes + self.key.nbytes + self.value.nbytes


def encode_coordinates(x, weights, sparse, *, code_dtype=torch.bfloat16):
    """Use Qwen4QSACode's fp32 selection, then store the code and exact spikes.

    Selection happens before code quantization. Keeping originals instead of a
    rounded residual lets a reader preserve each selected coordinate exactly.
    Encoder and decoder are independent learned matrices.
    """
    if x.ndim != 3 or x.shape[1:] != weights.mean.shape:
        raise ValueError("Input must be [token, head, dim]")
    if not 0 <= sparse <= x.shape[-1] <= 256:
        raise ValueError("Invalid sparse coordinate count or uint8 dimension")
    centered = x.float() - weights.mean
    latent = torch.einsum("thd,hrd->thr", centered, weights.encoder)
    reconstruction = torch.einsum("thr,hdr->thd", latent, weights.decoder)
    indices = (centered - reconstruction).abs().topk(sparse, dim=-1).indices
    return SparseCode(
        latent.to(code_dtype), indices.to(torch.uint8), x.gather(-1, indices)
    )


def encode_prefix(
    k, v, key_weights, value_weights, layout, *, code_dtype=torch.bfloat16
):
    if k.shape != v.shape or k.shape[-1] != layout.head_dim:
        raise ValueError("K/V shapes must agree with the code layout")
    if key_weights.encoder.shape[1:] != (
        layout.key_rank,
        layout.head_dim - layout.rotary_dim,
    ) or value_weights.encoder.shape[1:] != (layout.value_rank, layout.head_dim):
        raise ValueError("Code weight ranks do not match layout")
    return PrefixCode(
        # clone: the compact representation must not retain the full K storage.
        k[..., : layout.rotary_dim].clone(),
        encode_coordinates(
            k[..., layout.rotary_dim :],
            key_weights,
            layout.key_sparse,
            code_dtype=code_dtype,
        ),
        encode_coordinates(
            v, value_weights, layout.value_sparse, code_dtype=code_dtype
        ),
    )


def materialize_coordinates(code, weights):
    """Validation-only reconstruction; never call from absorbed serving reads."""
    rec = torch.einsum("thr,hdr->thd", code.latent.float(), weights.decoder)
    rec = rec + weights.mean
    return rec.scatter(-1, code.indices.long(), code.originals.float())


def sparse_corrections(code, weights):
    """Decode only the selected coordinates, without a full K/V reconstruction."""
    heads = torch.arange(weights.mean.shape[0], device=code.latent.device)[
        None, :, None
    ]
    index = code.indices.long()
    rows = weights.decoder[heads, index]
    decoded = (rows * code.latent.float().unsqueeze(-2)).sum(-1)
    return code.originals.float() - weights.mean[heads, index] - decoded


def absorbed_attention_reference(
    q,
    prefix,
    key_weights,
    value_weights,
    exact_k,
    exact_v,
    *,
    scale,
    code_mask=None,
    exact_mask=None,
):
    """Correctness oracle: queries share a prefix and an exact current-turn tail.

    q is [query, q_head, dim]. Masks are [query, token], true for visible keys.
    All contributions use a single softmax, including when either pool is empty.
    Float32 arithmetic makes the algebra testable independently of a fused GPU
    kernel's chosen accumulation and output precision.
    """
    if q.ndim != 3 or exact_k.shape != exact_v.shape:
        raise ValueError("Invalid query or exact K/V shapes")
    n, hq, d = q.shape
    hk = prefix.rotary.shape[1]
    if hq % hk or exact_k.shape[1:] != (hk, d):
        raise ValueError("Invalid GQA head layout")
    heads = torch.arange(hq, device=q.device) // (hq // hk)
    rot = prefix.rotary.shape[-1]
    qf = q.float()
    qn = qf[..., rot:]
    qcode = torch.einsum("qhd,hdr->qhr", qn, key_weights.decoder[heads])
    scores_code = torch.einsum(
        "qhd,thd->qht", qf[..., :rot], prefix.rotary[:, heads].float()
    )
    scores_code += torch.einsum(
        "qhr,thr->qht", qcode, prefix.key.latent[:, heads].float()
    )
    scores_code += (qn * key_weights.mean[heads]).sum(-1).unsqueeze(-1)
    ki = prefix.key.indices[:, heads].long()
    kc = sparse_corrections(prefix.key, key_weights)[:, heads]
    # Gather query coordinates, not decoded keys.
    sparse_q = (
        qn.unsqueeze(1)
        .expand(-1, ki.shape[0], -1, -1)
        .gather(-1, ki.unsqueeze(0).expand(n, -1, -1, -1))
    )
    scores_code += (sparse_q * kc.unsqueeze(0)).sum(-1).permute(0, 2, 1)
    scores_exact = torch.einsum("qhd,thd->qht", qf, exact_k[:, heads].float())
    if code_mask is not None:
        scores_code = scores_code.masked_fill(~code_mask[:, None, :], -torch.inf)
    if exact_mask is not None:
        scores_exact = scores_exact.masked_fill(~exact_mask[:, None, :], -torch.inf)
    scores = torch.cat((scores_code, scores_exact), dim=-1) * scale
    # A row with no selected tokens is an inert graph/padding row.
    empty = torch.isneginf(scores).all(-1, keepdim=True)
    probabilities = torch.softmax(scores.masked_fill(empty, 0), dim=-1).masked_fill(
        empty, 0
    )
    pc, pe = probabilities.split(
        (scores_code.shape[-1], scores_exact.shape[-1]), dim=-1
    )
    latent_sum = torch.einsum("qht,thr->qhr", pc, prefix.value.latent[:, heads].float())
    output = torch.einsum("qhr,hdr->qhd", latent_sum, value_weights.decoder[heads])
    output += pc.sum(-1, keepdim=True) * value_weights.mean[heads]
    output += torch.einsum("qht,thd->qhd", pe, exact_v[:, heads].float())
    vi = prefix.value.indices[:, heads].long()
    vc = sparse_corrections(prefix.value, value_weights)[:, heads]
    weighted = pc.permute(0, 2, 1).unsqueeze(-1) * vc.unsqueeze(0)
    # Sum sparse residuals into output coordinates, not per-token dense values.
    output.scatter_add_(
        -1,
        vi.permute(1, 0, 2).reshape(1, hq, -1).expand(n, -1, -1),
        weighted.permute(0, 2, 1, 3).reshape(n, hq, -1),
    )
    return output
