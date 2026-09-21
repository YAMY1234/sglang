"""Compact prefix-code arithmetic, independent of the serving cache allocator.

This module defines the format and a correctness oracle. It is not a serving
attention kernel. Spike payloads are original input coordinates; readers must
subtract the decoded coordinate before adding a sparse correction.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass(frozen=True)
class QSACodeLayout:
    key_rank: int = 96
    value_rank: int = 64
    key_sparse: int = 32
    value_sparse: int = 32
    rotary_dim: int = 64
    head_dim: int = 256
    stored_residuals: bool = False  # Experimental until model K1 passes.
    value_bitmap: bool = False  # Same 32 B, sorted residuals; experimental.

    def __post_init__(self):
        if self.value_bitmap and (
            not self.stored_residuals or self.head_dim != 256 or self.value_sparse != 32
        ):
            raise ValueError("V bitmap requires 256 dimensions and 32 stored residuals")
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
    reference: str = "qwen4"

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
        if self.reference not in ("qwen4", "qsav"):
            raise ValueError("Unknown QSA reference implementation")

    def tensors(self):
        return self.encoder, self.decoder, self.mean


@dataclass(frozen=True)
class SparseCode:
    latent: torch.Tensor  # [token, head, rank]
    indices: torch.Tensor  # [token, head, sparse], uint8
    originals: torch.Tensor  # [token, head, sparse], values (residuals if flagged)
    stored_residuals: bool = False
    coordinate_bitmap: bool = False

    def coordinate_indices(self):
        """Oracle-only unpacking; serving reads address bitmap words directly."""
        if not self.coordinate_bitmap:
            return self.indices.long()
        words = self.indices.contiguous().view(torch.int32).long() & 0xFFFFFFFF
        coordinates = torch.arange(256, device=words.device)
        selected = ((words[..., coordinates // 32] >> (coordinates % 32)) & 1).bool()
        ordered = (
            coordinates.expand(selected.shape)
            .masked_fill(~selected, 256)
            .sort(-1)
            .values
        )
        # Zero sentinel rows have no selected bits and zero residuals. Keep
        # their inert oracle scatter in bounds without inventing a selected bit.
        return ordered[..., : self.originals.shape[-1]].clamp_max(255)

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


def encode_coordinates(
    x,
    weights,
    sparse,
    *,
    code_dtype=torch.bfloat16,
    stored_residuals=False,
    coordinate_bitmap=False,
):
    """Use Qwen4QSACode's fp32 selection, then store the code and exact spikes.

    Selection happens before code quantization. Keeping originals instead of a
    rounded residual lets a reader preserve each selected coordinate exactly.
    Encoder and decoder are independent learned matrices.
    """
    if x.ndim != 3 or x.shape[1:] != weights.mean.shape:
        raise ValueError("Input must be [token, head, dim]")
    if not 0 <= sparse <= x.shape[-1] <= 256:
        raise ValueError("Invalid sparse coordinate count or uint8 dimension")
    # Invoke the copied reference's original [B,H,S,D] arithmetic. Only the
    # return boundary exposes components instead of throwing them away after
    # materialization. The service default uses x256's learned Qwen4QSACode;
    # the qsav basis path remains only for historical diagnostic reproduction.
    from sglang.srt.layers.attention.qsa.qsav_reference import apply as reference_apply
    from sglang.srt.layers.attention.qsa.qwen4_code_reference import Qwen4QSACode

    original_shape = x.permute(1, 0, 2).unsqueeze(0)
    if weights.reference == "qsav":
        latent, indices = reference_apply(
            0,
            original_shape,
            _model={"U": {0: weights.decoder}, "mean": {0: weights.mean}, "m": sparse},
            _return_components=True,
        )
    else:
        latent, indices = Qwen4QSACode._code(
            SimpleNamespace(sparse=sparse),
            original_shape,
            weights.encoder,
            weights.decoder,
            weights.mean,
            _return_components=True,
        )
    latent = latent[0].permute(1, 0, 2).contiguous()
    indices = indices[0].permute(1, 0, 2).contiguous()
    latent = latent.to(code_dtype)
    originals = x.gather(-1, indices)
    if stored_residuals:
        # The copied reference selects coordinates before quantizing z. Move
        # the existing read-time correction here, then measure its rounding.
        heads = torch.arange(weights.mean.shape[0], device=x.device)[None, :, None]
        decoded = (weights.decoder[heads, indices] * latent.float().unsqueeze(-2)).sum(
            -1
        )
        originals = (originals.float() - weights.mean[heads, indices] - decoded).to(
            x.dtype
        )
    if coordinate_bitmap:
        if not stored_residuals or x.shape[-1] != 256 or sparse != 32:
            raise ValueError("Bitmap encoding requires 32 residuals in 256 dimensions")
        indices, order = indices.sort(-1)
        originals = originals.gather(-1, order)
        words = torch.zeros(
            (*indices.shape[:-1], 8), dtype=torch.int64, device=x.device
        )
        words.scatter_add_(
            -1, indices // 32, torch.ones_like(indices) << (indices % 32)
        )
        indices = words.to(torch.int32).contiguous().view(torch.uint8)
    return SparseCode(
        latent, indices.to(torch.uint8), originals, stored_residuals, coordinate_bitmap
    )


def load_x256_weights(release, *, layer_ids, tp_rank, tp_size, device):
    """Read the published trained codes, never the spec's initialization path.

    Hash verification is deliberately outside the CUDA graph. Only the 72 code
    tensors are mapped; the 1.98 GB emitter/latent payload is not loaded to GPU.
    """
    import hashlib
    import json
    from pathlib import Path

    from safetensors import safe_open

    release = Path(release)
    spec = json.loads((release / "spec.json").read_text())
    manifest = json.loads((release / "manifest.json").read_text())
    expected_spec = {
        "qsa_k_rank": 96,
        "qsa_v_rank": 64,
        "qsa_sparse": 32,
        "generated_token_code_delay": 256,
    }
    if any(spec.get(k) != v for k, v in expected_spec.items()):
        raise ValueError("QSA service requires the published x256 96+32/64+32 spec")
    if manifest.get("name") != "duet-fn-x256" or manifest.get("delay") != 256:
        raise ValueError("Expected duet-fn-x256 release manifest")
    if tp_size not in (1, 2) or not 0 <= tp_rank < tp_size:
        raise ValueError("QSA x256 requires valid TP1/TP2 ranks")
    source = release / "duet_components.safetensors"
    with source.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    published_hash = "60bfc4f5f6d023e92e829a9417ef31b53d4f41e015917a49eeb0304a733b66d3"
    if actual_hash != published_hash or manifest.get("sha256") != published_hash:
        raise ValueError("x256 code weights do not match the published SHA256")
    if source.stat().st_size != manifest.get("file_bytes"):
        raise ValueError("x256 component file size mismatch")
    heads = 2 // tp_size
    lo, hi = tp_rank * heads, (tp_rank + 1) * heads
    weights = {}
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        code_names = [k for k in checkpoint.keys() if k.startswith("P.vlat.")]  # noqa: SIM118 -- safe_open is not a mapping
        if (
            len(code_names) != 72
            or manifest.get("totals", {}).get("kv_codes") != 1681920
        ):
            raise ValueError("Expected all 12 trained QSA layers in the release")
        for layer in layer_ids:
            pair = []
            for side, dim, rank in (("K", 192, 96), ("V", 256, 64)):
                tensors = []
                for field, shape in (
                    ("E", (2, rank, dim)),
                    ("D", (2, dim, rank)),
                    ("mean", (2, dim)),
                ):
                    name = f"P.vlat.{layer}.{field}_{side}"
                    tensor = checkpoint.get_tensor(name)
                    if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
                        raise ValueError(f"Invalid x256 code tensor: {name}")
                    entry = manifest["tensors"][name]
                    if (
                        tuple(entry["shape"]) != shape
                        or entry["bytes"] != tensor.nbytes
                    ):
                        raise ValueError(f"x256 tensor disagrees with manifest: {name}")
                    tensors.append(tensor[lo:hi].to(device=device).contiguous())
                pair.append(CodeWeights(*tensors))
            weights[layer] = tuple(pair)
    return weights


def load_stack5_weights(path, *, layer_ids, tp_rank, tp_size, device):
    """Load the fixed STACK5 basis; never substitute a recovery checkpoint."""
    basis = torch.load(path, map_location="cpu", weights_only=True)
    if int(basis["rot"]) != 64 or tp_size not in (1, 2):
        raise ValueError("STACK5 basis loader currently requires rotary64 and TP1/TP2")
    if not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid tensor-parallel rank")
    heads = 2 // tp_size
    lo, hi = tp_rank * heads, (tp_rank + 1) * heads
    weights = {}
    for layer in layer_ids:
        pair = []
        for basis_key, mean_key, dim, rank in (
            ("UK", "meanK", 192, 96),
            ("U", "mean", 256, 64),
        ):
            u, mean = basis[basis_key][layer], basis[mean_key][layer]
            if u.shape[:2] != (2, dim) or u.shape[2] < rank or mean.shape != (2, dim):
                raise ValueError(
                    f"Invalid STACK5 weights at layer {layer}: {basis_key}"
                )
            decoder = (
                u[lo:hi, :, :rank].to(device=device, dtype=torch.float32).contiguous()
            )
            encoder = decoder.transpose(1, 2).contiguous()
            pair.append(
                CodeWeights(
                    encoder,
                    decoder,
                    mean[lo:hi].to(device=device, dtype=torch.float32).contiguous(),
                    reference="qsav",
                )
            )
        weights[layer] = tuple(pair)
    return weights


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
            stored_residuals=layout.stored_residuals,
        ),
        encode_coordinates(
            v,
            value_weights,
            layout.value_sparse,
            code_dtype=code_dtype,
            stored_residuals=layout.stored_residuals,
            coordinate_bitmap=layout.value_bitmap,
        ),
    )


def materialize_coordinates(code, weights):
    """Validation-only reconstruction; never call from absorbed serving reads."""
    rec = torch.einsum("thr,hdr->thd", code.latent.float(), weights.decoder)
    rec = rec + weights.mean
    if code.stored_residuals:
        return rec.scatter_add(-1, code.coordinate_indices(), code.originals.float())
    return rec.scatter(-1, code.coordinate_indices(), code.originals.float())


def sparse_corrections(code, weights):
    """Decode only the selected coordinates, without a full K/V reconstruction."""
    if code.stored_residuals:
        return code.originals.float()
    heads = torch.arange(weights.mean.shape[0], device=code.latent.device)[
        None, :, None
    ]
    index = code.coordinate_indices()
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
    ki = prefix.key.coordinate_indices()[:, heads]
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
    vi = prefix.value.coordinate_indices()[:, heads]
    vc = sparse_corrections(prefix.value, value_weights)[:, heads]
    weighted = pc.permute(0, 2, 1).unsqueeze(-1) * vc.unsqueeze(0)
    # Sum sparse residuals into output coordinates, not per-token dense values.
    output.scatter_add_(
        -1,
        vi.permute(1, 0, 2).reshape(1, hq, -1).expand(n, -1, -1),
        weighted.permute(0, 2, 1, 3).reshape(n, hq, -1),
    )
    return output
