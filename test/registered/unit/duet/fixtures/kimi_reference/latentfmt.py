"""Storage format of the residual code and its quantizers (docs/56 §56.h), used by twinstar.duet.latent.LinearCode for both models.

A coded prompt token is  z (r coordinates)  +  m exact coordinates (value, index)  (+ the scales the z format itself carries).
The format names the storage precision of each part and is a field of the DuetSpec, so that

  * training sees the same numbers the served model stores (quantization-aware distillation: the quantizer runs in the
    training forward with a straight-through gradient), and
  * the byte account of the paper is read from the spec.  There is no evaluation-time override: a checkpoint is stored in the
    precision it was trained with (the pre-unification LATENT_STORE / LATENT_SPARSE_QUANT switches were removed on 2026-09-23).

Parts and formats
  z      fp32 | bf16 | fp8 (e4m3, one fp32 scale per token) | nvfp4 (e2m1, one e4m3 scale per 16 coordinates, one fp32 scale per token)
  value  fp32 | bf16 | fp8 (e4m3, one fp32 scale per token)
  index  uint16 | gap8 (indices sorted, gaps stored in one byte; a gap >= 255 is escaped as 255 + uint16) | packed (ceil(log2 d) bits)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
except ImportError:                      # the byte account (paper/scripts) runs where torch is absent; the quantizers need it
    torch = None

Z_FORMATS = ("fp32", "bf16", "fp8", "nvfp4")
VALUE_FORMATS = ("fp32", "bf16", "fp8")
INDEX_FORMATS = ("uint16", "gap8", "packed")

FP8_MAX = 448.0                  # e4m3fn
NVFP4_BLOCK = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)   # midpoints: round to nearest, ties to the even code (cvt.rn: 0.75 -> 1, 3.5 -> 4, 2.5 -> 2)


@dataclass(frozen=True)
class LatentFormat:
    z: str = "fp32"
    value: str = "fp32"
    index: str = "uint16"

    def __post_init__(self):
        if self.z not in Z_FORMATS:
            raise ValueError(f"latent z format {self.z!r}: expected one of {Z_FORMATS}")
        if self.value not in VALUE_FORMATS:
            raise ValueError(f"latent value format {self.value!r}: expected one of {VALUE_FORMATS}")
        if self.index not in INDEX_FORMATS:
            raise ValueError(f"latent index format {self.index!r}: expected one of {INDEX_FORMATS}")

    @property
    def quantized(self) -> bool:
        return self.z != "fp32" or self.value != "fp32"

    # ------------------------------------------------------------------ bytes
    def z_bytes(self, r: int) -> float:
        """Bytes of the r coordinates including their scales (fp8: one fp32 scale per token; nvfp4: one e4m3 scale per block + one fp32)."""
        if r == 0:
            return 0.0
        return {"fp32": 4.0 * r, "bf16": 2.0 * r, "fp8": r + 4.0,
                "nvfp4": 0.5 * r + math.ceil(r / NVFP4_BLOCK) + 4.0}[self.z]

    def value_bytes(self, m: int) -> float:
        if m == 0:
            return 0.0
        return {"fp32": 4.0 * m, "bf16": 2.0 * m, "fp8": m + 4.0}[self.value]

    def index_bytes(self, m: int, d: int) -> float:
        """Nominal index bytes; gap8 is data dependent (1 byte per index + 2 per escape) -- measure with gap8_bytes on real codes."""
        if m == 0:
            return 0.0
        return {"uint16": 2.0 * m, "gap8": 1.0 * m, "packed": math.ceil(m * math.ceil(math.log2(d)) / 8)}[self.index]

    def bytes_per_token(self, r: int, m: int, d: int) -> float:
        """Stored bytes of one coded token (z with its scales + exact values + indices); the same for both models."""
        return self.z_bytes(r) + self.value_bytes(m) + self.index_bytes(m, d)

    def describe(self) -> str:
        return f"z={self.z} value={self.value} index={self.index}"


def from_spec(spec) -> LatentFormat:
    """The format named by a spec (twinstar.duet.spec.DuetSpec, or the structural TwinStarSpec derived from it)."""
    return LatentFormat(z=spec.latent_z_format, value=spec.latent_value_format, index=spec.latent_index_format)


# ---------------------------------------------------------------------- quantizers (fake quantization in fp32, values on the storage grid)
def fq_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.float().to(torch.bfloat16).float()


def fq_fp8(x: torch.Tensor) -> torch.Tensor:
    """e4m3 with one fp32 absmax scale per row (last dim = the coordinates of one token)."""
    xf = x.float()
    s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return (xf / s).to(torch.float8_e4m3fn).float() * s


def _round_e2m1(y: torch.Tensor) -> torch.Tensor:
    """|y| <= 6 -> nearest e2m1 magnitude (0, .5, 1, 1.5, 2, 3, 4, 6); sign kept.  A value exactly on a midpoint goes to the level with
    the even code (codes 0..7 = the eight magnitudes), the round-to-nearest-even of the hardware conversion (checked against
    flashinfer's NVFP4 quantizer, twinstar/tests/test_nvfp4_kernel.py: every differing code before this rule was a midpoint)."""
    a = y.abs()
    edges = torch.tensor(_E2M1_EDGES, device=y.device, dtype=y.dtype)
    levels = torch.tensor(_E2M1, device=y.device, dtype=y.dtype)
    idx = torch.bucketize(a, edges, right=True)         # a <= 0.25 -> 0 ... a midpoint goes up here; the next line sends odd codes back down
    tie = torch.isin(a, edges) & (idx % 2 == 1)
    idx = torch.where(tie, idx - 1, idx)
    return levels[idx] * torch.sign(y)


def fq_nvfp4(x: torch.Tensor, block: int = NVFP4_BLOCK) -> torch.Tensor:
    """NVFP4: e2m1 values, one e4m3 scale per `block` consecutive coordinates, one fp32 scale per row.
    The row scale g maps the row's absmax onto 6 * 448 so that every block scale amax_b / 6 lies in the e4m3 range; block scales
    are rounded to nearest e4m3 and the scaled values clipped to [-6, 6], the convention of the NVFP4 recipe (Model Optimizer)."""
    xf = x.float()
    shp = xf.shape
    r = shp[-1]
    if r % block:
        raise ValueError(f"nvfp4 needs the row length ({r}) to be a multiple of {block}")
    rows = xf.reshape(-1, r)
    g = rows.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (6.0 * FP8_MAX)            # (N, 1) fp32 row scale
    blk = (rows / g).reshape(rows.shape[0], r // block, block)                            # values now in [-6*448, 6*448]
    sb = blk.abs().amax(-1, keepdim=True) / 6.0                                           # exact block scale in [0, 448]
    sb8 = sb.to(torch.float8_e4m3fn).float().clamp_min(2.0 ** -9)                        # smallest e4m3 subnormal; an all-zero block stays zero
    q = _round_e2m1((blk / sb8).clamp(-6.0, 6.0)) * sb8
    return (q.reshape(rows.shape) * g).reshape(shp)


QUANTIZERS = {"fp32": lambda x: x.float(), "bf16": fq_bf16, "fp8": fq_fp8, "nvfp4": fq_nvfp4}


def quantize(x: torch.Tensor, fmt_name: str) -> torch.Tensor:
    """x (fp32) -> fp32 values on the storage grid of fmt_name.  Under autograd the gradient passes straight through
    (x + (q(x) - x).detach()), which is what makes the training forward quantization-aware."""
    if fmt_name == "fp32":
        return x
    q = QUANTIZERS[fmt_name](x.detach())
    return x + (q - x).detach() if x.requires_grad else q


# ---------------------------------------------------------------------- gap-coded uint8 indices
GAP8_ESCAPE = 255


def pack_gap8(idx: torch.Tensor) -> torch.Tensor:
    """idx (m,) integer indices of one token (any order) -> uint8 stream: sorted, first index as gap from -1, each gap g < 255
    as one byte, g >= 255 as 255 followed by g as two little-endian bytes."""
    s = torch.sort(idx.reshape(-1).to(torch.int64)).values
    gaps = torch.diff(torch.cat([torch.tensor([-1], device=s.device), s])) - 1     # >= 0; consecutive indices give gap 0
    out = []
    for g in gaps.tolist():
        if g < GAP8_ESCAPE:
            out.append(g)
        else:
            out += [GAP8_ESCAPE, g & 0xFF, (g >> 8) & 0xFF]
    return torch.tensor(out, dtype=torch.uint8)


def unpack_gap8(stream: torch.Tensor, m: int) -> torch.Tensor:
    b = stream.tolist()
    out, pos, i = [], -1, 0
    while len(out) < m:
        g = b[i]; i += 1
        if g == GAP8_ESCAPE:
            g = b[i] | (b[i + 1] << 8); i += 2
        pos += g + 1
        out.append(pos)
    if i != len(b):
        raise ValueError("gap8 stream has trailing bytes")
    return torch.tensor(out, dtype=torch.int64)


def gap8_bytes(idx: torch.Tensor) -> torch.Tensor:
    """idx (..., m) -> bytes of the gap8 stream per token (..., ), vectorised: m + 2 * (#gaps >= 255)."""
    s = torch.sort(idx.to(torch.int64), dim=-1).values
    first = s[..., :1] + 1
    gaps = torch.cat([first, torch.diff(s, dim=-1)], -1) - 1
    if (gaps < 0).any():
        raise ValueError("duplicate indices in one token")
    return s.shape[-1] + 2 * (gaps >= GAP8_ESCAPE).sum(-1)


def measured_index_bytes(fmt: LatentFormat, idx: torch.Tensor, d: int) -> torch.Tensor:
    """Per-token index bytes for the format on actual indices (..., m); only gap8 is data dependent."""
    m = idx.shape[-1]
    if fmt.index == "gap8":
        return gap8_bytes(idx).float()
    return torch.full(idx.shape[:-1], float(fmt.index_bytes(m, d)), device=idx.device)
