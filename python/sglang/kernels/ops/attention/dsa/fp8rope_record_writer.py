"""nvfp4.fp8rope MLA records for SGLANG_DSA_D576_INPLACE (torch reference writer).

One 352-byte record per token and layer:
  [0, 256)   E2M1 NoPE, 512 values, even-indexed value in the low nibble
  [256, 320) RoPE cast to E4M3
  [320, 352) E4M3 scale of each 16-value NoPE block b at byte 320 + 8 * (b % 4) + b // 4,
             scale = amax(block) / 6, no global scale
"""

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg

FP8ROPE_RECORD_BYTES = 352
_NOPE_DIM = 512
_ROPE_DIM = 64
_BLOCK = 16

# Midpoints between consecutive E2M1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_midpoints_by_device = {}


def dsa_d576_inplace_enabled() -> bool:
    return (
        envs.SGLANG_DSA_D576_INPLACE.get()
        and get_disagg().disaggregation_mode == "decode"
    )


def _midpoints(device: torch.device) -> torch.Tensor:
    t = _midpoints_by_device.get(device)
    if t is None:
        t = torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=device)
        _midpoints_by_device[device] = t
    return t


def write_fp8rope_records(
    dst: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> None:
    """Quantize each token's latent into a record and scatter it to ``dst[loc]``.

    Rows whose loc is the reserved slot 0 are skipped, as in set_mla_kv_buffer_triton.
    """
    n = loc.shape[0]
    blocks = k_nope.reshape(n, _NOPE_DIM // _BLOCK, _BLOCK).float()
    scale = (blocks.abs().amax(-1) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    scale_f = scale.float().unsqueeze(-1)
    x = torch.where(scale_f > 0, blocks / scale_f, torch.zeros_like(blocks))
    mag = torch.bucketize(x.abs(), _midpoints(x.device), out_int32=True)
    codes = (mag | (((x < 0) & (mag > 0)).to(torch.int32) << 3)).view(n, -1, 2)
    nope = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    rope = k_rope.reshape(n, _ROPE_DIM).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = scale.view(torch.uint8).view(n, 8, 4).transpose(1, 2).reshape(n, 32)
    records = torch.cat([nope, rope, scales], dim=1)

    rows = dst.view(-1, FP8ROPE_RECORD_BYTES)
    loc = loc.long()
    records = torch.where((loc != 0).unsqueeze(1), records, rows[loc])
    rows[loc] = records
