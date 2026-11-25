"""NVFP4 encoding of FP8 KV for the HiCache host tier (--hicache-host-kv-dtype nvfp4).

A host row is the linear-scale NVFP4 KV format of FlashInfer's
``nvfp4_kv_quantize``: E2M1 values, two per byte with the low nibble first,
then one E4M3 scale per 16 values. The global scale is fixed at 1/6, so every
block scale equals its block's FP8 amax, which E4M3 holds exactly: no
calibration, and any process decodes any page (L2 or L3).

On CUDA the rows go through fused Triton kernels (nvfp4_host_triton), checked
byte for byte against the torch twin below at construction; FlashInfer's
kernels and then the twin itself are the fallbacks. Only torch is imported at
module level, and the transfer helpers take the copy primitives as arguments,
so the byte layout can be checked without a GPU.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

logger = logging.getLogger(__name__)

NVFP4_BLOCK_SIZE = 16
FP8_SOURCE_GLOBAL_SCALE = 1.0 / 6.0

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between E2M1 magnitudes; a tie rounds to the even code, which is
# the upper neighbour only for these three.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_TIES_UP = (0.75, 1.75, 3.5)


def nvfp4_row_bytes(dim: int) -> int:
    return dim // 2 + dim // NVFP4_BLOCK_SIZE


def quantize_nvfp4_torch(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch twin of flashinfer.nvfp4_kv_quantize: [n, d] -> ([n, d/2], [n, d/16]) uint8."""
    n, dim = x.shape
    blocks = x.float().view(n, dim // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    gs = global_scale.float().to(x.device)
    sf = (amax / (6.0 * gs)).to(torch.float8_e4m3fn)
    inv = torch.where(amax > 0, 1.0 / (sf.float() * gs), torch.zeros_like(amax))
    y = blocks * inv
    mag = y.abs().clamp(max=6.0)
    code = torch.bucketize(mag, mag.new_tensor(_E2M1_MIDPOINTS))
    code += torch.isin(mag, mag.new_tensor(_E2M1_TIES_UP)).to(code.dtype)
    code |= torch.signbit(y).to(code.dtype) << 3
    code = code.to(torch.uint8).view(n, dim)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, sf.view(torch.uint8).view(n, dim // NVFP4_BLOCK_SIZE)


def dequantize_nvfp4_torch(
    fp4: torch.Tensor, sf: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """torch twin of flashinfer.nvfp4_kv_dequantize (bf16 output)."""
    n = fp4.shape[0]
    dim = fp4.shape[1] * 2
    lut = torch.tensor(
        E2M1_VALUES + tuple(-v for v in E2M1_VALUES),
        dtype=torch.float32,
        device=fp4.device,
    )
    codes = torch.stack((fp4 & 0xF, fp4 >> 4), dim=-1).view(n, dim)
    vals = lut[codes.long()].view(n, dim // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    scale = sf.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    out = (vals * scale) * global_scale.float().to(fp4.device)
    return out.view(n, dim).to(torch.bfloat16)


def _resolve_kernels(device: torch.device, global_scale: torch.Tensor):
    """(quantize, dequantize, name) for the unfused path."""
    if device.type != "cuda":
        return quantize_nvfp4_torch, dequantize_nvfp4_torch, "torch"
    try:
        from flashinfer import nvfp4_kv_dequantize, nvfp4_kv_quantize

        # Build the JIT modules at startup rather than on the first eviction.
        probe = torch.zeros((1, NVFP4_BLOCK_SIZE), dtype=torch.bfloat16, device=device)
        nvfp4_kv_dequantize(*nvfp4_kv_quantize(probe, global_scale), global_scale)
        return nvfp4_kv_quantize, nvfp4_kv_dequantize, "flashinfer kernels"
    except Exception as err:
        return quantize_nvfp4_torch, dequantize_nvfp4_torch, f"torch ({err!r})"


def _probe_rows(dim: int, device: torch.device) -> torch.Tensor:
    """FP8 rows covering every finite byte, zero and -0 blocks, subnormal-only
    blocks, +-448, and blocks dense in E2M1 ties."""
    finite = torch.tensor(
        [b for b in range(256) if b not in (0x7F, 0xFF)], dtype=torch.uint8
    )
    rows = finite.repeat(64 * dim // finite.numel() + 1)[: 64 * dim].view(64, dim)
    rows = rows.clone()
    rows[0] = 0
    rows[1] = 0x80
    rows[2] = torch.arange(1, 8, dtype=torch.uint8).repeat(dim // 7 + 1)[:dim]
    rows[3, ::16] = 0x7E
    rows[4, ::16] = 0xFE
    # amax 384 (0x7C) with 320 (0x7A) and 64 (0x60): scaled values 5.0 and 1.0
    rows[5] = torch.tensor([0x7C, 0x7A, 0x60, 0xFA] * (dim // 4), dtype=torch.uint8)
    return rows.to(device)


def _resolve_fused(device: torch.device, global_scale: torch.Tensor, dim: int):
    """(nvfp4_host_triton module, None) if its kernels run and match the torch
    twin byte for byte on a probe, else (None, reason)."""
    try:
        from sglang.srt.mem_cache.pool_host import nvfp4_host_triton as fused

        x = _probe_rows(dim, device)
        n = x.shape[0]
        rows = torch.arange(n, device=device)
        row_bytes = nvfp4_row_bytes(dim)
        q, sf = quantize_nvfp4_torch(x.view(torch.float8_e4m3fn).float(), global_scale)
        want = torch.cat((q, sf), dim=1)
        got = torch.zeros((1, n, row_bytes), dtype=torch.uint8, device=device)
        fused.encode_rows(x, rows, got, global_scale, dim=dim, page_size=n)
        deq = dequantize_nvfp4_torch(q, sf, global_scale)
        want_fp8 = deq.to(torch.float8_e4m3fn).view(torch.uint8)
        got_fp8 = torch.zeros_like(x)
        fused.decode_rows(want, rows, got_fp8, global_scale, dim=dim)
        bad = (int((got.view(n, -1) != want).sum()), int((got_fp8 != want_fp8).sum()))
        if bad != (0, 0):
            return None, f"differ from the torch twin on the probe ({bad} bytes)"
        return fused, None
    except Exception as err:
        return None, repr(err)


class NVFP4RowCodec:
    """FP8 rows (uint8 storage, dim values) <-> NVFP4 rows (dim/2 + dim/16 bytes)."""

    def __init__(self, dim: int, device: torch.device | str):
        if dim % NVFP4_BLOCK_SIZE:
            raise ValueError(f"NVFP4 rows need a multiple of 16 values, got {dim}")
        device = torch.device(device)
        self.dim = dim
        self.data_bytes = dim // 2
        self.row_bytes = nvfp4_row_bytes(dim)
        self.global_scale = torch.tensor(
            [FP8_SOURCE_GLOBAL_SCALE], dtype=torch.float32, device=device
        )
        self._fused, fused_error = _resolve_fused(device, self.global_scale, dim)
        if self._fused is not None:
            self.backend = "triton fused kernels"
            # Encode writes into the caller's staging; decode reads only its stage.
            self.encode_scratch_bytes = 0
            self.decode_scratch_bytes = self.row_bytes
            return
        self._quantize, self._dequantize, name = _resolve_kernels(
            device, self.global_scale
        )
        self.backend = f"{name}, triton fused kernels failed: {fused_error}"
        if device.type == "cuda":
            logger.warning("HiCache NVFP4 host codec fallback: %s", self.backend)
        # Live at once per row: FP8 gather, bf16, NVFP4 out.
        self.encode_scratch_bytes = 3 * dim + self.row_bytes
        # NVFP4 stage, its split copies, bf16, FP8 out.
        self.decode_scratch_bytes = 2 * self.row_bytes + 3 * dim

    def encode_rows(
        self,
        src: torch.Tensor,
        rows: torch.Tensor,
        out: torch.Tensor,
        *,
        page_size: int,
    ) -> None:
        """Encode src[rows] (FP8 bytes) into out[page, unit, ..., row_bytes],
        unit = position in rows % page_size."""
        if self._fused is not None:
            self._fused.encode_rows(
                src, rows, out, self.global_scale, dim=self.dim, page_size=page_size
            )
            return
        x = src.index_select(0, rows).view(torch.float8_e4m3fn).view(-1, self.dim)
        fp4, sf = self._quantize(x.to(torch.bfloat16), self.global_scale)
        lead = out.shape[:-1]
        out[..., : self.data_bytes].copy_(fp4.view(*lead, self.data_bytes))
        out[..., self.data_bytes :].copy_(sf.view(*lead, -1))

    def decode_rows(
        self, src: torch.Tensor, rows: torch.Tensor, dst: torch.Tensor
    ) -> None:
        """Decode NVFP4 rows src[i] ([n, row_bytes]) into dst[rows[i]] (FP8 bytes)."""
        if self._fused is not None:
            self._fused.decode_rows(src, rows, dst, self.global_scale, dim=self.dim)
            return
        fp4 = src[:, : self.data_bytes].contiguous()
        sf = src[:, self.data_bytes :].contiguous()
        x = self._dequantize(fp4, sf, self.global_scale)
        fp8 = x.to(torch.float8_e4m3fn).view(torch.uint8)
        dst.index_copy_(0, rows, fp8.view(rows.numel(), *dst.shape[1:]))


class IndexKeyPageCodec:
    """One DSA index-key page: [keys FP8 | per-token FP32 scales] ->
    [NVFP4 key rows | the same scales]. The scales stay exact."""

    def __init__(self, page_size: int, head_dim: int, device: torch.device | str):
        self.page_size = page_size
        self.rows = NVFP4RowCodec(head_dim, device)
        self.key_bytes = page_size * head_dim
        self.encoded_key_bytes = page_size * self.rows.row_bytes
        self.scale_bytes = page_size * 4
        self.device_page_bytes = self.key_bytes + self.scale_bytes
        self.host_page_bytes = self.encoded_key_bytes + self.scale_bytes
        # Per page: the page gather and key reshape, then the row codec's.
        self.encode_scratch_bytes = (
            self.device_page_bytes
            + self.key_bytes
            + page_size * self.rows.encode_scratch_bytes
        )
        # Per page: the stage, key reshape and decoded keys, the row codec's
        # temporaries, then the reassembled page.
        self.decode_scratch_bytes = (
            self.host_page_bytes
            + self.encoded_key_bytes
            + self.key_bytes
            + page_size * self.rows.decode_scratch_bytes
            + self.device_page_bytes
        )
        self.backend = self.rows.backend

    def encode_rows(
        self,
        src: torch.Tensor,
        rows: torch.Tensor,
        out: torch.Tensor,
        *,
        page_size: int,
    ) -> None:
        """Encode the device pages src[rows] into out[page, 1, host_page_bytes]."""
        n = rows.numel()
        pages = src.index_select(0, rows)
        keys = pages[:, : self.key_bytes].reshape(n * self.page_size, self.rows.dim)
        host = out.view(n, self.host_page_bytes)
        self.rows.encode_rows(
            keys,
            torch.arange(n * self.page_size, device=keys.device),
            host[:, : self.encoded_key_bytes].view(
                n, self.page_size, self.rows.row_bytes
            ),
            page_size=self.page_size,
        )
        host[:, self.encoded_key_bytes :].copy_(pages[:, self.key_bytes :])

    def decode_rows(
        self, src: torch.Tensor, rows: torch.Tensor, dst: torch.Tensor
    ) -> None:
        """Decode host pages src[i] ([n, host_page_bytes]) into dst[rows[i]]."""
        n = rows.numel()
        key_rows = src[:, : self.encoded_key_bytes].reshape(
            n * self.page_size, self.rows.row_bytes
        )
        keys = torch.empty(
            (n * self.page_size, self.rows.dim), dtype=torch.uint8, device=src.device
        )
        self.rows.decode_rows(
            key_rows, torch.arange(n * self.page_size, device=src.device), keys
        )
        pages = torch.cat(
            (keys.view(n, self.key_bytes), src[:, self.encoded_key_bytes :]), dim=1
        )
        dst.index_copy_(0, rows, pages)


# Device scratch one encode or decode step may take on its transfer stream; the
# step's page count follows from it. Arbitrary: GLM-5.2 PP4 prefill ranks keep
# >= 55 GB free after the KV pool.
NVFP4_HOST_SCRATCH_BYTES = 256 << 20


def scratch_chunk_pages(scratch_bytes_per_page: int) -> int:
    return max(1, NVFP4_HOST_SCRATCH_BYTES // scratch_bytes_per_page)


def backup_all_layers(
    codec,
    device_layers,
    host_buffer: torch.Tensor,
    host_units: torch.Tensor,
    device_units: torch.Tensor,
    *,
    units_per_page: int,
    chunk_pages: int,
    arange: torch.Tensor,
    copy_pages_to_host: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
    ],
) -> None:
    """Encode every layer's units straight into page-major staging laid out like
    the host pages ([page, layer, unit, ...]), then copy whole pages to host.

    host_buffer is [host_pages, layers, units_per_page, ..., unit_bytes];
    host_units are page-aligned CPU indices; device_units index the device rows
    and already live on their device; arange is a CPU arange of >= chunk_pages.
    """
    total = host_units.numel()
    if total == 0:
        return
    step = chunk_pages * units_per_page
    staging = torch.empty(
        (
            min(total, step) // units_per_page,
            len(device_layers),
            units_per_page,
            host_buffer.shape[-1],
        ),
        dtype=torch.uint8,
        device=device_layers[0].device,
    )
    for start in range(0, total, step):
        h = host_units[start : start + step]
        d = device_units[start : start + step]
        pages = h.numel() // units_per_page
        chunk = staging[:pages]
        for j, layer in enumerate(device_layers):
            codec.encode_rows(layer, d, chunk[:, j], page_size=units_per_page)
        copy_pages_to_host(
            chunk.view(pages, -1),
            host_buffer,
            h[::units_per_page] // units_per_page,
            arange[:pages],
        )


def load_one_layer(
    codec,
    device_layer: torch.Tensor,
    host_buffer: torch.Tensor,
    host_layer: int,
    host_units: torch.Tensor,
    device_units: torch.Tensor,
    *,
    units_per_page: int,
    chunk_pages: int,
    arange: torch.Tensor,
    copy_layer_from_host: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor], None
    ],
) -> None:
    """Copy one host layer's units to a device stage, then decode them straight
    into the device rows."""
    total = host_units.numel()
    if total == 0:
        return
    step = chunk_pages * units_per_page
    for start in range(0, total, step):
        h = host_units[start : start + step]
        d = device_units[start : start + step]
        stage = torch.empty(
            (h.numel(), *host_buffer.shape[3:]),
            dtype=torch.uint8,
            device=device_layer.device,
        )
        copy_layer_from_host(
            host_buffer, stage, h, host_layer, units_per_page, arange[: h.numel()]
        )
        codec.decode_rows(stage.view(h.numel(), -1), d, device_layer)
