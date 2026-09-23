"""Packed scheme-C codec for the QAD Flash-Next release.

Arithmetic oracle: twinstar-pd-models 856dfa32, duet/latentfmt.py and
models/twinstar_model.py::LatentBottleneck. Keep its operation order, including
bucketize(right=False) at exact midpoints. E/D GEMM precision is an explicit
serving-policy choice; all non-GEMM arithmetic and stored byte formats stay fixed.
"""
from dataclasses import dataclass

import torch
from torch import nn

LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _pack_nvfp4_torch(x):
    """Return packed nibbles, e4m3 block scales, and the fp32 row scale."""
    if x.ndim != 2 or x.shape[-1] % 16:
        raise ValueError("nvfp4 requires 2D rows with width a multiple of 16")
    xf = x.float()
    n, r = xf.shape
    g = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (6.0 * 448.0)
    blk = (xf / g).reshape(n, r // 16, 16)
    sb = (blk.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    sbf = sb.float().clamp_min(2.0 ** -9)
    y = (blk / sbf.unsqueeze(-1)).clamp(-6.0, 6.0)
    edges = torch.tensor(EDGES, dtype=y.dtype, device=y.device)
    mag = torch.bucketize(y.abs(), edges, right=False).to(torch.uint8)
    # Keep negative zero after magnitude rounds to zero, as levels*sign does.
    code = (mag | ((y < 0).to(torch.uint8) << 3)).reshape(n, r)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, sb, g


def _unpack_nvfp4_torch(packed, block_scale, row_scale):
    n, half = packed.shape
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(n, half * 2)
    levels = torch.tensor(LEVELS, device=packed.device, dtype=torch.float32)
    q = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    q = q.reshape(n, half // 8, 16) * block_scale.float().clamp_min(2.0 ** -9).unsqueeze(-1)
    return q.reshape(n, half * 2) * row_scale


def gap8_capacity(width, sparse):
    # Sum of skipped coordinates <= width - sparse; every escape consumes >=255.
    return sparse + 2 * min(sparse, (width - sparse) // 255)


def _pack_gap8_torch(indices, *, width=10240):
    """GPU-vectorized reference byte stream, with zero padding and explicit length.

    Returns stream, length, sorting permutation. Values MUST use the same
    permutation. Padded capacity is storage overhead, not nominal payload bytes.
    """
    if indices.ndim != 2 or not 0 < indices.shape[1] <= width <= 65535:
        raise ValueError("gap8 expects nonempty rows of distinct uint16 coordinates")
    idx, order = indices.long().sort(dim=-1)
    if bool(((idx < 0) | (idx >= width)).any()):
        raise ValueError("gap8 index out of range")
    gaps = torch.cat((idx[:, :1], idx[:, 1:] - idx[:, :-1] - 1), -1)
    if bool((gaps < 0).any()):
        raise ValueError("gap8 duplicate indices")
    escape = gaps >= 255
    sizes = 1 + 2 * escape.long()
    offsets = sizes.cumsum(-1) - sizes
    cap = gap8_capacity(width, idx.shape[1])
    out = torch.zeros((idx.shape[0], cap), dtype=torch.int64, device=idx.device)
    out.scatter_add_(1, offsets, torch.where(escape, 255, gaps))
    for step, data in ((1, gaps & 255), (2, gaps >> 8)):
        dest = torch.where(escape, offsets + step, 0)
        out.scatter_add_(1, dest, torch.where(escape, data, 0))
    return out.to(torch.uint8), sizes.sum(-1, keepdim=True).to(torch.int16), order


def _unpack_gap8_torch(stream, lengths, *, sparse=512, width=10240):
    """Vectorized parsing; escape high bytes cannot equal 255 for this width.

    A literal 255 low byte is immediately preceded by its escape marker. With
    width <= 65280, no high byte is 255, so marker recognition is unambiguous.
    """
    if width > 65280 or stream.ndim != 2:
        raise ValueError("vectorized gap8 decoder requires width <= 65280")
    b = stream.long()
    marker = b == 255
    prev = torch.nn.functional.pad(marker[:, :-1], (1, 0))
    marker = marker & ~prev
    following = torch.nn.functional.pad(marker[:, :-1], (1, 0))
    following2 = torch.nn.functional.pad(marker[:, :-2], (2, 0))
    columns = torch.arange(b.shape[1], device=b.device)[None, :]
    live = columns < lengths.long()
    starts = live & ~following & ~following2
    low = torch.nn.functional.pad(b[:, 1:], (0, 1))
    high = torch.nn.functional.pad(b[:, 2:], (0, 2))
    gaps = torch.where(marker, low + (high << 8), b)
    if bool((starts.sum(-1) != sparse).any()):
        raise ValueError("malformed gap8 row length")
    if bool((marker & live & (columns + 2 >= lengths)).any()):
        raise ValueError("truncated gap8 escape")
    ranks = starts.long().cumsum(-1) - 1
    coords = torch.where(starts, gaps + 1, 0).cumsum(-1) - 1
    out = torch.zeros((b.shape[0], sparse), dtype=torch.int64, device=b.device)
    out.scatter_add_(1, ranks.clamp(0, sparse - 1), torch.where(starts, coords, 0))
    if bool(((out < 0) | (out >= width)).any()):
        raise ValueError("decoded gap8 index out of range")
    return out


def pack_nvfp4(x):
    if x.ndim != 2 or x.shape[-1] % 16:
        raise ValueError("nvfp4 requires 2D rows with width a multiple of 16")
    if x.is_cuda and 0 < x.shape[1] <= 8192:
        from sglang.srt.mem_cache import flashnext_scheme_c_kernels as kernels
        return kernels.pack_nvfp4(x)
    return _pack_nvfp4_torch(x)


def unpack_nvfp4(packed, block_scale, row_scale):
    if packed.is_cuda and 0 < packed.shape[1] <= 4096:
        from sglang.srt.mem_cache import flashnext_scheme_c_kernels as kernels
        return kernels.unpack_nvfp4(packed, block_scale, row_scale)
    return _unpack_nvfp4_torch(packed, block_scale, row_scale)


def pack_gap8(indices, *, width=10240, validate=True):
    if indices.ndim != 2 or not 0 < indices.shape[1] <= width <= 65535:
        raise ValueError("gap8 expects nonempty rows of distinct uint16 coordinates")
    if indices.is_cuda and indices.shape[1] <= 1024:
        from sglang.srt.mem_cache import flashnext_scheme_c_kernels as kernels
        return kernels.pack_gap8(indices, width, validate=validate)
    return _pack_gap8_torch(indices, width=width)


def unpack_gap8(stream, lengths, *, sparse=512, width=10240, validate=True):
    if width > 65280 or stream.ndim != 2:
        raise ValueError("vectorized gap8 decoder requires width <= 65280")
    if stream.is_cuda and 0 < stream.shape[1] <= 4096:
        from sglang.srt.mem_cache import flashnext_scheme_c_kernels as kernels
        return kernels.unpack_gap8(stream, lengths, sparse, width, validate=validate)
    return _unpack_gap8_torch(stream, lengths, sparse=sparse, width=width)


@dataclass
class SchemeCBatch:
    z: torch.Tensor
    z_block_scale: torch.Tensor
    z_scale: torch.Tensor
    rms: torch.Tensor
    spike_indices: torch.Tensor  # padded gap8 bytes
    spike_lengths: torch.Tensor
    spike_values: torch.Tensor  # sorted coordinate order, bf16
    sink_rows: torch.Tensor
    sink_values: torch.Tensor

    def nominal_payload_bytes(self):
        n = self.z.shape[0]
        return (self.z.nbytes + self.z_block_scale.nbytes + self.z_scale.nbytes
                + self.rms.nbytes + self.spike_values.nbytes
                + int(self.spike_lengths.sum().item()))

    def payload_bytes(self):
        return sum(getattr(self, name).nbytes for name in (
            "z", "z_block_scale", "z_scale", "rms", "spike_indices",
            "spike_lengths", "spike_values"))

    def sink_bytes(self):
        return self.sink_rows.nbytes + self.sink_values.nbytes


class FlashNextSchemeCCodec(nn.Module):
    WIDTH = 10240
    RANK = 4096
    SPIKES = 512
    PAYLOAD_BYTES = 3848  # nominal, excludes escapes and service metadata

    def __init__(self, *, device, compute_precision="fp32"):
        super().__init__()
        for name, shape in (("E", (self.RANK, self.WIDTH)),
                            ("D", (self.WIDTH, self.RANK)), ("mean", (self.WIDTH,))):
            self.register_buffer(name, torch.empty(shape, dtype=torch.float32, device=device))
        self._loaded = set()
        self.register_buffer("E_bf16", None, persistent=False)
        self.register_buffer("D_bf16", None, persistent=False)
        self.set_compute_precision(compute_precision)

    def set_compute_precision(self, precision):
        """Only call at initialization or after draining and flushing requests.

        Changing E/D with existing cached payloads would mix numerical policies.
        The published fp32 component buffers are never modified by this switch.
        """
        if precision not in ("fp32", "tf32", "bf16"):
            raise ValueError(f"unknown E/D compute precision: {precision}")
        self.compute_precision = precision
        if precision == "bf16" and self._loaded == {"E", "D", "mean"}:
            self.E_bf16 = self.E.to(torch.bfloat16)
            self.D_bf16 = self.D.to(torch.bfloat16)
        else:
            self.E_bf16 = self.D_bf16 = None

    def project(self, inputs, matrix):
        """E/D-only GEMM; restore backend precision before any base-model work.

        bf16 uses bf16 inputs/weights with an fp32 output accumulator on CUDA.
        CPU is a diagnostic fallback with fp32 accumulation of rounded inputs.
        Each serving worker executes this on its serial model-forward thread.
        """
        if matrix not in ("E", "D"):
            raise ValueError("only latent E/D projections may change precision")
        if self.compute_precision == "bf16":
            weight = getattr(self, matrix + "_bf16")
            if weight is None:
                raise RuntimeError("bf16 E/D buffers must be finalized before forward")
            rounded = inputs.to(torch.bfloat16)
            if inputs.is_cuda:
                output = torch.empty((*rounded.shape[:-1], weight.shape[0]),
                                     dtype=torch.float32, device=rounded.device)
                # Deterministic serving replaces aten::mm and aten::mm.dtype.
                # The explicit-output native overload preserves the requested
                # E/D precision without disabling deterministic base kernels.
                torch.mm(rounded.reshape(-1, rounded.shape[-1]), weight.T,
                         out_dtype=torch.float32, out=output.reshape(-1, weight.shape[0]))
                return output
            return rounded.float() @ weight.float().T
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = self.compute_precision == "tf32"
            weight = getattr(self, matrix)
            output = torch.empty((*inputs.shape[:-1], weight.shape[0]),
                                 dtype=torch.float32, device=inputs.device)
            torch.mm(inputs.float().reshape(-1, inputs.shape[-1]), weight.T,
                     out=output.reshape(-1, weight.shape[0]))
            return output
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    def load(self, name, tensor):
        if name not in ("E", "D", "mean"):
            raise KeyError(name)
        target = getattr(self, name)
        if tensor.dtype != torch.float32 or target.shape != tensor.shape:
            raise ValueError(f"invalid published {name} dtype/shape")
        # Actual release loader casts component tensors through bf16 before
        # attach_codes promotes them back to fp32. Keep this explicit.
        target.copy_(tensor.to(torch.bfloat16).float())
        if name in ("E", "D"):
            setattr(self, name + "_bf16", None)
        self._loaded.add(name)

    def finalize(self):
        if self._loaded != {"E", "D", "mean"}:
            raise ValueError(f"missing scheme-C matrices: {set(('E','D','mean')) - self._loaded}")
        if self.compute_precision == "bf16" and (self.E_bf16 is None or self.D_bf16 is None):
            self.set_compute_precision("bf16")

    @torch.no_grad()
    def encode(self, streams, positions, base):
        return self._encode(streams, positions, base, with_reconstruction=False)

    @torch.no_grad()
    def encode_and_decode(self, streams, positions, base):
        """Reuse the exact D product already needed to select sparse corrections."""
        return self._encode(streams, positions, base, with_reconstruction=True)

    def _encode(self, streams, positions, base, *, with_reconstruction):
        self.finalize()
        if (streams.ndim != 2 or streams.shape[1] != self.WIDTH
                or streams.dtype != torch.bfloat16 or base.shape != streams.shape
                or base.dtype != streams.dtype or base.device != streams.device
                or positions.shape != streams.shape[:1]
                or positions.device != streams.device or self.E.device != streams.device):
            raise ValueError("scheme-C requires matching bf16 residual/embedding and logical positions")
        residual = streams.float() - base.float()
        rms = residual.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        normalized = residual / rms
        z = self.project(normalized - self.mean, "E")
        packed, block_scale, scale = pack_nvfp4(z)
        reconstructed = self.mean + self.project(unpack_nvfp4(packed, block_scale, scale), "D")
        correction = normalized - reconstructed
        indices = correction.abs().topk(self.SPIKES, dim=-1).indices
        values = correction.gather(-1, indices).to(torch.bfloat16)
        # topk supplies distinct in-range indices. Avoid a GPU->CPU validation
        # barrier on this internal path; public codec calls still validate.
        gap, lengths, order = pack_gap8(indices, width=self.WIDTH, validate=False)
        sink_rows = (positions == 0).nonzero(as_tuple=True)[0]
        batch = SchemeCBatch(packed, block_scale, scale, rms, gap, lengths,
                             values.gather(-1, order), sink_rows, streams.index_select(0, sink_rows))
        if not with_reconstruction:
            return batch
        # Same elementwise operations as decode. Coordinates are unique, so
        # scattering the original top-k order is bitwise equal to sorted gap8.
        sparse = torch.zeros_like(reconstructed).scatter_(-1, indices, values.float())
        result = (reconstructed + sparse) * rms + base.float()
        result.index_copy_(0, sink_rows, batch.sink_values.float())
        return batch, result.to(torch.bfloat16)

    @torch.no_grad()
    def decode(self, batch, base):
        self.finalize()
        if (base.shape != (batch.z.shape[0], self.WIDTH) or base.dtype != torch.bfloat16
                or base.device != self.E.device):
            raise ValueError("scheme-C decode requires matching bf16 token embeddings")
        z = unpack_nvfp4(batch.z, batch.z_block_scale, batch.z_scale)
        reconstructed = self.mean + self.project(z, "D")
        indices = unpack_gap8(batch.spike_indices, batch.spike_lengths,
                              sparse=self.SPIKES, width=self.WIDTH, validate=False)
        correction = torch.zeros_like(reconstructed).scatter_(-1, indices, batch.spike_values.float())
        result = (reconstructed + correction) * batch.rms + base.float()
        result.index_copy_(0, batch.sink_rows, batch.sink_values.float())
        return result.to(torch.bfloat16)
