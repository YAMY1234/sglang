"""Flash-Next component loader over the shared packed residual codec.

All positions, including position zero, follow the same reference rounding.
Only the E/D arithmetic uses the selected TF32 policy.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from sglang.srt.duet.latent_codec import PackedResidualCode


@dataclass
class LatentBatch:
    """Compatibility record for the deferred private-cache materializer."""

    z: torch.Tensor
    z_scale: torch.Tensor
    rms: torch.Tensor
    spike_indices: torch.Tensor
    spike_values: torch.Tensor
    sink_rows: torch.Tensor
    sink_values: torch.Tensor


class FlashNextLatentCodec(nn.Module):
    def __init__(self, *, device, spec, width, compute_precision="fp32"):
        super().__init__()
        if compute_precision not in ("fp32", "tf32"):
            raise ValueError("Flash-Next LinearCode supports fp32 or tf32")
        self.spec = spec
        self.compute_precision = compute_precision
        self.WIDTH, self.RANK, self.SPIKES = (
            width,
            spec["latent_rank"],
            spec["latent_spikes"],
        )
        self.PAYLOAD_BYTES = self.RANK // 2 + self.RANK // 16 + 4 + 3 * self.SPIKES
        for name, shape in (
            ("E", (self.RANK, width)),
            ("D", (width, self.RANK)),
            ("mean", (width,)),
        ):
            self.register_buffer(
                name, torch.empty(shape, dtype=torch.float32, device=device)
            )
        self._loaded = set()
        self._codec = None

    @torch.no_grad()
    def load(self, name, tensor):
        name = "mean" if name == "mu" else name
        if name not in ("E", "D", "mean"):
            raise KeyError(name)
        target = getattr(self, name)
        if tensor.ndim == target.ndim + 1 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.dtype != torch.float32 or tensor.shape != target.shape:
            raise ValueError(f"invalid published {name} dtype/shape")
        target.copy_(tensor.to(torch.bfloat16).float())
        self._loaded.add(name)
        self._codec = None

    def finalize(self):
        if self._loaded != {"E", "D", "mean"}:
            raise ValueError(
                f"missing latent components: { {'E', 'D', 'mean'} - self._loaded }"
            )
        if self._codec is None:
            self._codec = PackedResidualCode(
                self.E,
                self.D,
                self.mean,
                self.SPIKES,
                id_side=self.spec["latent_id_side"],
            )

    @contextmanager
    def _precision(self):
        saved = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = self.compute_precision == "tf32"
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = saved

    @torch.no_grad()
    def encode(self, streams, positions, base):
        self.finalize()
        if (
            streams.shape != base.shape
            or streams.ndim != 2
            or streams.shape[1] != self.WIDTH
        ):
            raise ValueError(
                "Flash-Next latent requires matching residual and embedding streams"
            )
        if positions.shape != (streams.shape[0],):
            raise ValueError("one logical position is required per residual row")
        with self._precision():
            return self._codec.encode(streams, base, positions)

    @torch.no_grad()
    def decode(self, batch, base):
        self.finalize()
        with self._precision():
            return self._codec.decode(batch, base)

    def encode_and_decode(self, streams, positions, base):
        batch = self.encode(streams, positions, base)
        return batch, self.decode(batch, base)
