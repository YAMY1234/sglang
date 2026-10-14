"""Slot-owned KDA factors shared by Kimi and GLM; opt-in FP32/BF16 storage, FP32 arithmetic.

Dense workspaces belong only to unfinished prefills. A final prompt commits
factors and releases its continuation. No dense slot array/ring is allocated,
and decode neither creates nor reads dense storage.
"""

import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactoredKDAConfig:
    rank: int
    window: int
    storage: str = "fp32"
    deferred: bool = False
    immediate: bool = True

    @property
    def use_deferred(self):
        return self.storage == "bf16" or self.deferred

    @property
    def use_immediate(self):
        return self.immediate and not self.use_deferred

    @property
    def dtype(self):
        return {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp32_bf16": torch.bfloat16,
        }[self.storage]

    @property
    def coeff_dtype(self):
        return torch.float32 if self.use_immediate else self.dtype

    @classmethod
    def from_env(cls):
        value = os.environ.get("SGLANG_KDA_FACTORED_STATE", "")
        if value in ("", "0"):
            return None
        fields = dict(pair.split("=") for pair in value.split(","))
        if not {"r", "w"} <= set(fields) or set(fields) - {"r", "w", "dtype"}:
            raise ValueError(
                "SGLANG_KDA_FACTORED_STATE requires r=<rank>,w=<window>[,dtype=fp32|fp32_bf16]"
            )
        cfg = cls(int(fields["r"]), int(fields["w"]), fields.get("dtype", "fp32"))
        if cfg.storage not in ("fp32", "fp32_bf16"):
            raise ValueError(
                "KDA factor dtype must be fp32 or fp32_bf16 (U FP32, V BF16); deferred BF16 is rejected"
            )
        if min(cfg.rank, cfg.window) < 1 or cfg.rank + cfg.window > 64:
            raise ValueError("KDA factors require 1 <= r,w and r+w <= 64")
        return cfg

    @property
    def capacity(self):
        return self.rank + self.window

    def state_bytes_per_layer(self, shape):
        heads, values, keys = shape.temporal
        columns = self.capacity * (
            keys * self.coeff_dtype.itemsize + values * self.dtype.itemsize
        )
        delayed = (keys + self.capacity) * 4 if self.use_deferred else 0
        return heads * (keys * 4 + columns + delayed + 4)

    def per_req_bytes(self, params):
        from math import prod

        conv = sum(prod(s) for s in params.shape.conv) * params.dtype.conv.itemsize
        return len(params.layers) * (conv + self.state_bytes_per_layer(params.shape))

    def fixed_bytes(self, shape, layers):
        heads, values, keys = shape.temporal
        return 4 * (layers * heads * values + heads * keys * (self.rank + 8))


class FactoredKDAPool:
    def __init__(self, size, cache_params, layer_ids, device, cfg):
        self.cfg = cfg
        self.layer_ids = list(layer_ids)
        self.layer_map = {layer: i for i, layer in enumerate(layer_ids)}
        self.heads, self.values, self.keys = cache_params.shape.temporal
        if cfg.rank >= min(self.keys, self.values):
            raise ValueError("KDA content rank must be smaller than head dimensions")
        dims = (len(layer_ids), size + 1, self.heads)
        self.a = torch.zeros(*dims, self.keys, dtype=torch.float32, device=device)
        self.log_decay = torch.zeros_like(self.a) if cfg.use_deferred else None
        self.column_exponent = (
            torch.zeros(*dims, cfg.capacity, dtype=torch.float32, device=device)
            if cfg.use_deferred
            else None
        )
        self.basis = torch.zeros(
            *dims, cfg.capacity, self.values, dtype=cfg.dtype, device=device
        )
        self.coeff = torch.zeros(
            *dims, cfg.capacity, self.keys, dtype=cfg.coeff_dtype, device=device
        )
        self.count = torch.full(dims, cfg.rank, dtype=torch.int32, device=device)
        self.vbar = torch.zeros(
            len(layer_ids), self.heads, self.values, dtype=torch.float32, device=device
        )
        generator = torch.Generator(device=device).manual_seed(0)
        self.omega = torch.randn(
            1,
            self.heads,
            self.keys,
            cfg.rank + 8,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        self.directions_bound = False
        self._prefill_final = None
        self._continuations = {}
        logger.info(
            "KDA factor pool: slots=%d layers=%d heads=%d K=%d V=%d R=%d W=%d "
            "storage=%s capacity=%d factor_bytes=%d fixed_bytes=%d dense_temporal_bytes=0",
            size + 1,
            len(layer_ids),
            self.heads,
            self.keys,
            self.values,
            cfg.rank,
            cfg.window,
            cfg.storage,
            cfg.capacity,
            sum(t.numel() * t.element_size() for t in self.tensors()),
            self.vbar.numel() * self.vbar.element_size()
            + self.omega.numel() * self.omega.element_size(),
        )

    @contextmanager
    def prefill(self, slots, final):
        """GLM chunk plan, outside decode/capture, with scheduler final flags.

        Do not truncate intermediate chunks: their exact state lives only until
        the next chunk or final commit, as in GDN's exact prefill continuation.
        """
        if self._prefill_final is not None:
            raise RuntimeError("nested KDA prefill plan")
        self._prefill_final = dict(zip(slots.tolist(), final))
        try:
            yield
        finally:
            self._prefill_final = None

    def bind_directions(self, directions):
        for layer, index in self.layer_map.items():
            self.vbar[index].copy_(directions[layer])
        self.directions_bound = True
        return self

    def tensors(self):
        main = (self.a, self.basis, self.coeff)
        if self.log_decay is not None:
            main += (self.log_decay, self.column_exponent)
        return (*main, self.count)

    def reset_slots(self, indices):
        if self._continuations:
            selected = set(indices.tolist())
            self._continuations = {
                key: value
                for key, value in self._continuations.items()
                if key[1] not in selected
            }
        for tensor in self.tensors()[:-1]:
            tensor[:, indices] = 0
        self.count[:, indices] = self.cfg.rank

    def copy_slots(self, src, dst):
        if self._continuations:
            pairs = list(zip(src.tolist(), dst.tolist()))
            saved = {
                (li, target): self._continuations.get((li, source))
                for source, target in pairs
                for li in range(len(self.layer_ids))
            }
            for key, value in saved.items():
                if value is None:
                    self._continuations.pop(key, None)
                else:
                    self._continuations[key] = value.clone()
        for tensor in self.tensors():
            tensor[:, dst] = tensor[:, src]

    def get_cpu_slots(self, indices):
        ordinary = tuple(t[:, indices].cpu() for t in self.tensors())
        if not self._continuations:
            return ordinary
        temporary = {
            (li, row): self._continuations[(li, slot)].cpu()
            for row, slot in enumerate(indices.tolist())
            for li in range(len(self.layer_ids))
            if (li, slot) in self._continuations
        }
        return (*ordinary, temporary)

    def load_cpu_slots(self, data, indices):
        self.reset_slots(indices)
        tensors = self.tensors()
        for tensor, saved in zip(tensors, data[: len(tensors)]):
            tensor[:, indices] = saved.to(tensor.device)
        if len(data) == len(tensors) + 1:
            slots = indices.tolist()
            for (li, row), value in data[-1].items():
                self._continuations[li, slots[row]] = value.to(self.a.device)

    def dense(self, layer, slots):
        """Extend-only (batch,H,V,K) scratch; never called by decode."""
        li = self.layer_map[layer]
        idx = slots.long().clamp_min(0)
        mask = (
            torch.arange(self.cfg.capacity, device=self.a.device)
            < self.count[li, idx, :, None]
        )
        c = self.coeff[li, idx].float() * mask[..., None]
        if self.log_decay is not None:
            logscale = (
                self.log_decay[li, idx, :, None, :]
                + self.column_exponent[li, idx, :, :, None]
            )
            c = c.sign() * torch.exp(c.abs().log() + logscale)
        dense = self.basis[li, idx].float().transpose(-1, -2) @ c
        dense += self.vbar[li, :, :, None] * self.a[li, idx, :, None, :].float()
        if self._continuations:
            for row, slot in enumerate(slots.tolist()):
                pending = self._continuations.get((li, slot))
                if pending is not None:
                    dense[row].copy_(pending)
        return dense * (slots >= 0)[:, None, None, None]

    def commit(self, layer, slots, dense):
        """Prompt/emitter commit, one sink extraction and MGS factorization."""
        from sglang.srt.mem_cache.gdn_factored_pool import factorize_dense

        li = self.layer_map[layer]
        if self._prefill_final is not None:
            finished = []
            for row, slot in enumerate(slots.tolist()):
                if self._prefill_final[slot]:
                    self._continuations.pop((li, slot), None)
                    finished.append(row)
                else:
                    self._continuations[li, slot] = dense[row].clone()
            if not finished:
                return
            if len(finished) != slots.numel():
                slots, dense = slots[finished], dense[finished]
        vb = self.vbar[li]
        a = torch.einsum("bhvk,hv->bhk", dense.float(), vb)
        a /= vb.square().sum(-1).clamp_min(1.0e-12)[None, :, None]
        residual = dense.float() - vb[None, :, :, None] * a[:, :, None, :]
        # GDN sees S^T: its orthonormal key basis becomes our value basis.
        zero = vb.new_zeros(self.heads, self.keys)
        _, q, c = factorize_dense(
            residual.transpose(-1, -2).contiguous(),
            zero,
            self.cfg.rank,
            self.cfg.capacity,
            torch.float32,
            omega=self.omega.expand(dense.shape[0], -1, -1, -1),
        )
        self.a[li, slots] = a
        if self.log_decay is not None:
            self.log_decay[li, slots] = 0
            self.column_exponent[li, slots] = 0
        self.basis[li, slots] = q.to(self.cfg.dtype)
        self.coeff[li, slots] = c.to(self.cfg.coeff_dtype)
        self.count[li].index_fill_(0, slots.long(), self.cfg.rank)

    def finish_prefill(self, slots):
        """Commit a continuation when the final chunk contains only its boundary token.

        Such requests skip the prompt pass, so no layer/emitter calls commit.
        This runs before the boundary decode graph, never inside decode.
        """
        if not self._continuations:
            return
        selected = slots.tolist()
        for layer, li in self.layer_map.items():
            pending = [s for s in selected if (li, s) in self._continuations]
            if pending:
                dense = torch.stack([self._continuations[li, s] for s in pending])
                indices = torch.tensor(pending, device=self.a.device, dtype=torch.long)
                self.commit(layer, indices, dense)
                for slot in pending:
                    self._continuations.pop((li, slot), None)

    def decode(self, layer, mixed, gate, beta, indices):
        from sglang.srt.layers.attention.linear.kernels.kda_factored import (
            packed_decode,
        )

        if not self.directions_bound:
            raise RuntimeError("KDA factored pool has no model sink directions")
        li = self.layer_map[layer.layer_id]
        return packed_decode(
            mixed,
            gate,
            beta,
            layer.A_log,
            layer.dt_bias,
            self.vbar[li],
            self.a[li],
            self.basis[li],
            self.coeff[li],
            self.count[li],
            indices,
            self.cfg.rank,
            self.cfg.window,
            lower_bound=layer.lower_bound,
            immediate=self.cfg.use_immediate,
            log_decay=self.log_decay[li] if self.log_decay is not None else None,
            column_exponent=(
                self.column_exponent[li] if self.column_exponent is not None else None
            ),
        )

    # Compatibility with the existing DUET model boundary hooks. The step
    # kernel owns cadence; these hooks must never schedule cuts or read counts.
    def before_graph(self, batch):
        pass

    def after_graph(self, batch):
        pass

    def mark_prefix(self, layer, slots):
        pass

    def note_prompt_end(self, slots):
        pass

    def mark_prompt_end(self, requests, slots):
        pass

    def slots(self, batch):
        return self.pool.get_mamba_indices(
            batch.req_pool_indices[: batch.batch_size]
        ).long()

    def flush(self, slots, **kwargs):
        pass

    def extend(self, *args, **kwargs):
        pass

    def prefill_boundary(self, batch):
        return nullcontext()
