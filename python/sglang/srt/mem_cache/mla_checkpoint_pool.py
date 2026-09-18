"""Shared P29 prompt checkpoint plus exact native MLA continuation slots.

Experimental TP1/page1 AGG pool. Logical slots survive compression and radix hits;
only their native slab reservation is returned. Allocation and reclamation run on
one stream with overlap scheduling disabled. No per-layer prompt c is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, unwrap_write_loc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLACheckpointConfig:
    basis: str
    rank: int = 1024
    sparse: int = 128
    native_fraction: float = .5
    first_layer: int = 29
    layers: int = 47
    hidden: int = 2048

    @classmethod
    def from_env(cls):
        raw = os.environ.get("SGLANG_MLA_CHECKPOINT", "")
        if not raw:
            return None
        fields = dict(part.split("=", 1) for part in raw.split(","))
        unknown = fields.keys() - {"basis", "rank", "sparse", "native_fraction"}
        if unknown or not fields.get("basis"):
            raise ValueError(f"MLA checkpoint needs basis=path; unknown fields {unknown}")
        cfg = cls(basis=fields["basis"], rank=int(fields.get("rank", 1024)),
                  sparse=int(fields.get("sparse", 128)),
                  native_fraction=float(fields.get("native_fraction", .5)))
        if cfg.rank not in (512, 1024, 1536, 2048):
            raise ValueError("rank must be 512/1024/1536/2048")
        if cfg.sparse < 0 or cfg.sparse > 2048 or cfg.sparse % 8:
            raise ValueError("sparse must be a multiple of 8 and in [0,2048]")
        if not 0 < cfg.native_fraction <= 1:
            raise ValueError("native_fraction must be in (0,1]")
        return cfg

    @property
    def compact_bytes(self):
        # z + packed index + fp8 residual + scale + a/rho + position + native map.
        return 2*self.rank + self.sparse*5//2 + 4 + 4*(1+self.layers-self.first_layer) + 4 + 8

    @property
    def allocation_bytes_per_token(self):
        return self.first_layer*1152 + self.compact_bytes + 256 + math.ceil(
            self.native_fraction*(self.layers-self.first_layer)*1152)

    def validate_runtime(self, kvc):
        from sglang.srt.runtime_context import get_parallel, get_schedule, get_disagg, get_memory, get_spec
        hf = kvc.model_config.hf_config
        ts = getattr(hf, "twinstar", {})
        if (getattr(hf, "model_type", None) != "glm4_moe_lite"
            or hf.num_hidden_layers != 47 or hf.hidden_size != 2048
            or ts.get("p_layers") != list(range(29))
            or ts.get("emitters") != list(range(29, 47)) or ts.get("bridge", 0)):
            raise ValueError("MLA checkpoint requires the unbridged GLM-4.7-Flash eo P29 export")
        if (get_parallel().attn_tp_size != 1 or get_parallel().pp_size != 1
            or get_parallel().dcp_enabled or get_schedule().page_size != 1
            or not get_schedule().disable_overlap_schedule
            or get_disagg().disaggregation_mode != "null"
            or get_memory().enable_hierarchical_cache or get_memory().enable_unified_memory
            or kvc.kv_cache_dtype != torch.bfloat16):
            raise ValueError("MLA checkpoint requires TP1/PP1/page1/bf16 AGG, overlap/HiCache/unified disabled")
        if kvc.server_args.attention_backend != "triton" or kvc.server_args.speculative_algorithm:
            raise ValueError("MLA checkpoint requires triton attention and no speculative decoding")


def pack12(indices):
    a, b = indices.long().reshape(*indices.shape[:-1], indices.shape[-1]//2, 2).unbind(-1)
    return torch.stack((a & 255, (a >> 8) | ((b & 15) << 4), b >> 4), -1).flatten(-2).to(torch.uint8)


def unpack12(packed):
    a, b, c = packed.long().reshape(*packed.shape[:-1], packed.shape[-1]//3, 3).unbind(-1)
    return torch.stack((a | ((b & 15) << 8), (b >> 4) | (c << 4)), -1).flatten(-2)


class MLACheckpointPool(MLATokenToKVPool):
    def __init__(self, *args, checkpoint_config, **kwargs):
        self.checkpoint_config = checkpoint_config
        self.folded = {}
        self.allocator = None
        super().__init__(*args, **kwargs)
        logger.info("MLA checkpoint pool logical=%d native=%d payload=%d B/token total=%d B; native reservation included",
                    self.size, self.native_size, checkpoint_config.compact_bytes, self.get_kv_size_bytes())

    def _create_buffers(self):
        cfg = self.checkpoint_config
        assert self.start_layer == 0 and self.layer_num == cfg.layers and self.page_size == 1
        assert self.dtype == torch.bfloat16 and self.kv_cache_dim == 576
        self.native_size = max(1, math.floor(self.size*cfg.native_fraction))
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.kv_buffer = [torch.zeros(
                (self.size+1 if l < cfg.first_layer else self.native_size+1, 1, 576),
                dtype=self.dtype, device=self.device) for l in range(self.layer_num)]
            self.z = torch.zeros((self.size+1, cfg.rank), dtype=self.dtype, device=self.device)
            self.indices = torch.zeros((self.size+1, cfg.sparse*3//2), dtype=torch.uint8, device=self.device)
            self.values = torch.zeros((self.size+1, cfg.sparse), dtype=torch.uint8, device=self.device)
            self.residual_scale = torch.zeros(self.size+1, dtype=torch.float32, device=self.device)
            self.norms = torch.ones((self.size+1, 1+cfg.layers-cfg.first_layer), dtype=torch.float32, device=self.device)
            self.positions = torch.zeros(self.size+1, dtype=torch.int32, device=self.device)
            self.native_of = torch.full((self.size+1,), -1, dtype=torch.int64, device=self.device)
            self.native_of[0] = 0
            self.rope_scratch = torch.zeros((self.size+1, 64), dtype=torch.float32, device=self.device)
        self.basis = self.mean = None

    def get_kv_size_bytes(self):
        tensors = self.kv_buffer + [self.z, self.indices, self.values, self.residual_scale,
                                     self.norms, self.positions, self.native_of, self.rope_scratch]
        return sum(t.numel()*t.element_size() for t in tensors)

    def get_key_buffer(self, layer_id):
        if layer_id >= self.checkpoint_config.first_layer:
            raise RuntimeError("deep checkpoint pool needs mixed attention; no materialized key buffer")
        return super().get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id):
        if layer_id >= self.checkpoint_config.first_layer:
            raise RuntimeError("deep checkpoint pool needs mixed attention; no materialized value buffer")
        return super().get_value_buffer(layer_id)

    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, layer_id_override=None):
        loc, _, _ = unwrap_write_loc(loc_info)
        lid = layer.layer_id if layer_id_override is None else layer_id_override
        if lid < self.checkpoint_config.first_layer:
            return super().set_kv_buffer(layer, loc_info, cache_k, cache_v, layer_id_override)
        self.kv_buffer[lid][self.native_of[loc]] = cache_k.to(self.dtype).view(-1, 1, 576)

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope, layer_id_override=None):
        lid = layer.layer_id if layer_id_override is None else layer_id_override
        if lid < self.checkpoint_config.first_layer:
            return super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope, layer_id_override)
        buf = self.kv_buffer[lid]
        native = self.native_of[loc]
        buf[native, :, :512] = cache_k_nope.to(self.dtype).view(-1, 1, 512)
        buf[native, :, 512:] = cache_k_rope.to(self.dtype).view(-1, 1, 64)

    def register_emitters(self, emitters):
        if self.basis is not None:
            return
        cfg = self.checkpoint_config
        state = torch.load(cfg.basis, map_location="cpu", weights_only=True)
        self.basis = state["var"][:, :cfg.rank].to(self.device).float().contiguous()
        self.mean = state["mean"].to(self.device).float().contiguous()
        if cfg.rank == cfg.hidden:
            # Full-h absorption diagnostic: no PCA or bf16 coordinate rotation.
            self.basis = torch.eye(cfg.hidden, device=self.device)
            self.mean = torch.zeros(cfg.hidden, device=self.device)
        assert self.basis.shape == (cfg.hidden, cfg.rank)
        for lid in range(cfg.first_layer, cfg.layers):
            emitter = emitters[str(lid)]
            a = emitter.kv_a_proj_with_mqa.weight.float() * emitter.input_layernorm.weight.float()[None, :]
            p = a[:512] * emitter.kv_a_layernorm.weight.float()[:, None]
            self.folded[lid] = {"a": a, "p": p.contiguous(),
                "pe": (p @ self.basis).contiguous(), "pm": p @ self.mean,
                "re": (a[512:] @ self.basis).contiguous(), "rm": a[512:] @ self.mean,
                "eps_h": emitter.input_layernorm.variance_epsilon,
                "eps_c": emitter.kv_a_layernorm.variance_epsilon}
        logger.info("MLA checkpoint folded weights registered: %d layers", len(self.folded))

    def encode(self, h, positions, loc, exact_mask):
        """Keep exact emitter c for sink/boundary; compact all remaining prompt slots."""
        cfg = self.checkpoint_config
        loc = loc[~exact_mask]
        if loc.numel() == 0:
            return  # e.g. a two-token prompt contains only sink and boundary
        h = h[~exact_mask].float()
        positions = positions[~exact_mask]
        z = ((h-self.mean) @ self.basis).to(self.dtype)
        self.z[loc] = z
        if cfg.sparse:
            residual = h - (z.float() @ self.basis.T + self.mean)
            indices = residual.abs().topk(cfg.sparse, dim=-1, sorted=False).indices
            values = residual.gather(-1, indices)
            scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-30)/448
            self.indices[loc] = pack12(indices)
            self.values[loc] = (values/scale).to(torch.float8_e4m3fn).view(torch.uint8)
            self.residual_scale[loc] = scale[:, 0]
        a = (h.square().mean(-1)+self.folded[cfg.first_layer]["eps_h"]).sqrt()
        self.norms[loc, 0] = a
        for lid, w in self.folded.items():
            c = (h @ w["a"][:512].T) / a[:, None]
            self.norms[loc, 1+lid-cfg.first_layer] = (c.square().mean(-1)+w["eps_c"]).sqrt()
        self.positions[loc] = positions.to(torch.int32)
        self.allocator.release_native(loc)

    def get_contiguous_buf_infos(self):
        raise NotImplementedError("checkpoint pool has no PD transfer ABI")

    def get_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("checkpoint pool HiCache is not enabled")

    def load_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("checkpoint pool HiCache is not enabled")

    def move_kv_cache(self, *args, **kwargs):
        raise NotImplementedError("checkpoint pool speculative relocation is not enabled")


class MLACheckpointAllocator(TokenToKVPoolAllocator):
    requires_repeated_eviction = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kvcache.allocator = self

    def clear(self):
        super().clear()
        pool = self._kvcache
        self.native_free = torch.arange(1, pool.native_size+1, dtype=torch.int64, device=self.device)
        pool.native_of.fill_(-1)
        pool.native_of[0] = 0

    def logical_available_size(self):
        return super().available_size()

    def available_size(self):
        return min(self.logical_available_size(), len(self.native_free))

    def available_with_cache(self, tree_cache):
        """Admission must count reclaimable native slots, not compact tokens.

        The experimental page1 radix path walks unlocked nodes and performs one
        GPU reduction. This conservative admission cost is part of served perf.
        Logical cache statistics continue to use the stock token count.
        """
        logical = tree_cache.evictable_size()
        if not logical:
            return self.available_size()
        values = []
        stack = list(tree_cache.root_node.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if hasattr(node, "component_data"):
                from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE
                component = node.component_data[BASE_COMPONENT_TYPE]
            else:
                component = node
            if component.lock_ref == 0 and component.value is not None and component.value.numel():
                values.append(component.value)
        native = 0
        if values:
            loc = torch.cat(values).long()
            native = int((self._kvcache.native_of[loc] > 0).sum().item())
        return min(self.logical_available_size()+logical, len(self.native_free)+native)

    def alloc(self, need_size):
        if need_size > self.available_size():
            return None
        loc = super().alloc(need_size)
        if loc is None:
            return None
        native = self.native_free[:need_size]
        self.native_free = self.native_free[need_size:]
        self._kvcache.native_of[loc] = native
        return loc

    def release_native(self, loc):
        if loc.numel() == 0:
            return
        native = self._kvcache.native_of[loc]
        native = native[native > 0]
        self.native_free = torch.cat((self.native_free, native))
        self._kvcache.native_of[loc] = -1

    def free(self, free_index):
        if self.free_group is None:
            self.release_native(free_index)
        super().free(free_index)

    def verify_byte_accounting(self):
        free = self.native_free
        active = self._kvcache.native_of[1:]
        active = active[active > 0]
        all_native = torch.cat((free, active))
        expected = torch.arange(1, self._kvcache.native_size+1, device=self.device)
        if all_native.numel() != expected.numel() or not torch.equal(all_native.sort().values, expected):
            return ["MLA checkpoint native slots leaked or aliased"]
        return []
