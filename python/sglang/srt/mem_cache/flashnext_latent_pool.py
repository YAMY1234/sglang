"""Shared shallow KV/latent pages and separate request-private deep KV pages."""
from contextlib import nullcontext
import copy

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.flashnext_latent_layout import FlashNextLatentLayout, PrivatePageOwners
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

LATENT_SINK_LAYER_ID = (1 << 32) - 4
LATENT_BOUNDARY_LAYER_ID = (1 << 32) - 5


class LatentRequestState:
    """Exact sink follows shallow prefix COW; boundary is handoff-only state."""
    def __init__(self, size, device):
        self.transfer_enabled = True
        self.sink = torch.zeros((size + 1, 10240), dtype=torch.bfloat16, device=device)
        self.sink_valid = torch.zeros((size + 1, 1), dtype=torch.int32, device=device)
        self.boundary = torch.zeros_like(self.sink)
        self.boundary_position = torch.full((size + 1, 1), -1, dtype=torch.int64, device=device)

    def reset_slots(self, indices):
        self.sink[indices] = 0
        self.sink_valid[indices] = 0
        self.boundary[indices] = 0
        self.boundary_position[indices] = -1

    def copy_slots(self, src, dst):
        self.sink[dst] = self.sink[src]
        self.sink_valid[dst] = self.sink_valid[src]
        # A cached prefix cannot retain another request's D boundary.
        self.boundary[dst] = 0
        self.boundary_position[dst] = -1

    def get_cpu_slots(self, indices):
        return (self.sink[indices].cpu(), self.sink_valid[indices].cpu())

    def load_cpu_slots(self, data, indices):
        self.reset_slots(indices)
        self.sink[indices] = data[0].to(self.sink.device)
        self.sink_valid[indices] = data[1].to(self.sink.device)

    def iter_transfer_state_entries(self):
        if not self.transfer_enabled:
            return
        for name, tensor, layer in (
            ("latent_sink", self.sink, LATENT_SINK_LAYER_ID),
            ("latent_sink_valid", self.sink_valid, LATENT_SINK_LAYER_ID),
            ("latent_boundary", self.boundary, LATENT_BOUNDARY_LAYER_ID),
            ("latent_boundary_position", self.boundary_position, LATENT_BOUNDARY_LAYER_ID),
        ):
            yield name, tensor, None, layer


class FlashNextLatentPool(QSATokenToKVPool):
    """Only the shallow parent pool is exposed to radix and Mooncake KV APIs."""
    deep_pool_type = QSATokenToKVPool
    def __init__(self, *, private_tokens, tp_rank, tp_size, req_to_token_pool, scheme_c=False, **kwargs):
        layer_ids = kwargs.pop("full_attention_layer_ids")
        if layer_ids != list(range(3, 48, 4)):
            raise ValueError("v3 requires the complete Flash-Next QSA layer set")
        self.deep = None
        self.latent = {}
        self.layout = FlashNextLatentLayout(tp_size, scheme_c=scheme_c)
        self.tp_rank = tp_rank
        self.private_tokens = private_tokens
        self.request_pool = req_to_token_pool
        req_to_token_pool.flashnext_latent_pool = self
        super().__init__(full_attention_layer_ids=[l for l in layer_ids if l < 31], **kwargs)
        deep_args = dict(kwargs, size=private_tokens)
        self.deep = self.deep_pool_type(full_attention_layer_ids=[l for l in layer_ids if l >= 31], **deep_args)
        # Logical positions are layer-independent; materialization and the
        # boundary restore the same pending group as the shallow layers.
        self.deep.qsa_rope_position_buffer = self.qsa_rope_position_buffer
        self.deep_req_to_token = torch.zeros_like(req_to_token_pool.req_to_token)
        self.deep_request_pool = copy.copy(req_to_token_pool)
        self.deep_request_pool.req_to_token = self.deep_req_to_token
        self.private = PrivatePageOwners(private_tokens, self.page_size)
        self.request_slots = {}
        self.slot_requests = {}
        self.prompt_lengths = {}
        self.pending_mappings = {}
        self.materialized = set()
        self.stats = dict(materializations=0, materialized_tokens=0, materialization_ms=0.0)
        allocation = self.full_kv_pool
        with (allocation.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
              torch.cuda.use_mem_pool(allocation.custom_mem_pool) if allocation.enable_custom_mem_pool else nullcontext()):
            shapes = {"z": (self.layout.local_rank, torch.uint8),
                      "z_scale": (1, torch.float32), "rms": (1, torch.float32),
                      "spike_indices": (self.layout.local_sparse, torch.int16),
                      "spike_values": (self.layout.local_sparse, torch.float32),
                      "token_ids": (1, torch.int32)}
            if scheme_c:
                shapes.update(z=(self.layout.local_rank // 2, torch.uint8),
                              z_block_scale=(self.layout.local_rank // 16, torch.uint8),
                              spike_indices=(self.layout.local_gap_bytes, torch.uint8),
                              spike_lengths=(1, torch.int16),
                              spike_values=(self.layout.local_sparse, torch.bfloat16))
            self.latent = {name: torch.zeros((self.size + self.page_size, width), dtype=dtype, device=self.device)
                           for name, (width, dtype) in shapes.items()}
            self.request_state = LatentRequestState(self.mamba_pool.size, self.device)
        self.mamba_pool.register_slot_state(self.request_state)
        # Deep recurrent states are active-request state, never prefix state.
        self.mamba_pool.prefix_layer_limit = 48 if scheme_c else 31
        factored = getattr(req_to_token_pool, "factored_gdn_pool", None)
        if factored is None:
            raise ValueError("v3 requires the r8 GDN pool")
        factored.prefix_layer_limit = 48 if scheme_c else 31
        k, v = self.get_kv_size_bytes()
        self.mem_usage = (k + v) / (1 << 30)

    def request_bound(self, req):
        bound = len(req.origin_input_ids) + req.sampling_params.max_new_tokens
        if bound > self.deep_req_to_token.shape[1]:
            raise ValueError("request exceeds the private page table context bound")
        return bound

    def can_admit(self, req, pending=()):
        return self.private.can_admit(req.rid, self.request_bound(req),
            [(r.rid, self.request_bound(r)) for r in pending])

    def bind_request(self, req):
        slot = int(req.kv.req_pool_idx)
        if req.rid in self.request_slots:
            if self.request_slots[req.rid] != slot:
                raise RuntimeError("active private reservation changed request slot")
            return
        if slot in self.slot_requests:
            raise RuntimeError("request slot reused before private deep release")
        pages = self.private.bind(req.rid, self.request_bound(req))
        self.pending_mappings[slot] = pages
        self.request_slots[req.rid] = slot
        self.slot_requests[slot] = req.rid
        self.prompt_lengths[slot] = len(req.origin_input_ids)

    def release_request(self, req):
        slot = self.request_slots.pop(req.rid, None)
        if slot is None:
            return
        if self.slot_requests.pop(slot) != req.rid:
            raise RuntimeError("private request ownership mismatch")
        self.private.release(req.rid)
        self.materialized.discard(slot)
        self.prompt_lengths.pop(slot)
        self.pending_mappings.pop(slot, None)

    def prepare_request_mappings(self, host_indices):
        """Commit page tables on the forward stream after any old slot readers."""
        if host_indices is None:
            return  # graph capture only uses reserved request slot zero
        for slot in host_indices.tolist():
            pages = self.pending_mappings.pop(int(slot), None)
            if pages is None:
                continue
            ids = torch.tensor(pages, dtype=torch.int64, device=self.device)
            locs = (ids[:, None] * self.page_size + torch.arange(self.page_size, device=self.device)).flatten()
            n = min(locs.numel(), self.deep_req_to_token.shape[1])
            self.deep_req_to_token[slot, :n] = locs[:n].to(self.deep_req_to_token.dtype)
            ring = torch.arange(self.qsa_compress_ratio, device=self.device) + slot * self.qsa_compress_ratio
            for tensor in self.deep.qsa_key_state_buffer_pool:
                tensor[ring] = 0
            self.deep.qsa_rope_position_buffer[ring] = 0

    def deep_batch(self, fb):
        if getattr(fb, "flashnext_private_locations", False):
            return fb
        batch = copy.copy(fb)
        if fb.forward_mode.is_decode_or_idle():
            rows = fb.req_pool_indices.long()
            positions = fb.seq_lens.long() - 1
        else:
            lengths = torch.tensor(fb.extend_seq_lens_cpu, device=self.device)
            rows = torch.repeat_interleave(fb.req_pool_indices.long(), lengths)
            positions = fb.positions.long()
        batch.out_cache_loc = self.deep_req_to_token[rows, positions.clamp_min(0)].long()
        batch.flashnext_private_locations = True
        return batch

    def store_latent(self, locations, batch, token_ids):
        loc = locations.long()
        rz, rs = self.layout.local_rank, self.layout.local_sparse
        if self.layout.scheme_c:
            rz //= 2
        zslice = slice(self.tp_rank * rz, (self.tp_rank + 1) * rz)
        sslice = slice(self.tp_rank * rs, (self.tp_rank + 1) * rs)
        self.latent["z"][loc] = batch.z.view(torch.uint8)[:, zslice]
        self.latent["z_scale"][loc] = batch.z_scale
        self.latent["rms"][loc] = batch.rms
        if self.layout.scheme_c:
            gb = self.layout.local_gap_bytes
            self.latent["spike_indices"][loc] = batch.spike_indices[:, self.tp_rank*gb:(self.tp_rank+1)*gb]
            self.latent["spike_lengths"][loc] = batch.spike_lengths
            bs = self.layout.local_rank // 16
            self.latent["z_block_scale"][loc] = batch.z_block_scale.view(torch.uint8)[:, self.tp_rank*bs:(self.tp_rank+1)*bs]
        else:
            self.latent["spike_indices"][loc] = batch.spike_indices[:, sslice]
        self.latent["spike_values"][loc] = batch.spike_values[:, sslice]
        self.latent["token_ids"][loc] = token_ids.reshape(-1, 1).to(torch.int32)

    def load_latent(self, locations):
        from sglang.srt.distributed import get_tp_group
        out = {k: v.index_select(0, locations.long()) for k, v in self.latent.items()}
        group = get_tp_group()
        if group.world_size != self.layout.tp_size:
            raise RuntimeError("latent TP ownership differs from the materialization group")
        sharded = ["z", "spike_indices", "spike_values"]
        if self.layout.scheme_c:
            sharded.append("z_block_scale")
        for name in sharded:
            dtype = out[name].dtype
            # Byte views support int16/fp8 on every supported NCCL build.
            out[name] = group.all_gather(out[name].contiguous().view(torch.uint8), dim=-1).contiguous().view(dtype)
        if self.layout.scheme_c:
            out["z_block_scale"] = out["z_block_scale"].view(torch.float8_e4m3fn)
        else:
            out["z"] = out["z"].view(torch.float8_e4m3fn)
        return out

    def get_latent_state_buf_infos(self):
        return self._get_paged_state_buf_infos(list(self.latent.values()), self.page_size)

    def get_latent_state_layer_ids(self):
        return list(range(len(self.latent)))

    def get_kv_size_bytes(self):
        k, v = super().get_kv_size_bytes()
        if self.deep is not None:
            dk, dv = self.deep.get_kv_size_bytes()
            k += dk
            v += dv
        return k + sum(t.nbytes for t in self.latent.values()), v

    def _layer_pool(self, layer_id):
        return self.deep if layer_id >= 31 else self

    def get_key_buffer(self, layer_id, *args, **kwargs):
        return (self.deep.get_key_buffer(layer_id, *args, **kwargs) if layer_id >= 31
                else super().get_key_buffer(layer_id, *args, **kwargs))

    def get_value_buffer(self, layer_id, *args, **kwargs):
        return (self.deep.get_value_buffer(layer_id, *args, **kwargs) if layer_id >= 31
                else super().get_value_buffer(layer_id, *args, **kwargs))

    def get_kv_buffer(self, layer_id):
        return self.deep.get_kv_buffer(layer_id) if layer_id >= 31 else super().get_kv_buffer(layer_id)

    def set_kv_buffer(self, layer, *args, **kwargs):
        if layer.layer_id >= 31:
            return self.deep.set_kv_buffer(layer, *args, **kwargs)
        return super().set_kv_buffer(layer, *args, **kwargs)

    def get_qsa_key_state_buffer(self, layer_id):
        return (self.deep.get_qsa_key_state_buffer(layer_id) if layer_id >= 31
                else super().get_qsa_key_state_buffer(layer_id))

    def get_qsa_compressed_k_buffer(self, layer_id):
        return (self.deep.get_qsa_compressed_k_buffer(layer_id) if layer_id >= 31
                else super().get_qsa_compressed_k_buffer(layer_id))
