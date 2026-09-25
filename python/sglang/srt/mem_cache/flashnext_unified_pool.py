"""Request-private deep KV backed by the shared prefix arena (#379).

Physical units are layer-pages, each carrying exact bf16 K/V and the indexer.
Shared virtual pages own seven QSA units and two latent payload units. Deep
virtual pages own five QSA units. Both draw from one physical free list.
"""
from contextlib import nullcontext
import os

import numpy as np
import torch

from sglang.srt.mem_cache.flashnext_latent_pool import FlashNextLatentPool
from sglang.srt.mem_cache.flashnext_unified_layout import UnifiedPageOwners, UnifiedPrivatePageOwners
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool, QSA_ROPE_STATE_LAYER_ID
from sglang.srt.mem_cache.flashnext_pd_pages import request_layer_pages


class MappedQSA:
    def page_mapping(self, layer_id):
        return self.physical_page_map[:, self._transfer_full_attention_id(layer_id)]

    def translate_locations(self, layer_id, locations, *, compressed=False):
        if not hasattr(self, 'physical_page_map'):
            return locations
        page = self.page_size // self.qsa_compress_ratio if compressed else self.page_size
        ids = locations.long()
        return self.page_mapping(layer_id)[ids // page].long() * page + ids % page

    def translate_page_table(self, layer_id, table):
        return self.page_mapping(layer_id)[table.long()].to(table.dtype)

    def set_kv_buffer(self, layer, loc, *args, **kwargs):
        loc = self.translate_locations(layer.layer_id, loc)
        return super().set_kv_buffer(layer, loc, *args, **kwargs)

    def set_qsa_compressed_k_buffer(self, layer_id, loc, values):
        loc = self.translate_locations(layer_id, loc, compressed=True)
        return super().set_qsa_compressed_k_buffer(layer_id, loc, values)


class MappedQSAPool(MappedQSA, QSATokenToKVPool):
    pass


class FlashNextUnifiedLatentPool(MappedQSA, FlashNextLatentPool):
    deep_pool_type = MappedQSAPool
    shared_arena = True
    transfer_requires_final_chunk = True
    wire_field_order = ('z', 'z_block_scale', 'z_scale', 'rms', 'spike_indices',
                        'spike_lengths', 'spike_values', 'token_ids')

    def set_kv_buffer(self, layer, loc, *args, **kwargs):
        # Emitters enter through the parent pool with private virtual locations.
        # Dispatch before MappedQSA translates; each location must be mapped once.
        if layer.layer_id >= 31:
            return self.deep.set_kv_buffer(layer, loc, *args, **kwargs)
        return super().set_kv_buffer(layer, loc, *args, **kwargs)

    def set_qsa_compressed_k_buffer(self, layer_id, loc, values):
        if layer_id >= 31:
            return self.deep.set_qsa_compressed_k_buffer(layer_id, loc, values)
        return super().set_qsa_compressed_k_buffer(layer_id, loc, values)

    def __init__(self, *, private_tokens, tp_rank, tp_size, req_to_token_pool, scheme_c=False, **kwargs):
        if not scheme_c or tp_size != 2 or kwargs['page_size'] != 64:
            raise ValueError('shared arena currently requires final Scheme C, TP2, page64')
        if kwargs.get('enable_kv_cache_copy') or kwargs.get('post_capture_active'):
            raise ValueError('shared arena does not yet support speculative moves or VMM transport')
        size = kwargs['size']
        # Construct only small metadata/sentinel pools, then attach the common
        # backing. Never allocate the old fixed private reserve, even transiently.
        small = dict(kwargs, size=64)
        super().__init__(private_tokens=64, tp_rank=tp_rank, tp_size=tp_size,
                         req_to_token_pool=req_to_token_pool, scheme_c=True, **small)
        self.request_state.transfer_enabled = False  # D receives complete KV, no boundary replay
        self.size = size
        pages = size // self.page_size
        self.arena = UnifiedPageOwners(pages * UnifiedPageOwners.shared_units)
        physical_tokens = (self.arena.capacity + 1) * self.page_size
        allocation = self.full_kv_pool
        # MNNVL must see an exportable allocation, as with the ordinary QSA pool.
        with (torch.cuda.use_mem_pool(allocation.custom_mem_pool)
              if allocation.enable_custom_mem_pool else nullcontext()):
            self.unified_k = torch.zeros((physical_tokens, 1, 256), dtype=torch.bfloat16, device=self.device)
            self.unified_v = torch.zeros_like(self.unified_k)
            self.unified_index = torch.zeros((physical_tokens // 4, 1, 128), dtype=torch.bfloat16, device=self.device)
        self.physical_page_map = torch.zeros((pages+1, 9), dtype=torch.int32, device=self.device)
        self.private_tokens = (self.arena.capacity // 5) * self.page_size
        self.deep.physical_page_map = torch.zeros((self.private_tokens // self.page_size+1, 5),
                                                  dtype=torch.int32, device=self.device)
        self.private = UnifiedPrivatePageOwners(self.private_tokens, self.page_size, self.arena)
        self.deep.size = self.private_tokens
        self._attach(self, 7, physical_tokens)
        self._attach(self.deep, 5, physical_tokens)
        # Scalar/escape formats are unchanged; the wire row occupies 1,972 B per
        # rank. Two K/V units provide 2,048 B, plus unused indexer space (128 B).
        self.latent_fields = [(name, self.latent[name].shape[1], self.latent[name].dtype)
                              for name in self.wire_field_order]
        self.latent = {}
        self.mem_usage = sum(self.get_kv_size_bytes()) / (1 << 30)

    def _attach(self, pool, layers, physical_tokens):
        full = pool.full_kv_pool
        full.size = physical_tokens - self.page_size
        full.num_pages = physical_tokens // self.page_size
        full.k_buffer = [self.unified_k] * layers
        full.v_buffer = [self.unified_v] * layers
        # Slot movement is deliberately unsupported, but pointer metadata must
        # still describe the actual buffers for inventory/diagnostics.
        full.k_data_ptrs = torch.tensor([self.unified_k.data_ptr()]*layers, dtype=torch.uint64, device=self.device)
        full.v_data_ptrs = torch.tensor([self.unified_v.data_ptr()]*layers, dtype=torch.uint64, device=self.device)
        full.data_ptrs = torch.cat((full.k_data_ptrs, full.v_data_ptrs))
        full.data_strides = torch.full((2*layers,), 512, dtype=torch.int64, device=self.device)
        pool.qsa_compressed_flat = self.unified_index
        pool.qsa_compressed_k_buffer_pool = [self.unified_index] * layers
        pool.qsa_compressed_capacity = physical_tokens // 4

    def prepare_request_mappings(self, host_indices):
        super().prepare_request_mappings(host_indices)
        if host_indices is None:
            return
        for pending, table in ((self.arena.pending_shared, self.physical_page_map),
                               (self.arena.pending_deep, self.deep.physical_page_map)):
            if pending:
                ids = torch.tensor(list(pending), dtype=torch.long, device=self.device)
                table[ids] = torch.tensor(list(pending.values()), dtype=torch.int32, device=self.device)
                pending.clear()

    def admission_units(self, req, pending=()):
        candidates = {r.rid:r for r in (*pending, req)}
        deep = shared = 0
        for r in candidates.values():
            bound = self.request_bound(r)
            deep += self.private.pages_needed(r.rid, bound) * 5
            allocated = max(len(r.prefix_indices), int(getattr(r.kv, 'kv_allocated_len', 0)))
            shared += max(0, (bound+63)//64 - (allocated+63)//64) * 9
        return deep + shared

    def can_admit(self, req, pending=()):
        return (super().can_admit(req, pending)
                and self.admission_units(req, pending) <= len(self.arena.free))

    def _payload_locations(self, locations):
        ids = locations.long()
        pages = self.physical_page_map[ids // 64, 7:9].long()
        return pages * 64 + (ids % 64)[:, None]

    def store_latent(self, locations, batch, token_ids):
        rank = self.tp_rank
        fields = dict(z=batch.z[:, rank*1024:(rank+1)*1024],
                      z_scale=batch.z_scale, rms=batch.rms,
                      spike_indices=batch.spike_indices[:, rank*294:(rank+1)*294],
                      spike_lengths=batch.spike_lengths,
                      z_block_scale=batch.z_block_scale.view(torch.uint8)[:, rank*128:(rank+1)*128],
                      spike_values=batch.spike_values[:, rank*256:(rank+1)*256],
                      token_ids=token_ids.reshape(-1,1).to(torch.int32))
        n = locations.numel()
        payload = torch.zeros((n, 2048), dtype=torch.uint8, device=self.device)
        offset = 0
        for name, width, dtype in self.latent_fields:
            size = width * dtype.itemsize
            payload[:, offset:offset+size] = fields[name].contiguous().view(torch.uint8).reshape(n,size)
            offset += size
        assert offset == 1972
        ids = self._payload_locations(locations)
        for buffer, part, col in ((self.unified_k,0,0),(self.unified_k,1,1),
                                   (self.unified_v,2,0),(self.unified_v,3,1)):
            buffer[ids[:,col]] = payload[:,part*512:(part+1)*512].contiguous().view(torch.bfloat16).reshape(n,1,256)

    def load_latent(self, locations):
        from sglang.srt.distributed import get_tp_group
        ids = self._payload_locations(locations)
        n = locations.numel()
        payload = torch.cat([buffer[ids[:,col]].contiguous().view(torch.uint8).reshape(n,512)
                             for buffer,col in ((self.unified_k,0),(self.unified_k,1),
                                                (self.unified_v,0),(self.unified_v,1))], -1)
        if os.environ.get('SGLANG_FLASHNEXT_REBUILD_GRAPH', '0') == '1':
            from .flashnext_rebuild_graph import unpack_payload
            group = get_tp_group()
            if group.world_size != 2:
                raise RuntimeError('shared arena TP ownership changed')
            return unpack_payload(payload, self.latent_fields,
                                  gathered=group.all_gather(payload, dim=-1))
        out = {};offset = 0
        for name,width,dtype in self.latent_fields:
            size = width*dtype.itemsize
            out[name] = payload[:,offset:offset+size].contiguous().view(dtype)
            offset += size
        group = get_tp_group()
        if group.world_size != 2:
            raise RuntimeError('shared arena TP ownership changed')
        for name in ('z','spike_indices','spike_values','z_block_scale'):
            dtype = out[name].dtype
            out[name] = group.all_gather(out[name].contiguous().view(torch.uint8),dim=-1).contiguous().view(dtype)
        out['z_block_scale'] = out['z_block_scale'].view(torch.float8_e4m3fn)
        return out

    def get_kv_size_bytes(self):
        if not hasattr(self, 'unified_k'):
            return super().get_kv_size_bytes()
        return self.unified_k.nbytes+self.unified_index.nbytes, self.unified_v.nbytes

    def get_latent_state_buf_infos(self):
        # Complete KV is materialized on P; latent and exact sink stay in P radix.
        return [], [], []

    def get_contiguous_buf_infos(self):
        return self._get_paged_state_buf_infos(
            [self.unified_k] * 12 + [self.unified_v] * 12, self.page_size)

    def get_kv_layer_ids(self):
        return list(range(3, 48, 4)) * 2

    def get_qsa_pending_state_buf_infos(self):
        return self._get_paged_state_buf_infos(
            [*self.qsa_key_state_buffer_pool, *self.deep.qsa_key_state_buffer_pool,
             self.qsa_rope_position_buffer], self.qsa_compress_ratio)

    def get_qsa_pending_state_layer_ids(self):
        return [*range(3, 48, 4), QSA_ROPE_STATE_LAYER_ID]

    def get_qsa_compressed_state_buf_infos(self):
        return self._get_paged_state_buf_infos([self.unified_index] * 12,
                                               self.qsa_compressed_page_size)

    def get_qsa_compressed_state_layer_ids(self):
        return list(range(3, 48, 4))

    def get_kv_transfer_pages(self, req, shared_pages, start_token=0, *, compressed=False):
        layers = request_layer_pages(self.arena.shared, self.arena.deep,
                                     self.private.owners[req.rid], shared_pages,
                                     start_token, self.page_size)
        return layers if compressed else np.concatenate((layers, layers), axis=0)
