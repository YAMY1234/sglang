"""SGLang adapter for the x256 physical code/exact page store.

Virtual page lifetimes remain owned by SGLang's allocator/radix locks. The
store's named readers additionally protect exact views until prefix handoff or
the accepted-token age boundary. Scheduler hooks run outside CUDA graphs.
"""

from collections import defaultdict
from contextlib import nullcontext

import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.code import QSACodeLayout, load_x256_weights
from sglang.srt.layers.attention.qsa.code_kernel import write_exact_tokens
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.qsa_code_pool import QSAPrefixPageStore, _Page


class QSACodeServingPool(KVCache):
    def __init__(
        self,
        *,
        size,
        page_size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        device,
        enable_memory_saver,
        release,
        layer_ids,
        tp_rank,
        tp_size,
        exact_fraction=0.25,
        num_request_slots=4,
        enable_kv_cache_copy=False,
        quant_method=None,
        post_capture_active=False,
    ):
        if dtype != torch.bfloat16 or head_dim != 256 or head_num != 2 // tp_size:
            raise ValueError("x256 code pool requires bf16, head_dim256 and TP1/TP2")
        if (
            post_capture_active
            or getattr(quant_method, "name", "unquantized") != "unquantized"
        ):
            raise ValueError(
                "x256 code pages require static unquantized physical backing"
            )
        if not 0 < exact_fraction < 1 or page_size != 64:
            raise ValueError(
                "x256 currently uses page64 and an exact fraction in (0,1)"
            )
        super().__init__(size, page_size, dtype, layer_num, device, enable_memory_saver)
        self.head_num, self.head_dim = head_num, head_dim
        self.kv_cache_layout = "qsa_codes"
        self.quant_method = None
        self.layout = QSACodeLayout()
        loaded = load_x256_weights(
            release,
            layer_ids=layer_ids,
            tp_rank=tp_rank,
            tp_size=tp_size,
            device=device,
        )
        self.model_layer_ids = tuple(layer_ids)
        weights = {i: loaded[layer] for i, layer in enumerate(layer_ids)}
        pages = size // page_size
        exact_pages = int(pages * exact_fraction)
        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext(),
        ):
            self.store = QSAPrefixPageStore(
                code_pages=pages - exact_pages,
                exact_pages=exact_pages,
                page_size=page_size,
                layout=self.layout,
                weights=weights,
                device=device,
            )
        self.prefix_lengths = torch.zeros(
            num_request_slots, dtype=torch.int32, device=device
        )
        self._bound_prompts = {}
        self._finalize_allocation_log(size)

    def reserve_pages(self, virtual_ids):
        """Reserve physical exact backing for virtual pages chosen by allocator."""
        if virtual_ids.numel() == 0:
            return
        ids = virtual_ids.tolist()
        if len(ids) > self.store.available_exact_pages:
            raise MemoryError("QSA exact page capacity exhausted")
        if len(set(ids)) != len(ids) or any(v in self.store.pages for v in ids):
            raise ValueError("QSA virtual page is already allocated")
        physical = [self.store._free_exact.pop() for _ in ids]
        for virtual, exact in zip(ids, physical, strict=True):
            self.store.pages[virtual] = _Page(exact=exact, owners={"pending": "exact"})
        allocated = set(ids)
        self.store._free_virtual = [
            v for v in self.store._free_virtual if v not in allocated
        ]
        if ids:
            self.store.exact_page[virtual_ids.long()] = torch.tensor(
                physical, dtype=torch.int32, device=self.device
            )

    def free_pages(self, virtual_ids):
        """Called only when radix locks say the virtual pages have no readers."""
        grouped = defaultdict(list)
        for virtual in virtual_ids.tolist():
            if virtual not in self.store.pages:
                raise ValueError("QSA double free of virtual page")
            for owner in self.store.pages[virtual].owners:
                grouped[owner].append(virtual)
        for owner, pages in grouped.items():
            self.store.release(pages, owner=owner)

    def clear(self):
        if self.store.pages:
            ids = torch.tensor(
                list(self.store.pages), dtype=torch.int64, device=self.device
            )
            self.free_pages(ids)
        self.store._next_query.clear()
        self.store.delay_histogram.clear()
        self.store.conversion_deferred.clear()
        self.prefix_lengths.zero_()
        self._bound_prompts.clear()

    @staticmethod
    def request_owner(req):
        return f"request:{id(req)}"

    def _request_pages(self, req, req_to_token_pool, length):
        if not length:
            return []
        slots = req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : length : self.page_size
        ]
        return (slots // self.page_size).tolist()

    def bind_request(self, req, req_to_token_pool):
        owner = self.request_owner(req)
        prompt = (req.kv.req_pool_idx, max(0, len(req.origin_input_ids) - 1))
        if self._bound_prompts.get(owner) != prompt:
            self.prefix_lengths[prompt[0]] = prompt[1]
            self._bound_prompts[owner] = prompt
        for virtual in self._request_pages(
            req, req_to_token_pool, req.kv.kv_allocated_len
        ):
            page = self.store.pages[virtual]
            if "pending" in page.owners:
                del page.owners["pending"]
                page.owners[owner] = "exact"
            elif not page.code and owner not in page.owners:
                self.store.acquire_exact([virtual], owner=owner)

    def commit_request(self, req, req_to_token_pool, length):
        """Call after a successful forward or at its next scheduler boundary."""
        owner = self.request_owner(req)
        self.bind_request(req, req_to_token_pool)
        pages = self._request_pages(req, req_to_token_pool, length)
        ids, lengths = [], []
        for i, virtual in enumerate(pages):
            page = self.store.pages[virtual]
            if page.owners.get(owner) == "exact" and not page.code:
                ids.append(virtual)
                lengths.append(min(self.page_size, length - i * self.page_size))
        self.store.commit_serving_writes(ids, lengths, owner=owner)
        return pages

    def publish_prefix(self, req, req_to_token_pool, length, *, encode_length=None):
        pages = self.commit_request(req, req_to_token_pool, length)
        if encode_length is None:
            encode_length = length
        if not 0 <= encode_length <= length:
            raise ValueError("QSA prefix encoding boundary exceeds committed writes")
        full = pages[: encode_length // self.page_size]
        if not full:
            return
        owner = self.request_owner(req)
        # Already-coded radix hits need no work and retain the same representation.
        owned = [p for p in full if self.store.pages[p].owners.get(owner) == "exact"]
        if not owned:
            return
        temporary = object()
        self.store.acquire_prefix(owned, owner=temporary)
        for page in owned:
            self.store.pages[page].owners[owner] = "code"
        self.store.release(owned, owner=temporary)

    def prepare_decode(self, reqs, req_to_token_pool):
        for req in reqs:
            committed = req.kv.kv_committed_len
            pages = self.commit_request(req, req_to_token_pool, committed)
            owner = self.request_owner(req)
            generated, ends = [], []
            for i, virtual in enumerate(pages[: committed // self.page_size]):
                page = self.store.pages[virtual]
                if page.owners.get(owner) == "exact":
                    generated.append(virtual)
                    ends.append((i + 1) * self.page_size - 1)
            self.store.track_generated_pages(generated, ends, owner=owner)
            self.store.advance_generation(
                owner=owner,
                next_query_position=committed,
                committed_position=committed - 1,
            )
        if self.store.convert_aged_pages():
            self.write_audit()

    def finish_request(self, req):
        owner = self.request_owner(req)
        self.store._next_query.pop(owner, None)
        self._bound_prompts.pop(owner, None)
        self.write_audit()

    def write_audit(self):
        """Optional local benchmark artifact; never part of an inference response."""
        import json
        import os
        from pathlib import Path

        directory = os.environ.get("SGLANG_QSA_CODE_AUDIT_DIR")
        if not directory:
            return
        Path(directory).mkdir(parents=True, exist_ok=True)
        data = self.store.allocation_bytes()
        data.update(
            prefix_length_table=self.prefix_lengths.nbytes,
            virtual_token_capacity=self.size,
            exact_page_capacity=self.store.exact[0][0].shape[0] - 1,
            code_page_capacity=self.store.codes[0].rotary.shape[0] - 1,
            allocated_virtual_pages=len(self.store.pages),
            free_exact_pages=self.store.available_exact_pages,
            free_code_pages=self.store.available_code_pages,
            delay_histogram=dict(self.store.delay_histogram),
            conversion_deferred=dict(self.store.conversion_deferred),
        )
        (Path(directory) / f"pool-{os.getpid()}.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )

    def get_key_buffer(self, layer_id):
        # Shape/physical-buffer discovery only; code-aware backends use store.
        return self.store.exact[layer_id][0].flatten(0, 1)

    def get_value_buffer(self, layer_id):
        return self.store.exact[layer_id][1].flatten(0, 1)

    def get_kv_buffer(self, layer_id):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self, layer, loc, cache_k, cache_v, *args, layer_id_override=None, **kwargs
    ):
        layer_id = layer.layer_id if layer_id_override is None else layer_id_override
        write_exact_tokens(cache_k, cache_v, loc, self.store, layer_id)

    def get_kv_size_bytes(self):
        k = v = 0
        for layer, (ek, ev) in self.store.exact.items():
            code = self.store.codes[layer]
            k += ek.nbytes + code.rotary.nbytes + code.key.nbytes
            v += ev.nbytes + code.value.nbytes
        return k, v

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "QSA code pages need representation-aware PD transfer"
        )

    def move_kv_cache(self, tgt_loc, src_loc):
        # Accepted speculative tokens are always younger than 256; copy their
        # exact representation. Advanced indexing gathers before the writes.
        src_page = self.store.exact_page[src_loc // self.page_size].long()
        dst_page = self.store.exact_page[tgt_loc // self.page_size].long()
        src = src_page * self.page_size + src_loc % self.page_size
        dst = dst_page * self.page_size + tgt_loc % self.page_size
        for layer in self.store.exact:
            k, v = self.get_kv_buffer(layer)
            k[dst] = k[src]
            v[dst] = v[src]
