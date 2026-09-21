"""SGLang adapter for the x256 physical code/exact page store.

Virtual page lifetimes remain owned by SGLang's allocator/radix locks. The
store's named readers additionally protect exact views until prefix handoff or
the accepted-token age boundary. Scheduler hooks run outside CUDA graphs.
"""

from collections import defaultdict
from contextlib import nullcontext
import os

import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.code import load_x256_weights, serving_code_layout
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
        self.layout = serving_code_layout()
        tuning = os.environ.get("SGLANG_QSA_CODE_READ_TUNING", "0")
        if tuning not in ("0", "1"):
            raise ValueError("SGLANG_QSA_CODE_READ_TUNING must be 0 or 1")
        self.read_tuned = tuning == "1"
        self.exact_fraction = exact_fraction
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
        self._active_requests = {}
        self._prefix_reservations = {}
        self.indexer_bytes = {}
        self.workspace_bytes = {}
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
            if not any(owner in page.owners for page in self.store.pages.values()):
                self.store._next_query.pop(owner, None)
                prompt = self._bound_prompts.pop(owner, None)
                if prompt is not None:
                    self._active_requests.pop(prompt[0], None)
                self._prefix_reservations.pop(owner, None)
        self.store.reserved_code_pages = sum(self._prefix_reservations.values())

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
        self._active_requests.clear()
        self._prefix_reservations.clear()
        self.store.reserved_code_pages = 0

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
        self._active_requests[prompt[0]] = req
        if self._bound_prompts.get(owner) != prompt:
            self.prefix_lengths[prompt[0]] = prompt[1]
            self._bound_prompts[owner] = prompt
            allocated_prefix = self._request_pages(
                req, req_to_token_pool, min(prompt[1], req.kv.kv_allocated_len)
            )[: prompt[1] // self.page_size]
            coded = sum(bool(self.store.pages[p].code) for p in allocated_prefix)
            self._prefix_reservations[owner] = prompt[1] // self.page_size - coded
            self.store.reserved_code_pages = sum(self._prefix_reservations.values())
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

    def prefix_code_credit(self, req):
        return self._prefix_reservations.get(self.request_owner(req), 0)

    def publish_prefix(
        self,
        req,
        req_to_token_pool,
        length,
        *,
        encode_length=None,
        tree_cache=None,
        may_skip=False,
    ):
        pages = self.commit_request(req, req_to_token_pool, length)
        if encode_length is None:
            encode_length = length
        if not 0 <= encode_length <= length:
            raise ValueError("QSA prefix encoding boundary exceeds committed writes")
        full = pages[: encode_length // self.page_size]
        owner = self.request_owner(req)
        owned = [p for p in full if self.store.pages[p].owners.get(owner) == "exact"]
        needed = sum(not self.store.pages[p].code for p in owned)
        credit = self._prefix_reservations.pop(owner, 0)
        self.store.reserved_code_pages = sum(self._prefix_reservations.values())
        try:
            # Radix locks protect current/shared inputs. Evict only inactive
            # cache, never a live prefix, and recheck physical code capacity.
            while needed > self.store.available_code_pages and tree_cache is not None:
                if (
                    tree_cache.evict_full(
                        (needed - self.store.available_code_pages) * self.page_size
                    )
                    == 0
                ):
                    break
            if needed > self.store.available_code_pages and may_skip:
                self.store.conversion_deferred["finished_cache_insert_skipped"] += 1
                return False
            if not owned:
                return True
            temporary = object()
            self.store.acquire_prefix(owned, owner=temporary)
            for page in owned:
                self.store.pages[page].owners[owner] = "code"
            self.store.release(owned, owner=temporary)
        except Exception:
            self._prefix_reservations[owner] = credit
            self.store.reserved_code_pages = sum(self._prefix_reservations.values())
            raise
        return True

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

    def exact_prefill_locations(
        self, request_slots, sequence_lengths, req_to_token_pool
    ):
        """Eager-only exact chunk gather; any published code keeps mixed reads.

        Check this request's pages, including shared code/exact representations.
        The global presence of unrelated cached code pages is irrelevant.
        """
        if torch.is_tensor(request_slots):
            request_slots = request_slots.tolist()
        locations = []
        for slot, length in zip(request_slots, sequence_lengths, strict=True):
            logical = req_to_token_pool.req_to_token[int(slot), : int(length)]
            pages = (logical[:: self.page_size] // self.page_size).tolist()
            if any(self.store.pages[page].code for page in pages):
                return None
            if any(not self.store.pages[page].exact for page in pages):
                raise ValueError("Prefill page has neither exact nor code backing")
            physical = self.store.exact_page[logical.long() // self.page_size].long()
            locations.append(physical * self.page_size + logical % self.page_size)
        return locations

    def publish_before_boundary(
        self, request_slots, sequence_lengths, boundary_lengths, req_to_token_pool
    ):
        """P/D wrapper hook, after all P/emitter layers and before D attention.

        These are host-side request lengths; graph replay only sees the stable
        page tables updated here. Zero-boundary intermediate chunks stay exact.
        """
        if torch.is_tensor(request_slots):
            request_slots = request_slots.tolist()
        for slot, length, boundary in zip(
            request_slots, sequence_lengths, boundary_lengths, strict=True
        ):
            if boundary == 0:
                continue
            req = self._active_requests.get(int(slot))
            if req is None or not 0 < boundary <= length <= req.kv.kv_allocated_len:
                raise ValueError("QSA boundary handoff needs a bound allocated request")
            committed = int(length - boundary)
            self.publish_prefix(
                req,
                req_to_token_pool,
                committed,
                encode_length=min(committed, max(0, len(req.origin_input_ids) - 1)),
            )

    def finish_request(self, req):
        owner = self.request_owner(req)
        self.store._next_query.pop(owner, None)
        self._bound_prompts.pop(owner, None)
        if self._active_requests.get(req.kv.req_pool_idx) is req:
            self._active_requests.pop(req.kv.req_pool_idx)
        self._prefix_reservations.pop(owner, None)
        self.store.reserved_code_pages = sum(self._prefix_reservations.values())
        self.write_audit()

    def attach_indexer_buffers(self, indexer):
        self.indexer_bytes = {
            "indexer_compressed": indexer.qsa_compressed_flat.nbytes,
            "indexer_pending": sum(t.nbytes for t in indexer.qsa_key_state_buffer_pool)
            + indexer.qsa_rope_position_buffer.nbytes,
        }

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
        data.update(self.indexer_bytes)
        data["read_workspaces"] = dict(self.workspace_bytes)
        data["read_workspace_bytes"] = sum(self.workspace_bytes.values())
        data.update(
            spike_format="bitmap" if self.layout.value_bitmap else "original",
            read_tuned=self.read_tuned,
            prefix_length_table=self.prefix_lengths.nbytes,
            virtual_token_capacity=self.size,
            exact_page_capacity=self.store.exact[0][0].shape[0] - 1,
            code_page_capacity=self.store.codes[0].rotary.shape[0] - 1,
            allocated_virtual_pages=len(self.store.pages),
            free_exact_pages=self.store.available_exact_pages,
            free_code_pages=self.store.available_code_pages,
            physical_free_code_pages=len(self.store._free_code),
            reserved_prefix_code_pages=self.store.reserved_code_pages,
            delay_histogram=dict(self.store.delay_histogram),
            conversion_deferred=dict(self.store.conversion_deferred),
        )
        from sglang.srt.mem_cache.qsa_code_capacity import QSACodeCapacity

        expected = QSACodeCapacity(
            self.layer_num,
            self.head_num,
            self.exact_fraction,
        ).allocation(self.size, self.prefix_lengths.numel())
        # Physical tensors are authoritative; the sizing estimator is audited
        # independently, including sentinel pages and indexer replication.
        data["sizing_matches_allocation"] = all(
            data[key] == expected[key]
            for key in (
                "exact",
                "code",
                "page_tables",
                "weights",
                "prefix_length_table",
            )
        ) and all(data[key] == expected[key] for key in self.indexer_bytes)
        data["cuda_process_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
        data["cuda_process_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
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
