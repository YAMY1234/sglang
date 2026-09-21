"""SGLang adapter for the x256 physical code/exact page store.

Virtual page lifetimes remain owned by SGLang's allocator/radix locks. The
store's named readers additionally protect exact views until prefix handoff or
the accepted-token age boundary. Scheduler hooks run outside CUDA graphs.
"""

import heapq
import os
from collections import defaultdict
from contextlib import nullcontext

import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.qsa.code import load_x256_weights, serving_code_layout
from sglang.srt.layers.attention.qsa.code_kernel import write_exact_tokens
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.qsa_code_pool import QSAPrefixPageStore, _Page
from sglang.srt.mem_cache.qsa_code_capacity import qsa_fused_read_enabled


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
        exact_tokens=None,
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
        self.read_fused = qsa_fused_read_enabled()
        if self.read_fused and not (
            self.layout.stored_residuals and self.layout.value_bitmap
        ):
            raise ValueError("Fused QSA reads require residual/bitmap code pages")
        self.exact_fraction = exact_fraction
        self.exact_tokens = exact_tokens
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
        if exact_tokens is not None and (
            exact_tokens < page_size or exact_tokens % page_size or exact_tokens >= size
        ):
            raise ValueError(
                "Fixed QSA exact reserve must leave at least one code page"
            )
        exact_pages = (
            int(pages * exact_fraction)
            if exact_tokens is None
            else exact_tokens // page_size
        )
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
        self._request_page_cache = {}
        self._bound_page_lists = {}
        self._committed_lengths = {}
        self._tracked_full_pages = {}
        self._decode_event_plan = None
        self._requests_by_owner = {}
        self._tree_cache = None
        self.indexer_bytes = {}
        self.workspace_bytes = {}
        encoder_graph = os.environ.get("SGLANG_QSA_CODE_ENCODER_GRAPH", "0")
        if encoder_graph not in ("0", "1"):
            raise ValueError("SGLANG_QSA_CODE_ENCODER_GRAPH must be 0 or 1")
        if encoder_graph == "1":
            if self.enable_custom_mem_pool or enable_memory_saver:
                raise ValueError(
                    "QSA encoder graph requires the standard CUDA allocator"
                )
            self.store.prepare_encoder_graph()
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
        if any(v not in self.store._free_virtual for v in ids):
            raise ValueError("QSA virtual page is outside the free address space")
        physical = [self.store._free_exact.pop() for _ in ids]
        for virtual, exact in zip(ids, physical, strict=True):
            self.store.pages[virtual] = _Page(exact=exact, owners={"pending": "exact"})
        for virtual in ids:
            del self.store._free_virtual[virtual]
        if ids:
            self.store.exact_page[virtual_ids.long()] = torch.tensor(
                physical, dtype=torch.int32, device=self.device
            )

    def free_pages(self, virtual_ids):
        """Called only when radix locks say the virtual pages have no readers."""
        # Frees include radix deduplication and retraction. Invalidate all cached
        # row identities before an ID can be reused, even at the same length.
        self._invalidate_request_pages()
        grouped = defaultdict(list)
        for virtual in virtual_ids.tolist():
            if virtual not in self.store.pages:
                raise ValueError("QSA double free of virtual page")
            for owner in self.store.pages[virtual].owners:
                grouped[owner].append(virtual)
        for owner, pages in grouped.items():
            self.store.release(pages, owner=owner)
            if not any(owner in page.owners for page in self.store.pages.values()):
                self.store.stop_generation(owner)
                prompt = self._bound_prompts.pop(owner, None)
                if prompt is not None:
                    self._active_requests.pop(prompt[0], None)
                self._requests_by_owner.pop(owner, None)
                self._prefix_reservations.pop(owner, None)
        self.store.reserved_code_pages = sum(self._prefix_reservations.values())

    def clear(self):
        if self.store.pages:
            ids = torch.tensor(
                list(self.store.pages), dtype=torch.int64, device=self.device
            )
            self.free_pages(ids)
        self.store._next_query.clear()
        self.store._age_heaps.clear()
        self.store._age_ready.clear()
        self._invalidate_request_pages()
        self.store.delay_histogram.clear()
        self.store.conversion_deferred.clear()
        self.prefix_lengths.zero_()
        self._bound_prompts.clear()
        self._active_requests.clear()
        self._requests_by_owner.clear()
        self._prefix_reservations.clear()
        self.store.reserved_code_pages = 0

    @staticmethod
    def request_owner(req):
        return f"request:{id(req)}"

    def _invalidate_request_pages(self):
        self._request_page_cache.clear()
        self._bound_page_lists.clear()
        self._committed_lengths.clear()
        self._tracked_full_pages.clear()
        self._decode_event_plan = None

    def _request_pages(self, req, req_to_token_pool, length):
        pages = self._request_pages_batch([req], req_to_token_pool)[0]
        return pages[: (length + self.page_size - 1) // self.page_size]

    def _request_pages_batch(self, reqs, req_to_token_pool):
        """Read changed page rows only; ordinary decode steps need no D2H copy.

        Allocated virtual IDs are stable until a free/retraction/deduplication.
        free_pages invalidates the cache before reuse; slot/count changes miss.
        """
        result, misses = [], []
        for req in reqs:
            owner = self.request_owner(req)
            count = (req.kv.kv_allocated_len + self.page_size - 1) // self.page_size
            key = (id(req_to_token_pool), req.kv.req_pool_idx, count)
            cached = self._request_page_cache.get(owner)
            if cached is not None and cached[0] == key:
                result.append(cached[1])
            elif count == 0:
                result.append(())
                self._request_page_cache[owner] = (key, result[-1])
            else:
                prefix = (
                    cached[1]
                    if cached is not None
                    and cached[0][:2] == key[:2]
                    and len(cached[1]) < count
                    else ()
                )
                misses.append((len(result), owner, key, prefix))
                result.append(None)
        if misses:
            table = req_to_token_pool.req_to_token
            stride = table.shape[1]
            positions = [
                key[1] * stride + page * self.page_size
                for _, _, key, prefix in misses
                for page in range(len(prefix), key[2])
            ]
            rows = torch.tensor(positions, dtype=torch.int64, device=table.device)
            pages = (table.flatten().index_select(0, rows) // self.page_size).tolist()
            offset = 0
            for index, owner, key, prefix in misses:
                # Immutable snapshots allow identity checks on the hot path;
                # a caller cannot mutate a cached row behind ownership control.
                count = key[2] - len(prefix)
                result[index] = prefix + tuple(pages[offset : offset + count])
                offset += count
                self._request_page_cache[owner] = (key, result[index])
        return result

    def bind_requests(self, reqs, req_to_token_pool, *, tree_cache=None):
        # P31 publishes inside the model forward, after the scheduler has
        # admitted this batch using code + evictable capacity. Retain the same
        # radix context so that publication can realize those physical frees.
        # QSA code serving disables overlap; this is the scheduler's live cache.
        if tree_cache is not None:
            self._tree_cache = tree_cache
        pages = self._request_pages_batch(reqs, req_to_token_pool)
        for req, allocated in zip(reqs, pages, strict=True):
            self.bind_request(req, req_to_token_pool, _allocated_pages=allocated)

    def bind_request(self, req, req_to_token_pool, *, _allocated_pages=None):
        owner = self.request_owner(req)
        prompt = (req.kv.req_pool_idx, max(0, len(req.origin_input_ids) - 1))
        allocated = (
            self._request_pages(req, req_to_token_pool, req.kv.kv_allocated_len)
            if _allocated_pages is None
            else _allocated_pages
        )
        if not isinstance(allocated, tuple):
            allocated = tuple(allocated)
        self._active_requests[prompt[0]] = req
        self._requests_by_owner[owner] = req
        if self._bound_prompts.get(owner) != prompt:
            self.prefix_lengths[prompt[0]] = prompt[1]
            self._bound_prompts[owner] = prompt
            allocated_prefix = allocated[
                : (prompt[1] + self.page_size - 1) // self.page_size
            ]
            coded = sum(bool(self.store.pages[p].code) for p in allocated_prefix)
            self._prefix_reservations[owner] = (
                prompt[1] + self.page_size - 1
            ) // self.page_size - coded
            self.store.reserved_code_pages = sum(self._prefix_reservations.values())
        previous = self._bound_page_lists.get(owner, ())
        if previous is allocated:
            return
        if allocated[: len(previous)] == previous:
            newly_bound = allocated[len(previous) :]
        else:
            newly_bound = allocated
            self._committed_lengths.pop(owner, None)
            self._tracked_full_pages.pop(owner, None)
        for virtual in newly_bound:
            page = self.store.pages[virtual]
            if "pending" in page.owners:
                del page.owners["pending"]
                page.owners[owner] = "exact"
            elif not page.code and owner not in page.owners:
                self.store.acquire_exact([virtual], owner=owner)
        self._bound_page_lists[owner] = allocated

    def commit_request(self, req, req_to_token_pool, length, *, _allocated_pages=None):
        """Call after a successful forward or at its next scheduler boundary."""
        owner = self.request_owner(req)
        allocated = (
            self._request_pages(req, req_to_token_pool, req.kv.kv_allocated_len)
            if _allocated_pages is None
            else _allocated_pages
        )
        self.bind_request(req, req_to_token_pool, _allocated_pages=allocated)
        pages = allocated[: (length + self.page_size - 1) // self.page_size]
        ids, lengths = [], []
        previous = self._committed_lengths.get(owner, 0)
        if length == previous:
            return pages
        first = previous // self.page_size if length >= previous else 0
        for i in range(first, len(pages)):
            virtual = pages[i]
            page = self.store.pages[virtual]
            if page.owners.get(owner) in ("exact", "both"):
                ids.append(virtual)
                lengths.append(min(self.page_size, length - i * self.page_size))
        self.store.commit_serving_writes(ids, lengths, owner=owner)
        self._committed_lengths[owner] = length
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
        selected = pages[: (encode_length + self.page_size - 1) // self.page_size]
        lengths = [
            min(self.page_size, encode_length - i * self.page_size)
            for i in range(len(selected))
        ]
        owner = self.request_owner(req)
        needed = sum(not self.store.pages[p].code for p in selected)
        credit = self._prefix_reservations.pop(owner, 0)
        self.store.reserved_code_pages = sum(self._prefix_reservations.values())
        try:
            # Radix locks protect current/shared inputs. Evict only inactive
            # cache, never a live prefix, and recheck physical code capacity.
            if needed > self.store.available_code_pages and tree_cache is not None:
                tree_cache.token_to_kv_pool_allocator.flush_deferred_frees()
            while needed > self.store.available_code_pages and tree_cache is not None:
                evicted = tree_cache.evict(
                    EvictParams(
                        num_tokens=(needed - self.store.available_code_pages)
                        * self.page_size
                    )
                ).num_tokens_evicted
                tree_cache.token_to_kv_pool_allocator.flush_deferred_frees()
                if evicted == 0:
                    break
            if needed > self.store.available_code_pages and may_skip:
                self.store.conversion_deferred["finished_cache_insert_skipped"] += 1
                return False
            if not selected:
                return True
            temporary = object()
            self.store.acquire_prefix(selected, owner=temporary, valid_lengths=lengths)
            for virtual, code_length in zip(selected, lengths, strict=True):
                page = self.store.pages[virtual]
                # A prefix tail also contains this request's exact boundary /
                # future generated tokens. Keep both views until the page ages.
                page.owners[owner] = (
                    "both" if code_length < self.page_size and page.exact else "code"
                )
            self.store.release(selected, owner=temporary)
        except Exception:
            self._prefix_reservations[owner] = credit
            self.store.reserved_code_pages = sum(self._prefix_reservations.values())
            raise
        return True

    def prepare_decode(self, reqs, req_to_token_pool):
        allocated_batch = self._request_pages_batch(reqs, req_to_token_pool)
        with self.store.batch_page_tables():
            for req, allocated in zip(reqs, allocated_batch, strict=True):
                committed = req.kv.kv_committed_len
                pages = self.commit_request(
                    req, req_to_token_pool, committed, _allocated_pages=allocated
                )
                owner = self.request_owner(req)
                generated, ends, starts = [], [], []
                full = committed // self.page_size
                first = self._tracked_full_pages.get(owner, 0)
                for i in range(first if full >= first else 0, full):
                    virtual = pages[i]
                    page = self.store.pages[virtual]
                    if page.owners.get(owner) in ("exact", "both"):
                        generated.append(virtual)
                        ends.append((i + 1) * self.page_size - 1)
                        starts.append(
                            max(0, len(req.origin_input_ids) - 1 - i * self.page_size)
                        )
                if generated:
                    self.store.track_generated_pages(
                        generated, ends, owner=owner, start_offsets=starts
                    )
                self._tracked_full_pages[owner] = full
                self.store.advance_generation(
                    owner=owner,
                    next_query_position=committed,
                    committed_position=committed - 1,
                )
            # Shared readers need actual accepted positions when a conversion
            # is due, not a stale watermark from their previous 64-step event.
            # No scan of unrelated requests/pages occurs on ordinary steps.
            for page in list(self.store._age_ready.values()):
                for owner in page.generated_ends:
                    request = self._requests_by_owner.get(owner)
                    if request is not None:
                        committed = request.kv.kv_committed_len
                        self.store.advance_generation(
                            owner=owner,
                            next_query_position=committed,
                            committed_position=committed - 1,
                        )
            converted = self.store.convert_aged_pages()
        if converted:
            self.write_audit()

    def prepare_decode_batch(self, batch):
        """O(1) ordinary non-spec step; only due requests enter page control.

        SGLang replaces reqs on merge/filter. Keeping the actual list (not its
        id) prevents identity reuse; physical free/retraction invalidates the
        plan explicitly. Non-overlap decode accepts exactly one token per
        invocation. Variable speculative advances retain the existing path.
        """
        if not batch.spec_algorithm.is_none():
            self._decode_event_plan = None
            return self.prepare_decode(batch.reqs, batch.req_to_token_pool)
        plan = self._decode_event_plan
        if (
            plan is None
            or plan["reqs"] is not batch.reqs
            or plan["table"] is not batch.req_to_token_pool
            or plan["count"] != len(batch.reqs)
        ):
            self.prepare_decode(batch.reqs, batch.req_to_token_pool)
            queue = [
                (
                    (self.page_size - 1 - req.kv.kv_committed_len) % self.page_size
                    or self.page_size,
                    index,
                    req,
                )
                for index, req in enumerate(batch.reqs)
            ]
            heapq.heapify(queue)
            self._decode_event_plan = {
                "reqs": batch.reqs,
                "count": len(batch.reqs),
                "table": batch.req_to_token_pool,
                "step": 0,
                "queue": queue,
            }
            return
        plan["step"] += 1
        queue, step = plan["queue"], plan["step"]
        if not queue or queue[0][0] > step:
            return
        due = []
        while queue and queue[0][0] <= step:
            deadline, order, req = heapq.heappop(queue)
            due.append(req)
            heapq.heappush(queue, (deadline + self.page_size, order, req))
        self.prepare_decode(due, batch.req_to_token_pool)

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
                tree_cache=self._tree_cache,
            )

    def finish_request(self, req):
        owner = self.request_owner(req)
        self._requests_by_owner.pop(owner, None)
        self._decode_event_plan = None
        self.store.stop_generation(owner)
        self._request_page_cache.pop(owner, None)
        self._bound_page_lists.pop(owner, None)
        self._committed_lengths.pop(owner, None)
        self._tracked_full_pages.pop(owner, None)
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
            read_fused=self.read_fused,
            exact_token_reserve=self.exact_tokens,
            legacy_exact_fraction=self.exact_fraction,
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
            encoder_graph_enabled=self.store._encoder_graph is not None,
            encoder_graph_allocated_delta_bytes=self.store.encoder_graph_bytes,
            encoder_graph_reserved_delta=self.store.encoder_graph_reserved_delta,
            encoder_graph_pool_reserved_bytes=self.store.encoder_graph_pool_reserved_bytes,
            encoder_graph_workspace_bound_bytes=self.store.encoder_graph_workspace_bound_bytes,
            encoder_graph_resource_bound_bytes=self.store.encoder_graph_resource_bound_bytes,
            encoder_graph_captures=self.store.encoder_graph_captures,
            encoder_graph_evictions=self.store.encoder_graph_evictions,
        )
        from sglang.srt.mem_cache.qsa_code_capacity import QSACodeCapacity

        expected = QSACodeCapacity(
            self.layer_num,
            self.head_num,
            self.exact_fraction,
            exact_tokens=self.exact_tokens,
        ).allocation(self.size, self.prefix_lengths.numel())
        # Physical tensors are authoritative; the sizing estimator is audited
        # independently, including sentinel pages and indexer replication.
        data["sizing_matches_allocation"] = all(
            data[key] == expected[key]
            for key in (
                "exact",
                "code",
                "page_tables",
                "host_control",
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
