"""The tail server's forward, on the scheduler's own pools and model runner.

A tail is a request whose prefix [0, e - W) is preallocated as a PD decode's transfer
destination is (only the window pool's live tail gets window slots), and whose window
[e - W, e) then runs as one ordinary extend through the late layers: the model starts
at the last KV source layer from the residual stage 0 captured, and decoder SWA bounded
replay floors the late layers' window at s - W.
"""

from __future__ import annotations

import logging
from array import array
from typing import Dict, List, Optional, Sequence

import msgspec
import numpy as np
import torch
from sglang.srt.disaggregation.decode import alloc_for_decode_prealloc
from sglang.srt.disaggregation.dsv41_tail_stage.layout import c1_page_ids
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class TailRun(msgspec.Struct):
    """One tail in a forward: its request, its staging slot and its prompt length."""

    req: Req
    slot: int
    s: int


class TailStageForward:
    def __init__(
        self,
        *,
        model_runner,
        req_to_token_pool,
        allocator,
        tree_cache,
        model_config,
        num_slots: int,
        window: int,
        draft_aux: bool,
    ) -> None:
        self.model_runner = model_runner
        self.req_to_token_pool = req_to_token_pool
        self.allocator = allocator
        self.tree_cache = tree_cache
        self.model_config = model_config
        self.window = window
        self.page_size = allocator.page_size
        self.storage = kv_storage_of(req_to_token_pool, allocator)
        config = model_config.hf_text_config
        device = model_runner.device
        # [slots, W, hc_mult, hidden] bf16: the late layers' input stage 0 writes.
        self.residual = torch.zeros(
            num_slots,
            window,
            config.hc_mult,
            config.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.aux: Optional[torch.Tensor] = None
        if draft_aux:
            layers = list(config.dspark_target_layer_ids)
            model_runner.model.set_dspark_layers_to_capture(layers)
            # [slots, W, captures * hidden] bf16, in dspark_layers_to_capture order.
            self.aux = torch.zeros(
                num_slots,
                window,
                len(layers) * config.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )

    def admit(
        self, *, rid: str, e: int, tail_token_ids: Optional[Sequence[int]] = None
    ) -> Optional[Req]:
        """A request holding the prefix [0, e - W), ready for stage 0's c1 pages;
        None while the pool cannot hold its prefix and window."""
        prefix_len = e - self.window
        # The window's extend takes at most one page beyond the prefix's last.
        if self.allocator.owner_available_size() < prefix_len + 2 * self.page_size:
            return None
        ids = array("q", bytes(8 * e))
        if tail_token_ids is not None:
            ids[prefix_len:e] = array("q", tail_token_ids)
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=ids,
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        if self.storage.acquire([req]) is None:
            return None
        req.kv.commit_prealloc(prefix_len)
        window_tails = self.storage.window_tails(prefix_len, keep_insert_boundary=False)
        kv_loc = alloc_for_decode_prealloc(
            self.storage,
            self.allocator,
            req=req,
            fill_len=prefix_len,
            delta_len=prefix_len,
            prefix_len=0,
            total_prefix_len=0,
            prefix_indices=None,
            uses_swa_tail=True,
            window_tails=window_tails,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
        )
        assert kv_loc is not None, f"no KV for a {prefix_len}-token tail prefix"
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, len(kv_loc))), kv_loc
        )
        req.prefix_len = prefix_len
        req.extend_end = e
        return req

    def c1_pages(self, req: Req) -> np.ndarray:
        row = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        return c1_page_ids(row, e=req.prefix_len, page_size=self.page_size)

    def run(self, tails: List[TailRun]) -> None:
        """The late layers over every tail's window; fills each slot's aux."""
        self.execute(tails, self.prepare(tails))

    def prepare(self, tails: List[TailRun]) -> ForwardBatch:
        """Host side: the extend batch over the windows, with their late floors."""
        batch = ScheduleBatch.init_new(
            reqs=[t.req for t in tails],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        forward_batch = ForwardBatch.init_new(
            batch, self.model_runner, return_hidden_states_before_norm=False
        )
        floors = torch.tensor(
            [t.s - self.window for t in tails],
            dtype=torch.int64,
            device=self.residual.device,
        ).repeat_interleave(self.window)
        # Rows below the late-layer floor s - W attend to themselves only, as a
        # fed tail's do on decode.
        forward_batch.encoder_swa_row_floor = torch.minimum(
            floors, forward_batch.positions.to(torch.int64)
        )
        return forward_batch

    @torch.no_grad()
    def execute(self, tails: List[TailRun], forward_batch: ForwardBatch) -> None:
        slots = torch.tensor(
            [t.slot for t in tails], dtype=torch.int64, device=self.residual.device
        )
        residual = self.residual[slots].flatten(0, 1)
        out = self.model_runner.forward(
            forward_batch,
            pp_proxy_tensors=PPProxyTensors({"hidden_states": residual}),
        )
        if self.aux is not None:
            self.aux[slots] = out.logits_output.view(len(tails), self.window, -1)

    def release(self, req: Req) -> None:
        release_kv_cache(req, self.tree_cache, checkpoint=False)

    def page_source(self) -> TailPages:
        return TailPages(self)


class TailPages:
    """The slot table's PageSource on the scheduler's pools: a grant admits the
    tail's request with its prefix preallocated; a freed slot releases it."""

    def __init__(self, fwd: TailStageForward) -> None:
        self.fwd = fwd
        self.capacity = fwd.allocator.owner_available_size() // fwd.page_size
        self.reqs: Dict[int, Req] = {}

    def take(self, tail) -> Optional[List[int]]:
        req = self.fwd.admit(rid=tail.rid, e=tail.e)
        if req is None:
            return None
        self.reqs[tail.room] = req
        return self.fwd.c1_pages(req).tolist()

    def give_back(self, tail) -> None:
        self.fwd.release(self.reqs.pop(tail.room))
