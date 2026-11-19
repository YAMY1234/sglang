"""The tail server's forward, on memory it owns and the scheduler's model runner.

At startup the tail server takes its memory from the scheduler's pools once (memory.py):
a request row and two window pages per slot, and full pages it hands out per grant. A
tail's window [e - W, e) then runs as one extend through the late layers: the model
starts at the last KV source layer from the residual stage 0 captured, and decoder SWA
bounded replay floors the late layers' window at s - W. The extend batch is filled from
the grant's host-side layout; nothing is allocated per forward.
"""

from __future__ import annotations

import logging
from array import array
from contextlib import nullcontext
from typing import Dict, List, Sequence

import msgspec
import numpy as np
import torch
from sglang.srt.disaggregation.dsv41_tail_stage.memory import (
    SWA_PAGES_PER_SLOT,
    RoundPlan,
    TailMemory,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

# Arbitrary: tokens per allocator call while the full pool is taken at startup, to bound
# the temporary index tensor.
TAKE_CHUNK_TOKENS = 1 << 24


def window_row_floors(
    *, prompt_lens: Sequence[int], window: int, positions: torch.Tensor
) -> torch.Tensor:
    """Each window row's late-layer SWA floor: s - W, the floor bounded replay gives
    the prompt's last window; a row below it (1 or 2 per tail) attends to itself
    only, as a fed tail's do on decode (_floored_verify_swa)."""
    floors = torch.tensor(
        prompt_lens, dtype=torch.int64, device=positions.device
    ).repeat_interleave(window)
    return torch.minimum(floors - window, positions.to(torch.int64))


class TailRun(msgspec.Struct):
    """One tail in a forward: its slot, prompt length s, split e, and the full slots of
    its window (host, from the grant)."""

    slot: int
    s: int
    e: int
    window_full: np.ndarray


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
        max_tails: int,
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
        self.device = model_runner.device
        config = model_config.hf_text_config
        mem_pool = model_runner.token_to_kv_pool.custom_mem_pool
        # Mooncake reads and writes these as the KV pools: under a custom memory
        # pool (MNNVL) they must come from it, as the pools' buffers do.
        with torch.cuda.use_mem_pool(mem_pool) if mem_pool else nullcontext():
            # [slots, W, hc_mult, hidden] bf16: the late layers' input stage 0 writes.
            self.residual = torch.zeros(
                num_slots,
                window,
                config.hc_mult,
                config.hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.aux = None
            if draft_aux:
                layers = list(config.dspark_target_layer_ids)
                model_runner.model.set_dspark_layers_to_capture(layers)
                # [slots, W, captures * hidden] bf16, in dspark_layers_to_capture order.
                self.aux = torch.zeros(
                    num_slots,
                    window,
                    len(layers) * config.hidden_size,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
        self.memory = self._take_memory(num_slots)
        rows = self.memory.rows
        # One request per slot, bound to the slot's row for the server's lifetime.
        self.reqs = [self._slot_req(k, row) for k, row in enumerate(rows)]
        self._page_arange = torch.arange(
            self.page_size, dtype=torch.int64, device=self.device
        )
        # Tail token ids: the late layers read none (no vision routing).
        self._input_ids = torch.zeros(
            max_tails * window, dtype=torch.int64, device=self.device
        )
        self._sampling: Dict[int, SamplingBatchInfo] = {}
        rows_ = max_tails * (window + 2)
        self._staging = [
            torch.empty(rows_, dtype=torch.int64, pin_memory=True) for _ in range(2)
        ]
        self._staging_done = [None, None]
        self._next_staging = 0

    def _take_memory(self, num_slots: int) -> TailMemory:
        rows = self.req_to_token_pool.alloc_rows(num_slots)
        assert rows is not None, f"{num_slots} slots need {num_slots} request rows"
        swa_alloc = self.allocator.swa_attn_allocator
        swa_page_size = swa_alloc.page_size
        pool = self.model_runner.token_to_kv_pool
        assert swa_page_size == pool.swa_page_size, (swa_page_size, pool.swa_page_size)
        assert (
            pool.full_to_swa_index_mapping is self.allocator.full_to_swa_index_mapping
        ), "the pool and the allocator must share the full-to-SWA map"
        swa = swa_alloc.alloc(num_slots * SWA_PAGES_PER_SLOT * swa_page_size)
        assert swa is not None, "the SWA pool cannot hold two pages per slot"
        swa_pages = (swa[::swa_page_size] // swa_page_size).cpu().numpy()
        full_alloc = self.allocator.full_attn_allocator
        pages = []
        while True:
            want = min(
                full_alloc.available_size() // self.page_size * self.page_size,
                TAKE_CHUNK_TOKENS,
            )
            if want == 0:
                break
            got = full_alloc.alloc(want)
            pages.append((got[:: self.page_size] // self.page_size).cpu().numpy())
        full_pages = np.concatenate(pages)
        logger.info(
            "DSV4.1 tail stage owns %d request rows, %d SWA pages and %d full pages "
            "(%d tokens)",
            len(rows),
            swa_pages.size,
            full_pages.size,
            full_pages.size * self.page_size,
        )
        return TailMemory(
            rows=rows,
            swa_pages=swa_pages.reshape(num_slots, SWA_PAGES_PER_SLOT),
            full_pages=full_pages,
            page_size=self.page_size,
            swa_page_size=swa_page_size,
            window=self.window,
            write_round=self._write_round,
        )

    def _slot_req(self, slot: int, row: int) -> Req:
        req = Req(
            rid=f"dsv41-tail-slot-{slot}",
            origin_input_text="",
            origin_input_ids=array("q", [0]),
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        req.kv.req_pool_idx = row
        return req

    def _write_round(self, plan: RoundPlan) -> None:
        """One copy and two scatters for a whole grant round, on the default stream;
        forwards wait for it on theirs."""
        n_pages, n_window = len(plan.rows), len(plan.window_full)
        packed = torch.from_numpy(
            np.concatenate(
                [plan.rows, plan.cols, plan.pages, plan.window_full, plan.window_swa]
            )
        ).to(self.device)
        rows, cols, pages, full, swa = torch.split(
            packed, [n_pages, n_pages, n_pages, n_window, n_window]
        )
        ar = self._page_arange
        table = self.req_to_token_pool.req_to_token
        table[rows[:, None], cols[:, None] * self.page_size + ar] = (
            pages[:, None] * self.page_size + ar
        ).to(table.dtype)
        mapping = self.allocator.full_to_swa_index_mapping
        mapping[full] = swa.to(mapping.dtype)

    def prepare(self, runs: List[TailRun]) -> ForwardBatch:
        """The extend batch over the windows, filled as prepare_for_extend fills it for
        requests holding their window's KV: no allocation, no per-room tensors."""
        n, window = len(runs), self.window
        num_tokens = n * window
        seq_lens = [r.e for r in runs]
        rows = [self.memory.rows[r.slot] for r in runs]
        # [seq_lens | rows | window full slots], one pinned copy.
        host = np.concatenate([seq_lens, rows, *[r.window_full for r in runs]])
        device_view = self._to_device(host)
        batch = ScheduleBatch.init_new(
            reqs=[self.reqs[r.slot] for r in runs],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.forward_mode = ForwardMode.EXTEND
        batch.prefix_lens = [e - window for e in seq_lens]
        batch.extend_lens = [window] * n
        batch.extend_num_tokens = num_tokens
        batch.seq_lens = device_view[:n]
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = device_view[:n].to(torch.int32)
        batch.req_pool_indices = device_view[n : 2 * n]
        batch.req_pool_indices_cpu = torch.tensor(rows, dtype=torch.int64)
        batch.out_cache_loc = device_view[2 * n :]
        batch.input_ids = self._input_ids[:num_tokens]
        batch.prefill_input_ids_cpu = None
        batch.multimodal_inputs = [None] * n
        batch.seq_lens_sum = sum(seq_lens)
        # No logprobs: the start is the extend length, as compute_extend_logprob_start_len
        # gives for logprob_start_len -1.
        batch.extend_logprob_start_lens = [window] * n
        batch.extend_input_logprob_token_ids = None
        batch.sampling_info = self._sampling_info(batch)
        forward_batch = ForwardBatch.init_new(
            batch, self.model_runner, return_hidden_states_before_norm=False
        )
        forward_batch.encoder_swa_row_floor = window_row_floors(
            prompt_lens=[r.s for r in runs],
            window=window,
            positions=forward_batch.positions,
        )
        return forward_batch

    def _to_device(self, host: np.ndarray) -> torch.Tensor:
        """Through two alternating pinned buffers: a buffer is refilled only after the
        copy out of it two batches ago has finished."""
        k = self._next_staging
        self._next_staging ^= 1
        if self._staging_done[k] is not None:
            self._staging_done[k].synchronize()
        staging = self._staging[k][: len(host)]
        staging.numpy()[:] = host
        device = staging.to(self.device, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        self._staging_done[k] = done
        return device

    def _sampling_info(self, batch: ScheduleBatch) -> SamplingBatchInfo:
        """Every slot request samples alike (nothing is sampled): one per batch size."""
        n = len(batch.reqs)
        if n not in self._sampling:
            self._sampling[n] = SamplingBatchInfo.from_schedule_batch(
                batch, self.model_config.vocab_size
            )
        return self._sampling[n]

    @torch.no_grad()
    def execute(self, runs: List[TailRun], forward_batch: ForwardBatch) -> None:
        slots = torch.tensor(
            [r.slot for r in runs], dtype=torch.int64, device=self.residual.device
        )
        residual = self.residual[slots].flatten(0, 1)
        out = self.model_runner.forward(
            forward_batch,
            pp_proxy_tensors=PPProxyTensors({"hidden_states": residual}),
        )
        if self.aux is not None:
            self.aux[slots] = out.logits_output.view(len(runs), self.window, -1)

    def run(self, runs: List[TailRun]) -> None:
        self.execute(runs, self.prepare(runs))
