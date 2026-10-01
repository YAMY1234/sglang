"""Tier A orchestration; target verify/acceptance and KV allocation stay unchanged."""

import copy
import logging

import torch
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.eagle_extend_draft_cuda_graph_runner import (
    EAGLEExtendDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.eagle_worker_common import prepare_for_draft_extend
from sglang.srt.speculative.extend_draft_state import PendingExtendStore

logger = logging.getLogger(__name__)


class ExtendDraftCoordinator:
    @classmethod
    def create(cls, worker):
        if not envs.SGLANG_SPEC_FUSE_EXTEND_DRAFT.get():
            return None
        args = worker.server_args
        # Finished requests can leave pending KV; never publish it to a prefix cache.
        supported = (
            (
                args.disaggregation_mode == "decode"
                or (args.disaggregation_mode == "null" and args.dp_size == 1)
            )
            and args.disable_radix_cache
            and args.pp_size == 1
            and worker.topk == 1
            and worker.speculative_num_steps > 1
            and worker.speculative_num_draft_tokens == worker.speculative_num_steps + 1
            and type(worker.draft_runner.model).__name__ == "Qwen3_5ForCausalLMMTP"
            and type(worker.draft_attn_backend).__name__
            == "TRTLLMHAAttnMultiStepDraftBackend"
            and type(worker.draft_extend_attn_backend).__name__ == "TRTLLMHAAttnBackend"
            and worker.cuda_graph_runner is not None
            and worker.cuda_graph_runner_for_draft_extend is not None
            and not get_spec().speculative_adaptive
            and not get_spec().speculative_use_rejection_sampling
            and not args.enable_lora
            and not args.enable_two_batch_overlap
            and not args.enable_mixed_chunk
        )
        if not supported:
            logger.info(
                "extend+draft disabled: model=%s draft=%s extend=%s PP=%s DP=%s",
                type(worker.draft_runner.model).__name__,
                type(worker.draft_attn_backend).__name__,
                type(worker.draft_extend_attn_backend).__name__,
                args.pp_size,
                args.dp_size,
            )
            return None
        return cls(worker)

    def __init__(self, worker):
        self.worker = worker
        self.graph = EAGLEExtendDraftCudaGraphRunner(worker)
        ebuf = self.graph.extend.buffers
        self.store = PendingExtendStore(
            worker.req_to_token_pool,
            worker.speculative_num_draft_tokens,
            ebuf.hidden_states.shape[-1],
            ebuf.hidden_states.dtype,
            ebuf.hidden_states.device,
        )
        self.width = self.store.width
        self.flushes = self.deferred_rounds = 0
        worker.target_worker.model_runner.extend_draft_coordinator = self
        logger.info(
            "extend+draft captured buckets=%s pending_hidden_bytes=%d",
            self.graph.capture_bs,
            self.store.hidden.numel() * self.store.hidden.element_size(),
        )

    def vote(self, batch):
        # Three fields ride the existing scheduler metadata gather; no new D2H.
        if (
            batch is None
            or batch.forward_mode.is_prebuilt()
            or batch.forward_mode.is_idle()
        ):
            return (True, True, False)
        ready = self.store.ready(batch.req_pool_indices_cpu)
        eligible = (
            batch.forward_mode.is_decode()
            and batch.batch_size() <= self.graph.max_bs
            and (
                not self.graph.disable_padding
                or batch.batch_size() in self.graph.capture_bs
            )
            and all(x is None for x in batch.multimodal_inputs)
        )
        return (eligible, bool(ready.all()), bool(ready.any()))

    def decisions(self, batch):
        votes = batch.extend_draft_votes
        if votes is None:
            if self.worker.server_args.dp_size > 1:
                raise RuntimeError(
                    "extend+draft requires the existing DP scheduler vote"
                )
            votes = self.vote(batch)
            batch.extend_draft_votes = votes
        can_defer, all_pending, any_pending = votes
        return can_defer, can_defer and all_pending and any_pending, any_pending

    def _prepare(self, batch):
        slots = batch.req_pool_indices_cpu
        if slots is None:
            slots = torch.empty(0, dtype=torch.int64)
        indices = batch.req_pool_indices
        if indices is None:
            indices = self.store.tokens.new_empty(0)
        rows = self.store.take(slots, indices)
        pending = copy.copy(batch)
        pending.req_pool_indices = rows.indices
        pending.seq_lens = rows.seq_lens
        pending.seq_lens_cpu = pending.seq_lens_sum = None
        pending.out_cache_loc = rows.cache
        pending.multimodal_inputs = [None] * slots.numel()
        pending.forward_mode = (
            ForwardMode.IDLE if not slots.numel() else ForwardMode.DECODE
        )
        info = EagleDraftExtendInput(
            hidden_states=rows.hidden,
            num_correct_drafts=rows.accept_lens - 1,
            num_accept_tokens=rows.accept_lens,
            num_tokens_per_req=self.width,
            num_tokens_for_logprob_per_req=self.width,
        )
        fb = prepare_for_draft_extend(
            info,
            pending,
            rows.tokens,
            self.width,
            self.worker.draft_runner,
            self.worker.cuda_graph_runner_for_draft_extend,
            return_hidden_states_before_norm=False,
        )
        select = torch.arange(slots.numel(), device=rows.indices.device) * self.width
        select = select + rows.accept_lens - 1
        return rows, fb, select

    def before_draft(self, batch):
        _, fused, any_pending = self.decisions(batch)
        if not any_pending:
            return None
        rows, fb, select = self._prepare(batch)
        if fused:
            self.graph.stage_extend(fb, select)
            self.store.consumed(rows)
            return self.graph

        graph = self.worker.cuda_graph_runner_for_draft_extend
        if graph.can_run_graph(fb):
            out = graph.execute(fb, select)
        else:
            out = self.worker.draft_runner.forward(
                fb, skip_attn_backend_init=True
            ).logits_output
            out.next_token_logits = out.next_token_logits[select]
            out.hidden_states = out.hidden_states[select]
        mask = rows.valid_cpu.to(rows.indices.device, non_blocking=True).unsqueeze(1)
        seed = batch.spec_info
        seed.topk_index = torch.where(
            mask, out.next_token_logits.argmax(-1, keepdim=True), seed.topk_index
        )
        seed.topk_p = torch.where(mask, 1.0, seed.topk_p)
        seed.hidden_states = torch.where(mask, out.hidden_states, seed.hidden_states)
        self.store.consumed(rows)
        self.flushes += 1
        return None

    def after_verify(self, batch, result):
        can_defer, _, _ = self.decisions(batch)
        if not can_defer:
            return False
        slots = batch.req_pool_indices_cpu
        if slots is None:
            slots = torch.empty(0, dtype=torch.int64)
        self.store.put(
            slots,
            batch.req_pool_indices.long(),
            result.logits_output.hidden_states,
            result.next_token_ids,
            batch.out_cache_loc,
            batch.seq_lens,
            result.accept_lens,
        )
        n = slots.numel()
        seed = result.next_draft_input
        seed.topk_p = torch.zeros((n, 1), dtype=torch.float32, device=batch.device)
        seed.topk_index = torch.zeros((n, 1), dtype=torch.int64, device=batch.device)
        seed.hidden_states = self.store.hidden.new_zeros(
            (n, self.store.hidden.shape[-1])
        )
        self.deferred_rounds += 1
        return True

    def record_shared_read_done(self):
        # on_publish is verify-end; pending copies and graph reads need a later fence.
        event = torch.cuda.Event()
        event.record()
        self.worker.draft_runner.shared_read_done_event = event
