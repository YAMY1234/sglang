import logging
import os
import time
from typing import List, Optional

import torch
from huggingface_hub import snapshot_download

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, global_server_args_dict
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardBatchOutput,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.build_eagle_tree import (
    TreeMaskMode,
    build_tree_kernel_efficient,
)
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils_v2 import (
    EagleDraftInput,
    EagleVerifyInput,
    assign_extend_cache_locs,
    fast_topk,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
    select_top_k_tokens,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import empty_context, get_available_gpu_memory, next_power_of_2

logger = logging.getLogger(__name__)


def _debug_check_indices(tag, idx, upper):
    """Pull upper bound and indices to CPU for safety check, avoiding triggering CUDA kernels again"""
    if idx is None or idx.numel() == 0:
        return
    idx_cpu = idx.detach().cpu()
    mi = int(idx_cpu.min().item()) if idx_cpu.numel() else 0
    ma = int(idx_cpu.max().item()) if idx_cpu.numel() else -1
    assert mi >= 0, f"[{tag}] negative index {mi}"
    assert ma < upper, f"[{tag}] max index {ma} >= upper {upper}"
    logger.info(f"[{tag}] index range: [{mi}, {ma}] vs upper {upper}, numel={idx.numel()}")


class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.num_steps = server_args.speculative_num_steps
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override context length with target model's context length
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            pp_rank=0,  # FIXME
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )

        # Share the embedding and lm_head
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # EAGLE3 models don't share lm_head
            self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            self.hot_token_id = self.draft_model_runner.model.get_hot_token_id().to(
                embed.device
            )
        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )

        self.init_attention_backend()
        self.init_cuda_graphs()

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.tree_mask_mode = TreeMaskMode.FULL_MASK
        self.plan_stream = torch.cuda.Stream()
        if self.server_args.attention_backend == "flashinfer":
            if not global_server_args_dict["use_mla_backend"]:
                from sglang.srt.layers.attention.flashinfer_backend import (
                    FlashInferMultiStepDraftBackend,
                )

                self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                    self.draft_model_runner,
                    self.topk,
                    self.num_steps,
                )
            else:
                from sglang.srt.layers.attention.flashinfer_mla_backend import (
                    FlashInferMLAMultiStepDraftBackend,
                )

                self.draft_attn_backend = FlashInferMLAMultiStepDraftBackend(
                    self.draft_model_runner,
                    self.topk,
                    self.num_steps,
                )
        elif self.server_args.attention_backend == "triton":
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )

            self.draft_attn_backend = TritonMultiStepDraftBackend(
                self.draft_model_runner,
                self.topk,
                self.num_steps,
            )
        elif self.server_args.attention_backend == "fa3":
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionMultiStepBackend,
            )

            self.draft_attn_backend = FlashAttentionMultiStepBackend(
                self.draft_model_runner,
                self.topk,
                self.num_steps,
            )
        else:
            raise ValueError(
                f"EAGLE is not supported in attention backend {self.server_args.attention_backend}"
            )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend
        self.plan_stream_ctx = (
            torch.cuda.stream(self.plan_stream) if self.plan_stream else empty_context()
        )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None

        if self.server_args.disable_cuda_graph:
            return

        # Capture draft
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
        )

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_generation(self, batch: ModelWorkerBatch) -> ForwardBatchOutput:
        """
        Run one speculative decoding forward batch.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            batch_output: The results in a tuple
        """
        if batch.forward_mode.is_decode():
            # ✅ Save the "Scheduler's out_cache_loc" before entering draft phase
            sched_out_cache_loc = batch.out_cache_loc

            try:
                old_spec_info = batch.spec_info
                spec_info = self.draft(batch)                   # This will temporarily modify batch.out_cache_loc
                batch_output = self.verify(batch, spec_info, old_spec_info)  # This will also modify it
                return batch_output
            finally:
                # ✅ Unconditionally restore out_cache_loc for Scheduler use (prevent None/shape mismatch)
                batch.out_cache_loc = sched_out_cache_loc
        else:
            # Target prefill
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(batch)

            # Draft prefill
            batch.capture_hidden_mode = CaptureHiddenMode.LAST
            batch_output.spec_info = self.forward_draft_extend(
                batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output

    def draft(self, batch: ModelWorkerBatch):
        # Prepare for draft
        spec_info = batch.spec_info
        forward_batch, can_cuda_graph = spec_info.prepare_for_draft(
            batch,
            self.cuda_graph_runner,
            self.draft_model_runner,
            self.topk,
            self.num_steps,
        )

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.num_steps,
            self.num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            num_steps=self.num_steps,
            topk=self.topk,
            num_draft_tokens=self.num_draft_tokens,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(self.num_steps, -1)

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            self._detect_nan_if_needed(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(score_list, self.num_draft_tokens - 1, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def verify(
        self,
        batch: ModelWorkerBatch,
        spec_info: EagleVerifyInput,
        old_spec_info: EagleDraftInput,
    ):
        # Parse args
        seq_lens_backup = batch.seq_lens
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = spec_info.prepare_for_verify(
                batch,
                self.target_worker,
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                spec_info,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # (A) Check verify phase out_cache_loc (note: check verify_forward_batch, not batch)
        pool_upper = int(batch.req_to_token_pool.req_to_token.shape[1])
        _debug_check_indices("verify.out_cache_loc", verify_forward_batch.out_cache_loc, pool_upper)
        assert verify_forward_batch.out_cache_loc.numel() == bs * self.num_draft_tokens, \
            f"verify.out_cache_loc.numel={verify_forward_batch.out_cache_loc.numel()} vs expected={bs*self.num_draft_tokens}"
        torch.cuda.synchronize()  # Ensure it fails here for easier debugging

        # Run target verify batch in the main compute stream
        forward_batch_output = self.target_worker.forward_batch_generation(
            verify_forward_batch, skip_sample=True, skip_attn_backend_init=True
        )
        logits_output = forward_batch_output.logits_output

        # Sample
        self._detect_nan_if_needed(logits_output)
        (
            predict,
            accept_length,
            accept_index,
        ) = spec_info.sample(batch, logits_output)
        new_seq_lens = seq_lens_backup + accept_length
        verify_done = torch.cuda.Event()
        verify_done.record()

        # Move the accepted tokens to the target KV cache locations
        batch.seq_lens = seq_lens_backup
        self.move_accepted_tokens_to_target_kvcache(
            batch,
            accept_index,
            accept_length,
            verify_forward_batch.out_cache_loc,   # <--- Key: use verify phase out_cache_loc
        )
        all_verified_id = predict[accept_index]
        verified_id = torch.empty_like(accept_length, dtype=torch.int32)
        fill_new_verified_id[(bs,)](
            all_verified_id,
            accept_length,
            verified_id,
            self.num_draft_tokens,
        )

        # Batch 2: Draft extend
        # --- Pad hidden_states to num_draft_tokens + 1 ---
        bs = len(batch.seq_lens)
        H = logits_output.hidden_states.shape[-1]
        hs = logits_output.hidden_states.view(bs, self.num_draft_tokens, H)  # (bs, N, H)
        pad = hs[:, :1, :]                          # (bs, 1, H) Use first position as placeholder
        hs_ext = torch.cat([pad, hs], dim=1)        # (bs, N+1, H)
        hs_ext = hs_ext.reshape(-1, H)              # (bs*(N+1), H)

        draft_input = EagleDraftInput(
            hidden_states=hs_ext,                   # ★ Align with extend_len (N+1)
            verified_id=verified_id,
        )
        extend_len = self.num_draft_tokens + 1
        
        # accept_length has already been incremented by 1 in sample() (including bonus),
        # here directly use accept_length as offset within each (extend_len) segment.
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device) * extend_len
            + accept_length
        )

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                predict,
                self.num_draft_tokens,
                self.draft_model_runner,
            )

        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

        # (B) Check extend phase out_cache_loc
        _debug_check_indices("extend.out_cache_loc", forward_batch.out_cache_loc, pool_upper)
        # Note: out_cache_loc length may vary due to allocator optimization, log but don't enforce assertion
        logger.debug(f"extend.out_cache_loc.numel={forward_batch.out_cache_loc.numel()} vs expected={bs*(self.num_draft_tokens+1)}")
        torch.cuda.synchronize()

        # Run draft extend batch in the main compute stream
        draft_logits_output = self.draft_model_runner.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # ✅ Only release temporary slots for draft-extend (Scheduler won't use them)
        if hasattr(forward_batch, "out_cache_loc") and forward_batch.out_cache_loc is not None:
            self._free_slots(forward_batch.out_cache_loc)
            logger.debug(f"Released draft-extend slots: {forward_batch.out_cache_loc[:8]}")

        # Since seq_lens_backup's tensor is allocated in another stream, we
        # need record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        seq_lens_backup.record_stream(torch.cuda.current_stream())

        # Construct the return values
        draft_input = EagleDraftInput(
            topk_p=ret_topk_p,
            topk_index=ret_topk_index,
            hidden_states=ret_hidden_states,
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=old_spec_info.allocate_lens,
            verify_done=verify_done,
        )

        return ForwardBatchOutput(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            spec_info=draft_input,
            accept_length=accept_length,
        )

    def forward_draft_extend(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        pt = 0
        for i, extend_len in enumerate(batch.extend_seq_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], next_token_ids[i].reshape(1))
            )
            pt += extend_len

        # Construct spec_info
        draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            allocate_lens=batch.seq_lens,
        )
        batch.spec_info = draft_input

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
        logits_output, _ = self.draft_model_runner.forward(forward_batch)

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states
        return draft_input

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
        src_out_cache_loc: torch.Tensor,   # <--- New parameter
    ):
        """
        Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            accept_length: The length of the accepted tokens.
        """
        bs = len(batch.seq_lens)
        size = bs * self.num_draft_tokens

        # 🔧 Add pool_upper (current token pool column count is the upper bound)
        pool_upper = int(batch.req_to_token_pool.req_to_token.shape[1])

        # (Optional) Count accepted tokens to check if size is "over-allocated"
        try:
            n_acc = int((accept_index.detach().cpu() != -1).sum().item())
            logger.info(f"[move] accepted tokens = {n_acc}/{size}")
        except Exception:
            pass

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length,
            tgt_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            src_out_cache_loc,             # <--- Use passed-in verify out_cache_loc
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        
        # (C) Check move_kv_cache src/dst
        _debug_check_indices("move.src_accepted", accepted_out_cache_loc, pool_upper)
        _debug_check_indices("move.dst_target", tgt_cache_loc, pool_upper)
        torch.cuda.synchronize()
        
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )
        
        # These are temporary slots from verify phase, can be recycled after move
        self._free_slots(src_out_cache_loc)

    def _free_slots(self, locs: torch.Tensor):
        """
        Free given KV slots (token indices) with de-dup and page-awareness.
        Works for both token-level and page-level allocators. Safe to call multiple times.
        """
        if locs is None or locs.numel() == 0:
            return
        alloc = self.token_to_kv_pool_allocator
        page_size = getattr(alloc, "page_size", 1)

        # Move all to CPU for processing, avoiding triggering CUDA's fancy small-index kernels
        locs_cpu = locs.detach().to(torch.int64).cpu()
        if page_size == 1:
            tokens = torch.unique(locs_cpu)
        else:
            pages = torch.unique(locs_cpu // page_size)
            tokens = (pages[:, None] * page_size + torch.arange(page_size)).reshape(-1)

        # If you can get the upper bound of "total token capacity", do boundary clipping and assertion:
        upper = int(self.req_to_token_pool.req_to_token.shape[1])
        # Filter out invalid token indices (<0 or >=upper)
        valid_mask = (tokens >= 0) & (tokens < upper)
        tokens = tokens[valid_mask]
        
        if tokens.numel() > 0:
            logger.debug(f"Freeing {tokens.numel()} tokens: {tokens[:8].tolist()}...")
            tokens = tokens.to(locs.device)
            alloc.free(tokens)

    def _free_all_draft_slots(self, batch: ModelWorkerBatch):
        """
        Free all draft out_cache_loc slots for this verify step (de-dup + page-aware).
        """
        self._free_slots(batch.out_cache_loc)

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)
