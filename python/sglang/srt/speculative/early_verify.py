"""FN TP4: plan verify before draft replay; publish only its tokens afterwards.

Planning reads the committed prefix and fixed chain topology, never the draft
token values. The target model and its state readers still run after draft.
"""

import logging
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    prepare_eagle_verify_forward,
)

logger = logging.getLogger(__name__)


@dataclass
class EarlyVerify:
    verify_input: object
    prepared_forward: tuple
    target_input_ids: torch.Tensor


@triton.jit
def _publish_verify_tokens(
    BONUS,
    DRAFT,
    TREE,
    TARGET,
    DRAFT_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = tl.arange(0, BLOCK)
    drafted = tl.load(
        DRAFT + (col - 1) * DRAFT_STRIDE,
        mask=(col > 0) & (col < WIDTH),
        other=0,
    )
    root = tl.load(BONUS)
    tokens = tl.where(col == 0, root, drafted)
    tl.store(TREE + col, tokens, col < WIDTH)
    tl.store(TARGET + col, tokens, col < WIDTH)


def prepare_early_verify(
    worker,
    batch,
    draft_input,
    draft_forward_batch,
    kv_loc_plan,
    *,
    can_run_decode_cuda_graph,
    late_commit_join,
):
    """Return None without changing the batch when this fixed-shape path misses."""
    target = worker.target_worker.model_runner
    runner = target.decode_cuda_graph_runner
    parallel = get_parallel()
    if (
        not can_run_decode_cuda_graph
        or late_commit_join
        # Those native flags make draft positions/lengths/slots and hidden
        # private. Preloading the shared target buffers must not overwrite them.
        or envs.SGLANG_PERFLL_STEP2.get() < 3
        or not envs.SGLANG_PERFLL_STEP5A.get()
        or not batch.forward_mode.is_decode()
        or len(batch.seq_lens) != 1
        or not batch.seq_lens.is_cuda
        or batch.seq_lens_cpu is not None
        or batch.has_grammar
        or batch.return_logprob
        or any(x is not None for x in batch.multimodal_inputs)
        or worker.topk != 1
        or worker.speculative_num_steps != 3
        or worker.speculative_num_draft_tokens != 4
        or worker.tree_mask_mode != TreeMaskMode.QLEN_ONLY
        or get_spec().speculative_use_rejection_sampling
        or get_spec().speculative_adaptive
        or kv_loc_plan is None
        or worker.draft_owns_attention
        or parallel.tp_group.world_size != 4
        or parallel.pp_group.world_size != 1
        or parallel.attn_dp_enabled
        or parallel.dp_size != 1
        or parallel.moe_ep_size != 1
        or not isinstance(target.attn_backend, QwenSparseAttnBackend)
        or runner is None
        or 1 not in runner.capture_bs
        or runner.require_mlp_tp_gather
        or runner.enable_two_batch_overlap
        or runner.enable_pdmux
        or runner.ragged_verify_mode
        or target.lora_manager is not None
        or target.canary_manager is not None
        or worker.draft_runner.canary_manager is not None
        or draft_forward_batch.batch_size != 1
        or kv_loc_plan.is_translated_for(target.kv_index_translator)
    ):
        return None

    placeholder = getattr(worker, "_q38c_verify_placeholder", None)
    if placeholder is None:
        # These values are never consumed by the target. They let the native
        # planner build topology/positions before the real tokens exist.
        placeholder = torch.zeros((1, 3), dtype=torch.int64, device=batch.device)
        worker._q38c_verify_placeholder = placeholder

    verify_input = build_eagle_verify_input(
        batch,
        draft_input,
        None,
        None,
        placeholder,
        None,
        target_worker=worker.target_worker,
        topk=1,
        num_steps=3,
        num_draft_tokens=4,
        tree_mask_mode=worker.tree_mask_mode,
        device=worker.device,
    )
    verify_input.kv_loc_plan = kv_loc_plan
    verify_input.prepared_out_cache_loc = kv_loc_plan.write_virtual
    # prepare_for_draft already produced a distinct ForwardBatch holding the
    # original draft spec_info; changing ScheduleBatch does not rebind it.
    batch.spec_info = verify_input
    prepared = prepare_eagle_verify_forward(
        batch,
        target_worker=worker.target_worker,
        req_to_token_pool=worker.req_to_token_pool,
        plan_stream=worker.plan_stream,
        plan_stream_ctx=worker.plan_stream_ctx,
        device=worker.device,
    )
    target_ids = (
        runner.buffers.input_ids[:4] if prepared[1] else verify_input.draft_token
    )
    if not getattr(worker, "_q38c_early_verify_logged", False):
        logger.info(
            "Q38C_EARLY_VERIFY_READY tp=4 bs=1 width=4 graph=%s "
            "late_commit_join=0 token_publish=device",
            prepared[1],
        )
        worker._q38c_early_verify_logged = True
    return EarlyVerify(verify_input, prepared, target_ids)


def publish_early_verify(early, bonus_tokens, draft_tokens):
    """Same-stream producer edge: no host token read or new synchronization."""
    _publish_verify_tokens[(1,)](
        bonus_tokens,
        draft_tokens,
        early.verify_input.draft_token,
        early.target_input_ids,
        DRAFT_STRIDE=draft_tokens.stride(1),
        WIDTH=4,
        BLOCK=32,
        num_warps=1,
        enable_fp_fusion=False,
        launch_pdl=False,
    )
    early.verify_input._q38c_prepared_verify = early.prepared_forward
    return early.verify_input
