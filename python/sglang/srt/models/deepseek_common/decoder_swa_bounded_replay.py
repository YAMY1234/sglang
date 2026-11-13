from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len_from_batch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.device import async_h2d

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def late_layer_dp_counts(
    forward_batch: ForwardBatch, local_late_rows: int, *, in_graph: bool
) -> Optional[List[int]]:
    # The TP-MoE gather / combine in the late layers spans every DP rank, so each
    # rank needs all ranks' tail row counts.
    early = forward_batch.global_num_tokens_padded_cpu
    if early is None:
        early = forward_batch.global_num_tokens_cpu
    if early is None:
        return None
    if len(early) == 1:
        # Local-only counts (MegaMoE with a DP LM head): no peer geometry to share.
        counts = [local_late_rows]
    elif in_graph:
        # The capture bakes the tail block and replays never run this code. Every
        # rank shares one bucket whenever the graph holds a DP gather.
        counts = [local_late_rows] * len(early)
    elif forward_batch.global_late_layer_num_tokens_cpu is not None:
        # Exchanged in ForwardBatch.prepare_mlp_sync_batch, which a peer replaying
        # the prefill graph reaches too; a gather here would wait for it forever.
        counts = forward_batch.global_late_layer_num_tokens_cpu
        assert counts[get_parallel().attn_dp_rank] == local_late_rows, (
            counts,
            local_late_rows,
        )
    else:
        # Without the exchange (no extend in the batch) every rank takes this
        # path; the CPU group avoids a device sync.
        counts = torch.empty(len(early), dtype=torch.int64)
        all_gather_single(
            counts,
            torch.tensor([local_late_rows], dtype=torch.int64),
            group=get_parallel().tp_group.cpu_group,
        )
        counts = counts.tolist()
    return None if counts == list(early) else counts


def enter_late_layer_dp(forward_batch: ForwardBatch, counts: List[int]) -> tuple:
    fb = forward_batch
    saved = (
        fb.global_num_tokens_cpu,
        fb.global_num_tokens_gpu,
        fb.global_num_tokens_padded_cpu,
        fb.global_num_tokens_for_logprob_cpu,
        fb.global_num_tokens_for_logprob_gpu,
        fb.global_dp_buffer_len,
        fb.dp_padding_mode,
        fb.dp_local_start_pos,
        fb.dp_local_num_tokens,
        fb.global_num_token_non_padded,
        fb.global_num_token_non_padded_cpu,
        fb.num_token_non_padded,
    )
    fb.global_num_tokens_cpu = counts
    # Device ops only: this runs inside prefill graph capture, where a host copy
    # would be recorded against pageable memory.
    gpu = fb.global_num_tokens_gpu
    if len(set(counts)) == 1:
        fb.global_num_tokens_gpu = torch.full_like(gpu, counts[0])
    else:
        fb.global_num_tokens_gpu = async_h2d(counts, dtype=gpu.dtype, device=gpu.device)
    fb.global_num_tokens_padded_cpu = counts
    # The MoE gatherv reads its sizes from the logprob counts when present.
    fb.global_num_tokens_for_logprob_cpu = counts
    fb.global_num_tokens_for_logprob_gpu = fb.global_num_tokens_gpu
    fb.global_dp_buffer_len = sum(counts)
    # Tail rows are real rows only, so the late layers gather variable lengths.
    fb.dp_padding_mode = DpPaddingMode.SUM_LEN
    fb.dp_local_start_pos = fb.dp_local_num_tokens = None
    # With attention TP=1 the invariant scalar also describes local MoE rows.
    # Preserve an idle peer's zero bound even when it carries fabricated rows.
    local_rows = counts[get_parallel().attn_dp_rank if len(counts) > 1 else 0]
    fb.global_num_token_non_padded_cpu = min(
        local_rows, fb.global_num_token_non_padded_cpu
    )
    if fb.global_num_token_non_padded is not None:
        # Clamp the runner's static scalar so a graph replay rebounds it per batch.
        fb.global_num_token_non_padded = torch.clamp(
            fb.global_num_token_non_padded, max=local_rows
        )
        fb.num_token_non_padded = fb.global_num_token_non_padded
    set_dp_buffer_len_from_batch(fb)
    return saved


def exit_late_layer_dp(forward_batch: ForwardBatch, saved: tuple) -> None:
    fb = forward_batch
    (
        fb.global_num_tokens_cpu,
        fb.global_num_tokens_gpu,
        fb.global_num_tokens_padded_cpu,
        fb.global_num_tokens_for_logprob_cpu,
        fb.global_num_tokens_for_logprob_gpu,
        fb.global_dp_buffer_len,
        fb.dp_padding_mode,
        fb.dp_local_start_pos,
        fb.dp_local_num_tokens,
        fb.global_num_token_non_padded,
        fb.global_num_token_non_padded_cpu,
        fb.num_token_non_padded,
    ) = saved
    set_dp_buffer_len_from_batch(fb)


def enter_verify_late_rows(
    forward_batch: ForwardBatch, late_metadata, late_rows
) -> tuple:
    """SGLANG_DSV41_LATE_TAIL_ROWS: switch the late layers of a target verify onto the
    verify rows plus the late-only pseudo-slots; returns what exit_verify_late_rows
    restores. Every row is real for the MoE: the pseudo-slots follow the graph's
    padded verify rows, so the non-padded prefix bound cannot describe them."""
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    assert get_parallel().attn_dp_size == 1 or not get_moe_a2a_backend().is_none(), (
        "late-only verify rows need an MoE that dispatches each rank's own rows"
    )
    fb = forward_batch
    backend = get_attn_backend()
    saved = (
        backend.forward_metadata,
        fb.out_cache_loc,
        fb.positions,
        fb.req_pool_indices,
        fb.seq_lens,
        fb.batch_size,
        fb.num_token_non_padded,
        fb.global_num_token_non_padded,
        fb.global_num_token_non_padded_cpu,
    )
    rows = late_rows.positions.shape[0]
    backend.forward_metadata = late_metadata
    fb.out_cache_loc = late_rows.out_cache_loc
    fb.positions = late_rows.positions
    fb.req_pool_indices = late_rows.req_pool_indices
    fb.seq_lens = late_rows.seq_lens
    fb.batch_size = late_rows.req_pool_indices.shape[0]
    if fb.num_token_non_padded is not None:
        fb.num_token_non_padded = torch.full_like(fb.num_token_non_padded, rows)
    if fb.global_num_token_non_padded is not None:
        fb.global_num_token_non_padded = torch.full_like(
            fb.global_num_token_non_padded, rows
        )
    if fb.global_num_token_non_padded_cpu is not None:
        fb.global_num_token_non_padded_cpu = rows
    return saved


def exit_verify_late_rows(forward_batch: ForwardBatch, saved: tuple) -> None:
    from sglang.srt.model_executor.forward_context import get_attn_backend

    fb = forward_batch
    (
        get_attn_backend().forward_metadata,
        fb.out_cache_loc,
        fb.positions,
        fb.req_pool_indices,
        fb.seq_lens,
        fb.batch_size,
        fb.num_token_non_padded,
        fb.global_num_token_non_padded,
        fb.global_num_token_non_padded_cpu,
    ) = saved
