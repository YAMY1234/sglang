"""Stage the next EAGLE draft inputs in the preceding post-verify graph."""

from __future__ import annotations

from functools import cache
from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, get_schedule, get_spec


@cache
def nvidia_eagle_decode_fast_path_enabled() -> bool:
    """Whether the proven TP megakernel decode fast path applies to this worker."""
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled

    return (
        megakernel_enabled()
        and (get_spec().speculative_algorithm or "").upper() == "EAGLE"
        and (get_spec().speculative_eagle_topk or 1) == 1
        and envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.get()
        and not get_schedule().disable_overlap_schedule
        and get_parallel().moe_ep_size == 1
    )


class PreparedNextDraftInputs(msgspec.Struct, frozen=True, kw_only=True):
    """Inputs and KV write locations prepared for one subsequent draft round."""

    num_requests: int
    seq_lens: torch.Tensor
    positions: torch.Tensor
    mrope_positions: Optional[torch.Tensor]
    draft_out_cache_loc: torch.Tensor
    write_window: torch.Tensor


class _PreparedNextDraftState(msgspec.Struct, frozen=True, kw_only=True):
    request_slots: tuple[int, ...]
    uses_cuda_graph: bool
    has_reserved_window: bool
    inputs: Optional[PreparedNextDraftInputs]
    seq_lens: Optional[torch.Tensor]


def allocate_prepared_next_draft_inputs(
    *,
    num_requests: int,
    num_steps: int,
    window_size: int,
    with_mrope: bool,
    device,
    runner_buffers,
) -> PreparedNextDraftInputs:
    """Bind draft fields to graph-runner inputs and own the full write window."""

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.int64, device=device)

    return PreparedNextDraftInputs(
        num_requests=num_requests,
        seq_lens=runner_buffers.seq_lens[:num_requests],
        positions=runner_buffers.positions[:num_requests],
        mrope_positions=(
            runner_buffers.mrope_positions[:, :num_requests] if with_mrope else None
        ),
        draft_out_cache_loc=runner_buffers.out_cache_loc[: num_requests * num_steps],
        write_window=zeros(num_requests * window_size),
    )


def prepare_next_draft_inputs(
    prepared: PreparedNextDraftInputs,
    *,
    req_pool_indices: torch.Tensor,
    new_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    num_steps: int,
    window_size: int,
    seq_len_fill_value: int,
) -> None:
    """Populate the next draft's fields from the accepted lengths."""
    from sglang.kernels.ops.speculative.cache_locs import prepare_draft_window

    prepare_draft_window(
        req_pool_indices,
        new_seq_lens,
        req_to_token,
        num_steps=num_steps,
        window=window_size,
        seq_len_fill_value=seq_len_fill_value,
        seq_lens_out=prepared.seq_lens,
        positions=prepared.positions,
        mrope=prepared.mrope_positions,
        draft_out_cache_loc=prepared.draft_out_cache_loc,
        write_virtual=prepared.write_window,
    )


def can_use_prepared_next_draft_inputs(
    prepared: PreparedNextDraftInputs,
    *,
    num_requests: int,
    plans_window: bool,
    with_mrope: bool,
) -> bool:
    """Whether the prepared buffers exactly describe this draft round."""
    return (
        plans_window
        and prepared.num_requests == num_requests
        and (prepared.mrope_positions is not None) == with_mrope
    )


class NextDraftInputStager:
    """Match a decode round with inputs prepared by its immediate predecessor."""

    def __init__(self, *, window_size: int, graph_sizes) -> None:
        self.window_size = window_size
        self.graph_sizes = frozenset(graph_sizes or ())
        self._state: Optional[_PreparedNextDraftState] = None

    def note_decode(
        self,
        batch,
        *,
        prepared: Optional[PreparedNextDraftInputs],
        new_seq_lens: Optional[torch.Tensor],
    ) -> None:
        required = 3 * self.window_size
        self._state = _PreparedNextDraftState(
            request_slots=tuple(req.kv.req_pool_idx for req in batch.reqs),
            uses_cuda_graph=len(batch.reqs) in self.graph_sizes,
            has_reserved_window=all(
                req.kv.kv_allocated_len >= req.kv.kv_committed_len + required
                for req in batch.reqs
            ),
            inputs=prepared,
            seq_lens=new_seq_lens,
        )

    def forget(self) -> None:
        self._state = None

    def _matches(self, batch) -> bool:
        state = self._state
        return bool(
            state is not None
            and state.request_slots == tuple(req.kv.req_pool_idx for req in batch.reqs)
            and state.uses_cuda_graph
            and state.has_reserved_window
            and state.inputs is not None
        )

    def take_seq_lens(self, batch) -> Optional[torch.Tensor]:
        if not self._matches(batch):
            return None
        return self._state.seq_lens

    def take_inputs(self, batch) -> Optional[PreparedNextDraftInputs]:
        state, matches = self._state, self._matches(batch)
        self._state = None
        if not matches or state is None:
            return None
        return state.inputs
