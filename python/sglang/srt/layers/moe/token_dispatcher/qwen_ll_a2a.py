"""Fence-free MoE all-to-all for the FlashInfer dispatcher's decode path: target NVFP4 dispatch
(SGLANG_Q35MF_E2X_LLA2A) and draft BF16 dispatch (SGLANG_Q35MF_E2X_LLA2A_DRAFT).

Rows reach the MoE runner in the shapes and dtypes FlashInfer's dispatch produces, but at fixed positions:
source s token t is row s * tmax + t, and every row without an expert on this rank carries the invalid id.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.communication.qwen_dep_ll_a2a import (
    LL_A2A_MAX_TOKENS,
    ll_a2a_combine,
    ll_a2a_dispatch,
    ll_a2a_region_bytes,
)

logger = logging.getLogger(__name__)

_CHECK_FIELDS = ("calls", "rows", "dispatch_rows_diff", "combine_rows_diff")
# Per namespace (False = target, True = draft): one state and one checker each.
_states: dict[bool, LlA2AState] = {}
_checks: dict[bool, LlA2ACheck] = {}


class LlA2AState:
    """Exchange regions (MNNVL, mapped on every EP rank), dense receive rows and per-CTA call counters.

    One instance serves every MoE layer of a namespace (target or draft): their calls run in the same order on
    every rank. payload "nvfp4": x uint8 [tokens, H / 2] + scales uint8 [tokens, H / 16]; "bf16": x bf16 [tokens, H].
    """

    def __init__(
        self,
        *,
        mapping,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        hidden: int,
        num_experts: int,
        slots: int,
        payload: str,
    ) -> None:
        from flashinfer.comm.mnnvl import MnnvlMemory

        assert 0 < slots <= LL_A2A_MAX_TOKENS and num_experts % ep_size == 0
        assert payload in ("nvfp4", "bf16")
        self.payload = payload
        self.x_bytes = hidden // 2 if payload == "nvfp4" else hidden * 2
        self.sf_bytes = hidden // 16 if payload == "nvfp4" else 0
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.hidden = hidden
        self.num_experts = num_experts
        self.experts_per_rank = num_experts // ep_size
        self.slots = slots
        dispatch_bytes, combine_bytes = ll_a2a_region_bytes(
            ep_size, top_k, hidden, slots, self.x_bytes, self.sf_bytes
        )
        # MoeAlltoAll's constructor already initialized MNNVL and its communicator for this mapping.
        self._mem = MnnvlMemory(mapping, dispatch_bytes + combine_bytes)
        regions = self._mem.as_torch_strided_tensor(torch.uint8)
        local = regions[ep_rank]
        local[:dispatch_bytes].zero_()
        local[dispatch_bytes : dispatch_bytes + combine_bytes].fill_(0xFF)
        torch.cuda.synchronize()
        # No rank may write into a peer's region before that peer has initialized it.
        MnnvlMemory.allocated_map[self._mem.ptr].comm.barrier()
        bases = [regions[r].data_ptr() for r in range(ep_size)]
        # Read by the launcher on the host; model init runs under a CUDA default device, so say cpu.
        self.dispatch_peers = torch.tensor(bases, dtype=torch.int64, device="cpu")
        self.combine_peers = torch.tensor(
            [b + dispatch_bytes for b in bases], dtype=torch.int64, device="cpu"
        )
        device = regions.device
        # Every CTA spins on peer data, so the grid must fit on the GPU at once.
        ctas = min(128, torch.cuda.get_device_properties(device).multi_processor_count)
        self.dispatch_ctr = torch.zeros(ctas, dtype=torch.int32, device=device)
        self.combine_ctr = torch.zeros(ctas, dtype=torch.int32, device=device)
        self.err = torch.zeros(1, dtype=torch.int32, device=device)
        rows = ep_size * slots
        self.out_x = torch.empty((rows, self.x_bytes), dtype=torch.uint8, device=device)
        self.out_sf = torch.empty(
            (rows, self.sf_bytes), dtype=torch.uint8, device=device
        )
        self.out_ids = torch.empty((rows, top_k), dtype=torch.int32, device=device)
        self.out_wts = torch.empty((rows, top_k), dtype=torch.float32, device=device)
        self.recv_tokens = torch.zeros(ep_size, dtype=torch.int32, device=device)
        logger.info(
            "SGLANG_Q35MF_E2X_LLA2A buffers (%s): %d ranks, %d slots, %d + %d bytes, %d CTAs",
            payload,
            ep_size,
            slots,
            dispatch_bytes,
            combine_bytes,
            ctas,
        )

    def matches(self, x: torch.Tensor, sf: Optional[torch.Tensor]) -> bool:
        if self.payload == "nvfp4":
            return sf is not None and x.element_size() == 1
        return sf is None and x.dtype == torch.bfloat16

    def dispatch(
        self,
        *,
        x: torch.Tensor,
        sf: Optional[torch.Tensor],
        ids: torch.Tensor,
        wts: torch.Tensor,
        tmax: int,
        invalid_id: int,
    ) -> list[torch.Tensor]:
        """Returns the received payloads as [ep_size, tmax, ...] views in MoeAlltoAll.dispatch order:
        x, sf, ids, weights for nvfp4; x, ids, weights for bf16."""
        rows = self.ep_size * tmax
        out = [
            self.out_x[:rows],
            self.out_sf[:rows],
            self.out_ids[:rows],
            self.out_wts[:rows],
        ]
        if sf is None:
            sf = x.new_empty((x.shape[0], 0), dtype=torch.uint8)
        ll_a2a_dispatch(
            self.dispatch_peers,
            x.view(torch.uint8),
            sf.view(torch.uint8),
            ids,
            wts,
            self.dispatch_ctr,
            *out,
            self.recv_tokens,
            self.err,
            tmax=tmax,
            slots=self.slots,
            rank=self.ep_rank,
            experts_per_rank=self.experts_per_rank,
            invalid_id=invalid_id,
            hidden=self.hidden,
        )
        out[0] = out[0].view(x.dtype)
        out[1] = out[1].view(sf.dtype)
        if self.payload == "bf16":
            del out[1]
        return [t.view(self.ep_size, tmax, -1) for t in out]

    def combine(
        self, *, partial: torch.Tensor, ids: torch.Tensor, tmax: int
    ) -> torch.Tensor:
        """partial: the MoE output rows of the last dispatch; ids: this rank's own top-k ids [tokens, top_k]."""
        out = torch.empty(
            (ids.shape[0], self.hidden), dtype=partial.dtype, device=partial.device
        )
        ll_a2a_combine(
            self.combine_peers,
            partial,
            self.out_ids[: self.ep_size * tmax],
            self.recv_tokens,
            ids,
            out,
            self.combine_ctr,
            self.err,
            tmax=tmax,
            slots=self.slots,
            rank=self.ep_rank,
            experts_per_rank=self.experts_per_rank,
            num_experts=self.num_experts,
            x_bytes=self.x_bytes,
            sf_bytes=self.sf_bytes,
        )
        return out


class LlA2ACheck:
    """SGLANG_Q35MF_E2X_LLA2A_CHECK: the stock FlashInfer exchange of the same rows beside the new one, in the same
    (captured) forward. Per local token count: calls, FlashInfer rows compared, dispatch and combine rows that differ."""

    def __init__(self, device: torch.device, tag: str) -> None:
        from sglang.srt.layers.moe.qwen_router_front import start_check_reporter

        self.counts = torch.zeros(
            (LL_A2A_MAX_TOKENS + 1, len(_CHECK_FIELDS)),
            dtype=torch.int64,
            device=device,
        )
        start_check_reporter(
            f"SGLANG_Q35MF_E2X_LLA2A_CHECK {tag}", self.counts, _CHECK_FIELDS
        )
        self._rows: Optional[torch.Tensor] = None
        self._valid: Optional[torch.Tensor] = None
        self._dispatch_diff: Optional[torch.Tensor] = None

    def dispatch(
        self,
        *,
        moe_a2a,
        payloads: list[torch.Tensor],
        expert_id_payload_index: int,
        invalid_id: int,
        num_experts: int,
        ll_recv: list[torch.Tensor],
        tmax: int,
    ) -> None:
        ids = payloads[expert_id_payload_index]
        tokens = ids.shape[0]
        # Stock dispatch with the source token index as one more payload, to find each stock row's fixed slot.
        tag = torch.arange(tokens, dtype=torch.int32, device=ids.device).view(tokens, 1)
        recv = moe_a2a.dispatch(
            ids,
            payloads + [tag],
            tmax,
            invalid_token_expert_id=invalid_id,
            expert_id_payload_index=expert_id_payload_index,
        )
        ep = ll_recv[0].shape[0]
        stock_ids = recv[expert_id_payload_index]
        valid = (stock_ids[..., 0] >= 0) & (stock_ids[..., 0] < num_experts)
        source = torch.arange(ep, dtype=torch.int64, device=ids.device).view(ep, 1)
        rows = (source * tmax + recv[-1][..., 0].long()).clamp_(0, ep * tmax - 1)
        differ = torch.zeros_like(valid)
        for ll_t, stock_t in zip(ll_recv, recv[:-1]):
            mine = (
                ll_t.reshape(ep * tmax, -1)[rows]
                .view(torch.uint8)
                .reshape(ep, tmax, -1)
            )
            differ |= (
                mine != stock_t.contiguous().view(torch.uint8).reshape(ep, tmax, -1)
            ).any(-1)
        ll_ids = ll_recv[expert_id_payload_index].reshape(ep * tmax, -1)[:, 0]
        ll_routed = ((ll_ids >= 0) & (ll_ids < num_experts)).sum()
        self._dispatch_diff = (differ & valid).sum() + (ll_routed - valid.sum()).abs()
        self._rows, self._valid = rows, valid

    def combine(
        self, *, moe_a2a, partial: torch.Tensor, out: torch.Tensor, tmax: int
    ) -> None:
        ep = self._rows.shape[0]
        payload = partial[self._rows.view(-1)].view(ep, tmax, -1)
        ref = moe_a2a.combine(payload, tmax, payload_in_workspace=False)
        combine_diff = (ref.view(torch.int16) != out.view(torch.int16)).any(-1).sum()
        one = torch.ones((), dtype=torch.int64, device=out.device)
        self.counts[min(out.shape[0], LL_A2A_MAX_TOKENS)].add_(
            torch.stack([one, self._valid.sum(), self._dispatch_diff, combine_diff])
        )
        self._rows = self._valid = self._dispatch_diff = None


def get_ll_a2a_state(*, draft: bool, **kwargs) -> LlA2AState:
    if draft not in _states:
        _states[draft] = LlA2AState(**kwargs)
    return _states[draft]


def get_ll_a2a_check(device: torch.device, *, draft: bool) -> LlA2ACheck:
    if draft not in _checks:
        _checks[draft] = LlA2ACheck(device, "draft" if draft else "target")
    return _checks[draft]
