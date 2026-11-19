"""The tail server's place in each stage-0 bootstrap table, and who sends which entry.

Stage 0 covers layers [0, 21) and stage 1 layers [20, 40), so ownership goes per PD
state entry, not per layer range: stage 0 keeps its wire W0 and the metadata row;
stage 1 registers W0 (sending none of it) followed by the tail entries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import msgspec
from sglang.srt.disaggregation.base.conn import StateType

TAIL_STAGE_PP_SIZE = 2
TAIL_STAGE_PP_RANK = 1


def tail_entries(draft_aux: bool) -> List[List[Optional[str]]]:
    """The wire entries stage 1 owns, appended after W0 on stage 1 and decode."""
    entries = [[StateType.DSV41_TAIL_SWA.value, None]]
    if draft_aux:
        entries.append([StateType.DSV41_TAIL_DRAFT_AUX.value, None])
    return entries


def tail_stage_wire(
    stage0_wire: Optional[List[list]], draft_aux: bool
) -> List[List[Optional[str]]]:
    return [list(entry) for entry in stage0_wire or []] + tail_entries(draft_aux)


def entry_sender(index: int, stage0_wire_len: int) -> int:
    """The PP rank that sends wire entry `index`."""
    return 0 if index < stage0_wire_len else TAIL_STAGE_PP_RANK


def stage1_state_indices(
    *,
    stage0_wire_len: int,
    draft_aux: bool,
    swa_rows: Optional[List[int]],
    aux_slot: Optional[int],
) -> List[Optional[List[int]]]:
    """Stage 1's per-entry payloads for one room. None skips an entry: every W0
    entry (stage 0 sends them), and every entry of a pass-through room
    (swa_rows None)."""
    payloads: List[Optional[List[int]]] = [None] * stage0_wire_len
    if swa_rows is None:
        return payloads + [None] * len(tail_entries(draft_aux))
    payloads.append(list(swa_rows))
    if draft_aux:
        assert aux_slot is not None and aux_slot >= 0
        payloads.append([aux_slot])
    return payloads


def stage1_registration(
    stage0_payload: Dict[str, Any],
    *,
    rank_ip: str,
    rank_port: int,
    state_wire: List[list],
) -> Dict[str, Any]:
    """Stage 1's PUT to one stage-0 bootstrap server, from the payload stage 0 PUT
    there: same DP group and pool settings, so the server's consistency checks pass."""
    if stage0_payload.get("pp_size") != TAIL_STAGE_PP_SIZE:
        raise ValueError(
            f"stage 0 advertises pp_size {stage0_payload.get('pp_size')}, "
            f"the tail stage needs {TAIL_STAGE_PP_SIZE}"
        )
    if stage0_payload.get("pp_rank") != 0:
        raise ValueError("the tail stage pairs with PP rank 0 only")
    for key in ("attn_tp_size", "attn_cp_size"):
        if stage0_payload.get(key) != 1:
            # Stage 1 registers one rank per DP group: the table would wait
            # for tp x cp of them.
            raise ValueError(
                f"the tail stage needs stage 0 {key} 1, got {stage0_payload.get(key)}"
            )
    payload = dict(stage0_payload)
    payload.update(
        pp_size=TAIL_STAGE_PP_SIZE,
        pp_rank=TAIL_STAGE_PP_RANK,
        rank_ip=rank_ip,
        rank_port=rank_port,
        state_wire=state_wire,
    )
    return payload


def expected_registrations(payload: Dict[str, Any]) -> int:
    """What CommonKVBootstrapServer._is_ready waits for, from one PUT payload."""
    dp_size = (
        payload["attn_dp_size"]
        if payload["system_dp_size"] == 1
        else payload["system_dp_size"]
    )
    return (
        dp_size * payload["attn_cp_size"] * payload["attn_tp_size"] * payload["pp_size"]
    )


class Stage0Peer(msgspec.Struct):
    stage0_id: str
    bootstrap_addr: str
    registration: Dict[str, Any]
    last_seen: float
    welcomed: bool = False


class Stage0Peers:
    """Stage 0 peers by stage0_id; every peer must carry the same W0 and pools."""

    def __init__(self, *, page_size: int, window: int, draft_aux: bool) -> None:
        self.page_size = page_size
        self.window = window
        self.draft_aux = draft_aux
        self.stage0_wire: Optional[List[list]] = None
        self.layout: Optional[tuple] = None
        self.peers: Dict[str, Stage0Peer] = {}

    def check(self, hello) -> Optional[str]:
        """Why this Hello cannot pair with the tail stage, if it cannot."""
        if hello.page_size != self.page_size:
            return f"page_size {hello.page_size}, tail stage runs {self.page_size}"
        if hello.window != self.window:
            return f"window {hello.window}, tail stage runs {self.window}"
        wire = hello.registration.get("state_wire")
        if self.stage0_wire is not None and wire != self.stage0_wire:
            return (
                f"state wire {wire} differs from the first stage 0's {self.stage0_wire}"
            )
        layout = (
            tuple(hello.c1_kv_item_lens),
            tuple(hello.c1_index_item_lens),
            hello.residual_item_len,
        )
        if self.layout is not None and layout != self.layout:
            return (
                f"pool layout {layout} differs from the first stage 0's {self.layout}"
            )
        return None

    def add(self, hello, *, now: float) -> Stage0Peer:
        assert self.check(hello) is None
        self.stage0_wire = hello.registration.get("state_wire") or []
        self.layout = (
            tuple(hello.c1_kv_item_lens),
            tuple(hello.c1_index_item_lens),
            hello.residual_item_len,
        )
        peer = Stage0Peer(
            stage0_id=hello.stage0_id,
            bootstrap_addr=hello.bootstrap_addr,
            registration=dict(hello.registration),
            last_seen=now,
        )
        self.peers[hello.stage0_id] = peer
        return peer

    def wire(self) -> List[List[Optional[str]]]:
        return tail_stage_wire(self.stage0_wire, self.draft_aux)

    def touch(self, stage0_id: str, *, now: float) -> None:
        peer = self.peers.get(stage0_id)
        if peer is not None:
            peer.last_seen = now

    def silent(self, *, now: float, timeout_s: float) -> List[str]:
        return [
            stage0_id
            for stage0_id, peer in self.peers.items()
            if now - peer.last_seen > timeout_s
        ]
