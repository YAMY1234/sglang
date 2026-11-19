"""Messages between the encoder-only prefills (stage 0) and the tail server (stage 1).

Stage 1 binds one ZMQ ROUTER; each stage-0 scheduler connects a DEALER whose identity
is its stage0_id. Every message is one msgpack frame of a tagged struct.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import msgspec

TAIL_STAGE_PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    pass


class TailMessage(msgspec.Struct, tag_field="kind", kw_only=True, frozen=True):
    version: int = TAIL_STAGE_PROTOCOL_VERSION


class Hello(TailMessage, tag="hello"):
    """Stage 0 at start: where its bootstrap server is and how its pools look."""

    stage0_id: str
    bootstrap_addr: str
    # The exact payload stage 0 PUT to its own bootstrap server.
    registration: Dict[str, Any]
    mooncake_session: str
    page_size: int
    c1_kv_item_lens: List[int]
    c1_index_item_lens: List[int]
    residual_item_len: int
    window: int


class Welcome(TailMessage, tag="welcome"):
    """Stage 1, once registered as PP rank 1: where stage 0 writes a slot."""

    tail_session: str
    c1_kv_ptrs: List[int]
    c1_kv_item_lens: List[int]
    c1_index_ptrs: List[int]
    c1_index_item_lens: List[int]
    residual_ptr: int
    residual_item_len: int
    page_size: int
    num_slots: int
    registered_pp_rank: int = 1
    # Item lengths of the state entries stage 1 sends decode (DSV41_TAIL_SWA, then
    # DSV41_TAIL_DRAFT_AUX), for a decode stand-in to register matching buffers.
    tail_state_item_lens: List[List[int]] = []


class SlotRequest(TailMessage, tag="slot_request"):
    """One per room stage 0 bootstraps; e = 0 and late_fed False for pass-through."""

    room: int
    rid: str
    s: int
    e: int
    late_fed: bool
    tail_token_ids: Optional[List[int]] = None


class SlotGrant(TailMessage, tag="slot_grant"):
    """slot = -1 for a pass-through room. c1_pages cover [0, e - W) in position
    order; the tail server computes the window's own layer-20 c1 rows."""

    room: int
    slot: int
    residual_slot: int
    c1_pages: List[int]


class SlotDefer(TailMessage, tag="slot_defer"):
    room: int
    retry_ms: int


class TailReady(TailMessage, tag="tail_ready"):
    """Stage 0's writes into the slot have returned (ok), or it gave up (not ok)."""

    room: int
    slot: int
    ok: bool


class TailAbort(TailMessage, tag="tail_abort"):
    room: int


class TailDone(TailMessage, tag="tail_done"):
    room: int
    slot: int
    status: str  # "ok", "failed" or "aborted"
    reason: str = ""
    queue_ms: float = 0.0
    forward_ms: float = 0.0
    send_ms: float = 0.0


class Heartbeat(TailMessage, tag="heartbeat"):
    free_slots: int = 0
    queued: int = 0
    ready: int = 0
    tails_per_s: float = 0.0
    inflight: int = 0


class Error(TailMessage, tag="error"):
    reason: str
    room: Optional[int] = None


TailStageMessage = Union[
    Hello,
    Welcome,
    SlotRequest,
    SlotGrant,
    SlotDefer,
    TailReady,
    TailAbort,
    TailDone,
    Heartbeat,
    Error,
]

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(TailStageMessage)


def encode(msg: TailMessage) -> bytes:
    return _encoder.encode(msg)


def decode(data: bytes) -> TailStageMessage:
    try:
        msg = _decoder.decode(data)
    except msgspec.DecodeError as exc:
        raise ProtocolError(f"malformed tail-stage message: {exc}") from exc
    if msg.version != TAIL_STAGE_PROTOCOL_VERSION:
        raise ProtocolError(
            f"tail-stage protocol version {msg.version}, expected "
            f"{TAIL_STAGE_PROTOCOL_VERSION}"
        )
    return msg


def slot_request_error(req: SlotRequest, window: int) -> Optional[str]:
    """Why a request breaks the split rule both sides share, if it does."""
    if not req.late_fed:
        return None if req.e == 0 else f"pass-through room {req.room} with e={req.e}"
    if req.e % 2 or req.e < window + 2 or req.e >= req.s or req.e < req.s - 2:
        return f"room {req.room}: e={req.e} is not a late-fed split of s={req.s}"
    return None
