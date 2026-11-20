"""PP stage decoupling for the PD prefill loop (SGLANG_PP_DECOUPLE_HOST).

Send windows keep a bounded number of P2P send rounds un-waited, receive inboxes
run the blocking host receives of a relay channel on a background thread, and the
state digest lets a run check that every stage built the same batches and holds
the same request slots at the same micro-batch epoch
(SGLANG_DEBUG_PP_STATE_DIGEST_EVERY).
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import msgspec
import torch

logger = logging.getLogger(__name__)

DIGEST_LOG_PREFIX = "PP state digest: "


class PPSendWindow:
    """Outstanding send rounds of one P2P channel, oldest first. At most `bound`
    rounds stay un-waited; a round's works, and the tensors they hold, are
    dropped once it is waited."""

    def __init__(self, bound: int) -> None:
        assert bound >= 0, bound
        self._bound = bound
        self._rounds: Deque[Tuple[Any, ...]] = deque()

    def __len__(self) -> int:
        return len(self._rounds)

    def push(self, works: Sequence[Any]) -> int:
        """Queue one round, then wait on the oldest rounds beyond the bound;
        returns how many rounds were waited."""
        if works:
            # Copied: callers clear their work list after handing it over.
            self._rounds.append(tuple(works))
        waited = 0
        while len(self._rounds) > self._bound:
            self._wait_oldest()
            waited += 1
        return waited

    def drain(self) -> int:
        waited = len(self._rounds)
        while self._rounds:
            self._wait_oldest()
        return waited

    def _wait_oldest(self) -> None:
        for p2p_work in self._rounds.popleft():
            p2p_work.work.wait()


class PPSendWindows(msgspec.Struct, frozen=True):
    """One window per PD prefill send channel; every field is None when off."""

    req: Optional[PPSendWindow] = None
    proxy: Optional[PPSendWindow] = None
    output: Optional[PPSendWindow] = None
    transfer: Optional[PPSendWindow] = None
    release: Optional[PPSendWindow] = None

    @classmethod
    def create(cls, bound: int, tensor_bound: Optional[int] = None) -> "PPSendWindows":
        # Proxy and output rounds hold their device tensors until waited (~0.5 GB a
        # proxy round at a 16k chunk), so they get their own, smaller bound.
        tensor_bound = bound if tensor_bound is None else tensor_bound
        return cls(
            req=PPSendWindow(bound),
            proxy=PPSendWindow(tensor_bound),
            output=PPSendWindow(tensor_bound),
            transfer=PPSendWindow(bound),
            release=PPSendWindow(bound),
        )


class _InboxFailure(msgspec.Struct, frozen=True):
    exc: BaseException


class PPRecvInbox:
    """In-order receiver for one host P2P channel: a daemon thread runs the blocking
    `recv_one` back to back, so the sender's gloo send completes when the thread
    posts the receive. Gloo matches receives in posting order, so every receive of
    the channel must go through `pop`; `depth` bounds how far the thread reads ahead."""

    def __init__(self, recv_one: Callable[[], Any], depth: int, name: str) -> None:
        assert depth >= 1, depth
        self._recv_one = recv_one
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def ready(self) -> bool:
        return not self._queue.empty()

    def pop(self) -> Any:
        item = self._queue.get()
        if isinstance(item, _InboxFailure):
            # Keep failing on later pops too: the channel is unusable.
            self._queue.put(item)
            raise RuntimeError(
                f"PP receive thread {self._thread.name} failed"
            ) from item.exc
        return item

    def _run(self) -> None:
        try:
            while True:
                self._queue.put(self._recv_one())
        except BaseException as exc:
            # Surfaces on the scheduler thread at its next pop.
            self._queue.put(_InboxFailure(exc=exc))


_relay_inbox: Optional[PPRecvInbox] = None


def set_relay_inbox(inbox: Optional[PPRecvInbox]) -> None:
    """Route the PP request relay (world CPU group, from the previous stage)."""
    global _relay_inbox
    _relay_inbox = inbox


def relay_inbox() -> Optional[PPRecvInbox]:
    return _relay_inbox


# ===== State digest =====


class PPStateDigest(msgspec.Struct, frozen=True):
    pp_rank: int
    # Attention DP ranks schedule independently; stages are compared per DP rank.
    dp_rank: int
    epoch: int
    # Over the rank-invariant parts only; this is what stages are compared on.
    digest: str
    # Every part's own digest, including informational ones that may legitimately
    # differ across stages (device page order once HiCache write-back runs).
    parts: Dict[str, str] = msgspec.field(default_factory=dict)


class PPDigestReport(msgspec.Struct, frozen=True):
    # (epoch, dp_rank) -> {pp_rank: digest}, only where the logged digests differ.
    mismatched: Dict[Tuple[int, int], Dict[int, str]]
    # (epoch, dp_rank) logged by fewer than pp_size stages, e.g. a log cut short.
    incomplete: List[Tuple[int, int]]


def collect_pp_state(
    *, token_to_kv_pool_allocator, req_to_token_pool, tree_cache
) -> Tuple[Tuple[int, ...], Tuple[Optional[torch.Tensor], ...]]:
    """Sizes and index tensors that must match on every PP stage at one epoch:
    allocator free lists in order, request slots, and the radix tree's sizes,
    locked size and node values in traversal order."""
    tree_tokens = tree_cache.evictable_size() + tree_cache.protected_size()
    sizes = [
        token_to_kv_pool_allocator.available_size(),
        req_to_token_pool.available_size(),
        tree_cache.evictable_size(),
        tree_cache.protected_size(),
    ]
    total = tree_cache.total_size()
    sizes.extend(total if isinstance(total, tuple) else (total,))
    tensors = [
        token_to_kv_pool_allocator.free_pages,
        token_to_kv_pool_allocator.release_pages,
        # RadixCache.all_values_flatten raises on an empty tree.
        tree_cache.all_values_flatten() if tree_tokens else None,
    ]
    return tuple(int(size) for size in sizes), tuple(tensors)


class PPBatchLog:
    """Rolling digest of the batches this stage built, in build order. Every stage
    builds the same batches (rids, prefix and extend lengths) whatever its pace,
    because the proxy tensors it receives are shaped by them."""

    def __init__(self) -> None:
        self._hasher = hashlib.blake2b(digest_size=8)

    def add(self, batch: Any) -> None:
        if batch is None:
            self._hasher.update(b"-")
            return
        rows = [
            [req.rid, int(prefix), int(extend)]
            for req, prefix, extend in zip(
                batch.reqs, batch.prefix_lens, batch.extend_lens
            )
        ]
        self._hasher.update(json.dumps(rows).encode())

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def slot_map_digest(reqs: Iterable[Any]) -> str:
    """Request -> request-pool slot over the live requests, in any collection order."""
    slots = sorted(
        {
            (req.rid, int(req.req_pool_idx))
            for req in reqs
            if req.req_pool_idx is not None
        }
    )
    return hashlib.blake2b(json.dumps(slots).encode(), digest_size=8).hexdigest()


def combine_parts(parts: Dict[str, str], invariant: Sequence[str]) -> str:
    joined = "|".join(f"{name}={parts[name]}" for name in invariant)
    return hashlib.blake2b(joined.encode(), digest_size=8).hexdigest()


def pp_state_digest(
    *, sizes: Sequence[int], tensors: Sequence[Optional[torch.Tensor]]
) -> str:
    """Order-sensitive digest; copies each tensor to the host once."""
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(json.dumps(list(sizes)).encode())
    for tensor in tensors:
        if tensor is None:
            hasher.update(b"none")
            continue
        values = tensor.detach().to(device="cpu", dtype=torch.int64).contiguous()
        hasher.update(values.numel().to_bytes(8, "little"))
        hasher.update(values.numpy().tobytes())
    return hasher.hexdigest()


def format_digest_line(record: PPStateDigest) -> str:
    return DIGEST_LOG_PREFIX + msgspec.json.encode(record).decode()


def parse_digest_lines(lines: Iterable[str]) -> List[PPStateDigest]:
    records = []
    for line in lines:
        _, sep, payload = line.partition(DIGEST_LOG_PREFIX)
        if sep:
            records.append(msgspec.json.decode(payload.strip(), type=PPStateDigest))
    return records


def first_part_mismatches(
    records: Iterable[PPStateDigest],
) -> Dict[Tuple[str, int], int]:
    """(part, dp_rank) -> first epoch where that part differs across stages."""
    by_epoch: Dict[Tuple[int, int], List[PPStateDigest]] = defaultdict(list)
    for record in records:
        by_epoch[(record.epoch, record.dp_rank)].append(record)
    first: Dict[Tuple[str, int], int] = {}
    for (epoch, dp_rank), group in sorted(by_epoch.items()):
        names = {name for record in group for name in record.parts}
        for name in names:
            if (name, dp_rank) not in first and (
                len({record.parts.get(name) for record in group}) > 1
            ):
                first[(name, dp_rank)] = epoch
    return first


def find_digest_mismatches(
    records: Iterable[PPStateDigest], pp_size: int
) -> PPDigestReport:
    """Group digests by epoch, not by log order: stages reach an epoch at
    different times once they are decoupled."""
    by_epoch: Dict[Tuple[int, int], Dict[int, str]] = defaultdict(dict)
    for record in records:
        by_epoch[(record.epoch, record.dp_rank)][record.pp_rank] = record.digest
    groups = sorted(by_epoch.items())
    mismatched = {
        key: stages for key, stages in groups if len(set(stages.values())) > 1
    }
    incomplete = [key for key, stages in groups if len(stages) < pp_size]
    return PPDigestReport(mismatched=mismatched, incomplete=incomplete)
