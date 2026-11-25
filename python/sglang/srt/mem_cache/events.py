# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV cache placement event recording.

Produces the ``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared`` events
consumed by KV-aware routers (e.g. dynamo). A cache holds one recorder and calls
it; the recorder owns the queue and needs nothing back from its owner.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from array import array
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Iterable, Optional

import msgspec
import numpy as np

from sglang.srt.environ import envs
from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
    kv_event_namespace_seed,
    namespaced_block_hash,
)
from sglang.srt.mem_cache.utils import (
    compute_node_event_hash_values,
    compute_node_hash_values,
    hash_str_to_int64,
)

logger = logging.getLogger(__name__)

_KVEV_LAZY = os.environ.get("SGLANG_KV_EVENTS_LAZY", "0") == "1"
_KVEV_LAZY_VERIFY = int(os.environ.get("SGLANG_KV_EVENTS_LAZY_VERIFY", "0") or 0)

if TYPE_CHECKING:
    from sglang.srt.lora.lora_registry import LoRARef

# Bigram runs shorter than this keep the tuple list: numpy's per-call cost exceeds
# the tuple build below ~16 pages (desktop bench, raw-tokens/bench_raw.out).
_RAW_MIN_TOKENS = 1024
_RAW_TABLE_BITS = 18  # vocabularies up to 262,144 ids use the lookup table
_raw_table = None


def _msgpack_array_header(n: int) -> bytes:
    if n < 16:
        return bytes((0x90 | n,))
    if n < 1 << 16:
        return b"\xdc" + n.to_bytes(2, "big")
    return b"\xdd" + n.to_bytes(4, "big")


def _uint_msgpack_rows(v):
    """Per value: its msgpack uint encoding (fixint / 0xcc / 0xcd / 0xce, big-endian)
    left-aligned in 5 bytes, and the mask of used bytes."""
    v = v.astype(np.uint64)
    length = np.where(v < 128, 1, np.where(v < 256, 2, np.where(v < 65536, 3, 5)))
    cols = np.empty((v.size, 5), np.uint8)
    cols[:, 0] = np.where(length == 1, v, np.where(length == 2, 0xCC, np.where(length == 3, 0xCD, 0xCE)))
    cols[:, 1] = np.where(length == 2, v, np.where(length == 3, v >> 8, v >> 24)) & 255
    cols[:, 2] = np.where(length == 3, v, v >> 16) & 255
    cols[:, 3] = (v >> 8) & 255
    cols[:, 4] = v & 255
    return cols, np.arange(5)[None, :] < length[:, None]


def _bigram_pairs_body(ids, start: int, end: int) -> Optional[bytes]:
    """msgpack body (no array header) of list(pairwise(ids[start:end + 1])), byte-identical
    to msgspec's encoding of those int tuples; None if a token is outside [0, 2**32)."""
    global _raw_table
    if isinstance(ids, array) and ids.itemsize == 8 and ids.typecode in "lq":
        v = np.frombuffer(ids, dtype=np.int64)[start : end + 1]
    else:
        v = np.asarray(ids[start : end + 1], dtype=np.int64)
    lo, hi = v.min(), v.max()
    if lo < 0 or hi >= 1 << 32:
        return None
    if hi < 1 << _RAW_TABLE_BITS:
        if _raw_table is None:
            _raw_table = _uint_msgpack_rows(np.arange(1 << _RAW_TABLE_BITS))
        cols, keep = _raw_table[0][v], _raw_table[1][v]
    else:
        cols, keep = _uint_msgpack_rows(v)
    n = v.size - 1
    rows = np.empty((n, 11), np.uint8)
    used = np.empty((n, 11), bool)
    rows[:, 0] = 0x92  # fixarray of 2: one (a, b) pair
    used[:, 0] = True
    rows[:, 1:6], used[:, 1:6] = cols[:-1], keep[:-1]
    rows[:, 6:11], used[:, 6:11] = cols[1:], keep[1:]
    return rows[used].tobytes()


# p-micro v4 (SGLANG_PMICRO_HOSTSTATS): per-process counters, flushed by the scheduler's PMICRO_HOSTSTAT line.
PMICRO = dict(on=envs.SGLANG_PMICRO_HOSTSTATS.get(), d2h_subs=0, d2h_tokens=0, d2h_runs=0, rebacked=0,
              kv_events=0, kv_blocks=0, kv_bytes=0, stat_ns=0)
# Node ids whose host copy was removed (record_remove with medium CPU); a later backup of one is a re-backup.
PMICRO_HOST_REMOVED: set = set()


def pmicro_host_runs(host_indices, page_size: int) -> int:
    """Contiguous host-page runs in one submission's (CPU) host token indices."""
    if host_indices.is_cuda:
        return -1
    idx = host_indices.numpy()
    if idx.size == 0:
        return 0
    if idx.size % page_size == 0:
        pages = idx[::page_size] // page_size
    else:
        p = idx // page_size
        pages = p[np.r_[True, p[1:] != p[:-1]]]
    return 1 + int(np.count_nonzero(np.diff(pages) != 1))


class _RawTokenParts:
    """token_ids of a coalesced BlockStored while it is queued: msgpack array bodies
    (from Raw runs or encoded lists) joined into one msgspec.Raw by take()."""

    __slots__ = ("count", "parts")

    def __init__(self, count: int, parts: list):
        self.count, self.parts = count, parts

    @classmethod
    def of(cls, tokens) -> "_RawTokenParts":
        if isinstance(tokens, cls):
            return tokens
        if isinstance(tokens, msgspec.Raw):
            data = bytes(tokens)
            head = 1 if data[0] & 0xF0 == 0x90 else (3 if data[0] == 0xDC else 5)
            count = data[0] & 0x0F if head == 1 else int.from_bytes(data[1:head], "big")
            return cls(count, [data[head:]])
        data = msgspec.msgpack.encode(tokens)
        head = len(_msgpack_array_header(len(tokens)))
        return cls(len(tokens), [data[head:]])

    def extend(self, tokens) -> None:
        other = _RawTokenParts.of(tokens)
        self.count += other.count
        self.parts.extend(other.parts)

    def raw(self) -> msgspec.Raw:
        return msgspec.Raw(_msgpack_array_header(self.count) + b"".join(self.parts))


# Length of LoRARef.lora_id, a uuid4 or uuid5 hex string. Req adds the id to the
# end of extra_key, so the last 32 characters of extra_key are the id.
_LORA_ID_LEN = 32


class LoRANameTable:
    """Find the LoRA adapter name from the lora_id at the end of extra_key.

    Req adds lora_id to the end of extra_key, so a tree node has only the id.
    The table keeps all entries. A lora_id is not used again, and nodes of an
    unloaded adapter can publish stores until eviction.
    """

    def __init__(self):
        self._names: dict[str, str] = {}

    @classmethod
    def from_lora_refs(cls, lora_refs: Optional[Iterable[LoRARef]]) -> LoRANameTable:
        table = cls()
        for lora_ref in lora_refs or ():
            table.register(lora_id=lora_ref.lora_id, lora_name=lora_ref.lora_name)
        return table

    def register(self, *, lora_id: str, lora_name: str) -> None:
        self._names[lora_id] = lora_name

    def get(self, extra_key: Optional[str]) -> Optional[str]:
        return self.resolve(extra_key)

    def resolve(self, extra_key: Optional[str]) -> Optional[str]:
        if not extra_key:
            return None
        return self._names.get(extra_key[-_LORA_ID_LEN:])


class KvEventLoraNames(LoRANameTable):
    """Public names of selected LoRA namespaces, kept for the publisher lifetime.

    Write-back can publish a host store after the last GPU removal and after
    its request dies, so request or published-block counts cannot bound this.
    """

    def __init__(self):
        # Adapter-id registrations keep the existing Dynamo load-time lookup.
        self._adapter_names: dict[str, str] = {}
        self._names: dict[str, str] = {}

    def register(
        self,
        req: Any = None,
        lora_name: Optional[str] = None,
        *,
        lora_id: Optional[str] = None,
    ) -> None:
        """Name ``req``'s namespace; ``req`` needs ``extra_key`` and ``lora_id``."""
        if lora_id is not None:
            self._adapter_names[lora_id] = lora_name
            return
        if req.lora_id is None:
            return
        # Older IPC senders omit the optional name; admission still resolves an ID.
        lora_name = lora_name or self._adapter_names.get(req.lora_id)
        if not lora_name:
            return
        self._names[req.extra_key] = lora_name

    def get(self, extra_key: Optional[str]) -> Optional[str]:
        return self._names.get(extra_key)

    def resolve(self, extra_key: Optional[str]) -> Optional[str]:
        if not extra_key:
            return None
        return self._names.get(extra_key) or self._adapter_names.get(
            extra_key[-_LORA_ID_LEN:]
        )


class KVCacheEventRecorder:
    """Collects KV placement events for one cache.

    ``enabled=False`` makes every ``record_*`` call a no-op and ``take`` return an
    empty list, so callers never have to guard.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        page_size: int,
        dynamo_format: bool = False,
        lora_names: Optional[KvEventLoraNames] = None,
    ):
        self.enabled = enabled
        self._batch_events = envs.SGLANG_HICACHE_BATCH_EVENTS.get()
        # Bigram token_ids as pre-encoded msgpack (same wire bytes); needs batching.
        self._raw_tokens = self._batch_events and envs.SGLANG_KV_EVENTS_RAW_TOKENS.get()
        self._has_raw_parts = False
        # First N Raw runs are also built as tuples and byte-compared (a mismatch raises).
        self._raw_verify_left = envs.SGLANG_KV_EVENTS_RAW_TOKENS_VERIFY.get() if self._raw_tokens else 0
        self._raw_stats = [0, 0, 0, 0]  # raw runs, raw tokens, list runs, list tokens (bigram)
        self._raw_stats_t = time.monotonic()
        self.page_size = page_size
        self.dynamo_format = dynamo_format
        self.lora_names = lora_names if lora_names is not None else KvEventLoraNames()
        self._queue: list = []
        # Lazy mode: per-node inputs are snapshotted at record time, events are built in take().
        self._lazy = _KVEV_LAZY and self._batch_events and enabled
        self._pending: list = []
        self._lazy_verify_left = _KVEV_LAZY_VERIFY if self._lazy else 0
        self._shadow: list = []
        if self._lazy:
            logger.info("KV events lazy on (SGLANG_KV_EVENTS_LAZY): built in take(), verify %d takes", self._lazy_verify_left)
        if self._batch_events:
            self.record_store = self._record_store_fast
            if self.enabled:
                logger.info("KV event batching on (SGLANG_HICACHE_BATCH_EVENTS): one BlockStored per page run")
                if self._raw_tokens:
                    logger.info(
                        "KV event Raw token_ids on (SGLANG_KV_EVENTS_RAW_TOKENS): bigram runs >= %d tokens, verify %d runs",
                        _RAW_MIN_TOKENS, self._raw_verify_left,
                    )

    def enqueue(self, event) -> None:
        """Append an event, coalescing it with a compatible queue tail.

        KV event batches already support multiple block hashes.  Combining them
        here avoids emitting one event per page while preserving ordering and
        the parent-linked store chains consumers use to rebuild the cache tree.
        """
        if self._queue:
            tail = self._queue[-1]

            if isinstance(tail, BlockRemoved) and isinstance(event, BlockRemoved):
                if tail.medium == event.medium:
                    tail.block_hashes.extend(event.block_hashes)
                    return

            elif isinstance(tail, BlockStored) and isinstance(event, BlockStored):
                if (
                    tail.medium == event.medium
                    and tail.lora_id == event.lora_id
                    and tail.block_size == event.block_size
                    and tail.cache_salt == event.cache_salt
                    and tail.session_id == event.session_id
                    and tail.lora_name == event.lora_name
                    and tail.block_hashes
                    and event.parent_block_hash == tail.block_hashes[-1]
                ):
                    tail.block_hashes.extend(event.block_hashes)
                    if type(tail.token_ids) is list and type(event.token_ids) is list:
                        tail.token_ids.extend(event.token_ids)
                    else:
                        if not isinstance(tail.token_ids, _RawTokenParts):
                            tail.token_ids = _RawTokenParts.of(tail.token_ids)
                            self._has_raw_parts = True
                        tail.token_ids.extend(event.token_ids)
                    return

        self._queue.append(event)

    def _node_event_hash_values(self, node: Any) -> list:
        """Hash values to publish for ``node``, computing them if not yet set."""
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, self.page_size)
        if node.key.extra_key is None and node.key.cache_salt is None:
            return node.hash_value
        return compute_node_event_hash_values(node, self.page_size)

    def _parent_block_hash(self, node: Any) -> Optional[int]:
        """The hash the first page of ``node`` links back to.

        ``None`` when the parent is the tree root: a root carries an empty
        ``hash_value`` and no event hash, so it contributes no link. Every other
        node on the path has a parent, which is what distinguishes the two.
        """
        parent = node.parent
        if parent is None or parent.parent is None:
            return None
        if node.key.extra_key is not None or node.key.cache_salt is not None:
            parent_hash_values = parent.event_hash_value
            assert parent_hash_values is not None
        else:
            parent_hash_values = parent.hash_value
        if not parent_hash_values:
            return None
        return hash_str_to_int64(parent_hash_values[-1])

    def _namespace_seed(self, node: Any) -> Optional[bytes]:
        if not self.dynamo_format:
            return kv_event_namespace_seed(
                extra_key=(
                    node.key.extra_key
                    if self.lora_names.get(node.key.extra_key) is not None
                    else None
                ),
                cache_salt=None,
            )
        return kv_event_namespace_seed(
            extra_key=node.key.extra_key, cache_salt=node.key.cache_salt
        )

    def record_store(
        self, node: Any, medium=None, *, session_id: Optional[str] = None
    ) -> None:
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        event_hash_values = self._node_event_hash_values(node)
        namespace_seed = self._namespace_seed(node)
        parent_block_hash = self._parent_block_hash(node)
        if parent_block_hash is not None:
            parent_block_hash = namespaced_block_hash(
                parent_block_hash, namespace_seed=namespace_seed
            )
        lora_name = (
            self.lora_names.resolve(node.key.extra_key)
            if self.dynamo_format
            else self.lora_names.get(node.key.extra_key)
        )

        page_index = 0
        logical_len = len(node.key)
        is_bigram = node.key.is_bigram
        raw = node.key.token_ids
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue
            # Preserve historical event payload: bigram pages expose tuples.
            if is_bigram:
                page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
            else:
                page_tokens = list(raw[start:end])

            block_hash = namespaced_block_hash(
                hash_str_to_int64(event_hash_values[page_index]),
                namespace_seed=namespace_seed,
            )

            self.enqueue(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=page_tokens,
                    block_size=len(page_tokens),
                    lora_id=None,
                    medium=medium,
                    cache_salt=node.key.cache_salt,
                    session_id=session_id,
                    lora_name=lora_name,
                )
            )

            parent_block_hash = block_hash
            page_index += 1

    def _record_store_fast(
        self, node: Any, medium=None, *, session_id: Optional[str] = None
    ) -> None:
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        event_hash_values = self._node_event_hash_values(node)
        namespace_seed = self._namespace_seed(node)
        parent_block_hash = self._parent_block_hash(node)
        if parent_block_hash is not None:
            parent_block_hash = namespaced_block_hash(
                parent_block_hash, namespace_seed=namespace_seed
            )
        lora_name = (
            self.lora_names.resolve(node.key.extra_key)
            if self.dynamo_format
            else self.lora_names.get(node.key.extra_key)
        )

        if self._lazy:
            args = (SimpleNamespace(key=node.key), medium, session_id, event_hash_values,
                    namespace_seed, parent_block_hash, lora_name)
            self._lazy_record(0, args, node.key.extra_key is None and node.key.cache_salt is None)
            return
        self._record_store_batched(
            node,
            medium,
            session_id,
            event_hash_values,
            namespace_seed,
            parent_block_hash,
            lora_name,
        )

    def _record_store_batched(
        self,
        node,
        medium,
        session_id,
        event_hash_values,
        namespace_seed,
        parent_block_hash,
        lora_name,
    ) -> None:
        """Construct the same maximal equal-block-size runs as enqueue()."""
        logical_len = len(node.key)
        raw = node.key.token_ids
        full_end = logical_len // self.page_size * self.page_size
        # A short final page has a different block_size and cannot coalesce.
        # pairwise builds the historical tuple payload in C without per-token indexing.
        page_index = 0
        for start, end, block_size in (
            (0, full_end, self.page_size),
            (full_end, logical_len, logical_len - full_end),
        ):
            if end <= start:
                continue
            body = (
                _bigram_pairs_body(raw, start, end)
                if self._raw_tokens and node.key.is_bigram and end - start >= _RAW_MIN_TOKENS
                else None
            )
            if body is not None:
                page_tokens = msgspec.Raw(_msgpack_array_header(end - start) + body)
                self._raw_stats[0] += 1
                self._raw_stats[1] += end - start
                if self._raw_verify_left > 0:
                    self._verify_raw(page_tokens, raw, start, end)
            elif node.key.is_bigram:
                page_tokens = list(pairwise(raw[start : end + 1]))
                if self._raw_tokens:
                    self._raw_stats[2] += 1
                    self._raw_stats[3] += end - start
            else:
                page_tokens = list(raw[start:end])
            block_hashes = [
                namespaced_block_hash(
                    hash_str_to_int64(event_hash_values[i]),
                    namespace_seed=namespace_seed,
                )
                for i in range(
                    page_index,
                    page_index + (end - start + self.page_size - 1) // self.page_size,
                )
            ]
            self.enqueue(
                BlockStored(
                    block_hashes=block_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=page_tokens,
                    block_size=block_size,
                    lora_id=None,
                    medium=medium,
                    cache_salt=node.key.cache_salt,
                    session_id=session_id,
                    lora_name=lora_name,
                )
            )
            parent_block_hash = block_hashes[-1]
            page_index += len(block_hashes)

    def _verify_raw(self, page_tokens, raw, start: int, end: int) -> None:
        want = msgspec.msgpack.encode(list(pairwise(raw[start : end + 1])))
        if bytes(page_tokens) != want:
            raise RuntimeError(f"KV event Raw token_ids differ from the tuple encoding: tokens {start}-{end}")
        self._raw_verify_left -= 1
        logger.info(
            "KV_EVENTS_RAW_VERIFY_OK tokens %d, %d bytes, %d runs left", end - start, len(want), self._raw_verify_left
        )

    def record_remove(self, node: Any, medium=None) -> None:
        # One BlockRemoved per radix node.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        if PMICRO["on"] and medium == StorageMedium.CPU:
            PMICRO_HOST_REMOVED.add(node.id)
        if not self.enabled:
            return
        if medium is None:
            medium = StorageMedium.GPU

        # Hash values must match what was stored.
        event_hash_values = self._node_event_hash_values(node)
        namespace_seed = self._namespace_seed(node)
        if self._lazy:
            args = (event_hash_values, namespace_seed, len(node.key), medium)
            self._lazy_record(1, args, node.key.extra_key is None and node.key.cache_salt is None)
            return
        self._build_remove(event_hash_values, namespace_seed, len(node.key), medium)

    def _build_remove(self, event_hash_values, namespace_seed, logical_len, medium) -> None:
        block_hashes = []
        page_index = 0
        for start in range(0, logical_len, self.page_size):
            end = min(start + self.page_size, logical_len)
            if end <= start:
                continue

            block_hashes.append(
                namespaced_block_hash(
                    hash_str_to_int64(event_hash_values[page_index]),
                    namespace_seed=namespace_seed,
                )
            )
            page_index += 1

        if block_hashes:
            self.enqueue(BlockRemoved(block_hashes=block_hashes, medium=medium))

    def record_all_cleared(self) -> None:
        if not self.enabled:
            return
        self._flush_pending()
        if self._lazy_verify_left > 0:
            self._shadow.append(AllBlocksCleared())
        self.enqueue(AllBlocksCleared())

    def _lazy_record(self, kind, args, deferrable) -> None:
        if deferrable:
            self._pending.append((kind, args))
        else:  # salted / extra keys stay eager, in order
            self._flush_pending()
            self._build_pending(kind, args)
        if self._lazy_verify_left > 0:
            self._shadow_build(kind, args)

    def _build_pending(self, kind, args) -> None:
        if kind == 0:
            self._record_store_batched(*args)
        else:
            self._build_remove(*args)

    def _flush_pending(self) -> None:
        if self._pending:
            pending, self._pending = self._pending, []
            for kind, args in pending:
                self._build_pending(kind, args)

    def _shadow_build(self, kind, args) -> None:
        # Eager build into the shadow queue (verify mode only).
        q, self._queue = self._queue, self._shadow
        try:
            self._build_pending(kind, args)
        finally:
            self._shadow, self._queue = self._queue, q

    @staticmethod
    def _finalize_raw(events) -> None:
        for event in events:
            if isinstance(event, BlockStored) and isinstance(event.token_ids, _RawTokenParts):
                event.token_ids = event.token_ids.raw()

    def _verify_lazy(self, events) -> None:
        shadow, self._shadow = self._shadow, []
        self._finalize_raw(shadow)
        got, want = msgspec.msgpack.encode(events), msgspec.msgpack.encode(shadow)
        if got != want:
            raise RuntimeError(f"KV events lazy != eager: {len(events)} vs {len(shadow)} events, {len(got)} vs {len(want)} bytes")
        self._lazy_verify_left -= 1
        logger.info("KV_EVENTS_LAZY_VERIFY_OK events %d, %d bytes, %d takes left", len(events), len(got), self._lazy_verify_left)

    def take(self) -> list:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enabled:
            return []
        verify = self._lazy and self._lazy_verify_left > 0
        self._flush_pending()
        events = self._queue
        self._queue = []
        if self._raw_tokens and time.monotonic() - self._raw_stats_t >= 60:
            st, self._raw_stats = self._raw_stats, [0, 0, 0, 0]
            self._raw_stats_t = time.monotonic()
            logger.info(
                "KV_EVENTS_RAW_STATS raw_runs=%d raw_tokens=%d list_runs=%d list_tokens=%d", *st
            )
        if self._has_raw_parts:
            self._has_raw_parts = False
            self._finalize_raw(events)
        if verify:
            self._verify_lazy(events)
        if PMICRO["on"] and events:
            t0 = time.perf_counter_ns()
            PMICRO["kv_events"] += len(events)
            PMICRO["kv_blocks"] += sum(len(e.block_hashes) for e in events if isinstance(e, (BlockStored, BlockRemoved)))
            PMICRO["kv_bytes"] += len(msgspec.msgpack.encode(events))
            PMICRO["stat_ns"] += time.perf_counter_ns() - t0
        return events
