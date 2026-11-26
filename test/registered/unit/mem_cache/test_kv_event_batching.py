"""Unit tests for the opt-in KV event construction modes in mem_cache/events.

SGLANG_HICACHE_BATCH_EVENTS builds one BlockStored per run of equal-size pages,
SGLANG_KV_EVENTS_RAW_TOKENS hands long bigram (EAGLE) token_ids to the publisher
as pre-encoded msgpack, and SGLANG_KV_EVENTS_LAZY builds queued events in
take(). For the same cache updates, every mode must publish the msgpack bytes
the default per-page recorder publishes.
"""

import hashlib
import unittest
from array import array
from itertools import cycle, islice, pairwise
from unittest import mock

import msgspec

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    StorageMedium,
)
from sglang.srt.environ import envs
from sglang.srt.managers.req_types import MM_PAD_SHIFT_VALUE
from sglang.srt.mem_cache import events
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Token ids at each msgpack uint width boundary, all inside the lookup table.
NARROW = [0, 1, 127, 128, 255, 256, 65535, 65536, (1 << 18) - 1]
# Ids past the lookup table, up to the largest uint32.
WIDE = NARROW + [1 << 18, (1 << 32) - 1]
# Ids msgpack encodes as wider or signed ints: these runs keep the tuple list.
OUT_OF_RANGE = [5, 1 << 32, -1]
PADS = [MM_PAD_SHIFT_VALUE + 11, MM_PAD_SHIFT_VALUE + 12]
OTHER_PADS = [MM_PAD_SHIFT_VALUE + 90 + i for i in range(4)]
LABEL = "9e3779b97f4a7c15"

MODES = {
    "batch": dict(batch=True),
    "batch+raw": dict(batch=True, raw=True),
    "batch+lazy": dict(batch=True, lazy=True),
    "batch+raw+lazy": dict(batch=True, raw=True, lazy=True),
}


def ids(values, count):
    return list(islice(cycle(values), count))


def make_recorder(
    page_size, *, dynamo_format=False, batch=False, raw=False, lazy=False, verify=0
):
    with (
        envs.SGLANG_HICACHE_BATCH_EVENTS.override(batch),
        envs.SGLANG_KV_EVENTS_RAW_TOKENS.override(raw),
        envs.SGLANG_KV_EVENTS_RAW_TOKENS_VERIFY.override(verify),
        envs.SGLANG_KV_EVENTS_LAZY.override(lazy),
    ):
        return events.KVCacheEventRecorder(
            enabled=True, page_size=page_size, dynamo_format=dynamo_format
        )


class Node:
    """The radix node fields the recorder reads, with fixed per-page hashes."""

    def __init__(
        self,
        name,
        parent,
        token_ids,
        page_size,
        *,
        is_bigram=False,
        extra_key=None,
        cache_salt=None,
    ):
        self.parent = parent
        self.key = RadixKey(
            array("q", token_ids),
            extra_key,
            is_bigram=is_bigram,
            cache_salt=cache_salt,
        )
        num_pages = -(-len(self.key) // page_size)
        self.hash_value = [
            hashlib.sha256(f"{name}/{i}".encode()).hexdigest() for i in range(num_pages)
        ]
        keyed = extra_key is not None or cache_salt is not None
        self.event_hash_value = self.hash_value if keyed else None


def bigram_updates(recorder, take):
    """Stores, removes and clears over bigram nodes of every encoding path."""
    page = recorder.page_size
    root = Node("root", None, [], page)

    def node(name, parent, values, count, **kwargs):
        return Node(
            name, parent, ids(values, count + 1), page, is_bigram=True, **kwargs
        )

    # A long (pre-encodable) run, a short child run and a long grandchild run
    # that coalesce into one event.
    a = node("a", root, NARROW, 20 * 64)
    b = node("b", a, NARROW[::-1], 2 * 64)
    h = node("h", b, WIDE, 16 * 64 + 10)
    # A short run (3 tokens at page size 1), then a long child run past the
    # lookup table.
    c = node("c", root, NARROW, 3 * page)
    d = node("d", c, WIDE, 17 * 64 + 12)
    # Out-of-range ids, and a run longer than 65535 tokens with a short child.
    e = node("e", root, OUT_OF_RANGE, 18 * 64)
    f = node("f", root, NARROW, 1100 * 64)
    g = node("g", f, WIDE, 64)
    salted = node("s", root, WIDE, 17 * 64, cache_salt="salt")

    for n in (a, b, h, c, d, e, f, g, salted):
        recorder.record_store(n)
    take()
    recorder.record_store(a, StorageMedium.CPU, session_id="session")
    recorder.record_store(b, StorageMedium.CPU, session_id="session")
    recorder.record_store(h, StorageMedium.CPU, session_id="session")
    recorder.record_remove(h)
    recorder.record_remove(b)
    recorder.record_remove(d, StorageMedium.CPU)
    recorder.record_remove(a)
    take()
    recorder.record_store(c)
    recorder.record_all_cleared()
    recorder.record_store(e)
    recorder.record_remove(e)
    recorder.record_store(g)
    take()
    take()  # nothing recorded


def text_updates(recorder, take):
    """Text and multimodal nodes, a namespaced chain, and routing labels that
    change between a store and its take()."""
    page = recorder.page_size
    root = Node("root", None, [], page)
    text = Node("t", root, list(range(100, 100 + 3 * 64)), page)
    mm_tokens = list(range(200, 240)) + ids(PADS, 50) + list(range(300, 371))
    mm = Node("m", text, mm_tokens, page)
    salted = Node("s", root, list(range(400, 400 + 2 * 64)), page, cache_salt="x")
    salted_child = Node("sc", salted, list(range(500, 600)), page, cache_salt="x")
    keyed = Node("k", root, mm_tokens, page, extra_key="adapter")

    recorder.record_store(text)
    recorder.record_store(mm)  # no label registered yet
    events.MM_ROUTING_LABELS.register(PADS, LABEL)
    take()
    recorder.record_store(mm, StorageMedium.CPU)
    recorder.record_store(salted)
    recorder.record_store(salted_child, session_id="session")
    recorder.record_store(keyed)
    # New labels evict PADS from the 4-entry registry before take().
    events.MM_ROUTING_LABELS.register(OTHER_PADS, "0123456789abcdef")
    recorder.record_remove(salted_child)
    take()
    recorder.record_remove(mm)
    recorder.record_remove(text)
    recorder.record_all_cleared()
    recorder.record_store(text)
    take()


class TestKVEventConstructionModes(CustomTestCase):
    def setUp(self):
        self.addCleanup(setattr, events, "MM_ROUTING_LABELS", events.MM_ROUTING_LABELS)
        self.encoder = msgspec.msgpack.Encoder()

    def _published(self, updates, page_size, dynamo_format=False, **mode):
        """Encoded KVEventBatch of each take() while ``updates`` runs."""
        events.MM_ROUTING_LABELS = events.MmRoutingLabels(capacity=4)
        recorder = make_recorder(page_size, dynamo_format=dynamo_format, **mode)
        payloads = []

        def take():
            batch = KVEventBatch(ts=0.0, events=recorder.take())
            payloads.append(self.encoder.encode(batch))

        updates(recorder, take)
        return payloads

    def _assert_same_bytes(self, updates, page_size, dynamo_format=False):
        expected = self._published(updates, page_size, dynamo_format)
        for name, mode in MODES.items():
            with self.subTest(mode=name, page_size=page_size):
                got = self._published(updates, page_size, dynamo_format, **mode)
                self.assertEqual(len(got), len(expected))
                for take_index, (a, b) in enumerate(zip(got, expected)):
                    # Report the first differing byte; a full diff of MBs is slow.
                    if a != b:
                        offset = next(
                            (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                            min(len(a), len(b)),
                        )
                        self.fail(f"take {take_index}: bytes differ at {offset}")
        return b"".join(expected)

    def test_bigram_events_publish_the_same_bytes(self):
        for page_size in (1, 64):
            self._assert_same_bytes(bigram_updates, page_size)

    def test_text_events_publish_the_same_bytes(self):
        for dynamo_format in (False, True):
            with self.subTest(dynamo_format=dynamo_format):
                published = self._assert_same_bytes(text_updates, 64, dynamo_format)
                # Spans appear only in the dynamo format.
                self.assertEqual(b"mm_routing_spans" in published, dynamo_format)

    def test_long_bigram_runs_reach_the_publisher_pre_encoded(self):
        recorder = make_recorder(64, batch=True, raw=True)
        root = Node("root", None, [], 64)
        tokens = ids(WIDE, 20 * 64 + 1)
        recorder.record_store(Node("a", root, tokens, 64, is_bigram=True))
        (event,) = recorder.take()
        self.assertIsInstance(event.token_ids, msgspec.Raw)
        self.assertEqual(
            msgspec.msgpack.decode(event.token_ids),
            [list(pair) for pair in pairwise(tokens)],
        )

    def test_verify_checks_the_first_pre_encoded_runs(self):
        root = Node("root", None, [], 64)
        runs = [
            Node(name, root, ids(WIDE, 20 * 64 + 1), 64, is_bigram=True)
            for name in ("a", "b")
        ]
        encode = events._bigram_pairs_body

        def corrupted(token_ids, start, end):
            body = bytearray(encode(token_ids, start, end))
            body[-1] ^= 1
            return bytes(body)

        with mock.patch.object(events, "_bigram_pairs_body", corrupted):
            # Off: the corrupted encoding is published unchecked.
            recorder = make_recorder(64, batch=True, raw=True)
            recorder.record_store(runs[0])
            self.assertIsInstance(recorder.take()[0].token_ids, msgspec.Raw)
            # On: the first corrupted run raises.
            recorder = make_recorder(64, batch=True, raw=True, verify=2)
            with self.assertRaisesRegex(RuntimeError, "differ from the tuple"):
                recorder.record_store(runs[0])

        # Only the first N runs are checked: a corrupted second run passes
        # once one correct run used up a count of 1.
        recorder = make_recorder(64, batch=True, raw=True, verify=1)
        recorder.record_store(runs[0])
        with mock.patch.object(events, "_bigram_pairs_body", corrupted):
            recorder.record_store(runs[1])

    def test_batching_enqueues_one_event_per_run(self):
        root = Node("root", None, [], 64)
        node = Node("a", root, list(range(20 * 64 + 10)), 64)
        for mode, expected_calls in ((dict(), 21), (dict(batch=True), 2)):
            with self.subTest(mode=mode):
                recorder = make_recorder(64, **mode)
                with mock.patch.object(
                    recorder, "enqueue", wraps=recorder.enqueue
                ) as enqueue:
                    recorder.record_store(node)
                    self.assertEqual(enqueue.call_count, expected_calls)
                stored = recorder.take()
                self.assertEqual([len(e.block_hashes) for e in stored], [20, 1])
                self.assertEqual([e.block_size for e in stored], [64, 10])

    def test_lazy_mode_builds_events_in_take(self):
        root = Node("root", None, [], 64)
        node = Node("a", root, list(range(3 * 64)), 64)
        recorder = make_recorder(64, batch=True, lazy=True)
        with mock.patch.object(recorder, "enqueue", wraps=recorder.enqueue) as enqueue:
            recorder.record_store(node)
            recorder.record_remove(node)
            self.assertEqual(enqueue.call_count, 0)
            taken = recorder.take()
            self.assertEqual(enqueue.call_count, 2)
        self.assertEqual([type(event) for event in taken], [BlockStored, BlockRemoved])


if __name__ == "__main__":
    unittest.main()
