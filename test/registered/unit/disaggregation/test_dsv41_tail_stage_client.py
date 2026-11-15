"""SGLANG_DSV41_TAIL_STAGE on an encoder-only prefill: the tail server's client, the slot
writes and the bootstrap wait."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.dsv41_tail_stage import protocol as P
from sglang.srt.disaggregation.dsv41_tail_stage.registration import tail_stage_wire
from sglang.srt.disaggregation.dsv41_tail_stage_client import (
    HELLO_RETRY_S,
    UNREACHABLE,
    WRITE_PARTS,
    Stage0Pools,
    TailStageClient,
    push_blocks,
    split_blocks,
)
from sglang.srt.disaggregation.dsv41_tail_stage_decode import (
    TAIL_AUX_ABORT_REST_STEPS,
    TailAuxStaging,
    append_tail_stage_entries,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import append_state_component, state_wire
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

Mixin = SchedulerDisaggregationPrefillMixin
POOLS = Stage0Pools(
    page_size=256,
    c1_kv_ptrs=[1_000_000],
    c1_kv_item_lens=[100],
    c1_index_ptrs=[2_000_000, 3_000_000],
    c1_index_item_lens=[10, 2],
    residual_ptr=4_000_000,
    residual_item_len=5_242_880,
)


def _welcome(**fields):
    values = dict(
        tail_session="tail:1",
        c1_kv_ptrs=[10_000_000],
        c1_kv_item_lens=[100],
        c1_index_ptrs=[20_000_000, 30_000_000],
        c1_index_item_lens=[10, 2],
        residual_ptr=40_000_000,
        residual_item_len=5_242_880,
        page_size=256,
        num_slots=64,
    )
    values.update(fields)
    return P.Welcome(**values)


class _Socket:
    def __init__(self):
        self.sent = []
        self.inbox = []

    def send(self, data):
        self.sent.append(P.decode(data))

    def recv(self):
        return P.encode(self.inbox.pop(0)) if self.inbox else None

    def take(self, kind):
        out = [m for m in self.sent if isinstance(m, kind)]
        self.sent = [m for m in self.sent if not isinstance(m, kind)]
        return out


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _client(transfer=None, *, clock=None, heartbeat_s=1.0, heartbeat_thread=False):
    sock, clock = _Socket(), clock or _Clock()
    hello = P.Hello(
        stage0_id="p0:8998/dp0",
        bootstrap_addr="p0:8998",
        registration={"pp_size": 2, "pp_rank": 0},
        mooncake_session="p0:17000",
        page_size=256,
        c1_kv_item_lens=[100],
        c1_index_item_lens=[10, 2],
        residual_item_len=5_242_880,
        window=128,
    )
    client = TailStageClient(
        socket=sock,
        hello=hello,
        pools=POOLS,
        transfer=transfer or MagicMock(return_value=0),
        heartbeat_s=heartbeat_s,
        timeout_s=5.0,
        clock=clock,
        heartbeat_thread=heartbeat_thread,
    )
    return client, sock, clock


def _welcomed(transfer=None):
    client, sock, clock = _client(transfer)
    client.poll()
    sock.inbox.append(_welcome())
    client.poll()
    sock.sent.clear()
    return client, sock, clock


class TestPushBlocks(CustomTestCase):
    def test_pages_coalesce_per_entry_and_the_window_goes_last(self):
        grant = P.SlotGrant(room=7, slot=3, residual_slot=5, c1_pages=[40, 41, 42, 9])
        blocks = push_blocks(
            pools=POOLS,
            welcome=_welcome(),
            grant=grant,
            src_pages=np.array([4, 5, 6, 7]),
            capture_slot=2,
        )
        # src 4..7 against dst 40, 41, 42, 9: two runs per entry, in position order.
        want = []
        for src, dst, item in (
            (1_000_000, 10_000_000, 100),
            (2_000_000, 20_000_000, 10),
            (3_000_000, 30_000_000, 2),
        ):
            want += [
                (src + 4 * item, dst + 40 * item, 3 * item),
                (src + 7 * item, dst + 9 * item, item),
            ]
        r = POOLS.residual_item_len
        want.append((4_000_000 + 2 * r, 40_000_000 + 5 * r, r))
        self.assertEqual(blocks, want)

    def test_page_counts_must_match(self):
        grant = P.SlotGrant(room=7, slot=3, residual_slot=5, c1_pages=[1, 2])
        with self.assertRaises(AssertionError):
            push_blocks(
                pools=POOLS,
                welcome=_welcome(),
                grant=grant,
                src_pages=np.array([1, 2, 3]),
                capture_slot=0,
            )


class TestTailStageClient(CustomTestCase):
    def test_hello_repeats_until_welcomed_then_heartbeats(self):
        client, sock, clock = _client()
        client.poll()
        client.poll()
        self.assertEqual(len(sock.take(P.Hello)), 1)
        clock.t += HELLO_RETRY_S
        client.poll()
        self.assertEqual(len(sock.take(P.Hello)), 1)
        sock.inbox.append(_welcome())
        client.poll()
        self.assertIsNotNone(client.welcome)
        self.assertEqual(len(sock.take(P.Heartbeat)), 1)
        clock.t += 0.5
        client.poll()
        self.assertEqual(sock.take(P.Heartbeat), [])
        clock.t += HELLO_RETRY_S
        sock.inbox.append(P.Heartbeat(free_slots=64))
        client.poll()
        self.assertEqual(sock.take(P.Hello), [])
        self.assertEqual(len(sock.take(P.Heartbeat)), 1)

    def test_layout_mismatch_refuses_late_fed_rooms(self):
        client, sock, _ = _client()
        sock.inbox.append(_welcome(c1_kv_item_lens=[200]))
        client.poll()
        self.assertIsNone(client.welcome)
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        self.assertIn("differs", client.error(1))

    def test_late_fed_room_waits_for_its_grant_and_retries_a_defer(self):
        client, sock, clock = _welcomed()
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        (req,) = sock.take(P.SlotRequest)
        self.assertEqual((req.room, req.s, req.e, req.late_fed), (1, 1001, 1000, True))
        sock.inbox.append(P.SlotDefer(room=1, retry_ms=20))
        client.poll()
        self.assertIsNone(client.grant(1))
        self.assertEqual(sock.take(P.SlotRequest), [])
        clock.t += 0.02
        client.poll()
        self.assertEqual(len(sock.take(P.SlotRequest)), 1)
        grant = P.SlotGrant(room=1, slot=3, residual_slot=3, c1_pages=[0, 1, 2, 3])
        sock.inbox.append(grant)
        client.poll()
        self.assertEqual(client.grant(1), grant)
        self.assertIsNone(client.error(1))

    def test_pass_through_room_asks_once_and_waits_for_nothing(self):
        client, sock, _ = _welcomed()
        client.request(room=2, rid="r", s=100, e=0, late_fed=False)
        (req,) = sock.take(P.SlotRequest)
        self.assertEqual((req.e, req.late_fed), (0, False))
        self.assertIsNone(client.grant(2))
        # Its slot -1 answer is not a slot to give back.
        sock.inbox.append(P.SlotGrant(room=2, slot=-1, residual_slot=-1, c1_pages=[]))
        client.poll()
        self.assertEqual(sock.take(P.TailReady), [])
        client.abort(2)
        self.assertEqual(sock.take(P.TailAbort), [])

    def test_push_then_ready(self):
        transfer = MagicMock(return_value=0)
        client, sock, clock = _welcomed(transfer)
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(
            P.SlotGrant(room=1, slot=3, residual_slot=4, c1_pages=[8, 9, 10, 11])
        )
        client.poll()
        client.start_push(room=1, src_pages=np.array([0, 1, 2, 3]), capture_slot=6)
        client._rooms[1].push.result(timeout=5)
        self.assertTrue(client.push_result(1))
        self.assertEqual({c.args[0] for c in transfer.call_args_list}, {"tail:1"})
        written = _written(transfer)
        # One run per c1 entry, then the window in pieces.
        c1 = [b for b in written if b[0] < POOLS.residual_ptr]
        self.assertEqual(len(c1), 3)
        window = [b for b in written if b[0] >= POOLS.residual_ptr]
        self.assertEqual(window[0][0], POOLS.residual_ptr + 6 * 5_242_880)
        self.assertEqual(sum(b[2] for b in window), 5_242_880)
        client.ready(1, ok=True)
        (ready,) = sock.take(P.TailReady)
        self.assertEqual((ready.room, ready.slot, ready.ok), (1, 3, True))
        self.assertEqual(client.tails_pushed, 1)
        self.assertIsNone(client.grant(1))
        # Debug 2 reads the stage split from this summary.
        sock.inbox.append(
            P.TailDone(room=1, slot=3, status="ok", queue_ms=4, forward_ms=9, send_ms=2)
        )
        clock.t += 5
        with self.assertLogs(
            "sglang.srt.disaggregation.dsv41_tail_stage_client", "INFO"
        ) as logs:
            client.poll()
        line = logs.output[-1]
        for name in (
            "wait_grant",
            "grant_to_ready",
            "write_queue",
            "write_xfer",
            "ready_lag",
            "s1_queue 4.0/4.0",
        ):
            self.assertIn(name, line)

    def test_a_failed_write_reports_not_ok(self):
        client, sock, _ = _welcomed(MagicMock(return_value=-1))
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(P.SlotGrant(room=1, slot=3, residual_slot=4, c1_pages=[8]))
        client.poll()
        client.start_push(room=1, src_pages=np.array([0]), capture_slot=0)
        client._rooms[1].push.result(timeout=5)
        self.assertIs(client.push_result(1), False)
        client.ready(1, ok=False)
        self.assertFalse(sock.take(P.TailReady)[0].ok)

    def test_abort_waits_for_a_write_in_flight(self):
        """The tail server reuses a slot once it hears of the abort, so a write still
        in flight must land before the slot is given back."""
        gate = threading.Event()
        client, sock, _ = _welcomed(lambda *_: 0 if gate.wait(5) else -1)
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(P.SlotGrant(room=1, slot=3, residual_slot=4, c1_pages=[8]))
        client.poll()
        client.start_push(room=1, src_pages=np.array([0]), capture_slot=0)
        client.abort(1)
        client.poll()
        self.assertEqual(sock.take(P.TailReady), [])
        self.assertIsNone(client.grant(1))
        gate.set()
        client._rooms[1].push.result(timeout=5)
        client.poll()
        # The tail server holds a granted slot until a TailReady gives it back.
        (back,) = sock.take(P.TailReady)
        self.assertEqual((back.room, back.slot, back.ok), (1, 3, False))
        self.assertEqual(sock.take(P.TailAbort), [])
        self.assertNotIn(1, client._rooms)

    def test_abort_before_ready_and_a_late_grant_go_back(self):
        client, sock, _ = _welcomed()
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        client.abort(1)
        self.assertEqual(sock.take(P.TailAbort)[0].room, 1)
        sock.inbox.append(P.SlotGrant(room=1, slot=3, residual_slot=4, c1_pages=[8]))
        client.poll()
        (ready,) = sock.take(P.TailReady)
        self.assertEqual((ready.slot, ready.ok), (3, False))

    def test_room_error_and_silence_fail_a_waiting_room(self):
        client, sock, clock = _welcomed()
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(P.Error(reason="room_in_use", room=1))
        client.poll()
        self.assertEqual(client.error(1), "room_in_use")
        client.request(room=2, rid="r2", s=1001, e=1000, late_fed=True)
        self.assertIsNone(client.error(2))
        clock.t += 5.1
        self.assertEqual(client.error(2), UNREACHABLE)
        client.poll()
        self.assertEqual(len(sock.take(P.Hello)), 1)
        sock.inbox.append(P.Heartbeat(free_slots=64))
        client.poll()
        self.assertIsNone(client.error(2))

    def test_a_slot_failed_by_the_tail_server_is_never_written(self):
        """Debug 2: the tail server failed granted slots while stage 0 still held the
        grants; a write into a slot that may be granted again is refused, and the
        answer that frees the slot waits for any write in flight."""
        transfer = MagicMock(return_value=0)
        client, sock, _ = _welcomed(transfer)
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(P.SlotGrant(room=1, slot=3, residual_slot=3, c1_pages=[8]))
        sock.inbox.append(
            P.TailDone(room=1, slot=3, status="failed", reason="stage 0 silent")
        )
        client.poll()
        self.assertIsNone(client.grant(1))
        self.assertIn("stage 0 silent", client.error(1))
        self.assertFalse(
            client.start_push(room=1, src_pages=np.array([0]), capture_slot=0)
        )
        transfer.assert_not_called()
        client.abort(1)
        (back,) = sock.take(P.TailReady)
        self.assertEqual((back.room, back.slot, back.ok), (1, 3, False))

    def test_a_write_in_flight_when_its_slot_fails_does_not_count(self):
        gate = threading.Event()
        client, sock, _ = _welcomed(lambda *_: 0 if gate.wait(5) else -1)
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(P.SlotGrant(room=1, slot=3, residual_slot=3, c1_pages=[8]))
        client.poll()
        self.assertTrue(
            client.start_push(room=1, src_pages=np.array([0]), capture_slot=0)
        )
        sock.inbox.append(P.TailDone(room=1, slot=3, status="failed", reason="x"))
        client.poll()
        client.abort(1)
        self.assertEqual(sock.take(P.TailReady), [])
        gate.set()
        client._rooms[1].push.result(timeout=5)
        self.assertIs(client.push_result(1), False)
        client.poll()
        # The writer thread sends no TailReady(ok=True) for a failed slot.
        (back,) = sock.take(P.TailReady)
        self.assertFalse(back.ok)

    def test_the_writer_reports_the_slot_ready_without_a_scheduler_poll(self):
        """Twin r2: TailReady waited for the next scheduler iteration (50-80 ms under
        load), which lengthens TTFT and slot hold."""
        client, sock, _ = _welcomed()
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(P.SlotGrant(room=1, slot=3, residual_slot=3, c1_pages=[8]))
        client.poll()
        sock.sent.clear()
        client.start_push(room=1, src_pages=np.array([0]), capture_slot=0)
        client._rooms[1].push.result(timeout=5)
        (ready,) = sock.take(P.TailReady)
        self.assertEqual((ready.slot, ready.ok), (3, True))
        self.assertTrue(client.push_result(1))
        # Forgotten without a second message, and an abort now sends nothing.
        client.abort(1)
        self.assertEqual(sock.sent, [])
        self.assertNotIn(1, client._rooms)

    def test_heartbeats_go_out_while_the_scheduler_does_not_poll(self):
        """Debug 2: a 27 s cold prefill kept the scheduler from polling, and the tail
        server failed every room of that stage 0 as silent."""
        client, sock, _ = _client(
            clock=time.monotonic, heartbeat_s=0.02, heartbeat_thread=True
        )
        try:
            client.poll()
            sock.inbox.append(_welcome())
            client.poll()
            sock.sent.clear()
            time.sleep(0.3)
            self.assertGreaterEqual(len(sock.take(P.Heartbeat)), 3)
        finally:
            client.close()


class TestSplitBlocks(CustomTestCase):
    def test_parts_cover_every_byte_once_in_order_and_balance(self):
        # The c1 KV of a 130K-token prefix as one run dominates the bytes.
        blocks = [(1000, 9000, 46_000_000), (50, 60, 9_000_000), (7, 8, 5_242_880)]
        parts = split_blocks(blocks, parts=WRITE_PARTS, min_bytes=1 << 20)
        self.assertEqual(len(parts), WRITE_PARTS)
        pieces = [b for p in parts for b in p]
        covered = {}
        for src, dst, length in pieces:
            covered.setdefault((src - dst), []).append((src, length))
        for src, dst, length in blocks:
            spans = sorted(covered[src - dst])
            self.assertEqual(spans[0][0], src)
            self.assertEqual(sum(n for _, n in spans), length)
            for (a, n), (b, _) in zip(spans, spans[1:]):
                self.assertEqual(a + n, b)
        sizes = [sum(n for _, _, n in p) for p in parts]
        self.assertLessEqual(max(sizes) - min(sizes), max(sizes) // 2)
        small = [(0, 0, 7), (100, 100, 900)]
        self.assertEqual(split_blocks(small, parts=8, min_bytes=1 << 20), [small])


def _written(transfer):
    """Every (src, dst, length) a slot write sent, over all its Mooncake batches."""
    return sorted(b for call in transfer.call_args_list for b in zip(*call.args[1:]))


def _granted_queue(client, *, late_fed=True):
    queue = SimpleNamespace(
        tail_stage=client,
        kv_manager=MagicMock(),
        capture_waits=0,
    )
    req = SimpleNamespace(
        bootstrap_room=1,
        bootstrap_host="10.0.0.1",
        rid="r",
        dsv41_prompt_len=1001,
        origin_input_ids=list(range(1000 if late_fed else 873)),
        dsv41_late_fed=late_fed,
    )
    return queue, req


class TestBootstrapWait(CustomTestCase):
    def test_late_fed_request_bootstraps_without_a_slot(self):
        """Debug 2: slots granted at bootstrap were held through the prefill queue
        (grant to TailReady 70 s), so 64 slots served about one room per second."""
        client, sock, _ = _welcomed()
        queue, req = _granted_queue(client)
        self.assertTrue(PrefillBootstrapQueue._tail_stage_admits(queue, req))
        self.assertEqual(sock.take(P.SlotRequest), [])
        queue.kv_manager.conclude_failure.assert_not_called()

    def test_pass_through_request_is_announced_at_bootstrap(self):
        client, sock, _ = _welcomed()
        queue, req = _granted_queue(client, late_fed=False)
        self.assertTrue(PrefillBootstrapQueue._tail_stage_admits(queue, req))
        self.assertEqual(sock.take(P.SlotRequest)[0].e, 0)

    def test_unreachable_tail_stage_fails_the_room_for_decode_too(self):
        client, sock, clock = _welcomed()
        queue, req = _granted_queue(client)
        clock.t += 6
        self.assertFalse(PrefillBootstrapQueue._tail_stage_admits(queue, req))
        queue.kv_manager.conclude_failure.assert_called_once_with(
            bootstrap_room=1, failure_reason=UNREACHABLE
        )


def _push_sched(client):
    queue, req = _granted_queue(client)
    # e = 1100: the tail server takes the c1 pages of [0, e - 128), 4 not 5.
    req.origin_input_ids = list(range(1100))
    req.dsv41_prompt_len, req.dsv41_late_capture_start = 1101, 972
    req.dsv41_tail_written = False
    req.kv = SimpleNamespace(req_pool_idx=1)
    req.dsv41_late_capture_slot = 9
    req_to_token = torch.arange(2 * 2048).reshape(2, 2048) + 256 * 10
    sched = SimpleNamespace(
        disagg_prefill_bootstrap_queue=queue,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )
    for name in ("_request_dsv41_tail_slot", "_fail_dsv41_tail"):
        setattr(sched, name, getattr(Mixin, name).__get__(sched))
    return sched, queue, req


class TestSlotAtFinalChunk(CustomTestCase):
    def test_the_slot_is_asked_with_the_final_chunk_only(self):
        client, sock, _ = _welcomed()
        sched, _, req = _push_sched(client)
        sched._dsv41_late_capture = lambda: SimpleNamespace(bind=MagicMock())
        req.full_untruncated_fill_ids = req.origin_input_ids
        batch = SimpleNamespace(reqs=[req])
        req.extend_end = 512
        Mixin._prepare_dsv41_tail_batch(sched, batch)
        self.assertEqual(sock.take(P.SlotRequest), [])
        req.extend_end = 1100
        Mixin._prepare_dsv41_tail_batch(sched, batch)
        (asked,) = sock.take(P.SlotRequest)
        self.assertEqual((asked.room, asked.s, asked.e), (1, 1101, 1100))


class TestPushAfterDecodeSuccess(CustomTestCase):
    def test_slot_written_after_decode_success_then_released(self):
        client, sock, _ = _welcomed()
        sched, _, req = _push_sched(client)
        # Asked here if the final-chunk batch did not ask; held until granted.
        self.assertTrue(Mixin._dsv41_tail_push_pending(sched, req))
        self.assertEqual(len(sock.take(P.SlotRequest)), 1)
        self.assertTrue(Mixin._dsv41_tail_push_pending(sched, req))
        sock.inbox.append(
            P.SlotGrant(room=1, slot=2, residual_slot=2, c1_pages=[5, 6, 7, 8])
        )
        client.poll()
        self.assertTrue(Mixin._dsv41_tail_push_pending(sched, req))
        client._rooms[1].push.result(timeout=5)
        written = _written(client._transfer)
        # Row 1 holds locs 2048 + 2560 ...: pages 18..21 of [0, 972), one run.
        self.assertEqual(written[0][0], POOLS.c1_kv_ptrs[0] + 18 * 100)
        self.assertEqual(written[0][2], 4 * 100)
        window = [b for b in written if b[0] >= POOLS.residual_ptr]
        self.assertEqual(window[0][0], POOLS.residual_ptr + 9 * POOLS.residual_item_len)
        self.assertFalse(Mixin._dsv41_tail_push_pending(sched, req))
        self.assertTrue(sock.take(P.TailReady)[0].ok)
        # Nothing left to hold, and nothing asked again.
        self.assertFalse(Mixin._dsv41_tail_push_pending(sched, req))
        self.assertEqual(sock.take(P.SlotRequest), [])

    def test_a_room_the_tail_server_failed_fails_to_decode(self):
        client, sock, _ = _welcomed()
        sched, queue, req = _push_sched(client)
        Mixin._dsv41_tail_push_pending(sched, req)
        sock.inbox.append(P.SlotGrant(room=1, slot=2, residual_slot=2, c1_pages=[5]))
        sock.inbox.append(P.TailDone(room=1, slot=2, status="failed", reason="x"))
        client.poll()
        self.assertTrue(Mixin._dsv41_tail_push_pending(sched, req))
        client._transfer.assert_not_called()
        queue.kv_manager.conclude_failure.assert_called_once()
        self.assertFalse(sock.take(P.TailReady)[0].ok)


class TestAdvertisedStages(CustomTestCase):
    def test_prefill_registers_two_stages_only_under_the_switch(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.dist_init_addr = None
        mgr.bootstrap_host, mgr.bootstrap_port = "0.0.0.0", 8998
        mgr.attn_tp_size, mgr.attn_tp_rank = 1, 0
        mgr.attn_cp_size, mgr.attn_cp_rank = 1, 0
        mgr.attn_dp_size, mgr.attn_dp_rank = 1, 0
        mgr.pp_size, mgr.pp_rank = 1, 0
        mgr.system_dp_size, mgr.system_dp_rank = 1, 0
        mgr.local_ip, mgr.rank_port = "10.0.0.1", 17000
        mgr.kv_cache_dtype_str, mgr.kv_layout = "auto", None
        mgr.dsv41_spec_layout = mgr.spec_kv_layout = None
        mgr.kv_args = SimpleNamespace(page_size=256, rust_http_port=None)
        conn = "sglang.srt.disaggregation.common.conn"
        context = dict(
            get_parallel=lambda: SimpleNamespace(
                enable_cp_tp_group_sharing=False,
                load_balance_method="round_robin",
                enable_dsa_cache_layer_split=False,
                enable_kv_cache_sharding=False,
            ),
            get_memory=lambda: SimpleNamespace(
                enable_unified_cache_external_linker=False
            ),
            get_serving=lambda: SimpleNamespace(port=30000),
            get_disagg=lambda: SimpleNamespace(
                disaggregation_decode_allocation_policy="early"
            ),
        )
        for flag, pp_size in ((False, 1), (True, 2)):
            with (
                patch.multiple(conn, **context),
                patch(f"{conn}.dsv41_tail_stage_enabled", return_value=flag),
                patch.object(CommonKVManager, "_register_topology_row") as put,
            ):
                mgr.register_to_bootstrap()
            payload = put.call_args.args[1]
            self.assertEqual((payload["pp_size"], payload["pp_rank"]), (pp_size, 0))
            self.assertEqual(mgr.bootstrap_registration, payload)


def _aux(slots=2, window=4):
    return TailAuxStaging(
        slots=slots,
        window=window,
        captures=3,
        hidden_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


class TestDecodeTailEntries(CustomTestCase):
    def test_decode_wire_is_the_tail_servers(self):
        """Decode pairs the tail server's entries by position after W0, so its wire
        must be exactly the tail server's (registration.tail_stage_wire)."""
        layers = 40
        swa = SimpleNamespace(
            page_size=64,
            kv_buffer=[torch.zeros(3, 8, dtype=torch.uint8) for _ in range(layers)],
        )
        pool = SimpleNamespace(
            request_window=None,
            swa_kv_pool=swa,
            swa_page_size=128,
            swa_layer_ids=lambda: list(range(layers)),
        )
        for draft_aux in (False, True):
            kv_args = SimpleNamespace(
                state_types=[],
                state_sub_pools=[],
                state_wire_keys=[],
                state_data_ptrs=[],
                state_data_lens=[],
                state_item_lens=[],
                state_dim_per_tensor=[],
                state_conv_shard_groups=[],
                state_slice_outer_counts=[],
                state_layer_ids=[],
                state_window_tails=[],
            )
            append_state_component(
                kv_args, StateType.SWA, [1], [2], [3], sub_pool="swa", wire_key="swa"
            )
            kv_args.state_window_tails.append(2048)
            w0 = state_wire(kv_args)
            aux = _aux() if draft_aux else None
            append_tail_stage_entries(
                kv_args, pool=pool, layer_ids=range(20, 40), aux=aux
            )
            self.assertEqual(state_wire(kv_args), tail_stage_wire(w0, draft_aux))
            self.assertEqual(len(kv_args.state_window_tails), len(kv_args.state_types))
            # One item per logical SWA page of each of layers 20-39.
            self.assertEqual(kv_args.state_item_lens[1], [16] * 20)
            self.assertEqual(
                kv_args.state_data_ptrs[1],
                [buf.data_ptr() for buf in swa.kv_buffer[20:40]],
            )
            if draft_aux:
                self.assertEqual(kv_args.state_item_lens[2], [4 * 6 * 4])


class TestTailAuxStaging(CustomTestCase):
    def test_a_read_slot_frees_soon_and_an_aborted_one_rests(self):
        aux = _aux(slots=2)
        a, b, c = (SimpleNamespace(finished=lambda: False) for _ in range(3))
        self.assertEqual(aux.reserve(a, step=0), 0)
        self.assertEqual(aux.reserve(b, step=0), 1)
        self.assertIsNone(aux.reserve(c, step=0))
        aux.release(0, step=10, rest=TAIL_AUX_ABORT_REST_STEPS)
        aux.release(1, step=10, rest=2)
        self.assertIsNone(aux.reserve(c, step=11))
        self.assertEqual(aux.reserve(c, step=12), 1)
        self.assertIsNone(aux.reserve(a, step=10 + TAIL_AUX_ABORT_REST_STEPS - 1))
        self.assertEqual(aux.reserve(a, step=10 + TAIL_AUX_ABORT_REST_STEPS), 0)

    def test_finished_holders_are_the_aborted_ones(self):
        aux = _aux(slots=2)
        live = SimpleNamespace(finished=lambda: False)
        gone = SimpleNamespace(finished=lambda: True)
        aux.reserve(live, step=0)
        aux.reserve(gone, step=0)
        self.assertEqual(aux.finished_holders(), [gone])


class TestDraftKvAtAdmission(CustomTestCase):
    def test_window_rows_go_to_the_draft_in_admission_order(self):
        aux = _aux(slots=3, window=4)
        aux.buf.copy_(
            torch.arange(aux.buf.numel(), dtype=torch.float32).view_as(aux.buf)
        )
        self.assertEqual([aux.reserve(None, step=0) for _ in range(3)], [0, 1, 2])
        reqs = []
        for slot, (row, e) in zip((2, 0), ((5, 1000), (7, 2002))):
            reqs.append(
                SimpleNamespace(
                    dsv41_tail_aux_slot=slot,
                    kv=SimpleNamespace(req_pool_idx=row),
                    dsv41_tail_start=e,
                )
            )
        plain = SimpleNamespace(dsv41_tail_aux_slot=None)
        writer = MagicMock()
        sched = SimpleNamespace(
            disagg_decode_prealloc_queue=SimpleNamespace(tail_aux=aux),
            model_config=SimpleNamespace(sliding_window_size=4),
            forward_stream_ctx=MagicMock(),
            model_worker=SimpleNamespace(write_tail_draft_kv=writer),
            forward_ct=50,
        )
        sched._tail_aux = lambda: aux
        sched._release_dsv41_tail_aux = lambda req, rest: (
            SchedulerDisaggregationDecodeMixin._release_dsv41_tail_aux(
                sched, req, rest=rest
            )
        )
        SchedulerDisaggregationDecodeMixin._write_dsv41_tail_draft_kv(
            sched, [reqs[0], plain, reqs[1]]
        )
        kwargs = writer.call_args.kwargs
        self.assertEqual(kwargs["req_pool_indices"].tolist(), [5, 7])
        self.assertEqual(kwargs["starts"].tolist(), [996, 1998])
        self.assertTrue(
            torch.equal(kwargs["aux"], torch.cat([aux.rows(2), aux.rows(0)]))
        )
        self.assertEqual([r.dsv41_tail_aux_slot for r in reqs], [None, None])
        self.assertEqual(sched.dsv41_tail_admit_ct, 2)
        # Free two steps on, once the step that reads them has run.
        self.assertIsNone(aux.reserve(None, step=51))
        self.assertIsNotNone(aux.reserve(None, step=52))


class TestTailDraftKvLayout(CustomTestCase):
    def test_window_rows_address_their_positions_in_paged_and_ring_pools(self):
        from contextlib import nullcontext

        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        req_to_token = torch.arange(3 * 64).view(3, 64) + 1000
        full_to_swa = torch.arange(2000) * 10
        for ring in (None, 32):
            writer = MagicMock()
            worker = SimpleNamespace(
                _hosts_draft=True,
                draft_model_runner=SimpleNamespace(
                    token_to_kv_pool=SimpleNamespace(
                        swa_req_ring_size=ring, full_to_swa_index_mapping=full_to_swa
                    )
                ),
                model_runner=SimpleNamespace(
                    req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
                ),
                draft_model=SimpleNamespace(write_target_hidden_kv=writer),
                _draft_context=nullcontext,
            )
            aux = torch.randn(2 * 4, 6)
            DSparkWorkerV2.write_tail_draft_kv(
                worker,
                req_pool_indices=torch.tensor([2, 0]),
                starts=torch.tensor([10, 40]),
                aux=aux,
            )
            kwargs = writer.call_args.kwargs
            want_pos = [10, 11, 12, 13, 40, 41, 42, 43]
            self.assertEqual(kwargs["positions"].tolist(), want_pos)
            self.assertIs(kwargs["main_hidden"], aux)
            if ring is None:
                locs = [req_to_token[2, p] for p in want_pos[:4]] + [
                    req_to_token[0, p] for p in want_pos[4:]
                ]
                want = [int(full_to_swa[loc]) for loc in locs]
            else:
                want = [2 * ring + p % ring for p in want_pos[:4]] + [
                    0 * ring + p % ring for p in want_pos[4:]
                ]
            self.assertEqual(kwargs["swa_loc"].tolist(), want)


if __name__ == "__main__":
    unittest.main()
