"""SGLANG_DSV41_TAIL_STAGE on an encoder-only prefill: the tail server's client, the slot
writes and the bootstrap wait."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.dsv41_tail_stage import protocol as P
from sglang.srt.disaggregation.dsv41_tail_stage_client import (
    HELLO_RETRY_S,
    UNREACHABLE,
    Stage0Pools,
    TailStageClient,
    push_blocks,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
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


def _client(transfer=None):
    sock, clock = _Socket(), _Clock()
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
        heartbeat_s=1.0,
        timeout_s=5.0,
        clock=clock,
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
        client.abort(2)
        self.assertEqual(sock.take(P.TailAbort), [])

    def test_push_then_ready(self):
        transfer = MagicMock(return_value=0)
        client, sock, _ = _welcomed(transfer)
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(
            P.SlotGrant(room=1, slot=3, residual_slot=4, c1_pages=[8, 9, 10, 11])
        )
        client.poll()
        client.start_push(room=1, src_pages=np.array([0, 1, 2, 3]), capture_slot=6)
        client._rooms[1].push.result(timeout=5)
        self.assertTrue(client.push_result(1))
        session, src, dst, lengths = transfer.call_args.args
        self.assertEqual(session, "tail:1")
        self.assertEqual(len(src), 4)  # one run per c1 entry, plus the window
        client.ready(1, ok=True)
        (ready,) = sock.take(P.TailReady)
        self.assertEqual((ready.room, ready.slot, ready.ok), (1, 3, True))
        self.assertEqual(client.tails_pushed, 1)
        self.assertIsNone(client.grant(1))

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
        in flight must land before TailAbort leaves."""
        gate = threading.Event()
        client, sock, _ = _welcomed(lambda *_: 0 if gate.wait(5) else -1)
        client.request(room=1, rid="r", s=1001, e=1000, late_fed=True)
        sock.inbox.append(P.SlotGrant(room=1, slot=3, residual_slot=4, c1_pages=[8]))
        client.poll()
        client.start_push(room=1, src_pages=np.array([0]), capture_slot=0)
        client.abort(1)
        client.poll()
        self.assertEqual(sock.take(P.TailAbort), [])
        self.assertIsNone(client.grant(1))
        gate.set()
        client._rooms[1].push.result(timeout=5)
        client.poll()
        self.assertEqual(sock.take(P.TailAbort)[0].room, 1)
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


def _granted_queue(client, *, late_fed=True):
    queue = SimpleNamespace(
        tail_stage=client,
        kv_manager=MagicMock(),
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
    def test_late_fed_request_finalizes_only_once_granted(self):
        client, sock, _ = _welcomed()
        queue, req = _granted_queue(client)
        wait = PrefillBootstrapQueue._tail_stage_granted
        self.assertFalse(wait(queue, req))
        self.assertFalse(wait(queue, req))
        self.assertEqual(len(sock.take(P.SlotRequest)), 1)
        sock.inbox.append(
            P.SlotGrant(room=1, slot=0, residual_slot=0, c1_pages=[0] * 4)
        )
        client.poll()
        self.assertTrue(wait(queue, req))
        queue.kv_manager.conclude_failure.assert_not_called()

    def test_pass_through_request_does_not_wait(self):
        client, sock, _ = _welcomed()
        queue, req = _granted_queue(client, late_fed=False)
        self.assertTrue(PrefillBootstrapQueue._tail_stage_granted(queue, req))
        self.assertEqual(sock.take(P.SlotRequest)[0].e, 0)

    def test_unreachable_tail_stage_fails_the_room_for_decode_too(self):
        client, sock, clock = _welcomed()
        queue, req = _granted_queue(client)
        clock.t += 6
        self.assertFalse(PrefillBootstrapQueue._tail_stage_granted(queue, req))
        queue.kv_manager.conclude_failure.assert_called_once_with(
            bootstrap_room=1, failure_reason=UNREACHABLE
        )
        self.assertEqual(sock.take(P.TailAbort)[0].room, 1)


class TestPushAfterDecodeSuccess(CustomTestCase):
    def test_slot_written_after_decode_success_then_released(self):
        client, sock, _ = _welcomed()
        queue, req = _granted_queue(client)
        self.assertFalse(PrefillBootstrapQueue._tail_stage_granted(queue, req))
        sock.inbox.append(
            P.SlotGrant(room=1, slot=2, residual_slot=2, c1_pages=[5, 6, 7, 8])
        )
        client.poll()
        req.kv = SimpleNamespace(req_pool_idx=1)
        req.dsv41_late_capture_slot = 9
        req_to_token = torch.arange(2 * 1024).reshape(2, 1024) + 256 * 10
        sched = SimpleNamespace(
            disagg_prefill_bootstrap_queue=queue,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
        self.assertTrue(Mixin._dsv41_tail_push_pending(sched, req))
        client._rooms[1].push.result(timeout=5)
        src = client._transfer.call_args.args[1]
        # Row 1 holds locs 1024 + 2560 ...: pages 14..17 of [0, 1000).
        self.assertEqual(src[0], POOLS.c1_kv_ptrs[0] + 14 * 100)
        self.assertEqual(src[-1], POOLS.residual_ptr + 9 * POOLS.residual_item_len)
        self.assertFalse(Mixin._dsv41_tail_push_pending(sched, req))
        self.assertTrue(sock.take(P.TailReady)[0].ok)
        # Nothing left to hold: a second success check releases at once.
        self.assertFalse(Mixin._dsv41_tail_push_pending(sched, req))


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


if __name__ == "__main__":
    unittest.main()
