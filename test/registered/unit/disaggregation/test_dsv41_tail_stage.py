"""SGLANG_DSV41_TAIL_STAGE: the tail server's protocol, slots, batching and its
registration as PP rank 1 of every encoder-only prefill's bootstrap table."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.dsv41_tail_stage import protocol as P
from sglang.srt.disaggregation.dsv41_tail_stage.batcher import (
    TailBatchPolicy,
    parse_buckets,
)
from sglang.srt.disaggregation.dsv41_tail_stage.registration import (
    Stage0Peers,
    entry_sender,
    expected_registrations,
    stage1_registration,
    stage1_state_indices,
    tail_stage_wire,
)
from sglang.srt.disaggregation.dsv41_tail_stage.slots import SlotState, SlotTable
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

WINDOW = 128
PAGE = 256
# Stand-in for the encoder-only wire: a keyed window entry and a request-row entry.
W0 = [["swa", "swa", 2048], ["request_row", None]]


def _split(s):
    return (s - 1) // 2 * 2


def _req(room, s, *, late_fed=True):
    e = _split(s) if late_fed else 0
    return P.SlotRequest(room=room, rid=f"r{room}", s=s, e=e, late_fed=late_fed)


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _table(*, slots=2, pages=64, queue_cap=8, ready_timeout_s=10.0):
    clock = _Clock()
    table = SlotTable(
        num_slots=slots,
        num_pages=pages,
        page_size=PAGE,
        window=WINDOW,
        queue_cap=queue_cap,
        ready_timeout_s=ready_timeout_s,
        clock=clock,
    )
    return table, clock


def _msgs(replies, kind):
    return [r.msg for r in replies if isinstance(r.msg, kind)]


class TestTailStageProtocol(CustomTestCase):
    def test_every_message_round_trips(self):
        msgs = [
            P.Hello(
                stage0_id="h:1/dp0",
                bootstrap_addr="h:1",
                registration={"pp_size": 2, "state_wire": W0},
                mooncake_session="h:2",
                page_size=PAGE,
                c1_kv_item_lens=[73728],
                c1_index_item_lens=[17408],
                residual_item_len=5242880,
                window=WINDOW,
            ),
            P.Welcome(
                tail_session="t:3",
                c1_kv_ptrs=[1 << 40],
                c1_kv_item_lens=[73728],
                c1_index_ptrs=[2 << 40],
                c1_index_item_lens=[17408],
                residual_ptr=3 << 40,
                residual_item_len=5242880,
                page_size=PAGE,
                num_slots=64,
            ),
            _req((1 << 63) - 1, 142_801),
            P.SlotGrant(room=7, slot=3, residual_slot=3, c1_pages=[5, 6]),
            P.SlotDefer(room=7, retry_ms=20),
            P.TailReady(room=7, slot=3, ok=True),
            P.TailAbort(room=7),
            P.TailDone(room=7, slot=3, status="ok", forward_ms=12.5),
            P.Heartbeat(free_slots=60, tails_per_s=840.0),
            P.Error(reason="room_in_use", room=7),
        ]
        for msg in msgs:
            self.assertEqual(P.decode(P.encode(msg)), msg)

    def test_other_versions_and_garbage_are_refused(self):
        stale = P.TailAbort(room=1, version=P.TAIL_STAGE_PROTOCOL_VERSION + 1)
        with self.assertRaisesRegex(P.ProtocolError, "version"):
            P.decode(P.encode(stale))
        with self.assertRaisesRegex(P.ProtocolError, "malformed"):
            P.decode(b"\x01\x02")

    def test_requests_must_follow_the_shared_split_rule(self):
        self.assertIsNone(P.slot_request_error(_req(1, 142_801), WINDOW))
        self.assertIsNone(P.slot_request_error(_req(1, 142_802), WINDOW))
        self.assertIsNone(P.slot_request_error(_req(1, 100, late_fed=False), WINDOW))
        bad = [
            P.SlotRequest(room=1, rid="r", s=1001, e=999, late_fed=True),  # odd
            P.SlotRequest(room=1, rid="r", s=1001, e=996, late_fed=True),  # too short
            P.SlotRequest(room=1, rid="r", s=131, e=128, late_fed=True),  # below 130
            P.SlotRequest(room=1, rid="r", s=100, e=64, late_fed=False),
        ]
        for req in bad:
            self.assertIsNotNone(P.slot_request_error(req, WINDOW), req)


class TestTailStageSlots(CustomTestCase):
    def test_grant_takes_a_slot_and_the_prefix_pages(self):
        table, _ = _table()
        (grant,) = _msgs(table.request("a", _req(1, 1001)), P.SlotGrant)
        self.assertEqual(grant.slot, grant.residual_slot)
        # [0, 1000) needs 4 pages of 256; page 0 is the padding page.
        self.assertEqual(len(grant.c1_pages), 4)
        self.assertNotIn(0, grant.c1_pages)
        self.assertEqual(table.counts()["free_pages"], 60)

    def test_stage0_waits_until_a_slot_frees(self):
        table, _ = _table(slots=2)
        for room in (1, 2):
            self.assertEqual(
                len(_msgs(table.request("a", _req(room, 1001)), P.SlotGrant)), 1
            )
        self.assertEqual(table.request("b", _req(3, 1001)), [])
        self.assertEqual(table.counts()["queued"], 1)
        table.ready("a", P.TailReady(room=1, slot=0, ok=True))
        table.take(1)
        table.sending(1)
        replies = table.finish(1, status="ok")
        (done,) = _msgs(replies, P.TailDone)
        self.assertEqual((done.room, done.status), (1, "ok"))
        (grant,) = _msgs(replies, P.SlotGrant)
        self.assertEqual((grant.room, grant.slot), (3, 0))
        self.assertEqual([r.stage0_id for r in replies], ["a", "b"])

    def test_grants_are_first_come_even_when_a_later_one_fits(self):
        table, _ = _table(slots=4, pages=8)
        table.request("a", _req(1, 6 * PAGE + 1))  # 6 pages
        self.assertEqual(table.request("a", _req(2, 4 * PAGE + 1)), [])  # 4 > 2 free
        self.assertEqual(table.request("b", _req(3, 2 * PAGE - 1)), [])  # fits, waits
        self.assertEqual(table.counts()["queued"], 2)
        replies = table.abort(1, stage0_id="a")
        self.assertEqual([g.room for g in _msgs(replies, P.SlotGrant)], [2, 3])
        self.assertEqual(table.counts()["queued"], 0)

    def test_full_queue_defers(self):
        table, _ = _table(slots=1, queue_cap=1)
        table.request("a", _req(1, 1001))
        self.assertEqual(table.request("a", _req(2, 1001)), [])
        (defer,) = _msgs(table.request("a", _req(3, 1001)), P.SlotDefer)
        self.assertEqual(defer.room, 3)
        self.assertNotIn(3, table.tails)

    def test_a_prefix_beyond_the_budget_is_refused(self):
        table, _ = _table(pages=4)
        (err,) = _msgs(table.request("a", _req(1, 4 * PAGE + 3)), P.Error)
        self.assertIn("budget", err.reason)

    def test_rooms_belong_to_one_stage0_and_repeats_are_idempotent(self):
        table, _ = _table()
        (grant,) = _msgs(table.request("a", _req(1, 1001)), P.SlotGrant)
        (again,) = _msgs(table.request("a", _req(1, 1001)), P.SlotGrant)
        self.assertEqual(again, grant)
        (err,) = _msgs(table.request("b", _req(1, 1001)), P.Error)
        self.assertEqual(err.reason, "room_in_use")
        self.assertEqual(table.counts()["free_slots"], 1)

    def test_pass_through_takes_no_slot(self):
        table, _ = _table(slots=1)
        (grant,) = _msgs(table.request("a", _req(1, 90, late_fed=False)), P.SlotGrant)
        self.assertEqual((grant.slot, grant.c1_pages), (-1, []))
        self.assertEqual(table.counts()["free_slots"], 1)
        self.assertEqual(table.finish(1, status="ok"), [])
        self.assertEqual(table.counts()["passthrough"], 0)

    def test_ready_only_for_the_granted_slot(self):
        table, _ = _table()
        table.request("a", _req(1, 1001))
        self.assertTrue(
            _msgs(table.ready("a", P.TailReady(room=1, slot=1, ok=True)), P.Error)
        )
        self.assertTrue(
            _msgs(table.ready("b", P.TailReady(room=1, slot=0, ok=True)), P.Error)
        )
        self.assertEqual(table.ready("a", P.TailReady(room=1, slot=0, ok=True)), [])
        self.assertEqual(table.num_ready(), 1)

    def test_failed_ready_frees_the_slot(self):
        table, _ = _table(slots=1)
        table.request("a", _req(1, 1001))
        (done,) = _msgs(
            table.ready("a", P.TailReady(room=1, slot=0, ok=False)), P.TailDone
        )
        self.assertEqual(done.status, "failed")
        self.assertEqual(table.counts()["free_slots"], 1)

    def test_take_runs_the_oldest_ready_first(self):
        table, clock = _table(slots=4)
        for room in (1, 2, 3):
            table.request("a", _req(room, 1001))
        for room, slot in ((3, 2), (1, 0)):
            clock.t += 0.005
            table.ready("a", P.TailReady(room=room, slot=slot, ok=True))
        clock.t += 0.010
        self.assertAlmostEqual(table.oldest_ready_wait_ms(), 15.0)
        self.assertEqual([t.room for t in table.take(8)], [3, 1])
        self.assertEqual(table.num_ready(), 0)

    def test_abort_while_sending_waits_for_the_send(self):
        table, _ = _table(slots=1)
        table.request("a", _req(1, 1001))
        table.ready("a", P.TailReady(room=1, slot=0, ok=True))
        table.take(1)
        table.sending(1)
        self.assertEqual(table.abort(1), [])
        self.assertEqual(table.counts()["free_slots"], 0)
        (done,) = _msgs(table.finish(1, status="ok"), P.TailDone)
        self.assertEqual(done.status, "aborted")
        self.assertEqual(table.counts()["free_slots"], 1)

    def test_overdue_ready_and_silent_stage0_fail_their_slots(self):
        table, clock = _table(slots=4, ready_timeout_s=10.0)
        table.request("a", _req(1, 1001))
        table.request("b", _req(2, 1001))
        table.request("b", _req(3, 1001))
        table.ready("b", P.TailReady(room=3, slot=2, ok=True))
        clock.t += 5
        done = _msgs(table.expire(silent_stage0s=["b"]), P.TailDone)
        self.assertEqual([(d.room, d.reason) for d in done], [(2, "stage 0 silent")])
        self.assertIn(3, table.tails)  # ready: stage 0 is no longer needed
        clock.t += 6
        done = _msgs(table.expire(silent_stage0s=[]), P.TailDone)
        self.assertEqual([(d.room, d.reason) for d in done], [(1, "ready timeout")])

    def test_slots_and_pages_balance_over_a_long_run(self):
        table, clock = _table(slots=3, pages=40, queue_cap=100)
        rooms = list(range(1, 60))
        for room in rooms:
            table.request(f"s{room % 3}", _req(room, 1001 + 997 * (room % 7)))
        finished = 0
        while table.tails:
            for tail in list(table.tails.values()):
                if tail.state == SlotState.GRANTED:
                    table.ready(
                        tail.stage0_id,
                        P.TailReady(room=tail.room, slot=tail.slot, ok=True),
                    )
            for tail in table.take(2):
                table.sending(tail.room)
                table.finish(tail.room, status="ok")
                finished += 1
        self.assertEqual(finished, len(rooms))
        self.assertEqual(table.counts()["free_slots"], 3)
        self.assertEqual(sorted(table.free_pages), list(range(1, 41)))


class TestTailStageBatcher(CustomTestCase):
    def setUp(self):
        self.policy = TailBatchPolicy(
            min_tails=8, max_tails=32, wait_ms=15, buckets=parse_buckets("8,16,24,32")
        )

    def test_runs_at_min_tails_or_after_the_wait(self):
        self.assertEqual(self.policy.take_count(ready=0, oldest_wait_ms=99), 0)
        self.assertEqual(self.policy.take_count(ready=7, oldest_wait_ms=14.9), 0)
        self.assertEqual(self.policy.take_count(ready=7, oldest_wait_ms=15), 7)
        self.assertEqual(self.policy.take_count(ready=8, oldest_wait_ms=0), 8)
        self.assertEqual(self.policy.take_count(ready=50, oldest_wait_ms=0), 32)

    def test_pads_to_the_next_graph_bucket(self):
        self.assertEqual(
            [self.policy.bucket(n) for n in (1, 8, 9, 17, 24, 25, 32)],
            [8, 8, 16, 24, 24, 32, 32],
        )
        with self.assertRaises(ValueError):
            self.policy.bucket(33)

    def test_rejects_a_max_without_a_bucket(self):
        with self.assertRaises(AssertionError):
            TailBatchPolicy(min_tails=8, max_tails=40, wait_ms=15, buckets=(8, 16, 32))


class _Request:
    def __init__(self, data=None, query=None):
        self._data = data
        self.query = query or {}

    async def json(self):
        return self._data


def _stage0_payload(*, dp_rank=0, pp_size=2):
    return {
        "attn_tp_size": 1,
        "attn_tp_rank": 0,
        "attn_cp_size": 1,
        "attn_cp_rank": 0,
        "attn_dp_size": 1,
        "attn_dp_rank": dp_rank,
        "pp_size": pp_size,
        "pp_rank": 0,
        "system_dp_size": 1,
        "system_dp_rank": 0,
        "rank_ip": "10.0.0.1",
        "rank_port": 17000,
        "page_size": PAGE,
        "kv_cache_dtype": "auto",
        "kv_layout": "v41_new",
        "dsv41_spec_layout": None,
        "load_balance_method": "follow_bootstrap_room",
        "decode_allocation_policy": "early",
        "prefill_http_port": 30000,
        "state_wire": W0,
    }


def _put(server, payload):
    return asyncio.run(server._handle_route_put(_Request(payload)))


def _get(server, **query):
    defaults = {
        "prefill_dp_rank": "-1",
        "prefill_cp_rank": "-1",
        "target_tp_rank": "-1",
        "target_pp_rank": "-1",
    }
    return asyncio.run(server._handle_route_get(_Request(query={**defaults, **query})))


def _decode(wire):
    return SimpleNamespace(
        pp_size=1,
        pp_rank=0,
        kv_args=SimpleNamespace(
            state_types=[StateType(entry[0]) for entry in wire],
            state_sub_pools=[entry[1] for entry in wire],
            state_wire_keys=[entry[1] for entry in wire],
            state_window_tails=[entry[2] if len(entry) > 2 else None for entry in wire],
        ),
    )


class TestTailStageRegistration(CustomTestCase):
    def _hello(self, stage0_id="h:8998/dp0", **payload_kw):
        return P.Hello(
            stage0_id=stage0_id,
            bootstrap_addr="h:8998",
            registration=_stage0_payload(**payload_kw),
            mooncake_session="10.0.0.1:15000",
            page_size=PAGE,
            c1_kv_item_lens=[73728],
            c1_index_item_lens=[17408],
            residual_item_len=5242880,
            window=WINDOW,
        )

    def _register_both(self, server, *, draft_aux=True):
        stage0 = _stage0_payload()
        self.assertEqual(_put(server, stage0).status, 200)
        self.assertEqual(_get(server).status, 503)  # 1 of 2: decode waits for stage 1
        peers = Stage0Peers(page_size=PAGE, window=WINDOW, draft_aux=draft_aux)
        hello = self._hello()
        self.assertIsNone(peers.check(hello))
        peers.add(hello, now=0.0)
        stage1 = stage1_registration(
            hello.registration,
            rank_ip="10.0.0.9",
            rank_port=17100,
            state_wire=peers.wire(),
        )
        self.assertEqual(_put(server, stage1).status, 200)
        return stage0, stage1

    def test_stage1_completes_the_stage0_table_as_pp_rank_1(self):
        server = CommonKVBootstrapServer("127.0.0.1", 0)
        stage0, stage1 = self._register_both(server)
        self.assertEqual(server._registered_count, expected_registrations(stage0))
        self.assertEqual(expected_registrations(stage1), 2)
        info = PrefillServerInfo(**json.loads(_get(server).text))
        self.assertEqual(info.pp_size, 2)
        self.assertEqual(info.state_wires, [W0, tail_stage_wire(W0, True)])
        rank1 = json.loads(
            _get(
                server,
                prefill_dp_rank="0",
                prefill_cp_rank="0",
                target_tp_rank="0",
                target_pp_rank="1",
            ).text
        )
        self.assertEqual((rank1["rank_ip"], rank1["rank_port"]), ("10.0.0.9", 17100))
        rank0 = json.loads(
            _get(
                server,
                prefill_dp_rank="0",
                prefill_cp_rank="0",
                target_tp_rank="0",
                target_pp_rank="0",
            ).text
        )
        self.assertEqual(rank0["rank_port"], 17000)
        # A one-stage decode with the tail entries pairs with both stages.
        CommonKVManager._check_prefill_state_entries(
            _decode(tail_stage_wire(W0, True)), info
        )
        # Today's decode (W0 only) and a decode without the draft aux are refused.
        for wire in (W0, tail_stage_wire(W0, False)):
            with self.assertRaisesRegex(RuntimeError, "PD state entry mismatch"):
                CommonKVManager._check_prefill_state_entries(_decode(wire), info)
        server.close()

    def test_stage1_wire_without_the_stage0_prefix_is_refused_by_decode(self):
        server = CommonKVBootstrapServer("127.0.0.1", 0)
        _put(server, _stage0_payload())
        stage1 = stage1_registration(
            _stage0_payload(),
            rank_ip="10.0.0.9",
            rank_port=17100,
            state_wire=tail_stage_wire([], True),
        )
        _put(server, stage1)
        info = PrefillServerInfo(**json.loads(_get(server).text))
        with self.assertRaisesRegex(RuntimeError, "PD state entry mismatch"):
            CommonKVManager._check_prefill_state_entries(
                _decode(tail_stage_wire(W0, True)), info
            )
        server.close()

    def test_stage0_must_advertise_two_stages(self):
        with self.assertRaisesRegex(ValueError, "pp_size"):
            stage1_registration(
                _stage0_payload(pp_size=1), rank_ip="x", rank_port=1, state_wire=[]
            )

    def test_peers_must_share_wire_and_layout(self):
        peers = Stage0Peers(page_size=PAGE, window=WINDOW, draft_aux=True)
        peers.add(self._hello(), now=0.0)
        other = self._hello("h2:8998/dp0")
        self.assertIsNone(peers.check(other))
        bad_wire = P.Hello(
            **{
                **_fields(other),
                "registration": {**other.registration, "state_wire": W0[:1]},
            }
        )
        self.assertIn("state wire", peers.check(bad_wire))
        bad_layout = P.Hello(**{**_fields(other), "c1_kv_item_lens": [1]})
        self.assertIn("layout", peers.check(bad_layout))
        bad_page = P.Hello(**{**_fields(other), "page_size": 64})
        self.assertIn("page_size", peers.check(bad_page))

    def test_n_stage0_tables_each_get_one_stage1_row(self):
        servers = [CommonKVBootstrapServer("127.0.0.1", 0) for _ in range(3)]
        peers = Stage0Peers(page_size=PAGE, window=WINDOW, draft_aux=True)
        for i, server in enumerate(servers):
            _put(server, _stage0_payload())
            hello = self._hello(f"h{i}:8998/dp0")
            peers.add(hello, now=0.0)
            _put(
                server,
                stage1_registration(
                    hello.registration,
                    rank_ip="10.0.0.9",
                    rank_port=17100,
                    state_wire=peers.wire(),
                ),
            )
        for server in servers:
            self.assertEqual(server._registered_count, 2)
            self.assertEqual(_get(server).status, 200)
            server.close()

    def test_silent_peers(self):
        peers = Stage0Peers(page_size=PAGE, window=WINDOW, draft_aux=True)
        peers.add(self._hello("a"), now=0.0)
        peers.add(self._hello("b"), now=0.0)
        peers.touch("a", now=4.0)
        self.assertEqual(peers.silent(now=6.0, timeout_s=5.0), ["b"])


class TestTailStageOwnership(CustomTestCase):
    def test_stage0_sends_w0_and_stage1_the_tail_entries(self):
        wire = tail_stage_wire(W0, True)
        self.assertEqual(
            [entry_sender(i, len(W0)) for i in range(len(wire))], [0, 0, 1, 1]
        )
        self.assertEqual(
            wire[len(W0) :],
            [
                [StateType.DSV41_TAIL_SWA_ROWS.value, None],
                [StateType.DSV41_TAIL_DRAFT_AUX.value, None],
            ],
        )
        rows = list(range(640, 768))
        payloads = stage1_state_indices(
            stage0_wire_len=len(W0), draft_aux=True, swa_rows=rows, aux_slot=5
        )
        self.assertEqual(payloads, [None, None, rows, [5]])

    def test_pass_through_sends_no_entry(self):
        payloads = stage1_state_indices(
            stage0_wire_len=len(W0), draft_aux=True, swa_rows=None, aux_slot=None
        )
        self.assertEqual(payloads, [None] * 4)
        payloads = stage1_state_indices(
            stage0_wire_len=len(W0), draft_aux=False, swa_rows=[1], aux_slot=None
        )
        self.assertEqual(payloads, [None, None, [1]])


def _fields(msg):
    return {name: getattr(msg, name) for name in msg.__struct_fields__}


if __name__ == "__main__":
    unittest.main()
