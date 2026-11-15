"""SGLANG_DSV41_TAIL_STAGE: the tail server's protocol, slots, batching and its
registration as PP rank 1 of every encoder-only prefill's bootstrap table."""

import asyncio
import json
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import zmq
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.dsv41_tail_stage import protocol as P
from sglang.srt.disaggregation.dsv41_tail_stage import transport
from sglang.srt.disaggregation.dsv41_tail_stage.batcher import (
    TailBatchPolicy,
    parse_buckets,
)
from sglang.srt.disaggregation.dsv41_tail_stage.forward import window_row_floors
from sglang.srt.disaggregation.dsv41_tail_stage.layout import (
    c1_buf_infos,
    c1_page_ids,
    item_lens_error,
    tail_swa_buf_infos,
    tail_swa_page_ids,
)
from sglang.srt.disaggregation.dsv41_tail_stage.registration import (
    Stage0Peers,
    entry_sender,
    expected_registrations,
    stage1_registration,
    stage1_state_indices,
    tail_stage_wire,
)
from sglang.srt.disaggregation.dsv41_tail_stage.service import (
    SENDER_SWEEP_S,
    Backend,
    TailStageService,
)
from sglang.srt.disaggregation.dsv41_tail_stage.slots import (
    CountedPages,
    SlotState,
    SlotTable,
)
from sglang.srt.disaggregation.utils import check_state_wire, state_wire
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
        pages=CountedPages(num_pages=pages, page_size=PAGE, window=WINDOW),
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
        # The prefix [0, 1000 - 128) needs 4 pages of 256; page 0 is the padding page.
        self.assertEqual(len(grant.c1_pages), 4)
        self.assertNotIn(0, grant.c1_pages)
        self.assertEqual(len(table.pages.free), 60)

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
        # Stage 0 may still write a granted slot: it frees at its TailReady.
        self.assertEqual(table.abort(1, stage0_id="a"), [])
        self.assertEqual(table.counts()["queued"], 2)
        replies = table.ready("a", P.TailReady(room=1, slot=0, ok=True))
        (done,) = _msgs(replies, P.TailDone)
        self.assertEqual(done.status, "aborted")
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
        self.assertTrue(_msgs(table.request("a", _req(1, 4 * PAGE + 129)), P.SlotGrant))
        (err,) = _msgs(table.request("a", _req(2, 4 * PAGE + 131)), P.Error)
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

    def test_a_failed_granted_slot_is_held_until_stage0_answers(self):
        """A slot failed after its grant must not be granted again while its stage 0
        may still write it: it frees on that stage 0's TailReady or TailAbort."""
        table, clock = _table(slots=1, ready_timeout_s=10.0)
        table.request("a", _req(1, 1001))
        self.assertEqual(table.request("b", _req(2, 1001)), [])
        clock.t += 5
        (done,) = _msgs(table.expire(silent_stage0s=["a"]), P.TailDone)
        self.assertEqual((done.room, done.status), (1, "failed"))
        self.assertEqual(table.counts()["failed_held"], 1)
        self.assertEqual(_msgs(table.expire(silent_stage0s=[]), P.SlotGrant), [])
        # Decode's abort does not free it either; stage 0's answer does, quietly.
        self.assertEqual(table.abort(1), [])
        replies = table.ready("a", P.TailReady(room=1, slot=0, ok=False))
        self.assertEqual(_msgs(replies, P.TailDone), [])
        (grant,) = _msgs(replies, P.SlotGrant)
        self.assertEqual((grant.room, grant.slot), (2, 0))
        # A stage 0 that never answers gives the slot back after another timeout.
        clock.t += 11
        table.expire(silent_stage0s=[])
        self.assertEqual(table.counts()["failed_held"], 1)
        clock.t += 11
        table.expire(silent_stage0s=[])
        self.assertEqual(table.counts()["free_slots"], 1)

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
        self.assertEqual(sorted(table.pages.free), list(range(1, 41)))


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
                [StateType.DSV41_TAIL_SWA.value, None],
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


class _Pool:
    """The fields of DeepSeekV4TokenToKVPool the layout helpers read."""

    page_size = PAGE
    swa_page_size = PAGE
    request_window = None

    def __init__(self, layer_ids):
        self._layer_ids = list(layer_ids)
        self.kv_pools = {
            1: SimpleNamespace(kv_buffer=[torch.zeros(9, 288, dtype=torch.uint8)])
        }
        index_rows = [torch.zeros(36, 17, dtype=torch.uint8)]
        self.index_pools = {
            1: SimpleNamespace(
                page_size=64, contiguous_page_row_buffers=lambda: index_rows
            )
        }
        # Physical SWA pages of 64 rows under 256-row logical pages.
        self.swa_kv_pool = SimpleNamespace(
            page_size=64,
            kv_buffer=[torch.zeros(16, 32, dtype=torch.uint8) for _ in self._layer_ids],
        )
        self.full_to_swa_index_mapping = torch.zeros(64 * PAGE, dtype=torch.int64)

    def swa_layer_ids(self):
        return list(self._layer_ids)

    def translate_loc_from_full_to_swa(self, locs):
        return self.full_to_swa_index_mapping[locs]


def _row(pages, length):
    """A req_to_token row whose positions sit in `pages`, page-aligned."""
    positions = torch.arange(length)
    return torch.tensor(pages)[positions // PAGE] * PAGE + positions % PAGE


class TestTailStageLayout(CustomTestCase):
    def test_c1_items_are_whole_full_pages(self):
        kv, index = c1_buf_infos(_Pool(range(20, 40)))
        self.assertEqual(kv.item_lens, [288])
        # Four 64-slot index pages make one 256-token full page.
        self.assertEqual(index.item_lens, [17 * 4])
        row = _row([9, 4, 7, 30], 4 * PAGE)
        np.testing.assert_array_equal(
            c1_page_ids(row, e=600, page_size=PAGE), [9, 4, 7]
        )
        np.testing.assert_array_equal(
            c1_page_ids(row, e=768, page_size=PAGE), [9, 4, 7]
        )
        np.testing.assert_array_equal(
            c1_page_ids(row, e=769, page_size=PAGE), [9, 4, 7, 30]
        )

    def test_tail_swa_pairs_layer_by_layer_across_stage_ranges(self):
        decode, stage1 = _Pool(range(40)), _Pool(range(20, 40))
        late = range(20, 40)
        dst, src = tail_swa_buf_infos(decode, late), tail_swa_buf_infos(stage1, late)
        self.assertEqual(
            dst.ptrs, [b.data_ptr() for b in decode.swa_kv_pool.kv_buffer[20:]]
        )
        self.assertEqual(src.ptrs, [b.data_ptr() for b in stage1.swa_kv_pool.kv_buffer])
        self.assertEqual(src.item_lens, [32 * 4] * 20)
        self.assertIsNone(item_lens_error("swa", src.item_lens, dst.item_lens))
        self.assertIsNotNone(item_lens_error("swa", src.item_lens, dst.item_lens[:-1]))

    def test_tail_swa_pages_follow_positions_through_the_swa_mapping(self):
        pool = _Pool(range(40))
        row = _row(list(range(10, 15)), 5 * PAGE)
        # Positions 768-1023 live in SWA page 5, 1024-1279 in SWA page 2.
        for page, swa_page in ((3, 5), (4, 2)):
            full = row[page * PAGE : (page + 1) * PAGE]
            pool.full_to_swa_index_mapping[full] = swa_page * PAGE + torch.arange(PAGE)
        e = 1100
        pages = tail_swa_page_ids(pool, row, start=e - WINDOW, end=e)
        np.testing.assert_array_equal(pages, [5, 2])
        pages = tail_swa_page_ids(pool, row, start=1000 - WINDOW, end=1000)
        np.testing.assert_array_equal(pages, [5])


class TestTailStageFloors(CustomTestCase):
    def test_rows_take_the_prompt_window_floor_and_lower_rows_attend_to_themselves(
        self,
    ):
        """Stage 1 runs [e - W, e) but the late-layer floor is s - W (the prompt's
        last window): rows from s - W attend back to it, the 1-2 rows below it to
        themselves only, as decode floors a fed tail."""
        e = 1000
        positions = torch.cat([torch.arange(e - WINDOW, e)] * 2)
        floors = window_row_floors(
            prompt_lens=[e + 1, e + 2], window=WINDOW, positions=positions
        )
        first, second = floors[:WINDOW], floors[WINDOW:]
        self.assertEqual(first[0].item(), e - WINDOW)  # below s - W: itself
        self.assertTrue(bool((first[1:] == e + 1 - WINDOW).all()))
        self.assertEqual(second[:2].tolist(), [e - WINDOW, e - WINDOW + 1])
        self.assertTrue(bool((second[2:] == e + 2 - WINDOW).all()))


class TestTailStageKVArgs(CustomTestCase):
    def test_w0_entries_are_empty_and_the_tail_entries_carry_buffers(self):
        """Decode pairs entries by position: stage 1 must register W0 (sending none of
        it) and then exactly the tail entries, over the late layers only."""
        pool = _Pool(range(20, 40))
        aux = torch.zeros(3, WINDOW, 3 * 8, dtype=torch.bfloat16)
        fwd = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=pool, kv_cache_dtype_str="fp8"
            ),
            aux=aux,
        )
        with (
            patch.object(
                transport,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_ib_device=None),
            ),
            patch.object(
                transport, "get_device", return_value=SimpleNamespace(gpu_id=0)
            ),
            patch.object(transport, "get_pd_kv_layout", return_value=None),
        ):
            kv_args = transport.tail_kv_args(
                fwd, stage0_wire=W0, start_layer=20, end_layer=40
            )
        wire = state_wire(kv_args)
        self.assertEqual(wire, tail_stage_wire(W0, True))
        check_state_wire(wire, tail_stage_wire(W0, True))
        self.assertEqual(kv_args.state_data_ptrs[: len(W0)], [[]] * len(W0))
        self.assertEqual(
            kv_args.state_data_ptrs[len(W0)],
            [b.data_ptr() for b in pool.swa_kv_pool.kv_buffer],
        )
        self.assertEqual(kv_args.state_item_lens[len(W0) + 1], [aux[0].nbytes])
        self.assertEqual(
            (kv_args.pp_rank, kv_args.prefill_start_layer, kv_args.prefill_end_layer),
            (1, 20, 40),
        )
        self.assertEqual((kv_args.kv_data_ptrs, kv_args.aux_data_ptrs), ([], []))


class _Sender:
    def __init__(self, room):
        self.room = room
        self.status = KVPoll.Bootstrapping

    def poll(self):
        return self.status


class TestTailStageService(CustomTestCase):
    """The loop over real ZMQ sockets and a real bootstrap server; fake forward and
    decode transport."""

    def setUp(self):
        self.ctx = zmq.Context()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.bind(f"inproc://tail-{id(self)}")
        self.stage0 = self.ctx.socket(zmq.DEALER)
        self.stage0.setsockopt(zmq.IDENTITY, b"h:8998/dp0")
        self.stage0.connect(f"inproc://tail-{id(self)}")
        self.bootstrap = CommonKVBootstrapServer("127.0.0.1", 0)
        self.senders, self.sent, self.batches = {}, [], []
        welcome = P.Welcome(
            tail_session="10.0.0.9:15100",
            c1_kv_ptrs=[1],
            c1_kv_item_lens=[73728],
            c1_index_ptrs=[2],
            c1_index_item_lens=[17408],
            residual_ptr=3,
            residual_item_len=5242880,
            page_size=PAGE,
            num_slots=3,
        )

        def make_sender(addr, room):
            self.senders[room] = _Sender(room)
            return self.senders[room]

        backend = Backend(
            start_transport=lambda wire: ("10.0.0.9", 17100, welcome),
            make_sender=make_sender,
            send=lambda sender, tail: self.sent.append((sender.room, tail.slot)),
            run=lambda tails: self.batches.append([t.room for t in tails]),
            put=lambda addr, payload: _put(self.bootstrap, payload).status == 200,
        )
        self.clock = _Clock()
        table, _ = _table(slots=3)
        table.clock = self.clock
        self.service = TailStageService(
            backend=backend,
            socket=self.router,
            table=table,
            peers=Stage0Peers(page_size=PAGE, window=WINDOW, draft_aux=True),
            policy=TailBatchPolicy(min_tails=2, max_tails=4, wait_ms=15, buckets=(4,)),
            heartbeat_s=1e9,
            heartbeat_timeout_s=5.0,
            clock=self.clock,
        )

    def tearDown(self):
        self.bootstrap.close()
        self.stage0.close(0)
        self.router.close(0)
        self.ctx.term()

    def _exchange(self, *msgs):
        for msg in msgs:
            self.stage0.send(P.encode(msg))
        self.service.step(poll_ms=50)
        out = []
        while self.stage0.poll(10):
            out.append(P.decode(self.stage0.recv()))
        return out

    def test_a_room_from_hello_to_done(self):
        hello = TestTailStageRegistration._hello(None)
        _put(self.bootstrap, _stage0_payload())
        (welcome,) = self._exchange(hello)
        self.assertIsInstance(welcome, P.Welcome)
        self.assertEqual(self.bootstrap._registered_count, 2)
        got = self._exchange(
            _req(1, 4001), _req(2, 4001), _req(3, 90, late_fed=False), _req(4, 4001)
        )
        grants = {m.room: m for m in got if isinstance(m, P.SlotGrant)}
        self.assertEqual(sorted(grants), [1, 2, 3, 4])
        self.assertEqual(grants[3].slot, -1)
        self.assertEqual(sorted(self.senders), [1, 2, 3, 4])
        # One ready tail waits for a second (min_tails 2) or the 15 ms window.
        self._exchange(P.TailReady(room=1, slot=grants[1].slot, ok=True))
        self.assertEqual(self.batches, [])
        self._exchange(P.TailReady(room=2, slot=grants[2].slot, ok=True))
        self.assertEqual(self.batches, [[1, 2]])
        # Sends wait for decode's transfer info (WaitingForInput).
        self.assertEqual(self.sent, [])
        for room in (1, 2, 3):
            self.senders[room].status = KVPoll.WaitingForInput
        self._exchange()
        self.assertEqual(
            sorted(self.sent), [(1, grants[1].slot), (2, grants[2].slot), (3, -1)]
        )
        for room in (1, 2, 3):
            self.senders[room].status = KVPoll.Success
        done = [m for m in self._exchange() if isinstance(m, P.TailDone)]
        self.assertEqual(
            sorted((d.room, d.status) for d in done), [(1, "ok"), (2, "ok")]
        )
        self.assertEqual(self.service.table.counts()["free_slots"], 2)
        # Decode failing a granted room holds its slot until stage 0's TailReady; a
        # sender without a pending send is polled once per sweep.
        self.senders[4].status = KVPoll.Failed
        self.clock.t += SENDER_SWEEP_S
        self.assertEqual([m for m in self._exchange() if isinstance(m, P.TailDone)], [])
        self.assertEqual(self.service.table.counts()["free_slots"], 2)
        (done,) = self._exchange(P.TailReady(room=4, slot=grants[4].slot, ok=True))
        self.assertEqual((done.room, done.status), (4, "aborted"))
        self.assertEqual(self.service.table.counts()["free_slots"], 3)

    def _welcomed_with_slots(self, slots):
        table, _ = _table(slots=slots, pages=400)
        table.clock = self.clock
        self.service.table = table
        _put(self.bootstrap, _stage0_payload())
        self._exchange(TestTailStageRegistration._hello(None))

    def test_forwards_overlap_and_their_outputs_wait_for_them(self):
        """r2 at the multi-turn ramp: the loop waited out each forward and plateaued at
        about 340 tails/s. A forward is launched without waiting; the next one queues
        behind it only when full, and a slot is sent only after its forward ran."""
        finished = []

        def run(tails):
            self.batches.append([t.room for t in tails])
            flag = [False]
            finished.append(flag)
            return lambda: flag[0]

        self.service.backend.run = run
        self._welcomed_with_slots(8)
        got = self._exchange(*[_req(room, 4001) for room in range(1, 7)])
        slot = {m.room: m.slot for m in got if isinstance(m, P.SlotGrant)}
        ready = lambda room: P.TailReady(room=room, slot=slot[room], ok=True)
        self._exchange(ready(1), ready(2))
        self.assertEqual(self.batches, [[1, 2]])
        self._exchange(ready(3), ready(4), ready(5))
        self.assertEqual(self.batches, [[1, 2]])
        self._exchange(ready(6))
        self.assertEqual(self.batches, [[1, 2], [3, 4, 5, 6]])
        for room in range(1, 7):
            self.senders[room].status = KVPoll.WaitingForInput
        self._exchange()
        self.assertEqual(self.sent, [])
        finished[0][0] = True
        self._exchange()
        self.assertEqual(sorted(r for r, _ in self.sent), [1, 2])
        finished[1][0] = True
        self._exchange()
        self.assertEqual(sorted(r for r, _ in self.sent), [1, 2, 3, 4, 5, 6])

    def test_only_senders_with_a_send_are_polled_each_step(self):
        """r2: every open sender was polled on every loop step."""
        self._welcomed_with_slots(8)
        got = self._exchange(*[_req(room, 4001) for room in range(1, 5)])
        slot = {m.room: m.slot for m in got if isinstance(m, P.SlotGrant)}
        self._exchange(*[P.TailReady(room=r, slot=slot[r], ok=True) for r in (1, 2)])
        polls = Counter()
        for room, sender in self.senders.items():
            sender.poll = lambda room=room, sender=sender: (
                polls.update([room]) or sender.status
            )
        self._exchange()
        self.assertEqual(sorted(polls), [1, 2])
        self.clock.t += SENDER_SWEEP_S
        self._exchange()
        self.assertEqual(sorted(polls), [1, 2, 3, 4])

    def test_requests_before_hello_and_bad_versions_are_refused(self):
        (err,) = self._exchange(_req(1, 4001))
        self.assertEqual(err.reason, "no Hello")
        self.stage0.send(P.encode(P.TailAbort(room=1, version=99)))
        self.service.step(poll_ms=50)
        self.assertIn("version", P.decode(self.stage0.recv()).reason)


def _fields(msg):
    return {name: getattr(msg, name) for name in msg.__struct_fields__}


if __name__ == "__main__":
    unittest.main()
