"""SGLANG_DSV41_TAIL_STAGE across both lanes on CPU: the stage-0 client (prefill) against
the tail server's loop over real ZMQ sockets and a real bootstrap server; the forward,
Mooncake and decode's transport are fakes."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zmq
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.dsv41_tail_stage import protocol as P
from sglang.srt.disaggregation.dsv41_tail_stage.batcher import TailBatchPolicy
from sglang.srt.disaggregation.dsv41_tail_stage.registration import (
    Stage0Peers,
    tail_stage_wire,
)
from sglang.srt.disaggregation.dsv41_tail_stage.service import (
    Backend,
    TailStageService,
)
from sglang.srt.disaggregation.dsv41_tail_stage.slots import CountedPages, SlotTable
from sglang.srt.disaggregation.dsv41_tail_stage_client import (
    Stage0Pools,
    TailStageClient,
    ZmqDealer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGE, WINDOW = 256, 128
W0 = [["swa", None], ["request_row", None]]
POOLS = Stage0Pools(
    page_size=PAGE,
    c1_kv_ptrs=[1 << 20],
    c1_kv_item_lens=[73728],
    c1_index_ptrs=[2 << 20],
    c1_index_item_lens=[17408],
    residual_ptr=3 << 20,
    residual_item_len=5242880,
)


class _Request:
    def __init__(self, body=None, query=None):
        self._body, self.query, self.method = body, query or {}, "PUT"

    async def json(self):
        return self._body


class _Sender:
    def __init__(self, room):
        self.room, self.status = room, KVPoll.Bootstrapping

    def poll(self):
        return self.status


def _stage0_registration():
    """The PUT payload stage 0 builds (register_to_bootstrap, switch on)."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.dist_init_addr = None
    mgr.bootstrap_host, mgr.bootstrap_port = "127.0.0.1", 8998
    mgr.attn_tp_size, mgr.attn_tp_rank = 1, 0
    mgr.attn_cp_size, mgr.attn_cp_rank = 1, 0
    mgr.attn_dp_size, mgr.attn_dp_rank = 1, 0
    mgr.pp_size, mgr.pp_rank = 1, 0
    mgr.system_dp_size, mgr.system_dp_rank = 1, 0
    mgr.local_ip, mgr.rank_port = "10.0.0.1", 17000
    mgr.kv_cache_dtype_str, mgr.kv_layout = "auto", "v41_new"
    mgr.dsv41_spec_layout = mgr.spec_kv_layout = None
    mgr.kv_args = SimpleNamespace(page_size=PAGE, rust_http_port=None)
    conn = "sglang.srt.disaggregation.common.conn"
    with (
        patch.multiple(
            conn,
            get_parallel=lambda: SimpleNamespace(
                enable_cp_tp_group_sharing=False,
                load_balance_method="follow_bootstrap_room",
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
            state_wire=lambda kv_args: W0,
            dsv41_tail_stage_enabled=lambda: True,
        ),
        patch.object(CommonKVManager, "_register_topology_row"),
    ):
        mgr.register_to_bootstrap()
    return mgr.bootstrap_registration


class TestTailStageAcrossLanes(CustomTestCase):
    def setUp(self):
        self.bootstrap = CommonKVBootstrapServer("127.0.0.1", 0)
        self.router = zmq.Context.instance().socket(zmq.ROUTER)
        port = self.router.bind_to_random_port("tcp://127.0.0.1")
        self.senders, self.sent, self.batches, self.writes = {}, [], [], []
        welcome = P.Welcome(
            tail_session="10.0.0.9:15100",
            c1_kv_ptrs=[11 << 20],
            c1_kv_item_lens=POOLS.c1_kv_item_lens,
            c1_index_ptrs=[12 << 20],
            c1_index_item_lens=POOLS.c1_index_item_lens,
            residual_ptr=13 << 20,
            residual_item_len=POOLS.residual_item_len,
            page_size=PAGE,
            num_slots=2,
        )

        def make_sender(addr, room):
            self.senders[room] = _Sender(room)
            return self.senders[room]

        def put(addr, payload):
            put.addr = addr
            return (
                asyncio.run(self.bootstrap._handle_route_put(_Request(payload))).status
                == 200
            )

        self.put = put
        self.service = TailStageService(
            backend=Backend(
                start_transport=lambda wire: ("10.0.0.9", 17100, welcome),
                make_sender=make_sender,
                send=lambda sender, tail: self.sent.append((sender.room, tail.slot)),
                run=lambda tails: self.batches.append([t.room for t in tails]),
                put=put,
            ),
            socket=self.router,
            table=SlotTable(
                num_slots=2,
                pages=CountedPages(num_pages=64, page_size=PAGE, window=WINDOW),
                page_size=PAGE,
                window=WINDOW,
                queue_cap=8,
                ready_timeout_s=600.0,
            ),
            peers=Stage0Peers(page_size=PAGE, window=WINDOW, draft_aux=True),
            policy=TailBatchPolicy(min_tails=1, max_tails=4, wait_ms=15, buckets=(4,)),
            heartbeat_s=1.0,
            heartbeat_timeout_s=5.0,
        )
        self.registration = _stage0_registration()
        hello = P.Hello(
            stage0_id="127.0.0.1:8998/dp0",
            bootstrap_addr="127.0.0.1:8998",
            registration=self.registration,
            mooncake_session="10.0.0.1:15000",
            page_size=PAGE,
            c1_kv_item_lens=POOLS.c1_kv_item_lens,
            c1_index_item_lens=POOLS.c1_index_item_lens,
            residual_item_len=POOLS.residual_item_len,
            window=WINDOW,
        )

        def transfer(session, src, dst, lengths):
            self.writes.append((session, src, dst, lengths))
            return 0

        self.client = TailStageClient(
            socket=ZmqDealer(addr=f"127.0.0.1:{port}", identity=hello.stage0_id),
            hello=hello,
            pools=POOLS,
            transfer=transfer,
            heartbeat_s=1.0,
            timeout_s=5.0,
        )

    def tearDown(self):
        self.client.close()
        self.router.close(0)
        self.bootstrap.close()

    def _turn(self, rounds=3):
        for _ in range(rounds):
            self.client.poll()
            self.service.step(poll_ms=20)
        self.client.poll()

    def _info(self):
        query = dict.fromkeys(
            ("prefill_dp_rank", "prefill_cp_rank", "target_tp_rank", "target_pp_rank"),
            "-1",
        )
        resp = asyncio.run(self.bootstrap._handle_route_get(_Request(query=query)))
        return resp.status, (json.loads(resp.body) if resp.status == 200 else None)

    def test_a_late_fed_room_and_a_pass_through_room_end_to_end(self):
        # Stage 0's own PUT: two stages advertised, so decode waits for stage 1.
        self.assertTrue(self.put("127.0.0.1:8998", self.registration))
        self.assertEqual(self._info()[0], 503)
        self._turn()
        self.assertIsNotNone(self.client.welcome)
        self.assertIsNone(self.client.refused)
        status, info = self._info()
        self.assertEqual((status, info["pp_size"]), (200, 2))
        # Decode receives both stages' wires: W0, and W0 plus the tail entries.
        self.assertEqual(info["state_wires"], [W0, tail_stage_wire(W0, True)])

        self.client.request(room=7, rid="a", s=4101, e=4100, late_fed=True)
        self.client.request(room=8, rid="b", s=100, e=0, late_fed=False)
        self._turn()
        grant = self.client.grant(7)
        self.assertIsNotNone(grant)
        self.assertEqual(len(grant.c1_pages), 16)  # ceil((4100 - 128) / 256)
        self.assertIsNone(self.client.grant(8))
        self.assertEqual(sorted(self.senders), [7, 8])

        self.client.start_push(room=7, src_pages=np.arange(16), capture_slot=1)
        self.client._rooms[7].push.result(timeout=5)
        self.assertTrue(self.client.push_result(7))
        session, src, dst, lengths = self.writes[0]
        self.assertEqual(session, "10.0.0.9:15100")
        self.assertEqual(dst[-1], (13 << 20) + grant.residual_slot * 5242880)
        self.client.ready(7, ok=True)
        self._turn()
        self.assertEqual(self.batches, [[7]])
        for room in (7, 8):
            self.senders[room].status = KVPoll.WaitingForInput
        self._turn()
        self.assertEqual(sorted(self.sent), [(7, grant.slot), (8, -1)])
        for room in (7, 8):
            self.senders[room].status = KVPoll.Success
        with self.assertLogs(
            "sglang.srt.disaggregation.dsv41_tail_stage_client", "INFO"
        ):
            self._turn()
            self.client._timings_at -= 10
            self.client.poll()
        self.assertEqual(self.service.table.counts()["free_slots"], 2)

    def test_an_abort_before_the_write_frees_the_slot(self):
        self.put("127.0.0.1:8998", self.registration)
        self._turn()
        self.client.request(room=9, rid="c", s=4001, e=4000, late_fed=True)
        self._turn()
        self.assertIsNotNone(self.client.grant(9))
        self.client.abort(9)
        self._turn()
        self.assertEqual(self.service.table.counts()["free_slots"], 2)
        self.assertEqual(self.batches, [])


if __name__ == "__main__":
    unittest.main()
