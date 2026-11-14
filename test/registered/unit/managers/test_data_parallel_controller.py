"""DPBudget + DataParallelController dispatch tests.

The e2e counterpart, over the real scheduler load-report path, is
test/registered/disaggregation/test_disaggregation_dp_attention.py.

Fragility: scheduler tests bypass `DataParallelController.__init__` via `__new__` and
inject only the attrs the schedulers read (`workers`, `_active_workers`,
`round_robin_counter`, `dp_budget`). `status` / `dp_active` / `max_dp_size` are for
`update_active_ranks`. Update `_make_controller` if a scheduler starts reading another
attr. `maybe_external_dp_rank_routing` is exercised as the real method, no mock. The
refresh-throttle tests inject `load_snapshot_reader` and `_last_refresh_time` on top of
those.
"""

import threading
import time
import unittest
from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import msgspec.structs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import io_struct
from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
    WorkerPortHandshake,
)
from sglang.srt.managers.io_struct import ActiveRanksOutput
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


# Dispatch is asserted through worker.send_pyobj on SimpleNamespace reqs, which
# only the pickle codec sends.
_pickle_ipc = patch.object(io_struct, "_USE_PICKLE_IPC", True)


def setUpModule():
    _pickle_ipc.start()


def tearDownModule():
    _pickle_ipc.stop()


_BASE_LOAD = msgspec.structs.replace(
    LoadSnapshot(dp_rank=0),
    max_total_num_tokens=4096,
    max_running_requests=128,
)


def _load(**overrides) -> LoadSnapshot:
    return msgspec.structs.replace(_BASE_LOAD, **overrides)


def _make_controller(
    num_dp_ranks: int, max_dp_size: int | None = None
) -> DataParallelController:
    """Bypass __init__; inject only the attrs dispatch methods read."""
    if max_dp_size is None:
        max_dp_size = num_dp_ranks
    ctl = DataParallelController.__new__(DataParallelController)
    ctl.workers = [MagicMock(name=f"worker_{i}") for i in range(max_dp_size)]
    ctl.max_dp_size = max_dp_size
    ctl.launch_dp_size = num_dp_ranks
    ctl.dp_active = [True] * num_dp_ranks + [False] * (max_dp_size - num_dp_ranks)
    ctl.status = list(ctl.dp_active)
    ctl._active_workers = list(range(num_dp_ranks))
    ctl._active_count_cache = num_dp_ranks
    ctl.load_balance_method = LoadBalanceMethod.ROUND_ROBIN
    ctl.round_robin_counter = 0
    ctl.health_round_robin_counter = 0
    ctl.dp_budget = DPBudget(num_dp_ranks=num_dp_ranks)
    return ctl


def _req(
    routed_dp_rank=None,
    bootstrap_room=None,
    input_ids=None,
    rid="USER",
    time_stats=None,
    input_embeds=None,
):
    """Req stand-in; SimpleNamespace avoids pinning to the Req dataclass schema.

    As with a real input_embeds request, passing input_embeds leaves input_ids None.
    """
    return SimpleNamespace(
        time_stats=time_stats,
        rid=rid,
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
        input_ids=None if input_embeds is not None else (input_ids or []),
        input_embeds=input_embeds,
    )


class TestJoinerWorkerPortAdmission(CustomTestCase):
    def _launch_joiner(self, parallel, handshake, launch):
        controller = DataParallelController.__new__(DataParallelController)
        controller._receive_ports_as_client = MagicMock(return_value=handshake)
        controller.launch_tensor_parallel_group = launch
        execution = SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=True))

        with (
            patch(
                "sglang.srt.managers.data_parallel_controller.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.data_parallel_controller.get_exec",
                return_value=execution,
            ),
        ):
            controller.launch_dp_attention_schedulers(
                SimpleNamespace(), SimpleNamespace()
            )

    def test_topology_mismatch_is_rejected_before_scheduler_launch(self):
        mismatches = [
            # A wider joiner would map physical offset 4 onto active primary slot 2.
            ((1, 1), (2, 1), 1),
            # A narrower joiner would select beyond the primary's reserved slots.
            ((2, 1), (1, 1), 2),
        ]
        for primary, joining, num_dp_ranks in mismatches:
            with self.subTest(primary=primary, joining=joining):
                parallel = SimpleNamespace(
                    dist_init_addr="127.0.0.1:20000",
                    ep_join_rank_offset=4,
                    num_dp_ranks=num_dp_ranks,
                    attn_tp_size=joining[0],
                    attn_cp_size=joining[1],
                )
                handshake = WorkerPortHandshake(
                    worker_ports=[1000, 1001, 1002, 1003],
                    attn_tp_size=primary[0],
                    attn_cp_size=primary[1],
                )
                launch = MagicMock()

                with self.assertRaisesRegex(ValueError, "same attention topology"):
                    self._launch_joiner(parallel, handshake, launch)

                launch.assert_not_called()

    def test_matching_topology_selects_reserved_logical_dp_slot(self):
        parallel = SimpleNamespace(
            dist_init_addr="127.0.0.1:20000",
            ep_join_rank_offset=4,
            num_dp_ranks=1,
            attn_tp_size=2,
            attn_cp_size=1,
        )
        handshake = WorkerPortHandshake(
            worker_ports=[1000, 1001, 1002, 1003],
            attn_tp_size=2,
            attn_cp_size=1,
        )
        launch = MagicMock()

        self._launch_joiner(parallel, handshake, launch)

        self.assertEqual(launch.call_args.args[-1], [1002])


class TestLocalKvEventSources(CustomTestCase):
    @staticmethod
    def _source(rank, block_size=64):
        return {
            "dp_rank": rank,
            "endpoint": f"tcp://127.0.0.1:{5557 + rank}",
            "topic": "kv",
            "block_size": block_size,
        }

    def test_scheduler_reports_logical_block_size_or_no_source(self):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components.kv_events_publisher import (
            SchedulerKvEventsPublisher,
        )

        component = SchedulerKvEventsPublisher.__new__(SchedulerKvEventsPublisher)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_total_num_tokens = 64
        scheduler.max_req_input_len = 32
        scheduler.startup_time = {}
        scheduler.page_size = 64
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.kv_events_publisher = component
        publisher = MagicMock()
        publisher.describe_local_source.return_value = self._source(4, 256)
        for owned_publisher, expected in (
            (publisher, [self._source(4, 256)]),
            (None, []),
        ):
            with (
                self.subTest(has_publisher=owned_publisher is not None),
                get_context().override_server_args(grpc_port=50051, dcp_size=4),
            ):
                component.kv_event_publisher = owned_publisher
                self.assertEqual(
                    scheduler.get_init_info()["kv_event_sources"], expected
                )
        publisher.describe_local_source.assert_called_once_with(256)

    def test_controller_collects_all_scheduler_sources(self):
        from sglang.srt.managers import data_parallel_controller as module

        controller = DataParallelController.__new__(DataParallelController)
        controller.env_lock = threading.Lock()
        controller._scheduler_init_info_lock = threading.Lock()
        controller._scheduler_init_infos = []
        controller.scheduler_procs = []
        controller.local_kv_event_sources = []
        controller.run_scheduler_process_func = MagicMock()
        controller.attn_replica_size = 1
        controller.prelaunched_schedulers = None
        # TP8/DP4 over two nodes: this follower's four schedulers own ranks 2,3.
        readers = [
            MagicMock(
                recv=MagicMock(
                    return_value={
                        "max_total_num_tokens": 64,
                        "max_req_input_len": 32,
                        "kv_event_sources": sources,
                    }
                )
            )
            for sources in ([self._source(2)], [], [self._source(3)], [])
        ]
        with (
            get_context().override_server_args(
                tp_size=8,
                dp_size=4,
                enable_dp_attention=True,
                nnodes=2,
                node_rank=1,
                pp_size=1,
                attn_cp_size=1,
            ),
            patch.object(module.mp, "Process"),
            patch.object(
                module.mp,
                "Pipe",
                side_effect=[(reader, MagicMock()) for reader in readers],
            ),
            patch.object(module.PortArgs, "init_new", return_value=MagicMock()),
            patch.object(module.TorchMemorySaverAdapter, "create"),
            patch.object(module.numa_utils, "configure_subprocess"),
            patch.object(
                module, "maybe_reindex_device_id", return_value=nullcontext(0)
            ),
        ):
            controller.launch_tensor_parallel_group(
                MagicMock(), MagicMock(), 0, None, [7000, 7001, 7002, 7003]
            )
        self.assertEqual(
            controller.local_kv_event_sources, [self._source(2), self._source(3)]
        )


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=3, num_waiting_reqs=2),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=5, num_waiting_reqs=1),
            ]
        )
        self.assertEqual(budget.total_requests, [5, 6])

    def test_maps_num_total_tokens_not_num_used_tokens(self):
        budget = DPBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_used_tokens=100, num_total_tokens=150
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_used_tokens=80, num_total_tokens=80
                ),
            ]
        )
        self.assertEqual(budget.total_tokens, [150, 80])

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_running_reqs=10, num_total_tokens=100
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_running_reqs=20, num_total_tokens=200
                ),
                _load(
                    dp_rank=2, timestamp=1.0, num_running_reqs=30, num_total_tokens=300
                ),
            ]
        )
        budget.update_budget(
            [
                _load(
                    dp_rank=1,
                    timestamp=2.0,
                    num_running_reqs=1,
                    num_waiting_reqs=1,
                    num_total_tokens=50,
                )
            ]
        )
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])


class TestRoundRobinScheduler(CustomTestCase):
    def test_cycles_through_active_workers_in_order(self):
        ctl = _make_controller(num_dp_ranks=4)
        for _ in range(8):
            ctl.round_robin_scheduler(_req())
        # 8 reqs across 4 active workers — 2 each, in round-robin order
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(worker.send_pyobj.call_count, 2, f"worker {i} call count")

    def test_first_dispatch_picks_worker_zero(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()
        for i in (1, 2, 3):
            ctl.workers[i].send_pyobj.assert_not_called()

    def test_skips_inactive_workers(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl._active_workers = [0, 2]
        for _ in range(6):
            ctl.round_robin_scheduler(_req())
        # Only workers 0 and 2 are active — should split 6 reqs evenly
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 3)
        ctl.workers[1].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[2].send_pyobj.call_count, 3)
        ctl.workers[3].send_pyobj.assert_not_called()

    def test_raises_when_no_active_workers(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl._active_workers = []
        with self.assertRaisesRegex(RuntimeError, "No active DP workers"):
            ctl.round_robin_scheduler(_req())

    def test_routed_dp_rank_bypasses_counter(self):
        """External dp-rank routing must not advance the counter."""
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req(routed_dp_rank=2))
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(
            ctl.round_robin_counter,
            0,
            "external routing must not advance the round-robin counter",
        )
        # Subsequent round-robin req still lands on worker 0
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()


class TestDispatchingWithTrace(CustomTestCase):
    def _make_traced_controller(self) -> DataParallelController:
        ctl = _make_controller(num_dp_ranks=4)
        ctl.refresh_load_budget_on_dispatch = True
        ctl.refresh_load_budget = MagicMock()
        ctl.dispatching = MagicMock()
        ctl.dispatch_health_check = MagicMock()
        return ctl

    def test_health_and_user_dispatch_use_separate_state(self):
        for rid, is_health in [("HEALTH_CHECK_dispatch", True), ("user", False)]:
            with self.subTest(rid=rid):
                ctl = self._make_traced_controller()
                req = _req(rid=rid)
                ctl.dispatching_with_trace(req)
                self.assertEqual(ctl.refresh_load_budget.call_count, int(not is_health))
                self.assertEqual(
                    ctl.dispatching.call_args_list, [] if is_health else [call(req)]
                )
                self.assertEqual(
                    ctl.dispatch_health_check.call_args_list,
                    [call(req)] if is_health else [],
                )

    def test_batch_refreshes_budget_only_for_user_requests(self):
        for rids, refreshes in [
            (["HEALTH_CHECK_batch"], 0),
            (["HEALTH_CHECK_batch", "user"], 1),
        ]:
            with self.subTest(rids=rids):
                ctl = self._make_traced_controller()
                ctl.dispatching_with_trace = MagicMock()
                reqs = [_req(rid=rid) for rid in rids]
                ctl.dispatch_batch_generate(reqs)
                self.assertEqual(ctl.refresh_load_budget.call_count, refreshes)
                self.assertEqual(
                    ctl.dispatching_with_trace.call_args_list,
                    [call(req, refresh_load_budget=False) for req in reqs],
                )


class TestHealthCheckScheduler(CustomTestCase):
    def test_health_and_user_round_robin_state_are_independent(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req())
        self.assertEqual(ctl.health_round_robin_counter, 0)
        ctl.dispatch_health_check(_req(rid="HEALTH_CHECK"))
        self.assertEqual(ctl.round_robin_counter, 1)
        self.assertEqual(ctl.health_round_robin_counter, 1)
        ctl.round_robin_scheduler(_req())
        self.assertEqual(ctl.round_robin_counter, 2)
        self.assertEqual(ctl.health_round_robin_counter, 1)

    def test_health_round_robin_uses_only_active_workers(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl._active_workers = [1, 3]
        for _ in range(4):
            ctl.dispatch_health_check(_req(rid="HEALTH_CHECK_active"))
        ctl.workers[0].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[1].send_pyobj.call_count, 2)
        ctl.workers[2].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[3].send_pyobj.call_count, 2)

    def test_external_routing_does_not_advance_health_counter(self):
        ctl = _make_controller(num_dp_ranks=4)
        req = _req(routed_dp_rank=2, rid="HEALTH_CHECK_routed")
        ctl.dispatch_health_check(req)
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(ctl.health_round_robin_counter, 0)

    def test_health_preserves_follow_bootstrap_room_routing(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.load_balance_method = LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM

        req = _req(rid="HEALTH_CHECK_pd", bootstrap_room=3)

        ctl.dispatch_health_check(req)

        ctl.workers[3].send_pyobj.assert_called_once()
        self.assertEqual(ctl.health_round_robin_counter, 0)


class TestFollowBootstrapRoomScheduler(CustomTestCase):
    def test_dispatches_by_bootstrap_room_modulo(self):
        ctl = _make_controller(num_dp_ranks=4)
        for room, expected_rank in [
            (0, 0),
            (1, 1),
            (4, 0),
            (5, 1),
            (100, 0),
            (101, 1),
        ]:
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=room))
            ctl.workers[expected_rank].send_pyobj.assert_called()

    def test_requires_bootstrap_room(self):
        ctl = _make_controller(num_dp_ranks=4)
        with self.assertRaises(AssertionError):
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=None))

    def test_routed_dp_rank_bypasses_bootstrap_room(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.follow_bootstrap_room_scheduler(_req(routed_dp_rank=3, bootstrap_room=1))
        ctl.workers[3].send_pyobj.assert_called_once()
        ctl.workers[1].send_pyobj.assert_not_called()


class TestTotalRequestsScheduler(CustomTestCase):
    def test_dispatches_to_min_request_worker(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req())
        ctl.workers[2].send_pyobj.assert_called_once()
        for i in (0, 1, 3):
            ctl.workers[i].send_pyobj.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_requests[2],
            2,
            "DPBudget must record the dispatch by incrementing the counter",
        )

    def test_routed_dp_rank_bypasses_budget(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req(routed_dp_rank=0))
        ctl.workers[0].send_pyobj.assert_called_once()
        # DPBudget must not be touched when bypassed
        self.assertEqual(
            ctl.dp_budget.total_requests,
            [5, 3, 1, 4],
            "external routing must not mutate DPBudget state",
        )


class TestTotalTokensScheduler(CustomTestCase):
    def test_input_embeds_request_is_counted_by_embedding_rows(self):
        # input_embeds requests carry input_ids=None.
        ctl = _make_controller(num_dp_ranks=2)
        ctl.dp_budget.total_tokens = [10, 0]
        ctl.total_tokens_scheduler(_req(input_embeds=[[0.0] * 4] * 3))
        ctl.workers[1].send_pyobj.assert_called_once()
        ctl.workers[0].send_pyobj.assert_not_called()
        self.assertEqual(ctl.dp_budget.total_tokens, [10, 3])


class TestStatusAwarenessInconsistency(CustomTestCase):
    """Document a divergence: ``round_robin_scheduler`` routes only through
    ``_active_workers``, but ``total_requests_scheduler`` / ``total_tokens_scheduler``
    route purely by DPBudget — they do NOT consult ``self.status`` or
    ``_active_workers``. If a future change unifies this behaviour, this test will fail
    and force a reviewer to confirm intent."""

    def test_total_requests_ignores_status(self):
        ctl = _make_controller(num_dp_ranks=4)
        # Worker 2 is the global minimum AND marked inactive.
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.status[2] = False
        ctl.total_requests_scheduler(_req())
        # Current behaviour: still dispatches to the inactive worker.
        ctl.workers[2].send_pyobj.assert_called_once()


class TestUpdateActiveRanks(CustomTestCase):
    def test_refreshes_active_workers_and_masks_unprovisioned_slots(self):
        """Verify that status updates refresh the worker cache and preserve masking of
        unprovisioned slots."""
        ctl = _make_controller(num_dp_ranks=2, max_dp_size=4)
        # Rank 1 dies; trailing True values would leak onto slots 2 and 3
        # if we copied ranks.status without intersecting dp_active.
        ctl.update_active_ranks(ActiveRanksOutput(status=[True, False, True, True]))
        self.assertEqual(ctl.status, [True, False, False, False])
        self.assertEqual(ctl._active_workers, [0])
        self.assertEqual(ctl._active_count_cache, 1)

        ctl.round_robin_scheduler(_req())
        ctl.round_robin_scheduler(_req())
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 2)
        for i in (1, 2, 3):
            ctl.workers[i].send_pyobj.assert_not_called()

        ctl.update_active_ranks(ActiveRanksOutput(status=[True, True, False, False]))
        self.assertEqual(ctl._active_workers, [0, 1])

    def test_length_mismatch_is_ignored(self):
        ctl = _make_controller(num_dp_ranks=2, max_dp_size=4)
        before_status = list(ctl.status)
        before_active = list(ctl._active_workers)
        ctl.update_active_ranks(ActiveRanksOutput(status=[True, False]))
        self.assertEqual(ctl.status, before_status)
        self.assertEqual(ctl._active_workers, before_active)


class TestRefreshLoadBudgetThrottle(CustomTestCase):
    @staticmethod
    def _controller_with_reader(num_dp_ranks, snapshots):
        ctl = _make_controller(num_dp_ranks)
        ctl.load_snapshot_reader = MagicMock()
        ctl.load_snapshot_reader.read_all.return_value = snapshots
        return ctl

    def test_throttled_refresh_spreads_a_burst_across_ranks(self):
        idle = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(4)]
        ctl = self._controller_with_reader(num_dp_ranks=4, snapshots=idle)
        # A refresh stamp in the future keeps every call inside the window, so
        # the burst runs entirely on speculative counters.
        ctl._last_refresh_time = time.perf_counter() + 3600.0

        for _ in range(8):
            ctl.refresh_load_budget()
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 100))

        ctl.load_snapshot_reader.read_all.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [200, 200, 200, 200],
            "speculative increments should spread the burst evenly",
        )
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(
                worker.send_pyobj.call_count, 2, f"worker {i} should get 2 of 8 reqs"
            )

    def test_refresh_outside_window_overwrites_speculative_increments(self):
        reported = [
            _load(dp_rank=0, timestamp=2.0, num_total_tokens=10),
            _load(dp_rank=1, timestamp=2.0, num_total_tokens=20),
        ]
        ctl = self._controller_with_reader(num_dp_ranks=2, snapshots=reported)
        ctl._last_refresh_time = 0.0  # window has long passed
        ctl.dp_budget.total_tokens = [999, 999]

        ctl.refresh_load_budget()

        ctl.load_snapshot_reader.read_all.assert_called_once()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [10, 20],
            "a fresh snapshot must replace the speculative state",
        )
        self.assertGreater(ctl._last_refresh_time, 0.0)

    def test_unchanged_snapshot_does_not_reset_the_burst(self):
        frozen = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(2)]
        ctl = self._controller_with_reader(num_dp_ranks=2, snapshots=frozen)
        ctl._last_refresh_time = 0.0
        ctl.refresh_load_budget()  # adopts timestamp 1.0

        for _ in range(4):
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 50))
        after_burst = list(ctl.dp_budget.total_tokens)

        ctl._last_refresh_time = 0.0  # let the next refresh through the throttle
        ctl.refresh_load_budget()  # same timestamp -> update_budget skips it

        self.assertEqual(
            after_burst, [100, 100], "burst should have spread over both ranks"
        )
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            after_burst,
            "a stale-timestamp snapshot must not wipe the speculative state",
        )


class TestDadpWorkloadControls(CustomTestCase):
    @staticmethod
    def _generate(rid="user", length=8, routed_dp_rank=None):
        return io_struct.TokenizedGenerateReqInput(
            rid=rid,
            input_text=None,
            input_ids=array("q", range(length)),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=16, ignore_eos=True),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=True,
            routed_dp_rank=routed_dp_rank,
        )

    def test_replication_has_one_visible_original_and_abort_covers_shadows(self):
        from sglang.srt.managers.dadp import is_dadp_shadow

        ctl = _make_controller(4)
        req = self._generate(routed_dp_rank=2)
        with patch(
            "sglang.srt.managers.data_parallel_controller.DADP_ROUTE", "replicate"
        ):
            ctl.dispatch_dadp(req)
        envelope = ctl.workers[0].send_pyobj.call_args.args[0]
        self.assertIsInstance(envelope, io_struct.DadpReplicateReqInput)
        self.assertEqual([r.routed_dp_rank for r in envelope.replicas], [0, 1, 2, 3])
        self.assertEqual(
            [r.rid for r in envelope.replicas if not is_dadp_shadow(r.rid)], ["user"]
        )
        self.assertEqual(envelope.replicas[2].rid, "user")
        for replica in envelope.replicas:
            self.assertEqual(replica.input_ids, req.input_ids)
            self.assertTrue(io_struct.AbortReq(rid="user").matches(replica.rid))
            self.assertIsNot(replica.sampling_params, req.sampling_params)
        for worker in ctl.workers[1:]:
            worker.send_pyobj.assert_not_called()

    def test_replication_skips_inactive_ranks_and_checks_primary(self):
        ctl = _make_controller(4)
        ctl._active_workers = [0, 2]
        with patch(
            "sglang.srt.managers.data_parallel_controller.DADP_ROUTE", "replicate"
        ):
            ctl.dispatch_dadp(self._generate())
            envelope = ctl.workers[0].send_pyobj.call_args.args[0]
            self.assertEqual([r.routed_dp_rank for r in envelope.replicas], [0, 2])
            with self.assertRaisesRegex(ValueError, "not active"):
                ctl.dispatch_dadp(self._generate(routed_dp_rank=1))

    def test_replication_envelope_roundtrips_msgpack(self):
        from sglang.srt.managers.dadp import make_dadp_replicas

        req = self._generate()
        envelope = io_struct.DadpReplicateReqInput(
            replicas=make_dadp_replicas(req, [0, 1], 0)
        )
        result = io_struct.msgpack_decode(io_struct.msgpack_encode(envelope))
        self.assertEqual(
            [r.rid for r in result.replicas], [r.rid for r in envelope.replicas]
        )
        self.assertEqual(result.replicas[1].sampling_params.max_new_tokens, 16)

    def test_replicas_enter_each_rank_at_one_global_receive_boundary(self):
        from sglang.srt.managers.dadp import DadpReplicaInbox, make_dadp_replicas
        from sglang.srt.managers.scheduler_components import request_receiver as mod

        replicas = make_dadp_replicas(self._generate(), [0, 1], 0)
        envelope = io_struct.DadpReplicateReqInput(replicas=replicas)
        receiver = mod.SchedulerRequestReceiver.__new__(mod.SchedulerRequestReceiver)
        for rank in [0, 1]:
            for local_control in [True, False]:
                inbox = DadpReplicaInbox()
                with (
                    self.subTest(rank=rank, local_control=local_control),
                    patch.object(
                        mod,
                        "get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(is_ep_scale_joiner=False)
                        ),
                    ),
                    patch.object(mod, "DADP_ROUTE", "replicate"),
                    patch.object(mod, "DADP_REPLICA_INBOX", inbox),
                    patch.object(
                        mod,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            attn_dp_enabled=True,
                            attn_tp_rank=0,
                            attn_cp_rank=0,
                            attn_dp_rank=rank,
                            tp_size=2,
                            enable_dp_attention_local_control_broadcast=local_control,
                            tp_group=SimpleNamespace(
                                rank=rank, ranks=[0, 1], cpu_group=object()
                            ),
                        ),
                    ),
                    patch.object(
                        mod, "attn_cp_tp_broadcast_pyobj", side_effect=lambda x: x
                    ),
                    patch.object(
                        mod,
                        "broadcast_pyobj",
                        side_effect=(
                            [[envelope]] if local_control else [[], [envelope], [], []]
                        ),
                    ) as broadcast,
                ):
                    self.assertEqual(
                        receiver._broadcast_reqs_across_ranks(
                            [envelope] if rank == 0 else []
                        ),
                        [],
                    )
                    self.assertEqual(broadcast.call_count, 0 if local_control else 1)
                    self.assertEqual(len(inbox.envelopes), 1 if rank == 0 else 0)
                    self.assertEqual(receiver.has_pending(), rank == 0)
                    # The intervening existing MLP-sync delivers the same vote.
                    inbox.ready = True
                    self.assertTrue(receiver.has_pending())
                    result = receiver._broadcast_reqs_across_ranks([])
                    self.assertEqual([r.rid for r in result], [replicas[rank].rid])
                    self.assertFalse(inbox.ready)
                    self.assertFalse(inbox.envelopes)
                    self.assertFalse(receiver.has_pending())
                    self.assertEqual(receiver._broadcast_reqs_across_ranks([]), [])
                    self.assertEqual(broadcast.call_count, 1 if local_control else 4)

    def test_abort_deferred_with_envelopes_is_replayed_after_admission(self):
        from sglang.srt.managers.dadp import DadpReplicaInbox, make_dadp_replicas

        replicas = make_dadp_replicas(self._generate(), [0, 1], 0)
        envelope = io_struct.DadpReplicateReqInput(replicas=replicas)
        abort = io_struct.AbortReq(rid=replicas[0].rid)
        inbox = DadpReplicaInbox()
        inbox.defer([envelope], [])
        inbox.defer([], [abort])
        received = inbox.take()
        self.assertIs(received[0], envelope)
        self.assertIs(received[1], abort)
        self.assertTrue(all(abort.matches(r.rid) for r in replicas))
        self.assertFalse(inbox.envelopes)
        self.assertFalse(inbox.aborts)

    def test_heavy_threshold_overrides_explicit_route_only_for_long_requests(self):
        ctl = _make_controller(4)
        ctl.dispatching = MagicMock()
        with (
            patch(
                "sglang.srt.managers.data_parallel_controller.DADP_ROUTE",
                "heavy_to_rank0",
            ),
            patch.dict("os.environ", {"SGLANG_DADP_HEAVY_ISL": "8"}),
        ):
            long = self._generate(length=8, routed_dp_rank=3)
            short = self._generate(length=7, routed_dp_rank=3)
            ctl.dispatch_dadp(long)
            ctl.dispatch_dadp(short)
        ctl.workers[0].send_pyobj.assert_called_once_with(long, flags=0, protocol=5)
        ctl.dispatching.assert_called_once_with(short)

    def test_heavy_input_embeds_uses_row_count(self):
        ctl = _make_controller(4)
        req = self._generate()
        req.input_ids = None
        req.input_embeds = [[0.0]] * 8
        with (
            patch(
                "sglang.srt.managers.data_parallel_controller.DADP_ROUTE",
                "heavy_to_rank0",
            ),
            patch.dict("os.environ", {"SGLANG_DADP_HEAVY_ISL": "8"}),
        ):
            ctl.dispatch_dadp(req)
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 1)

    def test_prefill_cap_preserves_other_sampling_fields_and_zero(self):
        from sglang.srt.managers.dadp import clamp_dadp_max_new_tokens

        for limit, expected in [(16, 1), (None, 1), (0, 0)]:
            params = SamplingParams(
                max_new_tokens=limit, ignore_eos=True, temperature=0.2
            )
            before = msgspec.structs.asdict(params)
            clamp_dadp_max_new_tokens(params)
            self.assertEqual(params.max_new_tokens, expected)
            self.assertEqual(
                {
                    k: v
                    for k, v in msgspec.structs.asdict(params).items()
                    if k != "max_new_tokens"
                },
                {k: v for k, v in before.items() if k != "max_new_tokens"},
            )

    def test_prefill_cap_is_applied_by_tokenizer_before_dispatch(self):
        from sglang.srt.managers import tokenizer_manager as mod

        manager = mod.TokenizerManager.__new__(mod.TokenizerManager)
        manager.preferred_sampling_params = {}
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=32000)
        manager.tokenized_generate_req_type = io_struct.TokenizedGenerateReqInput
        manager.rid_to_state = {"user": SimpleNamespace(time_stats=MagicMock())}
        req = io_struct.GenerateReqInput(
            rid="user",
            input_ids=[1, 2],
            sampling_params={"max_new_tokens": 16, "ignore_eos": True},
        )
        results = []
        for flag in ["", "1"]:
            with (
                get_context().override_server_args(),
                patch.object(mod, "DADP_MAX_NEW_TOKENS", flag),
                patch.object(
                    mod,
                    "get_disagg",
                    return_value=SimpleNamespace(
                        disaggregation_transfer_backend="mooncake"
                    ),
                ),
            ):
                result = manager._create_tokenized_object(req, None, [1, 2])
                results.append(result.sampling_params)
        self.assertEqual([r.max_new_tokens for r in results], [16, 1])
        self.assertTrue(all(r.ignore_eos for r in results))

    def test_shadow_abort_and_release_never_reach_frontend(self):
        from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper

        socket = MagicMock()
        sender = SenderWrapper(socket)
        with patch(
            "sglang.srt.managers.scheduler_components.output_sender.DADP_ROUTE",
            "replicate",
        ):
            sender.send_output(io_struct.AbortReq(rid="user__dadp_shadow_rank_1"))
            sender.send_output(
                io_struct.DetokenizeReleaseReq(rid="user__dadp_shadow_rank_1")
            )
            socket.send_pyobj.assert_not_called()
            sender.send_output(io_struct.AbortReq(rid="user"))
        self.assertEqual(socket.send_pyobj.call_count, 1)

    def test_shadow_stream_is_suppressed_before_detokenization(self):
        from sglang.srt.managers.scheduler_components.output_streamer import (
            SchedulerOutputStreamer,
        )

        shadow = SimpleNamespace(
            rid="user__dadp_shadow_rank_1",
            finished=lambda: True,
            finished_output=False,
            finished_len=None,
            output_ids=[42],
        )
        original = SimpleNamespace(rid="user")
        streamer = SimpleNamespace(
            defer_outputs=False,
            is_generation=True,
            _stream_output_generation=MagicMock(),
        )
        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer.DADP_ROUTE",
            "replicate",
        ):
            SchedulerOutputStreamer.stream_output(streamer, [shadow, original], False)
        streamer._stream_output_generation.assert_called_once_with(
            [original], False, None
        )
        self.assertTrue(shadow.finished_output)


if __name__ == "__main__":
    unittest.main()
