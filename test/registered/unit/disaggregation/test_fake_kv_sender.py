import time
import unittest
from contextlib import ExitStack
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation import prefill as prefill_mod
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVSender
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestFakeKVSender(unittest.TestCase):
    """A FakeKVSender whose send() never comes must not pin the prefill
    inflight queue forever, and a healthy request must never be failed early."""

    def setUp(self):
        self.mgr = self._make_mgr()
        self.sender = self._make_sender()

    def _make_mgr(self) -> FakeKVManager:
        return FakeKVManager(
            args=KVArgs(),
            disaggregation_mode=DisaggregationMode.PREFILL,
            server_args=MagicMock(),
        )

    def _make_sender(self, mgr=None) -> FakeKVSender:
        return FakeKVSender(
            mgr=self.mgr if mgr is None else mgr,
            bootstrap_addr="fake_addr:1234",
            bootstrap_room=42,
        )

    def _expire_deadline(self, sender: FakeKVSender, by: float = 1.0) -> None:
        """Push the armed deadline `by` seconds past the timeout."""
        sender.waiting_since -= sender.waiting_timeout + by

    def test_polls_before_send_stay_waiting_for_input(self):
        """Repeated polling must not conclude on its own: the scheduler polls a
        waiting request once per loop iteration, and those are healthy requests."""
        self.sender.init(3, 5)
        for _ in range(1000):
            self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(self.sender.conclude_state)

    def test_zero_page_last_chunk_still_sends(self):
        """A zero-page last chunk must not be gated out: skipping it leaves the
        sender in WaitingForInput forever and pins the prefill inflight queue."""
        self.assertTrue(self.sender.should_send_kv_chunk(0, last_chunk=True))
        self.assertFalse(self.sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(self.sender.should_send_kv_chunk(3, last_chunk=False))

    def test_fully_cached_request_concludes(self):
        """Regression: a request whose last chunk carries no pages still reaches
        a terminal poll state instead of accumulating in the inflight queue."""
        self.sender.init(0, 0)
        page_indices = np.array([], dtype=np.int32)
        # Mirrors the send_kv_chunk gate in SchedulerDisaggregationPrefillMixin.
        if self.sender.should_send_kv_chunk(len(page_indices), True):
            self.sender.send(page_indices)
        self.assertEqual(self.sender.poll(), KVPoll.Success)

    def test_waiting_timeout_fails_a_stuck_sender(self):
        self.sender.init(1, 0)
        self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)

        self._expire_deadline(self.sender)
        self.assertEqual(self.sender.poll(), KVPoll.Failed)
        self.assertEqual(self.sender.conclude_state, KVPoll.Failed)
        # Terminal: stays Failed even if send() arrives late.
        self.sender.send(np.array([0], dtype=np.int32))
        self.assertEqual(self.sender.poll(), KVPoll.Failed)

    def test_poll_does_not_read_the_timeout_off_the_manager(self):
        """Regression: a FAKE_BOOTSTRAP_HOST req on a real transfer backend pairs
        FakeKVSender with that backend's KVManager, which defines no
        waiting_timeout in prefill mode. Reading the knob off the manager raised
        AttributeError in the scheduler loop on the first poll after init()."""

        class ManagerWithoutWaitingTimeout:
            pass

        sender = self._make_sender(mgr=ManagerWithoutWaitingTimeout())
        sender.init(1, 0)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

        self._expire_deadline(sender)
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_queue_and_prefill_time_is_not_charged_to_the_deadline(self):
        """The scheduler stops polling between init() and the last chunk, so the
        deadline starts at the first poll. Arming it at init() instead would fail
        a healthy request that merely queued for longer than the timeout."""
        self.sender.init(1, 0)
        self.assertIsNone(self.sender.waiting_since)

        after_init = time.monotonic()
        self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)
        self.assertGreaterEqual(self.sender.waiting_since, after_init)

    def test_no_timeout_before_init(self):
        """A sender still in the bootstrap queue has no deadline, matching the
        real backends, which cover that window with the bootstrap timeout."""
        self.assertFalse(self.sender.inited)
        for _ in range(100):
            self.assertEqual(self.sender.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(self.sender.waiting_since)


class _RecordingSender:
    """A real-backend stand-in: records send() and claims host-index support."""

    supports_host_tensor_indices = True

    def __init__(self):
        self.kv_mgr = SimpleNamespace(
            kv_shard_size=1, enable_all_cp_ranks_for_transfer=False
        )
        self.calls = []

    def get_max_transfer_tokens(self):
        return None

    def should_send_kv_chunk(self, num_pages, last_chunk):
        return num_pages > 0 or last_chunk

    def send(self, kv_indices, state_indices=None, num_kv_tokens=None, **kwargs):
        self.calls.append((kv_indices, state_indices, kwargs))


class TestFakeSenderAsyncIndices(unittest.TestCase):
    """SGLANG_DISAGG_FAKE_SENDER_ASYNC_INDICES: the fake sender takes Mooncake's
    async_d2h readback (no .cpu()) and still concludes; other senders keep the
    existing gate."""

    TOTAL, PAGE = 200, 8

    def setUp(self):
        self.req_to_token = (torch.arange(self.TOTAL, dtype=torch.int32) + 64).reshape(
            1, -1
        )

    def _make(self, sender, *, start=0, end=None, bootstrap_host="2.2.2.2"):
        req = SimpleNamespace(
            rid="r",
            kv=SimpleNamespace(req_pool_idx=0),
            origin_input_ids=[0] * self.TOTAL,
            extend_end=self.TOTAL if end is None else end,
            start_send_idx=start,
            disagg_decode_prefix_len=0,
            disagg_kv_sender=sender,
            bootstrap_host=bootstrap_host,
            bootstrap_room=7,
            pending_bootstrap=False,
            metadata_buffer_index=0,
            discard_output_reason=None,
            to_finish=None,
            finished_reason=None,
            return_logprob=False,
            return_sampling_mask=False,
            grammar=None,
        )
        scheduler = SimpleNamespace(
            enable_staging=False,
            chunked_req=None,
            kv_checksum_computer=None,
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=self.PAGE,
                translate_kv_indices_for_transfer=lambda x: x,
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=self.req_to_token,
                req_index_to_mamba_index_mapping=torch.tensor([17]),
                translate_mamba_indices=lambda x: x,
            ),
            disagg_metadata_buffers=Mock(),
            disagg_prefill_bootstrap_queue=SimpleNamespace(
                kv_manager=SimpleNamespace(
                    kv_args=SimpleNamespace(state_types=[StateType.MAMBA])
                )
            ),
            disagg_prefill_pending_chunk_rids=set(),
            disagg_prefill_early_sent_rids=set(),
            disagg_prefill_early_send_stats={"early": 0, "waited": 0},
        )
        mixin = SchedulerDisaggregationPrefillMixin
        scheduler._send_kv_chunk = partial(mixin._send_kv_chunk, scheduler)
        scheduler._settle_early_sent_last_chunk = partial(
            mixin._settle_early_sent_last_chunk, scheduler
        )
        return scheduler, req

    def _run(self, flag, fn, *args, backend="mooncake", **kwargs):
        """Run fn under the flag; return (result, .cpu() calls, async_d2h mock)."""
        real_cpu = torch.Tensor.cpu
        cpu_calls = []

        def counting_cpu(t, *a, **k):
            cpu_calls.append(t.shape)
            return real_cpu(t, *a, **k)

        d2h = Mock(side_effect=prefill_mod.async_d2h)
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_DISAGG_FAKE_SENDER_ASYNC_INDICES.override(flag)
            )
            stack.enter_context(
                get_context().override_server_args(
                    disaggregation_transfer_backend=backend
                )
            )
            stack.enter_context(patch.object(torch.Tensor, "cpu", counting_cpu))
            stack.enter_context(patch.object(prefill_mod, "async_d2h", d2h))
            stack.enter_context(patch.object(torch.cuda, "Event", Mock()))
            stack.enter_context(patch.object(torch.cuda, "current_stream", Mock()))
            out = fn(*args, **kwargs)
        return out, cpu_calls, d2h

    def _send(
        self, sender, flag, *, start=0, end=None, last_chunk=True, backend="mooncake"
    ):
        scheduler, req = self._make(sender, start=start, end=end)
        _, cpu_calls, d2h = self._run(
            flag,
            SchedulerDisaggregationPrefillMixin._send_kv_chunk,
            scheduler,
            req,
            last_chunk=last_chunk,
            backend=backend,
        )
        return req, cpu_calls, d2h

    @staticmethod
    def _host(pages):
        # The async path hands over the pinned tensor, as Mooncake receives it.
        return pages.numpy() if isinstance(pages, torch.Tensor) else pages

    def _expected_pages(self, start, end):
        return kv_to_page_indices(self.req_to_token[0, start:end], self.PAGE)

    def _fake_sender(self):
        # On the rig a fake sender shares the real Mooncake manager; early-send
        # settling reads its request_status / _staging_outstanding.
        sender = FakeKVSender(
            mgr=SimpleNamespace(
                request_status={},
                _staging_outstanding={},
                kv_shard_size=1,
                enable_all_cp_ranks_for_transfer=False,
            ),
            bootstrap_addr="fake_addr:1234",
            bootstrap_room=7,
        )
        sender.init(-(-self.TOTAL // self.PAGE), 0)
        sender.send = Mock(wraps=sender.send)
        return sender

    def test_flag_on_fake_sender_skips_cpu_sync_and_concludes(self):
        for backend in ("mooncake", "fake"):
            with self.subTest(backend=backend):
                self._check_flag_on_fake_send(backend)

    def _check_flag_on_fake_send(self, backend):
        sender = self._fake_sender()
        _, cpu_calls, d2h = self._send(sender, True, backend=backend)
        self.assertEqual(cpu_calls, [])
        # KV pages + mamba state, both through the pinned readback.
        self.assertEqual(d2h.call_count, 2)
        sender.send.assert_called_once()
        args, kwargs = sender.send.call_args
        self.assertIsNotNone(kwargs["indices_event"])
        self.assertIsInstance(args[0], torch.Tensor)
        expected = self._expected_pages(0, self.TOTAL)
        self.assertEqual(self._host(args[0]).dtype, expected.dtype)
        self.assertEqual(self._host(args[0]).tobytes(), expected.tobytes())
        self.assertEqual(int(args[1][0][0]), 17)
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_flag_on_fake_sender_chunked_sends_match_sync_pages(self):
        """Non-last chunk ends page-aligned; the last chunk sends the rest."""
        sender = self._fake_sender()
        req, cpu_calls, _ = self._send(sender, True, end=77, last_chunk=False)
        self.assertEqual(req.start_send_idx, 72)
        self.assertEqual(sender.poll(), KVPoll.Success)  # fake: any send concludes
        req2, cpu_calls2, _ = self._send(sender, True, start=72, last_chunk=True)
        self.assertEqual(cpu_calls + cpu_calls2, [])
        sent = [self._host(c.args[0]) for c in sender.send.call_args_list]
        self.assertEqual(sent[0].tobytes(), self._expected_pages(0, 72).tobytes())
        self.assertEqual(
            sent[1].tobytes(), self._expected_pages(72, self.TOTAL).tobytes()
        )
        self.assertEqual(req2.start_send_idx, self.TOTAL)

    def test_flag_off_fake_sender_unchanged(self):
        sender = self._fake_sender()
        _, cpu_calls, d2h = self._send(sender, False)
        d2h.assert_not_called()
        self.assertEqual(len(cpu_calls), 2)  # KV pages + mamba state
        args, kwargs = sender.send.call_args
        self.assertNotIn("indices_event", kwargs)
        self.assertEqual(
            args[0].tobytes(), self._expected_pages(0, self.TOTAL).tobytes()
        )
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_flag_on_non_fake_sender_keeps_backend_gate(self):
        for backend, expect_async in (("nixl", False), ("mooncake", True)):
            with self.subTest(backend=backend):
                sender = _RecordingSender()
                _, cpu_calls, d2h = self._send(sender, True, backend=backend)
                (kv_indices, _, kwargs), = sender.calls
                self.assertEqual("indices_event" in kwargs, expect_async)
                self.assertEqual(bool(d2h.call_count), expect_async)
                self.assertEqual(not cpu_calls, expect_async)
                self.assertEqual(
                    self._host(kv_indices).tobytes(),
                    self._expected_pages(0, self.TOTAL).tobytes(),
                )

    def test_early_send_eligibility_matches_a_real_sender(self):
        for flag in (False, True):
            with self.subTest(flag=flag):
                real_sched, real_req = self._make(
                    _RecordingSender(), bootstrap_host="10.0.0.1"
                )
                fake_sched, fake_req = self._make(self._fake_sender())
                mixin = SchedulerDisaggregationPrefillMixin
                eligible = mixin._pp_pd_early_send_eligible
                with envs.SGLANG_DISAGG_FAKE_SENDER_ASYNC_INDICES.override(flag):
                    self.assertTrue(eligible(real_sched, real_req))
                    self.assertEqual(eligible(fake_sched, fake_req), flag)
                    # The other gates still apply to the fake request.
                    fake_req.return_logprob = True
                    self.assertFalse(eligible(fake_sched, fake_req))

    def test_flag_on_fake_sender_early_send_then_result_concludes(self):
        """Non-last stage (no aux) and last stage (first token as aux): the early
        send uses the async readback, and the result's send is a no-op."""
        for first_output_id in (None, 5):
            with self.subTest(first_output_id=first_output_id):
                sender = self._fake_sender()
                scheduler, req = self._make(sender)
                wait_event = None if first_output_id is not None else object()
                _, cpu_calls, _ = self._run(
                    True,
                    SchedulerDisaggregationPrefillMixin._pp_pd_send_last_chunk_early,
                    scheduler,
                    req,
                    wait_event,
                    first_output_id,
                )
                self.assertEqual(cpu_calls, [])
                args, kwargs = sender.send.call_args
                self.assertIsNotNone(kwargs["indices_event"])
                self.assertEqual(
                    kwargs.get("send_aux", True), first_output_id is not None
                )
                self.assertEqual(
                    self._host(args[0]).tobytes(),
                    self._expected_pages(0, self.TOTAL).tobytes(),
                )
                if first_output_id is None:
                    scheduler.disagg_metadata_buffers.set_buf.assert_not_called()
                else:
                    scheduler.disagg_metadata_buffers.set_buf.assert_called_once_with(
                        req, first_output_id
                    )
                self.assertIn(req.rid, scheduler.disagg_prefill_early_sent_rids)
                self.assertEqual(sender.poll(), KVPoll.Success)
                # The result's last-chunk send settles at once and sends nothing.
                SchedulerDisaggregationPrefillMixin.send_kv_chunk(
                    scheduler, req, last_chunk=True
                )
                self.assertEqual(sender.send.call_count, 1)
                self.assertNotIn(req.rid, scheduler.disagg_prefill_early_sent_rids)
                self.assertEqual(
                    scheduler.disagg_prefill_early_send_stats["waited"], 0
                )


if __name__ == "__main__":
    unittest.main()
