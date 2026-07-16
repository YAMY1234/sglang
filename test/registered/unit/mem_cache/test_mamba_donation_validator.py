"""Regression tests for non-blocking Mamba slot donation validation."""

import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import HybridMambaDecodeReqToTokenPool
from sglang.srt.environ import envs
from sglang.srt.mem_cache.mamba_donation_validator import MambaDonationValidator
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-small")


class _DeviceScalar:
    device = types.SimpleNamespace(type="cuda")
    is_cuda = True
    shape = (1,)

    def __init__(
        self,
        value: int,
        *,
        fail_on_item: bool = False,
        item_counter: list[int] | None = None,
    ):
        self.value = value
        self.fail_on_item = fail_on_item
        self._item_counter = item_counter if item_counter is not None else [0]

    @property
    def item_calls(self):
        return self._item_counter[0]

    def numel(self):
        return 1

    def reshape(self, *_shape):
        return self

    def unsqueeze(self, _dim):
        return self

    def clone(self):
        return _DeviceScalar(
            self.value,
            fail_on_item=self.fail_on_item,
            item_counter=self._item_counter,
        )

    def item(self):
        self._item_counter[0] += 1
        if self.fail_on_item:
            raise AssertionError("device scalar was read by the host")
        return self.value


class _PingPongBuffer:
    def __init__(self, scalar: _DeviceScalar):
        self.scalar = scalar

    def __getitem__(self, _idx):
        return self.scalar

    def tolist(self):
        return [self.scalar.value]


class _UnreadableHostBuffer:
    def tolist(self):
        raise AssertionError("host buffer was read before its event completed")


class TestMambaDonationValidation(CustomTestCase):
    @staticmethod
    def _make_pool(*, debug: bool = False):
        pool = object.__new__(HybridReqToTokenPool)
        pool.enable_mamba_extra_buffer_lazy = False
        pool.mamba_ping_pong_track_buffer_size = 2
        pool._debug_mamba_donate = debug
        pool._mamba_donation_validator = MagicMock()
        pool.set_mamba_ping_pong_slot = MagicMock()
        return pool

    @staticmethod
    def _make_req(scalar: _DeviceScalar):
        return types.SimpleNamespace(
            rid="req-7",
            mamba_next_track_idx=0,
            mamba_ping_pong_track_buffer=_PingPongBuffer(scalar),
        )

    def test_default_donation_never_reads_device_scalar(self):
        """A stalled device scalar must not block the scheduler donation call."""
        pool = self._make_pool()
        scalar = _DeviceScalar(5, fail_on_item=True)
        req = self._make_req(scalar)

        donated = pool.donate_mamba_ping_pong_slot(req, [23])

        self.assertIsNot(donated, scalar)
        self.assertEqual(donated.value, scalar.value)
        self.assertEqual(scalar.item_calls, 0)
        pool._mamba_donation_validator.observe.assert_called_once_with(
            donated,
            kind="donated",
            rid="req-7",
            slot_idx=1,
            next_track_idx=0,
        )
        pool.set_mamba_ping_pong_slot.assert_called_once_with(req, 1, 23)

    def test_finished_request_never_reads_device_scalar(self):
        """The finished-request cache callsite must also avoid host scalar reads."""
        scalar = _DeviceScalar(5, fail_on_item=True)
        req = types.SimpleNamespace(
            rid="finished-7",
            origin_input_ids=[1],
            output_ids=[],
            req_pool_idx=0,
            mamba_last_track_seqlen=1,
            cache_protected_len=0,
            extra_key=None,
            mamba_next_track_idx=0,
            mamba_ping_pong_track_buffer=_PingPongBuffer(scalar),
            last_node=object(),
        )
        pool = types.SimpleNamespace(
            req_to_token=torch.tensor([[7]], dtype=torch.int64),
            mamba_ckpt_pool=None,
            get_mamba_ping_pong_keep_idx=MagicMock(return_value=0),
            validate_mamba_slot=MagicMock(),
            free_mamba_cache=MagicMock(),
            poll_mamba_slot_validation=MagicMock(),
        )
        cache = object.__new__(MambaRadixCache)
        cache.disable = False
        cache.enable_mamba_extra_buffer = True
        cache.req_to_token_pool = pool
        cache.token_to_kv_pool_allocator = MagicMock()
        cache.page_size = 1
        cache.insert = MagicMock(return_value=types.SimpleNamespace(mamba_exist=False))
        cache.dec_lock_ref = MagicMock()

        cache.cache_finished_req(req, kv_len_to_handle=1)

        self.assertEqual(scalar.item_calls, 0)
        validated = pool.validate_mamba_slot.call_args.args[0]
        self.assertIsNot(validated, scalar)
        self.assertEqual(validated.value, scalar.value)
        pool.validate_mamba_slot.assert_called_once_with(
            validated,
            req=req,
            slot_idx=0,
            kind="finished",
        )
        pool.poll_mamba_slot_validation.assert_called_once_with()

    def test_debug_donation_keeps_immediate_fail_fast_check(self):
        """The opt-in debug mode must retain request-local invariant failure."""
        pool = self._make_pool(debug=True)
        scalar = _DeviceScalar(-1)
        req = self._make_req(scalar)

        with self.assertRaisesRegex(AssertionError, "Donated Mamba slot is -1"):
            pool.donate_mamba_ping_pong_slot(req, [23])

        self.assertEqual(scalar.item_calls, 1)
        pool._mamba_donation_validator.observe.assert_not_called()
        pool.set_mamba_ping_pong_slot.assert_not_called()

    def test_debug_env_is_cached_when_pool_validator_is_initialized(self):
        pool = object.__new__(HybridReqToTokenPool)
        with envs.SGLANG_DEBUG_MAMBA_DONATE.override(True):
            pool._init_mamba_donation_validator()
        self.assertTrue(pool._debug_mamba_donate)

        with envs.SGLANG_DEBUG_MAMBA_DONATE.override(False):
            pool._init_mamba_donation_validator()
        self.assertFalse(pool._debug_mamba_donate)

    def test_not_ready_event_never_reads_or_waits(self):
        """A never-ready validation event must remain a non-blocking poll."""
        validator = MambaDonationValidator(check_interval=2)
        event = MagicMock()
        event.query.return_value = False
        event.synchronize.side_effect = AssertionError("must not synchronize")
        validator._copy_done = event
        validator._copy_pending = True
        validator._observations_since_poll = 1
        validator._host_values = _UnreadableHostBuffer()

        validator.observe(
            _DeviceScalar(8),
            kind="donated",
            rid="req-8",
            slot_idx=0,
            next_track_idx=0,
        )

        event.query.assert_called_once_with()
        event.synchronize.assert_not_called()
        self.assertTrue(validator._copy_pending)

    def test_ready_event_reports_request_context(self):
        """Completed D2H batches report the offending request without a wait."""
        validator = MambaDonationValidator(check_interval=2)
        event = MagicMock()
        event.query.return_value = True
        event.synchronize.side_effect = AssertionError("must not synchronize")
        validator._copy_done = event
        validator._copy_pending = True
        validator._observations_since_poll = 1
        validator._host_values = torch.tensor([-1, 9], dtype=torch.int64)
        validator._inflight_batch = object()
        validator._inflight_metadata = [
            ("donated", "bad-rid", 1, 0),
            ("donated", "good-rid", 0, 1),
        ]

        with self.assertLogs(
            "sglang.srt.mem_cache.mamba_donation_validator", level="ERROR"
        ) as logs:
            validator.observe(
                _DeviceScalar(10),
                kind="donated",
                rid="next-rid",
                slot_idx=0,
                next_track_idx=1,
            )

        self.assertIn("bad-rid", "\n".join(logs.output))
        event.synchronize.assert_not_called()
        self.assertFalse(validator._copy_pending)
        self.assertIsNone(validator._inflight_batch)

    def test_full_building_batch_repolls_before_dropping(self):
        """A newly ready inflight copy should free capacity before a drop."""
        validator = MambaDonationValidator(check_interval=2)
        event = MagicMock()
        event.query.side_effect = [False, True]
        event.synchronize.side_effect = AssertionError("must not synchronize")
        validator._copy_done = event
        validator._copy_pending = True
        validator._observations_since_poll = 1
        validator._host_values = torch.tensor([7], dtype=torch.int64)
        validator._inflight_batch = object()
        validator._inflight_metadata = [("donated", "inflight-rid", 0, 0)]
        validator._building_values = [_DeviceScalar(8), _DeviceScalar(9)]
        validator._building_metadata = [
            ("donated", "building-1", 0, 0),
            ("donated", "building-2", 1, 0),
        ]

        def start_copy():
            validator._building_values = []
            validator._building_metadata = []
            validator._copy_pending = True

        validator._start_copy = MagicMock(side_effect=start_copy)
        validator.observe(
            _DeviceScalar(10),
            kind="donated",
            rid="next-rid",
            slot_idx=0,
            next_track_idx=1,
        )

        self.assertEqual(event.query.call_count, 2)
        event.synchronize.assert_not_called()
        validator._start_copy.assert_called_once_with()
        self.assertEqual(validator._dropped_observations, 0)
        self.assertEqual(len(validator._building_values), 1)

    def test_decode_pool_clear_flushes_donation_validation(self):
        pool = object.__new__(HybridMambaDecodeReqToTokenPool)
        pool._alloc_size = 4
        pool.free_slots = []
        pool.mamba_allocator = MagicMock()
        pool.poll_mamba_slot_validation = MagicMock()

        pool.clear()

        pool.poll_mamba_slot_validation.assert_called_once_with(flush=True)
        self.assertEqual(pool.free_slots, [1, 2, 3])
        pool.mamba_allocator.clear.assert_called_once_with()

    def test_cuda_batch_copy_reports_after_event_completion(self):
        """A real pinned D2H batch detects -1 after, never during, donation."""
        validator = MambaDonationValidator(check_interval=2)
        validator.observe(
            torch.tensor([7], device="cuda"),
            kind="donated",
            rid="good-rid",
            slot_idx=0,
            next_track_idx=0,
        )
        validator.observe(
            torch.tensor([-1], device="cuda"),
            kind="donated",
            rid="bad-rid",
            slot_idx=1,
            next_track_idx=0,
        )
        torch.cuda.synchronize()

        validator.observe(
            torch.tensor([8], device="cuda"),
            kind="donated",
            rid="poll-1",
            slot_idx=0,
            next_track_idx=0,
        )
        with self.assertLogs(
            "sglang.srt.mem_cache.mamba_donation_validator", level="ERROR"
        ) as logs:
            validator.observe(
                torch.tensor([9], device="cuda"),
                kind="donated",
                rid="poll-2",
                slot_idx=1,
                next_track_idx=0,
            )

        self.assertIn("bad-rid", "\n".join(logs.output))
        torch.cuda.synchronize()

    def test_cuda_finished_value_flushes_partial_batch(self):
        """A request boundary submits fewer than 64 observations for checking."""
        validator = MambaDonationValidator(check_interval=64)
        validator.observe(
            torch.tensor([-1], device="cuda"),
            kind="finished",
            rid="partial-rid",
            slot_idx=0,
            next_track_idx=0,
        )
        self.assertTrue(validator._copy_pending)
        self.assertEqual(len(validator._inflight_metadata), 1)
        torch.cuda.synchronize()

        with self.assertLogs(
            "sglang.srt.mem_cache.mamba_donation_validator", level="ERROR"
        ) as logs:
            validator.poll()

        self.assertIn("partial-rid", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
