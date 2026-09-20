import unittest

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import abort_inflight_prefill_sender


class FakeSender:
    def __init__(self, poll_result):
        self.poll_result = poll_result
        self.abort_calls = 0
        self.poll_calls = 0

    def poll(self):
        self.poll_calls += 1
        if isinstance(self.poll_result, Exception):
            raise self.poll_result
        return self.poll_result

    def abort(self):
        self.abort_calls += 1


class TestAbortInflightPrefillSender(unittest.TestCase):
    def test_pp_completed_transfer_ignores_late_abort(self):
        sender = FakeSender(KVPoll.Success)

        aborted = abort_inflight_prefill_sender(sender, pp_size=2)

        self.assertFalse(aborted)
        self.assertEqual(sender.poll_calls, 1)
        self.assertEqual(sender.abort_calls, 0)

    def test_pp_transferring_sender_is_aborted(self):
        sender = FakeSender(KVPoll.Transferring)

        aborted = abort_inflight_prefill_sender(sender, pp_size=4)

        self.assertTrue(aborted)
        self.assertEqual(sender.poll_calls, 1)
        self.assertEqual(sender.abort_calls, 1)

    def test_pp_poll_error_falls_back_to_abort(self):
        sender = FakeSender(RuntimeError("poll failed"))

        aborted = abort_inflight_prefill_sender(sender, pp_size=2)

        self.assertTrue(aborted)
        self.assertEqual(sender.poll_calls, 1)
        self.assertEqual(sender.abort_calls, 1)

    def test_pp1_success_retains_abort_behavior(self):
        sender = FakeSender(KVPoll.Success)

        aborted = abort_inflight_prefill_sender(sender, pp_size=1)

        self.assertTrue(aborted)
        self.assertEqual(sender.poll_calls, 0)
        self.assertEqual(sender.abort_calls, 1)


if __name__ == "__main__":
    unittest.main()
