"""#884: every non-capture graph replay of the GDN backend joins the factored side stream first.

The TwinStar boundary DECODE graph (graph-on 1+2 arm) replays through init_forward_metadata_out_graph with a plain
DECODE batch; before the fix only target-verify replays joined (via snapshot_commit), so the boundary token could
invalidate a slot's P checkpoint before the side-stream radix final copy read it.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _Spec:
    def __init__(self):
        self.side = object()
        self.events = []

    def join(self):
        self.events.append("join")

    def snapshot_commit(self, slots):
        self.events.append("snapshot_commit")


def _backend(spec):
    backend = GDNAttnBackend.__new__(GDNAttnBackend)
    backend.factored = SimpleNamespace(spec_state=spec, snapshot_commit=spec.snapshot_commit)
    backend.forward_metadata = SimpleNamespace(mamba_cache_indices=[0, 1])
    return backend


class TestGraphReplayJoin(unittest.TestCase):
    def _replay(self, mode, in_capture=False):
        spec = _Spec()
        backend = _backend(spec)
        order = []

        def base(self, forward_batch, in_capture=False):
            order.append("metadata")
            spec.events.append("metadata")

        batch = SimpleNamespace(forward_mode=mode, actual_forward_mode=mode, batch_size=2, num_padding=0)
        with patch.object(MambaAttnBackendBase, "init_forward_metadata_out_graph", base):
            backend.init_forward_metadata_out_graph(batch, in_capture=in_capture)
        return spec.events

    def test_decode_replay_joins_before_metadata(self):
        self.assertEqual(self._replay(ForwardMode.DECODE), ["join", "metadata"])

    def test_verify_replay_still_snapshots(self):
        events = self._replay(ForwardMode.TARGET_VERIFY)
        self.assertEqual(events[:2], ["join", "metadata"])
        self.assertIn("snapshot_commit", events)

    def test_capture_does_not_join(self):
        self.assertEqual(self._replay(ForwardMode.DECODE, in_capture=True), ["metadata"])

    def test_no_side_stream_no_join(self):
        spec = _Spec()
        spec.side = None
        backend = _backend(spec)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, actual_forward_mode=ForwardMode.DECODE, batch_size=1)
        with patch.object(MambaAttnBackendBase, "init_forward_metadata_out_graph", lambda *a, **k: None):
            backend.init_forward_metadata_out_graph(batch)
        self.assertEqual(spec.events, [])


if __name__ == "__main__":
    unittest.main()
