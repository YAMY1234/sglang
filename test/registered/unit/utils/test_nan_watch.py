"""SGLANG_ENABLE_NAN_WATCH: per-row first appearance of a non-finite value and shared-slot counters, CPU only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.debug_utils import nan_capture, nan_watch
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors


class _Pool:
    """HybridReqToTokenPool stand-in: row -> Mamba slot, the slot free list and req_to_token."""

    def __init__(self, slots, free_slots, req_to_token):
        self.slots = torch.tensor(slots)
        self.mamba_allocator = SimpleNamespace(free_slots=torch.tensor(free_slots))
        self.req_to_token = req_to_token

    def get_mamba_indices(self, req_idx):
        return self.slots[req_idx]


def _runner(pool, free_pages):
    return SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=4, get_all_free_pages=lambda: torch.tensor(free_pages)
        ),
        kv_index_translator=SimpleNamespace(is_translating=False),
    )


def _batch(prefix_lens, extend_lens, out_cache_loc):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens_cpu=extend_lens,
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens=torch.tensor(extend_lens),
        req_pool_indices=torch.arange(len(extend_lens)),
        req_pool_indices_cpu=torch.arange(len(extend_lens)),
        out_cache_loc=torch.tensor(out_cache_loc),
    )


class TestNanWatch(unittest.TestCase):
    def setUp(self):
        nan_watch._pending.clear()
        nan_watch._active = None
        # Row 0: prefix 4 (slots 0-3), extend 2 (slots 40, 41); row 1: no prefix, extend 3 (slots 44-46).
        rtt = torch.zeros((2, 16), dtype=torch.int64)
        rtt[0, :4] = torch.arange(4)
        self.batch = _batch([4, 0], [2, 3], [40, 41, 44, 45, 46])
        self.rtt = rtt

    def _run(
        self,
        *,
        slots=(5, 6),
        free_slots=(7, 8),
        free_pages=(20,),
        hidden=None,
        state=None,
        logits=None,
        layer_outputs=(),
        params=None,
    ):
        pool = _Pool(list(slots), list(free_slots), self.rtt)
        proxy = PPProxyTensors(
            {"hidden_states": hidden if hidden is not None else torch.zeros(5, 3)}
        )
        with patch("sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool", _Pool):
            nan_watch.begin(
                self.batch,
                proxy,
                model_runner=_runner(pool, list(free_pages)),
                rids=["a", "b"],
                stage=3,
            )
        ssm = state if state is not None else torch.zeros(10, 2, 2)
        nan_watch.check_state(torch.tensor(list(slots)), ssm, torch.zeros(10, 3))
        for layer, point, h in layer_outputs:
            nan_watch.check_point(layer, point, h)
        if params is not None:
            nan_watch.capture_params(params[0], **params[1])
        nan_watch.end(
            logits=logits if logits is not None else torch.zeros(2, 8),
            forward_batch=self.batch,
        )
        with self.assertLogs(nan_watch.logger, level="INFO") as logs:
            nan_watch.poll()
            nan_watch.logger.info("end")
        return [m for m in logs.output if "NaN watch hit" in m or "slot anomaly" in m]

    def test_clean_forward_logs_nothing(self):
        self.assertEqual(self._run(), [])

    def test_each_place_is_attributed_to_its_row(self):
        hidden = torch.zeros(5, 3)
        hidden[3, 1] = float("nan")  # a token of row 1
        state = torch.zeros(10, 2, 2)
        state[5, 0, 0] = float("inf")  # row 0's slot
        logits = torch.zeros(2, 8)
        logits[1, 2] = float("nan")
        lines = self._run(hidden=hidden, state=state, logits=logits)
        self.assertEqual(len(lines), 2)
        row0 = next(m for m in lines if "rid=a " in m)
        row1 = next(m for m in lines if "rid=b " in m)
        self.assertIn("PP3", row0)
        self.assertIn("in_bad_tokens=0 state_bad_layers=1 out_bad=0", row0)
        self.assertIn("in_bad_tokens=1 state_bad_layers=0 out_bad=1", row1)
        self.assertIn("mamba_slot=6 kv_slots=[44,46]", row1)

    def test_first_bad_layer_point_and_token_offset(self):
        # Row 1 (tokens 2-4) turns non-finite at its second token in layer 39's attention output.
        clean = torch.zeros(5, 3)
        attn = torch.zeros(5, 3)
        attn[3:, 0] = float("nan")
        out = torch.zeros(5, 3)
        out[3:, 1] = float("nan")
        lines = self._run(layer_outputs=[(38, 1, clean), (39, 0, attn), (39, 1, out)])
        self.assertEqual(len(lines), 1)
        self.assertIn("rid=b ", lines[0])
        self.assertIn("points=[(39, 'attn', 2, 1), (39, 'out', 2, 1)]", lines[0])

    def test_transfer_checksums_pair_rows_and_catch_a_changed_row(self):
        received = torch.arange(15.0).reshape(5, 3)
        sent = received.clone()
        sent[3, 1] += 1  # a token of row 1
        pool = _Pool([5, 6], [7, 8], self.rtt)
        with patch("sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool", _Pool):
            nan_watch.begin(
                self.batch,
                PPProxyTensors({"hidden_states": received}),
                model_runner=_runner(pool, [20]),
                rids=["a", "b"],
                stage=3,
            )
        nan_watch.end(
            pp_proxy_tensors=PPProxyTensors({"hidden_states": sent}),
            forward_batch=self.batch,
        )
        with self.assertLogs(nan_watch.logger, level="INFO") as logs:
            nan_watch.poll()
        line = next(m for m in logs.output if "NaN watch xfer PP3" in m)
        got_in, got_out = re.search(r"in=\[(.*)\] out=\[(.*)\]", line).groups()
        got_in, got_out = got_in.split(", "), got_out.split(", ")
        self.assertEqual(got_in[0], got_out[0])
        self.assertNotEqual(got_in[1], got_out[1])

    def test_capture_saves_the_first_layer_bundle_of_a_produced_row(self):
        # Row 0 (prefix 4, tokens 0-1) turns non-finite in layer 39's attention output with a clean input and state.
        attn = torch.zeros(5, 3)
        attn[1, 0] = float("nan")
        later = torch.full((5, 3), float("nan"))  # a later layer: not captured
        with (
            tempfile.TemporaryDirectory() as d,
            envs.SGLANG_NAN_WATCH_CAPTURE_DIR.override(d),
            patch.object(nan_capture, "_MIN_PREFIX", 1),
            patch.object(nan_capture, "_dumps", [0]),
        ):
            self._run(
                layer_outputs=[
                    (39, 2, torch.zeros(5, 3)),
                    (39, 0, attn),
                    (40, 0, later),
                ],
                params=(39, {"A_log": torch.ones(4), "conv_bias": None}),
            )
            saved = os.listdir(d)
            self.assertEqual(saved, ["pp3-a-p4-e2.pt"])
            bundle = torch.load(os.path.join(d, saved[0]))
        self.assertEqual(bundle["meta"]["capture_layer"], 39)
        tensors = bundle["tensors"]
        self.assertEqual(
            sorted(tensors),
            ["attn", "conv", "in", "param.A_log", "recv.hidden_states", "ssm"],
        )
        self.assertTrue(torch.isnan(tensors["attn"][1, 0]))
        self.assertEqual(tuple(tensors["ssm"].shape), (1, 2, 2))

    def test_shared_slots_are_counted(self):
        # Both rows on Mamba slot 5, which is also free; new KV token 41 sits on free page 10.
        lines = self._run(slots=(5, 5), free_slots=(5, 9, 9), free_pages=(10,))
        self.assertEqual(len(lines), 1)
        self.assertIn("slot anomaly", lines[0])
        self.assertIn(
            "'dup_mamba_slot': 1, 'mamba_slot_free': 2, 'mamba_free_dup': 1", lines[0]
        )
        self.assertIn("free_dup_vals=[9] slots_on_free=[5, 5]", lines[0])
        self.assertIn("'new_kv_page_free': 2", lines[0])

    def test_new_kv_on_a_prefix_slot_is_counted(self):
        self.batch.out_cache_loc = torch.tensor([2, 41, 44, 45, 46])
        lines = self._run()
        self.assertIn("'new_kv_in_prefix': 1", lines[0])

    def test_only_extend_forwards_are_watched(self):
        self.batch.forward_mode = ForwardMode.DECODE
        hidden = torch.full((5, 3), float("nan"))
        self.assertEqual(self._run(hidden=hidden), [])


if __name__ == "__main__":
    unittest.main()
