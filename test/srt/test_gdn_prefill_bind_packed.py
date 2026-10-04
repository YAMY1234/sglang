"""Execute the real packing kernel on CPU, plus bank and T-reader ordering."""
from contextlib import ExitStack, nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch

from sglang.srt.mem_cache import gdn_prefill_batch_graph as batch
from sglang.srt.mem_cache import gdn_tracked_slot_side as side
from sglang.srt import runtime_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registered/unit/mem_cache"))
from test_gdn_prefill_batch_graph import fake_pool, make_plan
import test_gdn_tracked_slot_side as original


def buffers(pool, enabled, rows=8, tracked=8, **kwargs):
    with patch.dict(os.environ, {"SGLANG_GDN_PREFILL_BIND_PACKED": str(enabled)}), patch.object(
        runtime_context, "get_disagg", return_value=NS(disaggregation_mode="null")
    ):
        return batch.BatchBuffers(pool, rows, tracked, include_tail=False, **kwargs)


def fields(b):
    return [*b.normal, *(b.tracked or ()), b.slots, b.ring_dst,
            b.track_slots, b.final_src, b.final_dst, b.required, b.ring_pointers]


class PackedBinding(unittest.TestCase):
    def equal_bytes(self, actual, expected):
        for a, b in zip(fields(actual), fields(expected)):
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                self.assertTrue(torch.equal(a.contiguous().view(torch.uint8),
                                            b.contiguous().view(torch.uint8)))

    def test_36_layers_rows_strides_conversions_padding_and_three_launches(self):
        torch.manual_seed(63)
        p = fake_pool(layers=36)
        old, new = buffers(p, 0), buffers(p, 1)
        for count in (1, 2, 4, 8):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for tracked_rows in (None, 1, count):
                    with self.subTest(rows=count, dtype=dtype, tracked_rows=tracked_rows):
                        # Missing tracking uses a separately prewarmed signature.
                        if tracked_rows is None:
                            a, b = buffers(p, 0, tracked=None), buffers(p, 1, tracked=None)
                        else:
                            a, b = old, new
                        plan = make_plan(p, count)
                        states = [(torch.randn(count, 2, 16, 16).to(dtype).transpose(-1, -2),
                                   None if tracked_rows is None else
                                   torch.randn(tracked_rows, 2, 16, 16).to(dtype).transpose(-1, -2))
                                  for _ in range(36)]
                        ts = None if tracked_rows is None else torch.arange(20, 20 + tracked_rows)
                        args = (plan, states, ts, torch.tensor([1]), torch.tensor([38]))
                        a.bind(*args); b.bind(*args)
                        self.equal_bytes(a, b)
                        self.assertEqual(b.packed_bind.last_launches, 3)
                        self.assertEqual(b.packed_bind.fallbacks, {})

    def test_nan_payload_signed_zero_and_bf16_widening_are_byte_exact(self):
        p = fake_pool(layers=1, width=256, heads=1)
        a, b = buffers(p, 0, 1, None), buffers(p, 1, 1, None)
        # Every BF16 bit pattern, including signaling/quiet NaNs and subnormals.
        x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        args = (make_plan(p), [(x.reshape(1, 1, 256, 256), None)], None, None, None)
        a.bind(*args); b.bind(*args); self.equal_bytes(a, b)
        x = (torch.arange(65536, dtype=torch.int32) * 65537).view(torch.float32)
        args = (make_plan(p), [(x.reshape(1, 1, 256, 256), None)], None, None, None)
        a.bind(*args); b.bind(*args); self.equal_bytes(a, b)

    def test_ring_reallocation_is_in_same_packed_control_launch(self):
        p = fake_pool(layers=36)
        a, b = buffers(p, 0), buffers(p, 1)
        p.dense_ring = torch.ones_like(p.dense_ring); p.ring_generation += 1
        plan = make_plan(p, 3)
        args = (plan, [(torch.ones(3, 2, 16, 16), torch.zeros(1, 2, 16, 16))]*36,
                torch.tensor([20]), None, None)
        a.bind(*args); b.bind(*args); self.equal_bytes(a, b)
        self.assertEqual(b.packed_bind.last_launches, 3)
        self.assertEqual(b.ring_generation, p.ring_generation)

    def test_exact_self_views_skip_valid_rows_and_zero_only_padding(self):
        p = fake_pool(layers=2)
        a, b = buffers(p, 0), buffers(p, 1)
        for x in (a, b):
            for layer, tensor in enumerate(x.normal): tensor.fill_(3 + layer)
            for layer, tensor in enumerate(x.tracked): tensor.fill_(7 + layer)
            x.bind(make_plan(p, 3), [(n[:3], t[:1]) for n, t in zip(x.normal, x.tracked)],
                   torch.tensor([20]), None, None)
        self.equal_bytes(a, b)
        self.assertEqual(b.packed_bind.fallbacks, {})

    def test_cross_layer_alias_falls_back_before_any_upload_or_write(self):
        p = fake_pool(layers=2)
        b = buffers(p, 1)
        before = [None if x is None else x.clone() for x in fields(b)]
        args = (make_plan(p), [(b.normal[1][:1], b.tracked[0][:1]),
                               (b.normal[0][:1], b.tracked[1][:1])],
                torch.tensor([20]), None, None)
        self.assertFalse(b.packed_bind.run(*args))
        self.assertEqual(b.packed_bind.last_launches, 0)
        for a, x in zip(fields(b), before):
            if a is not None: self.assertTrue(torch.equal(a, x))

    def test_private_descriptor_bank_wait_precedes_refill(self):
        p = fake_pool(layers=2); b = buffers(p, 1)
        event = Mock(); event.query.return_value = False
        state = torch.ones(1, 2, 16, 16)
        event.synchronize.side_effect = lambda: state.fill_(4)
        b.packed_bind.events[0] = event
        b.bind(make_plan(p), [(state, state)]*2, torch.tensor([20]), None, None)
        event.synchronize.assert_called_once_with()
        self.assertEqual(b.packed_bind.bank_waits, 1)
        self.assertTrue(torch.equal(b.normal[0][:1], state))
        self.assertEqual(b.packed_bind.bank, 1)

    def test_default_off_and_PD_tail_keep_original_path(self):
        p = fake_pool(layers=2)
        self.assertIsNone(buffers(p, 0).packed_bind)
        with patch.dict(os.environ, {"SGLANG_GDN_PREFILL_BIND_PACKED": "1"}):
            for role in ("prefill", "decode"):
                with patch.object(runtime_context, "get_disagg", return_value=NS(disaggregation_mode=role)):
                    self.assertIsNone(batch.BatchBuffers(p, 1, 1, include_tail=False).packed_bind)
            with patch.object(runtime_context, "get_disagg", return_value=NS(disaggregation_mode="null")):
                self.assertIsNone(batch.BatchBuffers(p, 1, 1, include_tail=True).packed_bind)

    def test_tracked_side_off_on_reuses_original_slot_and_bank_fences(self):
        # Execute TrackedSlotSide.run itself: only the GPU graphs/events are
        # stand-ins. Both paths execute the production pack kernel interpreter.
        for tracked_side in (0, 1):
            with self.subTest(tracked_side=tracked_side):
                p, whole, obj, main, writer = original.SplitDispatch().setup_side()
                reference = buffers(p, 0, 1, 1)
                banks = [buffers(p, 1, 1, 1) for _ in range(2)]
                key = (1, 1, object(), (), False)
                order = []
                for index, b in enumerate(banks):
                    def check(b=b):
                        self.equal_bytes(b, reference); order.append("F")
                    obj.entries[index][key] = (b, NS(replay=check), NS(replay=lambda: order.append("T")))
                plan = make_plan(p)
                states = [(torch.randn(1, 2, 16, 16), torch.randn(1, 2, 16, 16).bfloat16())]*2
                args = (plan, states, torch.tensor([7]), None, None)
                reference.bind(*args)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
                    stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=main))
                    stack.enter_context(patch.object(torch.cuda, "Event", side_effect=original.Event))
                    stack.enter_context(patch.object(torch.cuda, "stream", return_value=nullcontext()))
                    for _ in range(3):
                        if tracked_side:
                            self.assertTrue(obj.run(key, *args))
                        else:
                            banks[0].bind(*args); self.equal_bytes(banks[0], reference)
                self.assertLessEqual(banks[0].packed_bind.last_launches, 4)
                if tracked_side:
                    self.assertEqual(order, ["F", "T"]*3)
                    self.assertEqual(obj.buffer_waits, 1)
                    self.assertEqual(obj.registry.publications, 3)


if __name__ == "__main__":
    unittest.main()
