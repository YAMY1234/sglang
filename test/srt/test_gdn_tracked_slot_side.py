"""CPU ordering/ownership plus the actual split graph body's bitwise gate.

CUDA stream scheduling itself is left to the independent GPU A/B. These tests
execute the production graph math with the Triton interpreter, not a second
implementation of the factorization.
"""
import copy
from contextlib import ExitStack, nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch

from sglang.srt.mem_cache import gdn_tracked_slot_side as side
from sglang.srt.mem_cache import gdn_prefill_batch_graph as batch
from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache.gdn_fulln_workspace import FullNWorkspace
from sglang.srt.environ import envs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registered/unit/mem_cache"))
from test_gdn_prefill_batch_graph import fake_pool, make_plan


class Event:
    def __init__(self, action=None):
        self.complete = False
        self.action = action
        self.producer = None

    def query(self):
        return self.complete

    def record(self, stream):
        self.producer = stream
        stream.order.append("record")


class Stream:
    def __init__(self, identity=1, order=None):
        self.cuda_stream = identity
        self.order = [] if order is None else order
        self.events = []
        self.priority = 0

    def wait_event(self, event):
        self.events.append(event)
        self.order.append("wait")
        if event.action is not None:
            event.action()
            event.action = None

    def wait_stream(self, stream):
        self.order.append("wait_stream")


class SlotEvents(unittest.TestCase):
    def test_no_pending_and_unrelated_decode_do_not_wait(self):
        registry, stream = side.SlotPublications(), Stream()
        self.assertEqual(registry.wait([1, 2], stream), 0)
        registry.publish([7, 8], Event())
        self.assertEqual(registry.wait([1, 2], stream), 0)
        self.assertEqual(registry.waits, 0)

    def test_reader_before_completion_waits_only_intersecting_event(self):
        registry, stream = side.SlotPublications(), Stream()
        a, b = Event(), Event()
        registry.publish([4], a); registry.publish([5, 6], b)
        self.assertEqual(registry.wait([5], stream), 1)
        self.assertEqual(stream.events, [b])
        registry.wait([6], stream)
        self.assertEqual(stream.events, [b])
        other = Stream(2)
        registry.wait([5], other)
        self.assertEqual(other.events, [b])
        b.complete = True
        registry.reap()
        self.assertEqual(set(registry.pending), {4})

    def test_cancel_and_reuse_keeps_old_writer_until_fenced(self):
        registry, stream = side.SlotPublications(), Stream()
        value = {7: "old"}
        old = Event(lambda: value.update({7: "T_old"}))
        registry.publish([7], old)
        # Request lifetime ends; its slot is returned to the allocator. There
        # is deliberately no per-request cancellation of the physical writer.
        self.assertEqual(set(registry.pending), {7})
        registry.wait([7], stream)
        value[7] = "reset"
        new = Event(lambda: value.update({7: "T_new"}))
        registry.publish([7], new)
        old.complete = True
        registry.reap()
        self.assertIs(registry.pending[7].event, new)
        registry.wait([7], stream)
        self.assertEqual(value[7], "T_new")
        self.assertEqual(stream.events, [old, new])

    def test_mutating_slot_tensor_invalidates_host_hint(self):
        slots = torch.tensor([1, 2])
        side.remember_slots(slots, [1, 2])
        self.assertEqual(side.slot_hint(slots), (1, 2))
        slots[0] = 9
        self.assertEqual(side.slot_hint(slots), (9, 2))

    def test_no_pending_does_not_read_gpu_indices(self):
        registry = side.SlotPublications()
        tensor = Mock()
        registry.wait_tensors((tensor,), "cuda")
        tensor.tolist.assert_not_called()
        tensor.numel.assert_not_called()

    def test_alias_and_tracked_final_source_use_unsplit_path(self):
        self.assertIsNone(side.alias_reason([1], [2], [1], [3]))
        for values in (([1], [2], [2], [3]), ([1], [1], [], []),
                       ([1], [2, 2], [], []), ([1], [2], [], [2])):
            self.assertIsNotNone(side.alias_reason(*values))


class ReaderTests(unittest.TestCase):
    def pool(self):
        p = fake_pool(layers=2)
        p.device = "cpu"
        p._tracked_slot_publications = side.SlotPublications()
        return p

    def contexts(self, stack, stream):
        stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=stream))
        stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))

    def test_hot_prefix_copy_waits_before_reading_state_and_validity(self):
        p, stream = self.pool(), Stream()
        def publish():
            p.a[:, 7].fill_(4); p.U[:, 7].fill_(5)
            p.W[:, 7].fill_(6); p.count[:, 7].fill_(8)
            p.prefix_valid[7] = 1
        event = Event(publish)
        p._tracked_slot_publications.publish([7], event)
        p._tracked_slot_publications.publish([9], Event())
        with ExitStack() as stack:
            self.contexts(stack, stream)
            native.FactoredGDNPool.copy_slots(p, torch.tensor([7]), torch.tensor([3]))
        self.assertEqual(stream.events, [event])
        for name in ("a", "U", "W", "count"):
            self.assertTrue(torch.equal(getattr(p, name)[:, 7], getattr(p, name)[:, 3]))
        self.assertEqual(p.prefix_valid[3], 1)

    def test_evicted_cancelled_slot_reset_fences_old_writer(self):
        p, stream = self.pool(), Stream()
        event = Event(lambda: p.U[:, 7].fill_(42))
        p._tracked_slot_publications.publish([7], event)
        with ExitStack() as stack:
            self.contexts(stack, stream)
            native.FactoredGDNPool.reset_slots(p, torch.tensor([7]))
        self.assertEqual(stream.events, [event])
        self.assertEqual(torch.count_nonzero(p.U[:, 7]), 0)
        self.assertEqual(p.prefix_valid[7], 0)

    def test_mixed_guard_counter_keeps_F_and_T_contributions(self):
        count = torch.tensor(0, dtype=torch.int64)
        for bad in (3, 17, 0):
            ok = torch.arange(1025) >= bad
            side.count_failures(count, ok)
        self.assertEqual(count.item(), 20)

    def test_false_track_mask_does_not_invalidate_async_prefix(self):
        p = self.pool(); p.prefix_valid[7] = 1
        side.invalidate_masked(p.prefix_valid, torch.tensor([7, 8]), torch.tensor([False, True]))
        self.assertEqual(p.prefix_valid[7], 1)
        self.assertEqual(p.prefix_valid[8], 0)

    def test_whole_pool_transport_is_explicit_all_slot_reader(self):
        p, stream = self.pool(), Stream()
        event = Event(); p._tracked_slot_publications.publish([7], event)
        with ExitStack() as stack:
            self.contexts(stack, stream)
            payload = list(native.FactoredGDNPool.iter_transfer_state_entries(p))
        self.assertEqual(stream.events, [event])
        self.assertEqual(len(payload), 4 * len(p.layer_ids))


class SplitDispatch(unittest.TestCase):
    def setup_side(self):
        p = fake_pool(layers=2)
        p.cfg.strict_chunk = True
        p._generic_prompt_only_state_cache = True
        p._tracked_slot_publications = side.SlotPublications()
        whole = batch.PrefillBatchGraph(include_tail=False)
        current, writer = Stream(1), Stream(2)
        with patch.object(torch.cuda, "Stream", return_value=writer), patch.object(
            torch.cuda, "graph_pool_handle", return_value=object()):
            obj = side.TrackedSlotSide(p, whole)
        return p, whole, obj, current, writer

    def test_default_off_and_PD_are_not_admitted(self):
        p, whole, obj, main, writer = self.setup_side()
        for role in ("null", "prefill", "decode"):
            for enabled in (False, True):
                self.assertEqual(side.eligible(p, role=role, enabled=enabled, include_tail=False),
                                 role == "null" and enabled)
        self.assertFalse(side.eligible(p, role="null", enabled=True, include_tail=True))
        with patch.dict(os.environ, {"SGLANG_GDN_TRACKED_SLOT_SIDE_STREAM": "0"}):
            side.install(p, whole, None)
        self.assertIsNone(whole.tracked_side)

    def test_split_routes_independently_of_fullN_and_overlap(self):
        for fulln in (0, 1):
            for overlap in (0, 1):
                with self.subTest(fulln=fulln, overlap=overlap), patch.dict(os.environ, {
                    "SGLANG_GDN_AGG_FULLN_PREFILL": str(fulln),
                    "SGLANG_GDN_AGG_FULLN_OVERLAP_OK": str(overlap),
                }):
                    p, whole, obj, main, writer = self.setup_side()
                    plan = make_plan(p); key = (1, 1, object(), (), False)
                    b = NS(bind=Mock(side_effect=lambda *a: main.order.append("bind")))
                    f = NS(replay=lambda: main.order.append("F"))
                    t = NS(replay=lambda: writer.order.append("T_store_publish"))
                    obj.entries = ({key: (b, f, t)}, {key: (b, f, t)})
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
                        stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=main))
                        stack.enter_context(patch.object(torch.cuda, "Event", side_effect=Event))
                        stack.enter_context(patch.object(torch.cuda, "stream", return_value=nullcontext()))
                        for _ in range(3):
                            self.assertTrue(obj.run(key, plan, [(None, None)]*2,
                                                   torch.tensor([7]), None, None))
                    self.assertEqual(main.order[:3], ["bind", "F", "record"])
                    self.assertEqual(writer.order[:3], ["wait", "T_store_publish", "record"])
                    self.assertEqual(obj.buffer_waits, 1)
                    self.assertEqual(obj.registry.publications, 3)
                    self.assertEqual(set(obj.registry.pending), {7})

    def test_prewarm_T_banks_do_not_alias_fullN_collector_slabs(self):
        p, whole, obj, main, writer = self.setup_side()
        workspace = FullNWorkspace(p, 8, 32768)
        whole.shared = workspace.shared
        base = batch.BatchBuffers(p, 1, 1, workspace.shared, include_tail=False)
        key = (1, 1, object(), (), False)
        whole.entries = {key: (base, None)}
        with ExitStack() as stack:
            for name, value in (("current_stream", main), ("synchronize", None),
                                ("CUDAGraph", Mock()), ("memory_allocated", 0)):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            for name in ("stream", "graph"):
                stack.enter_context(patch.object(torch.cuda, name, side_effect=lambda *a, **k: nullcontext()))
            stack.enter_context(patch.object(batch.BatchBuffers, "evaluate"))
            obj.prewarm(None)
        inputs = [obj.entries[b][key][0].tracked[0] for b in (0, 1)]
        pointers = [x.untyped_storage().data_ptr() for x in (*inputs, workspace.slabs["tracked"])]
        self.assertEqual(len(set(pointers)), 3)

    def test_default_whole_body_is_unchanged(self):
        import ast
        tree = ast.parse(Path(batch.__file__).read_text())
        node = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "BatchBuffers"
                    for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
        self.assertEqual(node.args.kw_defaults[0].value, "both")
        self.assertFalse(envs.SGLANG_GDN_TRACKED_SLOT_SIDE_STREAM.get())


class SplitBitwise(unittest.TestCase):
    def test_36_layer_r8_r16_same_math_pool_logits_and_nll(self):
        # Actual k31/factor/store kernels with fixed inputs. CPU uses fp64 eigh;
        # GPU graph arithmetic remains subject to dattr's independent gate.
        torch.set_num_threads(1)
        from test_gdn_prefill_k31_batch_graph import _pool, _plan
        for rank in (8, 16):
            for rows in (1, 3):
                with self.subTest(rank=rank, rows=rows):
                    p = _pool(native, 321, rank=rank)
                    q = copy.copy(p)
                    fields = ("a", "U", "W", "count", "stale", "dense_of", "dense_required",
                              "prefix_factored_valid", "dense_ring")
                    for name in fields:
                        setattr(q, name, getattr(p, name).clone())
                    p.cfg = copy.copy(p.cfg); q.cfg = p.cfg
                    torch.manual_seed(61)
                    states = [(torch.randn(rows, 2, 128, 128),
                               torch.randn(rows, 2, 128, 128).bfloat16()) for _ in range(36)]
                    plan = _plan(native, p, rows)
                    track = torch.arange(rows) + 4
                    # F->final copy must not read a tracked slot.
                    src, dst = torch.tensor([0]), torch.tensor([7])
                    bucket = 1 if rows == 1 else 4
                    for target, split in ((p, False), (q, True)):
                        self.addCleanup(patch.stopall)
                        patch.dict(os.environ, {
                            "SGLANG_GDN_K31_CHOLQR_MIXED": "1",
                            "SGLANG_GDN_TRACKED_SLOT_SIDE_STREAM": str(int(split)),
                        }).start()
                        buffers = batch.BatchBuffers(target, bucket, bucket, include_tail=False)
                        buffers.bind(plan, states, track, src, dst)
                        if split:
                            buffers.evaluate(native.factorize_layers, branch="normal")
                            # T is unready. F alone already has the exact same
                            # state needed by the boundary N-1 recurrent step.
                            for name in ("a", "U", "W", "count"):
                                self.assertTrue(torch.equal(getattr(p, name)[:, :rows],
                                                            getattr(q, name)[:, :rows]), name)
                            buffers.evaluate(native.factorize_layers, branch="tracked")
                        else:
                            buffers.evaluate(native.factorize_layers)
                        patch.stopall()
                    for name in fields:
                        self.assertTrue(torch.equal(getattr(p, name), getattr(q, name)), name)
                    # Fixed diagnostic readout, explicitly not full-model NLL.
                    logits_p = p.U[:, track, :, :rank].float().sum((0, 2, 3))
                    logits_q = q.U[:, track, :, :rank].float().sum((0, 2, 3))
                    self.assertTrue(torch.equal(logits_p, logits_q))
                    self.assertTrue(torch.equal(torch.log_softmax(logits_p, -1),
                                                torch.log_softmax(logits_q, -1)))


if __name__ == "__main__":
    unittest.main()
