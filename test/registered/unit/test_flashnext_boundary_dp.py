"""Boundary dispatch must rendezvous across DP2 x TP2, including empty ranks."""
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


class BoundaryDPTest(unittest.TestCase):
    def setUp(self):
        self.decode, self.idle, self.prebuilt = object(), object(), object()
        context = ModuleType('sglang.srt.runtime_context')
        context.get_disagg = lambda: NS(flashnext_pd_shallow_prefill=True)
        forward = ModuleType('sglang.srt.model_executor.forward_batch_info')
        forward.ForwardMode = NS(DECODE=self.decode, IDLE=self.idle)
        self.modules = patch.dict(sys.modules, {
            'sglang.srt.runtime_context': context,
            'sglang.srt.model_executor.forward_batch_info': forward,
        })
        self.modules.start()
        path = ROOT/'python/sglang/srt/disaggregation/flashnext_shallow.py'
        spec = importlib.util.spec_from_file_location('boundary_test', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.enabled = lambda _: True

    def tearDown(self):
        self.modules.stop()

    def run_layout(self, counts, legacy_rank=None):
        barrier = threading.Barrier(4, timeout=4)
        votes, callbacks, originals = {}, {}, {}

        def rank_main(rank):
            dp, tp = divmod(rank, 2)
            n = counts[dp]
            reqs = [NS(flashnext_pd_boundary_pending=True, output_ids=[-1]) for _ in range(n)]
            original = NS(reqs=reqs, forward_mode=self.prebuilt) if n else None
            if dp == legacy_rank:
                original = NS(reqs=[NS(output_ids=[42])], forward_mode=self.prebuilt)
            originals[rank] = original
            events = []
            def gather(local):
                events.append('gather')
                self.assertEqual(events[0], 'fence')
                if local is not None:
                    self.assertIs(local.forward_mode, self.decode)
                    self.assertIsNot(local, original)
                votes[rank] = len(local.reqs) if local else 0
                barrier.wait()
                self.assertEqual(votes[dp*2], votes[dp*2+1])
                global_counts = [votes[0], votes[2]]
                if not any(global_counts):
                    return None
                if local is None:
                    local = NS(reqs=[], forward_mode=self.idle)
                local.global_num_tokens = global_counts
                local.global_num_tokens_for_logprob = global_counts
                return local
            def complete(work, scheduler):
                self.assertEqual(work.global_num_tokens, list(counts))
                self.assertEqual(len(votes), 4)
                work.sampling_info = 'sampled'
                for req in work.reqs:
                    req.output_ids[:] = [7]
                    req.flashnext_pd_boundary_pending = False
                callbacks[rank] = len(work.reqs)
                # Every rank must reach the same deep-layer collective.
                barrier.wait()
            scheduler = NS(model_config=object(), enable_overlap=True,
                forward_stream=object(), schedule_stream=NS(wait_stream=lambda _:events.append('fence')),
                dp_attn_adapter=NS(ps=NS(attn_dp_size=2), prepare_mlp_sync_batch=gather),
                tp_worker=NS(model_runner=NS(model=NS(complete_pd_boundary=complete))))
            self.module.complete_prebuilt(scheduler, original)
            if n:
                self.assertIs(original.forward_mode, self.prebuilt)
                self.assertEqual(original.sampling_info, 'sampled')
                self.assertTrue(all(r.output_ids == [7] for r in original.reqs))
            elif original is not None:
                self.assertEqual(original.reqs[0].output_ids, [42])
                self.assertFalse(hasattr(original, 'sampling_info'))

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(rank_main, range(4)))
        self.assertEqual(len(callbacks), 4 if any(counts) else 0)
        return callbacks

    def test_boundary_on_dp0_empty_dp1(self):
        self.assertEqual(self.run_layout((3, 0)), {0:3, 1:3, 2:0, 3:0})

    def test_boundary_on_dp1_health_probe_on_dp0(self):
        self.run_layout((0, 1), legacy_rank=0)

    def test_uneven_boundaries(self):
        self.run_layout((1, 5))

    def test_no_boundary_work(self):
        self.run_layout((0, 0))

    def test_mixed_local_batch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'mixed legacy'):
            self.module.complete_prebuilt(NS(), NS(reqs=[
                NS(flashnext_pd_boundary_pending=True), NS()]))

    def test_scheduler_calls_boundary_even_without_new_batch(self):
        source = ROOT/'python/sglang/srt/disaggregation/decode.py'
        cls = next(n for n in ast.parse(source.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'SchedulerDisaggregationDecodeMixin')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'get_next_disagg_decode_batch_to_run')
        fn.decorator_list = []
        calls = []
        boundary = ModuleType('sglang.srt.disaggregation.flashnext_shallow')
        boundary.complete_prebuilt = lambda scheduler,batch:calls.append(('boundary',batch))
        scope = dict(NextBatchPlan=lambda **kwargs:NS(**kwargs))
        exec(compile('from __future__ import annotations\n'+ast.unparse(fn), str(source), 'exec'), scope)
        scheduler = NS(get_new_prebuilt_batch=lambda _:None,
            dp_attn_adapter=NS(maybe_prepare_mlp_sync_batch=lambda b:calls.append(('decode',b))))
        with patch.dict(sys.modules, {boundary.__name__:boundary}):
            scope[fn.name](scheduler, NS(is_empty=lambda:True))
        self.assertEqual(calls, [('boundary',None), ('decode',None)])


if __name__ == '__main__':
    unittest.main()
