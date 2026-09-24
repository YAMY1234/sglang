"""Real prebuilt scheduler entry: a fake probe can arrive with h31 work."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


class PrebuiltPhaseTest(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'python/sglang/srt/disaggregation/flashnext_shallow.py'
        context = ModuleType('sglang.srt.runtime_context')
        context.get_disagg = lambda: NS(disaggregation_decode_enable_radix_cache=False)
        self.modules = patch.dict(sys.modules, {'sglang.srt.runtime_context': context})
        self.modules.start()
        spec = importlib.util.spec_from_file_location('sglang.srt.disaggregation.flashnext_shallow', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.phase_module = patch.dict(sys.modules, {spec.name: self.module})
        self.phase_module.start()
        source = ROOT / 'python/sglang/srt/disaggregation/decode.py'
        cls = next(n for n in ast.parse(source.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'SchedulerDisaggregationDecodeMixin')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_new_prebuilt_batch')
        self.calls = []
        outer = self
        class Batch:
            @staticmethod
            def init_new(reqs, *args):
                outer.calls.append(tuple(r.rid for r in reqs))
                return NS(reqs=reqs, prepare_for_prebuilt=lambda: None,
                          process_prebuilt=lambda future: None)
        namespace = dict(ScheduleBatch=Batch, get_disagg=context.get_disagg,
                         set_time_batch=lambda *args: None)
        exec(compile('from __future__ import annotations\n' + ast.unparse(fn), str(source), 'exec'), namespace)
        self.entry = namespace['get_new_prebuilt_batch']

    def tearDown(self):
        self.phase_module.stop()
        self.modules.stop()

    def request(self, rid, phase=None):
        value = NS(rid=rid, kv=NS(kv_committed_len=None),
                   init_next_round_input=lambda tree: None)
        if phase is not None:
            value.flashnext_pd_boundary_pending = phase
        return value

    def scheduler(self, reqs, capacity=8):
        self.completed = []
        def complete(batch, scheduler):
            self.assertTrue(all(r.flashnext_pd_boundary_pending for r in batch.reqs))
            self.completed.extend(r.rid for r in batch.reqs)
            for req in batch.reqs:
                req.flashnext_pd_boundary_pending = False
        return NS(waiting_queue=list(reqs), grammar_manager=NS(has_waiting_grammars=lambda: False),
                  enable_priority_scheduling=False, req_to_token_pool=NS(size=capacity),
                  max_running_requests=capacity, tree_cache=object(), token_to_kv_pool_allocator=object(),
                  model_config=object(), enable_overlap=False, spec_algorithm=object(), future_map=object(),
                  tp_worker=NS(model_runner=NS(model=NS(complete_pd_boundary=complete))))

    def test_real_request_and_health_probe_arrive_together(self):
        real = self.request('a33bb787954642caadbc76cca2a47d8d', True)
        fake = self.request('HEALTH_CHECK_cb09c69afa2a4380b8277f040c468428')
        scheduler = self.scheduler([real, fake])
        idle = NS(batch_size=lambda: 0)
        self.entry(scheduler, running_batch=idle)
        self.assertEqual(scheduler.waiting_queue, [fake])
        self.entry(scheduler, running_batch=idle)
        self.assertEqual(self.calls, [(real.rid,), (fake.rid,)])
        self.assertEqual(self.completed, [real.rid])
        self.assertFalse(hasattr(fake, 'flashnext_pd_boundary_pending'))

    def test_fake_first_preserves_real_order_and_no_starvation(self):
        fake = self.request('HEALTH_CHECK')
        a, b, c = [self.request(x, True) for x in 'abc']
        scheduler = self.scheduler([fake, a, b], capacity=2)
        idle = NS(batch_size=lambda: 0)
        self.entry(scheduler, running_batch=idle)
        self.assertEqual(scheduler.waiting_queue, [a, b])
        scheduler.waiting_queue.append(c)
        self.entry(scheduler, running_batch=idle)
        self.assertEqual(self.calls, [('HEALTH_CHECK',), ('a', 'b')])
        self.assertEqual(scheduler.waiting_queue, [c])

    def test_stock_flag_off_and_uniform_boundary_keep_original_list(self):
        for reqs in ([], [self.request('stock')], [self.request('r', False)],
                     [self.request('a', True), self.request('b', True)]):
            selected, deferred = self.module.partition_prebuilt(reqs)
            self.assertIs(selected, reqs)
            self.assertEqual(deferred, [])
        scheduler = self.scheduler([self.request('stock1'), self.request('stock2')])
        self.entry(scheduler, running_batch=NS(batch_size=lambda: 0))
        self.assertEqual(self.calls, [('stock1', 'stock2')])
        self.assertEqual(self.completed, [])


if __name__ == '__main__':
    unittest.main()
