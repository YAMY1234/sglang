"""CPU state-transition checks for the opt-in PD admission prototype.

Load the actual methods without importing CUDA-only serving dependencies. Queue
doubles model ownership/credits; they do not implement the scheduling policy.
GPU execution, TP agreement and accuracy still require the paired serving runs.
"""
import ast
import logging
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

SOURCE = Path(__file__).resolve().parents[2] / "python/sglang/srt/disaggregation/decode.py"


def load_method(cls, name, namespace):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    code = ast.Module(body=future + [node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(SOURCE), "exec"), namespace)
    return namespace[name]


class AdmissionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.disagg = NS(disaggregation_decode_polling_interval=1,
                         disaggregation_decode_host_receive_threshold=0,
                         disaggregation_decode_enable_radix_cache=False)
        self.env = NS(SGLANG_PD_DECODE_ADMISSION_PIPELINE=NS(get=lambda: True))
        self.ns = dict(get_disagg=lambda: self.disagg, envs=self.env,
                       time=time, logger=logging.getLogger(__name__))
        self.process = load_method("SchedulerDisaggregationDecodeMixin",
                                   "_process_decode_queue_pipelined", self.ns)

    def scheduler(self, running=6, ready=0):
        pre = NS(queue=list(range(20)), retracted_queue=[], tp_rank=0)
        transfer = NS(queue=[], extend=lambda reqs: self.events.append(("extend", len(reqs))))
        transfer.pop_transferred = lambda: self.events.append("drain") or []
        pre.resume_retracted_reqs = lambda: self.events.append("resume") or []
        pre.pop_preallocated = lambda **kw: self.events.append(("allocate", kw["max_new_requests"])) or ([], [])
        return NS(disagg_decode_prealloc_queue=pre, disagg_decode_transfer_queue=transfer,
                  waiting_queue=[object() for _ in range(ready)], enable_hisparse=False,
                  running_batch=NS(reqs=[NS(finished=lambda: False) for _ in range(running)]),
                  req_to_token_pool=NS(size=8), max_running_requests=8)

    def test_completion_releases_metadata_before_new_allocation(self):
        s = self.scheduler()
        credit = [0]
        completed = object()
        def drain():
            credit[0] += 1
            return [completed]
        def allocate(**kw):
            self.assertEqual(credit[0], 1)
            self.assertIn(completed, s.waiting_queue)
            credit[0] -= 1
            return [object()], []
        s.disagg_decode_transfer_queue.pop_transferred = drain
        s.disagg_decode_prealloc_queue.pop_preallocated = allocate
        self.process(s)
        self.assertEqual(credit[0], 0)
        self.assertEqual(s._decode_admission_pipeline_stats["completed"], 1)

    def test_retraction_stall_does_not_starve_completed_transfer(self):
        s = self.scheduler()
        completed = object()
        s.disagg_decode_transfer_queue.pop_transferred = lambda: [completed]
        s.disagg_decode_prealloc_queue.retracted_queue = [object()]
        self.process(s)
        self.assertEqual(s.waiting_queue, [completed])
        self.assertFalse(any(isinstance(e, tuple) and e[0] == "allocate" for e in self.events))

    def test_active_decoder_bounds_bulk_work_and_refills_deficit(self):
        for running, ready, want in [(8, 0, 2), (6, 0, 2), (1, 0, 7), (1, 6, 2)]:
            with self.subTest(running=running, ready=ready):
                self.events.clear()
                self.process(self.scheduler(running, ready))
                self.assertIn(("allocate", want), self.events)
                self.assertLess(self.events.index("drain"), self.events.index(("allocate", want)))

    def test_idle_decoder_is_not_throttled(self):
        self.process(self.scheduler(0))
        self.assertIn(("allocate", None), self.events)

    def test_disabled_switch_keeps_legacy_prealloc_then_transfer_order(self):
        self.env.SGLANG_PD_DECODE_ADMISSION_PIPELINE.get = lambda: False
        self.disagg.disaggregation_decode_enable_offload_kvcache = False
        method = load_method("SchedulerDisaggregationDecodeMixin", "process_decode_queue", self.ns)
        s = self.scheduler()
        s.enable_decode_hicache = False
        s.disagg_decode_transfer_queue.resolve_deferred_releases = lambda: self.events.append("release")
        s.disagg_decode_prealloc_queue.pop_preallocated = lambda: self.events.append("allocate") or ([], [])
        method(s)
        self.assertLess(self.events.index("release"), self.events.index("allocate"))
        self.assertLess(self.events.index("allocate"), self.events.index("drain"))

    def test_poll_interval_preserved_while_retraction_resume_runs(self):
        self.disagg.disaggregation_decode_polling_interval = 3
        s = self.scheduler()
        self.process(s)
        self.process(s)
        self.assertEqual(self.events, ["resume", "resume"])
        self.process(s)
        self.assertEqual(self.events.count("drain"), 1)
        self.assertEqual(self.events.count("resume"), 3)

    def test_finished_batch_members_do_not_consume_admission_credit(self):
        method = load_method("SchedulerDisaggregationDecodeMixin", "_get_new_prebuilt_batch", self.ns)
        live = [object() for _ in range(6)]
        batch = NS(reqs=live+[object(), object()])
        batch.batch_size = lambda: len(batch.reqs)
        batch.filter_batch = lambda: setattr(batch, "reqs", live)
        reqs = [NS(last_node=None, kv=NS(kv_committed_len=None),
                   init_next_round_input=lambda _: None) for _ in range(4)]
        self.ns.update(set_time_batch=lambda *_: None,
                       ScheduleBatch=NS(init_new=lambda rs, *_: NS(reqs=rs, prepare_for_prebuilt=lambda: None)))
        s = NS(grammar_manager=NS(has_waiting_grammars=lambda: False), waiting_queue=reqs.copy(),
               enable_priority_scheduling=False, req_to_token_pool=NS(size=8), max_running_requests=8,
               tree_cache=object(), token_to_kv_pool_allocator=object(), model_config=object(),
               enable_overlap=False, spec_algorithm=object())
        admitted = method(s, batch)
        self.assertEqual(admitted.reqs, reqs[:2])
        self.assertEqual(s.waiting_queue, reqs[2:])

    def test_zero_burst_still_cleans_failed_handshakes(self):
        class Abort:
            pass
        self.ns["FINISH_ABORT"] = Abort
        method = load_method("DecodePreallocQueue", "pop_preallocated", self.ns)
        good = NS(req=NS(finished_reason=None), waiting_for_input=True)
        receiver = MagicMock()
        bad = NS(req=NS(finished_reason=Abort(), finished_output=False, return_logprob=False),
                 kv_receiver=receiver)
        s = self.scheduler()
        s.enable_priority_scheduling = False
        s.enable_lora = False
        s.running_batch.reqs = []
        s.output_streamer = MagicMock()
        q = NS(pp_size=1, queue=[good, bad], pending_reqs=[], scheduler=s,
               _resolve_pending_reqs=lambda: None, _update_handshake_waiters=lambda *_: None,
               _uses_swa_tail_prealloc=lambda: False, _uses_swa_reservation=lambda: False,
               _allocatable_token_budgets=lambda **_: 100, _hicache_pending_restore_tokens=lambda: 0)
        admitted, failed = method(q, max_new_requests=0)
        self.assertEqual(admitted, [])
        self.assertEqual(failed, [bad])
        self.assertEqual(q.queue, [good])
        receiver.clear.assert_called_once()
        self.assertIsNone(bad.kv_receiver)


if __name__ == "__main__":
    unittest.main()
