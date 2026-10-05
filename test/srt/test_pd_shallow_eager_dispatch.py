"""P241-PUB0: execute real eager batch copying before the exact-tail publisher.

CUDA/math/transport leaves use the established CPU backend. In particular,
EagerRunner.execute/load_batch and registry.extract_buffer are not replaced.
The old tests called model.forward on the *original* batch and missed this copy.
"""
from contextlib import contextmanager
from dataclasses import fields, replace
from types import MethodType
import unittest
from unittest.mock import patch

from pd_shallow_publication_cpu import (
    batch, candidate, field_bytes, model_runner, worker, warmup, torch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.cuda_graph_buffer_registry import CudaGraphBufferRegistry
from sglang.srt.model_executor.runner import eager_runner


def real_batch(rows):
    small = batch(rows)
    values = {k: v for k, v in vars(small).items() if k in {f.name for f in fields(ForwardBatch)}}
    values.update(input_ids=torch.arange(8192 * rows), positions=torch.arange(8192 * rows),
                  extend_seq_lens_cpu=[8192] * rows,
                  seq_lens=torch.full((rows,), 8192, dtype=torch.int64),
                  seq_lens_cpu=torch.full((rows,), 8192, dtype=torch.int64),
                  seq_lens_sum=8192 * rows, out_cache_loc=torch.arange(8192 * rows),
                  forward_metadata_ready=True)
    return ForwardBatch(**values)


@contextmanager
def eager_worker(flag='1', *, rank=16, overlap=False, no_copy='0'):
    def setup(w):
        mr = w.runner
        mr.max_total_num_tokens = 32768
        mr.support_pp = False
        mr.device_timer = None
        mr._pp_kwargs = MethodType(model_runner.ModelRunner._pp_kwargs, mr)
        mr._extend_forward_kwargs = MethodType(model_runner.ModelRunner._extend_forward_kwargs, mr)
        mr.forward = lambda fb: mr.eager_runner.execute(fb)
    recipe = {'SGLANG_GDN_PD_PUBLISH_OVERLAP_OK': str(int(overlap)),
              'SGLANG_EAGER_INPUT_NO_COPY': no_copy}
    with worker(flag, rank=rank, overlap=overlap, recipe=recipe, setup=setup) as w:
        yield w


def require_publication(w, fb):
    result = w.runner.forward(fb)
    assert w.pub.stats['submitted'] > 0 and w.pub.stats['launched'] > 0, w.pub.stats
    w.pub.ticket.complete()
    assert torch.all(w.pool.count[:24, 1:fb.batch_size+1] == w.pool.cfg.r + 1)
    assert torch.all(w.pool.count[24:, 1:fb.batch_size+1] == w.pool.cfg.r)
    return result


class ShallowEagerDispatchTest(unittest.TestCase):
    def test_default_service_warmup_through_real_eager_reaches_ready(self):
        for flag in ('0', '1'):
            with self.subTest(flag=flag), eager_worker(flag) as w:
                receipt = warmup(w, batch_factory=lambda: real_batch(1))
                receipt.killed.assert_not_called()
                receipt.ready.assert_called_once_with()
                if flag == '1':
                    self.assertGreater(w.pub.stats['submitted'], 0)
                    self.assertGreater(w.pub.stats['launched'], 0)

    def test_actual_eager_dispatch_rows_1_2_r8_r16_overlap_and_no_copy(self):
        for rank in (8, 16):
            for overlap in (False, True):
                for no_copy in ('0', '1'):
                    with self.subTest(rank=rank, overlap=overlap, no_copy=no_copy), eager_worker(
                            rank=rank, overlap=overlap, no_copy=no_copy) as w:
                        for rows in (1, 2):
                            require_publication(w, real_batch(rows))
                        self.assertEqual(w.pub.stats['submitted'], 2)
                        self.assertEqual(w.pub.stats['launched'], 2)

    def test_revert_to_dynamic_attribute_must_fail_execution_gate(self):
        # Reproduce exactly what dataclasses.replace did before the field was
        # declared: it discards the dynamically attached selection snapshot.
        original = CudaGraphBufferRegistry.extract_buffer
        def drop_selection(self, **kwargs):
            view = original(self, **kwargs)
            return replace(view, _pd_shallow_publication_selection=None)
        with eager_worker() as w, patch.object(CudaGraphBufferRegistry, 'extract_buffer', drop_selection):
            with self.assertLogs(candidate.logger, level='WARNING') as log:
                with self.assertRaises(AssertionError):
                    require_publication(w, real_batch(2))
            self.assertTrue(any('reason=selection_missing_after_dispatch' in line for line in log.output))
            self.assertEqual(w.pub.stats['submitted'], 0)
            self.assertEqual(w.pub.stats['launched'], 0)

    def test_bad_qualification_must_fail_execution_gate(self):
        with eager_worker() as w:
            fb = real_batch(1)
            fb.spec_info = object()
            with self.assertLogs(candidate.logger, level='WARNING') as log:
                with self.assertRaises(AssertionError):
                    require_publication(w, fb)
            self.assertTrue(any('reason=speculative' in line for line in log.output))
            self.assertEqual(w.pub.stats['submitted'], 0)

    def test_off_and_on_same_tensor_bytes_through_real_eager_views(self):
        for rows in (1, 2):
            values = []
            for flag in ('0', '1'):
                with eager_worker(flag) as w:
                    fb = real_batch(rows)
                    out = w.runner.forward(fb) if flag == '0' else require_publication(w, fb)
                    values.append((out, field_bytes(w)))
                    if flag == '0':
                        self.assertIsNone(w.pub)
                        self.assertIsNone(fb._pd_shallow_publication_selection)
            self.assertTrue(torch.equal(values[0][0], values[1][0]))
            self.assertEqual(values[0][1], values[1][1])

    def test_each_rejected_predicate_has_one_reason_line(self):
        with eager_worker() as w:
            for name, value, reason in (
                ('twinstar_prompt_final', None, 'missing_prompt_final'),
                ('extend_seq_lens_cpu', [1], 'empty_prefix_or_single_token'),
                ('spec_info', object(), 'speculative'),
                ('_pfactor_agg_contract', True, 'fulln_contract'),
            ):
                fb = real_batch(1)
                setattr(fb, name, value)
                with self.assertLogs(candidate.logger, level='WARNING') as log:
                    self.assertIsNone(candidate.select_batch(w.runner, fb))
                self.assertEqual(len(log.output), 1)
                self.assertIn('reason=' + reason, log.output[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
