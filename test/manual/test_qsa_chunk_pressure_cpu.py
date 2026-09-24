"""Run the real continuation admission method under shared-arena pressure."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'python/sglang/srt/managers/schedule_policy.py'
method = next(n for n in ast.walk(ast.parse(SOURCE.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == 'add_chunked_req')
scope = {'CLIP_MAX_NEW_TOKENS': 4096}
exec('from __future__ import annotations\n' + ast.unparse(method), scope)
admit = scope['add_chunked_req']


def case(remaining, *, prefix=16384, tail=35063, code=True):
    req = NS(prefix_indices=range(prefix), full_untruncated_fill_ids=range(prefix + tail),
             sampling_params=NS(max_new_tokens=1024), retracted_stain=False)
    def set_range(start, end): req.extend_range = NS(start=start, end=end, length=end-start)
    req.set_extend_range = set_range
    allocator = NS(available_for_prefill=lambda *a: remaining)
    if code: allocator.code_pool = object()
    debits = []
    adder = NS(_private_deep_admits=lambda r: True, dllm_config=None,
               rem_total_tokens=remaining, token_to_kv_pool_allocator=allocator,
               tree_cache=object(), rem_total_token_offset=0, rem_chunk_tokens=8192,
               page_size=64, is_hybrid_swa=False, prefill_delayer_single_pass=None,
               can_run_list=[], _mamba_gap_budget_for_req=lambda r: 0,
               _update_prefill_budget=lambda *a, **k: debits.append((a, k)))
    return adder, req, debits


class ChunkPressure(unittest.TestCase):
    def test_reported_6747_budget_keeps_next_qsa_prefix_group_aligned(self):
        adder, req, debits = case(6747)
        self.assertIs(admit(adder, req), req)
        self.assertEqual(req.extend_range.length, 6720)
        self.assertEqual(req.extend_range.end % 64, 0)
        self.assertEqual(req.extend_range.end % 16, 0)
        self.assertEqual(debits[0][0][1], 6720)
        self.assertEqual(debits[0][0][2], 0)

    def test_less_than_one_page_waits_without_consuming_or_losing_request(self):
        for remaining in (0, 1, 15, 63):
            adder, req, debits = case(remaining)
            self.assertIs(admit(adder, req), req)
            self.assertEqual(adder.can_run_list, [])
            self.assertEqual(debits, [])
            self.assertFalse(hasattr(req, 'extend_range'))

    def test_final_short_tail_is_not_dropped(self):
        adder, req, debits = case(6747, tail=6125)
        self.assertIsNone(admit(adder, req))
        self.assertEqual(req.extend_range.length, 6125)
        self.assertEqual(debits[0][0][2], 1024)

    def test_repeated_pressure_then_final_tail_covers_each_input_once(self):
        prefix = 16384; end = prefix + 35063; spans = []
        for remaining in (6747, 63, 9000, 4095, 64, 18000, 9000, 9000):
            adder, req, _ = case(remaining, prefix=prefix, tail=end-prefix)
            unfinished = admit(adder, req)
            if adder.can_run_list:
                spans.extend(range(req.extend_range.start, req.extend_range.end))
                prefix = req.extend_range.end
                if unfinished is not None: self.assertEqual(prefix % 64, 0)
            if unfinished is None: break
        self.assertEqual(spans, list(range(16384, end)))

    def test_stock_a_and_unconstrained_page_chunks_keep_existing_behavior(self):
        for code, remaining, expected in ((False, 6747, 6747), (True, 8192, 8192), (True, 9000, 8192)):
            adder, req, _ = case(remaining, code=code)
            admit(adder, req)
            self.assertEqual(req.extend_range.length, expected)


if __name__ == '__main__': unittest.main()
