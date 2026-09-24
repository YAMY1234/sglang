"""Use the production unified allocator constructor to cover its real marker."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from test_qsa_chunk_pressure_cpu import admit, case


def actual_unified_allocator():
    source=Path(__file__).resolve().parents[2]/'python/sglang/srt/mem_cache/allocator/flashnext_unified.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef))
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__']
    class PagedBase:
        def __init__(self,*args,**kwargs): self.page_size=64
    scope={'PagedTokenToKVPoolAllocator':PagedBase}
    exec(ast.unparse(cls),scope)
    return scope['FlashNextUnifiedAllocator'](kvcache=object())


def unified_case(remaining,**kwargs):
    adder,req,debits=case(remaining,code=False,**kwargs)
    adder.token_to_kv_pool_allocator=actual_unified_allocator()
    return adder,req,debits


class UnifiedChunkPressure(unittest.TestCase):
    def test_actual_unified_marker_routes_reported_7309_budget_to_alignment(self):
        adder,req,debits=unified_case(7309)
        self.assertFalse(hasattr(adder.token_to_kv_pool_allocator,'code_pool'))
        self.assertTrue(hasattr(adder.token_to_kv_pool_allocator,'unified_pool'))
        self.assertIs(admit(adder,req),req)
        self.assertEqual(req.extend_range.length,7296)
        self.assertEqual(req.extend_range.end%64,0)
        self.assertEqual(debits[0][0][1:3],(7296,0))

    def test_zero_negative_and_subpage_budget_park_without_overallocation(self):
        for remaining in (-100,0,1,15,63):
            adder,req,debits=unified_case(remaining)
            self.assertIs(admit(adder,req),req)
            self.assertEqual(adder.can_run_list,[]);self.assertEqual(debits,[])
            self.assertFalse(hasattr(req,'extend_range'))

    def test_final_nonpage_tail_is_preserved(self):
        adder,req,debits=unified_case(7309,tail=7101)
        self.assertIsNone(admit(adder,req));self.assertEqual(req.extend_range.length,7101)
        self.assertEqual(debits[0][0][2],1024)

    def test_repeated_actual_unified_pressure_covers_inputs_once(self):
        prefix=16384;end=prefix+35063;spans=[]
        for remaining in (7309,63,0,-1,8192,4095,64,18000,8192,8192):
            adder,req,_=unified_case(remaining,prefix=prefix,tail=end-prefix)
            unfinished=admit(adder,req)
            if adder.can_run_list:
                spans.extend(range(req.extend_range.start,req.extend_range.end))
                prefix=req.extend_range.end
                if unfinished is not None:self.assertEqual(prefix%64,0)
            if unfinished is None:break
        self.assertEqual(spans,list(range(16384,end)))

    def test_nonshared_allocator_behavior_is_unchanged(self):
        for remaining,expected in ((7309,7309),(8192,8192),(0,8192)):
            adder,req,_=case(remaining,code=False);admit(adder,req)
            self.assertEqual(req.extend_range.length,expected)


if __name__=='__main__':unittest.main()
