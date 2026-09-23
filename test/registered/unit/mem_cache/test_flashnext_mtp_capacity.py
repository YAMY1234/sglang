"""CPU regression for target latent geometry plus separate NEXTN KV bytes."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[4]/'python/sglang/srt'
path = ROOT/'model_executor/pool_configurator.py'
tree = ast.parse(path.read_text())
node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashNextLatentPoolConfigurator')
class NoImports(ast.NodeTransformer):
    def visit_ImportFrom(self,node): return None
node = NoImports().visit(node)
spec=importlib.util.spec_from_file_location('latent_layout',ROOT/'mem_cache/flashnext_latent_layout.py')
layout=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=layout;spec.loader.exec_module(layout)

class Base:
    def __init__(self,kvc): self._cell_size=1088*(12+kvc.draft_layers)
    def _compute_cell_size(self,kvc,layers): return 1088*layers
    def calculate_pool_sizes(self,budget,page): return budget//self._cell_size//page*page

scope=dict(DefaultPoolConfigurator=Base, fullstack_latent_config=lambda model:model.fs,
           FlashNextLatentLayout=layout.FlashNextLatentLayout, get_parallel=lambda:NS(attn_tp_size=2),
           get_schedule=lambda:NS(max_mamba_cache_size=20), get_spec=lambda:NS(speculative_num_draft_tokens=4))
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
Pool=scope['FlashNextLatentPoolConfigurator']

class CapacityTest(unittest.TestCase):
    def make(self,draft,shared):
        fs=dict(version=3,deep_private_tokens=4194304,deep_private_allocation='shared-arena' if shared else 'fixed')
        return Pool(NS(model_config=NS(fs=fs,context_len=262144),pool_page_size=64,draft_layers=draft,
                       spec_algorithm=NS(is_none=lambda:not draft),resolve_max_num_reqs=lambda n:4))

    def test_draft_retains_its_own_uncompressed_cost(self):
        for shared,target in ((True,9792),(False,9588)):
            for draft in (0,1,3):
                with self.subTest(shared=shared,draft=draft):
                    pool=self.make(draft,shared)
                    self.assertEqual(pool._cell_size,target+1088*draft)
                    budget=64<<30; tokens=pool.calculate_pool_sizes(budget,64)
                    self.assertLessEqual(tokens*pool._cell_size+pool._private_bytes,budget)
                    self.assertGreater((tokens+64)*pool._cell_size+pool._private_bytes,budget)

    def test_flagoff_private_bytes_unchanged_and_scratch_separate(self):
        off=self.make(0,True);on=self.make(1,True)
        original=5*262144*4+21*(2*10240*2+12)+(64<<20)
        self.assertEqual(off._private_bytes,original)
        self.assertEqual(on._private_bytes-original,5*4*(10240*2+8))

if __name__=='__main__': unittest.main()
