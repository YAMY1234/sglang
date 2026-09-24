"""CPU address/lifecycle tests; production I/O kernels require the GPU guard."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'python/sglang/srt/mem_cache/pool_host/flashnext_stock.py'


class Short:
    enabled = True


class Gram:
    enabled = True


class HostBase:
    def get_size_per_token(self):
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        return self.layer_num * 2 * 256 * 2

    def load_to_device_per_layer(self, *args, **kwargs):
        self.order.append('dense')


tree = ast.parse(SOURCE.read_text())
tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
scope = dict(os=os, torch=torch, ShortConvPool=Short, NGramPool=Gram,
             MambaPoolHost=HostBase, MHATokenToKVPoolHost=HostBase)
exec(compile(tree, str(SOURCE), 'exec'), scope)


class StockHiCacheCPU(unittest.TestCase):
    def test_compressed_address_mapping_preserves_noncontiguous_page_order(self):
        host = scope['FlashNextStockQSAHost'].__new__(scope['FlashNextStockQSAHost'])
        host.ratio = 16; host.page_size = 64
        indices = torch.cat([torch.arange(192, 256), torch.arange(64, 128)])
        self.assertEqual(host._group_indices(indices, 'cpu').tolist(),
                         [12, 13, 14, 15, 4, 5, 6, 7])
        with self.assertRaisesRegex(ValueError, 'complete KV pages'):
            host._group_indices(indices[:-1], 'cpu')

    def test_host_budget_counts_all_target_and_draft_index_bytes(self):
        host = scope['FlashNextStockQSAHost'].__new__(scope['FlashNextStockQSAHost'])
        host.target_layer_num = 12; host.mtp_draft_device_pools = (object(),)
        host.index_heads = 1; host.index_dim = 128
        host.index_dtype = torch.bfloat16; host.ratio = 16
        self.assertEqual(host.get_size_per_token(), 13 * (1024 + 16))

    def test_only_committed_ple_is_selected_and_unknown_siblings_fail(self):
        short = Short(); short.conv_state = torch.randn(3, 9, 2, 8)
        short.intermediate_conv_state = torch.randn(3, 9, 4, 2, 8)
        gram = Gram(); gram.context = torch.arange(36).view(9, 4)
        pool = NS(_slot_siblings=[short, gram],mamba_cache=NS(temporal=torch.ones(1)))
        rows = scope['ple_tensors'](pool)
        self.assertEqual([name for name, _ in rows], ['short_conv', 'ngram'])
        self.assertEqual(rows[0][1].data_ptr(),short.conv_state.data_ptr())
        self.assertEqual(rows[1][1].shape,(1,9,4))
        pool._slot_siblings.append(object())
        with self.assertRaisesRegex(ValueError,'factor/latent'):scope['ple_tensors'](pool)

    def test_ple_restores_on_first_layer_into_different_slots_before_cursor_reset(self):
        cls = scope['FlashNextStockMambaHost']; host = cls.__new__(cls)
        host.order = []; tensor = torch.zeros(3, 9, 2, 8)
        gram = torch.zeros(1, 9, 4, dtype=torch.int64)
        host.ple = [('short', tensor), ('ngram', gram)]
        host.ple_host = [torch.randn(10,3,1,2,8),torch.arange(40).view(10,1,1,4)]
        def copy(src,dst,si,di,layer,nlayers,backend):
            dst[di] = src[si,layer,0]; host.order.append('ple')
        host._copy_tensor_pf_lf=copy
        pool=NS(replayssm_write_pos=torch.full((9,),3))
        hi=torch.tensor([4,1]);di=torch.tensor([7,2])
        host.load_to_device_per_layer(pool,hi,di,1)
        self.assertEqual(host.order,['dense']);self.assertEqual(tensor.sum(),0)
        host.load_to_device_per_layer(pool,hi,di,0)
        self.assertTrue(torch.equal(tensor[:,di],host.ple_host[0][hi,:,0].transpose(0,1)))
        self.assertTrue(torch.equal(gram[:,di],host.ple_host[1][hi,:,0].transpose(0,1)))
        self.assertEqual(pool.replayssm_write_pos[di].tolist(),[0,0])
        self.assertEqual(pool.replayssm_write_pos[1].item(),3)
        self.assertEqual(host.order,['dense','dense','ple','ple','ple','ple'])

    def test_ngram_read_waits_for_completed_restore_only_on_opt_in_path(self):
        path=ROOT/'python/sglang/srt/mem_cache/memory_pool.py'
        tree=ast.parse(path.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_ngram_context')
        ns={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
        order=[]
        obj=NS(mamba_pool=NS(_hicache_restore_ple_before_prefill=True),
            layer_transfer_counter=NS(wait_until=lambda layer:order.append(('wait',layer))),
            ngram_pool=NS(get_context=lambda indices:order.append(('read',indices))))
        ns['get_ngram_context'](obj,5)
        self.assertEqual(order,[('wait',0),('read',5)])
        order.clear();obj.mamba_pool._hicache_restore_ple_before_prefill=False
        ns['get_ngram_context'](obj,5);self.assertEqual(order,[('read',5)])


if __name__=='__main__':unittest.main()
