"""Exercise production packing/mapping on CPU or FLASHNEXT_TEST_DEVICE=cuda."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

path = Path(__file__).resolve().parents[4]/'python/sglang/srt/mem_cache/flashnext_unified_pool.py'
tree = ast.parse(path.read_text())
scope = dict(torch=torch, QSATokenToKVPool=type('QSAStub',(),{}),
             FlashNextLatentPool=type('LatentStub',(),{}))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]),
             str(path),'exec'),scope)
Pool = scope['FlashNextUnifiedLatentPool']


class PayloadTest(unittest.TestCase):
    def test_parent_emitter_dispatches_before_virtual_address_translation(self):
        # Exercise the actual production MRO. A parent with shallow-only mapping
        # must never translate deep virtual slots or translate a slot twice.
        writes=[]
        def write(pool,layer,locations,*args,**kwargs):
            writes.append((layer.layer_id,locations.clone()))
        def compressed(pool,layer,locations,values):
            writes.append((layer,locations.clone()))
        parent=object.__new__(Pool)
        parent.page_size=64;parent.qsa_compress_ratio=4
        parent.physical_page_map=torch.tensor([[0],[7]],dtype=torch.int32)
        parent._transfer_full_attention_id=lambda layer:{3:0}[layer]
        deep=object.__new__(scope['MappedQSAPool'])
        deep.page_size=64;deep.qsa_compress_ratio=4
        deep.physical_page_map=torch.tensor([[0],[11]],dtype=torch.int32)
        deep._transfer_full_attention_id=lambda layer:{31:0}[layer]
        parent.deep=deep
        with patch.object(scope['QSATokenToKVPool'],'set_kv_buffer',write,create=True), \
             patch.object(scope['QSATokenToKVPool'],'set_qsa_compressed_k_buffer',compressed,create=True), \
             patch.object(scope['FlashNextLatentPool'],'set_kv_buffer',write,create=True), \
             patch.object(scope['FlashNextLatentPool'],'set_qsa_compressed_k_buffer',compressed,create=True):
            for layer in (3,31):
                parent.set_kv_buffer(SimpleNamespace(layer_id=layer),torch.tensor([65]))
                parent.set_qsa_compressed_k_buffer(layer,torch.tensor([17]),None)
        self.assertEqual([(layer,value.item()) for layer,value in writes],
                         [(3,449),(3,113),(31,705),(31,177)])

    def test_bitwise_payload_arbitrary_pages_and_single_token_alignment(self):
        device=os.environ.get('FLASHNEXT_TEST_DEVICE','cpu')
        for n in (1,17,128):
            with self.subTest(tokens=n):
                torch.manual_seed(381)
                fields = dict(z=torch.randint(0,256,(n,2048),dtype=torch.uint8,device=device),
                    z_block_scale=torch.randint(0,120,(n,256),dtype=torch.uint8,device=device),
                    z_scale=torch.randn(n,1,device=device), rms=torch.randn(n,1,device=device),
                    spike_indices=torch.randint(0,256,(n,588),dtype=torch.uint8,device=device),
                    spike_lengths=torch.randint(512,589,(n,1),dtype=torch.int16,device=device),
                    spike_values=torch.randn(n,512,device=device).bfloat16(),
                    token_ids=torch.arange(n,dtype=torch.int32,device=device).reshape(n,1))
                sharded=('z','spike_indices','spike_values','z_block_scale')
                locals=[{k:v.chunk(2,-1)[rank] if k in sharded else v for k,v in fields.items()} for rank in (0,1)]
                pools=[];locations=torch.arange(64,64+n,device=device)
                for rank in (0,1):
                    pool=object.__new__(Pool);pool.device=device;pool.tp_rank=rank
                    pool.unified_k=torch.zeros(32*64,1,256,dtype=torch.bfloat16,device=device)
                    pool.unified_v=torch.zeros_like(pool.unified_k)
                    pool.physical_page_map=torch.zeros(4,9,dtype=torch.int32,device=device)
                    # Nonadjacent physical backing, opposite order across logical pages.
                    pool.physical_page_map[1,7:9]=torch.tensor([13,6],device=device)
                    pool.physical_page_map[2,7:9]=torch.tensor([4,21],device=device)
                    pool.latent_fields=[(k,locals[rank][k].shape[1],locals[rank][k].dtype) for k in pool.wire_field_order]
                    pool.store_latent(locations,SimpleNamespace(**fields),fields['token_ids'])
                    pools.append(pool)
                for rank,pool in enumerate(pools):
                    names=iter(sharded)
                    class Group:
                        world_size=2
                        def all_gather(_,value,dim):
                            name=next(names)
                            self.assertTrue(torch.equal(value,locals[rank][name].contiguous().view(torch.uint8)))
                            return torch.cat([v[name].contiguous().view(torch.uint8) for v in locals],dim)
                    fake=SimpleNamespace(get_tp_group=lambda:Group())
                    with patch.dict(sys.modules,{'sglang.srt.distributed':fake}):
                        out=pool.load_latent(locations)
                    for name,value in fields.items():
                        self.assertTrue(torch.equal(out[name].view(torch.uint8),value.contiguous().view(torch.uint8)),name)
                    # The common padding unit and a live non-payload unit stayed untouched.
                    self.assertEqual(torch.count_nonzero(pool.unified_k[:64]).item(),0)
                    self.assertEqual(torch.count_nonzero(pool.unified_v[7*64:8*64]).item(),0)

    def test_full_and_compressed_addresses_share_page_identity(self):
        obj=object.__new__(scope['MappedQSA'])
        obj.page_size=64;obj.qsa_compress_ratio=4
        obj.physical_page_map=torch.tensor([[0,0],[7,4],[3,9]],dtype=torch.int32)
        obj._transfer_full_attention_id=lambda layer:layer
        positions=torch.tensor([0,64,67,127,128,191])
        for layer in (0,1):
            physical=obj.translate_locations(layer,positions)
            expected=obj.physical_page_map[positions//64,layer]*64+positions%64
            self.assertTrue(torch.equal(physical,expected))
            compressed=obj.translate_locations(layer,positions//4,compressed=True)
            self.assertTrue(torch.equal(physical//4,compressed))


if __name__=='__main__':unittest.main()
