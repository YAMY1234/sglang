"""CUDA parity for private-page planning and the unchanged fused K branch."""
from types import SimpleNamespace
import unittest

import torch

from sglang.srt.mem_cache.flashnext_materialization import make_batch
from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate as norm_rope


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class MaterializationTests(unittest.TestCase):
    def test_noncontiguous_pages_groups_and_short_tails(self):
        torch.manual_seed(403)
        mapping=torch.randperm(2048,device='cuda',dtype=torch.int64)[:2000].reshape(400,5).int()
        pool=SimpleNamespace(page_size=64,qsa_compress_ratio=4,physical_page_map=mapping)
        fb=SimpleNamespace(req_pool_indices=torch.tensor([3],device='cuda'),req_pool_indices_cpu=torch.tensor([3]))
        shapes=[(0,1),(0,3),(0,4),(0,8192),(8192,3463),(65536,3463)]
        # AgentX arrivals include previously unseen short and unaligned tails.
        # Exercise runtime N/G/START, including G=0 and different start offsets.
        shapes += [(8192*(i%29),n) for i,n in enumerate(
            (2,5,15,16,17,63,64,65,255,256,257,308,511,512,513,
             1023,1024,1025,1551,2047,2048,2049,4095,4096,4097,4815,8191))]
        for start,n in shapes:
            with self.subTest(start=start,n=n):
                pages=torch.randperm(398,device='cuda')[:(n+63)//64]+1
                loc=(pages[:,None]*64+torch.arange(64,device='cuda')).flatten()[:n]
                batch=make_batch(fb,0,start,start+n,torch.zeros(n,dtype=torch.int64,device='cuda'),loc,pool,True)
                plan=batch.flashnext_arrival_plan
                self.assertTrue(torch.equal(batch.positions,torch.arange(start,start+n,device='cuda')))
                for layer in range(5):
                    expected=mapping[loc//64,layer].long()*64+loc%64
                    self.assertTrue(torch.equal(plan.kv[layer],expected))
                    expected_c=expected[:n//4*4:4]//4
                    self.assertTrue(torch.equal(plan.compressed[layer].long(),expected_c))
                self.assertTrue(torch.equal(plan.group_rows.long(),torch.arange(n//4*4,device='cuda').reshape(-1,4)))
                self.assertTrue(torch.equal(plan.rope,batch.positions[:,None].expand(-1,3)))
                self.assertEqual(plan.tail,n%4)

    def test_zero_query_heads_keeps_every_key_bit(self):
        torch.manual_seed(403)
        for n in (1,17,8192):
            q=torch.randn(n,12*256*2,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(n,256*2,device='cuda',dtype=torch.bfloat16)[:,:256]
            q_weight=torch.randn(256,device='cuda',dtype=torch.bfloat16)
            k_weight=torch.randn_like(q_weight)
            cache=torch.randn(8200,64,device='cuda')
            positions=torch.arange(n,device='cuda')
            full=norm_rope(q,k,q_weight,k_weight,cache,positions,1e-6,12,1,256,64)[1]
            only=norm_rope(k[:,:0],k,q_weight,k_weight,cache,positions,1e-6,0,1,256,64,has_gate=False)[1]
            self.assertTrue(torch.equal(full.view(torch.int16),only.view(torch.int16)))


if __name__=='__main__':unittest.main()
