"""CPU regression for short unfinished prefill starvation (854917)."""
import unittest
import torch
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool, FactoredGDNConfig


def pool(ring=2):
    p=FactoredGDNPool.__new__(FactoredGDNPool)
    p.cfg=FactoredGDNConfig(ring=ring,strict_chunk=1);p.layer_ids=[0];p.layer_map={0:0};p.device='cpu'
    p.stale=torch.ones(12,dtype=torch.int32);p.dense_of=torch.full((12,),-1,dtype=torch.int32)
    p.dense_required=torch.zeros(12,dtype=torch.int32)
    p.prefix_dense_valid=None;p.prefix_factored_valid=None;p.ring_owner=[-1]*ring;p.ring_lru=list(range(ring))
    p.stats=dict(extends=0,rows=0,ring_src=0,ring_miss=0)
    return p


def own(p,slot,pos,required):
    p.ring_owner[pos]=slot;p.dense_of[slot]=pos;p.stale[slot]=0;p.dense_required[slot]=required


def plan(p,slots,lens,final):
    return p.plan_extend(torch.tensor(slots),lens,prompt_final=final)


class ExtendPriorityTest(unittest.TestCase):
    def test_short_unfinished_cannot_be_starved_by_completed_rows(self):
        p=pool();q=plan(p,[1,2,3],[100,99,1],[True,True,False])
        self.assertEqual(q.ring_dst.tolist(),[1,-1,0]);self.assertEqual(q.dense_required_after_commit.tolist(),[0,0,1])

    def test_optional_owned_slots_do_not_preempt_required_new_row(self):
        p=pool();own(p,1,0,False);own(p,2,1,False)
        p.dense_ring=torch.arange(2,dtype=torch.float32).reshape(1,2,1,1,1)
        q=plan(p,[1,2,3],[100,99,1],[True,True,False])
        self.assertGreaterEqual(q.ring_dst[2],0)
        self.assertEqual(q.use_ring.tolist(),[True,True,False])
        # Inputs retain their original sources even when destinations move.
        self.assertEqual(q.ring_src[:2].tolist(),[0,1])
        self.assertEqual(p.dense_ring[0][q.ring_src[:2]].flatten().tolist(),[0.,1.])

    def test_external_unfinished_owner_is_protected(self):
        p=pool();own(p,9,0,True)
        q=plan(p,[1,2],[100,1],[True,False])
        self.assertEqual(q.ring_dst.tolist(),[-1,1]);self.assertEqual(p.ring_owner[0],9)

    def test_true_capacity_exhaustion_still_fails_without_publication(self):
        p=pool();own(p,8,0,True);own(p,9,1,True)
        with self.assertRaisesRegex(RuntimeError,'exhausted'):
            plan(p,[1],[1],[False])
        self.assertEqual(p.ring_owner,[8,9]);self.assertEqual(p.dense_of[1].item(),-1)

    def test_owned_required_continuation_preserved(self):
        p=pool();own(p,3,0,True);own(p,1,1,False)
        q=plan(p,[1,2,3],[100,99,1],[True,True,False])
        self.assertEqual(q.ring_dst.tolist(),[1,-1,0]);self.assertTrue(q.use_ring[2])

    def test_all_final_keeps_existing_policy_and_ignores_padding(self):
        p=pool();own(p,2,1,False)
        q=plan(p,[1,2,-1],[100,1],[True,True])
        self.assertEqual(q.ring_dst.tolist(),[0,1,-1])

if __name__=='__main__':unittest.main()
