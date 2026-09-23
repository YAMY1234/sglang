"""Verify-window wrap must not overwrite members of an unfinished QSA group."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt/mem_cache"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("sglang.srt.mem_cache.gdn_factored_spec", "gdn_factored_spec.py")
qsa = load("qsa_verify_under_test", "qsa_verify_state.py")


class TestCausalRing(unittest.TestCase):
    def make(self, offset):
        old = torch.zeros(20,1,2, dtype=torch.bfloat16)
        rope = torch.zeros(20,3, dtype=torch.int64)
        req = torch.tensor([1,3]).repeat_interleave(4)
        start = 100 + offset
        positions = torch.arange(start,start+4).repeat(2)
        for r in (1,3):
            for p in range(start-4,start):
                old[r*4+p%4] = p+r*100
                rope[r*4+p%4] = p
        keys = (positions+req*100).to(torch.bfloat16)[:,None,None].expand(-1,1,2).clone()
        newrope = positions[:,None].expand(-1,3).clone()
        return old,rope,keys,newrope,req,positions

    def test_every_alignment_matches_tokenwise_ring(self):
        for offset in range(4):
            old,rope,keys,newrope,req,positions = self.make(offset)
            groups,coords = qsa.causal_groups(old,rope,keys,newrope,req,positions,4,4)
            for row,p in enumerate(positions.tolist()):
                expected = torch.arange(p-3,p+1) + req[row]*100
                self.assertTrue(torch.equal(groups[row,:,0,0],expected.to(torch.bfloat16)))
                self.assertEqual(coords[row,:,0].tolist(),list(range(p-3,p+1)))

    def test_previous_batch_store_corrupts_unaligned_completed_group(self):
        old,rope,keys,newrope,req,positions = self.make(3)
        # The old target-verify path stores the entire window before compress.
        overwritten = old.clone()
        overwritten[req*4+positions%4] = keys
        correct,_ = qsa.causal_groups(old,rope,keys,newrope,req,positions,4,4)
        self.assertFalse(torch.equal(overwritten[4:8],correct[0]))
        self.assertEqual(correct[0,:,0,0].tolist(),[200,201,202,203])

    def test_commit_zero_or_partial_prefix_and_repeated_wrap(self):
        old,rope,keys,newrope,req,positions = self.make(3)
        pool = SimpleNamespace(qsa_compress_ratio=4,qsa_key_state_buffer_pool=[old],
                               qsa_rope_position_buffer=rope,
                               _transfer_full_attention_id=lambda _:0,
                               get_qsa_key_state_buffer=lambda _:old)
        tx=qsa.QSAVerifyState(pool,3,4)
        for iteration in range(17):
            before=old.clone(); oldrope=rope.clone()
            tx.begin(torch.tensor([1,3]))
            shadow=tx.shadow(0)
            shadow.get_qsa_key_state_buffer(0)[req*4+positions%4]=keys
            tx.record(0,keys,newrope,positions)
            self.assertTrue(torch.equal(old,before))
            self.assertTrue(torch.equal(rope,oldrope))
            steps=torch.tensor([0,2 if iteration==0 else 0])
            tx.commit(steps)
            for row in range(8):
                if row%4<=steps[row//4]:
                    before[req[row]*4+positions[row]%4]=keys[row]
                    oldrope[req[row]*4+positions[row]%4]=newrope[row]
            self.assertTrue(torch.equal(old,before))
            self.assertTrue(torch.equal(rope,oldrope))
            with self.assertRaises(RuntimeError):
                tx.commit(steps)
            positions += (steps+1).repeat_interleave(4)
            keys = (positions+req*100).to(torch.bfloat16)[:,None,None].expand(-1,1,2).clone()
            newrope = positions[:,None].expand(-1,3).clone()


if __name__ == "__main__":
    unittest.main()
