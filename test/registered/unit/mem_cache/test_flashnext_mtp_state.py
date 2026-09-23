"""Production boundary publication and request-private verify page addressing."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


ROOT=Path(__file__).resolve().parents[4]/"python/sglang/srt/mem_cache"
source=ROOT/"flashnext_latent_pool.py"
tree=ast.parse(source.read_text())
scope=dict(torch=torch,copy=copy,QSATokenToKVPool=type("QSAStub",(),{}))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[]),
             str(source),"exec"),scope)
State,Pool=scope["LatentRequestState"],scope["FlashNextLatentPool"]
spec=importlib.util.spec_from_file_location("sglang.srt.mem_cache.gdn_factored_spec",ROOT/"gdn_factored_spec.py")
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)


class FinalMTPStateTest(unittest.TestCase):
    def test_boundary_publishes_consumed_input_only_and_preserves_sink(self):
        state=State(8,"cpu",spec_batch_size=3,spec_tokens=4)
        state.boundary.fill_(17);state.boundary_position.fill_(10)
        state.sink.fill_(31);state.sink_valid.fill_(1)
        candidates=torch.arange(12)[:,None].expand(-1,10240).to(torch.bfloat16)
        positions=torch.arange(61,73)
        state.record_verify_boundary(candidates,positions)
        self.assertTrue(bool((state.boundary==17).all()))
        state.commit_verify_boundary(torch.tensor([2,5]),torch.tensor([0,2]))
        self.assertEqual(state.boundary[2,0].item(),0)
        self.assertEqual(state.boundary[5,0].item(),6)
        self.assertEqual(state.boundary_position[[2,5],0].tolist(),[61,67])
        self.assertTrue(bool((state.boundary[[0,1,3,4,6,7,8]]==17).all()))
        self.assertTrue(bool((state.sink==31).all()))
        self.assertTrue(bool((state.sink_valid==1).all()))

    def test_flag_off_does_not_allocate_candidate_buffers(self):
        state=State(8,"cpu")
        self.assertIsNone(state.boundary_candidates)
        self.assertIsNone(state.position_candidates)

    def test_private_verify_locations_cross_page_without_using_extend_lengths(self):
        pool=object.__new__(Pool)
        pool.device="cpu";pool.speculative_tokens=4
        pool.deep_req_to_token=torch.zeros((4,256),dtype=torch.int32)
        pool.deep_req_to_token[1]=torch.arange(256)+640
        pool.deep_req_to_token[2]=torch.arange(256)+1280
        fm=SimpleNamespace(is_decode_or_idle=lambda:False,is_target_verify=lambda:True)
        fb=SimpleNamespace(forward_mode=fm,req_pool_indices=torch.tensor([1,2,0]),
            positions=torch.tensor([62,63,64,65,126,127,128,129,0,0,0,0]),
            spec_info=SimpleNamespace(draft_token_num=4,topk=1),extend_seq_lens_cpu=None)
        mapped=pool.deep_batch(fb)
        self.assertEqual(mapped.out_cache_loc.tolist(),[702,703,704,705,1406,1407,1408,1409,0,0,0,0])
        self.assertFalse(hasattr(fb,"flashnext_private_locations"))
        self.assertIs(pool.deep_batch(mapped),mapped)

    def test_private_reservation_covers_rejected_tail(self):
        pool=object.__new__(Pool)
        pool.speculative_tokens=4;pool.deep_req_to_token=torch.zeros((3,512))
        req=SimpleNamespace(origin_input_ids=[1]*256,sampling_params=SimpleNamespace(max_new_tokens=64))
        self.assertEqual(pool.request_bound(req),324)
        pool.speculative_tokens=0
        self.assertEqual(pool.request_bound(req),320)


if __name__=="__main__": unittest.main()
