"""Regression test for the degraded factored-GDN checkpoint guard (TwinStar docs/139, #1149); CPU only.

An invalid cached P checkpoint must not kill P when SGLANG_FLASHNEXT_FACTOR_GUARD_ABORT=1: the plan records the row,
the backend maps it to the request, and after the forward that request is aborted (chunked ones through the native
pending-chunk abort) with the checkpoint flags it published cleared; other requests are untouched. Unset, the guard
still raises.
"""
import os
import unittest
from types import SimpleNamespace as NS

import torch

from sglang.srt.disaggregation import factor_guard
from sglang.srt.disaggregation.utils import is_aborted
from sglang.srt.mem_cache import gdn_factored_pool as gfp
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool

CFG = "r=4,m=4,dtype=fp16,ring=2,strict_chunk=1,factored_prefix=1"


def pool():
    cp = NS(shape=NS(conv=[(64, 3)], temporal=(4, 16, 16)), dtype=NS(conv=torch.bfloat16, temporal=torch.bfloat16),
            is_kda=False, layers=[0, 1])
    return FactoredGDNPool(size=8, cache_params=cp, mamba_layer_ids=[0, 1], device="cpu",
                           cfg=FactoredGDNConfig.parse(CFG))


def request(rid, slot, tracks):
    return NS(rid=rid, return_logprob=False, finished_reason=None, to_finish=None,
              kv=NS(mamba_pool_idx=torch.tensor([slot]), mamba_ping_pong_track_buffer=torch.tensor(tracks)))


class FactorGuard(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("SGLANG_FLASHNEXT_FACTOR_GUARD_ABORT", None)
        gfp.pop_guard_aborts()

    def plan(self, p):
        return p.plan_extend(torch.tensor([1, 2]), [16, 16], prefix_lens=[64, 64], prompt_final=[True, True])

    def test_guard_still_raises_by_default(self):
        p = pool()
        p.prefix_valid[2] = 1
        with self.assertRaisesRegex(RuntimeError, "no factored GDN checkpoint"):
            self.plan(p)

    def test_invalid_row_is_recorded_and_only_its_request_aborted(self):
        os.environ["SGLANG_FLASHNEXT_FACTOR_GUARD_ABORT"] = "1"
        p = pool()
        p.prefix_valid[2] = 1
        self.plan(p)  # does not raise
        self.assertEqual(p.guard_rows[0], [0])
        # the GDN backend's mapping from plan rows to the forward batch's rids
        gfp.report_guard_abort([["r1", "r2"][i] for i in p.guard_rows[0]], p.guard_rows[1])
        bad, good = request("r1", 1, [3, -1]), request("r2", 2, [4, -1])
        p.prefix_valid[[1, 3, 4]] = 1  # what the guarded forward published
        aborted = []
        scheduler = NS(attn_tp_cpu_group=None, chunked_req=bad,
                       req_to_token_pool=NS(factored_gdn_pool=p, translate_mamba_indices=lambda x: x),
                       abort_request=lambda recv: aborted.append(recv.rid))
        before = factor_guard.ABORTED[0]
        self.assertEqual(factor_guard.apply(scheduler, NS(reqs=[bad, good])), 1)
        self.assertTrue(is_aborted(bad))
        self.assertFalse(is_aborted(good))
        self.assertEqual(aborted, ["r1"])  # in-flight chunked request -> native pending-chunk abort
        self.assertEqual(p.prefix_valid[[1, 3]].tolist(), [0, 0])  # nothing it published stays usable
        self.assertEqual(p.prefix_valid[[2, 4]].tolist(), [1, 1])
        self.assertEqual(factor_guard.ABORTED[0], before + 1)
        self.assertEqual(gfp.pop_guard_aborts(), {})

    def test_final_chunk_is_left_to_the_result_loop(self):
        os.environ["SGLANG_FLASHNEXT_FACTOR_GUARD_ABORT"] = "1"
        p = pool()
        gfp.report_guard_abort(["r1"], "cached x256 P prefix has no factored GDN checkpoint")
        req, aborted = request("r1", 1, [-1, -1]), []
        scheduler = NS(attn_tp_cpu_group=None, chunked_req=None,
                       req_to_token_pool=NS(factored_gdn_pool=p, translate_mamba_indices=lambda x: x),
                       abort_request=lambda recv: aborted.append(recv.rid))
        factor_guard.apply(scheduler, NS(reqs=[req]))
        self.assertTrue(is_aborted(req))
        self.assertEqual(aborted, [])


if __name__ == "__main__":
    unittest.main()
