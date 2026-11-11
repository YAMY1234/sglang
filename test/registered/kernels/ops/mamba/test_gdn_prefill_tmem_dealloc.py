import itertools
import os
import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# Qwen3.8's GDN layer: 16 key heads, 48 value heads, 128-wide.
HQ, HV, D = 16, 48, 128
# Several sequences run the persistent chunked kernel; one long sequence takes
# the CP kernels wherever the CP heuristic (4 * N * HV < SMs) holds.
CASES = {"chunked": (37, 300, 1024, 129), "single_long": (8192,)}


def _inputs(seed: int, seq_lens):
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(seq_lens)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    q = F.normalize(randn(total, HQ, D), dim=-1).to(torch.bfloat16)
    k = F.normalize(randn(total, HQ, D), dim=-1).to(torch.bfloat16)
    v = randn(total, HV, D).to(torch.bfloat16)
    alpha = 0.9 + 0.1 * torch.rand(total, HV, generator=g, device="cuda")
    beta = torch.rand(total, HV, generator=g, device="cuda")
    cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], dtype=torch.int64, device="cuda"
    )
    state = 0.1 * randn(len(seq_lens), HV, D, D)
    return q, k, v, alpha, beta, cu_seqlens, state


def _prefill(inputs):
    # The call gdn_flashinfer.py makes: fp32 gates and state, final state out.
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    q, k, v, alpha, beta, cu_seqlens, state = inputs
    out, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=alpha,
        beta=beta,
        scale=None,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    return out, final_state


class TestGdnPrefillTmemDealloc(CustomTestCase):
    """FlashInfer #8854 only adds a fence and a named barrier before the TMEM free, so
    the fixed kernels must reproduce the stock ones bit for bit, and install() must
    make FlashInfer's own launchers build them (both chunked variants on SM107)."""

    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("FlashInfer's CuTe-DSL GDN prefill kernels need SM10x")

    def _variants(self):
        # 0.7.3 builds RubinGatedDeltaNetChunkedKernel on SM107 unless told otherwise.
        if torch.cuda.get_device_capability() == (10, 7):
            return ("sm107", "sm100")
        return ("sm100",)

    def _run_all(self):
        results = {}
        for variant in self._variants():
            os.environ["FLASHINFER_GDN_SM107_PREFILL_KERNEL"] = variant
            for name, seq_lens in CASES.items():
                inputs = _inputs(len(seq_lens), seq_lens)
                results[variant, name] = (_prefill(inputs), _prefill(inputs))
        os.environ.pop("FLASHINFER_GDN_SM107_PREFILL_KERNEL")
        return results

    def test_fixed_kernels_match_stock(self):
        from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill, gdn_prefill

        from sglang.kernels.ops.mamba import gdn_cp_prefill_tmem_dealloc as cp_fix
        from sglang.kernels.ops.mamba import gdn_prefill_tmem_dealloc as fix

        stock = self._run_all()
        self.assertTrue(fix.install())
        self.assertIs(
            gdn_prefill._PREFILL_KERNEL_CLASSES["sm100"],
            fix.GatedDeltaNetChunkedKernelTmemDealloc,
        )
        self.assertIs(
            gdn_prefill._PREFILL_KERNEL_CLASSES["sm107"],
            fix.RubinGatedDeltaNetChunkedKernelTmemDealloc,
        )
        self.assertIs(
            gdn_cp_prefill.CPDeltaRulePrefillTcgen05Sm100,
            cp_fix.CPDeltaRulePrefillTcgen05Sm100TmemDealloc,
        )
        self.assertEqual(gdn_prefill._CUTE_DSL_MODULE, fix._CUTE_DSL_MODULE)
        fixed = self._run_all()
        for key, ((ref_out, ref_state), (ref_out2, ref_state2)) in stock.items():
            (out, state), (out2, state2) = fixed[key]
            with self.subTest(variant=key[0], case=key[1]):
                # The stock kernels are deterministic, so any difference is the fix's.
                self.assertTrue(torch.equal(ref_out, ref_out2))
                self.assertTrue(torch.equal(ref_state, ref_state2))
                self.assertTrue(bool(torch.isfinite(ref_out).all()))
                self.assertTrue(torch.equal(out, ref_out))
                self.assertTrue(torch.equal(state, ref_state))
                self.assertTrue(torch.equal(out2, ref_out))
                self.assertTrue(torch.equal(state2, ref_state))


if __name__ == "__main__":
    unittest.main()
