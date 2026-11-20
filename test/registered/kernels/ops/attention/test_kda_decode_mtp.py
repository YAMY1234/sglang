import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention import kda_decode_mtp
from sglang.kernels.ops.attention.kda_decode_mtp import fused_kda_decode_mtp_dspark
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=50, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _make_inputs(num_requests, num_spec, key_dim=128, num_heads=2):
    num_tokens = num_requests * (1 + num_spec)
    num_slots, conv_width = num_requests + 2, 4
    torch.manual_seed(5)
    x_q = torch.randn(
        1,
        num_tokens,
        num_heads,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x_k = torch.randn_like(x_q)
    x_v = torch.randn_like(x_q)
    gate = torch.randn_like(x_q)
    beta = torch.randn(
        1,
        num_tokens,
        num_heads,
        device="cuda",
        dtype=torch.bfloat16,
    )
    conv_weight = [
        torch.randn(
            num_heads * key_dim,
            conv_width,
            device="cuda",
        )
        * 0.1
        for _ in range(3)
    ]
    conv_state = [
        torch.randn(
            num_slots,
            num_heads * key_dim,
            conv_width - 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    slots = torch.arange(1, num_requests + 1, device="cuda", dtype=torch.int32)
    scratch = torch.arange(num_requests, device="cuda", dtype=torch.int32)
    state = torch.randn(
        num_slots,
        num_heads,
        key_dim,
        key_dim,
        device="cuda",
    )
    intermediate_conv = torch.zeros(
        num_requests,
        1 + num_spec,
        num_heads * key_dim,
        conv_width - 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return dict(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=conv_weight[0],
        w_k=conv_weight[1],
        w_v=conv_weight[2],
        cs_q=conv_state[0],
        cs_k=conv_state[1],
        cs_v=conv_state[2],
        g=gate,
        beta=beta,
        A_log=torch.randn(num_heads, device="cuda"),
        dt_bias=torch.randn(num_heads * key_dim, device="cuda"),
        recurrent_state=state,
        intermediate_state_indices=scratch,
        intermediate_conv_q=intermediate_conv.clone(),
        intermediate_conv_k=intermediate_conv.clone(),
        intermediate_conv_v=intermediate_conv.clone(),
        ssm_state_indices=slots,
        cu_seqlens=torch.arange(
            0,
            num_tokens + 1,
            1 + num_spec,
            device="cuda",
            dtype=torch.int32,
        ),
        lower_bound=-5.0,
    )


class TestKdaDecodeMtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def _run(self, num_requests, num_spec, *, onorm, block_threads=None, num_heads=2):
        kwargs = _make_inputs(num_requests, num_spec, num_heads=num_heads)
        _, num_tokens, num_heads, key_dim = kwargs["x_v"].shape
        intermediate = torch.zeros(
            num_requests,
            1 + num_spec,
            num_heads,
            key_dim,
            key_dim,
            device="cuda",
        )
        gate = torch.randn(
            1, num_tokens, num_heads, key_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.rand(key_dim, device="cuda") + 0.5
        if onorm:
            kwargs.update(onorm_gate=gate, onorm_weight=weight, onorm_eps=1e-6)
        with mock.patch.object(
            kda_decode_mtp,
            "_block_threads",
            (lambda **_: block_threads)
            if block_threads
            else kda_decode_mtp._block_threads,
        ):
            output = fused_kda_decode_mtp_dspark(
                intermediate_ssm=intermediate, **kwargs
            )
        return output, intermediate, gate, weight

    def test_output_norm(self):
        # 24 heads: the attention-TP4 rank, folded under SGLANG_K31_KDA_ONORM_FOLD_ANY_HEADS
        for num_spec, num_heads in ((0, 2), (2, 2), (4, 24)):
            with self.subTest(num_spec=num_spec, num_heads=num_heads):
                raw, raw_states, gate, weight = self._run(
                    4, num_spec, onorm=False, num_heads=num_heads
                )
                fused, fused_states, _, _ = self._run(
                    4, num_spec, onorm=True, num_heads=num_heads
                )
                raw = raw.float()
                expected = (
                    raw
                    * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + 1e-6)
                    * weight
                    * torch.sigmoid(gate.float())
                )
                torch.testing.assert_close(
                    fused.float(), expected, atol=3e-2, rtol=3e-2
                )
                torch.testing.assert_close(fused_states, raw_states, atol=0, rtol=0)

    def test_streaming_state_matches_resident(self):
        for onorm in (False, True):
            with self.subTest(onorm=onorm):
                resident = self._run(
                    4, 0, onorm=onorm, block_threads=kda_decode_mtp.BLOCK_THREADS_WIDE
                )
                streaming = self._run(
                    4, 0, onorm=onorm, block_threads=kda_decode_mtp.BLOCK_THREADS_NARROW
                )
                torch.testing.assert_close(
                    streaming[0], resident[0], atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    streaming[1], resident[1], atol=1e-4, rtol=1e-4
                )


class TestKdaVerifyFoldPending(CustomTestCase):
    """Pending accepted ring rows folded inside the next v2 verify (replayssm_pending)
    must match verify + eager commit fold bit for bit, on outputs, rings, conv scratch
    and states, over consecutive steps with mixed accept lengths."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def _steps(self, num_requests, num_heads, pending_mode, num_spec=4, ring_len=16):
        from sglang.kernels.ops.attention.kda_decode_mtp_v2 import (
            fused_kda_decode_mtp_dspark as verify_v2,
        )
        from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
            commit_kda_replayssm_spec,
        )

        kw = _make_inputs(num_requests, num_spec, num_heads=num_heads)
        num_slots, key_dim = kw["recurrent_state"].shape[0], kw["x_q"].shape[-1]
        gen = torch.Generator(device="cuda").manual_seed(11)
        rings = dict(
            replayssm_rawv=torch.zeros(
                num_slots,
                num_heads,
                ring_len,
                key_dim,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            replayssm_rawk=torch.zeros(
                num_slots,
                num_heads,
                ring_len,
                key_dim,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            replayssm_g=torch.zeros(
                num_slots, num_heads, ring_len, key_dim, device="cuda"
            ),
            replayssm_beta=torch.zeros(num_slots, num_heads, ring_len, device="cuda"),
        )
        pending = torch.zeros(num_slots + 1, device="cuda", dtype=torch.int32)
        slots = kw["ssm_state_indices"].to(torch.int64)
        records = []
        for _ in range(3):
            for name in ("x_q", "x_k", "x_v", "g", "beta"):
                kw[name] = torch.randn(kw[name].shape, device="cuda", generator=gen).to(
                    kw[name].dtype
                )
            accept = torch.randint(
                1, num_spec + 2, (num_requests,), device="cuda", generator=gen
            )
            extra = {"replayssm_pending": pending} if pending_mode else {}
            out = verify_v2(intermediate_ssm=None, **rings, **extra, **kw)
            records.append(
                [out.clone(), kw["recurrent_state"].clone()]
                + [kw[f"intermediate_conv_{c}"].clone() for c in "qkv"]
                + [r.clone() for r in rings.values()]
            )
            if pending_mode:
                pending[slots] = accept.to(torch.int32)
                continue
            commit_kda_replayssm_spec(
                kw["recurrent_state"],
                rings["replayssm_rawv"],
                rings["replayssm_rawk"],
                rings["replayssm_g"],
                rings["replayssm_beta"],
                slots,
                accept,
                max_cache_len=ring_len,
                num_k_heads=num_heads,
            )
        if pending_mode:
            # Flush: the eager fold with the pending counts.
            commit_kda_replayssm_spec(
                kw["recurrent_state"],
                rings["replayssm_rawv"],
                rings["replayssm_rawk"],
                rings["replayssm_g"],
                rings["replayssm_beta"],
                slots,
                pending[slots].to(torch.int64),
                max_cache_len=ring_len,
                num_k_heads=num_heads,
            )
        records.append([kw["recurrent_state"].clone()])
        return records

    def test_pending_fold_matches_eager(self):
        # 2 x 4 items take the wide block, 16 x 24 the narrow one (two fold row maps).
        for num_requests, num_heads in ((2, 4), (16, 24)):
            with self.subTest(num_requests=num_requests, num_heads=num_heads):
                eager = self._steps(num_requests, num_heads, pending_mode=False)
                pending = self._steps(num_requests, num_heads, pending_mode=True)
                for step, (a, b) in enumerate(zip(eager, pending)):
                    for i, (x, y) in enumerate(zip(a, b)):
                        self.assertTrue(
                            torch.equal(x, y), f"step {step} tensor {i} differs"
                        )


if __name__ == "__main__":
    unittest.main()
