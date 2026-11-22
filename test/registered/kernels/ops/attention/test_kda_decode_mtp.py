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

    def _run(self, num_requests, num_spec, *, onorm, block_threads=None):
        kwargs = _make_inputs(num_requests, num_spec)
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
        for num_spec in (0, 2):
            with self.subTest(num_spec=num_spec):
                raw, raw_states, gate, weight = self._run(4, num_spec, onorm=False)
                fused, fused_states, _, _ = self._run(4, num_spec, onorm=True)
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


class TestKdaVerifyPendingFold(CustomTestCase):
    """SGLANG_OPT_KDA_PENDING_FOLD: commits that only record the accepted rows, folded
    inside the next v2 verify, must match verify + eager commit fold bit for bit on
    outputs, rings, conv scratch and states, across finishing and joining requests."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def _run(self, *, num_requests, num_heads, num_layers, pending_mode, steps=4):
        from types import SimpleNamespace

        from sglang.kernels.ops.attention.kda_decode_mtp_v2 import (
            fused_kda_decode_mtp_dspark as verify_v2,
        )
        from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
            commit_kda_replayssm_after_verify,
            fold_kda_pending,
            reset_kda_pending,
        )

        num_spec, ring_len, dim, rows = 4, 16, 128, num_requests + 1
        kw = _make_inputs(rows, num_spec, num_heads=num_heads)
        gen = torch.Generator(device="cuda").manual_seed(7)
        num_slots = 2 * num_requests + 3
        for c in "qkv":
            kw[f"cs_{c}"] = torch.randn(
                num_slots, num_heads * dim, 3, device="cuda", generator=gen
            ).bfloat16()
        spec_state = SimpleNamespace(
            temporal=torch.randn(
                num_layers, num_slots, num_heads, dim, dim, device="cuda", generator=gen
            )
            * 0.05,
            replayssm_rawv=torch.zeros(
                num_layers, num_slots, num_heads, ring_len, dim, device="cuda"
            ).bfloat16(),
            replayssm_rawk=torch.zeros(
                num_layers, num_slots, num_heads, ring_len, dim, device="cuda"
            ).bfloat16(),
            replayssm_g=torch.zeros(
                num_layers, num_slots, num_heads, ring_len, dim, device="cuda"
            ),
            replayssm_beta=torch.zeros(
                num_layers, num_slots, num_heads, ring_len, device="cuda"
            ),
            conv=[],
            intermediate_conv_window=[],
        )
        pending = (
            torch.zeros(num_slots + 1, device="cuda", dtype=torch.int32)
            if pending_mode
            else None
        )
        # Row num_requests is CUDA-graph padding (slot -1).
        slots = torch.randperm(num_slots, device="cuda", generator=gen)[:rows]
        slots[-1] = -1
        free = sorted(set(range(num_slots)) - set(slots.tolist()))
        ring_names = (
            "replayssm_rawv",
            "replayssm_rawk",
            "replayssm_g",
            "replayssm_beta",
        )
        onorm = dict(
            onorm_gate=torch.randn_like(kw["x_v"]),
            onorm_weight=torch.rand(dim, device="cuda") + 0.5,
            onorm_eps=1e-5,
        )
        records = []
        for step in range(steps):
            if step == 2:
                # Request 0 finishes; a new request takes its slot, another a free
                # slot; both arrive with a transferred state (PD) and no history.
                old = int(slots[0])
                slots[0], slots[1] = old, free[0]
                for s in (old, free[0]):
                    spec_state.temporal[:, s].normal_(generator=gen).mul_(0.05)
                if pending_mode:
                    reset_kda_pending(pending=pending, slots=slots[:2])
            for name in ("x_q", "x_k", "x_v", "g", "beta"):
                kw[name] = torch.randn(kw[name].shape, device="cuda", generator=gen).to(
                    kw[name].dtype
                )
            kw["ssm_state_indices"] = slots.to(torch.int32)
            extra = {"replayssm_pending": pending} if pending_mode else {}
            outs = []
            for layer in range(num_layers):
                outs.append(
                    verify_v2(
                        recurrent_state=spec_state.temporal[layer],
                        intermediate_ssm=None,
                        **{n: getattr(spec_state, n)[layer] for n in ring_names},
                        **onorm,
                        **extra,
                        **{k: v for k, v in kw.items() if k != "recurrent_state"},
                    )
                )
            live = slots[:num_requests].to(torch.int64)
            records.append(
                outs
                + [spec_state.temporal[:, live].clone()]
                + [getattr(spec_state, n)[:, live].clone() for n in ring_names]
                + [kw[f"intermediate_conv_{c}"].clone() for c in "qkv"]
            )
            accept = torch.randint(
                1, num_spec + 2, (rows,), device="cuda", generator=gen
            )
            commit_kda_replayssm_after_verify(
                spec_state=spec_state,
                state_batch_indices=slots,
                accept_lens=accept,
                last_correct_step_indices=accept - 1,
                pending=pending,
            )
        if pending_mode:
            fold_kda_pending(spec_state=spec_state, pending=pending, slots=slots)
            self.assertEqual(int(pending[slots[:num_requests]].abs().sum()), 0)
        # A finished request's slot keeps its pending rows (its state is dropped).
        records.append([spec_state.temporal[:, slots[:num_requests]].clone()])
        return records

    def test_pending_fold_matches_eager(self):
        from sglang.srt.environ import envs

        # 3 x 4 items take the wide block, 58 / 77 x 24 the narrow one.
        cases = (
            (3, 4, 2, False),
            (3, 4, 2, True),
            (58, 24, 3, True),
            (77, 24, 3, True),
        )
        for num_requests, num_heads, num_layers, fast_fold in cases:
            with (
                self.subTest(
                    num_requests=num_requests, num_heads=num_heads, fast_fold=fast_fold
                ),
                envs.SGLANG_OPT_KDA_REPLAYSSM_FAST_FOLD.override(fast_fold),
            ):
                cfg = dict(
                    num_requests=num_requests,
                    num_heads=num_heads,
                    num_layers=num_layers,
                )
                eager = self._run(pending_mode=False, **cfg)
                pending = self._run(pending_mode=True, **cfg)
                for step, (a, b) in enumerate(zip(eager, pending)):
                    for i, (x, y) in enumerate(zip(a, b)):
                        self.assertTrue(
                            torch.equal(x, y), f"step {step} tensor {i} differs"
                        )


if __name__ == "__main__":
    unittest.main()
