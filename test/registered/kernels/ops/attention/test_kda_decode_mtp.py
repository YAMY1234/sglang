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


def _kda_flow_reference(*, x, out_gate, raw_gate, beta, conv_w, conv_state, a_log,
                        dt_bias, state, norm_w, eps, lower_bound, tokens, keys=None):  # fmt: skip
    """FP64 KDA verify of n requests x H heads in the stock RecoverSSM semantics
    (BF16 conv outputs, sequential delta rule from the checkpoint). Returns the
    gated-RMSNorm output [R, D], corrections [n, H, q, 128], its own conv outputs k
    [n, H, q, 128] and the per-token states [q, n, H, 128, 128]."""
    f64 = torch.float64
    rows, dim3 = x.shape
    n, d, heads = rows // tokens, dim3 // 3, a_log.shape[0]
    seq = torch.cat([conv_state.to(f64), x.view(n, tokens, dim3).to(f64)], 1)
    w = conv_w.to(f64).transpose(1, 2).reshape(dim3, 4)  # [3 D, 4]
    conv = torch.stack(
        [(seq[:, t : t + 4, :] * w.t()[None]).sum(1) for t in range(tokens)], 1
    )
    act = (conv / (1 + torch.exp(-conv))).bfloat16().to(f64)  # [n, q, 3 D]
    q, k_calc, v = (
        act[..., i * d : (i + 1) * d].view(n, tokens, heads, 128) for i in range(3)
    )
    # ``keys``: the kernel's BF16 conv outputs (its FP32 SiLU rounds a few differently).
    k = k_calc if keys is None else keys.to(f64)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = raw_gate.to(f64).view(n, tokens, heads, 128)
    decay = torch.exp(
        lower_bound
        * torch.sigmoid(torch.exp(a_log.to(f64))[:, None] * (g + dt_bias.to(f64).view(heads, 128)))
    )
    b = torch.sigmoid(beta.to(f64).view(n, tokens, heads))
    s = state.to(f64).clone()  # [n, H, V, K]
    outs, us, states = [], [], []
    for t in range(tokens):
        s = s * decay[:, t, :, None, :]
        u = b[:, t, :, None] * (v[:, t] - (s @ kn[:, t, :, :, None]).squeeze(-1))
        s = s + u[..., :, None] * kn[:, t, :, None, :]
        outs.append((s @ qn[:, t, :, :, None]).squeeze(-1).bfloat16().to(f64))
        us.append(u)
        states.append(s.clone())
    o = torch.stack(outs, 1)  # [n, q, H, V]
    rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
    gate = torch.sigmoid(out_gate.to(f64).view(n, tokens, heads, 128))
    out = (o * rstd * norm_w.to(f64) * gate).reshape(rows, heads * 128)
    return out, torch.stack(us, 2), k_calc.transpose(1, 2), torch.stack(states)


def _rel(got, want):
    return float((got.double() - want.double()).norm() / want.double().norm())


class TestKdaFlowVerify(CustomTestCase):
    """SGLANG_OPT_KDA_FLOW: the persistent verify folds the previous step's accepted
    rows from its RecoverSSM records and verifies q = 5 rows; over steps with partial
    acceptance, a padding row and a slot changing owner it must match an FP64
    sequential reference (outputs within BF16, states and corrections within FP32
    reduction order) and the reader flush must commit the same state. A_log, dt_bias,
    f_b and the norm weight are nn.Parameters that require grad, as in the model:
    exporting such a tensor through DLPack raises, which crashed every decode
    scheduler at the first target verify."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.cuda.get_device_capability() != (10, 7):
            raise unittest.SkipTest("KdaFlow needs Rubin's 320 KiB shared memory")

    def _run(self, *, num_requests, fused_fb, steps=4, heads=24, tokens=5, layers=2):
        from types import SimpleNamespace

        from sglang.kernels.ops.attention.kda_flow import (
            fold_kda_flow_pending,
            kda_flow_records,
            kda_flow_verify,
            prepare_conv_weight,
        )
        from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
            commit_kda_replayssm_after_verify,
            reset_kda_pending,
        )
        from sglang.srt.environ import envs

        gen = torch.Generator(device="cuda").manual_seed(11)
        dev, bf16, f32 = "cuda", torch.bfloat16, torch.float32
        d, ring, rows_n = heads * 128, 16, num_requests + 1  # + one padding row
        rows, num_slots = rows_n * tokens, 2 * num_requests + 3

        def randn(*shape, scale=1.0, dtype=f32):
            return (torch.randn(*shape, device=dev, generator=gen) * scale).to(dtype)

        conv_w = [prepare_conv_weight(randn(3 * d, 4, scale=0.5)) for _ in range(layers)]
        # the model's layer parameters, which require grad: the kernel must not export them as they are
        param = torch.nn.Parameter
        a_log = [param(randn(heads, scale=0.5)) for _ in range(layers)]
        dt = [param(randn(d, scale=0.5)) for _ in range(layers)]
        f_b = [param(randn(d, 128, scale=128**-0.5, dtype=bf16)) for _ in range(layers)]
        norm_w = param(randn(128, scale=0.1) + 1)
        spec = SimpleNamespace(
            temporal=randn(layers, num_slots, heads, 128, 128, scale=0.1),
            replayssm_rawv=torch.zeros(layers, num_slots, heads, ring, 128, device=dev, dtype=bf16),
            replayssm_rawk=torch.zeros(layers, num_slots, heads, ring, 128, device=dev, dtype=bf16),
            replayssm_g=torch.zeros(layers, num_slots, heads, ring, 128, device=dev),
            replayssm_beta=torch.zeros(layers, num_slots, heads, ring, device=dev),
            conv=[randn(layers, num_slots, 3, 3 * d, dtype=bf16)],
            intermediate_conv_window=[
                torch.zeros(layers, rows_n, tokens, 3, 3 * d, device=dev, dtype=bf16)
            ],
        )  # fmt: skip
        corrections, key_gate = kda_flow_records(
            g_ring=spec.replayssm_g, rawk_ring=spec.replayssm_rawk, tokens=tokens
        )
        pending = torch.zeros(num_slots + 1, device=dev, dtype=torch.int32)
        slots = torch.randperm(num_slots - 1, device=dev, generator=gen)[:rows_n] + 1
        slots[-1] = -1
        free = sorted(set(range(1, num_slots)) - set(slots.tolist()))
        scratch = torch.arange(rows_n, device=dev, dtype=torch.int32)
        committed = spec.temporal.clone().double()  # the reference checkpoints
        report = []
        for step in range(steps):
            if step == 2:  # request 0 finishes; new owners of its slot and of a free one
                old = int(slots[0])
                slots[0], slots[1] = old, free[0]
                for s in (old, free[0]):
                    fresh = randn(layers, heads, 128, 128, scale=0.1)
                    spec.temporal[:, s] = fresh
                    committed[:, s] = fresh.double()
                reset_kda_pending(pending=pending, slots=slots[:2])
            fused = randn(rows, 4 * d, dtype=bf16)
            bfa = randn(rows, 152, dtype=bf16)
            accept = torch.randint(1, tokens + 1, (rows_n,), device=dev, generator=gen)
            live = slots[:num_requests].long()
            for layer in range(layers):
                f_a = bfa[:, :128]
                raw = (f_a.float() @ f_b[layer].float().t()).to(bf16)
                conv_before = spec.conv[0][layer][live].clone()
                out = kda_flow_verify(
                    x=fused[:, : 3 * d], conv_weight=conv_w[layer],
                    conv_state=spec.conv[0][layer], gate_in=f_a if fused_fb else raw,
                    beta=bfa[:, 128:152], a_log=a_log[layer], dt_bias=dt[layer],
                    slots=slots.to(torch.int32), state=spec.temporal[layer],
                    out_gate=fused[:, 3 * d :], norm_weight=norm_w,
                    f_b=f_b[layer] if fused_fb else None,
                    conv_windows=spec.intermediate_conv_window[0][layer],
                    scratch_rows=scratch, corrections=corrections[layer],
                    key_gate=key_gate[layer], pending=pending, lower_bound=-5.0, eps=1e-5,
                )  # fmt: skip
                torch.cuda.synchronize()
                # The kernel's own BF16 raw gates (F_B MMA: within one ulp of cuBLAS).
                rec_gate = key_gate[layer][live][..., 128:].transpose(1, 2)
                gate = rec_gate.reshape(num_requests * tokens, d)
                if fused_fb:
                    torch.testing.assert_close(
                        gate.float(), raw[: num_requests * tokens].float(), rtol=2**-7, atol=1e-6
                    )
                ref_out, ref_u, ref_k, ref_states = _kda_flow_reference(
                    x=fused[: num_requests * tokens, : 3 * d],
                    out_gate=fused[: num_requests * tokens, 3 * d :],
                    raw_gate=gate, beta=bfa[: num_requests * tokens, 128:152],
                    conv_w=conv_w[layer], conv_state=conv_before, a_log=a_log[layer],
                    dt_bias=dt[layer], state=committed[layer][live], norm_w=norm_w,
                    eps=1e-5, lower_bound=-5.0, tokens=tokens,
                    keys=key_gate[layer][live][..., :128].transpose(1, 2),
                )  # fmt: skip
                r = dict(step=step, layer=layer)
                r["out"] = _rel(out[: num_requests * tokens], ref_out)
                r["state"] = _rel(spec.temporal[layer][live], committed[layer][live])
                r["corr"] = _rel(corrections[layer][live], ref_u)
                # Conv outputs rounded to BF16 from FP32 (kernel) vs FP64: one ulp.
                torch.testing.assert_close(
                    key_gate[layer][live][..., :128].float(), ref_k.float(),
                    rtol=2**-7, atol=1e-6,
                )  # fmt: skip
                if layer == 0:
                    r["v2_vs_ref"], r["flow_vs_v2"] = self._against_v2(
                        out=out, ref_out=ref_out, fused=fused, raw=raw, bfa=bfa,
                        conv_w=conv_w[0], conv_state=spec.conv[0][0], a_log=a_log[0],
                        dt=dt[0], state=committed[0], slots=slots, scratch=scratch,
                        norm_w=norm_w, tokens=tokens, num_requests=num_requests,
                    )  # fmt: skip
                windows = spec.intermediate_conv_window[0][layer][: num_requests]
                full = torch.cat([conv_before, fused[: num_requests * tokens, : 3 * d].view(num_requests, tokens, 3 * d)], 1)  # fmt: skip
                for t in range(tokens):
                    self.assertTrue(torch.equal(windows[:, t], full[:, t + 1 : t + 4]))
                self.assertTrue(torch.equal(out[num_requests * tokens :], torch.zeros_like(out[num_requests * tokens :])))
                torch.testing.assert_close(out[: num_requests * tokens].float(), ref_out.float(), rtol=2e-2, atol=2e-2)  # fmt: skip
                # The reference's own BF16 roundings of v (FP64 vs the kernel's FP32
                # SiLU) differ in rare elements; our v2 kernel sits at the same level.
                self.assertLess(r["state"], 1e-4, r)
                self.assertLess(r["corr"], 1e-4, r)
                self.assertLess(r["out"], 1e-2, r)
                report.append(r)
                # The reference commit: the state after each request's accepted rows.
                index = (accept[:num_requests] - 1).long()
                committed[layer][live] = ref_states[index, torch.arange(num_requests, device=dev)]
            with envs.SGLANG_OPT_KDA_FLOW.override(True):
                commit_kda_replayssm_after_verify(
                    spec_state=spec, state_batch_indices=slots, accept_lens=accept,
                    last_correct_step_indices=accept - 1, src_indices_raw=scratch,
                    pending=pending,
                )  # fmt: skip
        # The reader flush (retract / copy) commits the last step's accepted rows.
        a_all, dt_all = torch.stack(a_log), torch.stack(dt)
        fold_kda_flow_pending(
            state=spec.temporal, corrections=corrections, key_gate=key_gate,
            a_log=a_all, dt_bias=dt_all, lower_bound=-5.0, slots=slots, pending=pending,
        )  # fmt: skip
        live = slots[:num_requests].long()
        flushed = _rel(spec.temporal[:, live], committed[:, live])
        self.assertLess(flushed, 1e-4)
        return report, flushed

    def _against_v2(self, *, out, ref_out, fused, raw, bfa, conv_w, conv_state, a_log,
                    dt, state, slots, scratch, norm_w, tokens, num_requests):  # fmt: skip
        """Our recurrent v2 verify from the same checkpoint and inputs (rings to
        scratch): its error against the FP64 reference and its distance to KdaFlow."""
        from sglang.kernels.ops.attention.kda_decode_mtp_v2 import (
            fused_kda_decode_mtp_dspark as verify_v2,
        )

        rows, d = fused.shape[0], conv_w.shape[-1]
        heads, num_slots = a_log.shape[0], state.shape[0]
        as4 = lambda t: t.reshape(1, rows, heads, 128).contiguous()  # noqa: E731
        w = conv_w.transpose(1, 2).reshape(3 * d, 4)
        cs = conv_state.split([d, d, d], dim=-1)
        windows = torch.zeros(scratch.numel(), tokens, 3, 3 * d, device="cuda", dtype=torch.bfloat16)
        ic = windows.transpose(-1, -2).split([d, d, d], dim=-2)
        ring = lambda dtype: torch.zeros(num_slots, heads, 16, 128, device="cuda", dtype=dtype)  # noqa: E731
        got = verify_v2(
            x_q=as4(fused[:, :d]), x_k=as4(fused[:, d : 2 * d]), x_v=as4(fused[:, 2 * d : 3 * d]),
            w_q=w[:d], w_k=w[d : 2 * d], w_v=w[2 * d :],
            cs_q=cs[0].transpose(-1, -2), cs_k=cs[1].transpose(-1, -2), cs_v=cs[2].transpose(-1, -2),
            g=as4(raw), beta=bfa[:, 128:152].reshape(1, rows, heads).contiguous(),
            A_log=a_log, dt_bias=dt, recurrent_state=state.float().clone(),
            intermediate_ssm=None, intermediate_state_indices=scratch,
            intermediate_conv_q=ic[0], intermediate_conv_k=ic[1], intermediate_conv_v=ic[2],
            ssm_state_indices=slots.to(torch.int32),
            cu_seqlens=torch.arange(0, rows + 1, tokens, device="cuda", dtype=torch.int32),
            lower_bound=-5.0, replayssm_rawv=ring(torch.bfloat16),
            replayssm_rawk=ring(torch.bfloat16), replayssm_g=ring(torch.float32),
            replayssm_beta=torch.zeros(num_slots, heads, 16, device="cuda"),
            onorm_gate=as4(fused[:, 3 * d :]), onorm_weight=norm_w, onorm_eps=1e-5,
        ).reshape(rows, heads * 128)  # fmt: skip
        live = num_requests * tokens
        return _rel(got[:live], ref_out), _rel(out[:live], got[:live])

    def test_matches_fp64_reference(self):
        for num_requests, fused_fb in ((33, False), (58, False), (58, True), (112, True)):
            with self.subTest(num_requests=num_requests, fused_fb=fused_fb):
                report, flushed = self._run(num_requests=num_requests, fused_fb=fused_fb)
                keys = ("out", "state", "corr", "v2_vs_ref", "flow_vs_v2")
                worst = {k: max(r.get(k, 0.0) for r in report) for k in keys}
                self.assertLess(worst["flow_vs_v2"], 1e-2, worst)
                print(f"KdaFlow n{num_requests} fused_fb={fused_fb}: worst rel {worst}, flush {flushed:.2e}")


if __name__ == "__main__":
    unittest.main()
