"""DSV4.1 decode launch switches keep their outputs: the DSpark stacked wq_a | wkv projection,
the row-per-CTA RoPE launch and the PDL launch of the q_lora RMSNorm + MXFP8 kernel."""

import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4 import norm_rope
from sglang.kernels.ops.layernorm import rmsnorm_mxfp8 as rmsnorm_mxfp8_mod
from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4_dspark as dspark
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

Q_LORA, KV_DIM, HIDDEN, HEADS, ROWS = 8, 4, 16, 64, 6


def _int_tensor(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, shape, generator=g).float()


class _Linear:
    def __init__(self, weight):
        self.weight = weight
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x @ self.weight.t(), None


class _Stop(Exception):
    def __init__(self, q, kv):
        self.q, self.kv = q, kv


def _attention(stacked: bool):
    wq_a, wkv = _int_tensor(Q_LORA, HIDDEN, seed=1), _int_tensor(KV_DIM, HIDDEN, seed=2)
    attn = types.SimpleNamespace(
        rope_head_dim=2,
        alt_streams=None,
        _multi_stream_bs_limit=128,
        q_head_norm=False,
        n_local_heads=HEADS,
        head_dim=KV_DIM,
        q_lora_rank=Q_LORA,
        wq_a=_Linear(wq_a),
        wkv=_Linear(wkv),
        wq_b=_Linear(_int_tensor(HEADS * KV_DIM, Q_LORA, seed=3)),
        wqkv_a_stacked=_Linear(torch.cat([wq_a, wkv])) if stacked else None,
        q_norm=lambda t: t * 2,
        freqs_cis=None,
        attn=None,
        stored=[],
    )
    attn._compute_q = types.MethodType(dspark.DSparkAttention._compute_q, attn)
    attn.kv_proj_only = types.MethodType(dspark.DSparkAttention.kv_proj_only, attn)
    attn._store_block_kv = lambda **kw: attn.stored.append(kw["kv"].clone())
    attn._local_attn_sink = lambda: None
    return attn


def _project(attn, hidden):
    """Run DSparkAttention.forward up to the attention call; returns (q, kv, stored kv)."""
    backend = MagicMock()
    backend.forward.side_effect = lambda **kw: (_ for _ in ()).throw(
        _Stop(kw["q"], kw["k"])
    )
    batch = MagicMock()
    batch.forward_mode.is_idle.return_value = False
    with (
        patch.object(
            dspark,
            "_resolve_dspark_pool",
            return_value=types.SimpleNamespace(uniform_fp8=False),
        ),
        patch.object(dspark, "get_is_capture_mode", return_value=False),
        patch.object(dspark, "is_unified_kv_triton", return_value=False),
        patch.object(dspark, "fused_rope_inplace"),
        patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=backend,
        ),
    ):
        try:
            dspark.DSparkAttention.forward(
                attn, torch.zeros(ROWS, dtype=torch.long), hidden, batch
            )
        except _Stop as stop:
            return stop.q, stop.kv, attn.stored[0]
    raise AssertionError("attention was not reached")


class TestDSparkStackedProjection(unittest.TestCase):
    def test_stacked_projection_matches_the_two_gemms(self):
        hidden = _int_tensor(ROWS, HIDDEN, seed=4)
        ref = _attention(stacked=False)
        q_ref, kv_ref, stored_ref = _project(ref, hidden)
        fused = _attention(stacked=True)
        q, kv, stored = _project(fused, hidden)
        self.assertTrue(torch.equal(q, q_ref))
        self.assertTrue(torch.equal(kv, kv_ref))
        self.assertTrue(torch.equal(stored, stored_ref))
        self.assertEqual(
            (fused.wq_a.calls, fused.wkv.calls, fused.wqkv_a_stacked.calls), (0, 0, 1)
        )
        self.assertEqual((ref.wq_a.calls, ref.wkv.calls), (1, 1))

    def test_stacking_order_and_loaded_gate(self):
        def stage(seed):
            attn = types.SimpleNamespace(
                wq_a=types.SimpleNamespace(
                    weight=_int_tensor(Q_LORA, HIDDEN, seed=seed),
                    weight_scale_inv=_int_tensor(1, 1, seed=seed + 1),
                ),
                wkv=types.SimpleNamespace(
                    weight=_int_tensor(KV_DIM, HIDDEN, seed=seed + 2),
                    weight_scale_inv=_int_tensor(1, 1, seed=seed + 3),
                ),
                wqkv_a_stacked=types.SimpleNamespace(
                    weight=torch.zeros(Q_LORA + KV_DIM, HIDDEN),
                    weight_scale_inv=torch.zeros(2, 1),
                ),
            )
            return types.SimpleNamespace(self_attn=attn)

        stages = [stage(10), stage(20)]
        loaded = {f"stages.0.self_attn.{n}.weight" for n in ("wq_a", "wkv")}
        loaded.add("stages.1.self_attn.wq_a.weight")  # stage 1 lacks wkv: left alone
        dspark._stack_wqkv_a(stages=stages, loaded_params=loaded)
        a = stages[0].self_attn
        self.assertTrue(
            torch.equal(
                a.wqkv_a_stacked.weight, torch.cat([a.wq_a.weight, a.wkv.weight])
            )
        )
        self.assertTrue(
            torch.equal(
                a.wqkv_a_stacked.weight_scale_inv,
                torch.cat([a.wq_a.weight_scale_inv, a.wkv.weight_scale_inv]),
            )
        )
        self.assertFalse(stages[1].self_attn.wqkv_a_stacked.weight.any())


class TestRopeRowLaunch(unittest.TestCase):
    def _call(self, k):
        q = torch.zeros(3, 4, 64, dtype=torch.bfloat16)
        freqs = torch.polar(torch.ones(16, 32), torch.zeros(16, 32))
        old, rows = MagicMock(), MagicMock()
        with (
            patch.object(norm_rope, "_jit_fused_rope_module", return_value=old),
            patch.object(norm_rope, "_jit_fused_rope_rows_module", return_value=rows),
        ):
            norm_rope.fused_rope_inplace(q, k, freqs, torch.arange(3), inverse=True)
        return old, rows

    def test_switch_routes_q_only_calls(self):
        with envs.SGLANG_OPT_DSV4_ROPE_ROW_LAUNCH.override(True):
            old, rows = self._call(k=None)
            self.assertEqual((old.forward.call_count, rows.forward.call_count), (0, 1))
            self.assertIs(rows.forward.call_args.args[3], True)  # inverse
            old, rows = self._call(k=torch.zeros(3, 1, 64, dtype=torch.bfloat16))
            self.assertEqual((old.forward.call_count, rows.forward.call_count), (1, 0))
        with envs.SGLANG_OPT_DSV4_ROPE_ROW_LAUNCH.override(False):
            old, rows = self._call(k=None)
            self.assertEqual((old.forward.call_count, rows.forward.call_count), (1, 0))


class TestQNormPdlLaunch(unittest.TestCase):
    def _launch_kwargs(self):
        launches = []

        class _Kernel:
            src = rmsnorm_mxfp8_mod._rmsnorm_mxfp8_kernel.src

            def __getitem__(self, grid):
                return lambda *args, **kwargs: launches.append(kwargs)

        with (
            patch.object(rmsnorm_mxfp8_mod, "_rmsnorm_mxfp8_kernel", _Kernel()),
            patch(
                "sglang.kernels.jit.utils.arch.is_arch_support_pdl", return_value=True
            ),
        ):
            rmsnorm_mxfp8_mod.rmsnorm_mxfp8(
                torch.zeros(4, 1280, dtype=torch.bfloat16),
                torch.ones(1280, dtype=torch.bfloat16),
                1e-6,
            )
        return launches[0]

    def test_pdl_kwargs_only_when_switched_on(self):
        with envs.SGLANG_OPT_DSV4_Q_NORM_PDL.override(True):
            kw = self._launch_kwargs()
            self.assertIs(kw.get("USE_PDL"), True)
            self.assertIs(kw.get("launch_pdl"), True)
        with envs.SGLANG_OPT_DSV4_Q_NORM_PDL.override(False):
            kw = self._launch_kwargs()
            self.assertNotIn("USE_PDL", kw)
            self.assertNotIn("launch_pdl", kw)

    def test_no_producer_read_or_store_before_the_wait(self):
        src = rmsnorm_mxfp8_mod._rmsnorm_mxfp8_kernel.src
        live = src[
            src.index("if USE_PDL:") : src.index(
                "        else:\n            v = tl.load(X"
            )
        ]
        self.assertLess(live.index("tl.load(W"), live.index("gdc_wait()"))
        self.assertLess(live.index("gdc_wait()"), live.index("tl.load(X"))
        self.assertNotIn("tl.store", live[: live.index("gdc_wait()")])
        pad = src[src.rindex("    else:\n") :]
        self.assertLess(pad.index("gdc_wait()"), pad.index("tl.store"))


if __name__ == "__main__":
    unittest.main()
