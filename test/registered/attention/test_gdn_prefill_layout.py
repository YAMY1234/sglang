import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.fla import l2norm as l2norm_mod
from sglang.kernels.ops.attention.fla.l2norm import (
    gdn_prefill_qkv_prepare_fwd,
    l2norm_fwd,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import (
    RMSNorm,
    rms_norm_gated,
)
from sglang.kernels.ops.attention.linear import fused_gdn_gating as gating_mod
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")


class TestGdnSplitConvGate(unittest.TestCase):
    def test_split_conv_takes_only_the_channel_last_width4_extend(self):
        backend = object.__new__(GDNAttnBackend)

        def layer(q_dim=256, width=4, activation="silu"):
            return SimpleNamespace(
                q_dim=q_dim,
                k_dim=256,
                v_dim=768,
                conv_weights=torch.zeros(q_dim + 1024, width),
                activation=activation,
            )

        self.assertTrue(backend._split_conv_ok(layer(), torch.zeros(5, 1280)))
        for bad_layer, rows in (
            (layer(width=3), 5),
            (layer(q_dim=128), 5),
            (layer(activation="relu"), 5),
            (layer(), 1),
        ):
            x = torch.zeros(rows, bad_layer.q_dim + 1024)
            self.assertFalse(backend._split_conv_ok(bad_layer, x))
        # Column-major input (stride(0) == 1) keeps the stock conv.
        self.assertFalse(backend._split_conv_ok(layer(), torch.zeros(1280, 5).T))


class TestGdnPrefillLaunchChoice(unittest.TestCase):
    def test_gating_2d_takes_only_extend_sized_batches(self):
        with mock.patch.object(gating_mod, "_GATING_2D", False):
            self.assertIsNone(gating_mod._gating_2d_cfg(4096, None))
            self.assertEqual(gating_mod._gating_2d_cfg(8, (32, 4)), (32, 4))
        with mock.patch.object(gating_mod, "_GATING_2D", True):
            self.assertIsNone(gating_mod._gating_2d_cfg(511, None))
            self.assertEqual(gating_mod._gating_2d_cfg(512, None), (16, 2))

    def test_l2norm_retile_takes_only_extend_sized_128_wide_inputs(self):
        with mock.patch.object(l2norm_mod, "_L2NORM_RETILE", False):
            self.assertEqual(l2norm_mod._l2norm_launch_cfg(65536, 128, None), (16, 8))
            self.assertEqual(l2norm_mod._l2norm_launch_cfg(8, 128, (32, 4)), (32, 4))
        with mock.patch.object(l2norm_mod, "_L2NORM_RETILE", True):
            self.assertEqual(l2norm_mod._l2norm_launch_cfg(16383, 128, None), (16, 8))
            self.assertEqual(l2norm_mod._l2norm_launch_cfg(16384, 256, None), (16, 8))
            self.assertEqual(l2norm_mod._l2norm_launch_cfg(16384, 128, None), (64, 8))


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGdnPrefillLayout(unittest.TestCase):
    TOKENS = 33
    NUM_QK_HEADS = 2
    NUM_V_HEADS = 4
    HEAD_DIM = 128

    def test_compact_prefill_conv_matches_rectangular_output_and_cache(self):
        torch.manual_seed(0)
        for seq_lens in ([65, 1, 0, 1, 1, 1, 1, 1], [8065] + [1] * 127):
            with self.subTest(batch_size=len(seq_lens)):
                dim, width = 768, 4
                batch_size = len(seq_lens)
                lengths = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
                starts = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)]).int()
                # Preserve the token-major projection view, including the unused Z.
                qkv = torch.randn(
                    sum(seq_lens), dim + 256, device="cuda", dtype=torch.bfloat16
                )[:, :dim]
                cache_indices = torch.randperm(batch_size + 2, device="cuda")[
                    :batch_size
                ]
                cache_indices = cache_indices.int()
                cache_indices[-1] = -1
                states = torch.randn(
                    batch_size + 2, dim, width - 1, device="cuda", dtype=qkv.dtype
                )
                expected_states = states.clone()
                layer = SimpleNamespace(
                    conv_weights=torch.randn(
                        dim, width, device="cuda", dtype=qkv.dtype
                    ),
                    bias=torch.randn(dim, device="cuda", dtype=qkv.dtype),
                    activation="silu",
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.MIXED,
                    tbo_parent_token_range=None,
                    extend_seq_lens_cpu=seq_lens,
                    extend_prefix_lens=torch.ones(batch_size, device="cuda"),
                )
                batch.extend_prefix_lens[0] = 0
                backend = object.__new__(GDNAttnBackend)
                backend.device = "cuda"
                backend.mis_metadata = None
                backend.forward_metadata = ForwardMetadata(
                    query_start_loc=starts, mamba_cache_indices=cache_indices
                )
                expected = backend._convolve_prefill(
                    layer, batch, qkv, expected_states, cache_indices
                )
                backend._init_prefill_conv_block_table(batch)
                self.assertIsNotNone(backend.forward_metadata.conv_block_table)
                actual = backend._convolve_prefill(
                    layer, batch, qkv, states, cache_indices
                )
                # Padded rows have no defined output; all cache slots must still match.
                valid_tokens = sum(seq_lens[:-1])
                torch.testing.assert_close(
                    actual[:valid_tokens], expected[:valid_tokens], rtol=0, atol=0
                )
                torch.testing.assert_close(states, expected_states, rtol=0, atol=0)

    def _projection_views(self, dtype):
        qkv_dim = (
            2 * self.NUM_QK_HEADS * self.HEAD_DIM + self.NUM_V_HEADS * self.HEAD_DIM
        )
        qkvz = torch.randn(
            self.TOKENS,
            qkv_dim + self.NUM_V_HEADS * self.HEAD_DIM,
            dtype=dtype,
            device="cuda",
        )
        ba = torch.randn(
            self.TOKENS,
            2 * self.NUM_V_HEADS,
            dtype=dtype,
            device="cuda",
        )
        return (qkvz, ba), qwen3_5_gdn_prefill_projection_views(
            qkvz,
            ba,
            self.NUM_QK_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
        )

    def test_projection_views_preserve_layout(self):
        (qkvz, ba), (mixed_qkv, z, b, a) = self._projection_views(torch.bfloat16)
        qkv_dim = mixed_qkv.shape[1]

        self.assertFalse(mixed_qkv.is_contiguous())
        self.assertFalse(z.is_contiguous())
        self.assertFalse(b.is_contiguous())
        self.assertFalse(a.is_contiguous())
        torch.testing.assert_close(mixed_qkv, qkvz[:, :qkv_dim], rtol=0, atol=0)
        torch.testing.assert_close(
            z.reshape(self.TOKENS, -1), qkvz[:, qkv_dim:], rtol=0, atol=0
        )
        torch.testing.assert_close(b, ba[:, : self.NUM_V_HEADS], rtol=0, atol=0)
        torch.testing.assert_close(a, ba[:, self.NUM_V_HEADS :], rtol=0, atol=0)

    def test_qkv_prepare_preserves_dtype_and_matches_materialized_path(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                _, (mixed_qkv, _, _, _) = self._projection_views(dtype)
                q_dim = self.NUM_QK_HEADS * self.HEAD_DIM
                v_dim = self.NUM_V_HEADS * self.HEAD_DIM
                q = mixed_qkv[:, :q_dim].view(
                    self.TOKENS, self.NUM_QK_HEADS, self.HEAD_DIM
                )
                k = mixed_qkv[:, q_dim : 2 * q_dim].view(
                    self.TOKENS, self.NUM_QK_HEADS, self.HEAD_DIM
                )
                v = mixed_qkv[:, 2 * q_dim : 2 * q_dim + v_dim].view(
                    self.TOKENS, self.NUM_V_HEADS, self.HEAD_DIM
                )

                q_out, k_out, v_out = gdn_prefill_qkv_prepare_fwd(q, k, v)

                self.assertEqual(q_out.dtype, dtype)
                self.assertEqual(k_out.dtype, dtype)
                self.assertEqual(v_out.dtype, dtype)
                torch.testing.assert_close(
                    q_out, l2norm_fwd(q.contiguous()), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    k_out, l2norm_fwd(k.contiguous()), rtol=0, atol=0
                )
                torch.testing.assert_close(v_out, v.contiguous(), rtol=0, atol=0)

    def test_fused_split_flashinfer_prepare_reuses_contiguous_value(self):
        _, (mixed_qkv, _, _, _) = self._projection_views(torch.bfloat16)
        q, k, v = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            self.NUM_QK_HEADS,
            self.NUM_QK_HEADS,
            self.NUM_V_HEADS,
            self.HEAD_DIM,
            self.HEAD_DIM,
            self.HEAD_DIM,
        )

        q_out, k_out, v_out = gdn_prefill_qkv_prepare_fwd(q[0], k[0], v[0])

        self.assertEqual(v_out.data_ptr(), v.data_ptr())
        torch.testing.assert_close(q_out, l2norm_fwd(q[0]), rtol=0, atol=0)
        torch.testing.assert_close(k_out, l2norm_fwd(k[0]), rtol=0, atol=0)

    def test_strided_gate_matches_contiguous_gate(self):
        for dtype in (torch.bfloat16, torch.float16):
            for norm_before_gate in (True, False):
                with self.subTest(dtype=dtype, norm_before_gate=norm_before_gate):
                    _, (_, z, _, _) = self._projection_views(dtype)
                    x = torch.randn(
                        self.TOKENS * self.NUM_V_HEADS,
                        self.HEAD_DIM,
                        dtype=dtype,
                        device="cuda",
                    )
                    weight = torch.randn(self.HEAD_DIM, dtype=dtype, device="cuda")
                    expected = rms_norm_gated(
                        x=x,
                        weight=weight,
                        bias=None,
                        z=z.contiguous().view_as(x),
                        norm_before_gate=norm_before_gate,
                        is_rms_norm=True,
                    )
                    actual = rms_norm_gated(
                        x=x,
                        weight=weight,
                        bias=None,
                        z=z,
                        norm_before_gate=norm_before_gate,
                        is_rms_norm=True,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                    norm = RMSNorm(
                        self.HEAD_DIM,
                        dtype=dtype,
                        device="cuda",
                        norm_before_gate=norm_before_gate,
                    )
                    norm.weight.data.copy_(weight)
                    native = norm.forward_native(x, z)
                    torch.testing.assert_close(native, actual, rtol=1e-2, atol=1e-2)

    def test_gating_2d_launch_matches_stock_launch(self):
        num_heads = 48
        a_log = torch.randn(num_heads, device="cuda")
        dt_bias = torch.randn(num_heads, device="cuda")
        for batch in (1, 600):
            with self.subTest(batch=batch):
                ba = torch.randn(
                    batch, 2 * num_heads, dtype=torch.bfloat16, device="cuda"
                )
                b, a = ba[:, :num_heads], ba[:, num_heads:]
                with mock.patch.object(gating_mod, "_GATING_2D", False):
                    g, beta = gating_mod.fused_gdn_gating(a_log, a, b, dt_bias)
                    g_2d, beta_2d = gating_mod.fused_gdn_gating(
                        a_log, a, b, dt_bias, launch_cfg=(16, 2)
                    )
                torch.testing.assert_close(g_2d, g, rtol=0, atol=0)
                torch.testing.assert_close(beta_2d, beta, rtol=0, atol=0)

    def test_l2norm_retile_matches_stock_launch(self):
        for rows in (33, 20000):
            with self.subTest(rows=rows):
                x = torch.randn(
                    rows, self.HEAD_DIM, dtype=torch.bfloat16, device="cuda"
                )
                torch.testing.assert_close(
                    l2norm_fwd(x, launch_cfg=(64, 8)),
                    l2norm_fwd(x, launch_cfg=(16, 8)),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
