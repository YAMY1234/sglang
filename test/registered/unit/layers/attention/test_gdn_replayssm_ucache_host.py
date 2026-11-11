"""Host-side contract of the GDN u-cache verify (--enable-gdn-replayssm-ucache).

The kernels need SM100+ (test_gdn_replayssm_ucache.py); these checks cover what
runs on the host: ring allocation, the option gates and the backend routing.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear import gdn_replayssm_ucache as ucache
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.utils import gdn_ucache_track_rings_supported
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

H, HV, K, V = 2, 4, 128, 128


def _allocate(**kwargs):
    args = dict(
        num_layers=3,
        num_slots=5,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        max_num_draft_tokens=4,
        device="cpu",
    )
    args.update(kwargs)
    return ucache.allocate_gdn_ucache(**args)


class TestGdnUcacheHost(CustomTestCase):
    def test_slot_rings(self):
        buffers = _allocate()
        self.assertEqual(buffers.k_ring.shape, (3, 5, H, ucache.RING_SLOTS, K))
        self.assertEqual(buffers.u_ring.shape, (3, 5, HV, ucache.RING_SLOTS, V))
        self.assertEqual(buffers.g_ring.shape, (3, 5, HV, ucache.RING_SLOTS))
        self.assertEqual(
            (buffers.k_ring.dtype, buffers.u_ring.dtype, buffers.g_ring.dtype),
            (torch.bfloat16, torch.float16, torch.float32),
        )
        for cursor in (buffers.hist_len, buffers.cache_base):
            self.assertEqual(cursor.dtype, torch.int32)
            self.assertTrue(torch.equal(cursor, torch.zeros(5, dtype=torch.int32)))
        # The laziest fold threshold that still fits T appended records.
        self.assertEqual(buffers.flush_min, 16 - 4 + 1)
        ring_bytes = sum(
            t[:, 0].numel() * t.element_size()
            for t in (buffers.k_ring, buffers.u_ring, buffers.g_ring)
        )
        self.assertEqual(
            ucache.gdn_ucache_ring_bytes_per_slot(
                num_layers=3,
                num_k_heads=H,
                num_v_heads=HV,
                head_k_dim=K,
                head_v_dim=V,
            ),
            ring_bytes,
        )

    def test_request_rings(self):
        buffers = _allocate(num_slots=9, by_request_row=True)
        self.assertTrue(buffers.by_request_row)
        self.assertEqual(buffers.k_ring.shape[1], 9)
        self.assertEqual(buffers.hist_len.shape, (9,))
        self.assertFalse(_allocate().by_request_row)
        slots = torch.tensor([3, -1], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "needs its request rows"):
            ucache.gather_gdn_ucache_rows(slots=slots, buffers=buffers)
        with self.assertRaisesRegex(RuntimeError, "needs its request rows"):
            ucache.commit_gdn_ucache_cursors(
                buffers=buffers, slots=slots, num_accept_tokens=slots
            )

    def test_request_rings_follow_the_master(self):
        env = envs.SGLANG_OPT_GDN_UCACHE_REQUEST_RINGS
        with env.override("1"):
            for master in (False, True):
                with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master):
                    self.assertIs(ucache.gdn_ucache_request_rings(), master)

    def test_fp16_key_ring_needs_the_sm107_kernel(self):
        self.assertIs(ucache.gdn_ucache_key_ring_dtype("cpu"), torch.bfloat16)
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            envs.SGLANG_OPT_GDN_UCACHE_FP16_KEY_RING.override("1"),
        ):
            with patch("torch.cuda.get_device_capability", return_value=(10, 7)):
                self.assertIs(ucache.gdn_ucache_key_ring_dtype("cuda"), torch.float16)
                self.assertIs(_allocate().k_ring.dtype, torch.float16)
            with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                with self.assertRaisesRegex(ValueError, "SM107"):
                    ucache.gdn_ucache_key_ring_dtype("cuda")

    def _flush_options(self, capability, **env):
        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True)
            )
            for name, value in env.items():
                stack.enter_context(
                    getattr(envs, f"SGLANG_OPT_GDN_UCACHE_{name}").override(value)
                )
            stack.enter_context(
                patch("torch.cuda.get_device_capability", return_value=capability)
            )
            return ucache.gdn_ucache_flush_options("cuda")

    def test_flush_options_default_off(self):
        # Without options the capability is not read: SM100 / SM103 take the FlashInfer kernel.
        opts = self._flush_options((10, 0))
        self.assertEqual(opts, ucache.GdnUcacheFlushOptions())
        self.assertEqual(opts.kernel_kwargs(), {})

    def test_strided_qkv_needs_the_sm107_kernel(self):
        opts = self._flush_options((10, 7), STRIDED_QKV="1")
        self.assertTrue(opts.strided_qkv)
        self.assertEqual(opts.kernel_kwargs(), {"strided_inputs": True})
        with self.assertRaisesRegex(ValueError, "SM107"):
            self._flush_options((10, 0), STRIDED_QKV="1")

    def test_pdl_needs_the_sm107_kernel(self):
        opts = self._flush_options((10, 7), STRIDED_QKV="1", PDL="1")
        self.assertEqual(opts.kernel_kwargs(), {"strided_inputs": True, "pdl": True})
        with self.assertRaisesRegex(ValueError, "SM107"):
            self._flush_options((10, 3), PDL="1")

    def test_flush_first_needs_the_sm107_kernel(self):
        # The row order is a kernel argument, not a launch build: no kernel kwarg.
        opts = self._flush_options((10, 7), FLUSH_FIRST="1")
        self.assertTrue(opts.flush_first)
        self.assertEqual(opts.kernel_kwargs(), {})
        with self.assertRaisesRegex(ValueError, "SM107"):
            self._flush_options((10, 0), FLUSH_FIRST="1")

    def test_track_snapshots_need_slot_indexed_bf16_rings(self):
        """gdn_prefix_materialize reads the state and the rings at one slot id with one
        (bf16) ring dtype: snapshots engage only without request rings and the fp16 key
        ring, and their pool keeps a bf16 u ring."""
        self.assertIs(_allocate(track_snapshots=True).u_ring.dtype, torch.bfloat16)
        self.assertIs(_allocate().u_ring.dtype, torch.float16)
        with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
            self.assertTrue(gdn_ucache_track_rings_supported())
            for name in ("REQUEST_RINGS", "FP16_KEY_RING"):
                with getattr(envs, f"SGLANG_OPT_GDN_UCACHE_{name}").override("1"):
                    self.assertFalse(gdn_ucache_track_rings_supported(), name)

    def test_retraction_backup_is_refused_only_under_the_ucache(self):
        self.assertFalse(
            ucache.gdn_ucache_blocks_state_backup(SimpleNamespace(gdn_ucache=None))
        )
        self.assertTrue(
            ucache.gdn_ucache_blocks_state_backup(SimpleNamespace(gdn_ucache=object()))
        )

    def _verify_backend(self):
        backend = object.__new__(GDNAttnBackend)
        backend._gdn_ucache = object()
        backend._use_strided_target_verify_qkv = True
        return backend

    def test_target_verify_qkv_routing(self):
        """Contiguous q / k / v for the FlashInfer kernel; the packed conv-output
        views under SGLANG_OPT_GDN_UCACHE_STRIDED_QKV."""
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        for strided in (False, True):
            backend = self._verify_backend()
            backend._gdn_ucache_verifier = SimpleNamespace(
                options=ucache.GdnUcacheFlushOptions(strided_qkv=strided)
            )
            backend._init_target_verify_qkv_routing(batch)
            self.assertIs(backend._use_strided_target_verify_qkv, strided)

    def test_fused_conv_needs_the_strided_dependent_sm107_flush(self):
        backend = self._verify_backend()
        backend.is_draft_worker = False
        for capable, expected in ((False, False), (True, True)):
            backend._gdn_ucache_verifier = SimpleNamespace(fused_conv_capable=capable)
            with self.subTest(capable=capable):
                self.assertIs(backend._ucache_fused_conv_supported(), False)
                with (
                    envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                    envs.SGLANG_OPT_GDN_UCACHE_FUSED_CONV.override("1"),
                ):
                    self.assertIs(backend._ucache_fused_conv_supported(), expected)
                    backend.is_draft_worker = True
                    self.assertIs(backend._ucache_fused_conv_supported(), False)
                    backend.is_draft_worker = False

    def test_fused_conv_is_not_offered_when_off_or_on_a_tree(self):
        backend = self._verify_backend()
        for enabled, linear_chain in ((False, True), (True, False)):
            backend._ucache_fused_conv_enabled = enabled
            self.assertIsNone(
                backend._ucache_fused_conv(
                    layer=None,
                    mixed_qkv=None,
                    conv_states=None,
                    window=None,
                    window_indices=None,
                    batch_size=4,
                    draft_token_num=4,
                    linear_chain=linear_chain,
                )
            )

    def test_fused_norm_bounds(self):
        """The model's gated norm stays separate without a norm, when off, above T 4 and
        above SGLANG_GDN_UCACHE_FUSED_NORM_MAX_BATCH verify rows."""
        backend = self._verify_backend()
        backend._ucache_fused_norm_max_batch = 48
        norm = (torch.zeros(4, HV, V), torch.zeros(V), 1e-6)
        for enabled, norm_arg, tokens, T in (
            (True, None, 16, 4),
            (False, norm, 16, 4),
            (True, norm, 40, 8),
            (True, norm, 4 * 49, 4),
        ):
            backend._ucache_fused_norm_enabled = enabled
            with self.subTest(enabled=enabled, norm=norm_arg is not None, T=T):
                self.assertIsNone(
                    backend._ucache_fused_norm(
                        norm=norm_arg,
                        num_tokens=tokens,
                        num_draft_tokens=T,
                        num_v_heads=HV,
                    )
                )
        self.assertEqual(envs.SGLANG_GDN_UCACHE_FUSED_NORM_MAX_BATCH.get(), 48)

    def test_fused_norm_follows_the_stock_reduction_order(self):
        # The epilogue matches the stock gated-norm launch from 2 rows per program.
        for rows_per_block, expected in ((1, False), (2, True), (4, True)):
            with patch(
                "sglang.kernels.ops.attention.fla.layernorm_gated.calc_rows_per_block",
                return_value=rows_per_block,
            ):
                self.assertIs(
                    ucache.gdn_ucache_fused_norm_applies(64, "cuda"), expected
                )

    def test_plain_decode_is_refused(self):
        backend = self._verify_backend()
        qkv = torch.zeros(1, 3 * K)
        with self.assertRaisesRegex(RuntimeError, "plain GDN decode"):
            backend.forward_decode(None, None, qkv, None, None)


if __name__ == "__main__":
    unittest.main()
