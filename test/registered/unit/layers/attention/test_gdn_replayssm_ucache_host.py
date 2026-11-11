"""Host-side contract of the GDN u-cache verify (--enable-gdn-replayssm-ucache).

The kernels need SM100+ (test_gdn_replayssm_ucache.py); these checks cover what
runs on the host: ring allocation, the option gates and the backend routing.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear import gdn_replayssm_ucache as ucache
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
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

    def test_target_verify_reads_contiguous_qkv(self):
        backend = self._verify_backend()
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        backend._init_target_verify_qkv_routing(batch)
        self.assertFalse(backend._use_strided_target_verify_qkv)

    def test_plain_decode_is_refused(self):
        backend = self._verify_backend()
        qkv = torch.zeros(1, 3 * K)
        with self.assertRaisesRegex(RuntimeError, "plain GDN decode"):
            backend.forward_decode(None, None, qkv, None, None)


if __name__ == "__main__":
    unittest.main()
