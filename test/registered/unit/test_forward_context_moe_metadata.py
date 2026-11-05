"""Framework-owned batch/phase publication; no provider or CUDA execution."""

import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
    has_forward_context,
    set_forward_context,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestForwardContextMoeMetadata(unittest.TestCase):
    def setUp(self):
        self.previous = set_forward_context(None)

    def tearDown(self):
        set_forward_context(self.previous)

    def batch(self, rows, mode):
        return SimpleNamespace(
            batch_size=rows,
            forward_mode=mode,
            moe_num_token_non_padded=lambda: None,
        )

    def test_absent_metadata_stays_unknown(self):
        with forward_context(ForwardContext(attn_backend=None)):
            ctx = get_forward_context()
            self.assertIsNone(ctx.moe_batch_size)
            self.assertIsNone(ctx.moe_forward_mode)
        self.assertFalse(has_forward_context())

    def test_actual_batch_and_phase_are_published_without_inference(self):
        for batch_size in (0, 1, 3, 4):
            for mode in (
                ForwardMode.EXTEND,
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
            ):
                with self.subTest(batch_size=batch_size, mode=mode):
                    batch = self.batch(batch_size, mode)
                    with forward_context(
                        ForwardContext(attn_backend=None), forward_batch=batch
                    ):
                        ctx = get_forward_context()
                        self.assertEqual(ctx.moe_batch_size, batch_size)
                        self.assertIs(ctx.moe_forward_mode, mode)
                        self.assertIs(
                            ctx.moe_num_token_non_padded, batch.moe_num_token_non_padded
                        )

    def test_nested_override_inherits_original_batch_identity(self):
        batch = self.batch(3, ForwardMode.TARGET_VERIFY)
        with forward_context(ForwardContext(attn_backend=None), forward_batch=batch):
            outer = get_forward_context()
            with forward_context(ForwardContext(attn_backend="child")):
                inner = get_forward_context()
                self.assertEqual(inner.attn_backend, "child")
                self.assertEqual(inner.moe_batch_size, 3)
                self.assertIs(inner.moe_forward_mode, ForwardMode.TARGET_VERIFY)
                self.assertIs(
                    inner.moe_num_token_non_padded, batch.moe_num_token_non_padded
                )
            self.assertIs(get_forward_context(), outer)

    def test_explicit_nested_phase_and_zero_batch_are_preserved(self):
        with forward_context(
            ForwardContext(attn_backend=None),
            forward_batch=self.batch(4, ForwardMode.TARGET_VERIFY),
        ):
            with forward_context(
                ForwardContext(
                    attn_backend=None,
                    moe_batch_size=0,
                    moe_forward_mode=ForwardMode.IDLE,
                )
            ):
                ctx = get_forward_context()
                self.assertEqual(ctx.moe_batch_size, 0)
                self.assertIs(ctx.moe_forward_mode, ForwardMode.IDLE)

    def test_tbo_child_uses_its_own_actual_batch(self):
        with forward_context(
            ForwardContext(attn_backend=None),
            forward_batch=self.batch(4, ForwardMode.TARGET_VERIFY),
        ):
            child = self.batch(1, ForwardMode.DRAFT_EXTEND_V2)
            with forward_context(
                ForwardContext(attn_backend=None), forward_batch=child
            ):
                ctx = get_forward_context()
                self.assertEqual(ctx.moe_batch_size, 1)
                self.assertIs(ctx.moe_forward_mode, ForwardMode.DRAFT_EXTEND_V2)
                self.assertIs(
                    ctx.moe_num_token_non_padded, child.moe_num_token_non_padded
                )

    def test_existing_getter_override_semantics_are_unchanged(self):
        getter = lambda: None
        with forward_context(
            ForwardContext(attn_backend=None),
            forward_batch=self.batch(3, ForwardMode.TARGET_VERIFY),
        ):
            with forward_context(
                ForwardContext(attn_backend=None, moe_num_token_non_padded=getter)
            ):
                ctx = get_forward_context()
                self.assertIs(ctx.moe_num_token_non_padded, getter)
                self.assertIsNone(ctx.moe_batch_size)
                self.assertIsNone(ctx.moe_forward_mode)

    def test_exception_restores_parent_and_top_level_context(self):
        with forward_context(
            ForwardContext(attn_backend=None),
            forward_batch=self.batch(3, ForwardMode.TARGET_VERIFY),
        ):
            outer = get_forward_context()
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with forward_context(
                    ForwardContext(attn_backend=None),
                    forward_batch=self.batch(4, ForwardMode.EXTEND),
                ):
                    raise RuntimeError("failure")
            self.assertIs(get_forward_context(), outer)
        self.assertFalse(has_forward_context())

    def test_metadata_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            ForwardContext(attn_backend=None).moe_batch_size = 3


if __name__ == "__main__":
    unittest.main()
