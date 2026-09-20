import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    _pp_exchange_outputs_before_forward,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _batch(mode: ForwardMode, is_extend_in_batch: bool = False):
    return SimpleNamespace(forward_mode=mode, is_extend_in_batch=is_extend_in_batch)


class TestPPExchangeOrder(unittest.TestCase):
    """Under PP+spec a verify round must receive the relayed outputs before it
    launches, while an extend microbatch must launch first: exchanging first on
    extend serializes the ring and caps every stage at (pp_size-1)/pp_size."""

    def _decide(self, batch, **overrides):
        kwargs = dict(spec_relay=True, is_last_rank=False, async_batch_depth=0)
        kwargs.update(overrides)
        return _pp_exchange_outputs_before_forward(cur_batch=batch, **kwargs)

    def test_verify_round_exchanges_first(self):
        self.assertTrue(self._decide(_batch(ForwardMode.DECODE)))

    def test_extend_launches_first(self):
        self.assertFalse(self._decide(_batch(ForwardMode.EXTEND)))
        self.assertFalse(self._decide(_batch(ForwardMode.MIXED)))
        self.assertFalse(
            self._decide(_batch(ForwardMode.DECODE, is_extend_in_batch=True))
        )

    def test_last_rank_non_spec_and_idle_launch_first(self):
        self.assertFalse(self._decide(_batch(ForwardMode.DECODE), is_last_rank=True))
        self.assertFalse(self._decide(_batch(ForwardMode.DECODE), spec_relay=False))
        self.assertFalse(self._decide(None))

    def test_async_batch_depth_always_exchanges_first(self):
        self.assertTrue(self._decide(_batch(ForwardMode.EXTEND), async_batch_depth=1))
        self.assertTrue(
            self._decide(None, spec_relay=False, is_last_rank=True, async_batch_depth=1)
        )


if __name__ == "__main__":
    unittest.main()
