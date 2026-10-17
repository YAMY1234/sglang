import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4AttnMetadata,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROWS, _HEADS, _DIM = 3, 64, 8
_SWAPAB = "sglang.kernels.ops.attention.dsv4.decode_attention_sm100.swapab_attention"


def _backend():
    """A decode step on the SWA-only cache: every input of the attention choice."""
    pool = mock.Mock(spec=DeepSeekV4TokenToKVPool)
    pool.request_window = None
    pool.get_swa_key_buffer_radix.return_value = torch.zeros(5, 2 * 16)
    pool.get_swa_key_page_size.return_value = 2
    pool.get_swa_key_bytes_per_token.return_value = 16
    meta = mock.Mock(spec=DSV4AttnMetadata)
    meta.swa_page_indices = torch.zeros(_ROWS, 64, dtype=torch.int32)
    meta.swa_topk_lengths = torch.ones(_ROWS, dtype=torch.int32)
    meta.get_flashmla_metadata.return_value = "flashmla metadata"
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.__dict__.update(
        mtp_enabled=False,
        token_to_kv_pool=pool,
        forward_metadata=SimpleNamespace(core_attn_metadata=meta, late_layer_tail=None),
        trtllm_attn=False,
        is_dsv41=True,
        is_dspark_draft=False,
        head_dim_v=_DIM,
        softmax_scale=1.0,
        _use_sparse_prefill=lambda *a, **k: False,
    )
    return backend


@contextlib.contextmanager
def _sm100_swapab():
    """Yields the swap-AB kernel mock; the platform takes swap-AB at these rows."""
    platform = SimpleNamespace(is_sm100=True, is_sm120=False)
    with (
        mock.patch.object(backend_module, "get_platform", return_value=platform),
        mock.patch.object(
            backend_module, "can_use_swapab_attention", return_value=True
        ),
        mock.patch(_SWAPAB, return_value="stock swap-AB") as swapab,
    ):
        yield swapab


def _forward(backend, **kwargs):
    decode = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_idle=lambda: False,
            is_decode=lambda: True,
            is_target_verify=lambda: False,
        )
    )
    q = torch.randn(_ROWS, _HEADS, _DIM)
    attn_sink = torch.zeros(_HEADS)
    out = backend.forward(
        q=q,
        k=q,
        v=q,
        layer=SimpleNamespace(layer_id=0, tp_q_head_num=16),
        forward_batch=decode,
        compress_ratio=0,
        save_kv_cache=False,
        attn_sink=attn_sink,
        **kwargs,
    )
    return out, q, attn_sink


class TestSparseDecodeSlot(unittest.TestCase):
    def test_set_sparse_decode_replaces_only_the_attention_choice(self):
        """A caller's sparse decode gets the resolved caches and indices instead of swap-AB."""
        backend = _backend()
        sparse_decode = mock.Mock(return_value="caller attention")
        with _sm100_swapab() as swapab:
            out, q, attn_sink = _forward(backend, sparse_decode=sparse_decode)
        self.assertEqual(out, "caller attention")
        swapab.assert_not_called()
        kwargs = sparse_decode.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["q"], q.unsqueeze(1)))
        self.assertEqual(tuple(kwargs["swa_k_cache"].shape), (5, 2, 1, 16))
        self.assertEqual(tuple(kwargs["swa_page_indices"].shape), (_ROWS, 1, 64))
        self.assertIs(kwargs["attn_sink"], attn_sink)
        self.assertIsNone(kwargs["extra_k_cache"])
        self.assertIsNone(kwargs["extra_indices"])
        self.assertEqual(kwargs["flashmla_metadata"], "flashmla metadata")

    def test_unset_sparse_decode_runs_swap_ab_with_the_same_arguments(self):
        """Without a caller's sparse decode the backend still picks swap-AB."""
        backend = _backend()
        with _sm100_swapab() as swapab:
            out, q, attn_sink = _forward(backend)
        self.assertEqual(out, "stock swap-AB")
        args = swapab.call_args.args
        self.assertTrue(torch.equal(args[0], q.unsqueeze(1)[..., :16, :]))
        self.assertEqual(tuple(args[2].shape), (_ROWS, 1, 64))
        self.assertIs(args[4], attn_sink)


if __name__ == "__main__":
    unittest.main()
