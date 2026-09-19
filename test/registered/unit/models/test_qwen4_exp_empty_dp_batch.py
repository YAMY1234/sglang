import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpAttentionDecoderLayer,
    Qwen4ExpLinearDecoderLayer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_layer():
    hidden_states = torch.empty(0, 32)
    residual = torch.empty(0, 32)
    layer = SimpleNamespace(
        _prepare_qwen4_exp_attn=MagicMock(
            return_value=(hidden_states, residual)
        ),
        linear_attn=MagicMock(),
        self_attention=MagicMock(),
        _prepare_qwen4_exp_mlp=MagicMock(
            return_value=(hidden_states, residual)
        ),
        _run_qwen4_exp_mlp=MagicMock(return_value=hidden_states),
        _postprocess_qwen4_exp_layer=MagicMock(return_value=(hidden_states, None)),
    )
    return layer, hidden_states, residual


class TestQwen4ExpEmptyDPBatch(CustomTestCase):
    def setUp(self):
        self.forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False)
        )

    def test_linear_attention_skips_empty_non_idle_batch(self):
        layer, hidden_states, residual = make_layer()

        Qwen4ExpLinearDecoderLayer.forward(
            layer,
            hidden_states,
            residual,
            forward_batch=self.forward_batch,
            ple_batch=None,
        )

        layer.linear_attn.assert_not_called()
        layer._run_qwen4_exp_mlp.assert_called_once()

    def test_full_attention_skips_empty_non_idle_batch(self):
        layer, hidden_states, residual = make_layer()

        Qwen4ExpAttentionDecoderLayer.forward(
            layer,
            positions=None,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=self.forward_batch,
            ple_batch=None,
        )

        layer.self_attention.assert_not_called()
        layer._run_qwen4_exp_mlp.assert_called_once()

    def test_mtp_qsa_lookup_uses_metadata_request_rows(self):
        backend = object.__new__(QwenSparseAttnBackend)
        backend._mtp_shared_sparse_indices = MagicMock()
        metadata = SimpleNamespace(
            decode_logical_positions=torch.arange(7),
            req_pool_indices=torch.arange(10, 17),
        )
        backend.get_indexer_metadata = MagicMock(return_value=metadata)
        forward_batch = SimpleNamespace(req_pool_indices=torch.arange(8))

        QwenSparseAttnBackend.lookup_mtp_sparse_indices(
            backend, forward_batch, layer_id=4
        )

        args = backend._mtp_shared_sparse_indices.lookup.call_args.args
        torch.testing.assert_close(args[0], metadata.req_pool_indices)
        torch.testing.assert_close(args[1], metadata.decode_logical_positions)
        self.assertEqual(args[2], 4)

    def test_full_attention_skips_target_verify_batch_without_positions(self):
        layer, _, _ = make_layer()
        hidden_states = torch.empty(1, 32)
        residual = torch.empty(1, 32)
        layer._prepare_qwen4_exp_attn.return_value = (hidden_states, residual)
        layer._prepare_qwen4_exp_mlp.return_value = (hidden_states, residual)
        layer._run_qwen4_exp_mlp.return_value = hidden_states

        Qwen4ExpAttentionDecoderLayer.forward(
            layer,
            positions=None,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=self.forward_batch,
            ple_batch=None,
        )

        layer.self_attention.assert_not_called()
        layer._run_qwen4_exp_mlp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
