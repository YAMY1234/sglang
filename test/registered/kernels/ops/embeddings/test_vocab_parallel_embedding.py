import types
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.embeddings.vocab_parallel_embedding import (
    vocab_parallel_embedding,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
    pad_vocab_size,
)
from sglang.srt.speculative import perfll_rs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=13, suite="stage-b-test-1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)


def _reference_embedding(input_ids, weight, cfg):
    # The kernel's contract is bit-parity with the eager production path it
    # replaces, so use that exact path as the oracle.
    masked_input, input_mask = get_masked_input_and_mask(input_ids, **cfg)
    output = F.embedding(masked_input.long(), weight)
    output.masked_fill_(input_mask.unsqueeze(-1), 0)
    return output


def _run_case(input_ids, weight, cfg):
    expected = _reference_embedding(input_ids, weight, cfg)
    actual = vocab_parallel_embedding(input_ids, weight, **cfg)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hidden_dim", [7, 128, 6144])
def test_vocab_parallel_embedding_no_added_vocab(dtype, input_dtype, hidden_dim):
    cfg = dict(
        org_vocab_start_index=16,
        org_vocab_end_index=32,
        num_org_vocab_padding=0,
        added_vocab_start_index=64,
        added_vocab_end_index=64,
    )
    weight = torch.randn((16, hidden_dim), dtype=dtype, device="cuda")
    # Include negative and garbage ids: the mask is the only defense against
    # them when the OOB probe is disabled, so they must produce zero rows.
    token_ids = [-100, -1, 0, 16, 17, 31, 32, 63, 2**30]
    if input_dtype == torch.int64:
        token_ids += [torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max]
    input_ids = torch.tensor(token_ids, dtype=input_dtype, device="cuda")
    _run_case(input_ids, weight, cfg)


def test_vocab_parallel_embedding_added_vocab_with_padding():
    cfg = dict(
        org_vocab_start_index=10,
        org_vocab_end_index=18,
        num_org_vocab_padding=4,
        added_vocab_start_index=100,
        added_vocab_end_index=103,
    )
    weight = torch.randn((16, 257), dtype=torch.bfloat16, device="cuda")
    input_ids = torch.tensor(
        [[9, 10, 17], [18, 100, 102]], dtype=torch.int64, device="cuda"
    )
    _run_case(input_ids, weight, cfg)


def test_vocab_parallel_embedding_strided_weight():
    cfg = dict(
        org_vocab_start_index=10,
        org_vocab_end_index=18,
        num_org_vocab_padding=4,
        added_vocab_start_index=100,
        added_vocab_end_index=103,
    )
    # Column slice of a wider buffer: stride(0) != hidden_dim, stride(1) == 1.
    weight = torch.randn((16, 300), dtype=torch.bfloat16, device="cuda")[:, :257]
    assert weight.stride(0) == 300 and weight.stride(1) == 1
    input_ids = torch.tensor(
        [[9, 10, 17], [18, 100, 102]], dtype=torch.int64, device="cuda"
    )
    _run_case(input_ids, weight, cfg)


def test_vocab_parallel_embedding_empty_input():
    cfg = dict(
        org_vocab_start_index=0,
        org_vocab_end_index=8,
        num_org_vocab_padding=0,
        added_vocab_start_index=8,
        added_vocab_end_index=8,
    )
    weight = torch.randn((8, 64), dtype=torch.bfloat16, device="cuda")
    input_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    _run_case(input_ids, weight, cfg)


def _stub_layer(**overrides):
    # The gate reads only tp_size, quant_method, and weight, so a stub avoids
    # needing distributed init for tp_size > 1.
    layer = types.SimpleNamespace(
        tp_size=2,
        quant_method=UnquantizedEmbeddingMethod(),
        weight=torch.empty((16, 32), dtype=torch.bfloat16, device="cuda"),
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def test_use_triton_embedding_gate():
    # The gate's failure direction is silent (the eager fallback is
    # numerically correct), so pin eligibility and the exclusions that define
    # the fused kernel's scope.
    gate = VocabParallelEmbedding._use_triton_embedding
    input_ids = torch.zeros((4,), dtype=torch.int64, device="cuda")
    assert gate(_stub_layer(), input_ids)
    assert not gate(_stub_layer(tp_size=1), input_ids)
    assert not gate(_stub_layer(quant_method=object()), input_ids)
    unsupported_weights = (
        torch.empty((16, 32), dtype=torch.bfloat16, device="cpu"),
        torch.empty((16,), dtype=torch.bfloat16, device="cuda"),
        torch.empty((16, 64), dtype=torch.bfloat16, device="cuda")[:, ::2],
    )
    for weight in unsupported_weights:
        assert not gate(_stub_layer(weight=weight), input_ids)


@pytest.mark.parametrize("use_attn_tp_group", [False, True])
def test_empty_embedding_skips_all_reduce(use_attn_tp_group):
    input_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    output = torch.empty((0, 32), dtype=torch.bfloat16, device="cuda")
    layer = types.SimpleNamespace(
        tp_size=2,
        use_attn_tp_group=use_attn_tp_group,
        num_embeddings=128,
        full_table=None,
        _embed_local_shard=lambda _: output,
    )

    with (
        patch(
            "sglang.srt.layers.vocab_parallel_embedding.get_attn_tp_context",
            return_value=types.SimpleNamespace(input_scattered=False),
        ),
        patch(
            "sglang.srt.layers.vocab_parallel_embedding."
            "tensor_model_parallel_all_reduce"
        ) as tensor_parallel_all_reduce,
        patch(
            "sglang.srt.layers.vocab_parallel_embedding.attn_tp_all_reduce"
        ) as attn_tp_all_reduce,
    ):
        actual = VocabParallelEmbedding.forward(layer, input_ids)

    assert actual is output
    tensor_parallel_all_reduce.assert_not_called()
    attn_tp_all_reduce.assert_not_called()


class _GatherGroup:
    """The TP group of ``attach_full_table``: one call per rank, the last one gathers."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.rows = []

    def all_gather_into_tensor(self, output, input_):
        self.rows.append(input_.clone())
        if len(self.rows) == self.world_size:
            output.copy_(torch.cat(self.rows))


def _full_table_layers(table, tp_size):
    """Each rank's stub after ``attach_full_table`` over the shards of ``table``."""
    group = _GatherGroup(tp_size)
    per_rank = table.shape[0] // tp_size
    layers = []
    for rank in range(tp_size):
        layer = types.SimpleNamespace(
            weight=torch.nn.Parameter(
                table[rank * per_rank : (rank + 1) * per_rank].clone(),
                requires_grad=False,
            ),
            tp_size=tp_size,
            tp_group=group,
            full_table=None,
            output_dtype=None,
        )
        VocabParallelEmbedding.attach_full_table(layer)
        layers.append(layer)
    return layers


def test_full_table_rows_equal_the_all_reduced_shard_lookups():
    # SGLANG_PERFLL_RS_V3 drops the all-reduce, so its rows must be what the reduce of
    # every rank's shard lookup gives, bit for bit, including a -0.0 row turning +0.0.
    tp_size, vocab, hidden = 4, 1000, 96
    padded = pad_vocab_size(vocab, 64)
    table = torch.randn((padded, hidden), dtype=torch.bfloat16, device="cuda")
    table[vocab:] = 0
    table[3, :7] = -0.0
    table[257] = -0.0
    input_ids = torch.tensor(
        [0, 3, 255, 256, 257, 999, 1000, padded - 1, padded, -1],
        dtype=torch.int64,
        device="cuda",
    )
    per_rank = padded // tp_size
    partials = []
    for rank in range(tp_size):
        idx = VocabParallelEmbedding._get_indices(
            padded, padded, vocab, vocab, rank, tp_size
        )
        partials.append(
            vocab_parallel_embedding(
                input_ids,
                table[rank * per_rank : (rank + 1) * per_rank].contiguous(),
                idx.org_vocab_start_index,
                idx.org_vocab_end_index,
                idx.num_org_vocab_padding,
                idx.added_vocab_start_index,
                idx.added_vocab_end_index,
            )
        )
    reduced_fp32 = torch.stack(partials).float().sum(0).to(torch.bfloat16)
    reduced_bf16 = partials[0]
    for partial in partials[1:]:
        reduced_bf16 = reduced_bf16 + partial

    layers = _full_table_layers(table, tp_size)
    actual = VocabParallelEmbedding._embed_full_table(layers[-1], input_ids)
    assert layers[-1].full_table.shape == (padded, hidden)
    for expected in (reduced_fp32, reduced_bf16):
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_full_table_forward_skips_the_all_reduce_unless_captured_off():
    table = torch.randn((128, 32), dtype=torch.bfloat16, device="cuda")
    layer = _full_table_layers(table, 2)[-1]
    shard_rows = torch.zeros((3, 32), dtype=torch.bfloat16, device="cuda")
    layer.use_attn_tp_group = False
    layer.num_embeddings = 128
    layer.quant_method = UnquantizedEmbeddingMethod()
    layer._embed_local_shard = lambda _: shard_rows
    layer._use_triton_embedding = types.MethodType(
        VocabParallelEmbedding._use_triton_embedding, layer
    )
    layer._embed_full_table = types.MethodType(
        VocabParallelEmbedding._embed_full_table, layer
    )
    input_ids = torch.tensor([5, 64, 127], dtype=torch.int64, device="cuda")

    with (
        patch(
            "sglang.srt.layers.vocab_parallel_embedding.get_attn_tp_context",
            return_value=types.SimpleNamespace(input_scattered=False),
        ),
        patch(
            "sglang.srt.layers.vocab_parallel_embedding."
            "tensor_model_parallel_all_reduce",
            side_effect=lambda x: x,
        ) as all_reduce,
    ):
        full = VocabParallelEmbedding.forward(layer, input_ids)
        all_reduce.assert_not_called()
        with perfll_rs.capture_variant(frozenset({"v3"})):
            shard = VocabParallelEmbedding.forward(layer, input_ids)
        all_reduce.assert_called_once()

    assert torch.equal(full, table[input_ids])
    assert shard is shard_rows


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
