"""SGLANG_QSA_PREFILL_HOIST: the packed gathers equal the per-sequence ones."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_packed_token_locations,
    build_prefill_compressed_locs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RATIO = 4
LENGTHS = [0, 3, 4, 9, 64, 131]


def _table(lengths, width, seed=0):
    gen = torch.Generator().manual_seed(seed)
    # Page-aligned raw slots: each group of RATIO tokens is contiguous.
    groups = torch.randperm(4096, generator=gen)[: len(lengths) * width // RATIO]
    groups = groups.reshape(len(lengths), width // RATIO)
    slots = groups[:, :, None] * RATIO + torch.arange(RATIO)
    return slots.reshape(len(lengths), width).to(torch.int32)


def _pool(translate):
    buffer = torch.randn(4096, 1, 8)
    pool = SimpleNamespace(
        get_qsa_compressed_k_buffer=lambda layer_id: buffer,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=8,
    )
    if translate:
        pool.physical_page_map = True
        pool.translate_locations = lambda layer_id, locs, compressed=False: (locs * 7 + layer_id) % 4096
    return pool


def _metadata(lengths, pool, hoisted):
    width = 136
    table = _table(lengths, width)
    sequence_lengths = torch.tensor(lengths, dtype=torch.int32)
    token_to_batch_idx = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32), torch.tensor([min(l, 5) for l in lengths])
    )
    return QSAIndexerMetadata(
        sequence_lengths=sequence_lengths,
        token_to_batch_idx=token_to_batch_idx,
        token_slot_table=table,
        out_cache_loc=torch.zeros(0, dtype=torch.int64),
        token_to_kv_pool=pool,
        compress_ratio=RATIO,
        block_topk=16,
        prefill_compressed_locs=(
            build_prefill_compressed_locs(table, sequence_lengths, lengths, RATIO) if hoisted else None
        ),
    )


def test_compressed_locs_match_per_sequence_slices():
    table = _table(LENGTHS, 136)
    expected = torch.cat(
        [table[i, : (n // RATIO) * RATIO : RATIO].long() // RATIO for i, n in enumerate(LENGTHS) if n >= RATIO]
    )
    got = build_prefill_compressed_locs(table, torch.tensor(LENGTHS), LENGTHS, RATIO)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("translate", [False, True])
def test_prefill_mqa_inputs_are_identical(translate):
    pool = _pool(translate)
    positions = torch.cat([torch.arange(max(n - 5, 0), n) for n in LENGTHS])
    ref = _metadata(LENGTHS, pool, hoisted=False).get_prefill_mqa_inputs(3, positions)
    got = _metadata(LENGTHS, pool, hoisted=True).get_prefill_mqa_inputs(3, positions)
    for a, b in zip(ref, got):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_packed_token_locations_match_per_request_rows():
    req_to_token = torch.randint(0, 1 << 20, (8, 160), dtype=torch.int32)
    req = torch.tensor([5, 0, 7, 2, 3, 1])
    lengths = torch.tensor(LENGTHS)
    expected = torch.cat([req_to_token[int(r), :n].long() for r, n in zip(req, LENGTHS)])
    got = build_packed_token_locations(req_to_token, req, lengths, LENGTHS)
    assert torch.equal(got, expected)


def test_empty_batches():
    table = _table([3, 2], 8)
    assert build_prefill_compressed_locs(table, torch.tensor([3, 2]), [3, 2], RATIO).numel() == 0
    req_to_token = torch.zeros(2, 4, dtype=torch.int32)
    assert build_packed_token_locations(req_to_token, torch.tensor([0, 1]), torch.tensor([0, 0]), [0, 0]).numel() == 0
