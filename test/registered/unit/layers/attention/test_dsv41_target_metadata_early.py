"""SGLANG_OPT_DSV41_TARGET_METADATA_EARLY on the DSv4 backend: only the unfiltered
variant's target-verify prep builds the full metadata (into its own bucket entry, the
filtered variant keeping its in-graph upgrade), and the captured metadata keeps every
tensor the verify graph reads by address."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    _GraphBucket,
)
from sglang.srt.layers.attention.graph_variants import (
    DSV41_CANDIDATE_FILTERED,
    DSV41_CANDIDATE_UNFILTERED,
)
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_uses_early = DeepseekV4AttnBackend._uses_early_verify_metadata
_install = DeepseekV4AttnBackend._install_early_verify_metadata


class _Core:
    """Mirrors DSV4AttnMetadata.copy_: tensors copied in place, except
    swa_out_cache_loc, which copy_ assigns."""

    def __init__(self, page_table, swa_out_cache_loc):
        self.page_table = page_table
        self.swa_out_cache_loc = swa_out_cache_loc

    def copy_(self, other):
        self.page_table.copy_(other.page_table)
        self.swa_out_cache_loc = other.swa_out_cache_loc


class _Metadata:
    def __init__(self, page_table):
        # As the real full build: the SWA write locations are left to the in-graph op.
        self.core_attn_metadata = _Core(page_table, None)

    def copy_(self, other):
        self.core_attn_metadata.copy_(other.core_attn_metadata)


def _backend(*, early=True, captured=None, early_ids=()):
    bucket = {} if captured is None else {(2, "early"): captured}
    return SimpleNamespace(
        builds_verify_metadata_early=early,
        cuda_graph_metadata_of_bucket_and_bs={_GraphBucket.TARGET_VERIFY: bucket},
        _early_verify_metadata_ids=set(early_ids),
        low_ratios=(1, 2),
    )


def _building_backend():
    backend = _backend()

    def build(*, raw_metadata, online_c128_state_slot_offset, graph_backed):
        # In-graph semantics: no free-memory chunking of the indexer schedule.
        assert graph_backed
        return _Metadata(raw_metadata.clone())

    backend.make_forward_metadata_from_raw_verify = build
    backend.online_c128_mtp = SimpleNamespace(state_slot_offset=lambda: 0)
    backend.token_to_kv_pool = SimpleNamespace(
        request_window=None,
        swa_req_ring_size=None,
        translate_loc_from_full_to_swa=lambda loc: loc + 1000,
    )
    return backend


@pytest.mark.parametrize(
    "variant,early",
    [
        (DSV41_CANDIDATE_UNFILTERED, True),
        (DSV41_CANDIDATE_FILTERED, False),
        ("candidate_all", False),
        (None, False),
    ],
)
def test_only_the_unfiltered_variant_builds_early(monkeypatch, variant, early):
    monkeypatch.setattr(capture_mode, "_capture_attention_variant", variant)
    assert _uses_early(_backend(), ragged_layout=None) is early
    assert not _uses_early(_backend(early=False), ragged_layout=None)
    assert not _uses_early(_backend(), ragged_layout=object())


def test_every_capture_and_replay_writes_the_first_captured_early_object(monkeypatch):
    """The install replaces the in-graph op that translates the step's write
    locations; every later install writes them into the tensor the graph reads. The
    filtered variant's Raw entry of the same key is never touched."""
    monkeypatch.setattr(
        capture_mode, "_capture_attention_variant", DSV41_CANDIDATE_UNFILTERED
    )
    backend = _building_backend()
    raw = object()
    backend.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.TARGET_VERIFY][2] = raw
    for step in range(3):
        assert _uses_early(backend, ragged_layout=None)
        _install(
            backend,
            graph_key=2,
            raw_metadata=torch.tensor([1, 2]) + step,
            out_cache_loc=torch.tensor([5, 6]) + step,
        )
        if step == 0:
            captured = backend.forward_metadata
            swa = captured.core_attn_metadata.swa_out_cache_loc
            assert swa.dtype == torch.int32 and swa.tolist() == [1005, 1006]
    bucket = backend.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.TARGET_VERIFY]
    assert bucket[(2, "early")] is captured and backend.forward_metadata is captured
    assert bucket[2] is raw
    assert backend._early_verify_metadata_ids == {id(captured)}
    assert captured.core_attn_metadata.page_table.tolist() == [3, 4]
    # The verify graphs read the captured write locations by address.
    assert captured.core_attn_metadata.swa_out_cache_loc is swa
    assert swa.tolist() == [1007, 1008]


def test_verify_graph_records_only_the_topk_plans_for_early_metadata(monkeypatch):
    """The verify graph of early metadata records no Raw -> Full upgrade, only the
    indexers' top-k v2 plans, into the very buffers the installed metadata holds."""
    from sglang.kernels.ops.attention import dsv4 as dsv4_ops

    planned = []
    monkeypatch.setattr(
        dsv4_ops,
        "plan_topk_v2",
        lambda lens, out: planned.append((lens, out)),
    )

    def indexer(v2):
        return SimpleNamespace(
            use_topk_v2=v2,
            compressed_seq_lens=torch.ones(3),
            topk_metadata=torch.zeros(4),
        )

    c4, c1, c2 = indexer(True), indexer(True), indexer(False)
    early = SimpleNamespace(
        indexer_metadata=c4, c1_indexer_metadata=c1, c2_indexer_metadata=c2
    )
    # Any other attribute the in-graph build touches would raise on these stand-ins.
    backend = SimpleNamespace(
        forward_metadata=early, _early_verify_metadata_ids={id(early)}
    )
    DeepseekV4AttnBackend.init_forward_metadata_in_graph(backend, SimpleNamespace())
    assert backend.forward_metadata is early
    assert [
        (lens is m.compressed_seq_lens, out is m.topk_metadata)
        for (lens, out), m in zip(planned, (c4, c1))
    ] == [(True, True)] * 2
    assert len(planned) == 2  # c2 plans no top-k v2
