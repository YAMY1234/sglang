import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _batch(
    reuse: bool, carried, *, is_extend: bool = False, seed_buf=None, seed_select=None
) -> SimpleNamespace:
    return SimpleNamespace(
        reuse_dsa_topk_indices=reuse,
        forward_mode=ForwardMode.DRAFT_EXTEND_V2 if is_extend else ForwardMode.DECODE,
        spec_info=SimpleNamespace(
            dsa_topk_indices=carried,
            dsa_seed_topk_capture=seed_buf,
            dsa_seed_topk_select=seed_select,
        ),
    )


def test_mtp_carry_is_empty_when_reuse_disabled():
    batch = _batch(reuse=False, carried="old")
    state = IndexTopKShareState.from_mtp_carry(batch)

    assert state.topk_indices is None
    state.update("new")
    state.publish()

    assert state.topk_indices == "new"
    assert batch.spec_info.dsa_topk_indices == "old"


def test_mtp_carry_reads_and_publishes_when_reuse_enabled():
    batch = _batch(reuse=True, carried="old")
    state = IndexTopKShareState.from_mtp_carry(batch)

    assert state.topk_indices == "old"
    state.update("new")
    state.publish()

    assert state.topk_indices == "new"
    assert batch.spec_info.dsa_topk_indices == "new"


def test_target_carry_stays_local_without_publish():
    batch = _batch(reuse=False, carried="batch")
    state = IndexTopKShareState(batch, "layer")

    assert state.topk_indices == "layer"
    assert batch.spec_info.dsa_topk_indices == "batch"

    state.update(None)

    assert state.topk_indices is None
    assert batch.spec_info.dsa_topk_indices == "batch"


def test_target_none_does_not_fall_back_to_mtp_carry():
    batch = _batch(reuse=True, carried="batch")
    state = IndexTopKShareState(batch, None)

    assert state.topk_indices is None

    state.update("layer")

    assert state.topk_indices == "layer"
    assert batch.spec_info.dsa_topk_indices == "batch"


def test_publish_captures_draft_extend_seed():
    seed_buf = torch.zeros(2, 3, dtype=torch.int64)
    batch = _batch(reuse=False, carried=None, is_extend=True, seed_buf=seed_buf)
    state = IndexTopKShareState.from_mtp_carry(batch)

    assert state.should_publish
    state.update(torch.arange(12, dtype=torch.int64).view(4, 3))
    with patch.object(forward_mha, "resolve_attn_backend", return_value=None):
        state.publish()

    assert torch.equal(seed_buf, torch.arange(6, dtype=torch.int64).view(2, 3))
    assert batch.spec_info.dsa_topk_indices is None


def test_seed_buffer_is_ignored_outside_extend():
    seed_buf = torch.zeros(2, 3, dtype=torch.int64)
    batch = _batch(reuse=False, carried=None, is_extend=False, seed_buf=seed_buf)
    state = IndexTopKShareState.from_mtp_carry(batch)

    assert not state.should_publish
    state.update(torch.ones(4, 3, dtype=torch.int64))
    state.publish()

    assert torch.equal(seed_buf, torch.zeros(2, 3, dtype=torch.int64))


def test_mtp_iteration_clears_batch_state():
    batch = _batch(reuse=False, carried="stale")

    with IndexTopKShareState.mtp_iteration(batch) as state:
        assert state is not None
        assert batch.reuse_dsa_topk_indices
        assert batch.spec_info.dsa_topk_indices is None
        batch.spec_info.dsa_topk_indices = "draft-topk"

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices is None


def test_mtp_iteration_clears_batch_state_on_exception():
    batch = _batch(reuse=False, carried=None)

    with pytest.raises(RuntimeError, match="draft step blew up"):
        with IndexTopKShareState.mtp_iteration(batch):
            batch.spec_info.dsa_topk_indices = "draft-topk"
            raise RuntimeError("draft step blew up")

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices is None


def test_disabled_mtp_iteration_is_passthrough():
    batch = _batch(reuse=False, carried="untouched")

    with IndexTopKShareState.mtp_iteration(batch, enabled=False) as state:
        assert state is None
        assert not batch.reuse_dsa_topk_indices
        assert batch.spec_info.dsa_topk_indices == "untouched"

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices == "untouched"


def test_mtp_iteration_preserves_draft_extend_seed():
    batch = _batch(reuse=False, carried="extend-seed")

    with IndexTopKShareState.mtp_iteration(batch, keep_carry_seed=True) as state:
        assert state is not None
        assert batch.spec_info.dsa_topk_indices == "extend-seed"
        assert state.topk_indices == "extend-seed"

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices is None


def test_mtp_iteration_clears_missing_draft_extend_seed():
    batch = _batch(reuse=False, carried=None)

    with IndexTopKShareState.mtp_iteration(batch, keep_carry_seed=True):
        assert batch.spec_info.dsa_topk_indices is None

    assert not batch.reuse_dsa_topk_indices
    assert batch.spec_info.dsa_topk_indices is None


def _capture(
    indices, *, fused=True, ragged=True, flattened=True, select=None, path="mha"
):
    # Two requests share their first two physical slots. Logical index 5 is
    # outside the four-slot physical pool, but maps to the valid physical slot 2.
    table = torch.tensor([[3, 1, 0], [3, 1, 2]], dtype=torch.int32)
    metadata = SimpleNamespace(
        page_table_1=table,
        page_table_1_flattened=table.flatten() if flattened else None,
        indexer_seq_lens_cpu=torch.tensor([3, 3]),
    )
    # Exercise the real seed export method without initializing GPU resources.
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.use_fused_topk = fused
    backend.get_topk_transform_method = Mock(
        return_value=TopkTransformMethod.RAGGED if ragged else TopkTransformMethod.PAGED
    )
    backend.forward_metadata = metadata
    rows = len(indices) if select is None else len(select)
    capture = torch.full((rows + 1, indices.shape[1]), -99, dtype=torch.int32)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reuse_dsa_topk_indices=False,
        spec_info=SimpleNamespace(
            dsa_seed_topk_capture=capture,
            dsa_seed_topk_select=select,
        ),
    )
    with patch.object(forward_mha, "resolve_attn_backend", return_value=backend):
        if path == "mha":
            indexer = Mock(return_value=indices)
            forward_mha.forward_dsa_indexer_for_mha(
                indexer,
                hidden_states=None,
                q_lora=None,
                positions=None,
                forward_batch=batch,
                layer_id=0,
            )
            assert indexer.call_args.kwargs["return_indices"]
        else:
            # Long prefills take sparse MLA and export through publish().
            state = IndexTopKShareState.from_mtp_carry(batch)
            state.update(indices)
            state.publish()
    assert torch.all(capture[-1] == -99)
    return capture[:-1]


@pytest.mark.parametrize("path", ["mha", "publish"])
@pytest.mark.parametrize("flattened", [True, False])
@pytest.mark.parametrize("select", [None, torch.tensor([2, 0])])
def test_ragged_seed_maps_shared_prefix_and_preserves_padding(flattened, select, path):
    """Export physical slots, even when shared prefixes make ragged offsets larger."""
    indices = torch.tensor([[0, 2, -1], [3, 4, -1], [5, 1, -1]], dtype=torch.int32)
    original = indices.clone()
    expected = torch.tensor([[3, 0, -1], [3, 1, -1], [2, 1, -1]], dtype=torch.int32)
    if select is not None:
        expected = expected[select]
    actual = _capture(indices, flattened=flattened, select=select, path=path)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(indices, original)


@pytest.mark.parametrize("path", ["mha", "publish"])
@pytest.mark.parametrize("fused,ragged", [(True, False), (False, True), (False, False)])
def test_paged_and_unfused_seed_contracts_are_preserved(fused, ragged, path):
    indices = torch.tensor([[2, 0, -1], [1, 2, -1]], dtype=torch.int32)
    actual = _capture(indices, fused=fused, ragged=ragged, path=path)
    torch.testing.assert_close(actual, indices)


@pytest.mark.parametrize("path", ["mha", "publish"])
@pytest.mark.parametrize("invalid", [-2, 6])
def test_invalid_ragged_indices_are_not_clamped(invalid, path):
    with pytest.raises((IndexError, RuntimeError)):
        _capture(torch.tensor([[invalid, -1]], dtype=torch.int32), path=path)


# DCP decode top-k: the K31HT switches choose the candidate select, merge launch
# and push-combine kernel; off (the default), the stock kernels run.


@pytest.mark.parametrize("enabled", [False, True])
def test_k31ht_dsa_select_produces_the_dcp_merge_candidates(enabled):
    from sglang.srt.layers.attention.dsa import dsa_dcp_sharded

    logits = torch.randn(2, 16)
    local_lens = torch.tensor([16, 9], dtype=torch.int32)
    calls = []

    def select(logits_, lens, out, world, rank, topk):
        calls.append((world, rank, topk, lens.tolist()))
        return out.fill_(7)

    def stock(logits_, lens, *, out_candidates, world, rank, **kwargs):
        calls.append(("stock", world, rank))
        out_candidates.fill_(3)

    switch = SimpleNamespace(ENABLED=enabled, local_candidates=select)
    parallel = SimpleNamespace(attn_dcp_size=8, attn_dcp_rank=3)
    with (
        patch.object(dsa_dcp_sharded, "_k31ht_select", switch),
        patch.object(dsa_dcp_sharded, "get_parallel", lambda: parallel),
        patch.object(dsa_dcp_sharded, "topk_transform_candidates_v2", stock),
        patch.object(dsa_dcp_sharded, "candidates_cluster_size", lambda rows: 1),
    ):
        out = dsa_dcp_sharded._local_candidates(logits, local_lens, 4)

    assert tuple(out.shape) == (2, 4, 2) and out.dtype == torch.int32
    if enabled:
        assert calls == [(8, 3, 4, [16, 9])] and bool((out == 7).all())
    else:
        assert calls == [("stock", 8, 3)] and bool((out == 3).all())


@pytest.mark.parametrize("staged,min_blocks", [(False, 1), (True, 1), (True, 2)])
def test_dcp_topk_merge_staged_variant_launch_args(staged, min_blocks):
    from sglang.kernels.ops.attention.dsa import dcp_topk_merge
    from sglang.srt.environ import envs

    loaded = []
    with (
        envs.SGLANG_K31_DCP_MERGE_STAGED.override(staged),
        envs.SGLANG_K31_DCP_MERGE_MIN_BLOCKS.override(min_blocks),
        patch.object(dcp_topk_merge, "_staged_launch", dcp_topk_merge._staged_launch.__wrapped__),
        patch.object(dcp_topk_merge, "load_jit", lambda *a, **kw: loaded.append((a, kw))),
    ):
        dcp_topk_merge._jit_dcp_topk_merge_module.__wrapped__(8, 2048, 768, True)

    (args, kwargs), = loaded
    expected = (8, 2048, 768, 1) + ((1, min_blocks) if staged else ())
    assert args[0] == "dcp_topk_merge" and len(args) == 1 + len(expected)
    assert kwargs["cuda_wrappers"][0][1] == f"DcpTopKMergeKernel<{', '.join(map(str, expected))}>::run"


def test_dcp_topk_merge_min_blocks_fail_closed():
    from sglang.kernels.ops.attention.dsa import dcp_topk_merge
    from sglang.srt.environ import envs

    with (
        envs.SGLANG_K31_DCP_MERGE_STAGED.override(True),
        envs.SGLANG_K31_DCP_MERGE_MIN_BLOCKS.override(0),
        pytest.raises(ValueError, match="MIN_BLOCKS"),
    ):
        dcp_topk_merge._staged_launch.__wrapped__()


@pytest.mark.parametrize("env,world,expected", [(None, 4, 768), (1024, 4, 1024), (1024, 8, 384)])
def test_dcp4_push_cand_k_env(env, world, expected):
    from sglang.kernels.ops.attention.dsa import dcp_topk_merge
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa import dsa_dcp_sharded

    with envs.SGLANG_DCP4_PUSH_CAND_K.override(env or 768):
        assert dsa_dcp_sharded._merge_cand_k(world, 2048) == expected
    loaded = []
    with patch.object(dcp_topk_merge, "load_jit", lambda *a, **kw: loaded.append((a, kw))):
        dcp_topk_merge._jit_dcp_topk_merge_pushed_module.__wrapped__(world, 2048, expected)
    (args, kwargs), = loaded
    assert kwargs["cuda_wrappers"][0][1].startswith(f"DcpTopKMergePushedKernel<{world}, 2048, {expected}, ")


@pytest.mark.parametrize("cand_k", [500, 256, 4096])
def test_dcp4_push_cand_k_fail_closed(cand_k):
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa import dsa_dcp_sharded

    with envs.SGLANG_DCP4_PUSH_CAND_K.override(cand_k), pytest.raises(ValueError, match="CAND_K"):
        dsa_dcp_sharded._merge_cand_k(4, 2048)


@pytest.mark.parametrize("form", ["0", "v3"])
def test_dcp_push_combine_form_selects_the_kernel(form):
    import os

    from sglang.kernels.ops.attention import dcp_push_combine

    loaded = []
    with (
        patch.object(dcp_push_combine, "COMBINE_FORM", form),
        patch.object(dcp_push_combine, "load_jit", lambda *a, **kw: loaded.append((a, kw)) or "module"),
    ):
        assert dcp_push_combine._jit_module.__wrapped__(8) == "module"

    (args, kwargs), = loaded
    if form == "v3":
        assert args[0] == "dcp_push_combine_v3"
        assert os.path.isfile(kwargs["cuda_files"][0])
        assert kwargs["cuda_wrappers"][0][1].startswith("DcpPushCombineV3Kernel<8, ")
    else:
        assert args[0] == "dcp_push_combine"
        assert kwargs["cuda_files"] == ["dsa/dcp_push_combine.cuh"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
