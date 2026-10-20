"""SGLANG_OPT_DSV41_TARGET_METADATA_EARLY: verify inputs and their metadata glue
graph staged on a side stream beside the draft graph (waiting only for the head), and
the runner gating that keeps every other configuration and variant on the main path."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.graph_variants import (
    DSV41_CANDIDATE_FILTERED,
    DSV41_CANDIDATE_UNFILTERED,
    Dsv41CandidateGraphVariants,
)
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    GraphSlot,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as runner_mod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.early_verify_metadata import (
    STAGED_SLOTS,
    EarlyVerifyMetadata,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

KEY = (2, "TARGET_VERIFY", "TARGET_VERIFY", DSV41_CANDIDATE_UNFILTERED)
HEAD = object()  # the head's event the side stream waits for


def _registry(device) -> CudaGraphBufferRegistry:
    registry = CudaGraphBufferRegistry(
        device=torch.device(device), max_bs=4, max_num_tokens=12
    )
    for name in sorted(STAGED_SLOTS):
        tokens = name in ("positions", "out_cache_loc")
        registry.register_slot(
            GraphSlot(
                name,
                (lambda bs, mt: (mt,)) if tokens else (lambda bs, mt: (bs,)),
                torch.int64,
                axis="tokens" if tokens else "bs",
            )
        )
    return registry


def _sources(step: int, device="cpu") -> dict:
    return {
        "positions": torch.arange(6, device=device) + step,
        "out_cache_loc": torch.arange(6, device=device) + 100 + step,
        "req_pool_indices": torch.tensor([3, 1], device=device),
        "seq_lens": torch.tensor([10, 11], device=device) + step,
    }


class _Glue:
    def __init__(self):
        self.replays = []

    def replay(self, key):
        self.replays.append(key)


SIDE_WAITS = []


@pytest.fixture
def fake_streams(monkeypatch):
    """CPU stand-ins; returns the list of events the current stream waited on. The
    side stream records what it waited on in ``side_waits`` and refuses to wait for a
    whole stream (it must not order itself after the draft)."""
    waits = []
    current = SimpleNamespace(wait_event=waits.append)

    def no_stream_wait(other):
        raise AssertionError("the side stream must wait for the head event only")

    side = SimpleNamespace(
        wait_event=lambda e: SIDE_WAITS.append(e), wait_stream=no_stream_wait
    )
    SIDE_WAITS.clear()
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: side)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda *a: None)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None)
    return waits


def test_take_accepts_only_the_staged_key_and_views(fake_streams):
    registry, glue = _registry("cpu"), _Glue()
    early = EarlyVerifyMetadata(device="cpu", log_every=1000)

    sources = _sources(0)
    early.stage(glue=glue, key=KEY, registry=registry, sources=sources, after=HEAD)
    assert glue.replays == [KEY]
    assert registry.get_slot("seq_lens").buffer[:2].tolist() == [10, 11]
    assert early.take(key=KEY, forward_batch=SimpleNamespace(**sources))
    assert len(fake_streams) == 1
    assert SIDE_WAITS == [HEAD]

    # Equal values in another tensor: the staged copy may be stale, refill.
    sources = _sources(1)
    early.stage(glue=glue, key=KEY, registry=registry, sources=sources, after=HEAD)
    other = dict(sources, positions=sources["positions"].clone())
    assert not early.take(key=KEY, forward_batch=SimpleNamespace(**other))

    early.stage(glue=glue, key=KEY, registry=registry, sources=sources, after=HEAD)
    assert not early.take(key=(4, *KEY[1:]), forward_batch=SimpleNamespace(**sources))
    assert len(fake_streams) == 3

    # Nothing staged: no wait, no match.
    assert not early.take(key=KEY, forward_batch=SimpleNamespace(**sources))
    assert len(fake_streams) == 3
    assert (early.staged_ct, early.taken_ct, early.rejected_ct) == (3, 1, 2)


def test_every_staged_record_ends_in_a_wait(fake_streams):
    registry, glue = _registry("cpu"), _Glue()
    early = EarlyVerifyMetadata(device="cpu", log_every=1000)
    early.stage(glue=glue, key=KEY, registry=registry, sources=_sources(0), after=HEAD)
    # Restaging (the previous one never consumed) waits first.
    early.stage(glue=glue, key=KEY, registry=registry, sources=_sources(1), after=HEAD)
    assert len(fake_streams) == 1
    early.discard()
    early.discard()
    assert len(fake_streams) == 2


def _dspark_model_runner(*, dspark=True, draft=False, device="cuda"):
    return SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_dspark=lambda: dspark),
        is_draft_worker=draft,
        device=device,
    )


def _gating_runner(**overrides):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    fields = dict(
        model_runner=_dspark_model_runner(),
        capture_forward_mode=ForwardMode.TARGET_VERIFY,
        ragged_verify_mode=False,
        require_gathered_buffer=False,
        enable_two_batch_overlap=False,
    )
    fields.update(overrides)
    for name, value in fields.items():
        setattr(runner, name, value)
    return runner


@pytest.mark.parametrize(
    "override",
    [
        {"model_runner": _dspark_model_runner(draft=True)},
        {"model_runner": _dspark_model_runner(dspark=False)},
        {"model_runner": _dspark_model_runner(device="cpu")},
        {"capture_forward_mode": ForwardMode.DECODE},
        {"ragged_verify_mode": True},
        {"require_gathered_buffer": True},
        {"enable_two_batch_overlap": True},
    ],
)
def test_runner_wants_early_metadata_only_as_the_eligible_verify_runner(
    monkeypatch, override
):
    parallel = SimpleNamespace(attn_dp_enabled=False)
    monkeypatch.setattr(runner_mod, "get_parallel", lambda: parallel)
    with envs.SGLANG_OPT_DSV41_TARGET_METADATA_EARLY.override(True):
        assert _gating_runner()._wants_early_verify_metadata()
        assert not _gating_runner(**override)._wants_early_verify_metadata()
        parallel.attn_dp_enabled = True
        assert not _gating_runner()._wants_early_verify_metadata()
    parallel.attn_dp_enabled = False
    assert not _gating_runner()._wants_early_verify_metadata()


def test_staged_slots_exclude_the_draft_written_input_ids():
    assert "input_ids" not in STAGED_SLOTS  # the draft graph may write it in place


def _variants(span=16):
    return Dsv41CandidateGraphVariants(
        graph_limits=((DSV41_CANDIDATE_UNFILTERED, span),),
        capture_labels=(DSV41_CANDIDATE_UNFILTERED, DSV41_CANDIDATE_FILTERED),
        verify_extra_tokens=3,
    )


def _staging_runner(calls, *, captured=lambda key: key[0] != 4):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner._early_verify_metadata = SimpleNamespace(
        stage=lambda **kw: calls.append(("stage", kw["key"], kw["after"])),
        take=lambda **kw: calls.append(("take", kw["key"])) or True,
        discard=lambda: calls.append(("discard", None)),
    )
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner.capture_bs = [1, 2, 4]
    runner.captured_req_width = 3
    runner.buffer_registry = None
    runner.attention_graph_variants = _variants()
    runner._metadata_glue = SimpleNamespace(is_captured=captured)
    return runner


def test_runner_stages_only_captured_unpadded_early_variant_keys():
    calls = []
    runner = _staging_runner(calls)

    def stage(bs, max_seq_len=10):
        return runner.stage_verify_metadata_early(
            seq_lens=torch.zeros(bs),
            req_pool_indices=torch.zeros(bs),
            out_cache_loc=torch.zeros(bs * 3),
            positions=torch.zeros(bs * 3),
            max_seq_len=max_seq_len,
            after=HEAD,
        )

    assert not stage(3)  # padded to 4 at replay
    assert not stage(4)  # glue graph not captured yet
    assert not stage(2, max_seq_len=14)  # 14 + 3 > 16: the filtered variant
    assert not stage(2, max_seq_len=None)  # unknown length: the filtered fallback
    assert stage(2, max_seq_len=13)
    key = runner._glue_key(
        2, str(ForwardMode.TARGET_VERIFY), DSV41_CANDIDATE_UNFILTERED
    )
    assert key[-1] == DSV41_CANDIDATE_UNFILTERED
    assert calls == [("discard", None)] * 4 + [("stage", key, HEAD)]


def test_glue_key_is_unchanged_without_the_early_metadata():
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    mode = str(ForwardMode.TARGET_VERIFY)
    assert runner._glue_key(2, mode, DSV41_CANDIDATE_UNFILTERED) == (2, mode, mode)


def test_runner_takes_only_an_unpadded_captured_replay():
    calls = []
    runner = _staging_runner(calls)
    fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    mode = str(ForwardMode.TARGET_VERIFY)

    def take(*, raw_bs=2, bs=2, is_ragged=False):
        key = runner._glue_key(bs, mode, DSV41_CANDIDATE_UNFILTERED)
        return runner._take_early_verify_metadata(
            fb, raw_bs=raw_bs, bs=bs, is_ragged=is_ragged, key=key
        )

    assert not take(raw_bs=1)
    assert not take(is_ragged=True)
    assert not take(raw_bs=4, bs=4)  # its glue graph is gone
    assert take()
    key = runner._glue_key(2, mode, DSV41_CANDIDATE_UNFILTERED)
    assert calls == [("discard", None)] * 3 + [("take", key)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_side_stream_work_is_visible_after_take():
    """Real streams: what the side stream wrote is what the main stream reads after
    ``take``, step after step, while the main stream is busy past the head event."""
    device = torch.device("cuda")
    registry = _registry(device)
    out = torch.zeros(2, dtype=torch.int64, device=device)
    seq_lens_buf = registry.get_slot("seq_lens").buffer

    class Glue:
        def replay(self, key):
            torch.mul(seq_lens_buf[:2], 2, out=out)

    early = EarlyVerifyMetadata(device=device, log_every=1000)
    busy = torch.randn(2048, 2048, device=device)
    head = torch.cuda.Event()
    for step in range(4):
        sources = _sources(step, device=device)
        head.record()
        for _ in range(4):
            busy = busy @ busy.T / 2048  # the "draft", after the head
        early.stage(
            glue=Glue(), key=KEY, registry=registry, sources=sources, after=head
        )
        assert early.take(key=KEY, forward_batch=SimpleNamespace(**sources))
        assert (out + 0).tolist() == [2 * (10 + step), 2 * (11 + step)]
        positions = registry.get_slot("positions").buffer[:6]
        assert positions.tolist() == list(range(step, step + 6))
