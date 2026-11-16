"""CUDA graphs of the temperature-sampling verify accept of static DSpark decode steps
(SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH).

The static verify graph folds only the greedy accept (DsparkVerifyEpilogue). On an
all-sampled step the rest ran eagerly after the replay: the draft and target softmax,
the rejection kernel, the bonus gather, the accept-site TP broadcast, the engram
history commit, the future relay and the target-hidden KV commit. This module captures
that region once per verify-graph batch size and replays it right after the verify
graph.

The region reads the verify graph's static inputs and outputs (``static_io``) and the
draft sampler's static buffers, which hold this step's values after both replays, so a
replay stages nothing. Its accept outputs land in the epilogue's static accept buffers,
like the greedy fold. Per key the first step runs eagerly (it compiles kernels and warms
allocations), the second captures, later steps replay. The uniform draws come from
torch's default generator, whose graph-safe offsets make a replay draw what the eager
accept draws at the same generator state; with block-wise rejection they come from the
worker's one RejectionNoise, whose device step counter a replay advances like an eager
call. The DSPARK_ACCEPT_SAMPLE broadcast stays in
the region (pynccl inside the graph), so TP ranks agree as in the eager path. With the
deferred draft KV store, the verify graph already joined this step's projection, so
the region stages the stores from it.

The caller routes only all-sampled steps without top-k / top-p, logit adjustments or
grammar; everything else keeps the eager accept. Under SGLANG_SIMULATE_ACC_LEN the region
draws the bonus at the simulated row from the buffers the caller stages with
begin_sim_bonus.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from functools import partial
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark.blockwise_rejection import RejectionNoise
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    DsparkVerifyEpilogue,
    StaticVerifyIO,
    accept_draft_tokens,
    build_accept_outs,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite

logger = logging.getLogger(__name__)


class GraphSamplingInfo(msgspec.Struct, frozen=True):
    """The SamplingBatchInfo fields the sampling accept reads, for an all-sampled
    batch without top-k / top-p."""

    temperatures: torch.Tensor
    is_all_greedy: bool = False
    is_any_greedy: bool = False
    need_top_k_sampling: bool = False
    need_top_p_sampling: bool = False


def sampling_accept_commit_lens(
    io: StaticVerifyIO,
    *,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    gated: bool,
    draft_sampler,
    epilogue: DsparkVerifyEpilogue,
    tp_sync: SpecTpSync,
    block_verification: bool,
    rejection_noise: Optional[RejectionNoise],
    sim_bonus: Optional[Callable[[], torch.Tensor]] = None,
) -> Optional[torch.Tensor]:
    """The eager sampling accept and finalize into the epilogue's static buffers;
    returns the gated commit lengths when ``gated``. ``sim_bonus``: joins and returns
    the simulated bonus a side stream drew beside the accept."""
    bs, stride = io.candidates.shape
    params = draft_sampler.params
    draft_tokens = io.candidates[:, 1:]
    draft_block = DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=(
            None if draft_sampler.greedy_draft else draft_sampler.step_logits[:bs]
        ),
        greedy_mask=params.greedy_mask[:bs],
        temperatures=params.temperatures[:bs],
        greedy_draft=draft_sampler.greedy_draft,
    )
    correct_len, bonus, cap_trim_lens = accept_draft_tokens(
        candidates=io.candidates,
        target_logits=io.logits,
        draft_block=draft_block,
        # params.temperatures is the batch's, clamped at 1e-5 (resolve_draft_params).
        sampling_info=GraphSamplingInfo(temperatures=params.temperatures[:bs]),
        draft_input=None,
        gamma=stride - 1,
        verify_num_draft_tokens=stride,
        block_verification=block_verification,
        rejection_noise=rejection_noise,
    )
    bonus_logits = io.logits
    if epilogue.simulated_correct_len is not None:
        drawn = epilogue.sim_bonus(io.logits, bs) if sim_bonus is None else sim_bonus()
        bonus = drawn.to(bonus.dtype).view_as(bonus)
        bonus_logits = None
    commit_lens = build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=io.seq_lens[:bs],
        draft_tokens=draft_tokens.contiguous(),
        gamma=stride - 1,
        verify_num_draft_tokens=stride,
        tp_sync=tp_sync,
        sync_site=SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE,
        simulated_correct_len=epilogue.simulated_correct_len,
        target_logits=bonus_logits,
        out=epilogue.read_accept(bs),
        block_accept_lens_out=epilogue.read_block_accept_lens(bs),
    ).commit_lens
    if not gated:
        return None
    return torch.minimum(commit_lens, verify_lens.to(torch.int32)) * gate


class DsparkSamplingAcceptGraph:
    """Per-batch-size CUDA graphs of the sampling accept region of static verify."""

    # One key per verify-graph batch size is expected; more means addresses move,
    # and the further keys run eagerly. Arbitrary bound.
    MAX_GRAPHS = 32

    def __init__(
        self,
        *,
        epilogue: DsparkVerifyEpilogue,
        resolve_draft_sampler: Callable[[], Optional[object]],
        tp_sync: SpecTpSync,
        block_verification: bool,
        device,
        rejection_noise: Optional[RejectionNoise] = None,
        rank_local: bool = False,
        side_streams: bool = False,
    ) -> None:
        self._epilogue = epilogue
        # SGLANG_OPT_DSV41_DSPARK_ACCEPT_GRAPH_STREAMS: captures fork these; eager runs stay serial.
        self._forks = _RegionForks() if side_streams else None
        # Attention DP with attn_tp 1: the region holds no collective, and ranks capture
        # at different steps, so it is captured outside graph_capture() (whose exit
        # registers buffers with a collective over the TP group).
        self._rank_local = rank_local
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self.eager_fallbacks: dict[str, int] = {}
        # The static IO of the last region run (SGLANG_DSPARK_ACCEPT_GRAPH_CHECK_STEPS).
        self.last_io: Optional[StaticVerifyIO] = None
        self._resolve_draft_sampler = resolve_draft_sampler
        self._tp_sync = tp_sync
        self._block_verification = block_verification
        self._rejection_noise = rejection_noise
        # The region always commits; padded rows are gated by verify_lens.
        self._gate = torch.ones((1,), dtype=torch.int32, device=device)
        self._graphs: dict[Hashable, torch.cuda.CUDAGraph] = {}
        self._warmed_up: set[Hashable] = set()
        self._pool = None
        self.disabled = False
        self.num_replays = 0
        # Set by run(): whether this step's region committed the engram history.
        self.committed_engram = False

    def run(
        self, *, bs: int, target_logits: torch.Tensor, graph_bs: Optional[int] = None
    ) -> Optional[AcceptOuts]:
        """This step's accept through the region, or None when the step must take the
        eager path. Reads only host state every TP rank holds identically.
        ``graph_bs``: the replayed verify graph's batch size when it pads past ``bs``
        (attention DP pads to the largest DP batch); default the smallest one >= bs."""
        if self.disabled:
            return self._fallback("disabled")
        draft_sampler = self._resolve_draft_sampler()
        if draft_sampler is None or (
            not draft_sampler.greedy_draft and draft_sampler.step_logits is None
        ):
            return self._fallback("no draft sampler buffers")
        epilogue = self._epilogue
        if graph_bs is None:
            graph_bs = min((b for b in epilogue.static_io if b >= bs), default=None)
        if graph_bs is None or graph_bs < bs or graph_bs not in epilogue.static_io:
            return self._fallback("no recorded verify graph")
        io = epilogue.static_io[graph_bs]
        # The replayed verify graph must be the one whose buffers were recorded.
        if target_logits.data_ptr() != io.logits.data_ptr():
            return self._fallback("verify logits buffer moved")
        self.last_io = io
        key = (
            graph_bs,
            io.logits.data_ptr(),
            draft_sampler.params.temperatures.data_ptr(),
            0
            if draft_sampler.step_logits is None
            else draft_sampler.step_logits.data_ptr(),
        )
        region = partial(self._region, io, bs=graph_bs, draft_sampler=draft_sampler)
        self.committed_engram = epilogue.engram_history() is not None
        self._run(key, region)
        return epilogue.read_accept(bs)

    def _region(self, io: StaticVerifyIO, *, bs: int, draft_sampler) -> bool:
        epilogue = self._epilogue
        forks = self._forks if _capturing() else None
        projected_kvs = sim_bonus = None
        if forks is not None:
            if epilogue.folds_commit and epilogue.deferred_draft_kv is None:
                hidden = io.hidden[: io.positions.shape[0]]
                projected_kvs = forks.fork(
                    0, lambda: epilogue.commit_ctx.kv_injector.project_for_store(hidden)
                )
            if epilogue.simulated_correct_len is not None:
                sim_bonus = forks.fork(1, lambda: epilogue.sim_bonus(io.logits, bs))
        committed = epilogue.run_static(
            io,
            accept=partial(
                sampling_accept_commit_lens,
                draft_sampler=draft_sampler,
                epilogue=epilogue,
                tp_sync=self._tp_sync,
                block_verification=self._block_verification,
                rejection_noise=self._rejection_noise,
                sim_bonus=sim_bonus,
            ),
            gate=self._gate,
            kv_projected=True,
            projected_kvs=projected_kvs,
        )
        if forks is not None:
            forks.join_all()
        return committed

    def _fallback(self, reason: str) -> None:
        """Count an eager step by reason; the first of each reason is logged."""
        n = self.eager_fallbacks.get(reason, 0) + 1
        self.eager_fallbacks[reason] = n
        if n == 1:
            logger.info("DSpark sampling-accept graph: eager step (%s).", reason)
        return None

    def _run(self, key: Hashable, region: Callable[[], object]) -> None:
        graph = self._graphs.get(key)
        if graph is None:
            if key not in self._warmed_up:
                if len(self._warmed_up) < self.MAX_GRAPHS:
                    self._warmed_up.add(key)
                region()
                return
            graph = self._capture(key, region)
            if graph is None:
                return
        graph.replay()
        self.num_replays += 1

    def _capture(
        self, key: Hashable, region: Callable[[], object]
    ) -> Optional[torch.cuda.CUDAGraph]:
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        try:
            if self._rank_local:
                self._capture_rank_local(graph, region)
            else:
                # graph_capture() enables pynccl, which the in-graph accept broadcast needs.
                with (
                    graph_capture() as ctx,
                    torch.cuda.graph(
                        graph,
                        pool=self._pool,
                        stream=ctx.stream,
                        capture_error_mode="thread_local",
                    ),
                ):
                    region()
        except Exception:
            logger.warning(
                "DSpark sampling-accept graph capture failed for %s; sampling steps "
                "take the eager accept from now on.",
                key,
                exc_info=True,
            )
            self.disabled = True
            # A failed capture recorded the ops without running them.
            region()
            return None
        logger.info("Captured the DSpark sampling-accept region for %s.", key)
        self._graphs[key] = graph
        return graph

    def _capture_rank_local(
        self, graph: torch.cuda.CUDAGraph, region: Callable[[], object]
    ) -> None:
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream()
        stream = self._capture_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(
            graph, pool=self._pool, stream=stream, capture_error_mode="thread_local"
        ):
            region()
        torch.cuda.current_stream().wait_stream(stream)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class _RegionForks:
    """Side streams of a captured sampling-accept region: work whose inputs are final
    before the region (the verify graph's hidden and logits) runs beside the accept."""

    def __init__(self) -> None:
        self._streams: Optional[list] = None
        self._events: Optional[list] = None
        self._pending: set[int] = set()

    def fork(self, slot: int, fn: Callable[[], object]) -> Callable[[], object]:
        """Run ``fn`` on side stream ``slot``; the returned join waits for it and
        returns its result (once joined, later calls only return it)."""
        if self._streams is None:
            self._streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            self._events = [torch.cuda.Event(), torch.cuda.Event()]
        stream, event = self._streams[slot], self._events[slot]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out = fn()
            event.record()
        self._pending.add(slot)

        def join():
            if slot in self._pending:
                torch.cuda.current_stream().wait_event(event)
                self._pending.discard(slot)
            return out

        return join

    def join_all(self) -> None:
        """A capture must end with every side stream joined."""
        for slot in sorted(self._pending):
            torch.cuda.current_stream().wait_event(self._events[slot])
        self._pending.clear()


_ACCEPT_FIELDS = (
    "correct_len",
    "bonus",
    "cap_trim_lens",
    "commit_lens",
    "new_seq_lens",
    "out_tokens",
)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.numel() == b.numel() and torch.equal(
        a.reshape(-1).to(torch.int64), b.reshape(-1).to(torch.int64)
    )


class AcceptGraphCheck:
    """SGLANG_DSPARK_ACCEPT_GRAPH_CHECK_STEPS: compares the sampling-accept graph's
    outputs and the static inputs it read with the eager accept on the same step,
    bitwise, for the first N region steps. Debug only: it syncs the host."""

    _LOG_EVERY = 100

    def __init__(self, *, steps: int) -> None:
        self._left = int(steps)
        self._checked = 0
        self._padded = 0
        self._mismatched = 0
        self._bad_fields: dict[str, int] = {}

    def begin(self, rejection_noise: Optional[RejectionNoise]):
        """Before the region: the draw counter to rerun the eager accept from."""
        if self._left <= 0:
            return None
        return (
            rejection_noise,
            None if rejection_noise is None else rejection_noise.counter.clone(),
        )

    def compare(
        self,
        check,
        *,
        region_accept: AcceptOuts,
        reference: Callable[[], AcceptOuts],
        io: StaticVerifyIO,
        live_io: tuple,
        kv_fingerprint: Callable[[], torch.Tensor],
        eager_kv_commit: Optional[Callable[[torch.Tensor], None]],
    ) -> None:
        noise, counter_before = check
        if noise is not None:
            counter_after = noise.counter.clone()
            noise.counter.copy_(counter_before)
        ref = reference()
        if noise is not None:
            noise.counter.copy_(counter_after)
        bad = [
            f
            for f in _ACCEPT_FIELDS
            if not _same(getattr(region_accept, f), getattr(ref, f))
        ]
        bad += self._compare_inputs(io, live_io, bs=region_accept.commit_lens.shape[0])
        if eager_kv_commit is not None:
            # The eager commit rewrites the rows the region wrote; equal bytes keep the sums.
            in_region = kv_fingerprint()
            eager_kv_commit(ref.commit_lens)
            if not _same(in_region, kv_fingerprint()):
                bad.append("draft_kv")
        self._record(bad, padded=io.candidates.shape[0] != ref.commit_lens.shape[0])

    @staticmethod
    def _compare_inputs(io: StaticVerifyIO, live_io: tuple, *, bs: int) -> list:
        candidates, prefix_lens, req_pool_indices, positions, cache_loc = live_io
        n = bs * io.candidates.shape[1]
        pairs = (
            ("in:candidates", io.candidates[:bs], candidates),
            ("in:seq_lens", io.seq_lens[:bs], prefix_lens),
            ("in:req_pool_indices", io.req_pool_indices[:bs], req_pool_indices),
            ("in:positions", io.positions[:n], positions[:n]),
            (
                "in:cache_loc",
                io.out_cache_loc[:n],
                io.out_cache_loc[:n] if cache_loc is None else cache_loc[:n],
            ),
        )
        return [name for name, a, b in pairs if not _same(a, b)]

    def _record(self, bad: list, *, padded: bool) -> None:
        self._left -= 1
        self._checked += 1
        self._padded += int(padded)
        if bad:
            self._mismatched += 1
            for f in bad:
                self._bad_fields[f] = self._bad_fields.get(f, 0) + 1
            if self._mismatched <= 20:
                logger.warning(
                    "DSpark accept graph check: step %d mismatch in %s (padded %s)",
                    self._checked,
                    bad,
                    padded,
                )
        if self._checked % self._LOG_EVERY == 0 or self._left == 0:
            logger.info(
                "DSpark accept graph check: %d steps, %d padded, %d mismatched %s%s",
                self._checked,
                self._padded,
                self._mismatched,
                self._bad_fields,
                " (done)" if self._left == 0 else "",
            )
