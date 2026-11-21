"""The tail server's event loop, in place of a scheduler's (--dsv41-tail-stage-server)."""

from __future__ import annotations

import logging
import statistics
import time
from types import SimpleNamespace
from typing import Sequence

import torch
from sglang.srt.disaggregation.dsv41_tail_stage.batcher import parse_buckets
from sglang.srt.disaggregation.dsv41_tail_stage.forward import (
    TailRun,
    TailStageForward,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW

logger = logging.getLogger(__name__)


def build_forward(scheduler) -> TailStageForward:
    return TailStageForward(
        model_runner=scheduler.tp_worker.model_runner,
        req_to_token_pool=scheduler.req_to_token_pool,
        allocator=scheduler.token_to_kv_pool_allocator,
        tree_cache=scheduler.tree_cache,
        model_config=scheduler.model_config,
        num_slots=envs.SGLANG_DSV41_TAIL_STAGE_SLOTS.get(),
        max_tails=envs.SGLANG_DSV41_TAIL_STAGE_MAX_TAILS.get(),
        window=SWA_WINDOW,
        draft_aux=envs.SGLANG_DSV41_TAIL_STAGE_DRAFT_AUX.get(),
    )


def run_tail_stage(scheduler) -> None:
    fwd = build_forward(scheduler)
    counts = envs.SGLANG_DSV41_TAIL_STAGE_SELF_BENCH.get()
    if counts:
        for prefix_len in parse_buckets(
            envs.SGLANG_DSV41_TAIL_STAGE_SELF_BENCH_PREFIX.get()
        ):
            self_bench(
                fwd,
                counts=parse_buckets(counts),
                prefix_len=prefix_len,
                iters=envs.SGLANG_DSV41_TAIL_STAGE_SELF_BENCH_ITERS.get(),
            )
        logger.info("DSV4.1 tail stage bench: all done")
    if envs.SGLANG_DSV41_TAIL_STAGE_PORT.get():
        from sglang.srt.disaggregation.dsv41_tail_stage.service import build_service

        build_service(scheduler, fwd).serve_forever()
    logger.info("DSV4.1 tail stage: no port set, idle")
    while True:
        time.sleep(60)


def self_bench(
    fwd: TailStageForward,
    *,
    counts: Sequence[int],
    prefix_len: int,
    iters: int,
    warmup: int = 3,
) -> None:
    """Time the late layers over tails of a fixed prefix, largest count first so the
    first run warms the tail graphs' gate and captures the largest bucket first."""
    window = fwd.window
    e = prefix_len + window
    s = e + 1
    torch.manual_seed(0)
    fwd.residual.normal_()
    tail_graph = fwd.model_runner.model.model.decoder_replay_tail
    memory = fwd.memory
    for n in sorted(counts, reverse=True):
        prep_ms, fwd_ms = [], []
        for it in range(warmup + iters):
            grants = [SimpleNamespace(room=-(i + 1), slot=i, e=e) for i in range(n)]
            assert all(memory.take(g) is not None for g in grants), (
                "prefixes do not fit"
            )
            memory.end_round()
            tails = [
                TailRun(
                    slot=g.slot, s=s, e=e, window_full=memory.held[g.room].window_full
                )
                for g in grants
            ]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prepared = fwd.prepare(tails)
            t1 = time.perf_counter()
            fwd.execute(tails, prepared)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for g in grants:
                memory.give_back(g)
            if it >= warmup:
                prep_ms.append((t1 - t0) * 1e3)
                fwd_ms.append((t2 - t1) * 1e3)
        total = [a + b for a, b in zip(prep_ms, fwd_ms)]
        med = statistics.median(total)
        logger.info(
            "DSV4.1 tail stage bench: tails=%d rows=%d prefix=%d total_ms median %.2f "
            "min %.2f max %.2f | prepare_ms median %.2f | forward_ms median %.2f | "
            "%.0f tails/s | tail graph %s",
            n,
            n * window,
            prefix_len,
            med,
            min(total),
            max(total),
            statistics.median(prep_ms),
            statistics.median(fwd_ms),
            n / med * 1e3,
            "off"
            if tail_graph is None
            else f"warm={tail_graph.runner.warm} graphs={len(tail_graph.runner._graphs)}",
        )
    logger.info("DSV4.1 tail stage bench: done")
