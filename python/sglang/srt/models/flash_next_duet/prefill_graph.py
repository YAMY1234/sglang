"""Breakable prefill CUDA graphs for the TwinStar P sub-batch (#287).

The framework prefill runner captures the whole model forward, but a TwinStar
prefill decomposes one batch into a P sub-batch (31 layers + 17 emitters) and a
one-token boundary, so the graph must wrap the P sub-batch instead. Two bodies,
each behind its own switch and captured on the shared graph pool:

  trunk     embedding + P layers 0..30 (PLE and the QSA indexer run as eager
            breaks, GDN / QSA attention cores are the framework's breaks);
            returns the 4-stream residual (and the embedding for v3 latent).
  emitters  the emitters of the arm, reading the residual from a static buffer.

Attention metadata is planned once per P sub-batch by the caller, exactly as in
eager mode; the runners never plan (the factored GDN ring must not be reserved
twice). Rows past the live token count are padding and are never read back.
"""

import contextlib
import dataclasses
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

TRUNK = "TWINSTAR_PREFILL_GRAPH_TRUNK"
EMITTERS = "TWINSTAR_PREFILL_GRAPH_EMITTERS"


def enabled(name) -> bool:
    from .config import runtime_controls

    return runtime_controls()["prefill_graph"]


@contextlib.contextmanager
def _breakable_prefill_config(buckets):
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec

    graph = get_exec().graph
    saved = graph.cuda_graph_config.prefill
    graph.cuda_graph_config.prefill = dataclasses.replace(
        saved, backend=Backend.BREAKABLE, bs=list(buckets), max_bs=max(buckets)
    )
    try:
        yield
    finally:
        graph.cuda_graph_config.prefill = saved


class _Body:
    """layer_model for the runner: forward(input_ids, positions, forward_batch)."""

    def __init__(self, owner, trunk: bool, emit_ids):
        self.owner = owner
        self.trunk = trunk
        self.emit_ids = list(emit_ids)
        cfg = owner.config
        self.width = cfg.hc_count * cfg.hidden_size
        self.streams_in = None  # static input of an emitter-only body

    def allocate(self, max_tokens, device):
        if not self.trunk:
            self.streams_in = torch.zeros(
                max_tokens, self.width, dtype=torch.bfloat16, device=device
            )

    def forward(self, input_ids, positions, forward_batch):
        owner = self.owner
        latent_base = None
        if self.trunk:
            streams, latent_base = owner._p_trunk(forward_batch)
        else:
            streams = self.streams_in[: input_ids.shape[0]]
        for layer in self.emit_ids:
            owner.emitters[str(layer)].emit(streams, forward_batch)
        if not self.trunk:
            return None
        return (streams, latent_base) if latent_base is not None else streams


def _runner_class():
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
        _slice_output_rows,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    class TwinStarPrefillRunner(PrefillCudaGraphRunner):
        def __init__(self, model_runner, body, buckets, name, attributes):
            self.body = body
            self.name = name
            self.run_count = 0
            self.attributes = (
                attributes  # extra P sub-batch fields the capture batch needs
            )
            body.allocate(max(buckets), model_runner.device)
            with _breakable_prefill_config(buckets):
                super().__init__(model_runner)

        def capture(self):
            self.layer_model = self.body
            super().capture()

        def capture_prepare(self, num_tokens):
            forward_batch, backend = super().capture_prepare(num_tokens)
            # Capture requests complete their prompt: no dense-ring reservation.
            forward_batch.twinstar_prompt_final = [True] * forward_batch.batch_size
            for key, value in self.attributes.items():
                setattr(forward_batch, key, value)
            return forward_batch, backend

        def _init_forward_metadata_for_capture(self, forward_batch, num_tokens):
            super()._init_forward_metadata_for_capture(forward_batch, num_tokens)
            if not self.body.trunk:
                self._start_plan_at_emitters()

        def _start_plan_at_emitters(self):
            """An emitter-only capture forward runs without the P layers before it;
            in serving the trunk has already advanced the factored plan to here."""
            linear = getattr(
                self.model_runner.attn_backend, "linear_attn_backend", None
            )
            meta = getattr(linear, "forward_metadata", None)
            plan = getattr(meta, "factored_extend", None)
            if plan is None:
                return
            owner = self.body.owner
            gdn = [l for l in self.body.emit_ids if not owner.emitters[str(l)].is_attn]
            if gdn:
                plan.next_layer = linear.factored.layer_map[gdn[0]]

        def _prepare_forward_metadata_for_replay(
            self, forward_batch, static_forward_batch, num_tokens
        ):
            pass  # the caller planned this P sub-batch once

        def can_run(self, forward_batch) -> bool:
            if os.environ.get("SGLANG_PREFILL_GRAPH_CAPTURE_ONLY", "0") == "1":
                return False  # diagnostic: captured, never replayed
            tokens = int(forward_batch.input_ids.shape[0])
            return 0 < tokens <= self.max_num_tokens and self.can_run_graph(
                forward_batch
            )

        def run(self, forward_batch, streams=None):
            with self.backend.replay_session():
                static = self.load_batch(forward_batch)
                # TwinStar/FlashNext fields set on the P sub-batch after
                # construction; eager breaks read them from the static batch.
                for key, value in vars(forward_batch).items():
                    if key not in vars(static):
                        setattr(static, key, value)
                if getattr(self.body.owner, "pd_trunk_prefill_graph", False):
                    # These are declared ForwardBatch fields, so the generic
                    # extra-field copy above leaves their constructor defaults.
                    # Keep current host identity/phase values for eager breaks;
                    # captured device inputs and track buffers stay untouched.
                    for key in (
                        "twinstar_prompt_final",
                        "pd_factor_only_full_batch",
                        "req_pool_indices_cpu",
                    ):
                        setattr(static, key, getattr(forward_batch, key, None))
                padded = int(static.input_ids.shape[0])
                raw = self.raw_num_tokens
                if streams is not None:
                    self.body.streams_in[:raw].copy_(streams)
                    self.body.streams_in[raw:padded].zero_()
                with self._prefill_forward_context(
                    static, num_tokens=padded, raw_num_tokens=raw
                ):
                    out = self.backend.replay(ShapeKey(size=padded), static)
            self.run_count += 1
            return _slice_output_rows(out, raw) if out is not None else None

    return TwinStarPrefillRunner


def capture(owner, model_runner):
    """Build the enabled runners; returns {'trunk': runner|None, 'emitters': runner|None}."""
    from sglang.srt.arg_groups.cuda_graph_hook import (
        generate_prefill_cuda_graph_batch_sizes,
    )
    from sglang.srt.runtime_context import get_schedule

    runners = dict(trunk=None, emitters=None)
    want_trunk, want_emitters = enabled(TRUNK), enabled(EMITTERS)
    if not (want_trunk or want_emitters):
        return runners
    if os.environ.get("SGLANG_QWEN4_PREFILL_GRAPH", "0") != "1":
        raise ValueError(
            "TwinStar prefill graphs need the Qwen4 layer breaks (SGLANG_QWEN4_PREFILL_GRAPH=1)"
        )
    if not hasattr(model_runner, "attention_layers"):
        # Set by the framework only when its own prefill graph is enabled
        # (never for TwinStar arms): the breaks resolve layers through these.
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            index_attention_layers_by_global_id,
        )

        body = owner.model.model
        (
            model_runner.attention_layers,
            model_runner.moe_layers,
            model_runner.moe_fusions,
            model_runner.dsa_indexers,
            model_runner.mha_companion_layers,
        ) = model_runner.get_cuda_graph_layers(body)
        model_runner.attention_layers, model_runner.mha_companion_layers = (
            index_attention_layers_by_global_id(
                model_runner.attention_layers, model_runner.mha_companion_layers, body
            )
        )
    chunk = int(get_schedule().chunked_prefill_size)
    # TWINSTAR_PREFILL_GRAPH_MAX_TOKENS caps the captured buckets: the graph pool is sized by the largest one, and
    # final's eager latent codec needs that headroom at 32K. Larger P sub-batches run the eager trunk.
    limit = int(os.environ.get("TWINSTAR_PREFILL_GRAPH_MAX_TOKENS", chunk))
    buckets = [b for b in generate_prefill_cuda_graph_batch_sizes(chunk) if b <= limit]
    attributes = {}
    if owner.fullstack_v3_latent:
        attributes["flashnext_gdn_layer_range"] = (
            0,
            35 if owner.fullstack_final else 23,
        )
    emit_ids = owner._emit_ids()
    # final: the latent codec sits between the trunk and its (GDN) emitters.
    # PD replays the whole N-token trunk inside its per-layer GDN split. Its
    # emitters consume only N-1 and run after h31 publication/codec processing.
    # A joint graph would send the last prompt token through those emitters.
    joint = (
        want_trunk and want_emitters and not owner.fullstack_code
        and not getattr(owner, "pd_trunk_prefill_graph", False)
    )
    cls = _runner_class()
    plan = []
    if want_trunk:
        plan.append(("trunk", _Body(owner, True, emit_ids if joint else [])))
    if want_emitters and not joint and emit_ids:
        plan.append(("emitters", _Body(owner, False, emit_ids)))
    for name, body in plan:
        started = time.perf_counter()
        before = torch.cuda.mem_get_info()[0]
        runners[name] = cls(model_runner, body, buckets, name, attributes)
        used = (before - torch.cuda.mem_get_info()[0]) / 2**30
        logger.info(
            "TwinStar prefill graph captured %s: emitters=%s buckets=%d max=%d elapsed=%.1f s mem=%.2f GiB",
            name,
            body.emit_ids,
            len(buckets),
            max(buckets),
            time.perf_counter() - started,
            used,
        )
    if joint:
        runners["emitters"] = runners["trunk"]
    return runners
