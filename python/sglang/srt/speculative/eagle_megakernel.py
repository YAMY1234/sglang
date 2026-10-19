# SPDX-License-Identifier: Apache-2.0
"""SGLang's side of a megakernel's EAGLE entries (spec.ENTRIES) at the EAGLE workers'
and their graph runners' call sites; megakernels/eagle_binding.py documents the
entries and their structs."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.megakernels.eagle_binding import (
    EagleDraftExtendIO,
    EagleDraftLoopIO,
    EagleDraftRoundIO,
    EagleGdnExtendIO,
    EagleVerifyIO,
    EagleVerifyOutputs,
    GdnExtendTables,
    attn_step_metadata,
    kv_pool_buffers,
    mamba_pool_buffers,
    split_backend,
)
from sglang.srt.megakernels.runtime import megakernel_agree, megakernel_entry
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_spec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# the draft model's entries, and the target's
DRAFT_ENTRIES = ("draft_loop", "draft_extend", "begin_draft_round")
TARGET_ENTRIES = ("verify_sample", "gdn_extend")

# Eager rounds per post-verify key before its graphs are captured.
_NUM_WARMUP = 2
# XORed into the server seed so these draws differ from other samplers'.
_VERIFY_NOISE_DOMAIN = 0x5A17C0DE93E1B2A7
_SIM_ACC_LEN_DOMAIN = 0x0F3D6B8C2A915E47
# Arbitrary bound on post-verify graph sets (key x verify logits buffer); a batch
# past it runs SGLang's eager path.
_MAX_POST_VERIFY_STATES = 64


def forced_length_generator(seed: int) -> torch.Generator:
    """The forced acceptance lengths' generator (SGLANG_SIMULATE_ACC_LEN), seeded
    alike on every rank from the server seed."""
    return torch.Generator().manual_seed(_domain_key(seed, _SIM_ACC_LEN_DOMAIN))


def offers_eagle_entries(*, target_model: Any, draft_model: Any) -> bool:
    """Whether the target's or the draft's megakernel offers an EAGLE entry."""
    return any(
        megakernel_entry(model, name) is not None
        for model, names in (
            (draft_model, DRAFT_ENTRIES),
            (target_model, TARGET_ENTRIES),
        )
        for name in names
    )


def unservable_reasons(worker: Any) -> list[str]:
    """Why the EAGLE entries cannot run on this worker."""
    reasons = []
    if worker.topk != 1:
        reasons.append("the entries run a chain draft (topk 1)")
    if get_spec().speculative_use_rejection_sampling:
        reasons.append("the entries do not run rejection sampling")
    draft = worker.draft_worker
    return reasons + _backend_reasons(
        target=worker.target_worker.model_runner.attn_backend,
        draft=(
            *draft.draft_attn_backend.attn_backends,
            draft.draft_extend_attn_backend or draft.draft_runner.attn_backend,
        ),
    )


class EagleMegakernel:
    """The batch sizes whose captured graphs run each entry, their SGLang-owned
    outputs, where each draft round's input came from, and SGLang's post-verify
    graphs."""

    def __init__(
        self,
        *,
        target_worker: Any,
        draft_worker: Any,
        simulated_acc_len_rng: torch.Generator,
    ):
        self._target_worker = target_worker
        self._draft_worker = draft_worker
        draft_runner = draft_worker.draft_runner
        self._draft_runner = draft_runner
        self._draft_config = draft_runner.model_config
        self._draft_loop = megakernel_entry(draft_runner.model, "draft_loop")
        self._draft_extend = megakernel_entry(draft_runner.model, "draft_extend")
        self._begin_draft_round = megakernel_entry(
            draft_runner.model, "begin_draft_round"
        )
        # runner id -> what bind_*_graphs set up: bs -> step, and the outputs
        self._draft_graphs: dict[int, SimpleNamespace] = {}
        self._extend_graphs: dict[int, SimpleNamespace] = {}
        self._draft_kv = None
        # draft inputs produced so far, and (that count, rids) of the last one a
        # served draft-extend replay produced
        self._input_seq = 0
        self._served_input: Optional[tuple] = None
        self._verify_sample = megakernel_entry(
            target_worker.model_runner.model, "verify_sample"
        )
        self._gdn_extend = megakernel_entry(
            target_worker.model_runner.model, "gdn_extend"
        )
        self._mamba = None
        # bs -> verify_sample's step, from bind_verify_sample
        self._verify_steps: dict[int, Callable] = {}
        # the server seed is the same on every rank
        seed = target_worker.random_seed
        self._noise_key = _domain_key(seed, _VERIFY_NOISE_DOMAIN)
        self._noise_offset = torch.zeros(
            1, dtype=torch.int64, device=target_worker.device
        )
        self._post_verify = _PostVerifyGraphs(
            body=self._post_verify_body,
            commit=self._post_verify_commit,
            sim_rng=simulated_acc_len_rng,
        )

    # ----- the EAGLE draft and draft-extend graphs -----

    def bind_draft_graphs(self, runner: Any) -> frozenset[int]:
        """Before the runner captures: calls the draft's draft_loop factory once with
        the runner's outputs, and returns the batch sizes whose graphs run it."""
        if self._draft_loop is None:
            return frozenset()
        draft_tokens = torch.zeros(
            (runner.max_bs, runner.speculative_num_steps),
            dtype=torch.int64,
            device=runner.device,
        )
        steps = self._draft_loop(
            num_steps=runner.speculative_num_steps,
            max_bs=runner.max_bs,
            draft_tokens=draft_tokens,
        )
        served = _served(steps, runner.capture_bs, what="draft_loop batch sizes")
        if served:
            self._draft_graphs[id(runner)] = SimpleNamespace(
                steps=served, draft_tokens=draft_tokens
            )
        return frozenset(served)

    def run_draft_loop(self, *, runner: Any, forward_batch: ForwardBatch) -> tuple:
        """Inside the draft graph's capture: draft_loop's step in place of
        draft_forward, with the same outputs."""
        bs = int(forward_batch.batch_size)
        graphs = self._draft_graphs[id(runner)]
        draft_tokens = graphs.draft_tokens[:bs]
        spec_info = forward_batch.spec_info
        graphs.steps[bs](
            EagleDraftLoopIO(
                num_requests=bs,
                num_steps=runner.speculative_num_steps,
                topk_index=spec_info.topk_index,
                hidden_states=spec_info.hidden_states,
                positions=forward_batch.positions,
                mrope_positions=(
                    forward_batch.mrope_positions
                    if self._draft_config.model_is_mrope
                    else None
                ),
                out_cache_loc=forward_batch.out_cache_loc,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                steps=tuple(
                    map(attn_step_metadata, runner.draft_attn_backend.attn_backends)
                ),
                kv=self._kv(),
                draft_tokens_out=draft_tokens,
            )
        )
        worker = self._draft_worker
        # a topk-1 chain's tree is constant: the worker's preallocated parents
        return (
            worker._topk1_parents_prealloc[:bs],
            worker._topk1_score_indices_prealloc[:bs],
            draft_tokens,
            None,
        )

    def bind_extend_graphs(self, runner: Any) -> frozenset[int]:
        """Before the runner captures: calls the draft's draft_extend factory once
        with the runner's outputs, and returns the batch sizes whose graphs run it."""
        if self._draft_extend is None:
            return frozenset()
        out = SimpleNamespace(
            topk_index=torch.zeros(
                (runner.max_bs, 1), dtype=torch.int64, device=runner.device
            ),
            topk_p=torch.ones(
                (runner.max_bs, 1), dtype=torch.float32, device=runner.device
            ),
            hidden=torch.zeros(
                (runner.max_bs, self._draft_config.hidden_size),
                dtype=self._draft_config.dtype,
                device=runner.device,
            ),
        )
        steps = self._draft_extend(
            tokens_per_request=runner.captured_req_width,
            max_bs=runner.max_bs,
            topk_index=out.topk_index,
            topk_p=out.topk_p,
            hidden=out.hidden,
        )
        served = _served(steps, runner.capture_bs, what="draft_extend batch sizes")
        if served:
            self._extend_graphs[id(runner)] = SimpleNamespace(steps=served, out=out)
        return frozenset(served)

    def run_draft_extend(self, *, runner: Any, forward_batch: ForwardBatch) -> Any:
        """Inside the draft-extend graph's capture: draft_extend's step in place of
        the model's forward; the logits stay inside the megakernel."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        bs = int(forward_batch.batch_size)
        graphs = self._extend_graphs[id(runner)]
        out = graphs.out
        spec_info = forward_batch.spec_info
        graphs.steps[bs](
            EagleDraftExtendIO(
                num_requests=bs,
                tokens_per_request=runner.captured_req_width,
                input_ids=forward_batch.input_ids,
                positions=forward_batch.positions,
                mrope_positions=(
                    forward_batch.mrope_positions
                    if self._draft_config.model_is_mrope
                    else None
                ),
                hidden_states=spec_info.hidden_states,
                out_cache_loc=forward_batch.out_cache_loc,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                select_index=spec_info.select_index,
                num_accept_tokens=spec_info.num_accept_tokens,
                attn=attn_step_metadata(get_attn_backend()),
                kv=self._kv(),
                topk_index_out=out.topk_index[:bs],
                topk_p_out=out.topk_p[:bs],
                hidden_out=out.hidden[:bs],
            )
        )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=out.hidden[:bs]
        )

    # ----- the draft worker's host calls -----

    def note_draft_extend(self) -> None:
        """A draft extend (after a prefill or a verify) produces the next draft input."""
        self._input_seq += 1

    def extend_tokens(
        self, *, runner: Any, batch: ScheduleBatch
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """After a draft-extend replay: (topk_p, topk_index) of a graph draft_extend
        ran, else None for SGLang's top-1."""
        graphs = self._extend_graphs.get(id(runner))
        if graphs is None or runner.bs not in graphs.steps:
            return None
        self._served_input = (self._input_seq, _rids(batch))
        raw_bs = runner.raw_bs
        # the next replay rewrites the buffers before the next round reads them
        return (
            graphs.out.topk_p[:raw_bs].clone(),
            graphs.out.topk_index[:raw_bs].clone(),
        )

    def begin_draft_round(
        self, *, runner: Any, batch: ScheduleBatch, draft_input: Any
    ) -> None:
        """Before a draft graph replays; tells a served batch size's draft_loop where
        its input came from."""
        from_served_extend = self._served_input == (self._input_seq, _rids(batch))
        self._served_input = None
        graphs = self._draft_graphs.get(id(runner))
        bs = len(batch.seq_lens)
        if (
            self._begin_draft_round is None
            or graphs is None
            or _bucket(bs, runner.capture_bs) not in graphs.steps
        ):
            return
        self._begin_draft_round(
            EagleDraftRoundIO(
                num_requests=bs,
                from_served_extend=from_served_extend,
                topk_index=draft_input.topk_index,
                hidden_states=draft_input.hidden_states,
            )
        )

    # ----- the target's GDN extend in captured prefill buckets -----

    def gdn_extend_entry(self) -> Optional[Callable[..., None]]:
        """The GDN backend's megakernel_extend: the target's gdn_extend, or None for
        SGLang's extend in the captured buckets."""
        return None if self._gdn_extend is None else self._run_gdn_extend

    def _run_gdn_extend(
        self,
        *,
        layer_id: int,
        meta: Any,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if self._mamba is None:
            # the target's pool, allocated before any prefill graph is captured
            self._mamba = mamba_pool_buffers(
                self._target_worker.model_runner.req_to_token_pool
            )
        self._gdn_extend(
            EagleGdnExtendIO(
                layer_id=layer_id,
                tables=_gdn_extend_tables(meta),
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                out=out,
                mamba=self._mamba,
            )
        )

    # ----- the verify post-section -----

    def bind_verify_sample(self, *, max_bs: int, tokens_per_request: int) -> None:
        """Once the target graphs are captured: calls the target's verify_sample
        factory once; the batch sizes up to max_bs whose post-verify graphs run it."""
        if self._verify_sample is None:
            return
        steps = self._verify_sample(
            tokens_per_request=tokens_per_request,
            max_bs=max_bs,
            noise_key=self._noise_key,
            noise_offset=self._noise_offset,
        )
        self._verify_steps = _served(
            steps, range(1, max_bs + 1), what="verify_sample batch sizes"
        )

    def verify_post(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: Any,
        logits_output: Any,
        can_run_cuda_graph: bool,
    ) -> Optional[tuple]:
        """(predict, accept_lens, accept_index, new_seq_lens, bonus_tokens,
        draft-extend inputs) from the post-verify graphs, or None when SGLang's
        sampling and commit run eagerly."""
        self.join()
        key = self._post_verify_key(
            batch, verify_input, logits_output, can_run_cuda_graph
        )
        if key is None:
            return None
        return self._post_verify.run(
            key=key,
            inputs=_post_verify_inputs(batch, verify_input),
            make_slot=lambda staged: self._post_verify_slot(
                staged, batch, verify_input, logits_output
            ),
            sim_max_len=verify_input.max_tree_depth,
        )

    def join(self) -> None:
        """Order the current stream after the last side-stream GDN state commit; the
        worker calls it once the draft extend is enqueued."""
        self._post_verify.join()

    def _post_verify_key(
        self,
        batch: ScheduleBatch,
        verify_input: Any,
        logits_output: Any,
        can_run_cuda_graph: bool,
    ) -> Optional[tuple]:
        from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN

        info = batch.sampling_info
        watermark = self._target_worker.model_runner.watermark_state
        if (
            not can_run_cuda_graph
            or batch.forward_mode.is_idle()
            or verify_input.topk != 1
            or batch.has_grammar
            or batch.hisparse_coordinator is not None
            or verify_input.draft_probs is not None
            or info.acc_additive_penalties is not None
            or info.acc_scaling_penalties is not None
            or info.logit_bias is not None
            or info.sampling_seed is not None
            or info.need_min_p_sampling
            or (watermark is not None and info.has_watermark_candidates)
        ):
            return None
        return (
            len(batch.seq_lens),
            bool(info.is_all_greedy),
            bool(info.need_top_k_sampling),
            bool(info.need_top_p_sampling),
            _compact_max_top_k(batch),
            SIMULATE_ACC_LEN > 0,
            batch.mamba_track_indices is not None,
            verify_input.draft_token_num,
            verify_input.max_tree_depth,
            # the graphs read the verify logits in place; a runner may own several
            logits_output.next_token_logits.data_ptr(),
        )

    def _post_verify_slot(
        self,
        staged: dict[str, torch.Tensor],
        batch: ScheduleBatch,
        verify_input: Any,
        logits_output: Any,
    ) -> SimpleNamespace:
        from sglang.srt.runtime_context import mamba_track_grid
        from sglang.srt.speculative.spec_utils import SIMULATE_ACC_TOKEN_MODE

        info = batch.sampling_info
        bs = len(batch.seq_lens)
        logits = logits_output.next_token_logits
        io = EagleVerifyIO(
            num_requests=bs,
            tokens_per_request=verify_input.draft_token_num,
            max_tree_depth=verify_input.max_tree_depth,
            is_all_greedy=bool(info.is_all_greedy),
            need_top_k=bool(info.need_top_k_sampling),
            need_top_p=bool(info.need_top_p_sampling),
            sim_real_draft_tokens=SIMULATE_ACC_TOKEN_MODE == "real-draft-token",
            logits=logits,
            draft_tokens=staged["draft_tokens"],
            retrieve_index=staged["retrieve_index"],
            retrieve_next_token=staged["retrieve_next_token"],
            retrieve_next_sibling=staged["retrieve_next_sibling"],
            seq_lens=staged["seq_lens"],
            req_pool_indices=staged["req_pool_indices"],
            temperatures=staged.get("temperatures"),
            top_ks=staged.get("top_ks"),
            top_ps=staged.get("top_ps"),
            max_top_k=_compact_max_top_k(batch),
            sim_accept_len=staged.get("sim_accept_len"),
            track_indices=staged.get("track_indices"),
            noise_key=self._noise_key,
            noise_offset=self._noise_offset,
        )
        step, out = self._verify_steps.get(bs), None
        if step is not None:
            out = EagleVerifyOutputs(
                predict=torch.zeros(
                    (logits.shape[0],), dtype=torch.int32, device=logits.device
                ),
                accept_lens=torch.zeros((bs,), dtype=torch.int32, device=logits.device),
                accept_index=torch.full(
                    (bs, verify_input.max_tree_depth),
                    -1,
                    dtype=torch.int32,
                    device=logits.device,
                ),
            )
        return SimpleNamespace(
            io=io,
            step=step,
            out=out,
            device=batch.device,
            forward_mode=batch.forward_mode,
            tree_cache=batch.tree_cache,
            track_interval=(
                mamba_track_grid(batch.tree_cache.page_size)
                if batch.mamba_track_indices is not None
                else 0
            ),
        )

    def _post_verify_body(self, slot: SimpleNamespace) -> tuple:
        """Sampling and acceptance (verify_sample's step, else SGLang's), then the
        commit outputs; recorded in SGLang's post-verify graph."""
        from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs

        io = slot.io
        if slot.out is not None:
            slot.out.predict.zero_()
            slot.out.accept_index.fill_(-1)
            slot.step(io, slot.out)
            predict, accept_lens, accept_index = (
                slot.out.predict,
                slot.out.accept_lens,
                slot.out.accept_index,
            )
        else:
            predict, accept_lens, accept_index = _stock_sample(slot)
        new_seq_lens, bonus_tokens, commit_steps, draft_extend_inputs = (
            prepare_verify_commit_outputs(
                predict,
                accept_index,
                accept_lens,
                io.seq_lens,
                num_draft_tokens=io.tokens_per_request,
                mamba_track_interval=slot.track_interval,
            )
        )
        return (
            predict,
            accept_lens,
            accept_index,
            new_seq_lens,
            bonus_tokens,
            draft_extend_inputs,
            commit_steps,
        )

    def _post_verify_commit(self, slot: SimpleNamespace, outputs: tuple) -> None:
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        # the batch fields the commit reads, on the slot's staged copies
        batch_view = SimpleNamespace(
            forward_mode=slot.forward_mode,
            seq_lens=slot.io.seq_lens,
            req_pool_indices=slot.io.req_pool_indices,
            mamba_track_indices=slot.io.track_indices,
            tree_cache=slot.tree_cache,
        )
        commit_mamba_states_after_verify(
            self._target_worker,
            batch_view,
            outputs[1],
            outputs[2],
            slot.io.tokens_per_request,
            prepared_step_indices=outputs[6],
        )

    def _kv(self):
        # the draft's pool, allocated before any draft graph is captured
        if self._draft_kv is None:
            runner = self._draft_runner
            self._draft_kv = kv_pool_buffers(
                pool=runner.token_to_kv_pool,
                backend=split_backend(runner.attn_backend)[0],
                model=runner.model,
            )
        return self._draft_kv


def _stock_sample(slot: SimpleNamespace) -> tuple:
    """SGLang's eagle_sample over the slot's staged copies, with the injected
    acceptance length read on the device."""
    from sglang.srt.speculative.eagle_utils import eagle_sample

    io = slot.io
    sampling_info = SimpleNamespace(
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        has_watermark_candidates=False,
        is_all_greedy=io.is_all_greedy,
        need_top_k_sampling=io.need_top_k,
        need_top_p_sampling=io.need_top_p,
        need_min_p_sampling=False,
        temperatures=io.temperatures,
        top_ks=io.top_ks,
        top_ps=io.top_ps,
    )
    batch_view = SimpleNamespace(
        device=slot.device,
        forward_mode=slot.forward_mode,
        seq_lens=io.seq_lens,
        req_pool_indices=io.req_pool_indices,
        sampling_info=sampling_info,
        return_logprob=False,
    )
    verify_view = SimpleNamespace(
        draft_token=io.draft_tokens,
        draft_token_num=io.tokens_per_request,
        max_tree_depth=io.max_tree_depth,
        retrieve_index=io.retrieve_index,
        retrieve_next_token=io.retrieve_next_token,
        retrieve_next_sibling=io.retrieve_next_sibling,
        tree_topk=1,
        draft_probs=None,
    )
    return eagle_sample(
        verify_view,
        batch_view,
        SimpleNamespace(next_token_logits=io.logits),
        uno_target_max_top_k=io.max_top_k,
        simulated_acc_len=io.sim_accept_len,
        sparse_bonus_argmax=io.max_top_k is not None,
        pack_broadcast=True,
    )


class _PostVerifyGraphs:
    """SGLang-owned post-verify graphs: per key, two slots of fixed staging buffers
    and outputs alternate, so one round's outputs stay valid while the host consumes
    them; the GDN state commit replays on a side stream until join()."""

    def __init__(self, *, body: Callable, commit: Callable, sim_rng: torch.Generator):
        self._body, self._commit = body, commit
        # forced acceptance lengths: the worker's one stream, from which SGLang's eager
        # verify draws too; once per round, whatever else uses the global RNG
        self._sim_rng = sim_rng
        self._states: dict[tuple, SimpleNamespace] = {}
        self._pool = None
        self._side_stream = self._main_done = self._commit_done = None
        self._commit_pending = False
        # The host runs up to ~2 rounds ahead; a pinned slot is rewritten only after
        # the event behind its last copy has completed.
        self._sim_host: list[torch.Tensor] = []
        self._sim_done: list[Optional[torch.cuda.Event]] = []
        self._sim_turn = 0

    def run(
        self, *, key: tuple, inputs: dict, make_slot: Callable, sim_max_len: int
    ) -> Optional[tuple]:
        state = self._states.get(key)
        if state is None:
            if len(self._states) >= _MAX_POST_VERIFY_STATES:
                return None
            state = self._states[key] = SimpleNamespace(
                warmups=0, turn=0, slots=[None, None]
            )
        turn, state.turn = state.turn, state.turn ^ 1
        sim_len, sim_turn = self._draw_sim_len(sim_max_len)
        slot = state.slots[turn]
        if slot is None:
            staged = {name: src.clone() for name, src in inputs.items()}
            if sim_len is not None:
                staged["sim_accept_len"] = sim_len.to(
                    inputs["seq_lens"].device, copy=True
                )
            slot = state.slots[turn] = make_slot(staged)
            slot.staged, slot.graph, slot.commit_graph = staged, None, None
        self._load(slot.staged, inputs)
        if sim_len is not None:
            slot.staged["sim_accept_len"].copy_(sim_len, non_blocking=True)
            self._record_sim_done(sim_turn)
        if slot.graph is None:
            if state.warmups < _NUM_WARMUP * 2:
                state.warmups += 1
                outputs = self._body(slot)
                self._commit(slot, outputs)
                return outputs[:6]
            self._capture(slot)
        slot.graph.replay()
        self._main_done.record()
        self._side_stream.wait_event(self._main_done)
        with torch.cuda.stream(self._side_stream):
            slot.commit_graph.replay()
        self._commit_done.record(self._side_stream)
        self._commit_pending = True
        return slot.outputs[:6]

    def join(self) -> None:
        if self._commit_pending:
            torch.cuda.current_stream().wait_event(self._commit_done)
            self._commit_pending = False

    @staticmethod
    def _load(staged: dict, inputs: dict) -> None:
        # One multi-tensor copy per dtype; one copy per input costs ~4 us of GPU time.
        groups: dict = {}
        for name, src in inputs.items():
            dsts, srcs = groups.setdefault(src.dtype, ([], []))
            dsts.append(staged[name])
            srcs.append(src)
        for dsts, srcs in groups.values():
            torch._foreach_copy_(dsts, srcs, non_blocking=True)

    def _draw_sim_len(
        self, max_len: int
    ) -> tuple[Optional[torch.Tensor], Optional[int]]:
        from sglang.srt.speculative.spec_utils import (
            SIMULATE_ACC_LEN,
            SIMULATE_ACC_METHOD,
            sample_simulated_acc_len,
        )

        if SIMULATE_ACC_LEN <= 0:
            return None, None
        if not self._sim_host:
            self._sim_host = [
                torch.zeros((1,), dtype=torch.int64, pin_memory=True) for _ in range(4)
            ]
            self._sim_done = [None] * len(self._sim_host)
        turn = self._sim_turn
        self._sim_turn = (turn + 1) % len(self._sim_host)
        done = self._sim_done[turn]
        if done is not None and not done.query():
            done.synchronize()
        host = self._sim_host[turn]
        host[0] = sample_simulated_acc_len(
            SIMULATE_ACC_LEN, SIMULATE_ACC_METHOD, max_len, generator=self._sim_rng
        )
        return host, turn

    def _record_sim_done(self, turn: int) -> None:
        if self._sim_done[turn] is None:
            self._sim_done[turn] = torch.cuda.Event()
        self._sim_done[turn].record()

    def _capture(self, slot: SimpleNamespace) -> None:
        from sglang.srt.distributed.parallel_state import graph_capture

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._side_stream = torch.cuda.Stream()
            self._main_done, self._commit_done = torch.cuda.Event(), torch.cuda.Event()
        graph = torch.cuda.CUDAGraph()
        with graph_capture(stream=torch.cuda.current_stream()) as ctx:
            with torch.cuda.graph(
                graph,
                pool=self._pool,
                stream=ctx.stream,
                capture_error_mode="thread_local",
            ):
                outputs = self._body(slot)
        # Captured after the main graph in the same pool and replayed after it.
        commit_graph = torch.cuda.CUDAGraph()
        self._side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(
            commit_graph,
            pool=self._pool,
            stream=self._side_stream,
            capture_error_mode="thread_local",
        ):
            self._commit(slot, outputs)
        torch.cuda.current_stream().wait_stream(self._side_stream)
        slot.graph, slot.commit_graph, slot.outputs = graph, commit_graph, outputs
        logger.info(
            "NVIDIA megakernel: captured the EAGLE post-verify graphs for batch size %d",
            slot.io.num_requests,
        )


def _post_verify_inputs(
    batch: ScheduleBatch, verify_input: Any
) -> dict[str, torch.Tensor]:
    """The round's inputs the post-verify graphs read, staged into fixed buffers."""
    info = batch.sampling_info
    inputs = {
        "draft_tokens": verify_input.draft_token,
        "retrieve_index": verify_input.retrieve_index,
        "retrieve_next_token": verify_input.retrieve_next_token,
        "retrieve_next_sibling": verify_input.retrieve_next_sibling,
        "seq_lens": batch.seq_lens,
        "req_pool_indices": batch.req_pool_indices,
    }
    if not info.is_all_greedy:
        inputs.update(
            temperatures=info.temperatures, top_ks=info.top_ks, top_ps=info.top_ps
        )
    if batch.mamba_track_indices is not None:
        inputs["track_indices"] = batch.mamba_track_indices
    return inputs


def _compact_max_top_k(batch: ScheduleBatch) -> Optional[int]:
    """The largest top-k when every request samples with a finite one within the
    compact top-k branch's limit, else None (the dense branch)."""
    top_ks = [req.sampling_params.top_k for req in batch.reqs]
    if not top_ks or min(top_ks) <= 1:
        return None
    from sglang.srt.speculative.uno_utils import _SPARSE_TOP_K_LIMIT

    return max(top_ks) if max(top_ks) <= _SPARSE_TOP_K_LIMIT else None


def _gdn_extend_tables(meta: Any) -> GdnExtendTables:
    """A GDNPrefillGraphMetadata's tables, without its host staging."""
    return GdnExtendTables(
        num_tokens=meta.num_tokens,
        max_seqs=meta.max_seqs,
        max_chunks=meta.max_chunks,
        max_conv_blocks=meta.max_conv_blocks,
        track=meta.track,
        cu_seqlens=meta.cu_seqlens,
        chunk_indices=meta.chunk_indices,
        chunk_offsets=meta.chunk_offsets,
        conv_block_table=meta.conv_block_table,
        track_mode=meta.track_mode,
        track_h_row=meta.track_h_row,
        track_conv_start=meta.track_conv_start,
        num_logical_tokens=meta.num_logical_tokens,
        cache_indices=meta.cache_indices,
        has_initial_state=meta.has_initial_state,
        track_dst=meta.track_dst,
    )


def _domain_key(seed: int, domain: int) -> int:
    return (int(seed) ^ domain) & 0x7FFFFFFFFFFFFFFF


def _served(
    steps: Callable[[int], Optional[Callable]], sizes: list[int], *, what: str
) -> dict[int, Callable]:
    """bs -> step of the sizes a factory's result serves, alike on every rank."""
    served = {bs: steps(bs) for bs in sorted(sizes)}
    served = {bs: step for bs, step in served.items() if step is not None}
    # a rank on SGLang's code at a size would wait in collectives the step skips
    megakernel_agree(tuple(served), what=what)
    return served


def _backend_reasons(*, target: Any, draft: tuple) -> list[str]:
    """Why these backends' step metadata cannot be handed over."""
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    reasons = []
    if (
        not isinstance(target, HybridLinearAttnBackend)
        or not isinstance(target.full_attn_backend, TRTLLMHAAttnBackend)
        or not isinstance(target.linear_attn_backend, GDNAttnBackend)
    ):
        reasons.append(
            "the target must run GDN and trtllm_mha under a hybrid linear-attention "
            f"backend, not {type(target).__name__}"
        )
    elif target.full_attn_backend.use_sliding_window_kv_pool:
        # every full-attention layer writes its KV at out_cache_loc
        reasons.append("the target's full attention must not use a sliding-window pool")
    for backend in draft:
        backend = split_backend(backend)[0]
        if not isinstance(backend, TRTLLMHAAttnBackend):
            reasons.append(
                f"the draft must run trtllm_mha, not {type(backend).__name__}"
            )
    return reasons


def _rids(batch: ScheduleBatch) -> tuple[str, ...]:
    return tuple(req.rid for req in batch.reqs)


def _bucket(bs: int, capture_bs: list[int]) -> int:
    """The captured batch size a batch of bs requests replays."""
    return min((size for size in capture_bs if size >= bs), default=-1)
