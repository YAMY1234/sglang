"""Target-verify sampling on the vocab-parallel TP LM head under attention DP (SGLANG_Q35MF_LL16_VPSAMP).

In a captured verify graph, every rank first sends its rows' next draft tokens to every rank, the LM head writes
every gathered row's logits on this rank's vocab shard, and one launch reduces each row to per-chunk
(max, sum-exp, argmax), the draft token's logit and one token drawn from the shard's residual, sent to the row's
owner. No logits all-to-all runs, no rank reads another rank's logits, and every exchanged word validates itself,
so no system fence is needed. The owner samples its requests from the payloads (temperature 1, no top-k / top-p /
min-p, or greedy): the stock chain acceptance with the stock coins, then a shard by residual mass and that shard's
token. The argmax equals torch.argmax over the full row; sampled chains follow the distribution of
sample_chain_full_vocab. A batch that cannot take this path makes the scheduler run that round without graphs
(see decode_batch_eligible), so captured rounds never need the full rows.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.speculative.ll16_vp_verify import (
    CHUNKS,
    MAX_D,
    SLOT_MARKER,
    WORDS,
    vp_cand,
    vp_decide,
    vp_pull,
    vp_shard,
    vp_slots_shape,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_spec

logger = logging.getLogger(__name__)

_STATE: Optional[VpVerify] = None


def owner_of(logits: Optional[torch.Tensor]) -> Optional[VpVerify]:
    """The exchange whose symmetric shard buffer holds these verify logits, if any."""
    if _STATE is None or logits is None or not _STATE.owns(logits):
        return None
    return _STATE


def built() -> bool:
    return _STATE is not None


def _proof_on() -> bool:
    # The post-verify graph is captured outside the A/B harness's per-variant env, so read the current
    # variant's switch (graphs are keyed by variant).
    from sglang.srt.model_executor.runner_backend import ab_variants

    if ab_variants.enabled():
        return ab_variants.env("SGLANG_Q35MF_LL16_VPSAMP_PROOF", "0") in ("1", "true")
    return envs.SGLANG_Q35MF_LL16_VPSAMP_PROOF.get()


def all_temperature_one(reqs) -> bool:
    return all(req.sampling_params.temperature == 1.0 for req in reqs)


def decode_batch_eligible(batch) -> bool:
    """Whether this decode batch's verify can take the shard exchange; the scheduler's decode graph vote is
    False otherwise, so every captured round is eligible. Same conditions as VpVerify.ineligible_reason."""
    info = batch.sampling_info
    if (
        batch.return_logprob
        or batch.has_grammar
        or envs.SGLANG_SANITIZE_NAN_LOGITS.get()
        or info.logit_bias is not None
        or info.has_custom_logit_processor
        or info.has_watermark_candidates
        or (
            info.penalizer_orchestrator is not None
            and info.penalizer_orchestrator.is_required
        )
    ):
        return False
    if info.is_all_greedy:
        return True
    return (
        not info.need_top_k_sampling
        and not info.need_top_p_sampling
        and not info.need_min_p_sampling
        and all_temperature_one(batch.reqs)
    )


class VpVerify:
    def __init__(
        self,
        *,
        device_group,
        rank: int,
        world: int,
        max_rows: int,
        vocab_local: int,
        vocab_size: int,
        draft_token_num: int,
        seed: int,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        from sglang.srt.model_executor.runner_backend import ab_variants

        self.rank = rank
        self.world = world
        self.max_rows = max_rows
        self.vocab_local = vocab_local
        self.vocab_size = vocab_size
        self.draft_token_num = draft_token_num
        g_max = world * max_rows
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        with torch.inference_mode(False), torch.no_grad():
            self.logits = symm_mem.empty(
                (g_max, vocab_local), dtype=torch.bfloat16, device=device
            )
            self.slots = symm_mem.empty(
                vp_slots_shape(g_max, world), dtype=torch.int32, device=device
            )
            self.cand = symm_mem.empty((2, g_max), dtype=torch.int32, device=device)
        # Every exchanged word holds the marker before any peer can see the buffer (rendezvous is collective).
        self.slots.fill_(SLOT_MARKER)
        self.cand.fill_(SLOT_MARKER)
        torch.cuda.synchronize()
        ptrs = {}
        for name, buf in (
            ("logits", self.logits),
            ("slots", self.slots),
            ("cand", self.cand),
        ):
            hdl = symm_mem.rendezvous(buf, group=device_group)
            if hdl.rank != rank or int(hdl.buffer_ptrs[rank]) != buf.data_ptr():
                raise RuntimeError("LL16 VP: unexpected symmetric memory layout")
            ptrs[name] = torch.tensor(
                [int(p) for p in hdl.buffer_ptrs], dtype=torch.int64, device=device
            )
        # Peers' logits are read only by the proof's check.
        self.peer_logits = ptrs["logits"]
        self.peer_slots = ptrs["slots"]
        self.peer_cand = ptrs["cand"]
        self.epoch = torch.zeros(g_max, dtype=torch.int32, device=device)
        self.counter = torch.zeros(g_max, dtype=torch.int32, device=device)
        self.part = torch.zeros((g_max, CHUNKS, 2), dtype=torch.float32, device=device)
        self.xd = torch.zeros(g_max, dtype=torch.float32, device=device)
        # [seed, draw counter]: the counter advances once per verify round.
        self.rng = torch.tensor(
            [seed & ((1 << 63) - 1), 0], dtype=torch.int64, device=device
        )
        self.stats = torch.zeros(
            (max_rows, world, WORDS), dtype=torch.int32, device=device
        )
        # [0]: rows per rank of the last exchange (padded verify rows).
        self.meta = torch.zeros(4, dtype=torch.int32, device=device)
        self._begin = self.logits.data_ptr()
        self._end = self._begin + self.logits.numel() * self.logits.element_size()
        # Built when the process or any same-server A/B variant sets the proof switch; checks are
        # captured only where the switch is on.
        self.proof = (
            VpProof()
            if ab_variants.env_any("SGLANG_Q35MF_LL16_VPSAMP_PROOF", "1")
            else None
        )
        self._logged = False

    def owns(self, logits: torch.Tensor) -> bool:
        return (
            logits.dtype == torch.bfloat16
            and logits.dim() == 2
            and logits.shape[-1] == self.vocab_local
            and self._begin <= logits.data_ptr() < self._end
        )

    def fits(
        self, *, hidden_states: torch.Tensor, input_ids: torch.Tensor, rows: int
    ) -> bool:
        return (
            0 < rows <= self.max_rows
            and rows % self.draft_token_num == 0
            and hidden_states.shape[0] == rows * self.world
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and input_ids.shape[0] == rows
            and input_ids.dtype == torch.int64
        )

    def push_candidates(self, *, input_ids: torch.Tensor) -> None:
        """Send this rank's verify rows' next draft tokens to every rank (before the LM head GEMM)."""
        vp_cand(
            input_ids=input_ids,
            peer_cand=self.peer_cand,
            epoch=self.epoch,
            rng=self.rng,
            rank=self.rank,
            d=self.draft_token_num,
        )

    def lm_head(
        self, *, hidden_states: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """The stock vocab-parallel LM head GEMM, written into the symmetric shard buffer."""
        out = self.logits[: hidden_states.shape[0]]
        return torch.matmul(hidden_states, weight.T, out=out)

    def exchange(self, *, logits: torch.Tensor, rows: int) -> torch.Tensor:
        """Send every gathered row's payload to its owner; returns this rank's rows of the shard
        logits, the handle the verify sampler recognizes."""
        vp_shard(
            logits=logits,
            slots=self.slots,
            peer_slots=self.peer_slots,
            cand=self.cand,
            epoch=self.epoch,
            counter=self.counter,
            part=self.part,
            xd=self.xd,
            rng=self.rng,
            stats=self.stats,
            meta=self.meta,
            vocab_start=self.rank * self.vocab_local,
            rank=self.rank,
            rows=rows,
            # The proof's check reads peers' logits, so its graphs publish them first.
            fence=self.proof is not None and envs.SGLANG_Q35MF_LL16_VPSAMP_PROOF.get(),
        )
        if not self._logged:
            self._logged = True
            logger.info(
                "LL16 VP verify exchange active (world=%d, rank=%d, rows/rank=%d, vocab shard=%d).",
                self.world,
                self.rank,
                rows,
                self.vocab_local,
            )
        return logits[self.rank * rows : (self.rank + 1) * rows]

    def ineligible_reason(
        self,
        *,
        sampling_info,
        verify_input,
        batch,
        grammar_mask,
        watermark_state,
        use_rejection_sampling: bool,
        uno_target_max_top_k: Optional[int],
        all_t1: bool,
    ) -> Optional[str]:
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

        if (
            verify_input.tree_topk != 1
            or use_rejection_sampling
            or verify_input.draft_token_num != self.draft_token_num
        ):
            return "not the topk-1 chain the exchange was built for"
        spec = get_spec()
        if (
            spec.speculative_accept_threshold_single != 1.0
            or spec.speculative_accept_threshold_acc != 1.0
        ):
            return "acceptance thresholds below 1"
        if grammar_mask is not None or (
            watermark_state is not None and sampling_info.has_watermark_candidates
        ):
            return "grammar or watermark"
        if (
            sampling_info.acc_additive_penalties is not None
            or sampling_info.acc_scaling_penalties is not None
            or sampling_info.logit_bias is not None
            or (
                isinstance(sampling_info, SamplingBatchInfo)
                and sampling_info.has_custom_logit_processor
            )
        ):
            return "penalties, bias or custom processor"
        if batch.return_logprob:
            return "logprobs"
        if envs.SGLANG_SANITIZE_NAN_LOGITS.get():
            return "NaN sanitizing writes the logits"
        if sampling_info.is_all_greedy:
            return None
        if (
            sampling_info.need_top_k_sampling
            or sampling_info.need_top_p_sampling
            or sampling_info.need_min_p_sampling
            or uno_target_max_top_k is not None
        ):
            return "top-k / top-p / min-p"
        if not all_t1:
            return "temperature other than 1"
        return None

    def sample(
        self,
        *,
        candidates: torch.Tensor,
        retrieve_index: torch.Tensor,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        coins: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Argmax per row [bs, d] (int64); with coins, also writes the sampled chain into predict /
        accept_index / num_correct_drafts."""
        bs, d = candidates.shape
        argmax = torch.empty((bs * d,), dtype=torch.int64, device=candidates.device)
        sample = coins is not None
        no_coins = torch.empty((0,), dtype=torch.float32, device=candidates.device)
        vp_decide(
            stats=self.stats,
            candidates=candidates,
            retrieve_index=retrieve_index,
            coins=coins[0] if sample else no_coins,
            coins_final=coins[1] if sample else no_coins,
            predict=predict,
            accept_index=accept_index,
            num_correct_drafts=num_correct_drafts,
            argmax=argmax,
            vocab_local=self.vocab_local,
            sample=sample,
        )
        if self.proof is not None and _proof_on():
            self.proof.check(
                vp=self,
                argmax=argmax,
                candidates=candidates,
                retrieve_index=retrieve_index,
                predict=predict,
                accept_index=accept_index,
                num_correct_drafts=num_correct_drafts,
                coins=coins,
            )
        return argmax.view(bs, d)

    def pull_rows(self, *, rows: int) -> torch.Tensor:
        """This rank's first ``rows`` verify rows over the whole vocab, float32 [rows, vocab_size]: the proof's
        check, after an exchange that ran with the release fence."""
        out = torch.empty(
            (rows, self.vocab_size), dtype=torch.float32, device=self.logits.device
        )
        vp_pull(
            peer_logits=self.peer_logits,
            meta=self.meta,
            out=out,
            ld=self.vocab_local,
            vocab_local=self.vocab_local,
            rank=self.rank,
        )
        return out


def create(
    *,
    device_group,
    rank: int,
    world: int,
    max_rows: int,
    vocab_local: int,
    vocab_size: int,
    draft_token_num: int,
    seed: int,
) -> VpVerify:
    global _STATE
    if draft_token_num > MAX_D:
        raise ValueError(f"LL16 VP: {draft_token_num} draft tokens, at most {MAX_D}")
    _STATE = VpVerify(
        device_group=device_group,
        rank=rank,
        world=world,
        max_rows=max_rows,
        vocab_local=vocab_local,
        vocab_size=vocab_size,
        draft_token_num=draft_token_num,
        seed=seed,
    )
    logger.info(
        "LL16 VP verify buffers built (world=%d, rows/rank <= %d, vocab shard=%d).",
        world,
        max_rows,
        vocab_local,
    )
    return _STATE


class VpProof:
    """Every sampled round, pull the full rows and compare against torch.argmax and, with the same coins, the
    accepted counts of the one-rank chunked sampler. The final token is drawn independently, so it is checked in
    distribution against the full row's residual (fp64 softmax without the rejected draft): zero-mass draws are
    counted, and two randomized PITs are binned, uniform when sampling is exact. Counters live on the device and are
    mirrored into pinned host memory by a copy that is part of the check, so the reporter thread makes no CUDA call."""

    PERIOD_S = 30.0
    FIELDS = (
        "calls",
        "rows",
        "argmax_mismatch",
        "requests",
        "accept_mismatch",
        "final_zero_mass",
    )
    PIT_BINS = 20

    def __init__(self):
        n = len(self.FIELDS) + 2 * self.PIT_BINS
        self.counters = torch.zeros(
            (n,), dtype=torch.int64, device=torch.cuda.current_device()
        )
        self.host_counters = torch.zeros(
            (n,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        logger.info("LL16 VP proof active")
        threading.Thread(target=self._report, daemon=True).start()

    def check(
        self,
        *,
        vp: VpVerify,
        argmax: torch.Tensor,
        candidates: torch.Tensor,
        retrieve_index: torch.Tensor,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        coins: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        from sglang.srt.speculative.sharded_verify_sampling import (
            sample_chain_full_vocab,
        )

        bs, d = candidates.shape
        full = vp.pull_rows(rows=bs * d)
        self.counters[0].add_(1)
        self.counters[1].add_(bs * d)
        self.counters[2].add_(torch.count_nonzero(torch.argmax(full, dim=-1) != argmax))
        if coins is not None:
            ref_predict = torch.zeros((bs * d,), dtype=torch.int32, device=full.device)
            ref_accept = torch.full_like(accept_index, -1)
            ref_num = torch.empty_like(num_correct_drafts)
            sample_chain_full_vocab(
                logits=full,
                temperatures=torch.ones(
                    (bs, 1), dtype=torch.float32, device=full.device
                ),
                candidates=candidates,
                retrieve_index=retrieve_index,
                predict=ref_predict,
                accept_index=ref_accept,
                num_correct_drafts=ref_num,
                coins=coins,
            )
            self.counters[3].add_(bs)
            self.counters[4].add_(torch.count_nonzero(ref_num != num_correct_drafts))
            self._pit(
                full=full,
                candidates=candidates,
                retrieve_index=retrieve_index,
                predict=predict,
                num_correct_drafts=num_correct_drafts,
            )
        self.host_counters.copy_(self.counters, non_blocking=True)

    def _pit(
        self, *, full, candidates, retrieve_index, predict, num_correct_drafts
    ) -> None:
        bs, d = candidates.shape
        k = num_correct_drafts.long()
        rows = torch.arange(bs, device=full.device) * d + k
        p = torch.softmax(full[rows].double(), dim=-1)
        # The rejected draft token (none after a fully accepted chain) is not in the residual.
        excl = candidates.gather(1, (k + 1).clamp(max=d - 1).view(bs, 1))
        keep = (k < d - 1).view(bs, 1)
        p = p.scatter(1, excl, torch.where(keep, 0.0, p.gather(1, excl)))
        p = p / p.sum(dim=-1, keepdim=True)
        last = retrieve_index.gather(1, k.view(bs, 1)).view(bs)
        tok = predict[last].long().view(bs, 1)
        pt = p.gather(1, tok)
        # Token-id order catches shard / chunk selection bias; descending-probability order catches shape bias.
        below_id = (p.cumsum(dim=-1) - p).gather(1, tok)
        above = torch.where(p > pt, p, 0.0).sum(dim=-1, keepdim=True)
        tied = torch.where(p == pt, p, 0.0).sum(dim=-1, keepdim=True)
        r = torch.rand((bs, 2), dtype=torch.float64, device=full.device)
        u = torch.cat([below_id + r[:, :1] * pt, above + r[:, 1:] * tied], dim=1)
        # Rows with non-finite logits (graph padding) are left out.
        ok = torch.isfinite(u).all(dim=1, keepdim=True)
        self.counters[5].add_(torch.count_nonzero(ok & (pt == 0)))
        bins = (torch.where(ok, u, 0.0) * self.PIT_BINS).long().clamp(
            0, self.PIT_BINS - 1
        ) + self.PIT_BINS * torch.arange(2, device=full.device)
        self.counters[len(self.FIELDS) :].index_add_(
            0, bins.view(-1), ok.expand(bs, 2).reshape(-1).long()
        )

    def _report(self) -> None:
        while True:
            time.sleep(self.PERIOD_S)
            vals = self.host_counters.tolist()
            n, b = len(self.FIELDS), self.PIT_BINS
            logger.info(
                "LL16 VP proof: %s pit_id=%s pit_p=%s",
                " ".join(f"{k}={v}" for k, v in zip(self.FIELDS, vals)),
                ",".join(str(v) for v in vals[n : n + b]),
                ",".join(str(v) for v in vals[n + b :]),
            )
