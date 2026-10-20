"""Target-verify sampling on the vocab-parallel TP LM head under attention DP (SGLANG_Q35MF_LL16_VPSAMP).

The verify LM head writes every gathered row's logits on this rank's vocab shard into symmetric memory. One
launch reduces each row to per-chunk (max, sum-exp) and the argmax and sends that payload to the row's owner,
so no logits all-to-all runs. The owner samples its requests from the payloads (temperature 1, no top-k /
top-p / min-p), reading the draft tokens' logits and the drawn chunk from the shards that hold them; any
other batch pulls its whole rows from the shards (no collective) and takes the stock path. The argmax equals
torch.argmax over the full row; sampled chains follow the distribution of sample_chain_full_vocab.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.speculative.ll16_vp_verify import (
    MAX_D,
    SLOT_MARKER,
    WORDS,
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


def all_temperature_one(reqs) -> bool:
    return all(req.sampling_params.temperature == 1.0 for req in reqs)


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
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.rank = rank
        self.world = world
        self.max_rows = max_rows
        self.vocab_local = vocab_local
        self.vocab_size = vocab_size
        g_max = world * max_rows
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        with torch.inference_mode(False), torch.no_grad():
            self.logits = symm_mem.empty(
                (g_max, vocab_local), dtype=torch.bfloat16, device=device
            )
            self.slots = symm_mem.empty(
                vp_slots_shape(g_max, world), dtype=torch.int32, device=device
            )
        # Every slot word holds the marker before any peer can see the buffer (rendezvous is collective).
        self.slots.fill_(SLOT_MARKER)
        torch.cuda.synchronize()
        logits_hdl = symm_mem.rendezvous(self.logits, group=device_group)
        slots_hdl = symm_mem.rendezvous(self.slots, group=device_group)
        for hdl, buf in ((logits_hdl, self.logits), (slots_hdl, self.slots)):
            if hdl.rank != rank or int(hdl.buffer_ptrs[rank]) != buf.data_ptr():
                raise RuntimeError("LL16 VP: unexpected symmetric memory layout")
        self.peer_logits = torch.tensor(
            [int(p) for p in logits_hdl.buffer_ptrs], dtype=torch.int64, device=device
        )
        self.peer_slots = torch.tensor(
            [int(p) for p in slots_hdl.buffer_ptrs], dtype=torch.int64, device=device
        )
        self.epoch = torch.zeros(g_max, dtype=torch.int32, device=device)
        self.stats = torch.zeros(
            (max_rows, world, WORDS), dtype=torch.int32, device=device
        )
        # [0]: rows per rank of the last exchange (padded verify rows).
        self.meta = torch.zeros(4, dtype=torch.int32, device=device)
        self._begin = self.logits.data_ptr()
        self._end = self._begin + self.logits.numel() * self.logits.element_size()
        self.proof = VpProof() if envs.SGLANG_Q35MF_LL16_VPSAMP_PROOF.get() else None
        self._logged = False

    def owns(self, logits: torch.Tensor) -> bool:
        return (
            logits.dtype == torch.bfloat16
            and logits.dim() == 2
            and logits.shape[-1] == self.vocab_local
            and self._begin <= logits.data_ptr() < self._end
        )

    def fits(self, *, hidden_states: torch.Tensor, rows: int) -> bool:
        return (
            0 < rows <= self.max_rows
            and hidden_states.shape[0] == rows * self.world
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
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
            logits,
            self.slots,
            self.peer_slots,
            self.epoch,
            self.stats,
            self.meta,
            self.rank * self.vocab_local,
            self.rank,
            rows,
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
            or verify_input.draft_token_num > MAX_D
        ):
            return "not a topk-1 chain"
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
            meta=self.meta,
            peer_logits=self.peer_logits,
            candidates=candidates,
            retrieve_index=retrieve_index,
            coins=coins[0] if sample else no_coins,
            coins_final=coins[1] if sample else no_coins,
            predict=predict,
            accept_index=accept_index,
            num_correct_drafts=num_correct_drafts,
            argmax=argmax,
            ld=self.vocab_local,
            vocab_local=self.vocab_local,
            rank=self.rank,
            sample=sample,
        )
        if self.proof is not None:
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
        """This rank's first ``rows`` verify rows over the whole vocab, float32 [rows, vocab_size]."""
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
) -> VpVerify:
    global _STATE
    _STATE = VpVerify(
        device_group=device_group,
        rank=rank,
        world=world,
        max_rows=max_rows,
        vocab_local=vocab_local,
        vocab_size=vocab_size,
    )
    logger.info(
        "LL16 VP verify buffers built (world=%d, rows/rank <= %d, vocab shard=%d).",
        world,
        max_rows,
        vocab_local,
    )
    return _STATE


class VpProof:
    """Every sampled round, pull the full rows and compare against torch.argmax and the one-rank chunked
    sampler with the same coins, eager or in CUDA graphs. Counters live on the device and are mirrored into
    pinned host memory by a copy that is part of the check, so the reporter thread makes no CUDA call."""

    PERIOD_S = 30.0
    # [calls, rows, argmax mismatches, sampled requests, accept-count mismatches, predict-row mismatches]
    FIELDS = (
        "calls",
        "rows",
        "argmax_mismatch",
        "requests",
        "accept_mismatch",
        "predict_mismatch",
    )

    def __init__(self):
        self.counters = torch.zeros(
            (len(self.FIELDS),), dtype=torch.int64, device=torch.cuda.current_device()
        )
        self.host_counters = torch.zeros(
            (len(self.FIELDS),), dtype=torch.int64, device="cpu", pin_memory=True
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
            ref_predict = torch.zeros_like(predict)
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
            self.counters[5].add_(
                torch.count_nonzero((ref_predict != predict).view(bs, d).any(dim=1))
            )
        self.host_counters.copy_(self.counters, non_blocking=True)

    def _report(self) -> None:
        while True:
            time.sleep(self.PERIOD_S)
            logger.info(
                "LL16 VP proof: %s",
                " ".join(
                    f"{k}={v}" for k, v in zip(self.FIELDS, self.host_counters.tolist())
                ),
            )
