"""Greedy (topk=1) MTP drafting over a fixed subset of the vocabulary.

The draft scores only the listed rows of the shared bf16 lm_head, split evenly
across TP ranks; logits stay in subset order and every proposal is mapped back to
its vocabulary id where it is selected, so verify, the next draft step and a peer
drafting over the full vocabulary all see vocabulary ids."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def load_hot_vocab(path: str, vocab_size: int, tp_size: int) -> torch.Tensor:
    """Sorted unique int64 vocabulary ids from a ``torch.save``-d tensor or list."""
    ids = torch.as_tensor(torch.load(path, weights_only=True), dtype=torch.int64)
    ids = ids.flatten().cpu()
    if ids.numel() == 0 or ids.numel() % tp_size != 0:
        raise ValueError(
            f"hot vocabulary {path}: {ids.numel()} ids, need a multiple of tp_size={tp_size}"
        )
    if ids.numel() > 1 and not bool((ids[1:] > ids[:-1]).all()):
        raise ValueError(f"hot vocabulary {path}: ids must be sorted and unique")
    if int(ids[0]) < 0 or int(ids[-1]) >= vocab_size:
        raise ValueError(f"hot vocabulary {path}: ids outside [0, {vocab_size})")
    return ids


def build_hot_head_shard(
    *,
    weight: torch.Tensor,
    org_vocab_start: int,
    org_vocab_end: int,
    hot_ids: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """This rank's contiguous slice of the hot rows of a vocab-parallel weight."""
    rows = hot_ids.numel() // tp_size
    own = None
    for dst in range(tp_size):
        ids = hot_ids[dst * rows : (dst + 1) * rows].to(weight.device)
        buf = torch.zeros((rows, weight.shape[1]), dtype=weight.dtype, device=weight.device)
        pos = ((ids >= org_vocab_start) & (ids < org_vocab_end)).nonzero().squeeze(1)
        buf[pos] = weight[ids[pos] - org_vocab_start]
        if tp_size > 1:
            # One owner per row and integer zeros elsewhere: the sum keeps the bits.
            dist.all_reduce(buf.view(torch.int32), group=group)
        if dst == tp_rank:
            own = buf
        else:
            del buf
    return own


class HotVocabHead(nn.Module):
    """Plain bf16 head over this rank's hot rows (no quant method, no padding)."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)


def _unsupported_reason(model) -> Optional[str]:
    from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP

    if not isinstance(model, Qwen3_5ForCausalLMMTP):
        return f"draft model {type(model).__name__} is not the Qwen3.5 MTP head"
    if model.config.tie_word_embeddings:
        return "tied embeddings"
    parallel = get_parallel()
    if parallel.attn_dp_size != 1:
        return "data-parallel attention"
    lp = model.logits_processor
    if parallel.tp_size > 1 and (
        not lp.do_tensor_parallel_all_gather or lp.do_tensor_parallel_all_gather_dp_attn
    ):
        return "logits are not tp-gathered"
    if lp.use_attn_tp_group or lp.use_fp32_lm_head:
        return "attention-tp logits group or fp32 lm_head"
    if lp.logit_scale is not None or lp.final_logit_softcapping is not None:
        return "logit scaling or softcapping"
    head = model.lm_head
    if should_apply_lm_head_quant_method(head, getattr(head, "quant_method", None)):
        return "quantized lm_head"
    if head.weight.dtype != torch.bfloat16:
        return f"lm_head dtype {head.weight.dtype}"
    return None


class DraftHotVocab:
    """Owns the hot ids (device int64, sorted) and the draft's hot head."""

    def __init__(self, model, hot_ids: torch.Tensor) -> None:
        parallel = get_parallel()
        head = model.lm_head
        shard = head.shard_indices
        weight = build_hot_head_shard(
            weight=head.weight.data,
            org_vocab_start=shard.org_vocab_start_index,
            org_vocab_end=shard.org_vocab_end_index,
            hot_ids=hot_ids,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
            group=parallel.tp_group.device_group if parallel.tp_size > 1 else None,
        )
        self.vocab_size = model.config.vocab_size
        self.token_ids = hot_ids.to(weight.device)
        self.head = HotVocabHead(weight)
        # The target keeps its own module; only the draft's attribute is rebound.
        model.lm_head = self.head

    @classmethod
    def maybe_create(cls, model, *, path: Optional[str], topk: int) -> Optional["DraftHotVocab"]:
        if not path:
            return None
        reason = "topk != 1" if topk != 1 else _unsupported_reason(model)
        if reason is not None:
            logger.warning("Draft hot vocabulary disabled (%s); using the full lm_head.", reason)
            return None
        parallel = get_parallel()
        hot_ids = load_hot_vocab(path, model.config.vocab_size, parallel.tp_size)
        hv = cls(model, hot_ids)
        logger.info(
            "Draft hot vocabulary active: %d of %d ids (%d rows/rank, tp=%d) from %s.",
            hot_ids.numel(),
            model.config.vocab_size,
            hv.head.weight.shape[0],
            parallel.tp_size,
            path,
        )
        return hv
