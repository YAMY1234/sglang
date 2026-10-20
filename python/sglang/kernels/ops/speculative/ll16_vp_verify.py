from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Must match ll16_vp::kChunks / kWords / kMaxD / kMaxChunk / kMarker / kNoToken in ll16_vp_verify.cuh.
CHUNKS = 4
WORDS = (CHUNKS + 1) * 4
MAX_D = 16
MAX_CHUNK = 4096
SLOT_MARKER = -1
NO_TOKEN = 0x7FFFFFFF


@cache_once
def _module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "ll16_vp_verify",
        *args,
        cuda_files=["speculative/ll16_vp_verify.cuh"],
        cuda_wrappers=[
            ("cand", f"Ll16VpVerify<{args}>::cand"),
            ("shard", f"Ll16VpVerify<{args}>::shard"),
            ("decide", f"Ll16VpVerify<{args}>::decide"),
            ("pull", f"Ll16VpVerify<{args}>::pull"),
        ],
    )


def _no_trace(like: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.int64, device=like.device)


def vp_chunk(vocab_local: int) -> int:
    """Elements per chunk: the shard row in CHUNKS chunks of 16-byte vectors."""
    chunk = -(-vocab_local // CHUNKS)
    return chunk + (-chunk) % 8


def vp_slots_shape(max_gathered_rows: int, world: int) -> tuple:
    """[2 halves, gathered rows, sender rank, WORDS] int32: one payload per sender."""
    return (2, max_gathered_rows, world, WORDS)


def vp_cand(
    *,
    input_ids: torch.Tensor,
    peer_cand: torch.Tensor,
    epoch: torch.Tensor,
    rng: torch.Tensor,
    rank: int,
    d: int,
) -> None:
    """Send each of this rank's verify rows' next draft token (``input_ids`` [rows], d per request) to every
    rank's candidate words, and advance the draw counter ``rng`` [seed, counter]."""
    _module().cand(input_ids, peer_cand, epoch, rng, rank, d)


def vp_shard(
    *,
    logits: torch.Tensor,
    slots: torch.Tensor,
    peer_slots: torch.Tensor,
    cand: torch.Tensor,
    epoch: torch.Tensor,
    counter: torch.Tensor,
    part: torch.Tensor,
    xd: torch.Tensor,
    rng: torch.Tensor,
    stats: torch.Tensor,
    meta: torch.Tensor,
    vocab_start: int,
    rank: int,
    rows: int,
    fence: bool,
    trace: Optional[torch.Tensor] = None,
) -> None:
    """Reduce this rank's shard ``logits`` [world * rows, V_local] of every gathered row (rank-major), draw one
    token per row from the shard's residual and exchange the payloads; this rank's rows land in ``stats``
    [rows, world, WORDS]. ``trace`` (int64 [rows * world, 8], T1 only) gets globaltimer phase stamps."""
    _module().shard(
        logits,
        slots,
        peer_slots,
        cand,
        epoch,
        counter,
        part,
        xd,
        rng,
        stats,
        meta,
        _no_trace(logits) if trace is None else trace,
        vp_chunk(logits.shape[1]),
        vocab_start,
        rank,
        rows,
        int(fence),
    )


def vp_decide(
    *,
    stats: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    coins: torch.Tensor,
    coins_final: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    argmax: torch.Tensor,
    vocab_local: int,
    sample: bool,
    trace: Optional[torch.Tensor] = None,
) -> None:
    """Argmax of every row into ``argmax`` [bs * d]; with ``sample``, the temperature-1 chain into
    ``predict`` / ``accept_index`` / ``num_correct_drafts``."""
    _module().decide(
        stats,
        candidates,
        retrieve_index,
        coins,
        coins_final,
        predict,
        accept_index,
        num_correct_drafts,
        argmax,
        _no_trace(argmax) if trace is None else trace,
        vocab_local,
        int(sample),
    )


def vp_pull(
    *,
    peer_logits: torch.Tensor,
    meta: torch.Tensor,
    out: torch.Tensor,
    ld: int,
    vocab_local: int,
    rank: int,
) -> None:
    """This rank's rows of every shard into ``out`` [rows, vocab_size] float32."""
    _module().pull(peer_logits, meta, out, ld, vocab_local, rank)
