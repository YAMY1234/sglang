"""Sharded-DCP DSA prefill: the sparse MLA split by rows over the DCP group.

Under Kimi-K3.1's sharded DCP every rank of the DCP group attends over the same
gathered KV with the same per-row top-k; only its query heads differ (attention
TP shards them). trtllm-gen's sparse MLA is bound by the per-row KV gathers, not
by heads, so attending every row with a rank's 12 heads costs nearly as much as
attending them with all 96. Instead, rank r attends rows [b_r, b_{r+1}) with
every head of the group: q moves heads -> rows in one all-to-all and the output
moves rows -> heads in another. Given the group's w_vc, each head's output is
projected before the return trip, which then carries v_head_dim instead of
kv_lora_rank per head.

On the row side heads are peer-major: head p * H_local + h is peer p's local
head h, the order an all-gather of w_vc over the group produces.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch

# Rows [lo, hi) of each rank of the group in one pass.
Segments = List[Tuple[int, int]]
# exchange(sends, recvs): for every peer p but this rank, send sends[p] to p and
# receive recvs[p] from p (both contiguous; empty ones are skipped on both ends).
Exchange = Callable[[List[torch.Tensor], List[torch.Tensor]], None]


def row_split_weights(
    seq_lens: Sequence[int],
    extend_lens: Sequence[int],
    num_rows: int,
    topk: int,
) -> torch.Tensor:
    """Cumulative sparse MLA work of rows [0, i] (int64, num_rows long): a row
    gathers min(kv_len, topk) keys, plus a fixed share for its q and output.
    Rows past the extends (MLP-sync padding) weigh the fixed share; with
    inconsistent lengths every row weighs 1. Host-only, identical on all ranks."""
    seq = torch.as_tensor(seq_lens, dtype=torch.int64).flatten()
    ext = torch.as_tensor(list(extend_lens), dtype=torch.int64)
    fixed = max(topk // 8, 1)
    real = int(ext.sum()) if ext.numel() else 0
    if ext.numel() == seq.numel() and 0 < real <= num_rows and bool((ext >= 0).all()):
        row_start = torch.repeat_interleave(torch.cumsum(ext, 0) - ext, ext)
        row_kv = (
            torch.repeat_interleave(seq - ext, ext)
            + torch.arange(real, dtype=torch.int64)
            - row_start
            + 1
        )
        weights = torch.cat(
            [
                row_kv.clamp(min=1, max=topk) + fixed,
                torch.full((num_rows - real,), fixed, dtype=torch.int64),
            ]
        )
    else:
        weights = torch.ones(num_rows, dtype=torch.int64)
    return torch.cumsum(weights, 0)


def _weighted_cuts(
    cum: torch.Tensor, lo: int, hi: int, parts: int, nonempty: bool
) -> List[int]:
    # lo = c_0 <= ... <= c_parts = hi with near-equal work per piece; nonempty
    # (needs hi - lo >= parts) keeps every piece at least one row.
    if parts == 1 or hi == lo:
        return [lo] * parts + [hi]
    base = int(cum[lo - 1]) if lo else 0
    targets = (
        base
        + (int(cum[hi - 1]) - base) * torch.arange(1, parts, dtype=torch.int64) // parts
    )
    cuts = (torch.searchsorted(cum[lo:hi], targets, right=True) + lo).tolist()
    out = [lo]
    for i, cut in enumerate(cuts, start=1):
        if nonempty:
            out.append(min(max(cut, out[-1] + 1), hi - (parts - i)))
        else:
            out.append(min(max(cut, out[-1]), hi))
    out.append(hi)
    return out


def row_split_bounds(cum: torch.Tensor, world: int) -> List[int]:
    """Contiguous bounds 0 = b_0 < ... < b_world = rows (rows >= world) with
    near-equal work (cum from row_split_weights) per rank."""
    assert cum.numel() >= world >= 1
    return _weighted_cuts(cum, 0, cum.numel(), world, nonempty=True)


def row_split_passes(
    bounds: Sequence[int], pass_rows: int, cum: Optional[torch.Tensor] = None
) -> List[Segments]:
    """Each rank's rows cut into the same number of passes, enough that the
    largest rank has about pass_rows per pass (<= 0: one pass). Passes end in
    an exchange, so each is balanced by work (cum) when given, else by rows."""
    world = len(bounds) - 1
    sizes = [bounds[r + 1] - bounds[r] for r in range(world)]
    num_passes = 1 if pass_rows <= 0 else max(1, -(-max(sizes) // pass_rows))
    cuts = []
    for r in range(world):
        lo, hi = bounds[r], bounds[r + 1]
        if cum is not None:
            cuts.append(_weighted_cuts(cum, lo, hi, num_passes, nonempty=False))
        else:
            step = -(-(hi - lo) // num_passes)
            cuts.append([min(lo + p * step, hi) for p in range(num_passes + 1)])
    return [
        [(cuts[r][p], cuts[r][p + 1]) for r in range(world)] for p in range(num_passes)
    ]


def heads_to_rows(
    q_local: torch.Tensor,
    segs: Segments,
    rank: int,
    exchange: Exchange,
    out: torch.Tensor,
) -> torch.Tensor:
    """q_local [rows, H_local, D] holds this rank's heads of every row. Writes
    out[:m, :world * H_local] (out is [>= m, >= world * H_local, D]) with the
    rows segs[rank] and the group's heads, and returns that view."""
    world = len(segs)
    _, local_heads, dim = q_local.shape
    lo, hi = segs[rank]
    m = hi - lo
    recv = q_local.new_empty((world, m, local_heads, dim))
    exchange([q_local[a:b] for a, b in segs], list(recv.unbind(0)))
    recv[rank].copy_(q_local[lo:hi])
    dst = out[:m, : world * local_heads]
    dst.view(m, world, local_heads, dim).copy_(recv.transpose(0, 1))
    return dst


def rows_to_heads(
    o_rows: torch.Tensor,
    segs: Segments,
    rank: int,
    exchange: Exchange,
    out: torch.Tensor,
    v_weights: Optional[torch.Tensor] = None,
) -> None:
    """o_rows [m, >= world * H_local, D_latent] holds the rows segs[rank] with
    the group's heads. Fills out[a:b] (out is [rows, H_local, D_out]) for every
    peer's rows segs[p] with this rank's heads. With v_weights [world * H_local,
    D_latent, D_out] each head is multiplied by its w_vc before the trip."""
    world = len(segs)
    _, local_heads, dim_out = out.shape
    lo, hi = segs[rank]
    m = hi - lo
    send = out.new_empty((world, m, local_heads, dim_out))
    for p in range(world):
        heads = slice(p * local_heads, (p + 1) * local_heads)
        src = o_rows[:m, heads]
        dst = out[lo:hi] if p == rank else send[p]
        if v_weights is None:
            dst.copy_(src)
        else:
            torch.bmm(src.transpose(0, 1), v_weights[heads], out=dst.transpose(0, 1))
    exchange(list(send.unbind(0)), [out[a:b] for a, b in segs])


def _wire(t: torch.Tensor) -> torch.Tensor:
    # pynccl has no fp8 dtype; move raw bytes (the buffers are contiguous).
    assert t.is_contiguous()
    return t.reshape(-1).view(torch.uint8)


def pynccl_exchange(comm, rank: int) -> Exchange:
    """An Exchange over a PyNcclCommunicator: one NCCL group of send/recv pairs."""

    def exchange(sends: List[torch.Tensor], recvs: List[torch.Tensor]) -> None:
        with comm.change_state(enable=True):
            comm.group_start()
            for peer, (s, r) in enumerate(zip(sends, recvs)):
                if peer == rank:
                    continue
                if s.numel():
                    comm.send(_wire(s), peer)
                if r.numel():
                    comm.recv(_wire(r), peer)
            comm.group_end()

    return exchange
