# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DSA indexer top-k under decode context parallelism (DCP).

The index-K cache is sharded by token: global position p of a request lives on
rank p % N at local position p // N. Each rank selects the top-k of its own
keys, the ranks all-gather (score, global position) candidates, and every rank
selects the global top-k from them. A key in the global top-k ranks within the
top-k of its own shard, so the merge is exact.

Both selections run the v2 top-k kernel, which orders by score descending,
then column ascending. Local columns are in global position order, and the
merge scatters candidates into a row indexed by global position, so ties
resolve by global position at both steps. With 8 or 16 ranks and multicast,
decode rows instead merge in one kernel (``DcpTopKMerge``).
"""

from collections import OrderedDict
from functools import partial
from itertools import accumulate
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dcp_kernels import _gather_candidate_prefix_kernel
from sglang.kernels.ops.attention.dsa import k31ht_select as _k31ht_select
from sglang.kernels.ops.attention.dsa.dcp_topk_merge import (
    DCP_TOPK_MERGE_NUM_STATS,
    dcp_topk_merge,
    dcp_topk_merge_buffer_numel,
    dcp_topk_merge_row_slot,
    dcp_topk_merge_pushed,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    candidates_cluster_size,
    topk_transform_candidates_dcp_push,
    topk_transform_candidates_v2,
    topk_transform_packed_v2,
    topk_transform_paged_v2,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_one_nvlink_clique,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled
from sglang.srt.layers.dcp.comm import all_gather_kv_cache_for_dcp
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.srt.utils import print_info_once

_ZEROS: dict[torch.device, torch.Tensor] = {}
_SIDE_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _SIDE_STREAMS.get(device)
    if stream is None:
        stream = _SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
    return stream


class DcpCandidateAllGather:
    """All-gather of the top-k candidates over the DCP group through torch
    symmetric memory (NVLS multimem, as TorchSymmMemCommunicator does for
    all-reduce): every rank writes its candidates into one symmetric output
    buffer, rendezvoused once at init. Falls back to the group all-gather when
    multicast is unavailable or the payload does not fit. Only the first
    min(local_lens[row], topk) pairs in each row are valid.
    """

    def __init__(self, group, max_rows: int, topk: int):
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.max_rows = max_rows
        self.topk = topk
        self.merge_buffer: Optional[torch.Tensor] = None
        self.out: Optional[torch.Tensor] = None
        self.prefix_handle = None
        self.group_name = group.device_group.group_name
        if not all(in_the_same_node_as(group.cpu_group)) and not is_one_nvlink_clique(
            group.cpu_group, group.device
        ):
            return
        prefix = group.world_size == 8 and topk == 2048 and get_platform().is_sm100
        if prefix:
            symm_mem.set_signal_pad_size(
                max(symm_mem.get_signal_pad_size(), 32 * group.world_size * 4)
            )
        per_rank = max_rows * topk * 2  # int32 (score bits, position) pairs
        out = symm_mem.empty(
            group.world_size * per_rank, dtype=torch.int32, device=group.device
        )
        handle = symm_mem.rendezvous(out, self.group_name)
        if handle.multicast_ptr != 0:
            self.out = out
            if prefix:
                self.prefix_handle = handle

    def __call__(
        self,
        t: torch.Tensor,
        *,
        local_lens: torch.Tensor,
        compact_prefix: bool = False,
    ) -> torch.Tensor:
        n = t.numel()
        if (
            self.out is None
            or t.shape[1] != self.topk
            or n * self.group.world_size > self.out.numel()
        ):
            return self.group.all_gather(t, dim=0)
        out = self.out[: self.group.world_size * n]
        if self.prefix_handle is None:
            torch.ops.symm_mem.multimem_all_gather_out(t.view(-1), self.group_name, out)
        else:
            handle = self.prefix_handle
            _gather_candidate_prefix_kernel[(8 if t.shape[0] <= 512 else 32,)](
                t,
                local_lens,
                handle.multicast_ptr,
                handle.signal_pad_ptrs_dev,
                t.shape[0],
                COMPACT_PREFIX=compact_prefix,
                TOPK=t.shape[1],
                RANK=handle.rank,
                WORLD=handle.world_size,
                BLOCK=512,
                num_warps=16,
            )
        return out.view(self.group.world_size * t.shape[0], *t.shape[1:])

    def merge_scores(self, rows: int, width: int) -> torch.Tensor:
        # Like the candidate buffer, this scratch is consumed before the next
        # merge. Reserve it during eager warmup and reuse it across graph batch
        # sizes: a context-sized allocation per capture can exhaust VRAM.
        # Never replace storage that a captured graph may still reference.
        if (
            self.merge_buffer is None
            and 0 < rows <= self.max_rows
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.merge_buffer = torch.empty(
                (self.max_rows, width), dtype=torch.float32, device=self.group.device
            )
        if (
            self.merge_buffer is not None
            and rows <= self.max_rows
            and width <= self.merge_buffer.shape[1]
        ):
            return self.merge_buffer[:rows, :width]
        return torch.empty((rows, width), dtype=torch.float32, device=self.group.device)


_CANDIDATE_ALL_GATHER: Optional[DcpCandidateAllGather] = None


def init_dcp_candidate_all_gather(max_rows: int, topk: int) -> None:
    """Collective over the DCP group; call on every rank before CUDA graph
    capture."""
    global _CANDIDATE_ALL_GATHER
    if _CANDIDATE_ALL_GATHER is not None:  # one buffer per process
        return
    _CANDIDATE_ALL_GATHER = DcpCandidateAllGather(
        get_parallel().dcp_group, max_rows, topk
    )


_CANDIDATE_CLUSTER_FLOOR = 16384

# Candidates per rank in the merge. Over 109824 decode rows of six real
# workloads (32K and 100K contexts, all 24 DSA layers), a rank held at most 339
# of a row's global top-2048 keys under DCP8, so 384 needs no fallback there.
_DCP_MERGE_CAND_K = 384
# DCP4 (push mode only): SGLANG_DCP4_PUSH_CAND_K, default 768, which keeps the
# union at 3072 >= 2048 keys, as 8 x 384, so no rank can certify fewer than the
# global top-2048 it holds.


def _merge_cand_k(world: int, topk: int) -> int:
    if world != 4:
        return _DCP_MERGE_CAND_K
    cand_k = envs.SGLANG_DCP4_PUSH_CAND_K.get()
    if cand_k % 32 or not (topk <= world * cand_k and cand_k <= topk):
        raise ValueError(
            f"SGLANG_DCP4_PUSH_CAND_K must be a multiple of 32 in "
            f"[{topk // world}, {topk}], got {cand_k}"
        )
    return cand_k


class DcpTopKMerge:
    """Exact global top-k merge of decode rows in one kernel; see
    jit/csrc/dsa/dcp_topk_merge.cuh. Needs 8 or 16 ranks (one warp each), top-k
    2048 and multicast; ``usable`` is False otherwise. ``split``: candidates come
    from the context-split verify indexer. ``push`` (token interleave only): the
    local top-k selects only the candidates and pushes them (``merge_pushed``)."""

    cand_k = _DCP_MERGE_CAND_K
    merge_warps = 0

    def __init__(
        self,
        group,
        max_rows: int,
        topk: int,
        split: bool = False,
        push: bool = False,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.rank = group.rank_in_group
        self.world = group.world_size
        self.max_rows = max_rows
        self.split = split
        self.push = push and not split
        self.buffer: Optional[torch.Tensor] = None
        self.multicast_ptr = 0
        self.peer_ptrs: Optional[torch.Tensor] = None
        self.counters: Optional[torch.Tensor] = None
        # 1 where a row's candidates could not certify the result (for tests)
        self.fallback: Optional[torch.Tensor] = None
        # running totals of the kernels (DCP_TOPK_MERGE_NUM_STATS), reported by
        # log_line from a copy one report old, so reading them never syncs
        self.stats: Optional[torch.Tensor] = None
        self._stats_host: Optional[torch.Tensor] = None
        self._stats_copied: Optional[torch.cuda.Event] = None
        self._stats_reported = [0] * DCP_TOPK_MERGE_NUM_STATS
        self._report_index = 0
        self.union_scratch: Optional[torch.Tensor] = None

        world = group.world_size
        # 16 ranks only on megakernel paths; stock DCP16 keeps the all-gather.
        worlds = (8, 16) if split or megakernel_enabled() else (8,)
        if self.push:
            worlds += (4,)
        self.cand_k = _merge_cand_k(world, topk)
        self.merge_warps = envs.SGLANG_DCP4_PUSH_MERGE_WARPS.get() if world == 4 and self.cand_k == topk else 0
        if self.merge_warps:
            print_info_once(f"DSA DCP4 push merge: {self.merge_warps} warps per rank (cand_k {self.cand_k} == top-k)")
        if world not in worlds or topk != 2048:
            print_info_once(
                "DSA DCP top-k merge off: it needs 8 or 16 ranks and top-k 2048"
                if 16 in worlds
                else "DSA DCP top-k merge off: it needs 8 ranks and top-k 2048"
            )
            return
        if not all(in_the_same_node_as(group.cpu_group)) and not is_one_nvlink_clique(
            group.cpu_group, group.device
        ):
            return
        buffer = symm_mem.empty(
            dcp_topk_merge_buffer_numel(world, max_rows, topk, self.cand_k, self.push),
            dtype=torch.int64,
            device=group.device,
        )
        # Zero before the rendezvous: a peer pushes into this buffer as soon as
        # its own rendezvous returns.
        buffer.zero_()
        torch.cuda.synchronize(group.device)
        handle = symm_mem.rendezvous(buffer, group.device_group.group_name)
        if handle.multicast_ptr == 0:
            print_info_once("DSA DCP top-k merge off: no multicast")
            return
        self.buffer = buffer
        self.multicast_ptr = handle.multicast_ptr
        self.peer_ptrs = torch.tensor(
            handle.buffer_ptrs, dtype=torch.int64, device=group.device
        )
        self.counters = torch.zeros(max_rows, dtype=torch.int64, device=group.device)
        self.fallback = torch.zeros(max_rows, dtype=torch.int32, device=group.device)
        self.stats = torch.zeros(
            DCP_TOPK_MERGE_NUM_STATS, dtype=torch.int64, device=group.device
        )
        self._stats_host = torch.zeros_like(self.stats, device="cpu").pin_memory()
        if self.push:
            # the pushed merge's fallback reads the failed ranks' slots directly
            print_info_once(
                f"DSA DCP top-k merge: the local top-k pushes its best "
                f"{self.cand_k} keys per row to the merge, no local "
                f"top-{topk} (SGLANG_DSA_DCP_TOPK_PUSH)"
            )
            return
        self.union_scratch = torch.empty(
            (max_rows, world * topk), dtype=torch.int64, device=group.device
        )

    def usable(self, rows: int) -> bool:
        return self.buffer is not None and rows <= self.max_rows

    def __call__(
        self,
        candidates: torch.Tensor,
        global_lens: torch.Tensor,
        page_table: torch.Tensor,
        out: torch.Tensor,
        span: int,
    ) -> None:
        dcp_topk_merge(
            candidates,
            global_lens,
            page_table,
            buffer=self.buffer,
            multicast_ptr=self.multicast_ptr,
            peer_ptrs=self.peer_ptrs,
            counters=self.counters,
            union_scratch=self.union_scratch,
            out_slots=out,
            out_fallback=self.fallback[: out.shape[0]],
            rank=self.rank,
            span=span,
            cand_k=self.cand_k,
            split=self.split,
            stats=self.stats,
        )

    def merge_pushed(
        self,
        logits: torch.Tensor,
        local_lens: torch.Tensor,
        global_lens: torch.Tensor,
        page_table: torch.Tensor,
        out: torch.Tensor,
        span: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Push mode: this rank's top-``cand_k`` candidates straight
        into every rank's buffer, then the merge. Fills ``out`` (as ``__call__``)
        and returns the same keys localized, as ``dcp_localize_compact(out)``."""
        rows = out.shape[0]
        topk_transform_candidates_dcp_push(
            logits,
            local_lens,
            push_mc=self.multicast_ptr,
            counters=self.counters,
            world=self.world,
            rank=self.rank,
            cand_k=self.cand_k,
            slot=dcp_topk_merge_row_slot(self.world, out.shape[1], self.cand_k, push=True),
            cluster_size=candidates_cluster_size(rows),
            cluster_floor=_CANDIDATE_CLUSTER_FLOOR,
        )
        local = torch.empty_like(out)
        lengths = torch.empty(rows, dtype=torch.int32, device=out.device)
        dcp_topk_merge_pushed(
            logits,
            local_lens,
            global_lens,
            page_table,
            buffer=self.buffer,
            peer_ptrs=self.peer_ptrs,
            counters=self.counters,
            out_slots=out,
            out_local=local,
            out_lengths=lengths,
            out_fallback=self.fallback[:rows],
            rank=self.rank,
            span=span,
            cand_k=self.cand_k,
            stats=self.stats,
            warps_per_rank=self.merge_warps,
        )
        return local, lengths

    def log_line(self, decode_step: int) -> Optional[str]:
        """Fallback counts since the previous line, every
        SGLANG_DSA_DCP_TOPK_MERGE_LOG_INTERVAL decode steps (0: never)."""
        interval = envs.SGLANG_DSA_DCP_TOPK_MERGE_LOG_INTERVAL.get()
        if self.stats is None or interval <= 0 or decode_step // interval == self._report_index:
            return None
        self._report_index = decode_step // interval
        line = None
        if self._stats_copied is not None and self._stats_copied.query():
            now = self._stats_host.tolist()
            fallback, published, selected, rows = (
                a - b for a, b in zip(now, self._stats_reported)
            )
            self._stats_reported = now
            line = (
                f"DSA DCP top-k merge (DCP{self.world}, {self.cand_k} candidates): "
                f"certificate failed on {fallback} of {rows} merged rows "
                f"({100 * fallback / max(rows, 1):.4f}%); this rank published "
                f"{published} of them, {selected} needing a top-k select"
            )
        self._stats_host.copy_(self.stats, non_blocking=True)
        self._stats_copied = torch.cuda.Event()
        self._stats_copied.record()
        return line


_TOPK_MERGE: Optional[DcpTopKMerge] = None


def topk_merge_log_line(decode_step: int) -> Optional[str]:
    return None if _TOPK_MERGE is None else _TOPK_MERGE.log_line(decode_step)

# (slots, localized rows, counts) of the last pushed merge: dcp_localize_compact
# returns the last two for that very slots tensor instead of recomputing them
_PUSHED_LOCALIZED: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None


def init_dcp_topk_merge(max_rows: int, topk: int) -> None:
    """Collective over the DCP group; call on every rank before CUDA graph
    capture."""
    global _TOPK_MERGE
    if _TOPK_MERGE is not None:  # one buffer per process
        return
    parallel = get_parallel()
    group = parallel.dcp_group
    # the kernel's rank must be the one the local candidates encode positions with
    assert (group.world_size, group.rank_in_group) == (
        parallel.attn_dcp_size,
        parallel.attn_dcp_rank,
    )
    _TOPK_MERGE = DcpTopKMerge(
        group, max_rows, topk, push=envs.SGLANG_DSA_DCP_TOPK_PUSH.get()
    )


def _zeros_i32(n: int, device: torch.device) -> torch.Tensor:
    """A read-only int32 zero vector of length n from a shared buffer (grown
    outside CUDA graph capture only, so captured graphs never see it move)."""
    buf = _ZEROS.get(device)
    if buf is None or buf.numel() < n:
        if torch.cuda.is_current_stream_capturing():
            return torch.zeros(n, dtype=torch.int32, device=device)
        buf = _ZEROS[device] = torch.zeros(
            max(n, 4096), dtype=torch.int32, device=device
        )
    return buf[:n]


@triton.jit
def _fill_visible_kernel(dense, dense_stride, lens, BLOCK: tl.constexpr):
    # -inf over [0, lens[row]), the only columns the top-k reads; a fixed
    # number of programs per row strides over it
    row = tl.program_id(0)
    n = tl.load(lens + row)
    base = dense + row.to(tl.int64) * dense_stride
    for start in range(tl.program_id(1) * BLOCK, n, tl.num_programs(1) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        tl.store(base + offs, float("-inf"), mask=offs < n)


@triton.jit
def _scatter_candidates_kernel(
    gathered,
    num_rows,
    global_lens,
    dense,
    dense_stride,
    WORLD: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPACT_PREFIX: tl.constexpr = False,
):
    # one program per (rank, row) block of candidates
    src_row = tl.program_id(0)
    row = src_row % num_rows
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # Prefix-only gathers leave the padding undefined, including on graph
    # replays with shorter lengths. Never read it or scatter stale positions.
    n = (tl.load(global_lens + row) + WORLD - 1 - src_row // num_rows) // WORLD
    mask = offs < tl.minimum(n, TOPK)
    base = gathered + src_row.to(tl.int64) * TOPK * 2
    if COMPACT_PREFIX and n <= TOPK:
        score = tl.load(base + offs, mask=mask, other=0).to(tl.float32, bitcast=True)
        pos = offs * WORLD + src_row // num_rows
    else:
        score = tl.load(base + offs * 2, mask=mask, other=0).to(
            tl.float32, bitcast=True
        )
        pos = tl.load(base + offs * 2 + 1, mask=mask, other=-1)
    tl.store(
        dense + row.to(tl.int64) * dense_stride + pos,
        score,
        mask=mask & (pos >= 0),
    )


@triton.jit
def _localize_compact_kernel(
    slots,
    slots_stride,
    out,
    out_stride,
    lengths,
    dcp_size,
    dcp_rank,
    TOPK: tl.constexpr,
):
    # owned slots (slot % N == rank) -> local rows slot // N at the front, in
    # order, -1 after them; lengths[row] = number of owned slots
    row = tl.program_id(0)
    offs = tl.arange(0, TOPK)
    x = tl.load(slots + row.to(tl.int64) * slots_stride + offs)
    own = (x >= 0) & (x % dcp_size == dcp_rank)
    own_i = own.to(tl.int32)
    n = tl.sum(own_i, axis=0)
    dst = tl.where(
        own, tl.cumsum(own_i, axis=0) - 1, n + tl.cumsum(1 - own_i, axis=0) - 1
    )
    tl.store(
        out + row.to(tl.int64) * out_stride + dst, tl.where(own, x // dcp_size, -1)
    )
    tl.store(lengths + row, n)


def dcp_localize_compact(slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Global top-k slots [rows, topk] -> (this rank's rows compacted to the
    front, -1 after; [rows] counts) for FlashMLA's ``topk_length``."""
    if _PUSHED_LOCALIZED is not None and _PUSHED_LOCALIZED[0] is slots:
        return _PUSHED_LOCALIZED[1], _PUSHED_LOCALIZED[2]
    parallel = get_parallel()
    rows, topk = slots.shape
    out = torch.empty_like(slots)
    lengths = torch.empty(rows, dtype=torch.int32, device=slots.device)
    if rows:
        _localize_compact_kernel[(rows,)](
            slots,
            slots.stride(0),
            out,
            out.stride(0),
            lengths,
            parallel.attn_dcp_size,
            parallel.attn_dcp_rank,
            TOPK=topk,
        )
    return out, lengths


def _local_candidates(
    logits: torch.Tensor, local_lens: torch.Tensor, topk: int
) -> torch.Tensor:
    # [rows, topk, 2] int32 (fp32 score bits, global position or -1); row i of
    # logits scores its request's local positions 0 .. local_lens[i]-1
    parallel = get_parallel()
    rows = local_lens.shape[0]
    out = torch.empty((rows, topk, 2), dtype=torch.int32, device=local_lens.device)
    if rows == 0:
        return out
    if logits.shape[1] == 0:
        out[..., 1] = -1
        return out
    if _k31ht_select.ENABLED:  # K31HT_DSA_SELECT=gvrm (default off): same value-exact set, not tie-stable
        return _k31ht_select.local_candidates(
            logits, local_lens, out, parallel.attn_dcp_size, parallel.attn_dcp_rank, topk
        )
    # v2 ragged top-k with the candidate epilogue: local positions ascend
    # with global ones, so its tie order matches the merge's
    topk_transform_candidates_v2(
        logits,
        local_lens,
        out_candidates=out,
        world=parallel.attn_dcp_size,
        rank=parallel.attn_dcp_rank,
        stable_ties=True,
        cluster_size=candidates_cluster_size(rows),
        cluster_floor=_CANDIDATE_CLUSTER_FLOOR,
    )
    return out


def dcp_global_topk(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    topk: int,
    page_size: int,
    plan: torch.Tensor,
    *,
    all_gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Decode / target verify: global top-k over the DCP group as widened KV
    slots, -1 padded.

    Row i is a query row (decode: request i; target verify: request i // n,
    draft step i % n). ``local_lens`` / ``global_lens``: [rows] visible keys
    of this rank / of the whole sequence for that row. ``page_table``: [rows,
    pages] this rank's physical page ids, which are the widened page ids of the
    row's request:
    global position p maps to slot ``page_table[p // span] * span + p % span``
    with span ``page_size * dcp_size``. ``plan``: the v2 top-k plan of
    ``global_lens``. ``all_gather`` replaces the group all-gather (tests run
    every rank in one process). Every rank must call this with the same number
    of rows, also when it has no local keys.

    With ``DcpTopKMerge`` the result holds only this rank's slots, which is all
    ``dcp_localize_compact`` keeps.
    """
    parallel = get_parallel()
    rows = local_lens.shape[0]
    span = page_size * parallel.attn_dcp_size
    global_lens = global_lens.to(torch.int32)
    out = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    if rows == 0:
        return out
    if all_gather is None and _TOPK_MERGE is not None and _TOPK_MERGE.usable(rows):
        if _TOPK_MERGE.push:
            global _PUSHED_LOCALIZED
            local, lengths = _TOPK_MERGE.merge_pushed(
                logits, local_lens, global_lens, page_table, out, span
            )
            _PUSHED_LOCALIZED = (out, local, lengths)
            return out
        candidates = _local_candidates(logits, local_lens, topk)
        _TOPK_MERGE(candidates, global_lens, page_table, out, span)
        return out
    workspace = _CANDIDATE_ALL_GATHER if all_gather is None else None
    compact_prefix = (
        workspace is not None
        and workspace.prefix_handle is not None
        and rows <= workspace.max_rows
        and topk == workspace.topk
    )
    if all_gather is None and workspace is not None:
        all_gather = partial(
            workspace, local_lens=local_lens, compact_prefix=compact_prefix
        )
    if all_gather is None:
        all_gather = partial(parallel.dcp_group.all_gather, dim=0)

    # Row i of `dense` holds the candidate scores at their global positions.
    # Its -inf fill depends on the lengths only, so it overlaps the local top-k
    # and the all-gather, which run on a side stream. The fill is captured
    # first so that a CUDA graph continues on this stream after the join.
    width = triton.cdiv(page_table.shape[1] * span, 4) * 4
    dense = (
        workspace.merge_scores(rows, width)
        if workspace is not None
        else torch.empty((rows, width), dtype=torch.float32, device=logits.device)
    )
    stream = torch.cuda.current_stream()
    side = _side_stream(logits.device)
    side.wait_stream(stream)
    _fill_visible_kernel[(rows, 16)](dense, dense.stride(0), global_lens, BLOCK=2048)
    with torch.cuda.stream(side):
        candidates = _local_candidates(logits, local_lens, topk)
        gathered = all_gather(candidates)
    stream.wait_stream(side)
    block = 1024
    _scatter_candidates_kernel[(gathered.shape[0], triton.cdiv(topk, block))](
        gathered,
        rows,
        global_lens,
        dense,
        dense.stride(0),
        WORLD=parallel.attn_dcp_size,
        COMPACT_PREFIX=compact_prefix,
        TOPK=topk,
        BLOCK=block,
    )
    topk_transform_paged_v2(
        dense, global_lens, page_table, out, span, plan, stable_ties=True
    )
    return out


def dcp_all_visible(
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    topk: int,
    page_size: int,
) -> torch.Tensor:
    """Decode / target verify: every visible position as widened KV slots, for
    rows that see at most ``topk`` keys (the indexer's k-only path), -1 padded."""
    span = page_size * get_parallel().attn_dcp_size
    rows = global_lens.shape[0]
    scores = torch.zeros((rows, topk), dtype=torch.float32, device=global_lens.device)
    out = torch.empty((rows, topk), dtype=torch.int32, device=global_lens.device)
    if rows == 0:
        return out
    topk_transform_packed_v2(
        scores,
        global_lens.to(torch.int32),
        page_table,
        out,
        span,
        row_starts=_zeros_i32(rows, global_lens.device),
        stable_ties=True,
    )
    return out


def dcp_gather_request_rows(
    local_rows: torch.Tensor, seq_lens_cpu: torch.Tensor
) -> torch.Tensor:
    """Prefill: every request's cache rows from all ranks, in position order.

    ``local_rows``: [sum of this rank's shares, W] uint8, request by request
    in local position order. Returns [sum(seq_lens), W]: request after
    request, the rows of positions 0..seq_len-1 (the layout of the ragged
    top-k indices).
    """
    return all_gather_kv_cache_for_dcp(local_rows, local_rows[:, :0], seq_lens_cpu)


# Fused prefill gather (SGLANG_DSA_DCP_PREFILL_FUSED_GATHER): one gather kernel
# packs this rank's rows straight from their source into the all-gather send
# buffer, and one writes the request-ordered result from the rank-major receive
# buffer. It replaces dcp_gather_request_rows' cat / per-request pad copies,
# the transpose reorder, the per-request cat, and the caller's pad or split.
@triton.jit
def _gather_rows_kernel(
    src,
    src_pitch,
    idx,
    idx2,
    dst,
    dst_pitch,
    n_rows,
    n_idx,
    WIDTH: tl.constexpr,
    HAS_IDX2: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # dst[row, :WIDTH] = src[idx2[idx[row]] if HAS_IDX2 else idx[row], :WIDTH];
    # rows past n_idx or with a negative index become zeros. Pointers and
    # pitches are in 32-bit words.
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    in_range = rows < n_rows
    src_rows = tl.load(idx + rows, mask=rows < n_idx, other=-1).to(tl.int64)
    if HAS_IDX2:
        src_rows = tl.load(idx2 + src_rows, mask=src_rows >= 0, other=-1).to(tl.int64)
    valid = src_rows >= 0
    src_base = src + src_rows * src_pitch
    dst_base = dst + rows.to(tl.int64) * dst_pitch
    for col in tl.static_range(0, WIDTH, BLOCK_W):
        cols = col + tl.arange(0, BLOCK_W)
        col_ok = cols < WIDTH
        value = tl.load(
            src_base[:, None] + cols[None, :],
            mask=valid[:, None] & col_ok[None, :],
            other=0,
        )
        tl.store(
            dst_base[:, None] + cols[None, :],
            value,
            mask=in_range[:, None] & col_ok[None, :],
        )


def _gather_rows(
    src: torch.Tensor,
    idx: torch.Tensor,
    dst: torch.Tensor,
    idx2: Optional[torch.Tensor] = None,
) -> None:
    """``dst[i] = src[idx2[idx[i]]]`` (or ``src[idx[i]]``) for 2D int32 views
    whose rows may be strided; rows past ``idx`` or with a negative index
    become zeros."""
    rows, width = dst.shape
    if rows == 0:
        return
    assert src.dtype == dst.dtype == torch.int32 and src.shape[1] == width
    assert src.stride(1) == 1 and dst.stride(1) == 1 and idx.numel() <= rows
    block_w = min(triton.next_power_of_2(width), 64)
    block_r = max(1, 2048 // block_w)
    _gather_rows_kernel[(triton.cdiv(rows, block_r),)](
        src,
        src.stride(0),
        idx,
        idx if idx2 is None else idx2,
        dst,
        dst.stride(0),
        rows,
        idx.numel(),
        WIDTH=width,
        HAS_IDX2=idx2 is not None,
        BLOCK_R=block_r,
        BLOCK_W=block_w,
        num_warps=4,
    )


class DcpRowGatherPlan:
    """Row maps of a fused DCP prefill gather; they depend only on the request
    lengths and the group, so a forward's layers (KV and index-K) share one.

    Global position p of a request lives on rank p % N at local position
    p // N. Rank r sends ceil(len / N) rows per request (its rows, then zero
    padding), so the all-gather output is [N, rows_per_rank] rank-major.
    """

    def __init__(
        self,
        seq_lens: list[int],
        world: int,
        rank: int,
        device: torch.device,
    ):
        padded = [(n + world - 1) // world for n in seq_lens]
        local = [max(0, (n - rank + world - 1) // world) for n in seq_lens]
        p_starts = [0, *accumulate(padded)]
        l_starts = [0, *accumulate(local)]
        s_starts = [0, *accumulate(seq_lens)]
        self.rows_per_rank = p_starts[-1]
        self.num_local = l_starts[-1]
        self.total = s_starts[-1]
        table = torch.tensor(
            [padded, local, p_starts[:-1], l_starts[:-1], s_starts[:-1], seq_lens],
            dtype=torch.int64,
            pin_memory=device.type == "cuda",
        ).to(device, non_blocking=True)
        padded_t, local_t, p_t, l_t, s_t, lens_t = table.unbind(0)
        reqs = torch.arange(len(seq_lens), device=device)

        # send row -> this rank's local row (or -1 for padding)
        req = torch.repeat_interleave(reqs, padded_t, output_size=self.rows_per_rank)
        j = torch.arange(self.rows_per_rank, device=device) - p_t[req]
        self.send_local = torch.where(j < local_t[req], l_t[req] + j, -1).to(
            torch.int32
        )
        # output row -> all-gather row
        req = torch.repeat_interleave(reqs, lens_t, output_size=self.total)
        p = torch.arange(self.total, device=device) - s_t[req]
        self.out_src = (
            (p % world) * self.rows_per_rank + p_t[req] + p // world
        ).to(torch.int32)


_ROW_GATHER_PLANS: "OrderedDict[tuple, DcpRowGatherPlan]" = OrderedDict()


def dcp_row_gather_plan(
    seq_lens_cpu: torch.Tensor, device: torch.device
) -> DcpRowGatherPlan:
    parallel = get_parallel()
    key = (
        tuple(seq_lens_cpu.tolist()),
        parallel.dcp_size,
        parallel.dcp_rank,
        device,
    )
    plan = _ROW_GATHER_PLANS.get(key)
    if plan is None:
        plan = DcpRowGatherPlan(list(key[0]), *key[1:])
        _ROW_GATHER_PLANS[key] = plan
        while len(_ROW_GATHER_PLANS) > 4:
            _ROW_GATHER_PLANS.popitem(last=False)
    return plan


def dcp_fused_gather_supported(*widths_bytes: int) -> bool:
    return (
        envs.SGLANG_DSA_DCP_PREFILL_FUSED_GATHER.get()
        and all(w % 4 == 0 for w in widths_bytes)
        and not torch.cuda.is_current_stream_capturing()
    )


def dcp_fused_gather_rows(
    sources: list[tuple[torch.Tensor, Optional[torch.Tensor]]],
    plan: DcpRowGatherPlan,
    outputs: list[torch.Tensor],
) -> None:
    """Prefill: ``dcp_gather_request_rows`` without its pack / reorder copies.

    ``sources``: (rows, index) pairs, each [*, W_i] byte rows of this rank; the
    send row of local row l is ``rows[index[l]]`` (``rows[l]`` without an
    index). ``outputs``: [>= plan.total, W_i] byte tensors that receive the
    request-ordered rows (zeros past plan.total).
    """
    print_info_once(
        "DSA DCP prefill: fused KV / index-K gather "
        "(SGLANG_DSA_DCP_PREFILL_FUSED_GATHER)"
    )
    parallel = get_parallel()
    device = outputs[0].device
    widths = [out.shape[1] // 4 for out in outputs]
    width = sum(widths)
    send = torch.empty((plan.rows_per_rank, width), dtype=torch.int32, device=device)
    col = 0
    for (rows, index), w in zip(sources, widths):
        _gather_rows(
            rows.view(torch.int32), plan.send_local, send[:, col : col + w], index
        )
        col += w
    recv = torch.empty(
        (parallel.dcp_size * plan.rows_per_rank, width),
        dtype=torch.int32,
        device=device,
    )
    parallel.dcp_group.all_gather_into_tensor(
        recv.view(torch.uint8), send.view(torch.uint8)
    )
    col = 0
    for out, w in zip(outputs, widths):
        _gather_rows(recv[:, col : col + w], plan.out_src, out.view(torch.int32))
        col += w
