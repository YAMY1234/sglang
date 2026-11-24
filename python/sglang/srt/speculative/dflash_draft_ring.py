"""A per-request ring of draft KV rows for an all-sliding-window MLA DFLASH draft.

The draft attends only to the last ``window_left + 1`` positions, and the KV of
a position depends on that position's target hidden state alone. So instead of
one draft row per target slot, each request owns ``rows_per_req`` rows per DCP
rank, enough for the window plus a verify block:

    local row(req, pos) = 1 + req * rows_per_req + (pos // dcp) % rows_per_req
    loc(req, pos)       = local row * dcp + pos % dcp

``loc`` keeps the DCP owner rule of the shared pool (``loc % dcp`` is the owner
rank, the row is ``loc // dcp``), so the existing sharded writers apply as is.
Row 0 is the write sink. The window kernel computes the same rows directly
(``mla_window_sparse_indices(ring_rows_per_req=...)``).

A request's ring holds no positions from before its first prefill chunk (its
prefix-cache hit) or from a previous request in the same req pool slot;
``min_pos`` floors the window there.

Under PD disaggregation prefill sends a request's last window of ring rows
(``transfer_rows``, positionally matched like DeepSeek-V4's SWA ring) and its
``min_pos`` row; a decode rank keeps the positions it owns.

A paged GQA reader (the DSpark draft on trtllm_mha) needs the ring page-aligned:
``sink_rows`` is then a whole page and ``rows_per_req`` a multiple of the page, so
``loc // page_size = sink_pages + req * pages_per_req + (pos // page_size) % pages_per_req``
and ``page_table_into`` gives the kernel the same pages ``req_to_token`` would if it
held ring locs. Such a draft is head-sharded on decode while a TP-1 prefill holds
every head, so decode receives whole rows into a staging ring and ``unstage`` copies
its heads into the ring before the request's first draft step.

A paged reader cannot floor its window, so a page-aligned ring is rebased on
``min_pos``: position ``pos`` lives at ``pos - min_pos[req]`` in the formulas above
and the reader gets ``window_seq_lens`` (``seq_lens - min_pos``); the window then
starts at the floor, never on rows from before it. Decode lands PD rows at their
absolute positions and ``unstage`` rotates them once the floor has landed.
"""

import logging
from collections import deque
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.kernels.ops.speculative.dflash_ring_locs import dflash_ring_block_locs
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def draft_kv_ring_enabled() -> bool:
    """The flag as resolved at startup (``_resolve_dflash_draft_kv_ring``)."""
    return envs.SGLANG_DFLASH_DRAFT_KV_RING.get()


def dspark_draft_kv_ring_enabled() -> bool:
    """The DSpark flag as resolved at startup (``_resolve_dspark_draft_kv_ring``)."""
    return envs.SGLANG_DSPARK_DRAFT_KV_RING.get()


def draft_kv_ring_active(spec_algorithm) -> bool:
    """Whether this algorithm's draft keeps its KV in a ring."""
    return (spec_algorithm.is_dflash() and draft_kv_ring_enabled()) or (
        spec_algorithm.is_dspark() and dspark_draft_kv_ring_enabled()
    )


DSPARK_DRAFT_KV_RING_BACKENDS = ("trtllm_mha",)


def dspark_draft_kv_ring_unsupported_reason(
    draft_hf_config, *, draft_backend: Optional[str]
) -> Optional[str]:
    """Why a page-aligned draft KV ring cannot serve this DSpark draft, or None."""
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_layer_types,
        is_dflash_mla_draft_config,
    )

    if is_dflash_mla_draft_config(draft_hf_config):
        return "the draft is MLA"
    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        return f"not every draft layer is sliding_attention ({layer_types})"
    if draft_backend not in DSPARK_DRAFT_KV_RING_BACKENDS:
        return f"draft backend {draft_backend!r} has no ring page-table path"
    return None


def draft_kv_ring_unsupported_reason(
    draft_hf_config, *, draft_backend: Optional[str], draft_window_size
) -> Optional[str]:
    """Why a draft KV ring cannot serve this DFLASH draft, or None if it can."""
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_layer_types,
        is_dflash_mla_draft_config,
    )
    from sglang.srt.speculative.draft_worker_common import (
        DFLASH_MLA_ONLY_DRAFT_BACKENDS,
    )

    if not is_dflash_mla_draft_config(draft_hf_config):
        return "the draft is not MLA"
    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        return f"not every draft layer is sliding_attention ({layer_types})"
    if draft_backend not in DFLASH_MLA_ONLY_DRAFT_BACKENDS:
        return f"draft backend {draft_backend!r} has no window index path"
    if draft_window_size is not None:
        return "--speculative-draft-window-size keeps the compact draft table"
    return None


def draft_kv_ring_window_left(draft_hf_config) -> int:
    """The draft's sliding window (window_left); a ring needs every layer to slide."""
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
        get_dflash_layer_types,
    )

    layer_types = get_dflash_layer_types(draft_hf_config)
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        raise ValueError(
            "SGLANG_DFLASH_DRAFT_KV_RING needs a draft whose every layer is "
            f"sliding_attention; got layer_types={layer_types}."
        )
    return get_dflash_attention_sliding_window_size(draft_hf_config)


def draft_kv_ring_rows_per_req(
    *, window_left: int, block_size: int, dcp_size: int
) -> int:
    """Rows per request per DCP rank: a window plus one verify block, plus one
    row so the block never lands on a key the window still reads."""
    span = window_left + 1 + block_size
    return -(-span // dcp_size) + 1


def dspark_draft_kv_ring_rows_per_req(
    *, window_left: int, block_size: int, page_size: int
) -> int:
    """Rows per request of a page-aligned ring: the pages of a window plus one
    verify block, plus one page each way so a block never lands on a page the
    window still reads."""
    span = window_left + 1 + block_size
    return (-(-span // page_size) + 2) * page_size


def draft_kv_ring_geometry(
    spec_algorithm, draft_hf_config, *, block_size: int, dcp_size: int, page_size: int
) -> Tuple[int, int]:
    """(rows_per_req, sink_rows) of this algorithm's draft KV ring."""
    window_left = draft_kv_ring_window_left(draft_hf_config)
    if spec_algorithm.is_dspark():
        return (
            dspark_draft_kv_ring_rows_per_req(
                window_left=window_left, block_size=block_size, page_size=page_size
            ),
            page_size,
        )
    return (
        draft_kv_ring_rows_per_req(
            window_left=window_left, block_size=block_size, dcp_size=dcp_size
        ),
        1,
    )


def draft_kv_ring_pool_rows(
    *, req_pool_rows: int, rows_per_req: int, page_size: int, sink_rows: int = 1
) -> int:
    rows = sink_rows + req_pool_rows * rows_per_req
    return -(-rows // page_size) * page_size


class DraftKVRingStaging:
    """Decode-side PD landing rows of a head-sharded draft KV ring: the same rows
    as the ring with every KV head (a TP-1 prefill's row), per layer K then V."""

    def __init__(
        self,
        *,
        pool,
        num_rows: int,
        total_kv_heads: int,
        mem_pool: Optional[torch.cuda.MemPool] = None,
    ):
        self.page_size = pool.page_size
        ring = pool._pd_registerable_tensors()
        with torch.cuda.use_mem_pool(mem_pool) if mem_pool else nullcontext():
            self._buffers = [
                torch.zeros(
                    (num_rows, total_kv_heads, b.shape[-1]),
                    dtype=b.dtype,
                    device=b.device,
                )
                for b in ring
            ]

    def ring_buffers(self) -> List[torch.Tensor]:
        return self._buffers

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return (
            [b.data_ptr() for b in self._buffers],
            [b.nbytes for b in self._buffers],
            [b[0].nbytes * self.page_size for b in self._buffers],
        )


class DFlashDraftKVRing:
    def __init__(
        self,
        *,
        rows_per_req: int,
        dcp_size: int,
        req_pool_rows: int,
        device: torch.device,
        window_left: int = 0,
        dcp_rank: int = 0,
        pool=None,
        mem_pool: Optional[torch.cuda.MemPool] = None,
        sink_rows: int = 1,
        page_size: Optional[int] = None,
    ):
        self.rows_per_req = rows_per_req
        self.dcp_size = dcp_size
        self.window_left = window_left
        self.dcp_rank = dcp_rank
        self.pool = pool
        # Rows before the first request's ring: row 0 is the write sink.
        self.sink_rows = sink_rows
        self.page_size = page_size
        if page_size is not None and (
            dcp_size != 1 or sink_rows % page_size or rows_per_req % page_size
        ):
            raise ValueError(
                "A page-aligned draft KV ring needs dcp_size 1 and page-multiple sink "
                f"and request rows: dcp={dcp_size}, sink={sink_rows}, "
                f"rows={rows_per_req}, page={page_size}."
            )
        # A head-sharded ring's staging ring and this rank's head columns in it.
        self.staging = None
        self._pending_unstage: deque = deque()
        # Positions a ring holds without wrapping onto each other.
        self.span = rows_per_req * dcp_size
        # PD writes min_pos rows remotely; Mooncake's NVLink transport only
        # reaches memory from the KV pools' cuMemCreate pool.
        with torch.cuda.use_mem_pool(mem_pool) if mem_pool else nullcontext():
            self.min_pos = torch.zeros(req_pool_rows, dtype=torch.int32, device=device)
        self._floor_rids: Dict[int, str] = {}
        self._fused_block_locs = envs.SGLANG_OPT_DFLASH_RING_LOCS_FUSED.get()
        # A page-aligned ring stores positions relative to min_pos (module doc);
        # the host floors are prefill's, for transfer_rows.
        self.rebased = page_size is not None
        self._floor_host: Dict[int, int] = {}
        # page_table_into's per-column ring pages, by table (width, dtype).
        self._page_cols: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}

    def locs(
        self, req_pool_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Ring locs of ``positions``, ``req_pool_indices`` given per row."""
        positions = positions.to(torch.int64)
        req_pool_indices = req_pool_indices.to(torch.int64)
        if self.rebased:
            positions = positions - self.min_pos[req_pool_indices]
        local = (
            self.sink_rows
            + req_pool_indices * self.rows_per_req
            + torch.remainder(positions // self.dcp_size, self.rows_per_req)
        )
        return local * self.dcp_size + torch.remainder(positions, self.dcp_size)

    def block_locs(
        self, req_pool_indices: torch.Tensor, positions: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Ring locs of a draft block: ``block_size`` positions per request."""
        if not self._fused_block_locs:
            return self.locs(req_pool_indices.repeat_interleave(block_size), positions)

        return dflash_ring_block_locs(
            req_pool_indices,
            positions,
            block_size=block_size,
            rows_per_req=self.rows_per_req,
            dcp_size=self.dcp_size,
            sink_rows=self.sink_rows,
            floors=self.min_pos if self.rebased else None,
        )

    def window_seq_lens(
        self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """A paged reader's sequence lengths: a rebased ring's window starts at
        ``min_pos``."""
        if not self.rebased:
            return seq_lens
        return seq_lens - self.min_pos[req_pool_indices.to(torch.int64)].to(
            seq_lens.dtype
        )

    def extend_locs(
        self,
        *,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        extend_lens: torch.Tensor,
        seq_ends: torch.Tensor,
    ) -> torch.Tensor:
        """Ring locs for one extend's tokens (request-major). A chunk longer than
        the ring wraps onto itself; only its last ``span`` positions, the ones a
        later window can read, keep a ring row, the rest go to the sink."""
        num_tokens = int(positions.numel())
        req_of_token = torch.repeat_interleave(
            req_pool_indices.to(torch.int64), extend_lens, output_size=num_tokens
        )
        end_of_token = torch.repeat_interleave(
            seq_ends.to(torch.int64), extend_lens, output_size=num_tokens
        )
        locs = self.locs(req_of_token, positions)
        keep = positions.to(torch.int64) >= end_of_token - self.span
        return torch.where(keep, locs, torch.zeros_like(locs))

    def record_floors(self, reqs, prefix_lens) -> None:
        """On a request's first extend, its prefix-cache hit length becomes the
        earliest position its window may read."""
        slots, floors = [], []
        for req, prefix_len in zip(reqs, prefix_lens):
            slot = req.kv.req_pool_idx
            if self._floor_rids.get(slot) != req.rid:
                self._floor_rids[slot] = req.rid
                self._floor_host[slot] = int(prefix_len)
                slots.append(slot)
                floors.append(int(prefix_len))
        if slots:
            self.min_pos[
                torch.tensor(slots, dtype=torch.int64).to(
                    self.min_pos.device, non_blocking=True
                )
            ] = torch.tensor(floors, dtype=torch.int32).to(
                self.min_pos.device, non_blocking=True
            )

    def transfer_rows(self, req_pool_idx: int, seq_len: int) -> np.ndarray:
        """This rank's ring rows of the last window before ``seq_len``, in
        position order; row 0 (the sink) for a position another rank owns. A
        rebased ring sends from its floor and lands at absolute positions
        (``reset_floor``)."""
        positions = np.arange(
            max(0, seq_len - self.window_left - 1), seq_len, dtype=np.int64
        )
        base = self._floor_host.get(req_pool_idx, 0) if self.rebased else 0
        rows = (
            self.sink_rows
            + req_pool_idx * self.rows_per_req
            + ((positions - base) // self.dcp_size) % self.rows_per_req
        )
        rows[positions % self.dcp_size != self.dcp_rank] = 0
        return rows.astype(np.int32)

    def page_table_into(
        self, req_pool_indices: torch.Tensor, out: torch.Tensor
    ) -> None:
        """A paged reader's block table for these requests (``out`` is
        ``[>= bs, width]``): logical page p of a request maps to its ring page
        ``p % pages_per_req``. The reader bounds its reads by the sequence lengths
        and the window, which never span more than ``pages_per_req`` pages."""
        pages_per_req = self.rows_per_req // self.page_size
        bs = req_pool_indices.shape[0]
        key = (out.shape[1], out.dtype)
        cols = self._page_cols.get(key)
        if cols is None:
            cols = self._page_cols[key] = (
                self.sink_rows // self.page_size
                + torch.arange(out.shape[1], device=out.device) % pages_per_req
            ).to(out.dtype)
        torch.add(
            cols,
            req_pool_indices.to(device=out.device, dtype=out.dtype).unsqueeze(1),
            alpha=pages_per_req,
            out=out[:bs],
        )

    def attach_staging(self, staging_pool, head_start: int, num_heads: int) -> None:
        """Receive PD rows (every head) into ``staging_pool``, a ring of the same
        rows; ``unstage`` copies heads ``[head_start, head_start + num_heads)``."""
        self.staging = (staging_pool, head_start, num_heads)

    def mark_unstage(self, req_pool_idx: int) -> None:
        """A request's PD rows landed (in the staging ring, or at absolute
        positions of a rebased ring); copy them into place before its first draft
        step (``flush_unstage``, on the forward stream)."""
        if self.staging is not None or self.rebased:
            self._pending_unstage.append(int(req_pool_idx))

    def flush_unstage(self) -> None:
        reqs = []
        while self._pending_unstage:
            reqs.append(self._pending_unstage.popleft())
        if not reqs:
            return
        if envs.SGLANG_LOG_MEM_CHECKPOINTS.get():
            logger.info(
                "UNSTAGE_FLUSH entries %d unique %d", len(reqs), len(set(reqs))
            )
        device = self.min_pos.device
        req = torch.tensor(reqs, dtype=torch.int64).to(device, non_blocking=True)
        first = self.sink_rows + req.unsqueeze(1) * self.rows_per_req
        cols = torch.arange(self.rows_per_req, device=device, dtype=torch.int64)
        rows = (first + cols).reshape(-1)
        dst_rows = rows
        if self.rebased:
            # Row j holds a position p = j mod rows_per_req; it belongs at p - floor.
            dst_rows = (
                first
                + torch.remainder(
                    cols - self.min_pos[req].unsqueeze(1), self.rows_per_req
                )
            ).reshape(-1)
        dsts = self.pool._pd_registerable_tensors()
        if self.staging is None:
            srcs, heads = dsts, slice(None)
        else:
            staging_pool, head_start, num_heads = self.staging
            srcs = staging_pool.ring_buffers()
            heads = slice(head_start, head_start + num_heads)
        for src, dst in zip(srcs, dsts):
            # The gather copies first, so an in-place rotation reads unrotated rows.
            dst[dst_rows] = src[rows][:, heads]

    def reset_floor(self, req_pool_idx: int, seq_len: int) -> None:
        """PD decode, at preallocation: no draft context until prefill's floor
        row lands, so a prefill that ships no ring leaves an empty window
        rather than a stale one."""
        self.min_pos[req_pool_idx].fill_(seq_len)
        self._floor_host.pop(req_pool_idx, None)

    def ring_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """The PD buffers (the staging ring's when there is one) with one item per
        ring row."""
        pool = self.pool if self.staging is None else self.staging[0]
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        return ptrs, lens, [n // pool.page_size for n in item_lens]

    def floor_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """``min_pos`` with one item per req pool row."""
        return (
            [self.min_pos.data_ptr()],
            [self.min_pos.nbytes],
            [self.min_pos.element_size()],
        )


_active_ring: Optional[DFlashDraftKVRing] = None


def set_draft_kv_ring(ring: Optional[DFlashDraftKVRing]) -> None:
    global _active_ring
    _active_ring = ring


def get_draft_kv_ring() -> Optional[DFlashDraftKVRing]:
    """This process's draft KV ring, for PD state registration and payloads."""
    return _active_ring
