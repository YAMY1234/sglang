"""Fused DSA indexer scorer + threshold top-k (FSTK) for the packed PAGED prefill.

Scores are bitwise DeepGEMM's legacy ``fp8_mqa_logits`` (32 heads x 128, FP8, FP32
reduction); the [rows, num_kv] score matrix is never written. Each row first gets a
threshold ``tau`` from every ``sample_stride``-th key of its window (the ``rank``-th largest
sample score its threads kept), the main pass appends the scores >= tau of the row's window
to per-warp candidate sub-lists, and the exact top-k (equal scores: smaller column first) is
selected and mapped through the page table. Rows whose sub-list overflowed or that hold fewer
than ``topk`` candidates (tau too high) are rescored on the device by a pass whose sub-lists
compact (they keep a superset of each warp's exact top-k), so every row is exact with no host
sync.
"""

from __future__ import annotations

import logging
import math
import os
from typing import NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit

logger = logging.getLogger(__name__)

MODE_LOGITS, MODE_NO_EPI, MODE_NO_STORE, MODE_FILTER, MODE_TAU, MODE_FILTER_COMPACT = 0, 1, 2, 3, 4, 5
# layout -> (rows per q-block, keys per split); see LayoutTraits in the kernel.
LAYOUTS = {0: (4, 256), 1: (4, 256), 2: (8, 128), 3: (4, 256), 4: (8, 128), 5: (4, 256), 15: (4, 256)}
DEFAULT_LAYOUT = 5  # measured fastest for the filter pass (VR, 11-20)
# Layout 5 with DeepGEMM's BF16 head reduction (scores are its BF16 logits); serving builds it too.
BF16_LAYOUT = 15
MATH_WARPS = 8  # candidate sub-lists per row in layouts 0-4: one per math warp of the row's CTA
SUB_LISTS = 4  # layout 5 (the fused path): one per warp of the row's warpgroup
SELECT_KEYS, SELECT_REPAIR_KEYS = 8192, 16384  # select kernel shared-memory capacity, in keys
# Device counters (int64) of fused_indexer_topk; see Stat in the kernel.
STAT_CALLS, STAT_REPAIR_ROWS, STAT_MAX_REPAIR, STAT_COMPACTIONS, STAT_UNREPAIRED, STAT_REPAIR_CALLS = range(6)
NUM_STATS = 8


def _deep_gemm_include() -> tuple[str, bool]:
    """DeepGEMM's header root, and whether the MQA logits kernel sits under legacy/."""
    import deep_gemm

    root = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
    legacy = os.path.exists(os.path.join(root, "deep_gemm/legacy/impls/sm100_mqa_logits.cuh"))
    if not legacy and not os.path.exists(os.path.join(root, "deep_gemm/impls/sm100_mqa_logits.cuh")):
        raise RuntimeError(f"fused indexer top-k: no sm100_mqa_logits.cuh under {root}")
    return root, legacy


@cache_once
def _jit_module(cuh_path: Optional[str] = None, serving: bool = False):
    """serving builds only what the prefill path runs (DEFAULT_LAYOUT and BF16_LAYOUT: threshold,
    filter and repair passes)."""
    dg_root, dg_legacy = _deep_gemm_include()
    return load_jit(
        "dsa_fused_indexer_topk",
        int(dg_legacy),
        int(serving),
        cuda_files=[cuh_path or "dsa/fused_indexer_topk.cuh"],
        cuda_wrappers=[
            ("score", "fstk::FusedIndexerTopK::score"),
            ("select", "fstk::FusedIndexerTopK::select"),
            ("repair_prep", "fstk::FusedIndexerTopK::repair_prep"),
        ],
        # DeepGEMM's own JIT flags; no fast-math, the bitwise contract depends on it.
        extra_cuda_cflags=[
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--compiler-options=-fconcepts",
            "--ptxas-options=--register-usage-level=10",
            f"-DFSTK_DG_LEGACY={int(dg_legacy)}",
            f"-DFSTK_SERVING={int(serving)}",
            f"-DFSTK_SERVING_LAYOUT={DEFAULT_LAYOUT}",
        ],
        extra_include_paths=[dg_root],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def _empty(device, dtype=torch.float32):
    return torch.empty(0, device=device, dtype=dtype)


def lpt_order(ks: torch.Tensor, ke: torch.Tensor, num_kv: int, layout: int = DEFAULT_LAYOUT) -> torch.Tensor:
    """q-blocks by descending KV span (the kernel deals them out snake-wise)."""
    block_q, split_kv = LAYOUTS[layout]
    t = ks.shape[0]
    nb = (t + block_q - 1) // block_q
    pad = nb * block_q - t
    s = ks.clamp(max=num_kv).to(torch.int64)
    e = ke.clamp(max=num_kv).to(torch.int64)
    if pad:
        s = torch.cat([s, s[-1:].expand(pad)])
        e = torch.cat([e, e[-1:].expand(pad)])
    s = s.view(nb, block_q).amin(1) // 4 * 4
    e = e.view(nb, block_q).amax(1)
    span = (e - s + split_kv - 1) // split_kv
    return torch.argsort(span, descending=True, stable=True).to(torch.int32)


def score(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    mode: int,
    order: Optional[torch.Tensor] = None,
    logits: Optional[torch.Tensor] = None,
    tau: Optional[torch.Tensor] = None,
    cand_key: Optional[torch.Tensor] = None,
    cand_count: Optional[torch.Tensor] = None,
    tau_out: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
    row_map: Optional[torch.Tensor] = None,
    num_rows: Optional[torch.Tensor] = None,
    stats: Optional[torch.Tensor] = None,
    topk: int = 0,
    rank: int = 0,
    keep_all: int = 0,
    ks_div: int = 1,
    reuse_kv: bool = False,
    layout: int = DEFAULT_LAYOUT,
    alu_heads: int = 0,
    num_ctas: Optional[int] = None,
    cuh_path: Optional[str] = None,
    serving: bool = False,
) -> None:
    dev = q_fp8.device
    if num_ctas is None:
        num_ctas = _num_sms(dev)
    i32 = _empty(dev, torch.int32)
    _jit_module(cuh_path, serving).score(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        ks,
        ke,
        order if order is not None else i32,
        logits if logits is not None else _empty(dev),
        tau if tau is not None else _empty(dev),
        cand_key if cand_key is not None else _empty(dev, torch.int64),
        cand_count if cand_count is not None else i32,
        tau_out if tau_out is not None else _empty(dev),
        lengths if lengths is not None else i32,
        row_map if row_map is not None else i32,
        num_rows if num_rows is not None else i32,
        stats if stats is not None else _empty(dev, torch.int64),
        topk,
        rank,
        keep_all,
        ks_div,
        mode,
        int(reuse_kv),
        layout,
        alu_heads,
        num_ctas,
    )


@cache_once
def _num_sms_cached(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _num_sms(dev: torch.device) -> int:
    return _num_sms_cached(dev.index if dev.index is not None else torch.cuda.current_device())


class _Carver:
    """Hands out 256-byte aligned views of one byte workspace (no allocation per call)."""

    def __init__(self, ws: Optional[torch.Tensor]):
        self.ws, self.off = ws, 0

    def take(self, shape, dtype, device) -> torch.Tensor:
        if self.ws is None:
            return torch.empty(shape, device=device, dtype=dtype)
        n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        start = (self.off + 255) // 256 * 256
        if start + n > self.ws.numel():
            raise ValueError(f"fused indexer top-k: workspace of {self.ws.numel()} B is too small")
        self.off = start + n
        return self.ws[start : start + n].view(dtype).view(shape)


def _sub_cap(cand_cap: int) -> int:
    return cand_cap // SUB_LISTS


def scratch_bytes_per_row(num_kv: int, sample_stride: int = 16, cand_cap: int = 16384) -> int:
    """Device scratch of fused_indexer_topk per query row: candidates, threshold, counts, and
    the compact copy of a row for the device-side repair."""
    del num_kv, sample_stride
    return 8 * cand_cap + 4 * SUB_LISTS + 4 + 4 + (32 * 128 + 4 * 32 + 3 * 4)


def workspace_fixed_bytes(num_kv: int, sample_stride: int = 16) -> int:
    """Workspace bytes of a call that do not scale with its rows (sample keys, alignment)."""
    num_samples = (num_kv + sample_stride - 1) // sample_stride
    return num_samples * (128 + 4) + 16 + 16 * 256


class FusedTopKResult(NamedTuple):
    out: torch.Tensor  # [rows, topk] int32 physical KV slots, -1 padded
    num_repaired: torch.Tensor  # [1] int32: rows rescored on the device with tau = -inf
    repaired_rows: torch.Tensor  # [rows] int32, first num_repaired valid
    tau: torch.Tensor
    cand_count: torch.Tensor


def fused_indexer_topk(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    lengths: torch.Tensor,
    row_to_batch: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    topk: int,
    *,
    sample_stride: int = 16,
    rank: int = 192,
    cand_cap: int = 16384,
    keep_all: Optional[int] = None,
    lpt: bool = False,
    select_keys: int = SELECT_KEYS,
    bf16_reduce: bool = False,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    stats: Optional[torch.Tensor] = None,
    cuh_path: Optional[str] = None,
    serving: bool = False,
) -> FusedTopKResult:
    """Exact top-k of every packed row, written to ``out``; no host sync.

    ``workspace`` (uint8, at least ``rows * scratch_bytes_per_row + workspace_fixed_bytes``)
    holds every scratch buffer, so a call allocates nothing large. ``stats``: optional
    [NUM_STATS] int64 device counters. ``lpt``: visit q-blocks longest span first (else in
    order; the kernel deals them out snake-wise either way). ``bf16_reduce``: score with
    DeepGEMM's BF16 head reduction (the exact top-k of its BF16 logits) instead of the FP32 one."""
    dev = q_fp8.device
    ws = _Carver(workspace)
    rows = q_fp8.shape[0]
    num_kv = kv_fp8.shape[0]
    sub_cap = _sub_cap(cand_cap)
    # Rows this short keep every score: at most keep_all / SUB_LISTS per sub-list.
    keep_all = cand_cap // 2 if keep_all is None else keep_all
    jit = _jit_module(cuh_path, serving)
    layout = BF16_LAYOUT if bf16_reduce else DEFAULT_LAYOUT
    common = dict(layout=layout, cuh_path=cuh_path, serving=serving, stats=stats, topk=topk)
    i32, i64 = _empty(dev, torch.int32), _empty(dev, torch.int64)
    st = stats if stats is not None else i64
    sms = _num_sms(dev)

    # Sample keys: every sample_stride-th flattened position; a row's window [ks, ke)
    # holds the samples [ceil(ks / s), ceil(ke / s)) (the kernel divides).
    n_s = (num_kv + sample_stride - 1) // sample_stride
    kv_s = ws.take((n_s, kv_fp8.shape[1]), kv_fp8.dtype, dev)
    kv_s.copy_(kv_fp8[::sample_stride])
    sc_s = ws.take((n_s,), kv_scale.dtype, dev)
    sc_s.copy_(kv_scale[::sample_stride])
    sample_order = order = None
    if lpt:
        sks = (ks + sample_stride - 1) // sample_stride
        sample_order = lpt_order(sks, (ke + sample_stride - 1) // sample_stride, n_s, layout)
        order = lpt_order(ks, ke, num_kv, layout)
    tau = ws.take((rows,), torch.float32, dev)
    score(
        q_fp8, kv_s, sc_s, weights, ks, ke, mode=MODE_TAU, order=sample_order, tau_out=tau,
        lengths=lengths, rank=rank, keep_all=keep_all, ks_div=sample_stride, **common,
    )

    # Select keys: (order-preserving score bits << 32) | ~window column, one sub-list per warp.
    cand_key = ws.take((rows, SUB_LISTS, sub_cap), torch.int64, dev)
    counts = ws.take((rows * SUB_LISTS + 1,), torch.int32, dev)
    counts.zero_()
    cand_count = counts[: rows * SUB_LISTS].view(rows, SUB_LISTS)
    num_repair = counts[rows * SUB_LISTS:]
    score(
        q_fp8, kv_fp8, kv_scale, weights, ks, ke, mode=MODE_FILTER, order=order,
        tau=tau, cand_key=cand_key, cand_count=cand_count, **common,
    )
    # The select kernel writes rows of exactly topk entries.
    dst = out if out is not None and out.is_contiguous() else None
    if dst is None:
        dst = torch.empty(rows, topk, device=dev, dtype=torch.int32)
    repair_rows = ws.take((rows,), torch.int32, dev)
    jit.select(cand_key, cand_count, lengths, row_to_batch, page_table, dst,
               repair_rows, num_repair, i32, i32, st, page_size, sms, select_keys)

    # Device-side repair of the listed rows (overflowed, or tau too high): rescored by the
    # compacting filter and selected again; three launches that do nothing when the list is empty.
    q_c = ws.take((rows, 32, 128), q_fp8.dtype, dev)
    w_c = ws.take((rows, 32), torch.float32, dev)
    ks_c = ws.take((rows,), torch.int32, dev)
    ke_c = ws.take((rows,), torch.int32, dev)
    tau_c = ws.take((rows,), torch.float32, dev)
    jit.repair_prep(repair_rows, num_repair, q_fp8, weights, ks, ke, q_c, w_c, ks_c, ke_c, tau, tau_c,
                    cand_count, st, sub_cap, 4 * sms)
    score(
        q_c, kv_fp8, kv_scale, w_c, ks_c, ke_c, mode=MODE_FILTER_COMPACT, tau=tau_c,
        cand_key=cand_key, cand_count=cand_count, row_map=repair_rows, num_rows=num_repair, **common,
    )
    jit.select(cand_key, cand_count, lengths, row_to_batch, page_table, dst,
               i32, i32, repair_rows, num_repair, st, page_size, 4 * sms, SELECT_REPAIR_KEYS)
    if out is None:
        out = dst
    elif dst is not out:
        out.copy_(dst)
    return FusedTopKResult(out, num_repair, repair_rows, tau, cand_count)


class DeviceStatsReader:
    """Reads the [NUM_STATS] device counters without a host sync: a snapshot is copied to
    pinned memory behind an event and read once the event has completed."""

    def __init__(self, device: torch.device):
        self.stats = torch.zeros(NUM_STATS, device=device, dtype=torch.int64)
        self._host = torch.zeros(NUM_STATS, dtype=torch.int64, pin_memory=True)
        self._event: Optional[torch.cuda.Event] = None
        self.last: Optional[list] = None

    def poll(self) -> Optional[list]:
        """The newest completed snapshot (and starts the next one)."""
        if self._event is not None and self._event.query():
            self.last = self._host.tolist()
            self._event = None
        if self._event is None:
            self._host.copy_(self.stats, non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record()
        return self.last


class FusedTopKStats:
    """Process-wide counters of the serving path, logged every ``log_interval`` calls. Host
    counters are exact; the device counters (repaired rows, compactions) come from the newest
    completed snapshot, so they may trail by one interval."""

    _DEVICE_KEYS = ("dev_calls", "repair_rows", "max_repair", "compactions", "unrepaired", "repair_calls")

    def __init__(self, log_interval: int, check_interval: int):
        self.log_interval = log_interval
        self.check_interval = check_interval
        self.total = dict(calls=0, rows=0, long_rows=0, chunks=0, budget_mb=0, gated_calls=0, gated_rows=0,
                          check_rows=0, check_differ=0, check_tie_explained=0, check_tie_checked=0)
        self.window = dict(self.total)
        self.events = 0
        self.device: Optional[DeviceStatsReader] = None

    def device_stats(self, device: torch.device) -> torch.Tensor:
        if self.device is None:
            self.device = DeviceStatsReader(device)
        return self.device.stats

    def _tick(self) -> None:
        self.events += 1
        if self.log_interval and self.events % self.log_interval == 0:
            snap = self.device.poll() if self.device is not None else None
            dev = dict(zip(self._DEVICE_KEYS, snap)) if snap is not None else {}
            logger.info("FSTK window %s total %s device %s", self.window, self.total, dev)
            self.window = {k: 0 for k in self.window}

    def record(self, *, rows: int, long_rows: int, chunks: int = 1, budget: int = 0) -> bool:
        """Counts one call; True when this call should also run the stock-path check."""
        for d in (self.total, self.window):
            d["calls"] += 1
            d["rows"] += rows
            d["long_rows"] += long_rows
            d["chunks"] += chunks
            d["budget_mb"] = budget >> 20
        self._tick()
        return bool(self.check_interval) and self.total["calls"] % self.check_interval == 0

    def record_gated(self, rows: int) -> None:
        """A call left to the unfused path because its rows are too short to pay."""
        for d in (self.total, self.window):
            d["gated_calls"] += 1
            d["gated_rows"] += rows
        self._tick()

    def record_check(self, *, rows: int, differ: int, tie_explained: int, tie_checked: int) -> None:
        for d in (self.total, self.window):
            d["check_rows"] += rows
            d["check_differ"] += differ
            d["check_tie_explained"] += tie_explained
            d["check_tie_checked"] += tie_checked
        logger.info("FSTK check rows=%d differ=%d tie_explained=%d/%d", rows, differ, tie_explained, tie_checked)
