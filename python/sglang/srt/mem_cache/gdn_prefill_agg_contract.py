"""P48 uses the native AGG full-prompt state; P31 keeps its boundary adapter."""
import logging
import os
import inspect
import atexit
from functools import wraps

import torch

from .gdn_prefill_batch_graph import BatchCollector, PrefillBatchGraph

FLAG = "SGLANG_GDN_PREFILL_AGG_CONTRACT"
AGG_FLAG = "SGLANG_GDN_AGG_FULLN_PREFILL"
logger = logging.getLogger(__name__)
_TRACK_FIELDS = ("mamba_last_track_idx", "mamba_next_track_idx", "mamba_last_track_seqlen")


class FullNSummary:
    """Completed prefill forwards only: captures and decode steps are excluded."""
    def __init__(self, role, interval):
        if interval < 1:
            raise ValueError("SGLANG_GDN_AGG_FULLN_LOG_INTERVAL must be positive")
        self.role, self.interval = role, interval
        self.forwards = self.trunk_replays = self.batch_publications = self.fallbacks = 0
        self.rows = 0

    def record(self, rows, *, trunk=False, published=False, fallback=False):
        self.forwards += 1
        self.trunk_replays += int(trunk)
        self.batch_publications += int(published)
        self.fallbacks += int(fallback)
        self.rows = rows
        if self.forwards == 1 or self.forwards % self.interval == 0:
            self.log("periodic")

    def log(self, reason="shutdown"):
        logger.info("GDN full-N prefill: role=%s forwards=%d trunk_replays=%d "
                    "batch_publications=%d fallbacks=%d rows=%d reason=%s",
                    self.role, self.forwards, self.trunk_replays,
                    self.batch_publications, self.fallbacks, self.rows, reason)


def enabled():
    if os.environ.get(FLAG, "1") != "1":
        return False
    marker = os.environ.get(FLAG + "_FILE")
    return not marker or os.path.exists(marker)


def agg_enabled():
    from sglang.srt.environ import envs

    return envs.SGLANG_GDN_AGG_FULLN_PREFILL.get()


def eligible(batch):
    mode = batch.forward_mode
    if (not mode.is_extend() or mode.is_mixed()
            or getattr(batch, "_pfactor_legacy_mixed", False)
            or getattr(batch, "spec_info", None) is not None
            or getattr(batch, "can_run_tbo", False)
            or getattr(batch, "tbo_split_seq_index", None) is not None):
        return False
    if hasattr(batch, "reqs"):
        rows = len(batch.reqs)
        lengths = [req.extend_range.length for req in batch.reqs]
    else:
        rows, lengths = batch.batch_size, batch.extend_seq_lens_cpu
        if lengths is None or sum(lengths) != batch.input_ids.shape[0]:
            return False
    return 0 < rows <= 16 and len(lengths) == rows and all(int(n) > 1 for n in lengths)


def agg_eligible(batch):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return (not get_is_capture_mode() and batch.forward_mode == ForwardMode.EXTEND
            and getattr(batch, "input_embeds", None) is None
            and getattr(batch, "tbo_parent_token_range", None) is None
            and eligible(batch))


def install_contracts(forward_cls, schedule_cls, backend_cls, handoff_cls, *, agg_mode=False,
                      workspace_limits=None, overlap=None):
    """Select the track extent before backend metadata and ring ownership exist."""
    if getattr(forward_cls, "_pfactor_agg_contract_installed", False):
        return
    legacy_track = schedule_cls._mamba_radix_cache_v2_req_prepare_for_extend
    native_track = inspect.unwrap(legacy_track)
    legacy_init = forward_cls.init_new.__func__
    legacy_metadata = backend_cls.init_forward_metadata
    legacy_send = handoff_cls.before_send
    native_send = inspect.unwrap(legacy_send)
    if not agg_mode and (native_track is legacy_track or native_send is legacy_send):
        raise ValueError("P48 AGG contract requires the frozen factor-only wrappers")

    def select(batch):
        selected = (agg_enabled() and agg_eligible(batch) if agg_mode
                    else enabled() and eligible(batch))
        if selected and workspace_limits is not None:
            rows, tokens = workspace_limits
            lengths = ([r.extend_range.length for r in batch.reqs]
                       if hasattr(batch, "reqs") else batch.extend_seq_lens_cpu)
            selected = len(lengths) <= rows and sum(lengths) <= tokens
        return selected

    def prepare_track(batch, req, selected):
        # Native DUET also applies N-1 inside prompt_p_extent. Unwrapping the
        # external factor-only adapter alone does not select full-N tracking.
        if overlap is not None:
            from .gdn_fulln_overlap import track_selection
            with track_selection(req, selected):
                return (native_track if selected else legacy_track)(batch, req)
        req._pfactor_agg_contract = selected
        return (native_track if selected else legacy_track)(batch, req)

    @wraps(legacy_track)
    def track(self, req):
        selected = select(self)
        record = None
        if overlap is not None:
            from .gdn_fulln_overlap import TrackSnapshot
            record = getattr(self, "fulln_overlap_record", None)
            if record is None:
                record = overlap.begin(self, selected)
            if record.sealed or record.selected != selected:
                raise RuntimeError("full-N checkpoint selection changed within a batch")
        before = tuple(getattr(req.kv, name) for name in _TRACK_FIELDS)
        result = prepare_track(self, req, selected)
        if record is not None:
            record.tracks[id(req)] = TrackSnapshot(
                req, before, tuple(getattr(req.kv, name) for name in _TRACK_FIELDS), selected)
        else:
            req._pfactor_track_before = self, before, selected
