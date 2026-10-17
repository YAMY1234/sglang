"""P48 uses the native AGG full-prompt state; P31 keeps its boundary adapter."""
import logging
import os
import inspect
import atexit
from functools import wraps

import torch

from .gdn_prefill_batch_graph import BatchCollector, PrefillBatchGraph
from .gdn_prefill_recipe_guard import warn_install_rejection

FLAG = "SGLANG_GDN_PREFILL_AGG_CONTRACT"
AGG_FLAG = "SGLANG_GDN_AGG_FULLN_PREFILL"
# K4.2 (docs/223): AGG prompt-end k31 on the publication side stream after logits.
AGG_DEFERRED_FLAG = "SGLANG_GDN_AGG_BATCH_PUBLISH_DEFERRED"
# K4.1 prerequisite (docs/223): dump the first N prompt-end dense states (debug only).
STATE_DUMP_DIR = "SGLANG_GDN_K31_STATE_DUMP_DIR"
STATE_DUMP_MAX = "SGLANG_GDN_K31_STATE_DUMP_MAX"
# K4.2 checksum replay (debug only): a control file read per full-N forward. Its
# first character selects the route ("0" = forward stream, other = side stream);
# "c" enables the factor-row digests and "d" the state dump for that forward;
# "n" keeps the batch on the original N-1 + boundary route (full-N A/B).
K42_CONTROL = "SGLANG_GDN_K42_CONTROL"
K42_CHECKSUM = "SGLANG_GDN_K42_CHECKSUM"
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


class StateDump:
    """Writes the collected per-layer prompt-end states before factorization.

    Host-synchronizing; only for the first SGLANG_GDN_K31_STATE_DUMP_MAX full-N
    forwards (default 8) of a dump run, never in a measured window.
    """
    def __init__(self, pool, directory, limit):
        self.pool, self.directory, self.limit = pool, directory, limit
        self.count = 0
        os.makedirs(directory, exist_ok=True)

    @classmethod
    def from_env(cls, pool):
        directory = os.environ.get(STATE_DUMP_DIR)
        if not directory:
            return None
        return cls(pool, directory, int(os.environ.get(STATE_DUMP_MAX, "8")))

    def __call__(self, plan, forward_batch, token_count):
        if self.count >= self.limit:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        p = self.pool

        def stack(index):
            if any(pair[index] is None for pair in plan.pending):
                return None
            return torch.stack([pair[index].float() for pair in plan.pending], dim=1).cpu()

        def host(name):
            value = getattr(forward_batch, name, None)
            return None if value is None else [int(x) for x in value]

        payload = dict(
            S=stack(0), S_tracked=stack(1), vbar=p.vbar.float().cpu(),
            omega=None if getattr(p, "k31_omega", None) is None else p.k31_omega.float().cpu(),
            meta=dict(rank=rank, layer_ids=list(p.layer_ids), r=p.cfg.r, rmax=p.cfg.rmax,
                      rows=int(plan.slots.numel()), token_count=int(token_count),
                      extend_lengths=host("extend_seq_lens_cpu"),
                      prefix_lengths=host("extend_prefix_lens_cpu")))
        path = os.path.join(self.directory, f"k31dump-r{rank}-{os.getpid()}-{self.count:03d}.pt")
        torch.save(payload, path)
        self.count += 1
        logger.info("K31_STATE_DUMP %s rows=%d tokens=%d", path, payload["meta"]["rows"], token_count)


class FactorChecksum:
    """Bit-pattern digests of the a/U/W/count rows a k31 publication wrote.

    Enqueued on the stream that replayed the factorization, before any reader
    join; read back without blocking once complete (debug replay only).
    """
    def __init__(self, pool):
        self.pool, self.serial, self.pending = pool, 0, []
        import atexit as _atexit
        _atexit.register(self.drain, True)

    @staticmethod
    def digest(tensor, index, valid):
        rows = tensor.index_select(1, index).contiguous()
        bits = rows.view(torch.int16 if rows.element_size() == 2 else torch.int32).to(torch.int64)
        bits = (bits * valid.view(1, -1, *([1] * (bits.dim() - 2)))).reshape(-1)
        weight = torch.arange(bits.numel(), device=bits.device, dtype=torch.int64) % 65521 + 1
        return torch.stack((bits.sum(), (bits * weight).sum()))

    def enqueue(self, slots, route, rows):
        p = self.pool
        index = torch.cat([s.reshape(-1).to(torch.long) for s in slots if s is not None])
        valid = (index >= 0).to(torch.int64)
        index = index.clamp_min(0)
        sums = torch.cat([self.digest(t, index, valid) for t in (p.a, p.U, p.W, p.count)])
        done = torch.cuda.Event()
        done.record()
        self.serial += 1
        self.pending.append((self.serial, route, rows, sums, done))

    def drain(self, block=False):
        while self.pending and (block or self.pending[0][4].query()):
            serial, route, rows, sums, done = self.pending.pop(0)
            done.synchronize()
            logger.info("K42_CHECKSUM serial=%d route=%s rows=%d sums=%s", serial, route, rows,
                        ",".join(str(v) for v in sums.tolist()))


# docs/219 §8 in-process A/B (ab_measure, scope='prefill'): host-only route
# override set by apply_A / apply_B at extend-forward entry; forward() latches
# it once into its local route. "forward" = today's forward-stream k31,
# None = the installed publication (K4.2 side stream, launched where the
# SGLANG_GDN_K42_LAUNCH_AT_LAST_LAYER default says), "late" / "early" = that
# publication launched after logits / from the last GDN layer's add (P1,
# docs/229 §3). Joins stay unconditional.
_AB_ROUTE = {"value": None, "logged": set()}
K42_ROUTES = ("forward", "late", "early", None)


def set_k42_route(route):
    if route not in K42_ROUTES:
        raise ValueError("K4.2 A/B route must be 'forward', 'late', 'early' or None")
    _AB_ROUTE["value"] = route
    if route not in _AB_ROUTE["logged"]:
        _AB_ROUTE["logged"].add(route)
        logger.info("K42_AB_ROUTE_ACTIVE route=%s", route or "side")


def launch_early(route, default, *, digest=None, dump=None, flags=None, graph=None):
    """P1: whether this forward's publication launches from the last layer's add.

    Off for the forward-stream route, the debug digest / state dump / control
    file (they read the collected states or pick the route after the trunk)
    and without the compact workspace (collected states must be slab views).
    """
    if route is None or digest is not None or dump is not None or flags is not None:
        return False
    ab = _AB_ROUTE["value"]
    early = ab == "early" or (ab is None and default)
    if early and getattr(graph, "workspace", None) is None:
        if not _AB_ROUTE.get("no_workspace_logged"):
            _AB_ROUTE["no_workspace_logged"] = True
            logger.warning("K42 launch at last layer needs SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS=1; "
                           "launching after logits")
        return False
    return early


# docs/219 §8.3 dual prewarm (docs/223 §9.1 stage 1): with SGLANG_GDN_K31_AB_DUAL=1 the
# whole-prefix k31 graphs are captured twice, with the env switch named by
# SGLANG_GDN_K31_AB_ENV (default SGLANG_GDN_K31_FUSED) at "0" (label A) and "1"
# (label B), sharing the static input buffers. set_k31_ab_label picks the replayed
# set per extend forward (host only) and mirrors the switch into os.environ so
# eager fallback paths follow the same label.
K31_AB_DUAL = "SGLANG_GDN_K31_AB_DUAL"
K31_AB_ENV = "SGLANG_GDN_K31_AB_ENV"
_K31_AB = {"label": None, "logged": set()}


def k31_ab_env():
    return os.environ.get(K31_AB_ENV, "SGLANG_GDN_K31_FUSED")


def set_k31_ab_label(label):
    if label not in ("A", "B"):
        raise ValueError("k31 A/B label must be 'A' or 'B'")
    _K31_AB["label"] = label
    os.environ[k31_ab_env()] = "1" if label == "B" else "0"
    if label not in _K31_AB["logged"]:
        _K31_AB["logged"].add(label)
        logger.info("K31_AB_LABEL_ACTIVE label=%s %s=%s", label, k31_ab_env(), os.environ[k31_ab_env()])


def k31_ab_graph(default, dual):
    """The replay graph for the current label (dual = {'A': g, 'B': g} or None)."""
    if dual is None or _K31_AB["label"] is None:
        return default
    return dual[_K31_AB["label"]]


def capture_dual(build):
    """build() -> graph; returns {'A': ..., 'B': ...} captured with the switch at 0 / 1."""
    name = k31_ab_env()
    saved = os.environ.get(name)
    out = {}
    try:
        for label, value in (("A", "0"), ("B", "1")):
            os.environ[name] = value
            out[label] = build()
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved
    logger.info("K31_AB_DUAL_PREWARM %s: A=0 B=1 (two graph sets, shared static inputs)", name)
    return out


def _control_route(path):
    try:
        with open(path) as stream:
            return stream.read().strip()
    except OSError:
        return ""


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

    control = os.environ.get(K42_CONTROL) if agg_mode else None

    def select(batch):
        selected = (agg_enabled() and agg_eligible(batch) if agg_mode
                    else enabled() and eligible(batch))
        if selected and control is not None and "n" in _control_route(control):
            selected = False  # debug in-process A/B: original route for this batch
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
        return result

    @classmethod
    @wraps(legacy_init)
    def initialize(cls, batch, model_runner, **kwargs):
        result = legacy_init(cls, batch, model_runner, **kwargs)
        if overlap is not None and not result.forward_mode.is_extend():
            # Decode keeps the original overlap path: no snapshot, event,
            # request walk, or pool-plan mutation from this opt-in contract.
            return result
        selected = select(result)
        record = None
        if overlap is not None:
            record = getattr(batch, "fulln_overlap_record", None)
            if record is None:
                record = overlap.begin(batch, select(batch))
            # A late view may reject a plan but must not newly select full-N
            # after checkpoint ownership was prepared for the fallback.
            selected = selected and record.selected
        if agg_mode and selected:
            trunk = model_runner.model._prefill_runners.get("trunk")
            # Decide before backend planning so a graph rejection restores the
            # original N-1 checkpoint and reserves ownership only once.
            selected = trunk is not None and trunk.can_run(result)
        changes = []
        for i, req in enumerate(batch.reqs):
            if record is not None:
                saved = record.tracks.get(id(req))
                snapshot = None if saved is None else (batch, saved.before, saved.selected)
            else:
                snapshot = getattr(req, "_pfactor_track_before", None)
            if snapshot is not None:
                source, before, previous = snapshot
                if previous != selected:
                    for name, value in zip(_TRACK_FIELDS, before, strict=True):
                        setattr(req.kv, name, value)
                    entry = prepare_track(source, req, selected)
                    changes.append((i, entry))
                if record is None:
                    del req._pfactor_track_before
                else:
                    from .gdn_fulln_overlap import TrackSnapshot
                    record.tracks[id(req)] = TrackSnapshot(
                        req, before, tuple(getattr(req.kv, name) for name in _TRACK_FIELDS), selected)
            if record is None:
                req._pfactor_agg_contract = selected
            if selected:
                req.factored_prefill_boundary_steps = 0
        if changes:
            # TBO/mixed decisions can arrive after checkpoint preparation.
            for name, field in (("mamba_track_mask", "track_mask"),
                                ("mamba_track_indices", "track_index"),
                                ("mamba_track_seqlens", "track_seqlen")):
                source = getattr(batch, name).clone()
                for i, entry in changes:
                    source[i] = getattr(entry, field)
                setattr(batch, name, source)
                setattr(result, name, source.to(model_runner.device))
        if record is not None:
            record.seal(selected)
            # Rejected/mixed/TBO batches use the original result path; the
            # per-batch restore above already selected their N-1 checkpoints.
            batch.fulln_overlap_record = record if selected else None
            result.fulln_overlap_record = record if selected else None
        result._pfactor_agg_contract = selected
        if not agg_mode and result.forward_mode.is_extend():
            result.pd_factor_only_full_batch = True
            result._pfactor_legacy_mixed = result.forward_mode.is_mixed()
            if result._pfactor_legacy_mixed:
                # Eager normalizes MIXED to EXTEND after checkpoint selection.
                result.twinstar_prompt_final = [
                    req.extend_range.end >= len(req.origin_input_ids)
                    for req in batch.reqs
                ]
        return result

    @wraps(legacy_metadata)
    def metadata(self, batch):
        if not getattr(batch, "_pfactor_agg_contract", False):
            return legacy_metadata(self, batch)
        previous = getattr(batch, "pd_factor_only_full_batch", False)
        batch.pd_factor_only_full_batch = False
        try:
            output = legacy_metadata(self, batch)
            if overlap is not None:
                record = batch.fulln_overlap_record
                if record.plan is not None:
                    raise RuntimeError("full-N immutable batch planned more than once")
                record.plan = self.forward_metadata.factored_extend
            return output
        finally:
            batch.pd_factor_only_full_batch = previous

    @wraps(legacy_send)
    def before_send(self, req):
        if not getattr(req, "_pfactor_agg_contract", False):
            return legacy_send(self, req)
        req.factored_prefill_boundary_steps = 0
        return native_send(self, req)

    schedule_cls._mamba_radix_cache_v2_req_prepare_for_extend = track
    forward_cls.init_new = initialize
    backend_cls.init_forward_metadata = metadata
    if not agg_mode:
        handoff_cls.before_send = before_send
    forward_cls._pfactor_agg_contract_installed = True


@warn_install_rejection("full-N")
def install(runner):
    owner = runner.model
    role = runner.server_args.disaggregation_mode
    agg_mode = role == "null"
    if agg_mode:
        if not agg_enabled():
            return
    elif (role != "prefill" or os.environ.get(FLAG, "1") != "1"
          or os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL") != "1"
          or getattr(owner, "pd_shallow_role", None) == "prefill"):
        return
    if getattr(owner, "_pfactor_agg_installed", False):
        return
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.disaggregation.state_handoff import FactorStateHandoff
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.runtime_context import get_schedule

    pool = runner.req_to_token_pool.factored_gdn_pool
    if (getattr(owner, "n_layers", None) != 48
            or list(pool.layer_ids) != [i for i in range(48) if i % 4 != 3]
            or not pool.cfg.strict_chunk
            or not pool.cfg.factored_prefix or pool.cfg.init_method != "k31"
            or pool.prefix_dense is not None or not pool.batch_prefill
            or (not agg_mode and not getattr(owner, "_exact_tail_installed", False))):
        raise ValueError("AGG contract requires the full-depth strict k31 P48 recipe")
    if owner.fullstack:
        from sglang.srt.models.flash_next_duet.pd_shallow_install import factor_only_contract

        factor_only_contract(owner)
    elif owner.emitters or getattr(owner, "twinstar", None) is not None:
        raise ValueError("AGG contract requires native DUET P48 or legacy factor-only P48")
    from sglang.srt.environ import envs

    overlap_enabled = (agg_mode and not get_schedule().disable_overlap_schedule
                       and envs.SGLANG_GDN_AGG_FULLN_OVERLAP_OK.get())
    if overlap_enabled and runner.server_args.dp_size != 1:
        raise ValueError("AGG full-N overlap first version requires DP1")
    if agg_mode:
        if (not owner.fullstack or (not get_schedule().disable_overlap_schedule and not overlap_enabled)
                or runner.server_args.is_embedding or runner.server_args.pp_size != 1
                or runner.server_args.speculative_algorithm
                or os.environ.get("SGLANG_GDN_PREFILL_COMMIT_GRAPH") != "1"
                or os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL") == "1"
                or os.environ.get("SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH") == "1"):
            raise ValueError("AGG full-N requires native P48 generation PP1, isolated scheduling or OVERLAP_OK=1, "
                             "COMMIT_GRAPH=1 and no PD factor-only/exact-tail adapter")
    # The flag-off external model delegates here; wrapper depth is not an ABI.
    from sglang.srt.environ import envs

    workspace_limits = None
    if agg_mode and envs.SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS.get():
        from .gdn_fulln_workspace import row_capacity

        chunk = runner.server_args.chunked_prefill_size
        capacity = row_capacity(chunk, runner.server_args.max_running_requests)
        workspace_limits = capacity, chunk
        pool._agg_fulln_workspace_limits = workspace_limits
    overlap = None
    if overlap_enabled:
        from .gdn_fulln_overlap import FullNOverlap
        overlap = FullNOverlap(runner.device)
        pool._agg_fulln_overlap = overlap
    native_forward = owner.model.forward
    publication = None
    if not agg_mode:
        from sglang.srt.environ import envs
        if (envs.SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD.get()
                and not envs.SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED.get()):
            raise ValueError("PD publish join offload requires PD_BATCH_PUBLISH_DEFERRED=1")
        if envs.SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED.get():
            from .gdn_pd_publication import install as install_publication
            install_publication(pool, runner)
            publication = pool._pd_batch_publication
    elif os.environ.get(AGG_DEFERRED_FLAG, "0") == "1":
        from sglang.srt.runtime_context import get_memory

        if get_memory().enable_hierarchical_cache:
            # The factor HiCache backup (pool_host/flashnext_factored.py) does
            # not join the side stream yet: keep the forward-stream publication.
            logger.warning("%s=1 ignored with --enable-hierarchical-cache: k31 publication stays "
                           "on the forward stream", AGG_DEFERRED_FLAG)
        else:
            from .gdn_pd_publication import install as install_publication
            install_publication(pool, runner, agg=True)
            publication = pool._pd_batch_publication
            # Fallback batches (original N-1 route) keep the forward-stream k31:
            # their boundary decode reads these factors inside the same forward.
    launch_at_last_layer = False
    if agg_mode and envs.SGLANG_GDN_K42_LAUNCH_AT_LAST_LAYER.get():
        if publication is None:
            logger.warning("SGLANG_GDN_K42_LAUNCH_AT_LAST_LAYER=1 ignored: needs %s=1", AGG_DEFERRED_FLAG)
        else:
            launch_at_last_layer = True
            logger.info("K42 launch at last layer enabled (docs/229 P1): AGG full-N k31 publication launches "
                        "from the last GDN layer's collector add, controls bound on the first add; "
                        "debug dump/digest/control forwards keep the after-logits launch")
    state_dump = StateDump.from_env(pool) if agg_mode else None
    k42_control = os.environ.get(K42_CONTROL) if agg_mode else None
    checksum = FactorChecksum(pool) if agg_mode and os.environ.get(K42_CHECKSUM) == "1" else None
    install_contracts(ForwardBatch, ScheduleBatch, GDNAttnBackend, FactorStateHandoff,
                      agg_mode=agg_mode, workspace_limits=workspace_limits, overlap=overlap)
    legacy_forward = owner.forward
    summary = FullNSummary(role, envs.SGLANG_GDN_AGG_FULLN_LOG_INTERVAL.get())
    owner._agg_fulln_summary = summary
    # Workers that exit normally get a final partial interval. Abruptly killed
    # workers need interval=1 (or an explicit summary.log at the window boundary).
    atexit.register(summary.log)

    if agg_mode:
        legacy_prepare = owner.prepare_forward_batch

        @wraps(legacy_prepare)
        def prepare(batch):
            if getattr(batch, "_pfactor_agg_contract", False) and not get_is_capture_mode():
                # #17 otherwise defers the plan for its N-1 sub-batch. This
                # route needs exactly one full-N plan before entering layers.
                batch._twinstar_defer_factor_plan = False
                batch._twinstar_prefill_eligible = False
                return
            return legacy_prepare(batch)

        owner.prepare_forward_batch = prepare

    @wraps(legacy_forward)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        if get_is_capture_mode():
            return legacy_forward(input_ids, positions, forward_batch, *args, **kwargs)
        if not getattr(forward_batch, "_pfactor_agg_contract", False):
            output = legacy_forward(input_ids, positions, forward_batch, *args, **kwargs)
            if forward_batch.forward_mode.is_extend() or forward_batch.forward_mode.is_mixed():
                summary.record(forward_batch.batch_size, fallback=True)
            return output
        if not (agg_eligible(forward_batch) if agg_mode else eligible(forward_batch)):
            raise RuntimeError("AGG batch contract changed after checkpoint planning")
        linear = get_attn_backend().linear_attn_backend
        plan = (forward_batch.fulln_overlap_record.plan if overlap is not None
                else linear.forward_metadata.factored_extend)
        if overlap is not None and linear.forward_metadata.factored_extend is not plan:
            raise RuntimeError("full-N backend metadata does not match its batch plan")
        if plan is None or getattr(pool, "_exact_tail_transaction", None) is not None:
            raise RuntimeError("AGG prefill needs one native full-N plan without a tail transaction")
        route = publication
        if _AB_ROUTE["value"] == "forward":
            route = None  # ab_measure label A: today's forward-stream k31 for this forward
        flags = None if k42_control is None else _control_route(k42_control)
        if flags is not None and flags.startswith("0"):
            route = None  # in-process A/B: this forward publishes on the forward stream
        digest = checksum if checksum is not None and (flags is None or "c" in flags) else None
        dump = state_dump if state_dump is not None and (flags is None or "d" in flags) else None
        if checksum is not None:
            checksum.drain()
        graph = k31_ab_graph(pool._agg_prefill_graph, getattr(pool, "_agg_prefill_graph_ab", None))
        early = agg_mode and launch_early(route, launch_at_last_layer, digest=digest, dump=dump,
                                          flags=flags, graph=graph)
        collector = BatchCollector(pool, plan, graph=graph, token_count=input_ids.shape[0],
                                   publication=route, launch_at_last_layer=early)
        with collector:
            if early:
                # Set before the trunk: the launch happens inside it. Debug
                # digests keep the after-logits launch (launch_early).
                route.after_replay = None
            if agg_mode:
                trunk = owner._prefill_runners["trunk"]
                if not trunk.can_run(forward_batch):
                    raise RuntimeError("AGG full-N graph eligibility changed after planning")
                # Same whole-token trunk and output tail as the all-off graph.
                # Attention breaks still execute in Python and fill the collector.
                graph_output = trunk.run(forward_batch)
                streams = graph_output[0] if isinstance(graph_output, tuple) else graph_output
                owner.model.model.last_hc_hidden_states = streams
                hidden, _ = owner.model.model.hyper_connection_mixer.mix(streams)
                output = owner.model.logits_processor(
                    input_ids, hidden, owner.model.lm_head, forward_batch)
                output.hidden_states = streams
                owner._agg_fulln_trunk_replays += 1
            else:
                # PD's admitted full-N route has already selected its checkpoint
                # extent. Reuse that collector/plan for an opt-in owner trunk;
                # a missing or rejected graph keeps the same full-N eager route.
                trunk = (
                    owner._pd_trunk_prefill_graph_runner(
                        forward_batch, kwargs.get("get_embedding", False),
                        kwargs.get("pp_proxy_tensors"),
                    )
                    if getattr(owner, "pd_trunk_prefill_graph", False) and not args
                    else None
                )
                if trunk is not None:
                    output = owner._pd_trunk_prefill_graph_forward(
                        trunk, input_ids, forward_batch
                    )
                    owner._agg_fulln_trunk_replays += 1
                else:
                    output = native_forward(input_ids, positions, forward_batch, *args, **kwargs)
            if linear.forward_metadata.factored_extend is not plan:
                raise RuntimeError("native AGG forward replaced its full-N state plan")
            if dump is not None:
                dump(plan, forward_batch, input_ids.shape[0])
        if agg_mode and route is not None:
            if digest is not None:
                written = (plan.slots, collector.controls[0])
                route.after_replay = lambda: digest.enqueue(written, "side", plan.slots.numel())
            else:
                route.after_replay = None
        elif digest is not None:
            digest.enqueue((plan.slots, collector.controls[0]), "forward", plan.slots.numel())
        if agg_mode and route is not None:
            # K4.2: launch first, then hand the side-stream event to the CPU
            # ownership fences in place of a forward-stream event.
            route.start_after_forward()
            if overlap is not None:
                overlap.publish(forward_batch.fulln_overlap_record,
                                done=route.transfer_event())
        elif overlap is not None:
            overlap.publish(forward_batch.fulln_overlap_record)
        if not agg_mode and publication is not None:
            # The full-N trunk, mixer and logits have all been enqueued. The
            # side stream waits for them. Readers retain either the original
            # join or an opt-in slot/transport fence before consuming factors.
            publication.start_after_forward()
        # The graph publishes r, with no boundary update or manual count edit.
        forward_batch.factored_prefill_boundary_steps = 0
        owner._agg_fulln_prefills += 1
        summary.record(forward_batch.batch_size, trunk=agg_mode, published=collector.published)
        return output

    owner.forward = forward
    owner._pfactor_agg_installed = True
    owner._agg_fulln_prefills = 0
    owner._agg_fulln_trunk_replays = 0
    pool._agg_prefill_enabled = True
    logger.info("GDN P48 AGG contract installed: role=%s full-N, deferred commit, phase=0 count=%d; "
                "original fallback for mixed/TBO/empty-prefix batches", role, pool.cfg.r)


def prewarm(pool):
    if not getattr(pool, "_agg_prefill_enabled", False):
        return
    from .gdn_factored_pool import (
        ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense, factorize_layers,
    )
    tail_graph = getattr(pool, "_prefill_batch_graph", None)
    limits = getattr(pool, "_agg_fulln_workspace_limits", None)
    workspace = None
    if limits is not None:
        from .gdn_fulln_workspace import FullNWorkspace, unique_state_bytes

        workspace = FullNWorkspace(pool, *limits)
        logger.info("GDN full-N workspace: row_capacity=%d chunk_tokens=%d "
                    "state_bytes=%d (two shared slabs, independent of trunk token buckets)",
                    limits[0], limits[1], unique_state_bytes(workspace.shared))
    shared = (workspace.shared if workspace is not None else
              None if tail_graph is None else tail_graph.shared)

    def build():
        graph = PrefillBatchGraph(include_tail=False, workspace=workspace, shared=shared)
        graph.prewarm(pool, eager=factorize_layers,
                      policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense))
        return graph

    if os.environ.get(K31_AB_DUAL, "0") == "1":
        if shared is None:
            shared = {}  # both label sets must bind the same static inputs
        dual = capture_dual(build)
        pool._agg_prefill_graph_ab = dual
        pool._agg_prefill_graph = dual["A"]
    else:
        pool._agg_prefill_graph = build()
