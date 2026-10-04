"""Startup-only recipe receipts. Observe installers without changing policy."""
from functools import wraps
import json
import logging
import os

logger = logging.getLogger(__name__)
EXACT = "SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH"
FACTOR_ONLY = "TWINSTAR_PD_FACTOR_ONLY_TAIL"
COMMIT = "SGLANG_GDN_PREFILL_COMMIT_GRAPH"
FULL_N = "SGLANG_GDN_PREFILL_AGG_CONTRACT"
AGG_FULL_N = "SGLANG_GDN_AGG_FULLN_PREFILL"
SWITCHES = (EXACT, FACTOR_ONLY, COMMIT, FULL_N, AGG_FULL_N)


def _context(runner):
    owner = runner.model
    pool = getattr(runner.req_to_token_pool, "factored_gdn_pool", None)
    fullstack = getattr(owner, "fullstack", None)
    rank = getattr(getattr(pool, "cfg", None), "r", 0)
    if not rank and isinstance(fullstack, dict):
        rank = fullstack.get("gdn_rank", 0)
    return dict(role=runner.server_args.disaggregation_mode, factor=bool(rank),
                rank=rank, shallow=getattr(owner, "pd_shallow_role", None) == "prefill")


def _missing(route, context):
    """Report prerequisites, not a substitute for an installer's eligibility gate."""
    required = []
    if context["role"] == "prefill":
        required = [EXACT, COMMIT]
        if not context["shallow"]:
            required.append(FACTOR_ONLY)
        if route == "full-N":
            required.append(FULL_N)
    elif context["role"] == "null" and route == "full-N":
        required = [AGG_FULL_N, COMMIT]
    return [key + "=1" for key in required
            if os.environ.get(key, "1" if key == FULL_N else "0") != "1"]


def _emit(route, context, *, installed, status, reason, warning):
    # json escapes embedded newlines in exceptions: one grep-able startup line.
    record = dict(route=route, **context, installed=bool(installed), status=status,
                  reason=reason, missing_switches=_missing(route, context) if warning else [],
                  switches={key: os.environ.get(key, "<unset>") for key in SWITCHES})
    log = logger.warning if warning else logger.info
    log("GDN recipe audit: %s %s", "WARN" if warning else "INFO",
        json.dumps(record, sort_keys=True, ensure_ascii=True))


def warn_install_rejection(route):
    """Keep return values, original exception objects and install ordering intact."""
    def decorate(install):
        @wraps(install)
        def observed(runner, *args, **kwargs):
            try:
                return install(runner, *args, **kwargs)
            except Exception as error:
                _emit(route, _context(runner), installed=False, status="rejected",
                      reason=f"{type(error).__name__}: {error}", warning=True)
                raise
        return observed
    return decorate


def warn_factor_only_constructor(initialize):
    """The external legacy guard can reject native DUET before runner init."""
    @wraps(initialize)
    def observed(owner, *args, **kwargs):
        try:
            return initialize(owner, *args, **kwargs)
        except Exception as error:
            if os.environ.get(FACTOR_ONLY, "0") == "1":
                from sglang.srt.runtime_context import get_disagg

                fullstack = getattr(owner, "fullstack", None)
                rank = fullstack.get("gdn_rank") if isinstance(fullstack, dict) else None
                context = dict(role=get_disagg().disaggregation_mode,
                               factor=bool(rank) if rank is not None else "requested",
                               rank=rank,
                               shallow=getattr(owner, "pd_shallow_role", None) == "prefill")
                _emit("exact-tail", context, installed=False, status="constructor-rejected",
                      reason=f"{type(error).__name__}: {error}", warning=True)
            raise
    return observed


def report_startup(runner):
    """Called once after existing installer/prewarm calls; never from forward."""
    context = _context(runner)
    role, factor, shallow = context["role"], context["factor"], context["shallow"]
    exact_requested = os.environ.get(EXACT, "0") == "1"
    agg_requested = os.environ.get(AGG_FULL_N, "0") == "1"
    if not factor and not exact_requested and not agg_requested:
        return
    exact_installed = bool(getattr(runner.model, "_exact_tail_installed", False))
    fulln_installed = bool(getattr(runner.model, "_pfactor_agg_installed", False))
    for route, installed in (("exact-tail", exact_installed), ("full-N", fulln_installed)):
        applicable = ((role == "prefill" and factor) if route == "exact-tail"
                      else (role == "prefill" and factor and not shallow) or (role == "null" and agg_requested))
        requested = exact_requested if route == "exact-tail" else agg_requested
        warning = not installed and (applicable or requested)
        if installed:
            status, reason = "installed", "installer completion marker present"
        elif warning:
            status = "not-installed"
            missing = _missing(route, context)
            reason = ("missing or disabled prerequisites: " + ", ".join(missing) if missing
                      else "installation not reached or unsupported by this role/build; inspect install guard")
        else:
            status, reason = "not-applicable", "no installation requested for this role/arm"
        _emit(route, context, installed=installed, status=status, reason=reason, warning=warning)
