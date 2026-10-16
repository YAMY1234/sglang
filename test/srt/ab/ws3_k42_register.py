# docs/219 §8 ab_measure register file (WS-3, K4.2). Executed once per TP process with `register` in globals.
# Needs the service launched with SGLANG_GDN_AGG_BATCH_PUBLISH_DEFERRED=1 on the AGG full-N route
# (SGLANG_GDN_AGG_FULLN_PREFILL=1); A = forward-stream k31 (today), B = K4.2 side-stream publication.
# Host-only, idempotent, no allocation; the route is latched once per extend forward in
# gdn_prefill_agg_contract.forward; joins stay unconditional. Do not also set SGLANG_GDN_K42_CONTROL.
from sglang.srt.mem_cache.gdn_prefill_agg_contract import set_k42_route


def _apply_a():
    set_k42_route("forward")


def _apply_b():
    set_k42_route(None)


register("k42_publication_route", _apply_a, _apply_b, scope="prefill")  # noqa: F821
