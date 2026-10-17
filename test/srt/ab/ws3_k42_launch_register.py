# docs/219 §8 ab_measure register file (WS-3, P1 docs/229 §3). Executed once per TP process with `register` in
# globals. Needs the service launched with SGLANG_GDN_AGG_BATCH_PUBLISH_DEFERRED=1 and
# SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS=1 on the AGG full-N route (SGLANG_GDN_AGG_FULLN_PREFILL=1); the
# SGLANG_GDN_K42_LAUNCH_AT_LAST_LAYER default does not matter (both labels override it).
# A = K4.2 side-stream publication launched after logits (today), B = launched from the last GDN layer's
# collector add. Host-only, idempotent, no allocation; the route is latched once per extend forward in
# gdn_prefill_agg_contract.forward; joins stay unconditional. Do not also set SGLANG_GDN_K42_CONTROL,
# SGLANG_GDN_K42_CHECKSUM or SGLANG_GDN_K31_STATE_DUMP_DIR (they keep the after-logits launch).
from sglang.srt.mem_cache.gdn_prefill_agg_contract import set_k42_route


def _apply_a():
    set_k42_route("late")


def _apply_b():
    set_k42_route("early")


register("k42_launch", _apply_a, _apply_b, scope="prefill")  # noqa: F821
