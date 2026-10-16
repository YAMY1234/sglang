# docs/219 §8 ab_measure register file (WS-3, R1 / docs/223 §9.1). Executed once per TP process with `register`
# in globals. Requires SGLANG_GDN_K31_AB_DUAL=1 at launch (the k31 graphs are prewarmed once per label with the
# switch named by SGLANG_GDN_K31_AB_ENV, default SGLANG_GDN_K31_FUSED, at 0 / 1). A = switch off, B = switch on;
# host only: picks the replayed graph set and mirrors the switch into os.environ for eager fallbacks.
from sglang.srt.mem_cache.gdn_prefill_agg_contract import set_k31_ab_label


def _apply_a():
    set_k31_ab_label("A")


def _apply_b():
    set_k31_ab_label("B")


register("k31_ab", _apply_a, _apply_b, scope="prefill")  # noqa: F821
