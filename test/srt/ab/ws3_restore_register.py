# docs/219 §8 ab_measure register file (WS-3, fused prefix restore, docs/223 §3.9). Executed once per TP process
# with `register` in globals. Launch the service with SGLANG_GDN_PREFIX_RESTORE_FUSED=1 so prewarm JIT-warms the
# fused kernel; PrefillRestoreGraph.run reads the variable on every call. A = restore graph, B = fused kernel.
import os


def _apply_a():
    os.environ["SGLANG_GDN_PREFIX_RESTORE_FUSED"] = "0"


def _apply_b():
    os.environ["SGLANG_GDN_PREFIX_RESTORE_FUSED"] = "1"


register("prefix_restore_fused", _apply_a, _apply_b, scope="prefill")  # noqa: F821
