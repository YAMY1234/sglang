"""CPU-only guard for the opt-in L3 tiering / prefetch-anchor flags.

Regression: ``_prefetch_anchor_full_kv`` was only assigned inside the
``SGLANG_HICACHE_PREFETCH_ANCHOR_FULL_KV`` branch of ``init_hicache`` but read
unconditionally by ``storage_prefetch_anchor``; with the flag unset the first
L3 prefetch raised ``AttributeError``. Every such flag must get a default in
``UnifiedRadixCache.__init__`` at the top level (not under an ``if``).
"""

from __future__ import annotations

import ast
import inspect
import re
import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Attributes introduced by the exclusive-tiering / eager-Mamba / prefetch-anchor
# commits. The regex also catches any future ``self._l3_*`` / ``self._write_behind_*``
# / ``self._prefetch_anchor_*`` read added without a default.
_FLAG_RE = re.compile(r"self\.(_l3_\w+|_write_behind_\w+|_prefetch_anchor_\w+)\b")
_EXPECTED = {
    "_prefetch_anchor_full_kv",
    "_l3_write_on_host_evict",
    "_l3_evict_write_reserve_fraction",
    "_l3_mamba_eager_write",
    "_l3_tier_stats",
    "_write_behind_inflight",
    "_write_behind_step",
    "_write_behind_clean_tokens",
}
# Methods, not state; they are defined on the class, not assigned in __init__.
_METHODS = {"_write_behind_host_tail"}


def _top_level_self_assigns(func) -> set[str]:
    """Names assigned as ``self.<name> = ...`` directly in the function body."""
    src = inspect.getsource(func)
    tree = ast.parse(inspect.cleandoc(src) if src[0] == " " else src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    names: set[str] = set()
    for stmt in fn.body:  # direct children only: no if/for/with/try nesting
        targets = []
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                names.add(t.attr)
    return names


class TestL3FlagDefaults(unittest.TestCase):
    def test_every_flag_read_has_an_unconditional_default(self):
        assigned = _top_level_self_assigns(UnifiedRadixCache.__init__)
        class_src = inspect.getsource(UnifiedRadixCache)
        read = set(_FLAG_RE.findall(class_src)) - _METHODS
        self.assertTrue(_EXPECTED <= read, f"audit list drifted: {_EXPECTED - read}")
        missing = sorted(read - assigned)
        self.assertEqual(
            missing,
            [],
            f"flags read somewhere in UnifiedRadixCache but not assigned at the top "
            f"level of __init__ (conditional-only init => AttributeError): {missing}",
        )

    def test_storage_prefetch_anchor_with_flag_unset_is_identity(self):
        # Stub cache exactly as __init__ leaves the flag when the env var is unset.
        cache = object.__new__(UnifiedRadixCache)
        cache._prefetch_anchor_full_kv = False
        cache.tree_core = None  # must not be touched when the flag is off
        req = SimpleNamespace(full_kv_last_node=None, full_kv_hit_length=0)
        self.assertEqual(
            cache.storage_prefetch_anchor(req, anchor=7, matched_len=3), (7, 3)
        )
        req = SimpleNamespace(full_kv_last_node=42, full_kv_hit_length=128)
        self.assertEqual(
            cache.storage_prefetch_anchor(req, anchor=7, matched_len=3), (7, 3)
        )


if __name__ == "__main__":
    unittest.main()
