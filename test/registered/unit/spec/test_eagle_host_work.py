import __future__
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load_function(path, name, namespace):
    tree = ast.parse((SRT / path).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    exec(
        compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[name]


class TestEagleHostWork(unittest.TestCase):
    def test_no_growth_skips_h2d_only_when_enabled(self):
        for enabled in (False, True):
            req = NS(decode_batch_idx=0, kv=NS(allocate_spec_decode=Mock()))
            batch = NS(
                reqs=[req],
                seq_lens_cpu=None,
                maybe_evict_swa=Mock(),
                token_to_kv_pool_allocator=NS(page_size=1),
                device="cpu",
                sampling_info=NS(penalizer_orchestrator=NS(is_required=False)),
                tree_cache=object(),
                req_to_token_pool=object(),
                req_pool_indices=object(),
            )
            tensor = Mock()
            tensor.tolist.return_value = [64]
            tensor.max.return_value = 64
            torch = NS(tensor=Mock(return_value=tensor), int32=object())
            alloc = load_function(
                "mem_cache/allocation.py", "alloc_for_spec_decode", {}
            )
            fn = load_function(
                "speculative/eagle_utils.py",
                "eagle_prepare_for_decode",
                {
                    "torch": torch,
                    "_fused_accept_enabled": lambda: enabled,
                    "get_alloc_reserve_per_decode": lambda: 32,
                    "page_aligned_decode_alloc_lens": lambda *a, **kw: ([64], [64], 0),
                    "alloc_for_spec_decode": alloc,
                },
            )
            fn(batch)
            self.assertEqual(tensor.to.call_count, 0 if enabled else 2)
            req.kv.allocate_spec_decode.assert_called_once_with(64)
            batch.maybe_evict_swa.assert_called_once()
            self.assertEqual((req.decode_batch_idx, batch.max_seq_len_host), (1, 64))


if __name__ == "__main__":
    unittest.main()
