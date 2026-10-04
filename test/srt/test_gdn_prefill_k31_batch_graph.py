"""AGG prompt-end k31 commit through PrefillBatchGraph (CPU, no CUDA graph).

Dispatch cases check when the whole-prefix graph replaces the eager last-layer
group. The equivalence case runs the graph body (BatchBuffers.evaluate, the
code that is captured) against the eager group on CPU; it does not cover GPU
kernel selection inside a captured graph.
"""

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
ROOT = Path(__file__).resolve().parents[2] / "python"
LAYERS, HV, K, V = 36, 2, 128, 128

try:
    import torch
    import triton  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _modules():
    """Import the pool normally, or through namespace packages when the full
    sglang runtime (and its heavy package __init__ files) is unavailable."""
    try:
        pool = importlib.import_module("sglang.srt.mem_cache.gdn_factored_pool")
    except ImportError:
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.utils",
            "sglang.srt.mem_cache",
            "sglang.srt.layers",
            "sglang.srt.layers.attention",
            "sglang.srt.layers.attention.linear",
            "sglang.srt.layers.attention.linear.kernels",
            "sglang.srt.configs",
            "sglang.srt.model_executor",
            "sglang.srt.duet",
        ):
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
            sys.modules[name] = module
        pool = importlib.import_module("sglang.srt.mem_cache.gdn_factored_pool")
    batch = importlib.import_module("sglang.srt.mem_cache.gdn_prefill_batch_graph")
    return pool, batch


def _pool(fp, seed):
    vbar = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
    fixed = torch.Generator().manual_seed(7)
    torch.save({"vbar": {l: torch.randn(HV, V, generator=fixed) for l in range(LAYERS)}}, vbar)
    cfg = fp.FactoredGDNConfig.parse(
        "r=16,m=16,dtype=fp16,ring=16,async=1,strict_chunk=1,init_method=k31,"
        f"decode_method=iter,vbar={vbar},factored_prefix=1"
    )
    params = types.SimpleNamespace(shape=types.SimpleNamespace(temporal=(HV, V, K)))
    torch.manual_seed(seed)
    pool = fp.FactoredGDNPool(
        size=8, cache_params=params, mamba_layer_ids=list(range(LAYERS)),
        device="cpu", cfg=cfg, tp_rank=0, max_running_requests=8,
    )
    g = torch.Generator().manual_seed(seed)
    for t in (pool.a, pool.U, pool.W):
        t.copy_(torch.randn(t.shape, generator=g).to(t.dtype))
    pool.count.fill_(cfg.r)
    return pool


def _plan(fp, pool, B):
    return fp.FactoredExtendPlan(
        slots=torch.arange(B), use_ring=torch.zeros(B, dtype=torch.bool),
        ring_src=torch.zeros(B, dtype=torch.long),
        ring_dst=torch.tensor([0] + [-1] * (B - 1)), ring_dst_rows=torch.tensor([0]),
        last_layer=LAYERS - 1,
        dense_required_after_commit=torch.ones(B, dtype=pool.dense_required.dtype),
    )


class _CudaLike:
    """A pending dense state that reports CUDA residency for dispatch only."""

    is_cuda = True

    def __init__(self, rows):
        self.shape = (rows, HV, V, K)

    def numel(self):
        return rows_bytes(self.shape)

    def element_size(self):
        return 4


def rows_bytes(shape):
    n = 1
    for d in shape:
        n *= d
    return n


@unittest.skipUnless(HAVE_TORCH, "needs torch + triton")
class K31BatchGraphDispatchTest(unittest.TestCase):
    def setUp(self):
        self.fp, _ = _modules()
        self.pool = _pool(self.fp, 0)
        self.calls = []
        self.pool._commit_extend_group = lambda *a: self.calls.append("eager")

    def _graph(self):
        calls = self.calls

        class Graph:
            warmed = True

            def run(self, pool, plan, states, *rest, **kw):
                calls.append(("graph", len(states)))

        self.pool._k31_batch_graph = Graph()
        self.pool._k31_batch_graph_max = 4

    def _commit(self, rows, checkpoint=False):
        from unittest import mock

        plan = _plan(self.fp, self.pool, rows)
        if checkpoint:
            plan.checkpoint_group = object()
        # CPU-only torch cannot query stream capture; serving is never capturing here.
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            for lid in range(LAYERS):
                self.pool.commit_extend_batched(lid, plan, _CudaLike(rows))
        return plan

    def test_full_plan_replays_one_graph_with_every_layer(self):
        self._graph()
        for rows in (1, 2, 4):
            plan = self._commit(rows)
            self.assertEqual(plan.pending, [])
        self.assertEqual(self.calls, [("graph", LAYERS)] * 3)

    def test_ineligible_plans_keep_the_eager_group(self):
        self._commit(1)  # switch off: no graph installed
        self._graph()
        self._commit(5)  # above the prewarmed bucket cap
        self._commit(1, checkpoint=True)  # PD checkpoint group owns tracked rows
        self.assertEqual(self.calls, ["eager", "eager", "eager"])


class K31BatchGraphStartupRoutingTest(unittest.TestCase):
    """Runs the real ModelRunner.init_cuda_graphs body (GPU capture stubbed):
    every role that commits prompt-end factors must prewarm the k31 graph once."""

    def test_prewarm_reaches_agg_and_prefill_but_not_decode(self):
        import ast
        from unittest.mock import patch

        from sglang.srt.environ import envs

        path = ROOT / "sglang/srt/model_executor/model_runner.py"
        tree = ast.parse(path.read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "init_cuda_graphs")
        calls = []
        contract = types.ModuleType("sglang.srt.mem_cache.gdn_prefill_agg_contract")
        contract.prewarm = lambda pool: calls.append("pd_contract_prewarm")
        contract.install = lambda runner: calls.append("fulln_install")
        saved = sys.modules.get(contract.__name__)
        sys.modules[contract.__name__] = contract
        scope = dict(os=os, envs=envs, capture_cuda_graphs=lambda **kw: types.SimpleNamespace(
            eager_runner=None, prefill=types.SimpleNamespace(runner=None),
            decode=types.SimpleNamespace(runner=None), memory_usage=0, time_usage=0))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                     str(path), "exec"), scope)
        counts = {}
        try:
            for enabled in ("0", "1"):
                with patch.dict(os.environ, {"SGLANG_GDN_AGG_FULLN_PREFILL": enabled}):
                    for role in ("null", None, "prefill", "decode"):
                        calls.clear()
                        pool = types.SimpleNamespace(
                            prewarm_commit_graph=lambda: calls.append("commit"),
                            prewarm_k31_batch_graph=lambda: calls.append("k31"))
                        runner = types.SimpleNamespace(
                            model=types.SimpleNamespace(fullstack={"gdn_rank": 16}),
                            req_to_token_pool=types.SimpleNamespace(factored_gdn_pool=pool),
                            server_args=types.SimpleNamespace(disaggregation_mode=role))
                        scope["init_cuda_graphs"](runner)
                        counts[enabled, role] = (calls.count("k31"), calls.count("commit"))
                        self.assertEqual(calls.count("fulln_install"),
                                         int(enabled == "1" and role == "null"))
        finally:
            if saved is None:
                sys.modules.pop(contract.__name__, None)
            else:
                sys.modules[contract.__name__] = saved
        # Default-off AGG keeps its original k31-only prewarm. Full-N opt-in
        # prewarms its independent collector without warming k31 twice.
        expected = {"null": (1, 0), None: (1, 0), "prefill": (1, 1), "decode": (0, 0)}
        self.assertEqual(counts, {
            (enabled, role): ((1, 1) if enabled == "1" and role == "null" else value)
            for enabled in ("0", "1") for role, value in expected.items()
        })


@unittest.skipUnless(HAVE_TORCH, "needs torch + triton")
class K31BatchGraphEquivalenceTest(unittest.TestCase):
    """The captured body must leave the pool exactly as the eager group does."""

    def _run(self, B, tracked, final):
        fp, bg = _modules()
        g = torch.Generator().manual_seed(100 + B)
        states = [
            (torch.randn(B, HV, V, K, generator=g),
             torch.randn(B, HV, V, K, generator=g) if tracked else None)
            for _ in range(LAYERS)
        ]
        track = torch.arange(B) + B if tracked else None
        src = torch.tensor([B]) if final else None
        dst = torch.tensor([7]) if final else None
        results = []
        for graph_body in (False, True):
            pool = _pool(fp, 1)
            plan = _plan(fp, pool, B)
            plan.pending = [(s.clone(), None if t is None else t.clone()) for s, t in states]
            if graph_body:
                bucket = next(b for b in bg.BATCH_BUCKETS if B <= b)
                rows = 1 if B == 1 else bucket
                buffers = bg.BatchBuffers(pool, rows, rows if tracked else None, {},
                                          include_tail=False)
                buffers.bind(plan, plan.pending, track, src, dst)
                buffers.evaluate(fp.factorize_layers)
            else:
                pool._commit_extend_group(LAYERS - 1, plan, *plan.pending[-1], track, src, dst)
            results.append(pool)
        eager, body = results
        for name in ("a", "U", "W", "count", "stale", "dense_of", "dense_required", "prefix_valid"):
            self.assertTrue(torch.equal(getattr(eager, name), getattr(body, name)), name)
        for x, y in zip(eager.dense_ring, body.dense_ring):
            self.assertTrue(torch.equal(x, y))

    def test_singleton_with_checkpoint_and_final_copy(self):
        self._run(1, tracked=True, final=True)

    def test_padded_bucket(self):
        self._run(3, tracked=True, final=False)


if __name__ == "__main__":
    unittest.main()
