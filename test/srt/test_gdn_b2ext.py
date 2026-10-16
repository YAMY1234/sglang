"""docs/219 §11.4 batch 2 (extend host path): CPU-level bitwise / behaviour checks for the two switches.

(a) SGLANG_GDN_FACTORED_HOST_SYNC_FREE in the v7 regime: K5-ring RING=2, so the continuation ring is full on almost
    every extend and plan_extend takes the eviction branch (owner H2D + two more D2H reads per extend when off).
    Off vs on must give the same plan fields, pool tensors, ring_owner and ring_lru after every extend of a sequence.
(b) SGLANG_GDN_K42_FAST_BIND (K4.2c, WS-3 0742a22c88d ported onto v7): BatchBuffers.bind(states_bound=True) must leave
    every static buffer identical to the full bind when the states are the entry's own buffers; holds() must reject
    every non-resident case; PrefillBatchGraph.run must fall back to the full bind when holds() fails;
    PDBatchPublication must pass states_bound / drop the state record_streams only with the switch on and a workspace.
CUDA-only case: with HOST_SYNC_FREE the full-ring plan_extend performs exactly one synchronization.

  PYTHONPATH=python python3 -m pytest -q test/srt/test_gdn_b2ext.py
"""
import importlib
import logging
import os
import sys
import types
import unittest
import warnings
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    import triton  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    import test_gdn_factored_host_sync as hs  # noqa: E402  (pool builder + stub-package import of the pool module)

POOL_FIELDS = ("a", "U", "W", "count", "stale", "dense_of", "dense_required", "prefix_valid")
PLAN_TENSORS = ("slots", "use_ring", "ring_src", "ring_dst", "ring_dst_rows", "dense_required_after_commit",
                "use_prefix")
PLAN_SCALARS = ("n_ring_src", "n_ring_miss", "all_fresh", "next_layer", "last_layer")


def _ring_pool(fp, enabled, ring, device="cpu"):
    """test_gdn_factored_host_sync._pool with the dense ring size of the served recipe (K5-ring RING=2)."""
    import tempfile

    vbar = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
    fixed = torch.Generator().manual_seed(7)
    torch.save({"vbar": {l: torch.randn(hs.HV, hs.V, generator=fixed) for l in range(hs.LAYERS)}}, vbar)
    cfg = fp.FactoredGDNConfig.parse(
        f"r=8,m=8,dtype=fp16,ring={ring},async=1,strict_chunk=1,init_method=k31,"
        f"decode_method=iter,vbar={vbar},factored_prefix=1")
    params = types.SimpleNamespace(shape=types.SimpleNamespace(temporal=(hs.HV, hs.V, hs.K)))
    with hs._envs(enabled):
        pool = fp.FactoredGDNPool(size=8, cache_params=params, mamba_layer_ids=list(range(hs.LAYERS)),
                                  device=device, cfg=cfg, tp_rank=0, max_running_requests=8)
    g = torch.Generator().manual_seed(3)
    for t in (pool.a, pool.U, pool.W):
        t.copy_(torch.randn(t.shape, generator=g).to(t.dtype).to(t.device))
    # device state within its real domains: dense_of is a ring position or -1, stale is 0 / 1
    pool.stale.copy_(torch.randint(0, 2, pool.stale.shape, generator=g).to(pool.stale.dtype).to(pool.stale.device))
    pool.dense_of.copy_(torch.randint(-1, ring, pool.dense_of.shape, generator=g).to(pool.dense_of.dtype)
                        .to(pool.dense_of.device))
    pool.count.copy_(torch.randint(0, 3, pool.count.shape, generator=g).to(pool.count.dtype).to(pool.count.device))
    assert pool.cfg.ring == ring, pool.cfg.ring
    pool.prefix_valid.fill_(1)  # prefix rows have a checkpoint (no guard abort)
    pool.dense_required.zero_()
    return pool


# (slots, extend_lens, prefix_lens, prompt_final): cold rows fill the 2-position ring, then prefix hits and new rows
# arrive while it is full (eviction branch), then an owner comes back (ring reuse) and a padded attn-TP row.
FULL_RING_SEQUENCE = [
    ([1], [64], [0], [True]),
    ([2], [64], [0], [True]),
    ([3], [32], [64], [True]),
    ([4, 5], [32, 128], [64, 0], [True, True]),
    ([2], [16], [128], [True]),
    ([6, -1], [64, 0], [0, 0], [True, True]),
    ([1, 3, 7], [32, 32, 32], [64, 64, 0], [True, True, True]),
]


@unittest.skipUnless(HAVE_TORCH, "needs torch + triton")
class HostSyncFreeFullRingTest(unittest.TestCase):
    def setUp(self):
        self.fp = hs._module()

    def _compare(self, pools, plans, tag):
        for field in PLAN_TENSORS:
            hs._same(self, getattr(plans[0], field), getattr(plans[1], field), (tag, field))
        for field in PLAN_SCALARS:
            self.assertEqual(getattr(plans[0], field), getattr(plans[1], field), (tag, field))
        for name in POOL_FIELDS:
            hs._same(self, getattr(pools[0], name), getattr(pools[1], name), (tag, name))
        self.assertEqual(pools[0].ring_owner, pools[1].ring_owner, tag)
        self.assertEqual(pools[0].ring_lru, pools[1].ring_lru, tag)

    def test_sequence_ring2_matches(self):
        for ring in (2, 4):
            pools = [_ring_pool(self.fp, enabled, ring) for enabled in (False, True)]
            self.assertEqual([p.host_sync_free for p in pools], [False, True])
            full_seen = False
            for step, (slots, lens, prefix, final) in enumerate(FULL_RING_SEQUENCE):
                full_seen = full_seen or all(o >= 0 for o in pools[0].ring_owner)
                plans = [p.plan_extend(torch.tensor(slots), lens, prefix_lens=prefix, prompt_final=final)
                         for p in pools]
                self._compare(pools, plans, (ring, step, slots))
                # recycle one slot between extends, as the scheduler does for finished requests
                for p in pools:
                    p.reset_slots(torch.tensor([slots[0]]) if slots[0] >= 0 else torch.tensor([], dtype=torch.long))
                    p.prefix_valid.fill_(1)  # the recycled slot's next prefix hit restores a checkpoint again
                for name in POOL_FIELDS:
                    hs._same(self, getattr(pools[0], name), getattr(pools[1], name), (ring, step, "reset", name))
            if ring == 2:
                self.assertTrue(full_seen, "the ring-2 sequence never reached the eviction branch")

    def test_runtime_toggle_is_equivalent(self):
        # the in-process A/B flips pool.host_sync_free per extend forward: a toggled pool must stay on the same
        # trajectory as an always-off pool
        pools = [_ring_pool(self.fp, False, 2) for _ in range(2)]
        for step, (slots, lens, prefix, final) in enumerate(FULL_RING_SEQUENCE):
            pools[1].host_sync_free = bool(step % 2)
            plans = [p.plan_extend(torch.tensor(slots), lens, prefix_lens=prefix, prompt_final=final) for p in pools]
            self._compare(pools, plans, ("toggle", step))

    @unittest.skipUnless(HAVE_TORCH and torch.cuda.is_available(), "CUDA sync counting")
    def test_cuda_full_ring_plan_has_one_sync(self):
        counts = {}
        for enabled in (False, True):
            pool = _ring_pool(self.fp, enabled, 2, device="cuda")
            for slots, lens, prefix, final in FULL_RING_SEQUENCE[:2]:
                pool.plan_extend(torch.tensor(slots, device="cuda"), lens, prefix_lens=prefix, prompt_final=final)
            self.assertTrue(all(o >= 0 for o in pool.ring_owner), pool.ring_owner)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pool.plan_extend(torch.tensor([3, 4], device="cuda"), [32, 32], prefix_lens=[64, 0],
                                     prompt_final=[True, True])
            finally:
                torch.cuda.set_sync_debug_mode(0)
            counts[enabled] = len([w for w in caught if "synchroniz" in str(w.message)])
        print("FULL_RING_SYNCS", counts)
        self.assertEqual(counts[True], 1, counts)
        self.assertGreater(counts[False], 1, counts)


def _graph_module():
    try:
        return importlib.import_module("sglang.srt.mem_cache.gdn_prefill_batch_graph")
    except ImportError:
        hs._module()  # installs the stub packages
        return importlib.import_module("sglang.srt.mem_cache.gdn_prefill_batch_graph")


def _publication_module():
    try:
        return importlib.import_module("sglang.srt.mem_cache.gdn_pd_publication")
    except ImportError:
        hs._module()
        return importlib.import_module("sglang.srt.mem_cache.gdn_pd_publication")


class _FakePool:
    def __init__(self, layers=3, hv=2, v=4, k=4, ring=2):
        self.a = torch.zeros(1)
        self.hv, self.v, self.k = hv, v, k
        self.layer_ids = list(range(layers))
        self.dense_ring = [torch.zeros(ring, hv, v, k) for _ in self.layer_ids]
        self.ring_generation = 0

    def init_omega(self, batch):
        return None


def _controls(batch, tracked, seed):
    g = torch.Generator().manual_seed(seed)
    rows = max(1, batch - 1)
    plan = types.SimpleNamespace(
        slots=torch.randint(0, 50, (rows,), generator=g),
        ring_dst=torch.randint(-1, 2, (rows,), generator=g),
        dense_required_after_commit=torch.randint(0, 2, (rows,), generator=g, dtype=torch.int32))
    track = None if tracked is None else torch.randint(0, 50, (max(1, tracked - 1),), generator=g)
    final_src = torch.randint(-1, 50, (rows,), generator=g)
    final_dst = torch.randint(-1, 50, (rows,), generator=g)
    return plan, track, final_src, final_dst


def _buffers_state(buffers):
    out = {}
    for role in ("normal", "tracked"):
        for i, t in enumerate(getattr(buffers, role) or []):
            out[f"{role}{i}"] = t.clone()
    for name in ("slots", "track_slots", "ring_dst", "final_src", "final_dst", "required", "ring_pointers"):
        t = getattr(buffers, name)
        out[name] = None if t is None else t.clone()
    return out


@unittest.skipUnless(HAVE_TORCH, "needs torch + triton")
class FastBindTest(unittest.TestCase):
    def setUp(self):
        self.gm = _graph_module()

    def _pair(self, batch, tracked):
        pool = _FakePool()
        made = [self.gm.BatchBuffers(pool, batch, tracked, {}, include_tail=False) for _ in range(2)]
        for buffers in made:
            for role in ("normal", "tracked"):
                for t in getattr(buffers, role) or []:
                    t.copy_(torch.randn(t.shape, generator=torch.Generator().manual_seed(t.numel() + batch)))
        return pool, made

    def test_states_bound_bind_is_bitwise_full_bind(self):
        for batch, tracked in ((1, None), (4, None), (4, 4), (1, 1)):
            pool, (full, fast) = self._pair(batch, tracked)
            for seed in range(3):
                plan, track, fsrc, fdst = _controls(batch, tracked, seed)
                for buffers, bound in ((full, False), (fast, True)):
                    states = [(buffers.normal[i], None if buffers.tracked is None else buffers.tracked[i])
                              for i in range(len(pool.layer_ids))]
                    self.assertTrue(buffers.holds(states))
                    buffers.bind(plan, states, track, fsrc, fdst, states_bound=bound)
                a, b = _buffers_state(full), _buffers_state(fast)
                self.assertEqual(sorted(a), sorted(b))
                for key in a:
                    if a[key] is None:
                        self.assertIsNone(b[key], key)
                    else:
                        self.assertTrue(torch.equal(a[key], b[key]), (batch, tracked, seed, key))

    def test_holds_rejects_non_resident_states(self):
        pool, (buffers, _) = self._pair(4, 4)
        own = [(buffers.normal[i], buffers.tracked[i]) for i in range(3)]
        self.assertTrue(buffers.holds(own))
        copy = [(n.clone(), t) for n, t in own]
        self.assertFalse(buffers.holds(copy), "different storage")
        short = [(n[:3], t) for n, t in own]
        self.assertFalse(buffers.holds(short), "padding rows")
        untracked = [(n, None) for n, _ in own]
        self.assertFalse(buffers.holds(untracked), "tracked branch missing")
        alias = [(own[0][0], t) for _, t in own]
        self.assertFalse(buffers.holds(alias), "layer 0 buffer for every layer")
        _, (plain, _) = self._pair(4, None)
        self.assertFalse(plain.holds([(n, own[0][1]) for n in plain.normal]), "unexpected tracked state")

    def test_run_falls_back_to_full_bind_when_not_held(self):
        pool, (buffers, reference) = self._pair(4, None)
        graph = self.gm.PrefillBatchGraph(include_tail=False)
        replays = []
        key = graph.key(4, None, "eager", "policy", False)
        graph.entries[key] = (buffers, types.SimpleNamespace(replay=lambda: replays.append(1)))
        graph.warmed = True
        plan, track, fsrc, fdst = _controls(4, None, 5)
        foreign = [(torch.randn(3, pool.hv, pool.v, pool.k), None) for _ in pool.layer_ids]
        calls = []
        original = buffers.bind

        def spy(*a, **kw):
            calls.append(kw.get("states_bound"))
            return original(*a, **kw)
        buffers.bind = spy
        graph.run(pool, plan, foreign, track, fsrc, fdst, eager="eager", policy="policy",
                  join_branches=False, states_bound=True)
        self.assertEqual(calls, [False])
        reference.bind(plan, foreign, track, fsrc, fdst)
        a, b = _buffers_state(buffers), _buffers_state(reference)
        for k in a:
            if a[k] is not None:
                self.assertTrue(torch.equal(a[k], b[k]), k)
        own = [(buffers.normal[i], None) for i in range(3)]
        graph.run(pool, plan, own, track, fsrc, fdst, eager="eager", policy="policy",
                  join_branches=False, states_bound=True)
        self.assertEqual(calls, [False, True])
        self.assertEqual(len(replays), 2)

    def test_negative_control_unchecked_fast_bind_is_detected(self):
        # a fast bind WITHOUT the holds() proof (states are foreign tensors) must leave different static buffers:
        # the bitwise comparison above can see a broken fast path
        pool, (buffers, reference) = self._pair(4, None)
        graph = self.gm.PrefillBatchGraph(include_tail=False)
        graph.entries[graph.key(4, None, "eager", "policy", False)] = (buffers, types.SimpleNamespace(replay=lambda: None))
        graph.warmed = True
        buffers.holds = lambda states: True  # the broken variant
        plan, track, fsrc, fdst = _controls(4, None, 9)
        foreign = [(torch.randn(4, pool.hv, pool.v, pool.k), None) for _ in pool.layer_ids]
        graph.run(pool, plan, foreign, track, fsrc, fdst, eager="eager", policy="policy",
                  join_branches=False, states_bound=True)
        reference.bind(plan, foreign, track, fsrc, fdst)
        a, b = _buffers_state(buffers), _buffers_state(reference)
        self.assertFalse(all(torch.equal(a[k], b[k]) for k in a if a[k] is not None),
                         "negative control not detected")


class _Runtime:
    def __init__(self):
        self.inputs = None

    def launch_after_forward(self, operation, inputs):
        self.inputs = list(inputs)
        operation()
        return "ticket"

    def join(self, done):
        pass


class _Graph:
    include_tail = False
    warmed = True

    def __init__(self, workspace):
        self.workspace = workspace
        self.calls = []

    def run(self, pool, plan, states, *controls, eager, policy, **kw):
        self.calls.append(kw)


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class PublicationFastBindTest(unittest.TestCase):
    def setUp(self):
        self.pm = _publication_module()
        self.saved = self.pm._FAST_BIND

    def tearDown(self):
        self.pm._FAST_BIND = self.saved

    def _launch(self, fast, workspace, *, agg=True, hook=None):
        self.pm._FAST_BIND = fast
        pool = types.SimpleNamespace(a=torch.zeros(1))
        runtime = _Runtime()
        publication = self.pm.PDBatchPublication(pool, runtime, agg=agg)
        publication.after_replay = hook
        graph = _Graph(workspace)
        states = [(torch.zeros(2), torch.ones(2)), (torch.zeros(3), None)]
        controls = (torch.arange(2), None, None)
        plan = types.SimpleNamespace(slots=torch.arange(2), ring_dst=torch.arange(2),
                                     dense_required_after_commit=None)
        publication.submit(graph, plan, states, controls, eager="e", policy="p")
        publication.start_after_forward()
        ids = {id(t) for pair in states for t in pair if t is not None}
        return publication, runtime, graph, ids

    def test_default_off_is_unchanged(self):
        _, runtime, graph, ids = self._launch(False, object())
        self.assertEqual(graph.calls, [{}])
        self.assertTrue(ids <= {id(t) for t in runtime.inputs if t is not None})

    def test_on_with_workspace_binds_controls_only(self):
        publication, runtime, graph, ids = self._launch(True, object())
        self.assertEqual(graph.calls, [{"states_bound": True}])
        self.assertFalse(ids & {id(t) for t in runtime.inputs if t is not None})
        self.assertTrue(getattr(publication, "_fast_bind_logged", False))

    def test_on_without_workspace_or_agg_is_unchanged(self):
        for kwargs in (dict(workspace=None), dict(workspace=object(), agg=False)):
            _, runtime, graph, ids = self._launch(True, **kwargs)
            self.assertEqual(graph.calls, [{}], kwargs)
            self.assertTrue(ids <= {id(t) for t in runtime.inputs if t is not None}, kwargs)

    def test_after_replay_hook_keeps_states_bound(self):
        seen = []
        _, _, graph, _ = self._launch(True, object(), hook=lambda: seen.append(1))
        self.assertEqual(graph.calls, [{"states_bound": True}])
        self.assertEqual(seen, [1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
