"""CPU tests of real factor payloads and local P/D lifecycle (no CUDA imports)."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


factors = load("_pd_test_factor_pool", "mem_cache/gdn_factored_pool.py")
handoff = load("_pd_test_handoff", "disaggregation/state_handoff.py")


class TestFactoredPDHandoff(unittest.TestCase):
    def setUp(self):
        self.pool = factors.FactoredGDNPool(
            size=8, cache_params=SimpleNamespace(shape=SimpleNamespace(temporal=(2, 4, 4))),
            mamba_layer_ids=[0, 2], device="cpu",
            cfg=factors.FactoredGDNConfig(r=2, m=2, ring=3),
        )
        self.req = SimpleNamespace(kv=SimpleNamespace(
            mamba_pool_idx=torch.tensor(2),
            mamba_ping_pong_track_buffer=torch.tensor([3, 4]),
            mamba_last_track_idx=1, mamba_last_track_seqlen=128,
            mamba_cow_src_index=torch.tensor([7]), mamba_needs_clear=True,
        ))
        self.handler = handoff.FactorStateHandoff(self.pool)

    def payload(self):
        return [x.clone() for x in (self.pool.a, self.pool.U, self.pool.W, self.pool.count)]

    def test_native_preallocation_synchronizes_before_slot_reset(self):
        import ast
        import os

        path = ROOT / "disaggregation/decode.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "DecodePreallocQueue")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_pre_alloc")
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
            names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        for overlap in (False, True):
            for obsolete_switch in ("0", "1"):
                with self.subTest(overlap=overlap, obsolete_switch=obsolete_switch):
                    log = []
                    req = self.req
                    req.kv.req_pool_idx = 0
                    req.origin_input_ids, req.output_ids = list(range(8)), []
                    req.set_extend_range = lambda *args: log.append("range")
                    original = self.handler.prepare_receive
                    def prepare(request):
                        log.append("reset")
                        original(request)
                    def allocate(allocator, **kwargs):
                        self.assertIs(kwargs["req"], req)
                        self.assertEqual(log, ["slot", "sync", "reset"] if overlap
                                         else ["slot", "reset"])
                        log.append("pages")
                        return torch.arange(8)
                    pool = SimpleNamespace(
                        alloc=lambda reqs: log.append("slot") or [0],
                        write=lambda *args: log.append("write"),
                        pd_state_handoffs={handoff.HandoffKind.STATE_FACTOR: self.handler},
                    )
                    queue = SimpleNamespace(
                        req_to_token_pool=pool,
                        scheduler=SimpleNamespace(enable_overlap=overlap, enable_hisparse=False,
                            forward_stream=SimpleNamespace(synchronize=lambda: log.append("sync"))),
                        token_to_kv_pool_allocator=object(),
                        _pre_alloc_fill_len=lambda req: 8,
                        _required_alloc_tokens=lambda **kwargs: 8,
                        _uses_swa_tail_prealloc=lambda: False, _swa_tail_len=lambda n: 0,
                    )
                    scope = dict(torch=torch, alloc_for_decode_prealloc=allocate,
                        get_disagg=lambda: SimpleNamespace(disaggregation_decode_enable_radix_cache=False))
                    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
                    before = self.payload()
                    with patch.dict(os.environ, SGLANG_FLASHNEXT_ASYNC_FACTOR_RECEIVE=obsolete_switch), \
                         patch.dict(sys.modules, {"sglang.srt.disaggregation.state_handoff": handoff}), \
                         patch.object(self.handler, "prepare_receive", side_effect=prepare):
                        actual = scope["_pre_alloc"](queue, req=req, prefix_indices=None,
                                                      prefix_len=0, total_prefix_len=0)
                    self.assertTrue(torch.equal(actual, torch.arange(8)))
                    self.assertFalse(hasattr(req, "_pd_deferred_factor_receive"))
                    for actual, expected in zip(self.payload(), before):
                        self.assertTrue(torch.equal(actual, expected))

    def test_receive_overwrites_authority_without_touching_payload(self):
        self.pool.a.normal_(); self.pool.U.normal_(); self.pool.W.normal_()
        self.pool.ring_owner[:] = [2, 3, 7]
        self.pool.stale.zero_(); self.pool.dense_of.fill_(1)
        before = self.payload()
        self.handler.prepare_receive(self.req)
        # Emulate RDMA into destination, then a successful metadata gate.
        self.pool.a[:, 2].add_(10)
        expected = self.payload()
        self.handler.commit_receive(self.req)
        self.handler.commit_receive(self.req)  # duplicate success/retry
        for actual, want in zip(self.payload(), expected):
            self.assertTrue(torch.equal(actual, want))
        self.assertFalse(torch.equal(before[0], expected[0]))
        self.assertEqual(self.pool.ring_owner, [-1, -1, 7])
        self.assertEqual(self.pool.stale[2:5].tolist(), [1, 1, 1])
        self.assertEqual(self.pool.dense_of[2:5].tolist(), [-1, -1, -1])
        self.assertEqual(self.pool.stale[7].item(), 0)
        self.assertIsNone(self.req.kv.mamba_cow_src_index)
        self.assertFalse(self.req.kv.mamba_needs_clear)

    def test_cancel_then_reuse_does_not_reuse_dense_ring(self):
        self.handler.prepare_receive(self.req)
        self.pool.a[:, 2].fill_(123)  # cancelled transfer payload, never committed
        self.pool.ring_owner[0] = 2
        self.pool.dense_of[2] = 0
        self.pool.stale[2] = 0
        self.handler.prepare_receive(self.req)  # only after old writer drained
        self.pool.a[:, 2].fill_(456)
        self.handler.commit_receive(self.req)
        self.assertEqual(self.pool.ring_owner[0], -1)
        self.assertTrue(bool((self.pool.a[:, 2] == 456).all()))
        self.assertEqual(self.pool.stale[2].item(), 1)

    def test_send_requires_final_truncation(self):
        self.handler.before_send(self.req)
        self.pool.count[1, 2, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "final prefill"):
            self.handler.before_send(self.req)

    def test_wire_entries_exclude_local_metadata(self):
        entries = list(self.pool.iter_transfer_state_entries())
        self.assertEqual(len(entries), 8)
        self.assertEqual({x[0] for x in entries}, {
            "gdn_factored_a", "gdn_factored_u", "gdn_factored_w", "gdn_factored_count"})
        for _, tensor, axis, lid in entries:
            self.assertEqual(tensor.shape[0], 9)
            self.assertEqual(axis, 0)
            self.assertIn(lid, [0, 2])

    def test_fp16_sizing_matches_wire_payload(self):
        cfg = factors.FactoredGDNConfig.parse("r=8,m=8,dtype=fp16,strict_chunk=1,factored_prefix=1")
        shape = SimpleNamespace(temporal=(2, 4, 4))
        pool = factors.FactoredGDNPool(size=8, cache_params=SimpleNamespace(shape=shape),
            mamba_layer_ids=[0, 2], device="cpu", cfg=cfg)
        actual = sum(x[0, 0].nbytes for x in (pool.a, pool.U, pool.W, pool.count))
        self.assertEqual(actual, cfg.state_bytes_per_layer(shape))
        self.assertEqual(pool.U.dtype, torch.float16)
        self.assertEqual(pool.W.dtype, torch.float16)
        self.assertIsNone(pool.prefix_dense)
        self.assertEqual({x[1].dtype for x in pool.iter_transfer_state_entries()},
                         {torch.float16, torch.float32, torch.int32})

    def test_strict_prefix_local_metadata_is_invalidated_on_receive(self):
        self.pool.dense_required = torch.ones(9, dtype=torch.int32)
        self.pool.prefix_dense_valid = torch.ones(9, dtype=torch.int32)
        payload = self.payload()
        self.handler.prepare_receive(self.req)
        self.handler.commit_receive(self.req)
        self.assertEqual(self.pool.dense_required[2:5].tolist(), [0, 0, 0])
        self.assertEqual(self.pool.prefix_dense_valid[2:5].tolist(), [0, 0, 0])
        self.assertEqual(self.pool.prefix_dense_valid[7].item(), 1)
        for actual, want in zip(self.payload(), payload):
            self.assertTrue(torch.equal(actual, want))

    def test_factored_prefix_validity_is_local_to_receiver(self):
        self.pool.cfg.factored_prefix = 1
        self.pool.dense_required = torch.ones(9, dtype=torch.int32)
        self.pool.prefix_factored_valid = torch.ones(9, dtype=torch.int32)
        payload = self.payload()
        self.handler.prepare_receive(self.req)
        self.handler.commit_receive(self.req)
        self.assertEqual(self.pool.prefix_factored_valid[2:5].tolist(), [0, 0, 0])
        self.assertEqual(self.pool.prefix_factored_valid[7].item(), 1)
        for actual, want in zip(self.payload(), payload):
            self.assertTrue(torch.equal(actual, want))

    def test_explicit_p31_boundary_preserves_r_plus_one(self):
        self.pool.cfg.strict_chunk = True
        self.req.factored_prefill_boundary_steps = 1
        with self.assertRaises(RuntimeError):
            self.handler.before_send(self.req)
        self.pool.count[:, 2] = self.pool.cfg.r + 1
        payload = self.payload()
        self.handler.before_send(self.req)
        for actual, want in zip(self.payload(), payload):
            self.assertTrue(torch.equal(actual, want))
        self.req.factored_prefill_boundary_steps = 0
        with self.assertRaises(RuntimeError):
            self.handler.before_send(self.req)
        self.req.factored_prefill_boundary_steps = 2
        with self.assertRaises(RuntimeError):
            self.handler.before_send(self.req)

    def test_flag_off_and_explicit_extension_registration(self):
        pool = SimpleNamespace()
        before = vars(self.req.kv).copy()
        handoff.dispatch_handoff(pool, "commit_receive", self.req)
        self.assertEqual(vars(self.req.kv), before)
        handoff.register_handoff(pool, handoff.HandoffKind.STATE_FACTOR, self.handler)
        with self.assertRaises(ValueError):
            handoff.register_handoff(pool, handoff.HandoffKind.STATE_FACTOR, self.handler)
        with self.assertRaises(TypeError):
            handoff.register_handoff(pool, handoff.HandoffKind.LATENT, object())
        with self.assertRaises(ValueError):
            handoff.dispatch_handoff(pool, "invented", self.req)

    def test_invalid_slot_cannot_mutate_pool(self):
        before = self.payload()
        for indices in (torch.tensor([0]), torch.tensor([9]), torch.tensor([-1])):
            with self.assertRaises(ValueError):
                self.pool.mark_transferred_slots(indices)
        for actual, want in zip(self.payload(), before):
            self.assertTrue(torch.equal(actual, want))

    def test_transfer_payload_uses_shareable_allocator_only(self):
        active = [False]
        allocations = {}
        @contextmanager
        def shared(pool):
            self.assertEqual(pool, 'mooncake')
            active[0] = True
            try:
                yield
            finally:
                active[0] = False
        def record(fn):
            def alloc(*args, **kwargs):
                tensor = fn(*args, **kwargs)
                allocations[tensor.data_ptr()] = active[0]
                return tensor
            return alloc
        with patch.object(torch.cuda,'use_mem_pool',shared), \
             patch.object(torch,'zeros',record(torch.zeros)), \
             patch.object(torch,'full',record(torch.full)), \
             patch.object(torch,'ones',record(torch.ones)):
            pool = factors.FactoredGDNPool(
                size=8,cache_params=SimpleNamespace(shape=SimpleNamespace(temporal=(2,4,4))),
                mamba_layer_ids=[0,2],device='cpu',
                cfg=factors.FactoredGDNConfig(r=2,m=2,ring=3),custom_mem_pool='mooncake')
        for tensor in (pool.a,pool.U,pool.W,pool.count):
            self.assertTrue(allocations[tensor.data_ptr()])
        for tensor in (pool.stale,pool.dense_of,pool.dense_ring,pool.vbar):
            self.assertFalse(allocations[tensor.data_ptr()])


if __name__ == "__main__":
    unittest.main()
