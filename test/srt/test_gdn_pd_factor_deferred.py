"""PD 2b/A: real reader order, native graph-body bytes, dense recurrence.

CPU interpreter and stream stubs are separate tests. Neither substitutes for
PD NLL/needle or hardware replay admission. Legacy and new wire tags are tested
through the actual SplitBoundaryPhase/FactorStateHandoff implementations.
"""

import ast
import copy
import importlib
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("TRITON_INTERPRET", "1")
from test_gdn_tracked_factor_side import controller, extract, function

BASE = Path(__file__).resolve().parents[2] / "python/sglang/srt"
SOURCE = BASE / "mem_cache/gdn_pd_factor_deferred.py"
OWNER = BASE / "models/flash_next_duet/pd_factor_deferred.py"


class StreamOrderTest(unittest.TestCase):
    def test_T_after_all_live_work_and_A_F_after_T(self):
        for final in (False, True):
            side, cuda = controller(deferred=True)
            scope = dict(torch=cuda, graph_shape=lambda n, t: (n, t))
            plan = types.SimpleNamespace(slots=types.SimpleNamespace(numel=lambda: 1))
            tracks = types.SimpleNamespace(numel=lambda: 1)
            extract([function(SOURCE, "launch_side")], scope)
            cuda.trace.append(("live_complete", cuda.main))
            scope["launch_side"](
                side,
                plan,
                [object()],
                (tracks, None, None),
                final=final,
                eager="eager",
                policy="policy",
            )
            names = [r[0] for r in cuda.trace]
            self.assertEqual(
                names,
                ["live_complete", "bind", "record", "wait", "tracked"]
                + (["final"] if final else [])
                + ["record"],
            )
            self.assertIs(cuda.trace[1][1], cuda.main)
            self.assertIs(cuda.trace[4][1], side.stream)
            self.assertIs(side.done, side.set_done[0])
            side.join()
            side.join()
            self.assertEqual(cuda.trace[-1], ("wait", cuda.main, side.done))
            cuda.trace.clear()
            # Reader fence plus parity-specific buffer reuse, including next F.
            for _ in range(2):
                scope["launch_side"](
                    side,
                    plan,
                    [],
                    (tracks, None, None),
                    final=final,
                    eager="eager",
                    policy="policy",
                )
            names = [r[0] for r in cuda.trace]
            pos = names.index("bind")
            self.assertEqual(cuda.trace[pos - 1], ("wait", cuda.main, side.set_done[0]))

    def test_default_off_and_capture_fallback_before_tensor_reads(self):
        env = (BASE / "environ.py").read_text()
        for name in (
            "SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_PD_P",
            "SGLANG_GDN_FINAL_FACTOR_DEFERRED_PD",
        ):
            self.assertIn(name + " = EnvBool(False)", env)
        side, cuda = controller(deferred=True)
        cuda.capturing = True
        reasons = []
        scope = dict(torch=cuda)
        extract([function(OWNER, "transaction_for")], scope)
        owner = types.SimpleNamespace(
            pd_factor_deferred=types.SimpleNamespace(
                fallback=lambda reason: reasons.append(reason)
            )
        )
        scope["transaction_for"](owner, None, None, None, None, None)
        self.assertEqual(reasons, ["capture"])

    def test_native_transfer_entry_dispatches_handoff_before_send(self):
        path = BASE / "disaggregation/prefill.py"
        fn = function(path, "send_kv_chunk")
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        handoff = [
            n.lineno
            for n in calls
            if isinstance(n.func, ast.Name) and n.func.id == "dispatch_handoff"
        ]
        sends = [
            n.lineno
            for n in calls
            if isinstance(n.func, ast.Attribute) and n.func.attr == "send"
        ]
        self.assertTrue(handoff and sends)
        self.assertLess(min(handoff), min(sends))


try:
    import torch
    import triton  # noqa: F401 — gate requires the interpreter

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def modules():
    from test_gdn_prefill_k31_batch_graph import _modules

    fp, bg = _modules()
    return (
        fp,
        bg,
        importlib.import_module("sglang.srt.mem_cache.gdn_pd_factor_deferred"),
    )


@unittest.skipUnless(HAVE_TORCH, "same-image gate requires torch/triton; rejects skips")
class WireAndTransactionTest(unittest.TestCase):
    def setUp(self):
        self.fp, self.bg, self.mod = modules()
        torch.set_num_threads(1)

    def wire(self, tag, counts, final=True, role="prefill"):
        from twinstar_sgl.pd_shallow import BoundaryState, SplitBoundaryPhase
        from sglang.srt.disaggregation.state_handoff import FactorStateHandoff

        events = []
        state = BoundaryState(size=2, device="cpu")
        state.valid[1] = tag
        pool = types.SimpleNamespace(
            cfg=types.SimpleNamespace(r=8, strict_chunk=True),
            layer_ids=[0, 31],
            count=torch.tensor(counts).reshape(2, 1, 1).expand(2, 3, 1).clone(),
            pside_join=lambda: events.append("join"),
            mark_transferred_slots=lambda slots: events.append("received"),
        )
        original = SplitBoundaryPhase(FactorStateHandoff(pool), state)
        handoff = self.mod.PDDeferredHandoff(
            original, pool, state, final=final, role=role
        )
        req = types.SimpleNamespace(
            kv=types.SimpleNamespace(
                mamba_pool_idx=torch.tensor(1), mamba_ping_pong_track_buffer=None
            )
        )
        return handoff, req, state, pool, events

    def test_legacy_phase_and_wire_bytes_unchanged(self):
        handoff, req, state, pool, events = self.wire(1, [9, 8])
        before = [
            t.clone() for t in (state.hidden, state.position, state.valid, pool.count)
        ]
        handoff.before_send(req)
        self.assertEqual(events, ["join"])
        for a, b in zip(
            before, (state.hidden, state.position, state.valid, pool.count)
        ):
            self.assertTrue(torch.equal(a, b))
        handoff.role = "decode"
        handoff.commit_receive(req)
        self.assertEqual(int(state.valid[1]), 1)
        self.assertFalse(hasattr(req, "pd_final_factor_deferred_received"))

    def test_new_phase_join_then_host_read_then_wire_copy_and_D_normalization(self):
        h, req, state, pool, events = self.wire(2, [0, 0])

        def joined():
            events.append("join")
            pool.count[:, 1] = 8

        pool.pside_join = joined
        h.before_send(req)  # The count is invalid until the producer fence runs.
        self.assertEqual(events, ["join"])
        wire = tuple(
            t.clone() for t in (state.hidden[1], state.position[1], state.valid[1])
        )
        d, dreq, ds, dp, dev = self.wire(0, [8, 8], role="decode")
        for target, value in zip((ds.hidden, ds.position, ds.valid), wire):
            target[1] = value
        d.commit_receive(dreq)
        self.assertEqual(dev, ["received"])
        self.assertTrue(dreq.pd_final_factor_deferred_received)
        self.assertEqual(int(ds.valid[1]), 1)
        self.assertTrue(torch.equal(ds.hidden[1], state.hidden[1]))
        self.assertTrue(torch.equal(ds.position[1], state.position[1]))
        self.assertEqual(dp.count[:, 1, 0].tolist(), [8, 8])
        # The unchanged decoder consumes only emitter_ids (deep); it does not
        # repeat the 24 shallow tails already in the received factors.
        helper = Path(importlib.import_module("twinstar_sgl.pd_shallow").__file__)
        body = ast.unparse(function(helper, "decode_boundary"))
        self.assertIn("run_deep_boundary(owner, fb, hidden)", body)
        self.assertIn(
            "owner.emitter_ids", ast.unparse(function(helper, "run_deep_boundary"))
        )

    def test_mismatch_unpublished_and_unknown_tags_fail_closed(self):
        for tag, counts, final, role in (
            (2, [8, 8], False, "decode"),
            (2, [9, 8], True, "decode"),
            (3, [8, 8], True, "decode"),
        ):
            h, req, *_ = self.wire(tag, counts, final=final, role=role)
            with self.assertRaises(RuntimeError):
                h.commit_receive(req)
        h, req, *_ = self.wire(2, [9, 8])
        with self.assertRaises(RuntimeError):
            h.before_send(req)

    def test_return_join_precedes_new_valid_word(self):
        events = []
        valid = types.SimpleNamespace(index_fill_=lambda *a: events.append("valid2"))
        tx = object.__new__(self.mod.PDDeferredTransaction)
        tx.published = True
        tx.final = True
        tx.plan = types.SimpleNamespace(slots=torch.tensor([1]))
        tx.pool = types.SimpleNamespace(
            launch_pending_tracked=lambda: events.append("boundary_done"),
            pside_join=lambda: events.append("join"))
        tx.controller = types.SimpleNamespace(state=types.SimpleNamespace(valid=valid))
        tx.finish_return()
        self.assertEqual(events, ["boundary_done", "join", "valid2"])

    def test_mutated_owner_and_missing_tail_rejected_before_publication(self):
        tx = object.__new__(self.mod.PDDeferredTransaction)
        tx.published = False
        tx.pool = types.SimpleNamespace(layer_ids=[0, 31])
        tx.states = [(None, None)] * 2
        tx.tail_layers = set()
        with self.assertRaisesRegex(RuntimeError, "all prefix"):
            tx.publish()
        tx.tail_layers = {0}
        tx.ids = torch.tensor([0])
        tx.generations = torch.tensor([1])
        tx.controller = types.SimpleNamespace(
            request_pool=types.SimpleNamespace(req_generation=torch.tensor([2]))
        )
        with self.assertRaisesRegex(RuntimeError, "generation"):
            tx.publish()

    def test_native_dense_tail_same_kernel_inputs_state_and_output(self):
        from test_gdn_prefill_k31_batch_graph import HV, V, K
        from test_gdn_final_factor_deferred import _interpreter_exp
        from sglang.kernels.ops.attention.fla import fused_recurrent as recurrent
        import functools

        scope = dict(
            torch=torch,
            fused_recurrent_gated_delta_rule_packed_decode=recurrent.fused_recurrent_gated_delta_rule_packed_decode,
        )
        extract(
            [
                function(
                    BASE / "layers/attention/linear/kernels/gdn_triton.py",
                    "packed_decode",
                )
            ],
            scope,
        )
        native = functools.partial(scope["packed_decode"], None)
        g = torch.Generator().manual_seed(143)
        state = torch.randn(1, HV, V, K, generator=g)
        expected_state = state.clone()
        tx = object.__new__(self.mod.PDDeferredTransaction)
        tx.pool = types.SimpleNamespace(layer_map={0: 0})
        tx.states = [(state, None)]
        tx.final = True
        tx.tail_layers = set()
        tx.boundary_slots = torch.tensor([7], dtype=torch.int32)
        layer = types.SimpleNamespace(
            layer_id=0,
            head_k_dim=K,
            head_v_dim=V,
            num_v_heads=HV,
            A_log=torch.randn(HV, generator=g),
            dt_bias=torch.randn(HV, generator=g),
        )
        mixed = torch.randn(1, (2 + HV) * K, generator=g).to(torch.bfloat16)
        a = torch.randn(1, HV, generator=g).to(torch.bfloat16)
        b = torch.randn(1, HV, generator=g).to(torch.bfloat16)
        calls = []
        backend = types.SimpleNamespace(
            kernel_dispatcher=types.SimpleNamespace(packed_decode=native),
            _track_mamba_state_decode=lambda *args: calls.append("track"),
        )
        with patch.object(recurrent, "exp", _interpreter_exp):
            actual = tx.decode(
                backend, layer, None, mixed, a, b, None, None, tx.boundary_slots
            )
            expected = native(
                mixed,
                a,
                b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=K**-0.5,
                ssm_states=expected_state,
                cache_indices=torch.tensor([0], dtype=torch.int32),
                num_v_heads=HV,
                head_v_dim=V,
            )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(state, expected_state))
        self.assertEqual(calls, ["track"])
        with self.assertRaisesRegex(RuntimeError, "once"):
            tx.decode(backend, layer, None, mixed, a, b, None, None, tx.boundary_slots)

    def test_startup_PD_only_explicit_pair_and_invalid_modes(self):
        from sglang.srt.disaggregation.state_handoff import HandoffKind

        spec = importlib.util.spec_from_file_location("pd_deferred_owner_test", OWNER)
        owner_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner_mod)
        flags = {
            "SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_PD_P": "1",
            "SGLANG_GDN_FINAL_FACTOR_DEFERRED_PD": "1",
            "SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31": "1",
        }

        def fixture(role):
            pool = types.SimpleNamespace(
                cfg=types.SimpleNamespace(
                    r=8,
                    strict_chunk=True,
                    factored_prefix=role == "prefill",
                    init_method="k31",
                    decode_method="iter",
                ),
                prefix_dense=None,
                layer_ids=list(range(36)),
                prefix_layer_count=lambda: 36,
                _generic_prompt_only_state_cache=True,
                _final_factor_deferred=None,
            )
            rp = types.SimpleNamespace(
                factored_gdn_pool=pool,
                pd_boundary_state=object(),
                pd_state_handoffs={HandoffKind.STATE_FACTOR: object()},
            )
            args = types.SimpleNamespace(
                pp_size=1,
                disable_overlap_schedule=True,
                is_embedding=False,
                enable_two_batch_overlap=False,
                enable_torch_compile=False,
                enable_linear_replayssm=False,
            )
            runner = types.SimpleNamespace(
                req_to_token_pool=rp,
                server_args=args,
                is_draft_worker=False,
                lora_manager=None,
                spec_algorithm=types.SimpleNamespace(is_none=lambda: True),
                attn_backend=types.SimpleNamespace(
                    linear_attn_backend=types.SimpleNamespace(
                        kernel_dispatcher=types.SimpleNamespace(
                            supports_packed_decode=True
                        )
                    )
                ),
            )
            owner = types.SimpleNamespace(
                pd_shallow_role=role, fullstack_v3_latent=False
            )
            return owner, runner, pool

        with patch.dict(os.environ, flags):
            for role in ("prefill", "decode", "null"):
                owner, runner, pool = fixture(role)
                runner.server_args.disable_overlap_schedule = role != "decode"
                owner_mod.install(owner, runner)
                self.assertEqual(
                    owner.pd_factor_deferred is not None, role == "prefill"
                )
                self.assertEqual(
                    isinstance(
                        runner.req_to_token_pool.pd_state_handoffs[
                            HandoffKind.STATE_FACTOR
                        ],
                        self.mod.PDDeferredHandoff,
                    ),
                    role != "null",
                )
            for field, value in (
                ("pp_size", 2),
                ("is_embedding", True),
                ("disable_overlap_schedule", False),
                ("enable_linear_replayssm", True),
                ("enable_two_batch_overlap", True),
            ):
                owner, runner, _ = fixture("prefill")
                setattr(runner.server_args, field, value)
                with self.assertRaises(ValueError):
                    owner_mod.install(owner, runner)
            with patch.dict(os.environ, {"SGLANG_GDN_PREFILL_JOIN_BRANCHES": "1"}):
                owner, runner, _ = fixture("prefill")
                with self.assertRaisesRegex(ValueError, "conflicting"):
                    owner_mod.install(owner, runner)
        with patch.dict(os.environ, {k: "0" for k in flags}):
            owner, runner, _ = fixture("prefill")
            old = runner.req_to_token_pool.pd_state_handoffs.copy()
            owner_mod.install(owner, runner)
            self.assertIsNone(owner.pd_factor_deferred)
            self.assertEqual(old, runner.req_to_token_pool.pd_state_handoffs)

    def test_request_rejections_and_logprob_uses_candidate(self):
        from test_gdn_prefill_k31_batch_graph import _pool, _plan

        spec = importlib.util.spec_from_file_location("pd_deferred_owner_test", OWNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pool = _pool(self.fp, 23)
        plan = _plan(self.fp, pool, 1)
        key = (
            1,
            1,
            self.fp.factorize_layers,
            (self.fp.ORTH_METHOD, self.fp.ORTH_WARPS_OVERRIDE, self.fp.factorize_dense),
            False,
        )
        pool._tracked_factor_side = types.SimpleNamespace(
            deferred=True,
            whole_graph=types.SimpleNamespace(key=lambda *a: a),
            entries={key: object()},
            alt_entries={key: object()},
        )
        state = types.SimpleNamespace(valid=torch.zeros(9, 1, dtype=torch.int32))
        rp = types.SimpleNamespace(req_generation=torch.tensor([4]))
        ctl = self.mod.PDDeferredController(pool, rp, state, final=True)
        owner = types.SimpleNamespace(pd_factor_deferred=ctl, state_audit_dir=None)
        batch = types.SimpleNamespace(
            batch_size=1,
            twinstar_prompt_final=[True],
            spec_info=None,
            return_logprob=True,
            forward_mode=types.SimpleNamespace(is_mixed=lambda: False),
            req_pool_indices_cpu=torch.tensor([0]),
            capture_hidden_mode=types.SimpleNamespace(is_full=lambda: False),
        )
        boundary = types.SimpleNamespace(batch_size=1, mamba_track_mask=None)
        meta = types.SimpleNamespace(
            factored_extend=plan,
            has_mamba_track_mask=True,
            track_ssm_h_dst=torch.tensor([2]),
            track_ssm_final_src=None,
            track_ssm_final_dst=None,
        )
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            result = module.transaction_for(
                owner, batch, batch, boundary, meta, object()
            )
            self.assertIsInstance(result, self.mod.PDDeferredTransaction)
            batch.batch_size = 17
            self.assertIsNone(
                module.transaction_for(owner, batch, batch, boundary, meta, object())
            )
            batch.batch_size = 1
            meta.track_ssm_final_src = torch.tensor([0])
            meta.track_ssm_final_dst = torch.tensor([3])
            self.assertIsNone(
                module.transaction_for(owner, batch, batch, boundary, meta, object())
            )
            ctl.final = False
            meta.track_ssm_final_src = torch.tensor([2])
            self.assertIsNone(
                module.transaction_for(owner, batch, batch, boundary, meta, object())
            )
            self.assertEqual(ctl.stats["fallback"], 3)

    def test_PD_2b_full_layer_bytes_match_native_per_layer_commit(self):
        from test_gdn_prefill_k31_batch_graph import _pool, _plan, LAYERS, HV, V, K

        fp, bg = self.fp, self.bg
        g = torch.Generator().manual_seed(47)
        states = [
            (
                torch.randn(1, HV, V, K, generator=g),
                torch.randn(1, HV, V, K, generator=g),
            )
            for _ in range(LAYERS)
        ]
        tracks, empty = torch.tensor([2]), torch.empty(0, dtype=torch.long)
        pools = []
        for deferred in (False, True):
            parse = fp.FactoredGDNConfig.parse
            with patch.object(
                fp.FactoredGDNConfig,
                "parse",
                side_effect=lambda value: parse(value.replace("r=16,m=16", "r=8,m=8")),
            ):
                pool = _pool(fp, 19)
            plan = _plan(fp, pool, 1)
            # Exercise real live final-copy bookkeeping as well as tracked T.
            src, dst = torch.tensor([0]), torch.tensor([7])
            meta = types.SimpleNamespace(
                track_ssm_h_dst=tracks, track_ssm_final_src=src, track_ssm_final_dst=dst
            )
            rp = types.SimpleNamespace(req_generation=torch.tensor([4]))
            ctl = self.mod.PDDeferredController(pool, rp, None, final=False)
            tx = self.mod.PDDeferredTransaction(
                ctl,
                types.SimpleNamespace(req_pool_indices_cpu=torch.tensor([0])),
                plan,
                meta,
            )

            def tail(layer, batch, mixed, a, b, conv, temporal, slots):
                # Unchanged recurrence entry is checked separately. A visible
                # append here detects the old all-layer final-copy pitfall.
                pool.count[layer.layer_id, 0] += 1
                return torch.tensor([layer.layer_id])

            backend = types.SimpleNamespace(_forward_decode_factored=tail)

            def replay(side, replay_plan, saved, controls, **kwargs):
                bufs = bg.BatchBuffers(pool, 1, 1, include_tail=False)
                bufs.bind(replay_plan, saved, *controls)
                bufs.evaluate(fp.factorize_layers, branch="tracked")

            pool._tracked_factor_side = None
            with patch.object(self.mod, "launch_side", replay):
                if deferred:
                    tx.__enter__()
                for lid, (dense, tracked) in enumerate(states):
                    if deferred:
                        tx.add(lid, plan, dense, tracked, tracks, src, dst)
                    else:
                        one = copy.copy(plan)
                        one.next_layer = one.last_layer = lid
                        one.pending = []
                        pool.commit_extend_batched(
                            lid, one, dense, tracked, tracks, None, None
                        )
                        pool.copy_slots_layer(lid, src, dst)
                    if lid < 31:
                        layer = types.SimpleNamespace(layer_id=lid)
                        if deferred:
                            tx.decode(
                                backend, layer, None, None, None, None, None, None, None
                            )
                        else:
                            tail(layer, None, None, None, None, None, None, None)
                if deferred:
                    tx.__exit__(None, None, None)
            pools.append(pool)
        for name in (
            "a",
            "U",
            "W",
            "count",
            "stale",
            "dense_of",
            "dense_required",
            "prefix_valid",
            "dense_ring",
        ):
            a = getattr(pools[0], name)
            b = getattr(pools[1], name)
            if isinstance(a, list):
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)), name)
            else:
                self.assertTrue(torch.equal(a, b), name)
        self.assertEqual(pools[1].count[0, 7, 0].item(), pools[1].cfg.r)

    def test_PD_A_graph_body_matches_main_for_same_mixed_S_N_S_N1(self):
        from test_gdn_prefill_k31_batch_graph import _pool, _plan, LAYERS, HV, V, K

        g = torch.Generator().manual_seed(48)
        states = [
            (
                torch.randn(3, HV, V, K, generator=g),
                torch.randn(3, HV, V, K, generator=g),
            )
            for _ in range(LAYERS)
        ]
        outputs = []
        for side in (False, True):
            parse = self.fp.FactoredGDNConfig.parse
            with patch.object(
                self.fp.FactoredGDNConfig,
                "parse",
                side_effect=lambda value: parse(value.replace("r=16,m=16", "r=8,m=8")),
            ):
                pool = _pool(self.fp, 20)
            plan = _plan(self.fp, pool, 3)
            bufs = self.bg.BatchBuffers(pool, 4, 4, include_tail=False)
            bufs.bind(plan, states, torch.tensor([3, 4, 5]), None, None)
            for branch in ("tracked", "normal") if side else ("both",):
                bufs.evaluate(self.fp.factorize_layers, branch=branch)
            outputs.append(pool)
        for name in (
            "a",
            "U",
            "W",
            "count",
            "stale",
            "dense_of",
            "dense_required",
            "prefix_valid",
            "dense_ring",
        ):
            a, b = getattr(outputs[0], name), getattr(outputs[1], name)
            self.assertTrue(
                all(torch.equal(x, y) for x, y in zip(a, b))
                if isinstance(a, list)
                else torch.equal(a, b),
                name,
            )


if __name__ == "__main__":
    unittest.main()
