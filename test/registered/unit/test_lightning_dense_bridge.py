"""CPU numerical/lifecycle gates for the native-dense Lightning bridge.

Run with --reference-repo to enforce the pinned cold/warm StateFactor gate.
CUDA qualification is separate: the CPU native stand-ins preserve the reference
operation sequence, and cannot qualify Triton's conv/readout reductions.
"""

import argparse
import ast
import copy
import json
import os
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/models")
)
from lightning_duet._common import load
from test_lightning_duet import load_reference

Bridge = load("dense_bridge").DenseBridgeStatePool
ReferencePool = load("state_pool").LeftSinkStatePool
REFERENCE = (
    load_reference(os.environ["DUET_REFERENCE_REPO"])
    if os.environ.get("DUET_REFERENCE_REPO")
    else None
)


def runtime_class():
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/models/lightning_duet/engine.py"
    )
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Runtime"
    )
    scope = {
        "torch": torch,
        "F": F,
        "copy": copy,
        "bridge_conv_update": cpu_conv,
        "bridge_state_update": cpu_ssu,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
    return scope["Runtime"]


def cpu_conv(x, conv, weight, *, activation, conv_state_indices, enable_fp_fusion):
    assert activation is None and not enable_fp_fusion
    out = torch.zeros_like(x)
    for row, slot in enumerate(conv_state_indices.tolist()):
        if slot == -1:
            continue
        full = torch.cat((conv[slot], x[row, :, None]), -1)
        out[row] = (full * weight).sum(-1)
        conv[slot] = full[:, 1:]
    return out


def cpu_ssu(state, x, dt, a, b, c, d, *, state_batch_indices, out, reference_decay):
    for row, slot in enumerate(state_batch_indices.tolist()):
        if slot == -1:
            continue
        bb, cc = [v[row].repeat_interleave(x.shape[1] // b.shape[1], 0) for v in (b, c)]
        current = (
            state[slot] * reference_decay[row, :, None, None]
            + (dt[row] * x[row])[:, :, None] * bb[:, None]
        )
        out[row] = torch.einsum("hpn,hn->hp", current, cc) + x[row] * d
        state[slot] = current


class DenseBridgeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261013)
        torch.set_num_threads(1)

    def pair(self, layers=2, heads=4, p=64, n=128, rank=16, window=16, size=5):
        directions = torch.randn(layers, heads, p)
        args = (size, tuple(range(layers)), directions)
        kwargs = {
            "n": n,
            "rank": rank,
            "window": window,
            "conv_dim": 4,
            "conv_width": 3,
        }
        new, old = Bridge(*args, **kwargs), ReferencePool(*args, **kwargs)
        for layer in range(layers):
            for slot in (1, 2, 3):
                state, conv = torch.randn(heads, p, n), torch.randn(4, 3)
                new.initialize(layer, slot, state, conv)
                old.initialize(layer, slot, state, conv)
                torch.testing.assert_close(
                    new.materialize(layer, slot),
                    old.materialize(layer, slot),
                    rtol=0,
                    atol=0,
                )
        return new, old

    def advance(self, new, old, slots):
        new.begin_step(slots, torch.tensor(slots, dtype=torch.int32))
        heads, p = new.directions.shape[1:]
        returned = {}
        for layer in new.layer_ids:
            i = new.layer_map[layer]
            for slot in slots:
                if slot == -1:
                    continue
                decay, x, b = (
                    torch.rand(heads),
                    torch.randn(heads, p),
                    torch.randn(heads, new.n),
                )
                conv = torch.randn(4, 3)
                current = (
                    new.dense[i, slot] * decay[:, None, None]
                    + x[:, :, None] * b[:, None]
                )
                expected = old.step(layer, slot, decay, x, b, conv)
                torch.testing.assert_close(current, expected, rtol=0, atol=0)
                returned[layer, slot] = current.clone()
                new.dense[i, slot] = current
                new.conv[i, slot] = conv
            new.note_layer_step(layer)
        new.finish_step()
        for layer in new.layer_ids:
            for slot in set(slots) - {-1}:
                torch.testing.assert_close(
                    new.materialize(layer, slot),
                    old.materialize(layer, slot),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    new.warm[layer, slot], old.warm[layer, slot], rtol=0, atol=0
                )
                self.assertEqual(
                    int(new.count[layer, slot]), int(old.count[layer, slot])
                )
        return returned

    def test_cold_warm_pinned_reference_real_head_geometry(self):
        if REFERENCE is None:
            self.skipTest(
                "set DUET_REFERENCE_REPO for the pinned reference test; the full CPU gate requires --reference-repo"
            )
        heads, p, n, rank = 64, 64, 128, 16
        directions = torch.randn(1, heads, p)
        pool = Bridge(
            3, (0,), directions, n=n, rank=rank, window=16, conv_dim=4, conv_width=3
        )
        policies = []
        for slot in (1, 2):
            policy = REFERENCE[0].StateFactor(1, heads, p, "left", rank, True, 16)
            policy.sink_dir[0].copy_(directions[0])
            state = torch.randn(1, heads, p, n)
            pool.initialize(0, slot, state[0], torch.zeros(4, 3))
            torch.testing.assert_close(
                pool.dense[0, slot], policy(0, state)[0], rtol=0, atol=0
            )
            policies.append(policy)
        # A single combined warm call; every slot retains its independent seed.
        states = pool.dense[0, 1:3] * 0.8 + torch.randn(2, heads, p, n) * 0.1
        pool._prune(torch.tensor([0, 0]), torch.tensor([1, 2]), states, warm=True)
        for row, policy in enumerate(policies):
            torch.testing.assert_close(
                pool.dense[0, row + 1],
                policy(0, states[row : row + 1], warm=True)[0],
                rtol=0,
                atol=0,
            )

    def test_batched_cold_warm_solver_edge_states(self):
        directions = torch.randn(2, 4, 64)
        pool = Bridge(
            4, (0, 1), directions, n=128, rank=16, window=16, conv_dim=4, conv_width=3
        )
        ref = ReferencePool(
            4, (0, 1), directions, n=128, rank=16, window=16, conv_dim=4, conv_width=3
        )
        states = [
            torch.zeros(4, 64, 128),
            torch.randn(4, 64, 128) * 1e-15,
            torch.randn(4, 64, 1) @ torch.randn(4, 1, 128),
        ]
        flat = torch.zeros(4, 64, 128)
        flat[:, :17, :17] = torch.eye(17)
        states.append(flat)
        for state in states:
            layers, slots = torch.tensor([0, 1]), torch.tensor([1, 2])
            for layer, slot in zip(layers.tolist(), slots.tolist()):
                pool.initialize(layer, slot, state.clone(), torch.zeros(4, 3))
                ref.initialize(layer, slot, state.clone(), torch.zeros(4, 3))
            # Cold and warm both share the per-request seeded probe; no batch
            # random stream or batch-dependent orientation changes the result.
            for warm in (False, True):
                pool._prune(layers, slots, torch.stack([state, state]), warm=warm)
                for layer, slot in zip(layers.tolist(), slots.tolist()):
                    ref.prune(layer, slot, state, warm=warm)
                    torch.testing.assert_close(
                        pool.dense[layer, slot],
                        ref.materialize(layer, slot),
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(bool(torch.isfinite(pool.dense[layer, slot]).all()))

    def test_33_steps_mixed_phases_and_changing_batch_bitwise(self):
        new, old = self.pair()
        for step in range(33):
            slots = [1, 2] if step < 17 else ([2, -1, 3, 1] if step % 2 else [3, 1])
            returned = self.advance(new, old, slots)
            if step in (0, 14, 15, 16, 30, 31, 32):
                # Current output is unpruned even on W16; the next state is pruned.
                if step in (15, 31):
                    self.assertFalse(torch.equal(returned[0, 1], new.dense[0, 1]))
                self.assertEqual(int(new.count[0, 1]), (step + 1) % 16)

    def test_23_layers_due_call_is_batched_and_due_only(self):
        new, old = self.pair(layers=23, heads=2)
        initial_calls = new.prune_calls
        initial_matrices = new.prune_matrices
        for _ in range(15):
            self.advance(new, old, [1, 2])
        self.assertEqual(new.prune_calls, initial_calls)
        self.advance(new, old, [1, 2])
        self.assertEqual(new.prune_calls - initial_calls, 1)
        self.assertEqual(new.prune_matrices - initial_matrices, 23 * 2)
        self.advance(new, old, [2])
        self.assertEqual(new.prune_calls - initial_calls, 1)

    def test_roundtrip_at_phases_0_15_16_17(self):
        for phase in (0, 15, 16, 17):
            with self.subTest(phase=phase):
                new, old = self.pair(layers=1)
                for _ in range(phase):
                    self.advance(new, old, [1])
                for pool in (new, old):
                    pool.copy_slots(torch.tensor([1]), torch.tensor([2]))
                    snapshot = pool.get_cpu_slots(torch.tensor([2]))
                    pool.reset_slots(torch.tensor([1, 2]))
                    pool.load_cpu_slots(snapshot, torch.tensor([2]))
                with self.assertRaises(RuntimeError):
                    new.materialize(0, 1)
                self.advance(new, old, [2])
                torch.testing.assert_close(
                    new.dense[0, 2], old.materialize(0, 2), rtol=0, atol=0
                )

    def test_native_storage_alias_and_lifecycle(self):
        dense = torch.zeros(2, 4, 4, 64, 128)
        conv = torch.zeros(2, 4, 4, 3)
        pool = Bridge(
            4,
            (0, 1),
            torch.randn(2, 4, 64),
            n=128,
            rank=16,
            window=16,
            conv_dim=4,
            conv_width=3,
            native_dense=dense,
            native_conv=conv,
        )
        self.assertEqual(pool.dense.data_ptr(), dense.data_ptr())
        self.assertEqual(pool.conv.data_ptr(), conv.data_ptr())
        self.assertFalse(hasattr(pool, "ring_x"))
        for layer in (0, 1):
            pool.initialize(layer, 1, torch.randn(4, 64, 128), torch.randn(4, 3))
        pool.copy_slots(torch.tensor([1]), torch.tensor([2]))
        snapshot = pool.get_cpu_slots(torch.tensor([2]))
        pool.reset_slots(torch.tensor([2]))
        pool.load_cpu_slots(snapshot, torch.tensor([2]))
        torch.testing.assert_close(dense[:, 2:3], snapshot["dense"], rtol=0, atol=0)
        names = [x[0] for x in pool.iter_transfer_state_entries()]
        self.assertNotIn("duet_dense", names)
        self.assertNotIn("duet_conv", names)

    def test_native_binding_rejects_wrong_geometry_and_dtype(self):
        kwargs = {"n": 128, "rank": 16, "window": 16, "conv_dim": 4, "conv_width": 3}
        for dense, conv in (
            (torch.zeros(1, 3, 4, 64, 128), None),
            (torch.zeros(2, 3, 4, 64, 128).bfloat16(), None),
            (None, torch.zeros(2, 3, 4, 4)),
            (None, torch.zeros(2, 3, 4, 3).bfloat16()),
        ):
            with self.assertRaises(ValueError):
                Bridge(
                    3,
                    (0, 1),
                    torch.randn(2, 4, 64),
                    native_dense=dense,
                    native_conv=conv,
                    **kwargs,
                )

    def test_native_parent_lifecycle_has_one_dense_writer(self):
        pool, _ = self.pair()
        sibling = pool.native_sibling()
        before = pool.get_cpu_slots(torch.tensor([1, 2]))
        # Simulate MambaPool.copy_from: native storage first, sidecar second.
        for name in ("dense", "conv"):
            value = getattr(pool, name)
            value[:, [2, 1]] = value[:, [1, 2]]
        sibling.copy_slots(torch.tensor([1, 2]), torch.tensor([2, 1]))
        after = pool.get_cpu_slots(torch.tensor([2, 1]))
        for name in pool.fields:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
        snapshot = sibling.get_cpu_slots(torch.tensor([1]))
        self.assertNotIn("dense", snapshot)
        self.assertNotIn("conv", snapshot)
        old_dense = pool.dense[:, 1].clone()
        sibling.reset_slots(torch.tensor([1]))
        torch.testing.assert_close(pool.dense[:, 1], old_dense, rtol=0, atol=0)
        sibling.load_cpu_slots(snapshot, torch.tensor([1]))
        self.assertTrue(bool(pool._host_valid[:, 1].all()))

    def test_single_writer_and_transaction_lifecycle(self):
        new, _ = self.pair()
        before = new.get_cpu_slots(torch.tensor([1]))
        with self.assertRaises(ValueError):
            new.begin_step([1, 1], torch.tensor([1, 1]))
        new.begin_step([1], torch.tensor([1]))
        for operation in (
            lambda: new.reset_slots(torch.tensor([1])),
            lambda: new.copy_slots(torch.tensor([1]), torch.tensor([2])),
            lambda: new.get_cpu_slots(torch.tensor([1])),
            lambda: new.load_cpu_slots(before, torch.tensor([1])),
            lambda: new.begin_step([2], torch.tensor([2])),
        ):
            with self.assertRaises(RuntimeError):
                operation()
        new.note_layer_step(0)
        with self.assertRaises(RuntimeError):
            new.note_layer_step(0)
        with self.assertRaises(RuntimeError):
            new.finish_step()
        new.note_layer_step(1)
        new.finish_step()
        self.assertEqual(int(new.count[0, 1]), 1)

    def test_padding_invalid_and_stale_metadata(self):
        new, old = self.pair()
        before = new.get_cpu_slots(torch.tensor([0]))
        self.advance(new, old, [-1, 1, -1])
        after = new.get_cpu_slots(torch.tensor([0]))
        for name in new.fields:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        for slots in ([0], [-2], [new.size]):
            with self.assertRaises((RuntimeError, IndexError)):
                new.begin_step(slots, torch.tensor(slots))
        with self.assertRaises(RuntimeError):
            new.begin_step([1], torch.tensor([2]))
        new.count[:, 1] += 1
        with self.assertRaises(RuntimeError):
            new.begin_step([1], torch.tensor([1]))
        new.count[:, 1] -= 1
        new.valid[:, 1] = False
        with self.assertRaises(RuntimeError):
            new.begin_step([1], torch.tensor([1]))

    def test_snapshot_validation_before_write_and_overlapping_copy(self):
        new, _ = self.pair()
        before = new.get_cpu_slots(torch.tensor([1, 2]))
        for bad in (
            {k: v for k, v in before.items() if k != "dense"},
            dict(before, count=before["count"] + 16),
            dict(before, dense=before["dense"].double()),
        ):
            with self.assertRaises(ValueError):
                new.load_cpu_slots(bad, torch.tensor([1, 2]))
        new.copy_slots(torch.tensor([1, 2]), torch.tensor([2, 1]))
        after = new.get_cpu_slots(torch.tensor([2, 1]))
        for name in new.fields:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)

    def test_exact_and_prompt_only_modes(self):
        for rank, window in ((0, 16), (16, 0)):
            new, old = self.pair(layers=1, rank=rank, window=window)
            for _ in range(33):
                self.advance(new, old, [1, 2])
            self.assertTrue(bool((new.count == 0).all()))

    def test_native_batched_decode_reference_norm_and_successor(self):
        # Exercise the actual Runtime method with native CPU stand-ins, checking
        # all intermediate states and BF16 output against the released row path.
        g = types.SimpleNamespace(
            heads=64,
            head_dim=64,
            groups=8,
            state_dim=128,
            intermediate=4096,
            conv_dim=6144,
            bc_dim=1024,
        )
        rt = runtime_class().__new__(runtime_class())
        rt.geometry = g
        rt.components = types.SimpleNamespace(eps=1e-5)

        class Projection:
            def __init__(self, output=None):
                self.output = output

            def __call__(self, hidden):
                return (hidden if self.output is None else self.output, None)

        mixer = types.SimpleNamespace(
            conv1d=types.SimpleNamespace(
                weight=torch.randn(g.conv_dim, 1, 4).bfloat16(),
                bias=torch.randn(g.conv_dim).bfloat16(),
            ),
            dt_bias=torch.randn(64),
            A=-torch.rand(64),
            D=torch.randn(64),
            norm=types.SimpleNamespace(weight=torch.randn(4096).bfloat16()),
            out_proj=Projection(),
        )
        rt.body = types.SimpleNamespace(layers=[types.SimpleNamespace(mixer=mixer)])
        for batch in (1, 2, 8):
            directions = torch.randn(1, 64, 64)
            rt.pool = Bridge(
                batch + 1,
                (0,),
                directions,
                n=128,
                rank=16,
                window=16,
                conv_dim=6144,
                conv_width=3,
            )
            for slot in range(1, batch + 1):
                rt.pool.initialize(
                    0, slot, torch.randn(64, 64, 128), torch.randn(6144, 3)
                )
            for _ in range(2):
                hidden = torch.randn(batch, 4096).bfloat16()
                projected = torch.randn(batch, 4096 + 6144 + 64).bfloat16()
                mixer.in_proj = Projection(projected)
                expected = []
                states, histories = [], []
                for row, slot in enumerate(range(1, batch + 1)):
                    gate, xbc, raw_dt = projected[row].split([4096, 6144, 64])
                    full = torch.cat((rt.pool.conv[0, slot], xbc[:, None]), -1)
                    convolved = (full.float() * mixer.conv1d.weight[:, 0].float()).sum(
                        -1
                    ) + mixer.conv1d.bias.float()
                    x, b, c = F.silu(convolved).split([4096, 1024, 1024])
                    x = x.reshape(64, 64)
                    b, c = [v.reshape(8, 128).repeat_interleave(8, 0) for v in (b, c)]
                    dt = F.softplus(raw_dt.float() + mixer.dt_bias.float())
                    state = (
                        rt.pool.dense[0, slot] * (dt * mixer.A).exp()[:, None, None]
                        + (dt[:, None] * x)[:, :, None] * b[:, None]
                    )
                    y = (
                        torch.einsum("hpn,hn->hp", state, c)
                        + x * mixer.D.float()[:, None]
                    )
                    grouped = (y.reshape(4096) * F.silu(gate.float())).reshape(8, 512)
                    grouped = grouped * torch.rsqrt(
                        grouped.square().mean(-1, keepdim=True) + 1e-5
                    )
                    expected.append(
                        (grouped.reshape(4096) * mixer.norm.weight).bfloat16()
                    )
                    states.append(state)
                    histories.append(full[:, 1:])
                rt.decode_slots = rt.pool.begin_step(
                    range(1, batch + 1), torch.arange(1, batch + 1, dtype=torch.int32)
                )
                out = rt.mamba_decode(0, hidden, None)
                rt.pool.finish_step()
                torch.testing.assert_close(out, torch.stack(expected), rtol=0, atol=0)
                torch.testing.assert_close(
                    rt.pool.dense[0, 1 : batch + 1], torch.stack(states), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    rt.pool.conv[0, 1 : batch + 1],
                    torch.stack(histories),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-repo", required=True)
    args = parser.parse_args()
    REFERENCE = load_reference(args.reference_repo)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DenseBridgeTest)
    )
    print(
        json.dumps(
            {
                "cell": "cpu-lightning-dense-bridge",
                "status": "pass"
                if result.wasSuccessful() and not result.skipped
                else "fail",
                "tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "reference_commit": "dd9c7bdbd9550a5d86781ebfa3d965d01fc1e78a",
                "cuda_initialized": torch.cuda.is_initialized(),
            }
        )
    )
    sys.exit(not result.wasSuccessful() or bool(result.skipped))
