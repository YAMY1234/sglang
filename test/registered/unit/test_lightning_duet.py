"""CPU algebra and lifecycle tests, independent of SGLang GPU imports.

Optional --reference-repo points to an existing twinstar checkout; reference
files are read from pinned Git objects without making a source copy.
"""

import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/models"))
from lightning_duet.components import base_rms_norm, final_mamba_state
from lightning_duet.boundary import mark_runner_dummy_batches, prefill_count
from lightning_duet.latent import ResidualCode, pack_gap8, unpack_gap8, pack_nvfp4, unpack_nvfp4
from lightning_duet.state import LightningMambaStatePool, factorize

REFERENCE = None


def load_reference(path):
    source = subprocess.check_output(
        ["git", "-C", path, "show", "dd9c7bdbd9550a5d86781ebfa3d965d01fc1e78a:twinstar/duet/state.py"], text=True
    )
    state = types.ModuleType("pinned_state")
    exec(compile(source, "pinned_state.py", "exec"), state.__dict__)
    source = subprocess.check_output(
        ["git", "-C", path, "show", "dd9c7bdbd9550a5d86781ebfa3d965d01fc1e78a:twinstar/duet/latentfmt.py"], text=True
    )
    fmt = types.ModuleType("pinned_fmt")
    sys.modules[fmt.__name__] = fmt
    exec(compile(source, "pinned_fmt.py", "exec"), fmt.__dict__)
    return state, fmt


class LightningDuetTest(unittest.TestCase):
    def test_base_norm_preserves_reference_bf16_rounding(self):
        x, residual = torch.randn(5, 2688).bfloat16(), torch.randn(5, 2688).bfloat16()
        weight = torch.randn(2688).bfloat16()
        norm = types.SimpleNamespace(weight=weight, variance_epsilon=1e-5)
        h = x + residual
        expected = weight * (h.float() * torch.rsqrt(h.float().square().mean(-1, keepdim=True) + 1e-5)).bfloat16()
        out, saved = base_rms_norm(norm, x, residual)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(saved, h, rtol=0, atol=0)
        self.assertNotEqual(saved.data_ptr(), residual.data_ptr())
        fp32_sum = x.float() + residual.float()
        stock_order = (fp32_sum * torch.rsqrt(fp32_sum.square().mean(-1, keepdim=True) + 1e-5) * weight).bfloat16()
        self.assertGreater((stock_order != out).sum().item(), 0)

    def test_shallow_side_embedding_survives_native_inplace_residual(self):
        path = Path(__file__).resolve().parents[3] / "python/sglang/srt/models/lightning_duet/engine.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
        backend = types.SimpleNamespace(init_forward_metadata=lambda fb: None)
        scope = dict(torch=torch, get_attn_backend=lambda: backend, get_token_to_kv_pool=lambda: None,
                     ATTENTION_EMITTERS=(), MAMBA_EMITTERS=(), log=types.SimpleNamespace(info=lambda *args: None))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        runtime = scope["Runtime"].__new__(scope["Runtime"])
        class Layer:
            def forward(self, *, hidden_states, residual, forward_batch):
                if residual is None:
                    residual = hidden_states
                else:
                    residual.add_(1)  # Real fused RMSNorm's aliasing contract.
                return torch.zeros_like(hidden_states), residual
        def embed(ids):
            return ids.float()[:, None].expand(-1, 3).clone()
        def encode(hidden, embeddings, ids):
            torch.testing.assert_close(embeddings, embed(ids), rtol=0, atol=0)
            torch.testing.assert_close(hidden, embed(ids) + 32, rtol=0, atol=0)
            return types.SimpleNamespace(token_ids=ids, nbytes=12)
        runtime.body = types.SimpleNamespace(embed_tokens=embed, layers=[Layer() for _ in range(33)])
        runtime.components = types.SimpleNamespace(code=types.SimpleNamespace(encode=encode, decode=lambda record, emb: emb))
        runtime.pool = types.SimpleNamespace(reset_slots=lambda slots: None)
        runtime.ensure_pool = lambda: None
        runtime.mamba_ids = ()
        runtime.shallow_handoff(types.SimpleNamespace(input_ids=torch.tensor([2, 7])), 1)

    def test_batch_slice_on_nightly_without_cpu_request_indices(self):
        # Execute the actual runtime class with CPU metadata stand-ins, without
        # importing SGLang's CUDA-only dependencies. The September image lacks
        # req_pool_indices_cpu, although this later fork defines it.
        path = Path(__file__).resolve().parents[3] / "python/sglang/srt/models/lightning_duet/engine.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
        scope = dict(torch=torch, copy=copy, ForwardMode=types.SimpleNamespace(DECODE="decode", EXTEND="extend"))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        runtime = scope["Runtime"].__new__(scope["Runtime"])
        fb = types.SimpleNamespace(input_ids=torch.arange(12), positions=torch.arange(12),
            req_pool_indices=torch.tensor([3, 8]), seq_lens=torch.tensor([5, 7]),
            orig_seq_lens=None, out_cache_loc=torch.arange(12) + 100, out_cache_loc_virtual=None)
        sub = runtime.slice_batch(fb, 1, 8, 9, 4, decode=True)
        self.assertEqual(sub.req_pool_indices.tolist(), [8])
        self.assertEqual(sub.input_ids.tolist(), [8])
        self.assertEqual(sub.out_cache_loc.tolist(), [108])
        self.assertEqual(sub.seq_lens.tolist(), [4])
        self.assertEqual(sub.extend_prefix_lens_cpu, [3])
        self.assertEqual(fb.input_ids.numel(), 12)

    def test_runner_dummy_marker_is_explicit_and_instance_local(self):
        calls = []
        runner = types.SimpleNamespace(prepare_dummy_forward_batch=lambda batch: calls.append(batch) or batch)
        untouched = types.SimpleNamespace()
        mark_runner_dummy_batches(runner)
        mark_runner_dummy_batches(runner)
        dummy = runner.prepare_dummy_forward_batch(types.SimpleNamespace())
        self.assertTrue(dummy._lightning_dummy_batch)
        self.assertEqual(len(calls), 1)
        self.assertFalse(hasattr(untouched, "_lightning_dummy_batch"))

    def test_boundary_and_teacher_forcing_alignment(self):
        self.assertEqual(prefill_count(4096), 4095)
        # Prefix=4096; hidden at 4095 predicts target at 4096. All 256
        # target logprobs use full-depth recurrence after the same prefix.
        self.assertEqual(prefill_count(4096 + 256, 4095), 4095)
        self.assertEqual(prefill_count(1), 0)
        self.assertEqual(prefill_count(17, 0), 0)
        for length, start in ((0, None), (3, -1), (3, 3)):
            with self.assertRaises(ValueError):
                prefill_count(length, start)

    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_nvfp4_roundtrip_matches_quantizer(self):
        z = torch.randn(9, 2048) * torch.logspace(-5, 4, 9)[:, None]
        z[0] = 0
        packed, scales, g = pack_nvfp4(z)
        out = unpack_nvfp4(packed, scales, g)
        self.assertEqual(packed.shape, (9, 1024))
        self.assertTrue(torch.isfinite(out).all())
        if REFERENCE:
            torch.testing.assert_close(out, REFERENCE[1].fq_nvfp4(z), rtol=0, atol=0)

    def test_gap8_escapes_and_corruption(self):
        indices = torch.tensor([0, 1, 255, 511, 1800, 2687])
        stream = pack_gap8(indices)
        self.assertEqual(unpack_gap8(stream, len(indices)), indices.tolist())
        for bad in [stream[:-1], stream + [0]]:
            with self.assertRaises(ValueError):
                unpack_gap8(bad, len(indices))
        with self.assertRaises(ValueError):
            pack_gap8(torch.tensor([4, 4]))

    def test_latent_side_channel_and_stored_values(self):
        h, emb = torch.randn(5, 48), torch.randn(5, 48)
        e, d, mu = torch.randn(1, 32, 48) / 8, torch.randn(1, 48, 32) / 8, torch.randn(1, 48)
        code = ResidualCode(e, d, mu, spikes=8)
        record = code.encode(h, emb, torch.arange(5))
        result = code.decode(record, emb)
        c = h.float() - emb.float() - mu[0]
        z = c @ e[0].T
        packed, scale, global_scale = pack_nvfp4(z)
        rec = unpack_nvfp4(packed, scale, global_scale) @ d[0].T
        residual = c - rec
        index = residual.abs().topk(8, dim=-1).indices
        expected = mu[0] + rec + torch.zeros_like(rec).scatter(-1, index, residual.gather(-1, index).bfloat16().float()) + emb
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(record.values.dtype, torch.bfloat16)
        self.assertEqual(record.gaps.dtype, torch.uint8)
        self.assertGreater(record.nbytes, 0)

    def test_factorization_matches_reference_and_preserves_sink(self):
        direction = torch.randn(2, 12)
        state = torch.randn(1, 2, 12, 16)
        coeff, left, right, warm = factorize(state, direction, 4)
        result = direction[None, :, :, None] * coeff[:, :, None] + left @ right
        before = torch.einsum("bhpn,hp->bhn", state, direction)
        after = torch.einsum("bhpn,hp->bhn", result, direction)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)
        if REFERENCE:
            policy = REFERENCE[0].StateFactor(1, 2, 12, "left", 4, True, 16)
            policy.sink_dir[0].copy_(direction)
            torch.testing.assert_close(result, policy(0, state), rtol=0, atol=0)
            state = result * .8 + torch.randn_like(state) * .1
            c, u, v, _ = factorize(state, direction, 4, warm)
            torch.testing.assert_close(direction[None, :, :, None] * c[:, :, None] + u @ v,
                                       policy(0, state, warm=True), rtol=0, atol=0)

    def test_ring_cadence_and_slot_roundtrip(self):
        directions = torch.randn(2, 2, 12)
        pool = LightningMambaStatePool.for_test(4, directions, 16, 4, 16)
        state = torch.randn(2, 12, 16)
        conv = torch.randn(4, 3)
        pool.initialize(0, 1, state, conv)
        dense = pool.materialize(0, 1)
        for step in range(33):
            decay, x, b = torch.rand(2), torch.randn(2, 12), torch.randn(2, 16)
            dense = dense * decay[:, None, None] + x[:, :, None] * b[:, None]
            returned = pool.step(0, 1, decay, x, b, conv)
            torch.testing.assert_close(returned, dense, rtol=0, atol=0)
            self.assertEqual(pool.count[0, 1].item(), (step + 1) % 16)
            if (step + 1) % 16 == 0:
                dense = pool.materialize(0, 1)
        pool.copy_slots(torch.tensor([1]), torch.tensor([2]))
        torch.testing.assert_close(pool.materialize(0, 2), dense, rtol=0, atol=0)
        snapshot = pool.get_cpu_slots(torch.tensor([2]))
        pool.reset_slots(torch.tensor([2]))
        with self.assertRaises(RuntimeError):
            pool.materialize(0, 2)
        pool.load_cpu_slots(snapshot, torch.tensor([2]))
        torch.testing.assert_close(pool.materialize(0, 2), dense, rtol=0, atol=0)
        pool.reset_slots(torch.tensor([1]))
        torch.testing.assert_close(pool.materialize(0, 2), dense, rtol=0, atol=0)

    def test_write_scan_against_recurrence(self):
        for length in [1, 15, 16, 17, 35]:
            x, dt = torch.randn(1, length, 4, 8), torch.rand(1, length, 4) * .3
            a, b = -torch.rand(4), torch.randn(1, length, 2, 6)
            result = final_mamba_state(x, dt, a, b, chunk=16)
            state = torch.zeros(1, 4, 8, 6)
            for t in range(length):
                state = (state * (dt[:, t] * a).exp()[..., None, None]
                         + (x[:, t] * dt[:, t, :, None])[..., None] * b[:, t].repeat_interleave(2, 1)[:, :, None])
            torch.testing.assert_close(result, state, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-repo")
    args, rest = parser.parse_known_args()
    if args.reference_repo:
        REFERENCE = load_reference(args.reference_repo)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LightningDuetTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"cell": "cpu-lightning-duet", "status": "pass" if result.wasSuccessful() else "fail",
                      "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                      "reference_commit": "dd9c7bdbd955" if REFERENCE else None}))
    sys.exit(not result.wasSuccessful())
