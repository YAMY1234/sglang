"""CPU checks of the GLM-5.3-Flash DUET adapter's torch-free / torch-only parts (python/sglang/srt/models/glm_duet; twinstar docs/79):
the release contract derived from the spec and the base config, the release verification against it, the DSA emitter's FP8 weight
grid, the tensor-parallel head slice of the state cut's random sketch, the host-side cut schedule against the parent's
on-device schedule, and the launcher's DUET arm.

Run from this directory:  python -m unittest test_glm_duet
"""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
# the policy base class by path, registered under its import name so the policies load without sglang/__init__
_KDA = "sglang.srt.layers.attention.linear.kda_state_prune"
if _KDA not in sys.modules:
    _kda_spec = importlib.util.spec_from_file_location(_KDA, ROOT / "python/sglang/srt/layers/attention/linear/kda_state_prune.py")
    sys.modules[_KDA] = importlib.util.module_from_spec(_kda_spec)
    _kda_spec.loader.exec_module(sys.modules[_KDA])

from lightning_duet._common import load as _load  # noqa: E402
from glm_duet import emitter_math  # noqa: E402
from glm_duet import geometry as glm_geometry  # noqa: E402
from glm_duet import state_policy as glm_state_policy  # noqa: E402

release, duet_spec = (_load(n) for n in ("release", "spec"))
from glm_duet import _state_factor as state_factor
from glm_duet import _dense_state_policy as dense_state_policy

# GLM-5.3-Flash's text config (results/models/GLM-5.3-Flash/config.json, the fields the geometry reads)
GLM = {"text_config": {
    "hidden_size": 4096, "hc_mult": 4, "num_hidden_layers": 45, "mhc": True, "qk_rope_head_dim": 0, "kv_lora_rank": 512,
    "index_head_dim": 128, "index_kpool_compress": True, "index_kpool_always_select_tail": True,
    "layer_types": ["full_attention" if l % 4 == 3 else "linear_attention" for l in range(45)],
    "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
                           "kda_layers": [l for l in range(45) if l % 4 != 3]}}}
SPEC = dict(model="glm5-next", prefill_depth=31, latent_rank=2048, latent_spikes=128, latent_id_side=True, latent_z_format="nvfp4",
            latent_value_format="bf16", latent_index_format="gap8", state_rank=16, state_every=16, state_sink="explicit",
            latent_init="", state_init="", name="test-glm")


def tiny_config():
    """A small GLM-shaped config (the same layer pattern) for a release that fits in memory."""
    tc = json.loads(json.dumps(GLM["text_config"]))
    tc.update(hidden_size=16, num_hidden_layers=8, kv_lora_rank=8, index_head_dim=4,
              layer_types=["full_attention" if l % 4 == 3 else "linear_attention" for l in range(8)])
    tc["linear_attn_config"].update(num_heads=2, head_dim=4, kda_layers=[l for l in range(8) if l % 4 != 3])
    return {"text_config": tc}


def write_release(root, spec, config):
    """A release directory as twinstar.duet.export_release writes it: fp32 tensors of the derived contract, spec, manifest."""
    from safetensors.torch import save_file
    g = glm_geometry.Geometry(config)
    contract = release.build_contract(spec, g)
    gen = torch.Generator().manual_seed(0)
    tensors = {name: torch.randn(*shape, generator=gen) for name, (_, shape) in contract.expected.items()}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "duet_components.safetensors"
    save_file(tensors, str(path), metadata={"format": "pt"})
    manifest = {"name": spec["name"], "kind": "duet_glm5-next", "base_model": glm_geometry.BASE_HF,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "file_bytes": path.stat().st_size,
                "total_params": sum(t.numel() for t in tensors.values()), "provenance": {"spec": spec, "trained_spec": spec},
                "tensors": {k: {"shape": list(v.shape), "bytes": v.numel() * 4} for k, v in sorted(tensors.items())}}
    (root / "spec.json").write_text(json.dumps(spec))
    (root / "manifest.json").write_text(json.dumps(manifest))
    return tensors


class GeometryContract(unittest.TestCase):
    def test_glm_contract_shapes_and_kinds(self):
        g = glm_geometry.validate(SPEC, GLM)
        self.assertEqual((g.residual_dim, g.num_layers, g.state_heads, g.state_side_dim), (16384, 45, 64, 128))
        self.assertEqual([l for l in range(31, 45) if not g.is_kda(l)], [31, 35, 39, 43])
        c = release.build_contract(SPEC, g)
        self.assertEqual(c.expected["latent.code.E"][1], (1, 2048, 16384))
        self.assertEqual(c.expected["state.sink_dir"][1], (45, 64, 128))
        self.assertEqual(c.expected["emitters.31.mixer.kv_a_proj_with_mqa.weight"][1], (512, 4096))
        self.assertEqual(c.expected["emitters.31.collapse.hc_attn_fn"][1], (24, 16384))
        self.assertEqual(c.expected["emitters.32.mixer.A_log"][1], (1, 1, 64, 1))
        self.assertNotIn("emitters.32.mixer.q_proj.weight", c.expected)             # the q path is inherited from the base layer
        self.assertIn("model.language_model.layers.32.self_attn.q_proj.weight", c.inherited)
        self.assertEqual(sorted({int(n.split(".")[1]) for n in c.expected if n.startswith("emitters.")}), list(range(31, 45)))
        # emitter parameters at k=31: 10 KDA + 4 DSA layers
        kda = sum(1 for l in range(31, 45) if g.is_kda(l))
        self.assertEqual(kda, 10)

    def test_validate_refuses_other_models_and_out_of_range_cuts(self):
        for bad in ({"model": "kimi-linear"}, {"prefill_depth": 45}, {"prefill_depth": 0}, {"state_rank": 128}, {"latent_rank": 2050}):
            with self.assertRaises((ValueError, NotImplementedError)):
                glm_geometry.validate({**SPEC, **bad}, GLM)
        self.assertEqual(glm_geometry.MODEL, "glm5-next")


class ReleaseVerification(unittest.TestCase):
    def test_round_trip_and_tamper(self):
        cfg = tiny_config()
        spec = {**SPEC, "prefill_depth": 3, "latent_rank": 32, "latent_spikes": 4, "state_rank": 2, "state_every": 2}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "rel"
            tensors = write_release(root, spec, cfg)
            g = glm_geometry.validate(spec, cfg)
            ident = release.verify_release(root, geometry=g, model="glm5-next")
            self.assertEqual(ident.audit["tensor_count"], len(tensors))
            loaded = release.load_components(root, release.build_contract(spec, g))
            self.assertTrue(all(torch.equal(loaded[k], tensors[k]) for k in tensors))
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["tensors"].pop("state.sink_dir")
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                release.verify_release(root, geometry=g, model="glm5-next")


class EmitterMath(unittest.TestCase):
    """The served emitters' arithmetic (emitter_math): the packed varlen convolution equals the per-request ShortConvF32 form bit for
    bit; the FP8 activation grid is exact in bf16 and idempotent; the collapse matches an fp64 transcription."""

    def test_packed_conv_equals_per_request_short_conv(self):
        gen = torch.Generator().manual_seed(3)
        C, K, lens = 6, 4, [5, 1, 9]
        w = torch.randn(C, K, generator=gen)
        hist = [torch.zeros(K - 1, C), torch.randn(K - 1, C, generator=gen), torch.randn(K - 1, C, generator=gen)]
        xs = [torch.randn(n, C, generator=gen) for n in lens]
        rows, starts, at = [], [], 0
        for h, x in zip(hist, xs):
            starts.append(at); rows += [h, x]; at += K - 1 + x.shape[0]
        got = emitter_math.conv_taps(torch.cat(rows), w, starts, lens, K)
        want = []
        for h, x in zip(hist, xs):                                   # twinstar ShortConvF32.prefill / _taps, (C, K-1+T) layout
            full = torch.cat([h, x]).t()[None]
            y = full[:, :, 0:x.shape[0]] * w[None, :, 0:1]
            for k in range(1, K):
                y = y + full[:, :, k:k + x.shape[0]] * w[None, :, k:k + 1]
            want.append(torch.nn.functional.silu(y)[0].t())
        self.assertTrue(torch.equal(got, torch.cat(want)))

    def test_fp8_activation_grid(self):
        x = (torch.randn(5, 256, generator=torch.Generator().manual_seed(4)) * 3).to(torch.bfloat16)
        q = emitter_math.fake_quant_act(x, "ue8m0")
        self.assertTrue(torch.equal(q, q.to(torch.bfloat16).float()))            # e4m3 x 2^e is exact in bf16
        self.assertTrue(torch.equal(emitter_math.fake_quant_act(q.to(torch.bfloat16), "ue8m0"), q))
        self.assertLess(float(((q - x.float()).abs() / x.float().abs().amax(-1, keepdim=True)).max()), 2 ** -3)
        s = emitter_math.ceil_ue8m0(torch.tensor([0.3, 1.0, 1.5, 4.0]))
        self.assertEqual(s.tolist(), [0.5, 1.0, 2.0, 4.0])

    def test_collapse_against_fp64(self):
        gen = torch.Generator().manual_seed(5)
        n, D, T = 4, 32, 7
        h = torch.randn(T, n * D, generator=gen).to(torch.bfloat16)
        fn, scale, base, w = torch.randn(24, n * D, generator=gen) * 0.1, torch.rand(3, generator=gen), torch.randn(24, generator=gen), \
            torch.rand(D, generator=gen) + 0.5
        got = emitter_math.collapse(h, fn, scale, base, w, hc_mult=n, eps=1e-5, hc_eps=1e-6)
        hd = h.double().view(T, n, D)
        rs = torch.rsqrt(hd.view(T, -1).pow(2).mean(-1, keepdim=True) + 1e-5)
        pre = torch.sigmoid((hd.view(T, -1) @ fn[:n].double().t()) * rs * scale[0].double() + base[:n].double()) + 1e-6
        acc = (pre[:, :, None] * hd).sum(1)
        want = acc * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + 1e-5) * w.double()
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertLess(float((got.double() - want).abs().max() / want.abs().max()), 1e-2)


class StateSketchHeadSlice(unittest.TestCase):
    def test_rank_slice_of_heads_equals_the_unsharded_projection(self):
        gen = torch.Generator().manual_seed(2)
        S = torch.randn(3, 8, 16, 16, generator=gen)
        d = torch.randn(8, 16, generator=gen)
        full, _ = state_factor.project_state_batched(S, d, 4, [None] * 3)
        for rank in range(4):
            h = slice(2 * rank, 2 * rank + 2)
            part, bases = state_factor.project_state_batched(S[:, h].contiguous(), d[h], 4, [None] * 3, omega_heads=(2 * rank, 8))
            torch.testing.assert_close(part, full[:, h], rtol=1e-4, atol=1e-4)
            warm, _ = state_factor.project_state_batched(S[:, h].contiguous(), d[h], 4, bases, omega_heads=(2 * rank, 8))
            self.assertEqual(tuple(warm.shape), (3, 2, 16, 16))
        # without the head offset a slice draws other numbers (the reason for omega_heads)
        other, _ = state_factor.project_state_batched(S[:, 2:4].contiguous(), d[2:4], 4, [None] * 3)
        self.assertFalse(torch.allclose(other, full[:, 2:4], rtol=1e-6, atol=1e-6))


class _Pool:
    """The slice of HybridReqToTokenPool the policies read: the dense KDA pool (layers, slots, H, V, K), the layer map, the request ->
    slot map and the sibling registration."""

    def __init__(self, temporal, slot_of_request):
        self.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(temporal=temporal), register_slot_state=lambda state: None)
        self.slot_of_request = slot_of_request

    def mamba2_layer_index(self, layer):
        return layer

    def get_mamba_indices(self, req_pool_indices):
        return self.slot_of_request[req_pool_indices]


class HostCutSchedule(unittest.TestCase):
    def test_host_schedule_cuts_like_the_parent(self):
        """Requests arriving, decoding in changing batches and leaving: the host-scheduled GLM policy leaves the pool, the warm bases
        and the cut counts bit-identical to DenseStatePolicy's on-device schedule (the flow the adapter used before)."""
        layers, slots, heads, v, k, every = 2, 5, 2, 8, 6, 3
        gen = torch.Generator().manual_seed(3)
        directions = torch.randn(layers, heads, v, generator=gen)
        spec = {"state_rank": 2, "state_every": every, "state_sink": "explicit"}
        slot_of_request = torch.full((16,), -1, dtype=torch.long)
        start = torch.randn(layers, slots, heads, v, k, generator=gen)
        pools = [_Pool(start.clone(), slot_of_request), _Pool(start.clone(), slot_of_request)]
        parent = dense_state_policy.DenseStatePolicy(pools[0], directions, spec, list(range(layers)), side="right",
                                                     pool_transposed=True, tag="parent")
        host = glm_state_policy.GlmStatePolicy(pools[1], directions, spec, list(range(layers)), omega_heads=None)
        policies = (parent, host)

        def batch(requests, with_cpu=True):
            rpi = torch.tensor(requests, dtype=torch.long)
            return SimpleNamespace(req_pool_indices=rpi, req_pool_indices_cpu=rpi.clone() if with_cpu else None, batch_size=len(requests))

        def step(requests, with_cpu=True):
            b = batch(requests, with_cpu)
            decay = torch.rand(heads, 1, 1, generator=gen) * 0.5 + 0.5
            write = torch.randn(len(requests), heads, v, k, generator=gen)
            for pol, pool in zip(policies, pools):
                pol.before_graph(b)
                temporal = pool.mamba_pool.mamba_cache.temporal
                for i, r in enumerate(requests):
                    s = int(slot_of_request[r])
                    temporal[:, s] = temporal[:, s] * decay + write[i]
                pol.after_graph(b)

        running, free, next_request = [], list(range(slots)), 0
        for tick in range(60):
            if free and (tick % 4 == 0 or not running):       # admit: slot reset, prompt-final states, boundary step alone
                s, r = free.pop(0), next_request
                next_request = (next_request + 1) % 16
                slot_of_request[r] = s
                prompt = torch.randn(layers, heads, v, k, generator=gen)
                for pol, pool in zip(policies, pools):
                    pol.reset_slots(torch.tensor([s]))
                    pool.mamba_pool.mamba_cache.temporal[:, s] = prompt
                parent.pending_prefix[:, s] = True
                parent.count[:, s] = 0
                host.mark_prompt_end([r], [s])
                step([r])
                running.append(r)
            step(list(reversed(running)) if tick % 3 else running, with_cpu=tick % 5 != 0)
            if tick % 7 == 6 and len(running) > 1:            # a request leaves; its slot is reset at the next admission
                r = running.pop(1)
                free.append(int(slot_of_request[r]))
            self.assertTrue(torch.equal(pools[0].mamba_pool.mamba_cache.temporal, pools[1].mamba_pool.mamba_cache.temporal), tick)
            self.assertEqual(sorted(parent.warm), sorted(host.warm), tick)
            self.assertEqual((parent.prefix_cuts, parent.decode_cuts), (host.prefix_cuts, host.decode_cuts), tick)
        self.assertGreater(host.decode_cuts, 10 * layers)
        self.assertGreater(host.prefix_cuts, 5 * layers)


class LauncherDuetArm(unittest.TestCase):
    def test_duet_arm_differs_from_the_base_arm_only_by_the_release(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("launch_duet", ROOT / "benchmark/duet_eval/launch_duet.py")
        ld = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ld)
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "glm"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next"}))
            rel = Path(d) / "rel"
            write_release(rel, {**SPEC, "prefill_depth": 3, "latent_rank": 32, "latent_spikes": 4, "state_rank": 2, "state_every": 2},
                          tiny_config())
            argv = {}
            for arm in ("stock", "duet"):
                out = Path(d) / arm
                self.assertEqual(ld.main(["--model", str(model), "--duet-release", str(rel), "--arm", arm, "--port", "30600",
                                          "--gpus", "0,1,2,3", "--max-running-requests", "256", "--dry-run", "--out", str(out)]), 0)
                record = json.loads((out / "launch.json").read_text())
                argv[arm] = record["argv"]
                if arm == "duet":
                    self.assertEqual(record["env"]["SGLANG_DUET_DIR"], str(rel.resolve()))
            extra = argv["duet"][len(argv["stock"]):]
            self.assertEqual(argv["duet"][:len(argv["stock"])], argv["stock"])
            self.assertEqual(extra, ["--duet-release", str(rel.resolve()), "--duet-numerics", "reference"])
            out = Path(d) / "abl"
            ld.main(["--model", str(model), "--duet-release", str(rel), "--arm", "duet", "--port", "30600", "--extra=--decode-ssm-r 0",
                     "--disable-overlap-schedule", "--dry-run", "--out", str(out)])
            a = json.loads((out / "launch.json").read_text())["argv"]
            self.assertIn("--disable-overlap-schedule", a)
            self.assertEqual(a[a.index("--decode-ssm-r") + 1], "0")
            ident = {"name": "x", "sha256": "c" * 64, "spec": {"prefill_depth": 3, "state_rank": 2, "state_every": 2}}
            served = "GLM DUET loaded: x tensors=3 sha256=%s\n" % ("c" * 64) + "DUET served cut: k=3 trim=True r=2 W=2 sha256=%s\n" % ("c" * 64)
            self.assertTrue(ld.check_arm("glm", "duet", served * 4, ident, {})[0])
            self.assertFalse(ld.check_arm("glm", "duet", served.replace("k=3", "k=4"), ident, {})[0])
            self.assertFalse(ld.check_arm("glm", "stock", served, ident, {})[0])


if __name__ == "__main__":
    unittest.main()
