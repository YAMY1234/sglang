"""CPU reference trajectory, state lifecycle and spec/CLI/env contract checks."""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4] / 'python/sglang/srt'


def load(name, relative):
    path = ROOT / relative
    if not path.is_file():
        path = Path(os.environ['DUET_BASE_SRT_ROOT']) / relative
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


policy = load('sglang.srt.model_executor.duet_policy', 'model_executor/duet_policy.py')
ref = load('sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference',
           'layers/attention/linear/kernels/gdn_prefill_reference.py')
pools = load('duet_pool_test', 'mem_cache/gdn_factored_pool.py')


def reference_source(relative):
    path = Path('/ref') / relative
    if path.is_file():
        return path.read_text()
    repo = os.environ.get('DUET_REFERENCE_REPO')
    if not repo:
        raise unittest.SkipTest('set DUET_REFERENCE_REPO or mount the pinned reference at /ref')
    return subprocess.check_output(['git', '-C', repo, 'show', f'origin/minma/0913:{relative}'], text=True)


@contextmanager
def temporary_modules(entries):
    saved = {key: sys.modules.get(key) for key in entries}
    sys.modules.update(entries)
    try:
        yield
    finally:
        # Restore only our namespace: torch may lazily register modules/operators.
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class PolicyTest(unittest.TestCase):
    def test_cli_environment_precedence_and_spec_defaults(self):
        spec = dict(state_rank=12, state_every=4)
        expected = dict(prefill_layer_trim=True, prefill_saving_policy='kv-and-ssm', decode_ssm_r=12, decode_ssm_w=4)
        from dataclasses import asdict
        common = load('sglang.srt.duet.options', 'duet/options.py')
        self.assertEqual(asdict(common.DuetOptions.resolve(spec, environ={})), expected)
        cli = SimpleNamespace(decode_ssm_r=6, prefill_layer_trim=False)
        env = dict(SGLANG_DUET_DECODE_SSM_R='5', SGLANG_DUET_DECODE_SSM_W='2',
                   SGLANG_DUET_PREFILL_LAYER_TRIM='1')
        from dataclasses import asdict
        common = load('sglang.srt.duet.options', 'duet/options.py')
        got = asdict(common.DuetOptions.resolve(spec, cli, env))
        self.assertEqual((got['decode_ssm_r'], got['decode_ssm_w'], got['prefill_layer_trim']), (6, 2, False))
        env['SGLANG_DUET_PREFILL_LAYER_TRIM'] = 'false'
        self.assertFalse(common.DuetOptions.resolve(spec, environ=env).prefill_layer_trim)
        for invalid in ('latent-only', 'latent-and-kv', 'latent-and-ssm'):
            with self.assertRaises(NotImplementedError):
                common.DuetOptions.resolve(spec, environ={'SGLANG_DUET_PREFILL_SAVING_POLICY': invalid})
        self.assertEqual(common.DuetOptions.resolve(spec, environ={'SGLANG_DUET_DECODE_SSM_W': '0'}).decode_ssm_w, 0)
        self.assertIsNone(pools.FactoredGDNConfig.parse('r=0,m=0'))

    def test_linear_code_all_tokens_and_storage_against_published_reference(self):
        from types import ModuleType
        codec = load('duet_linear_code_test', 'mem_cache/flashnext_scheme_c.py')
        parent, package = ModuleType('twinstar'), ModuleType('twinstar.duet')
        fmt, latent = ModuleType('twinstar.duet.latentfmt'), ModuleType('twinstar.duet.latent')
        package.latentfmt, parent.duet = fmt, package
        with temporary_modules({'twinstar': parent, 'twinstar.duet': package,
                'twinstar.duet.latentfmt': fmt, 'twinstar.duet.latent': latent}):
            exec(compile(reference_source('twinstar/duet/latentfmt.py'), 'reference/latentfmt.py', 'exec'), fmt.__dict__)
            exec(compile(reference_source('twinstar/duet/latent.py'), 'reference/latent.py', 'exec'), latent.__dict__)
            for width, rank, spikes in ((64, 32, 8), (1024, 32, 2)):
                with self.subTest(width=width), torch.no_grad():
                    torch.manual_seed(width)
                    original = latent.LinearCode(1, width, rank, spikes, fmt.LatentFormat('nvfp4', 'bf16', 'gap8'))
                    served = codec.FlashNextSchemeCCodec(device='cpu', width=width, rank=rank,
                        sparse=spikes, rms_normalize=False, linear_code=True)
                    for key, shape in (('E', (1, rank, width)), ('D', (1, width, rank)), ('mu', (1, width))):
                        raw = torch.randn(shape) / width ** .5
                        getattr(original, key).copy_(raw.bfloat16().float())
                        served.load(key, raw)
                    h, base = torch.randn(4, width).bfloat16(), torch.randn(4, width).bfloat16()
                    expected = original(h[:,None,:], base[:,None,:])[:,0,:]
                    payload, actual = served.encode_and_decode(h, torch.arange(4), base)
                    self.assertEqual(payload.sink_rows.numel(), 0)
                    self.assertFalse(torch.equal(expected[0], h[0]))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(served.decode(payload, base), expected, rtol=0, atol=0)

    @patch.dict(os.environ, {'SGLANG_EXTERNAL_MODEL_PACKAGE': 'twinstar_sgl', 'TWINSTAR_FULLSTACK': '1'})
    def test_generic_fullstack_uses_spec_and_no_latent_pool(self):
        fullstack = load('sglang.srt.model_executor.fullstack_policy', 'model_executor/fullstack_policy.py')
        spec = dict(latent_rank=2048, latent_spikes=128, latent_z_format='nvfp4', state_sink='explicit',
                    latent_id_side=True, latent_value_format='bf16', latent_index_format='gap8')
        fs = dict(release='/release', version=3, duet_spec=spec, latent='on', latent_rank=2048, latent_sparse=128,
                  latent_store='nvfp4', state_sink='explicit', latent_id_side=True,
                  latent_value_format='bf16', latent_index_format='gap8', latent_rms=False,
                  gdn_state='rank:12', gdn_rank=12, gdn_every=4, prefill_saving_policy='kv-and-ssm',
                  state_sink_vbar=__file__)
        config = SimpleNamespace(hf_config=SimpleNamespace(twinstar={'fullstack': fs}))
        self.assertIsNone(fullstack.fullstack_latent_config(config))
        value = fullstack.fullstack_state_config(config, radix=True)
        self.assertIn('r=12,m=4,dtype=fp32', value)
        self.assertIn('decode_method=warm', value)
        self.assertIn('exact_prefix=1', value)
        fs['latent_sparse'] = 129
        with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
            fullstack.fullstack_state_config(config)

    @patch.dict(os.environ, {'SGLANG_GDN_PREFILL_FACTOR_GRAPH': '0'})
    def test_warm_projection_and_slot_copy_restore(self):
        torch.manual_seed(23)
        r, every, h, v, k = 16, 16, 2, 64, 64
        cfg = pools.FactoredGDNConfig.parse(f'r={r},m={every},dtype=fp32,ring=2,strict_chunk=1,exact_prefix=1,init_method=k31,decode_method=warm')
        vb = torch.randn(h, v)
        def constants(pool, path, tp):
            pool.heads_total = h
            return vb.unsqueeze(0)
        cp = SimpleNamespace(shape=SimpleNamespace(temporal=(h, v, k)))
        with patch.object(pools.FactoredGDNPool, '_load_vbar', constants):
            pool = pools.FactoredGDNPool(size=3, cache_params=cp, mamba_layer_ids=[7], device='cpu', cfg=cfg)
        # Execute the actual pinned reference; do not mirror its implementation.
        from types import ModuleType
        source = reference_source('twinstar/duet/state.py')
        native = ModuleType('duet_pinned_reference_state')
        exec(compile(source, 'origin/minma/0913:twinstar/duet/state.py', 'exec'), native.__dict__)
        state = native.StateFactor(1, h, v, 'right', r, True, every)
        state.set_sink_dirs(vb.unsqueeze(0))
        def reference(s, prev):
            out = state(0, s, warm=prev is not None)
            return out, state._warm[0]
        dense = torch.randn(1, h, k, v)
        exact, prev = reference(dense, None)
        a, u, w = ref.factorize_prefill_k31(dense.transpose(-1,-2), vb, r, cfg.rmax, torch.float32, pool.init_omega(1))
        pool.a[0,1], pool.U[0,1], pool.W[0,1] = a[0], u[0], w[0]
        pool.save_warm_basis(7, torch.tensor([1]), w)
        for iteration in range(3):
            # Append independent rank-one updates, exactly representing the same dense matrix.
            for j in range(every):
                left, right = torch.randn(1,h,k), torch.randn(1,h,v)
                pool.U[0,1,:,r+j], pool.W[0,1,:,r+j] = left[0], right[0]
                exact = exact + left[...,None] * right[...,None,:]
            pool.count[0,1] = cfg.rfull
            exact, prev = reference(exact, prev)
            pool.truncate_warm(7, torch.tensor([1]))
            actual = pools.densify(pool.a[0,1:2], pool.U[0,1:2], pool.W[0,1:2], pool.count[0,1:2], vb).transpose(-1,-2)
            torch.testing.assert_close(actual, exact, rtol=3e-5, atol=3e-5)
            torch.testing.assert_close(pool.warm_v[0,1:2], prev, rtol=5e-4, atol=5e-4)
            if iteration == 0:
                pool.copy_slots(torch.tensor([1]), torch.tensor([2]))
                payload = pool.get_cpu_slots(torch.tensor([2]))
                pool.reset_slots(torch.tensor([1]))
                pool.load_cpu_slots(payload, torch.tensor([1]))
                torch.testing.assert_close(pool.warm_v[0,1], pool.warm_v[0,2], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
