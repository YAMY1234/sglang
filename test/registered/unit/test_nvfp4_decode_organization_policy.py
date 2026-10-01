"""CPU regression for spec-driven pool/dispatcher organization agreement."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[3] / 'python/sglang/srt/model_executor/fullstack_policy.py'


class OrganizationPolicy(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('policy_under_test', SOURCE)
        self.policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.policy)
        self.fs = dict(duet_spec={}, gdn_rank=16, gdn_every=16,
                       state_sink_vbar=str(SOURCE), duet_prefix_state='factored')
        for attr, value in (
            ('fullstack_enabled', lambda m: True),
            ('fullstack_config', lambda m: self.fs),
            ('fullstack_v3_config', lambda m: self.fs),
            ('_duet_options', lambda: NS(resolve_prefix_state=lambda opts: opts.duet_prefix_state)),
        ):
            p = patch.object(self.policy, attr, value);p.start();self.addCleanup(p.stop)

    def resolve(self, **env):
        with patch.dict(os.environ, env, clear=True):
            return self.policy.fullstack_state_config(None, radix=True)

    def test_default_string_preserved(self):
        self.assertEqual(self.resolve(),
            f'r=16,m=16,dtype=fp32,ring=16,async=1,strict_chunk=1,init_method=k31,'
            f'decode_method=iter,vbar={SOURCE},factored_prefix=1')

    def test_three_organizations(self):
        for kernel, async_flag, effective in [('split','1',1),('split','0',0),('fused','1',0),('fused','0',0)]:
            with self.subTest(kernel=kernel, async_flag=async_flag):
                value=self.resolve(SGLANG_GDN_FACTORED_KERNEL=kernel,
                                   SGLANG_GDN_FACTORED_ASYNC_TRUNC=async_flag)
                self.assertIn(f',async={effective},',value)
                self.assertTrue(value.endswith(',kernel='+kernel))

    def test_invalid_and_warm_combinations_fail(self):
        for env in [dict(SGLANG_GDN_FACTORED_KERNEL='typo'),
                    dict(SGLANG_GDN_FACTORED_ASYNC_TRUNC='2'),
                    dict(SGLANG_DUET_DECODE_METHOD='warm',SGLANG_GDN_FACTORED_KERNEL='fused'),
                    dict(SGLANG_DUET_DECODE_METHOD='warm',SGLANG_GDN_FACTORED_ASYNC_TRUNC='1')]:
            with self.subTest(env=env),self.assertRaises(ValueError):self.resolve(**env)
        value=self.resolve(SGLANG_DUET_DECODE_METHOD='warm')
        self.assertIn(',async=0,',value);self.assertIn(',decode_method=warm,',value)

    def test_legacy_bf16_release_ignores_spec_only_controls(self):
        self.fs=dict(gdn_state='rank:8',version=3,duet_prefix_state='factored')
        default=self.resolve()
        self.assertEqual(default,self.resolve(SGLANG_GDN_FACTORED_KERNEL='fused',SGLANG_GDN_FACTORED_ASYNC_TRUNC='0'))
        self.assertEqual(default,self.policy.FULLSTACK_R8_RADIX_STATE)


if __name__ == '__main__':unittest.main()
