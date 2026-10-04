"""CPU-only default/override contract against the actual policy module.

This policy has no Torch dependency; load it directly so default selection can
be checked on a developer machine without importing the serving stack.
"""
import ast
import importlib.util
import sys
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

POLICY_PATH = (Path(__file__).resolve().parents[2] / 'python/sglang/srt/'
               'model_executor/fullstack_policy.py')
FLAG = 'SGLANG_GDN_PROMPT_ONLY_STATE_CACHE'


def load_policy():
    spec = importlib.util.spec_from_file_location('pfactor4_policy_under_test', POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pool(strict, factored, exact):
    return NS(factored_gdn_pool=NS(cfg=NS(
        strict_chunk=strict, factored_prefix=factored, exact_prefix=exact)))


class PromptOnlyPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()

    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.model = NS(hf_config=NS())

    def initialized(self, p, **kw):
        self.policy.initialize_checkpoint_policy(self.model, p, **kw)
        return p

    def test_unset_defaults_on_only_for_eligible_factors(self):
        for strict in (0, 1):
            for factored in (0, 1):
                for exact in (0, 1):
                    with self.subTest(strict=strict, factored=factored, exact=exact):
                        p = self.initialized(pool(strict, factored, exact))
                        expected = bool(strict and (factored or exact))
                        self.assertEqual(self.policy.generic_prompt_only_state_cache(p), expected)
                        self.assertEqual(self.policy.prompt_only_state_cache(self.model, p), expected)

    def test_explicit_zero_restores_generic_control(self):
        os.environ[FLAG] = '0'
        for factored, exact in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(factored=factored, exact=exact):
                p = self.initialized(pool(1, factored, exact))
                self.assertFalse(self.policy.generic_prompt_only_state_cache(p))
                self.assertFalse(self.policy.prompt_only_state_cache(self.model, p))

    def test_explicit_one_matches_default_eligibility(self):
        for strict in (0, 1):
            for factored in (0, 1):
                for exact in (0, 1):
                    with self.subTest(strict=strict, factored=factored, exact=exact):
                        os.environ.pop(FLAG, None)
                        p = self.initialized(pool(strict, factored, exact))
                        default = self.policy.prompt_only_state_cache(self.model, p)
                        os.environ[FLAG] = '1'
                        p = self.initialized(pool(strict, factored, exact))
                        self.assertEqual(self.policy.prompt_only_state_cache(self.model, p), default)

    def test_stock_and_absent_factor_pool_are_unchanged(self):
        for p in (None, NS(), NS(factored_gdn_pool=None), NS(factored_gdn_pool=NS(cfg=None))):
            with self.subTest(pool=p):
                self.initialized(p)
                self.assertFalse(self.policy.generic_prompt_only_state_cache(p))
                self.assertFalse(self.policy.prompt_only_state_cache(self.model, p))

    def test_explicit_zero_preserves_existing_fullstack_policy(self):
        os.environ.update({FLAG: '0', 'SGLANG_DUET_DIR': 'cpu-contract-release'})
        self.model.hf_config.twinstar = {'fullstack': {'version': 3}}
        self.assertTrue(self.policy.prompt_only_state_cache(self.model, self.initialized(NS())))
        os.environ.pop('SGLANG_DUET_DIR')
        self.assertFalse(self.policy.prompt_only_state_cache(self.model, self.initialized(NS())))

    def test_all_enables_stock_decode_but_preserves_prefill(self):
        os.environ[FLAG] = 'all'
        p = self.initialized(NS(mamba_allocator=NS(size=480)))
        self.assertTrue(self.policy.prompt_only_state_cache(self.model, p))
        self.assertFalse(self.policy.prefill_prompt_only_state_cache(self.model, p))
        self.assertTrue(self.policy.prefill_prompt_only_state_cache(self.model, self.initialized(pool(1,1,0))))
        with self.assertLogs(self.policy.__name__, level='INFO') as logs:
            self.policy.report_checkpoint_policy(self.model, p)
            self.policy.report_checkpoint_policy(self.model, p)
        self.assertEqual(len(logs.output), 1)
        self.assertIn('p_only_radix=1 prefill_p_only=0', logs.output[0])
        self.assertIn('STATE_CACHE=all state_slots=480', logs.output[0])

    def test_invalid_override_is_rejected(self):
        for value in ('', 'true', '2', '-1'):
            with self.subTest(value=value):
                os.environ[FLAG] = value
                with self.assertRaises(ValueError):
                    self.initialized(pool(1, 1, 0))


    def test_invalid_override_fails_for_stock_and_fullstack_too(self):
        self.model.hf_config.twinstar = {'fullstack': {'version': 3}}
        os.environ['SGLANG_DUET_DIR'] = 'cpu-contract-release'
        for value in ('', 'true', '2', '-1'):
            with self.subTest(value=value), patch.dict(os.environ, **{FLAG:value}):
                with self.assertRaises(ValueError):
                    self.initialized(NS())

    def test_pool_parsing_is_once_and_hot_policy_never_reads_environment(self):
        for flag in ('0', '1', 'all'):
            with self.subTest(flag=flag), patch.dict(os.environ, **{FLAG:flag}):
                p = pool(1, 1, 0)
                self.policy.initialize_prompt_only_state_cache(p.factored_gdn_pool, p.factored_gdn_pool.cfg)
                os.environ[FLAG] = 'invalid-after-pool-construction'
                self.initialized(p)  # reuses the factor-pool value
                expected = flag != '0'
                with patch.object(os.environ, 'get', side_effect=AssertionError('hot env read')):
                    self.policy.initialize_prompt_only_state_cache(p.factored_gdn_pool, p.factored_gdn_pool.cfg)
                    self.initialized(p)
                    for _ in range(10):
                        self.assertEqual(self.policy.prompt_only_state_cache(self.model, p), expected)
                        self.assertEqual(self.policy.prefill_prompt_only_state_cache(self.model, p), expected)
                        self.assertEqual(self.policy.generic_prompt_only_state_cache(p), expected)

    def test_generic_p_only_speculation_is_rejected_at_startup(self):
        for flag in ('1', 'all'):
            for algorithm in ('EAGLE', 'NEXTN', 'STANDALONE'):
                for factored, exact in ((1,0), (0,1)):
                    with self.subTest(flag=flag, algorithm=algorithm, factored=factored), patch.dict(os.environ, **{FLAG:flag}):
                        with self.assertRaisesRegex(ValueError, 'generic prompt_only_state_cache'):
                            self.initialized(pool(1, factored, exact), speculative_algorithm=algorithm)

    def test_speculation_guard_does_not_claim_to_admit_other_engine_combinations(self):
        # Other factor/MTP restrictions still apply independently; this test
        # only checks the additional generic-P-only rejection boundary.
        with patch.dict(os.environ, **{FLAG:'0'}):
            self.initialized(pool(1,1,0), speculative_algorithm='EAGLE')
        with patch.dict(os.environ, **{FLAG:'all'}):
            self.initialized(NS(), speculative_algorithm='EAGLE')
        self.initialized(pool(0,1,0), speculative_algorithm='EAGLE')
        self.initialized(pool(1,1,0), speculative_algorithm=None)

    def test_startup_logs_both_policies_with_reason_and_only_once(self):
        for flag, factor, effective, reason in (
            ('0', True, 0, 'explicit-zero'), ('1', True, 1, 'eligible-factor-prefix'),
            ('1', False, 0, 'ineligible-or-dense-default'), ('all', False, 1, 'explicit-all-control')
        ):
            with self.subTest(flag=flag, factor=factor), patch.dict(os.environ, **{FLAG:flag}):
                p = pool(1,1,0) if factor else NS()
                p.mamba_allocator = NS(size=480)
                with self.assertLogs(self.policy.__name__, level='INFO') as logs:
                    self.policy.report_checkpoint_policy(self.model, p)
                    self.policy.report_checkpoint_policy(self.model, p)
                self.assertEqual(len(logs.output), 1)
                self.assertIn(f'prompt_only_state_cache={effective} ({reason})', logs.output[0])

    def test_fullstack_prefill_and_publication_are_cached_together(self):
        os.environ.update({FLAG:'0', 'SGLANG_DUET_DIR':'cpu-contract-release'})
        self.model.hf_config.twinstar = {'fullstack': {'version':3}}
        p = self.initialized(NS())
        with patch.object(os.environ, 'get', side_effect=AssertionError('hot fullstack env read')):
            self.assertTrue(self.policy.prompt_only_state_cache(self.model, p))
            self.assertTrue(self.policy.prefill_prompt_only_state_cache(self.model, p))


    def test_actual_runner_startup_forwards_algorithm_before_canary(self):
        text = POLICY_PATH.with_name('model_runner.py').read_text()
        node = next(n for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.FunctionDef) and n.name == '_init_post_memory_pool_components')
        def forbidden_canary(**kw):
            raise AssertionError('unsupported policy reached graph/canary startup')
        namespace = dict(get_spec=lambda: NS(speculative_algorithm='EAGLE'),
                         install_canary=forbidden_canary)
        exec('from __future__ import annotations\n' + ast.unparse(node), namespace)
        p = pool(1,1,0);p.mamba_allocator = NS(size=480)
        runner = NS(model_config=self.model, req_to_token_pool=p,
                    init_kv_index_translator=lambda: None)
        with patch.dict(sys.modules, {'sglang.srt.model_executor.fullstack_policy':self.policy}):
            with self.assertRaisesRegex(ValueError, 'generic prompt_only_state_cache'):
                namespace['_init_post_memory_pool_components'](runner)


def run_default_policy_checks():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PromptOnlyPolicyTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('M11 default/override CPU policy gate failed')
    return dict(passed=True, tests_run=result.testsRun)


if __name__ == '__main__':
    run_default_policy_checks()
