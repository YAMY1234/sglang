"""CPU-only default/override contract against the actual policy module.

This policy has no Torch dependency; load it directly so default selection can
be checked on a developer machine without importing the serving stack.
"""
import importlib.util
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

    def test_unset_defaults_on_only_for_eligible_factors(self):
        for strict in (0, 1):
            for factored in (0, 1):
                for exact in (0, 1):
                    with self.subTest(strict=strict, factored=factored, exact=exact):
                        p = pool(strict, factored, exact)
                        expected = bool(strict and (factored or exact))
                        self.assertEqual(self.policy.generic_prompt_only_state_cache(p), expected)
                        self.assertEqual(self.policy.prompt_only_state_cache(self.model, p), expected)

    def test_explicit_zero_restores_generic_control(self):
        os.environ[FLAG] = '0'
        for factored, exact in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(factored=factored, exact=exact):
                p = pool(1, factored, exact)
                self.assertFalse(self.policy.generic_prompt_only_state_cache(p))
                self.assertFalse(self.policy.prompt_only_state_cache(self.model, p))

    def test_explicit_one_matches_default_eligibility(self):
        for strict in (0, 1):
            for factored in (0, 1):
                for exact in (0, 1):
                    with self.subTest(strict=strict, factored=factored, exact=exact):
                        p = pool(strict, factored, exact)
                        os.environ.pop(FLAG, None)
                        default = self.policy.prompt_only_state_cache(self.model, p)
                        os.environ[FLAG] = '1'
                        self.assertEqual(self.policy.prompt_only_state_cache(self.model, p), default)

    def test_stock_and_absent_factor_pool_are_unchanged(self):
        for p in (None, NS(), NS(factored_gdn_pool=None), NS(factored_gdn_pool=NS(cfg=None))):
            with self.subTest(pool=p):
                self.assertFalse(self.policy.generic_prompt_only_state_cache(p))
                self.assertFalse(self.policy.prompt_only_state_cache(self.model, p))

    def test_explicit_zero_preserves_existing_fullstack_policy(self):
        os.environ.update({FLAG: '0', 'TWINSTAR_FULLSTACK': '1',
                           'SGLANG_EXTERNAL_MODEL_PACKAGE': 'twinstar_sgl'})
        self.model.hf_config.twinstar = {'fullstack': {'version': 3}}
        self.assertTrue(self.policy.prompt_only_state_cache(self.model, NS()))
        os.environ['TWINSTAR_FULLSTACK'] = '0'
        self.assertFalse(self.policy.prompt_only_state_cache(self.model, NS()))

    def test_invalid_override_is_rejected(self):
        for value in ('', 'true', '2', '-1'):
            with self.subTest(value=value):
                os.environ[FLAG] = value
                with self.assertRaises(ValueError):
                    self.policy.generic_prompt_only_state_cache(pool(1, 1, 0))


def run_default_policy_checks():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PromptOnlyPolicyTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('M11 default/override CPU policy gate failed')
    return dict(passed=True, tests_run=result.testsRun)


if __name__ == '__main__':
    run_default_policy_checks()
