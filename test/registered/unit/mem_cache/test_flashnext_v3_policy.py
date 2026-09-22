"""CPU regression: Step A must never select latent/private pool machinery."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

root=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('policy',root/'python/sglang/srt/model_executor/fullstack_policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


class StepAPolicy(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'SGLANG_EXTERNAL_MODEL_PACKAGE':'twinstar_sgl','TWINSTAR_FULLSTACK':'1'})
        self.env.start();self.addCleanup(self.env.stop)
        self.fs=dict(version=2,status='component-candidate',release_name='duet-fn-v3-r4096',latent='off',
            latent_id_side=True,latent_store='fp8',latent_weight_precision='bf16-roundtrip-fp32',
            latent_rank=4096,latent_sparse=512,latent_payload_bytes=7176,
            qsa_code='off',gdn_state='rank:8',gdn_rank=8,gdn_every=8)
        self.model=SimpleNamespace(hf_config=SimpleNamespace(twinstar={'fullstack':self.fs}))

    def test_step_a_has_full_kv_and_factor_prefix(self):
        self.assertIsNone(policy.fullstack_latent_config(self.model))
        self.assertEqual(policy.fullstack_state_config(self.model,radix=True),policy.FULLSTACK_R8_RADIX_STATE)
        self.assertEqual(policy.fullstack_state_config(self.model,disaggregation_mode='decode'),
            policy.FULLSTACK_R8_STATE.replace('dtype=fp32','dtype=fp16'))
        self.assertFalse(policy.fullstack_qsa_config(self.model)['qsa_code_prefix'])

    def test_stale_private_pool_configuration_rejected(self):
        self.fs['deep_private_tokens']=4194304
        with self.assertRaises(ValueError):policy.fullstack_latent_config(self.model)

    def test_full_flag_off_preserves_stock_even_with_invalid_candidate(self):
        self.fs['latent']='invalid'
        os.environ['TWINSTAR_FULLSTACK']='0'
        self.assertIsNone(policy.fullstack_latent_config(self.model))
        self.assertIsNone(policy.fullstack_state_config(self.model,radix=True))
        self.assertIsNone(policy.fullstack_qsa_config(self.model))

    def test_legacy_latent_candidate_stays_explicit(self):
        self.fs.update(latent='on',status='latent-serving-candidate',deep_private_tokens=4194304,materialization_chunk=8192)
        self.assertIs(policy.fullstack_latent_config(self.model),self.fs)


if __name__=='__main__':unittest.main()
