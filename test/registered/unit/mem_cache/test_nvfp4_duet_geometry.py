"""CPU regression for 8192+256 geometry and the distinct inference r16/W16 policy."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


layout = module("nvfp4_layout", "mem_cache/flashnext_latent_layout.py")
module("sglang.srt.model_executor.duet_policy", "model_executor/duet_policy.py")
policy = module(
    "sglang.srt.model_executor.fullstack_policy", "model_executor/fullstack_policy.py"
)


class GeometryTest(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_rank16_k31_does_not_require_grouped_lu_kernel(self):
        cfg = SimpleNamespace(r=16, strict_chunk=1, init_method="k31")
        self.assertFalse(policy.factored_batch_layers_enabled(cfg))
        cfg.r = 8
        self.assertTrue(policy.factored_batch_layers_enabled(cfg))
        cfg.r, cfg.init_method = 16, "iter"
        self.assertTrue(policy.factored_batch_layers_enabled(cfg))
        os.environ["SGLANG_GDN_FACTORED_BATCH_LAYERS"] = "0"
        self.assertFalse(policy.factored_batch_layers_enabled(cfg))
        self.assertFalse(policy.factored_batch_layers_enabled(None))

    def test_tp_geometry_and_capacity(self):
        one = layout.FlashNextLatentLayout(1, rank=8192, sparse=256, scheme_c=True)
        two = layout.FlashNextLatentLayout(2, rank=8192, sparse=256, scheme_c=True)
        self.assertEqual((one.local_rank, two.local_rank), (8192, 4096))
        self.assertEqual((one.local_sparse, two.local_sparse), (256, 128))
        self.assertEqual(two.token_bytes * 2 - one.token_bytes, 14)
        self.assertGreater(one.token_bytes, 5380)  # escape capacity and metadata
        with self.assertRaises(ValueError):
            layout.FlashNextLatentLayout(2, rank=8192, sparse=255, scheme_c=True)

    def test_codec_buffers_follow_spec(self):
        codec = module("nvfp4_codec", "mem_cache/flashnext_scheme_c.py")
        model = codec.FlashNextSchemeCCodec(
            device="meta", rank=8192, sparse=256, rms_normalize=False
        )
        self.assertEqual(tuple(model.E.shape), (8192, 10240))
        self.assertEqual(tuple(model.D.shape), (10240, 8192))
        self.assertEqual(model.PAYLOAD_BYTES, 5380)
        old = codec.FlashNextSchemeCCodec(device="meta")
        self.assertEqual((old.RANK, old.SPIKES, old.PAYLOAD_BYTES), (4096, 512, 3848))

    def test_fp4_roundtrip_8192_coordinates(self):
        import torch

        codec = module("nvfp4_codec_roundtrip", "mem_cache/flashnext_scheme_c.py")
        x = torch.zeros(2, 8192)
        x[1] = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 1024)
        z, scales, global_scale = codec.pack_nvfp4(x)
        self.assertEqual(tuple(z.shape), (2, 4096))
        decoded = codec.unpack_nvfp4(z, scales, global_scale)
        # The reciprocal global scale incurs one fp32 multiplication rounding.
        torch.testing.assert_close(x, decoded, rtol=0, atol=1e-6)
        self.assertTrue(torch.equal(decoded[0], x[0]))

    @patch.dict(
        os.environ,
        {
            "SGLANG_EXTERNAL_MODEL_PACKAGE": "twinstar_sgl",
            "TWINSTAR_FULLSTACK": "1",
            "SGLANG_FLASHNEXT_DENSE_STATE_ABLATION": "0",
        },
    )
    def test_policy_and_flagoff(self):
        fs = dict(
            release="/release",
            version=3,
            release_name="checkpoint-defined-name",
            latent="on",
            status="latent-serving-candidate",
            latent_id_side=True,
            latent_store="nvfp4",
            latent_weight_precision="bf16-roundtrip-fp32",
            latent_rank=8192,
            latent_sparse=256,
            latent_payload_bytes=5380,
            latent_rms=False,
            latent_value_format="bf16",
            latent_index_format="gap8",
            deep_gdn_prefix=True,
            qad=True,
            qsa_code="off",
            gdn_state="rank:16",
            gdn_rank=16,
            gdn_every=16,
            state_sink="explicit",
            gdn_prefill_truncation="k31-warm-subspace",
            state_sink_vbar=str(Path(__file__)),
            deep_private_tokens=65536,
            materialization_chunk=8192,
        )
        fs.update(
            prefill_saving_policy="kv-and-ssm",
            duet_spec=dict(
                latent_rank=8192,
                latent_spikes=256,
                latent_z_format="nvfp4",
                state_sink="explicit",
                latent_id_side=True,
                latent_value_format="bf16",
                latent_index_format="gap8",
            ),
        )
        config = SimpleNamespace(hf_config=SimpleNamespace(twinstar={"fullstack": fs}))
        self.assertIs(policy.fullstack_v3_config(config), fs)
        self.assertTrue(policy.fullstack_state_config(config).startswith("r=16,m=16,"))
        fs["gdn_every"] = (
            8  # An explicit decode override is independent of training provenance.
        )
        self.assertTrue(policy.fullstack_state_config(config).startswith("r=16,m=8,"))
        fs["latent_sparse"] = 128
        with self.assertRaises(ValueError):
            policy.fullstack_state_config(config)
        config.hf_config.twinstar = None
        os.environ.pop("SGLANG_DUET_DIR", None)
        self.assertIsNone(policy.fullstack_state_config(config))
        self.assertIsNone(policy.fullstack_latent_config(config))


if __name__ == "__main__":
    unittest.main()
