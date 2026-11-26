"""DSA target-verify q-prep on alt_stream: parity with the raw-FP8 backend path
under CUDA graph capture and changed-input replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.utils import mla_quantize_and_rope_for_fp8
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    DeepseekMLAForwardMixin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_HEADS = 16
_NOPE = 192
_RANK = 512
_ROPE = 64
_SLOTS = 256
_MAX_POS = 4096


class _KVLayers:
    """Stands in for the KV layer router: store_mla writes the FP8 bytes by slot."""

    def __init__(self):
        self.cache = torch.zeros(
            _SLOTS, _RANK + _ROPE, dtype=torch.uint8, device="cuda"
        )

    def layer(self, layer_id):
        return self

    def store_mla(self, layer, loc_info, cache_k_nope, cache_k_rope):
        self.cache[loc_info.loc] = torch.cat(
            (cache_k_nope.view(torch.uint8), cache_k_rope.view(torch.uint8)), dim=-1
        )


class TestVerifyQprepAltStream(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("FP8 RoPE/quantize needs SM90+")

    def test_capture_replay_matches_backend_path(self):
        cos_sin_cache = torch.randn(_MAX_POS, _ROPE, device="cuda", dtype=torch.float32)
        for num_tokens in (8, 24):
            with self.subTest(num_tokens=num_tokens):
                # Graph inputs, refilled before each replay.
                q_src = torch.empty(
                    _HEADS, num_tokens, _RANK, device="cuda", dtype=torch.bfloat16
                )
                q_b_out = torch.empty(
                    num_tokens,
                    _HEADS,
                    _NOPE + _ROPE,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                latent = torch.empty(
                    num_tokens, _RANK + _ROPE, device="cuda", dtype=torch.bfloat16
                )
                forward_batch = SimpleNamespace(
                    positions=torch.zeros(num_tokens, dtype=torch.int64, device="cuda"),
                    out_cache_loc=torch.zeros(
                        num_tokens, dtype=torch.int64, device="cuda"
                    ),
                    out_cache_loc_is_physical=False,
                )
                owner = SimpleNamespace(
                    rotary_emb=SimpleNamespace(
                        cos_sin_cache=cos_sin_cache, is_neox_style=False
                    ),
                    kv_lora_rank=_RANK,
                    qk_rope_head_dim=_ROPE,
                    attn_mqa=SimpleNamespace(layer_id=0, is_cross_attention=False),
                    _kv_cache_saved_early=False,
                )
                alt_stream = torch.cuda.Stream()
                side_kv, ref_kv = _KVLayers(), _KVLayers()

                def refill():
                    q_src.normal_()
                    q_b_out.normal_()
                    latent.normal_()
                    forward_batch.positions.copy_(
                        torch.randint(0, _MAX_POS, (num_tokens,), device="cuda")
                    )
                    forward_batch.out_cache_loc.copy_(
                        torch.randperm(_SLOTS, device="cuda")[:num_tokens]
                    )

                def inputs():
                    # Main-stream producers with the model's layouts: the absorbed
                    # bmm output is [H, T, rank] transposed, q_pe and k_pe are views
                    # of q_b_proj and the latent, k_nope is the kv_a_layernorm output.
                    q_nope_out = (q_src * 1).transpose(0, 1)
                    q_pe = (q_b_out * 1)[..., _NOPE:]
                    k_nope = (latent[..., :_RANK] * 1).unsqueeze(1)
                    k_pe = (latent * 1)[..., _RANK:].unsqueeze(1)
                    return q_nope_out, q_pe, k_nope, k_pe

                def side():
                    q_nope_out, q_pe, k_nope, k_pe = inputs()
                    current_stream = torch.cuda.current_stream()
                    alt_stream.wait_stream(current_stream)
                    with torch.cuda.stream(alt_stream):
                        q_fp8, _, _ = (
                            DeepseekMLAForwardMixin._rope_quantize_and_store_fp8(
                                owner, q_nope_out, q_pe, k_nope, k_pe, forward_batch
                            )
                        )
                    current_stream.wait_stream(alt_stream)
                    # Main-stream consumer after the join, as attention is.
                    return q_fp8.view(torch.uint8).clone()

                def backend_path():
                    # What the raw-FP8 DSA backend does with q_rope and cos_sin_cache.
                    q_nope_out, q_pe, k_nope, k_pe = inputs()
                    q_fp8, k_fp8, k_rope_fp8 = mla_quantize_and_rope_for_fp8(
                        q_nope_out,
                        q_pe,
                        k_nope.squeeze(1),
                        k_pe.squeeze(1),
                        forward_batch.positions,
                        cos_sin_cache,
                        False,
                        _RANK,
                        _ROPE,
                    )
                    ref_kv.store_mla(
                        owner.attn_mqa,
                        KVWriteLoc.for_layer(forward_batch, owner.attn_mqa),
                        k_fp8,
                        k_rope_fp8,
                    )
                    return q_fp8.view(torch.uint8)

                path = (
                    "sglang.srt.models.deepseek_common.attention_forward_methods."
                    "forward_mla.get_kv_layers"
                )
                with patch(path, return_value=side_kv):
                    refill()
                    # Warm up allocations/JIT outside capture.
                    side()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = side()
                self.assertTrue(owner._kv_cache_saved_early)

                for _ in range(3):
                    # Changed inputs expose stale reads or missing dependencies.
                    refill()
                    side_kv.cache.zero_()
                    ref_kv.cache.zero_()
                    expected = backend_path()
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(captured, expected), "q_fp8")
                    self.assertTrue(
                        torch.equal(side_kv.cache, ref_kv.cache), "kv cache"
                    )


if __name__ == "__main__":
    unittest.main()
