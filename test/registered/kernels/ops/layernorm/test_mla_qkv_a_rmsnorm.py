import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.mla_qkv_a_rmsnorm import mla_qkv_a_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="mla_qkv_a_rmsnorm is tested on CUDA",
)

# MLA kv_lora_rank and qk_rope_head_dim; q_lora_rank is parametrized.
KV_LORA, ROPE = 512, 64
Q_EPS, KV_EPS = 1e-6, 1e-5


def _norms(q_lora_rank):
    from sglang.srt.layers.layernorm import RMSNorm

    q_norm = RMSNorm(q_lora_rank, eps=Q_EPS).cuda().bfloat16()
    kv_norm = RMSNorm(KV_LORA, eps=KV_EPS).cuda().bfloat16()
    with torch.no_grad():
        q_norm.weight.uniform_(0.5, 1.5)
        kv_norm.weight.uniform_(0.5, 1.5)
    return q_norm, kv_norm


@pytest.mark.parametrize("q_lora_rank", [1536, 2048])
@pytest.mark.parametrize("num_tokens", [0, 1, 7, 128])
@pytest.mark.parametrize("row_pad", [0, 64])
def test_matches_two_rmsnorms(q_lora_rank, num_tokens, row_pad):
    """One launch matches the unfused path's two RMSNorms on the projection's
    column slices, up to the reduction order."""
    torch.manual_seed(num_tokens)
    width = q_lora_rank + KV_LORA + ROPE
    buf = torch.randn(num_tokens, width + row_pad, device="cuda", dtype=torch.bfloat16)
    latent = buf[:, :width]
    snapshot = latent.clone()
    q_norm, kv_norm = _norms(q_lora_rank)

    q, k_nope = mla_qkv_a_rmsnorm(latent, q_norm.weight, Q_EPS, kv_norm.weight, KV_EPS)

    # forward_absorb_prepare's unfused path: split, then one RMSNorm per part.
    q_ref, latent_cache = latent.split([q_lora_rank, KV_LORA + ROPE], dim=-1)
    q_ref = q_norm(q_ref)
    k_nope_ref = kv_norm(latent_cache[..., :KV_LORA])

    assert q.shape == (num_tokens, q_lora_rank) and q.is_contiguous()
    assert k_nope.shape == (num_tokens, KV_LORA) and k_nope.is_contiguous()
    torch.testing.assert_close(q, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_nope, k_nope_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(latent, snapshot, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
