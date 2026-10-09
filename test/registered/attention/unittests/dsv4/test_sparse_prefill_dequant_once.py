"""Compressed-workspace reuse must preserve each layer's SWA region."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.srt.layers.attention import deepseek_v4_backend as backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")

@pytest.mark.parametrize("reuse", [False, True])
def test_reuse_and_invalidation(reuse):
    attn = backend.DeepseekV4AttnBackend.__new__(backend.DeepseekV4AttnBackend)
    workspace = torch.empty((16, 1, 4), dtype=torch.bfloat16)
    ids = torch.arange(4)
    source = torch.full((8,), 11.0)
    swas = [torch.full((8,), float(i + 20)) for i in range(8)]
    cache = SimpleNamespace(
        swa_token_ids=torch.arange(2),
        swa_page_size=1,
        c0_combined_indices=torch.zeros((2, 1), dtype=torch.int32),
        c0_combined_lens=torch.ones(2, dtype=torch.int32),
    )
    cache.layer_inputs = lambda *args: (
        ids, cache.c0_combined_indices, cache.c0_combined_lens
    )
    attn.forward_metadata = SimpleNamespace(sparse_prefill_cache=cache)
    attn.sparse_prefill_workspace = SimpleNamespace(get=lambda n: workspace[:n])
    attn._compressed_in_workspace = None
    attn.softmax_scale, attn.head_dim_v = 1.0, 4
    pool = SimpleNamespace(
        get_extra_key_page_size=lambda _: 1,
        get_extra_key_buffer=lambda _: source,
        get_extra_key_layout=lambda _: None,
        get_swa_key_buffer_radix=lambda i: swas[i],
        get_swa_key_layout=lambda: None,
        request_window=None,
    )
    compressed_calls = []
    snapshots = []

    def dequant(buf, token_ids, *, page_size, out, layout):
        if buf is source:
            compressed_calls.append((source.data_ptr(), token_ids.data_ptr()))
        out.fill_(buf[0])

    def flash(**kwargs):
        snapshots.append(kwargs["kv"].clone())
        return torch.zeros_like(kwargs["q"]), None, None

    q = torch.zeros((2, 1, 1, 4), dtype=torch.bfloat16)
    with (
        envs.SGLANG_DSV4_PREFILL_DEQUANT_PER_SOURCE.override(reuse),
        mock.patch.object(backend, "dequantize_k_cache_paged", dequant),
        mock.patch("sgl_kernel.flash_mla.flash_mla_sparse_fwd", flash),
    ):
        def forward(layer, ratio=1):
            attn._forward_prefill_sparse(q, layer, ratio, None, pool, None, None)
            if ratio:
                assert torch.all(snapshots[-1][:len(ids)] == source[0])
                assert torch.all(snapshots[-1][len(ids):] == swas[layer][0])

        forward(0)
        forward(1)
        assert len(compressed_calls) == (1 if reuse else 2)
        # A distinct KV source or token selection must refresh compressed values.
        source = torch.full((8,), 33.0)
        forward(2)
        ids = ids.clone()
        forward(3)
        # A smaller compressed region changes where SWA begins.
        ids = ids[:2]
        forward(4)
        # SWA-only attention overwrites the beginning of the shared workspace.
        forward(5, ratio=0)
        forward(6)
        assert len(compressed_calls) == (5 if reuse else 6)

        # Updating the same source storage between steps must also refresh it.
        attn.mtp_enabled = False
        attn.enable_decoder_swa_bounded_replay = False
        attn.token_to_kv_pool = pool
        attn._build_forward_metadata = lambda batch: SimpleNamespace(
            sparse_prefill_cache=cache
        )
        attn.init_forward_metadata_in_graph = lambda batch: None
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, encoder_swa_replay=False
        )
        source.fill_(44)
        attn.init_forward_metadata(batch)
        forward(7)
        assert len(compressed_calls) == (6 if reuse else 7)
