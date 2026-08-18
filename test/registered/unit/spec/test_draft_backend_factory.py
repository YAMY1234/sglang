from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeBackend:
    needs_cpu_seq_lens = False

    def __init__(self, name):
        self.name = name


def test_trtllm_mha_draft_extend_keeps_configured_prefill_backend():
    server_args = SimpleNamespace(
        speculative_draft_attention_backend="trtllm_mha",
        speculative_attention_mode="decode",
        prefill_attention_backend="triton",
        attention_backend="flashinfer",
    )
    runner = SimpleNamespace(
        server_args=server_args,
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        model_config=SimpleNamespace(context_len=2048, hf_config=object()),
    )
    factory = DraftBackendFactory(
        server_args=server_args,
        draft_model_runner=runner,
        topk=1,
        speculative_num_steps=3,
    )
    prefill = _FakeBackend("prefill")
    draft_extend = _FakeBackend("draft_extend")

    with (
        patch(
            "sglang.srt.speculative.draft_utils.get_spec",
            return_value=SimpleNamespace(speculative_attention_mode="decode"),
        ),
        patch.object(
            factory,
            "_create_triton_prefill_backend",
            return_value=prefill,
        ),
        patch.object(
            factory,
            "_create_trtllm_mha_prefill_backend",
            return_value=draft_extend,
        ),
    ):
        backend = factory.create_draft_extend_backend()

    assert isinstance(backend, HybridAttnBackend)
    assert backend._select_backend(ForwardMode.EXTEND) is prefill
    assert backend._select_backend(ForwardMode.DRAFT_EXTEND_V2) is draft_extend
