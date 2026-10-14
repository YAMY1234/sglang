"""DSpark draft context KV on a side stream (SGLANG_OPT_DSV41_DSPARK_CONTEXT_SIDE_STREAM).

The verify epilogue projects the draft context KV beside the accept only inside a CUDA
graph capture, and the KV store then uses that projection instead of recomputing it.
"""

from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.speculative.dspark_components.dspark_verify import DsparkVerifyEpilogue
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _epilogue(stream):
    epilogue = DsparkVerifyEpilogue.__new__(DsparkVerifyEpilogue)
    epilogue._context_kv_stream = stream
    epilogue.commit_ctx = SimpleNamespace(
        resolve_pool=lambda: SimpleNamespace(set_swa_key_buffer_radix_fused_norm_rope=None),
        draft_model=SimpleNamespace(
            project_context_kv=lambda **kw: (_ for _ in ()).throw(AssertionError("forked"))
        ),
    )
    return epilogue


def test_no_fork_without_the_stream_or_outside_capture(monkeypatch):
    hidden = torch.zeros(6, 8)
    assert _epilogue(None)._fork_context_kv(hidden) is None
    # Eager steps keep the serial order even with the stream configured.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert _epilogue(object())._fork_context_kv(hidden) is None
    assert _epilogue(object())._join_context_kv(None) is None


def test_store_uses_the_precomputed_context_kv():
    stored = []
    pool = SimpleNamespace(
        set_swa_key_buffer_radix_fused_norm_rope=lambda **kw: stored.append(kw["kv"]),
        uniform_fp8=False,
    )
    stages = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                layer_id=i,
                kv_norm=SimpleNamespace(weight=SimpleNamespace(data=None)),
                eps=1e-6,
                freqs_cis=None,
            )
        )
        for i in range(3)
    ]

    def projected(*args, **kwargs):
        raise AssertionError("the store must not project again")

    model = SimpleNamespace(
        stages=stages, project_target_hidden=projected, _commit_kv_proj=projected
    )
    kvs = [torch.full((6, 4), float(i)) for i in range(3)]
    DeepseekV4ForCausalLMDSpark.write_target_hidden_kv(
        model,
        main_hidden=torch.zeros(6, 8),
        swa_loc=torch.zeros(6, dtype=torch.int64),
        positions=torch.zeros(6, dtype=torch.int64),
        pool=pool,
        context_kv=kvs,
    )
    assert stored == kvs
