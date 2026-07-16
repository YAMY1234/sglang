import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    should_skip_auto_prefill_cuda_graph_for_memory,
)
from sglang.srt.model_executor.runner import base_cuda_graph_runner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_auto_prefill_cuda_graph_memory_gate():
    assert should_skip_auto_prefill_cuda_graph_for_memory(3.99, set())
    assert not should_skip_auto_prefill_cuda_graph_for_memory(4.0, set())


def test_explicit_prefill_backend_bypasses_memory_gate():
    assert not should_skip_auto_prefill_cuda_graph_for_memory(
        0.0, {(Phase.PREFILL, "backend")}
    )


def test_legacy_flashinfer_gdn_shares_graph_coverage_with_draft():
    configured_bs = [1, 64]
    server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL, bs=configured_bs)
        ),
        speculative_eagle_topk=1,
        speculative_adaptive=False,
        enable_pdmux=False,
        enable_lora=False,
        disable_cuda_graph_padding=False,
        enable_two_batch_overlap=False,
        _generate_decode_cuda_graph_batch_sizes=lambda max_bs: configured_bs + [max_bs],
    )
    target = SimpleNamespace(
        server_args=server_args,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
        attn_backend=SimpleNamespace(
            linear_attn_backend=SimpleNamespace(
                kernel_dispatcher=SimpleNamespace(
                    verify_kernel=SimpleNamespace(
                        requires_exact_batch_graph_coverage=True
                    )
                )
            )
        ),
        req_to_token_pool=SimpleNamespace(size=128),
    )
    draft = SimpleNamespace(
        server_args=server_args,
        req_to_token_pool=SimpleNamespace(size=128),
    )

    base_cuda_graph_runner.maybe_set_legacy_flashinfer_gdn_graph_bs_override(target)
    with patch.multiple(
        base_cuda_graph_runner,
        require_gathered_buffer=lambda _: False,
        get_parallel=lambda: SimpleNamespace(attn_cp_size=1),
        get_flags=lambda: SimpleNamespace(
            capture=SimpleNamespace(enable_torch_compile=False)
        ),
    ):
        assert max(base_cuda_graph_runner.get_batch_sizes_to_capture(draft)[0]) == 128
    assert max(server_args.cuda_graph_config.decode.bs) == 64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
