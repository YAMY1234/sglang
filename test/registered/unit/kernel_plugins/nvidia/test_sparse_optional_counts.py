"""Unrequested FN select counts do not launch work or change failure policy."""

from unittest.mock import patch

import pytest
import torch
from test_sparse_score_producer import flags, indexer, tensors

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.layers.attention.qsa import qsa_indexer as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def test_unrequested_counts_do_not_clamp():
    obj = indexer()
    with flags():
        declared = obj.numeric_preparation_signatures(32768, 16)
        seen = []

        def choose(signature, *, prepared):
            assert prepared and signature in declared

            def run(io):
                if signature.operation == NumericOperation.SPARSE_SELECT:
                    assert io.counts_out is None
                    io.selected_indices_out.fill_(7)
                    seen.append(io)

            return run

        with (
            patch.object(numeric_binding, "numeric_operator", side_effect=choose),
            patch.object(
                torch.Tensor, "clamp", side_effect=AssertionError("unrequested counts")
            ),
        ):
            result = obj._numeric_score_select(*tensors(8192))
    assert len(seen) == 1 and seen[0].selected_indices_out is result
    assert torch.equal(result, torch.full((4, 512), 7, dtype=torch.int32))


@pytest.mark.parametrize(
    "failing_operation", [NumericOperation.SPARSE_SCORE, NumericOperation.SPARSE_SELECT]
)
def test_selected_write_failure_does_not_retry_stock(failing_operation):
    obj = indexer()
    calls, written = [], []
    with flags():
        declared = obj.numeric_preparation_signatures(32768, 16)

        def choose(signature, *, prepared):
            assert prepared and signature in declared

            def run(io):
                calls.append(signature.operation)
                output = (
                    io.scores_out
                    if signature.operation == NumericOperation.SPARSE_SCORE
                    else io.selected_indices_out
                )
                output.fill_(7)
                written.append(output)
                if signature.operation == failing_operation:
                    raise RuntimeError("selected write failed")

            return run

        args = tensors(8192)
        with (
            patch.object(numeric_binding, "numeric_operator", side_effect=choose),
            patch.object(impl, "qsa_mqa_decode") as stock,
        ):
            with pytest.raises(RuntimeError, match="selected write failed"):
                obj.select_decode_tokens(
                    *args[:5], torch.arange(4), args[3], True, args[5]
                )
            stock.assert_not_called()
    assert calls == (
        [NumericOperation.SPARSE_SCORE]
        if failing_operation == NumericOperation.SPARSE_SCORE
        else [NumericOperation.SPARSE_SCORE, NumericOperation.SPARSE_SELECT]
    )
    assert all(torch.all(output == 7) for output in written)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
