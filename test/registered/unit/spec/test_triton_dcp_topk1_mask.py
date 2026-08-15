import pytest

from sglang.srt.layers.attention.triton_backend import (
    can_use_dcp_causal_chain_mask,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def test_dcp_topk1_target_verify_mask_is_causal_chain():
    assert can_use_dcp_causal_chain_mask(
        forward_mode=ForwardMode.TARGET_VERIFY,
        causal=True,
        backend_topk=1,
        spec_topk=1,
        spec_steps=3,
        draft_token_num=4,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"forward_mode": ForwardMode.EXTEND},
        {"causal": False},
        {"backend_topk": 2},
        {"spec_topk": 2},
        {"spec_steps": 2},
        {"draft_token_num": 0},
    ],
)
def test_dcp_arbitrary_custom_mask_stays_rejected(override):
    args = dict(
        forward_mode=ForwardMode.TARGET_VERIFY,
        causal=True,
        backend_topk=1,
        spec_topk=1,
        spec_steps=3,
        draft_token_num=4,
    )
    args.update(override)
    assert not can_use_dcp_causal_chain_mask(**args)
