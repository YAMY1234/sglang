import pytest
import torch

from sglang.srt.layers.attention.triton_backend import (
    can_use_dcp_causal_chain_mask,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tree-mask kernel")
def test_topk1_tree_kernel_emits_prefix_plus_causal_triangle():
    bs = 2
    num_steps = 3
    num_draft_tokens = num_steps + 1
    # The CUDA tree builder consumes the scheduler's int64 sequence lengths.
    seq_lens = torch.tensor([3, 5], dtype=torch.int64, device="cuda")
    bonus_tokens = torch.zeros(bs, dtype=torch.long, device="cuda")
    parent_list = torch.arange(-1, num_steps - 1, device="cuda").repeat(bs, 1)
    top_scores_index = torch.arange(num_steps, device="cuda").repeat(bs, 1)
    draft_tokens = torch.zeros(
        (bs, num_steps), dtype=torch.long, device="cuda"
    )

    tree_mask, *_ = build_tree_kernel_efficient(
        bonus_tokens,
        parent_list,
        top_scores_index,
        draft_tokens,
        seq_lens,
        int(seq_lens.sum().item()),
        topk=1,
        spec_steps=num_steps,
        num_verify_tokens=num_draft_tokens,
        tree_mask_mode=TreeMaskMode.FULL_MASK,
    )

    offset = 0
    causal = torch.tril(
        torch.ones(
            num_draft_tokens,
            num_draft_tokens,
            dtype=torch.bool,
            device="cuda",
        )
    )
    for seq_len in seq_lens.tolist():
        width = seq_len + num_draft_tokens
        request_mask = tree_mask[offset : offset + num_draft_tokens * width].view(
            num_draft_tokens, width
        )
        assert request_mask[:, :seq_len].all()
        torch.testing.assert_close(request_mask[:, seq_len:], causal)
        offset += num_draft_tokens * width
