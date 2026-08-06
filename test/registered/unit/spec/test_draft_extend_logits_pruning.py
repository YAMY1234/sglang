from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _DraftExtendMode:
    @staticmethod
    def is_decode_or_idle():
        return False

    @staticmethod
    def is_target_verify():
        return False

    @staticmethod
    def is_draft_extend_v2():
        return True


def test_draft_extend_prunes_logits_rows():
    hidden_states = torch.arange(32).view(8, 4)
    metadata = LogitsMetadata(
        forward_mode=_DraftExtendMode(),
        draft_extend_select_index=torch.tensor([2, 7]),
    )

    pruned_states = LogitsProcessor._get_pruned_states(
        None, hidden_states, None, None, metadata
    )[0]

    torch.testing.assert_close(pruned_states, hidden_states[[2, 7]])


def test_draft_extend_without_selection_keeps_all_rows():
    hidden_states = torch.arange(32).view(8, 4)
    metadata = LogitsMetadata(forward_mode=_DraftExtendMode())

    pruned_states = LogitsProcessor._get_pruned_states(
        None, hidden_states, None, None, metadata
    )[0]

    assert pruned_states.data_ptr() == hidden_states.data_ptr()


def test_metadata_propagates_draft_extend_selection():
    select_index = torch.tensor([1, 6])
    batch = SimpleNamespace(
        forward_mode=_DraftExtendMode(),
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_logprob_start_lens_cpu=None,
        extend_input_logprob_token_ids_gpu=None,
        capture_hidden_mode=None,
        next_token_logits_buffer=None,
        padded_static_len=-1,
        is_prefill_only=False,
        global_num_tokens_gpu=None,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_dp_buffer_len=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_tokens_for_logprob_gpu=None,
        mm_input_embeds=None,
        spec_info=SimpleNamespace(select_index=select_index),
    )

    metadata = LogitsMetadata.from_forward_batch(batch)

    assert metadata.draft_extend_select_index is select_index
