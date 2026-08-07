from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _DraftExtendMode:
    @staticmethod
    def is_extend():
        return False

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


def test_metadata_keeps_eager_gathered_draft_extend_rows():
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
        spec_info=EagleDraftInput(),
    )

    metadata = LogitsMetadata.from_forward_batch(batch)

    assert metadata.draft_extend_select_index is None


def test_eager_draft_extend_captures_selected_hidden_rows():
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        input_ids=torch.arange(8),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8, 9], dtype=torch.int32),
        seq_lens_sum=17,
    )
    spec_algorithm = Mock()
    spec_algorithm.is_standalone.return_value = False
    attn_backend = Mock()
    draft_model_runner = SimpleNamespace(
        spec_algorithm=spec_algorithm,
        attn_backend=attn_backend,
    )
    draft_input = EagleDraftInput()
    forward_batch = SimpleNamespace()

    with patch(
        "sglang.srt.speculative.eagle_info_v2.ForwardBatch.init_new",
        return_value=forward_batch,
    ):
        actual = draft_input.prepare_for_extend_to_fill_draft_kvcache(
            batch=batch,
            predict=torch.arange(8),
            num_draft_tokens=4,
            draft_model_runner=draft_model_runner,
            cuda_graph_runner=None,
        )

    assert actual is forward_batch
    assert batch.capture_hidden_mode == CaptureHiddenMode.LAST
    attn_backend.init_forward_metadata.assert_called_once_with(forward_batch)
