from unittest.mock import patch

import torch

from sglang.srt.managers.utils import GenerationBatchResult


class _Event:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True


def test_stage_keeps_device_fields_until_commit():
    token_ids = torch.tensor([11, 12, 13])
    accept_lens = torch.tensor([3])
    result = GenerationBatchResult(
        next_token_ids=token_ids,
        accept_lens=accept_lens,
        copy_done=_Event(),
    )

    assert result.can_stage_early_cpu_result(False, False)
    with patch(
        "sglang.srt.managers.utils._async_d2h", side_effect=lambda value: value.clone()
    ):
        result.stage_early_cpu_result()

    token_ids.fill_(-1)
    accept_lens.fill_(-1)
    assert result.copy_done.recorded
    assert result.next_token_ids.tolist() == [-1, -1, -1]
    assert result.accept_lens.tolist() == [-1]

    result.commit_early_cpu_result()
    assert result.next_token_ids.tolist() == [11, 12, 13]
    assert result.accept_lens.tolist() == [3]
    assert not result.has_staged_early_cpu_result


def test_optional_payloads_keep_the_late_copy_path():
    result = GenerationBatchResult(next_token_ids=torch.tensor([1]))
    assert not result.can_stage_early_cpu_result(True, False)
    assert not result.can_stage_early_cpu_result(False, True)
