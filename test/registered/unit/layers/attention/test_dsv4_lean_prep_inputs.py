"""SGLANG_OPT_DSV41_LEAN_PREP_INPUTS on the DSv4 backend: the per-step extend-length
fill / upload is skipped only while the destination still holds the same values,
and never counts when recorded under graph capture."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_fill = DeepseekV4AttnBackend._fill_extend_seq_lens
_upload = DeepseekV4AttnBackend._upload_extend_lens


@pytest.fixture(autouse=True)
def not_capturing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _backend(lean: bool):
    backend = SimpleNamespace(
        extend_seq_lens_buffer=torch.zeros(8, dtype=torch.int32),
        _lean_prep=lean,
        _extend_fill_record=None,
        _extend_upload_record=None,
        uploads=0,
    )

    def move(x):
        backend.uploads += 1
        return torch.tensor(x, dtype=torch.int32)

    backend._move_to_device = move
    return backend


def test_fill_skipped_only_while_unchanged():
    b = _backend(lean=True)
    assert _fill(b, 2, 6).tolist() == [6, 6]
    b.extend_seq_lens_buffer[:2] = -1  # a skipped fill leaves this visible
    assert _fill(b, 2, 6).tolist() == [-1, -1]
    # Another batch size or value rewrites.
    assert _fill(b, 3, 6).tolist() == [6, 6, 6]
    assert _fill(b, 3, 5).tolist() == [5, 5, 5]
    # Another writer (the ragged verify copy) drops the record.
    b._extend_fill_record = None
    b.extend_seq_lens_buffer[:3] = 0
    assert _fill(b, 3, 5).tolist() == [5, 5, 5]


def test_fill_flag_off_always_writes():
    b = _backend(lean=False)
    _fill(b, 2, 6)
    b.extend_seq_lens_buffer[:2] = -1
    assert _fill(b, 2, 6).tolist() == [6, 6]


def test_fill_under_capture_never_counts(monkeypatch):
    b = _backend(lean=True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _fill(b, 2, 6)
    assert b._extend_fill_record is None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    b.extend_seq_lens_buffer[:2] = -1
    assert _fill(b, 2, 6).tolist() == [6, 6]


@pytest.mark.parametrize("lean, expected_uploads", [(True, 2), (False, 3)])
def test_upload_reused_while_host_list_unchanged(lean, expected_uploads):
    b = _backend(lean=lean)
    first = _upload(b, [5, 5])
    again = _upload(b, [5, 5])
    assert (again is first) == lean
    assert _upload(b, [5, 5, 5]).tolist() == [5, 5, 5]
    assert b.uploads == expected_uploads
