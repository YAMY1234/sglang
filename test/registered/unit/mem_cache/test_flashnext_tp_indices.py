"""TP coordinate identity, associated values, and unchanged-path regression."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / 'python/sglang/srt/mem_cache/flashnext_scheme_c.py'
spec = importlib.util.spec_from_file_location('scheme_c_tp_test', PATH)
codec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = codec
spec.loader.exec_module(codec)


def make(monkeypatch, enabled):
    monkeypatch.setenv('SGLANG_FLASHNEXT_LATENT_CANONICAL_TP', str(int(enabled)))
    cls = type('SmallCodec', (codec.FlashNextSchemeCCodec,), dict(WIDTH=128, RANK=32, SPIKES=16))
    result = cls(device='cpu')
    torch.manual_seed(630)
    for name, shape in [('E', (32, 128)), ('D', (128, 32)), ('mean', (128,))]:
        result.load(name, torch.randn(shape) * 0.03)
    return result


def group(monkeypatch, broadcast, size=2):
    fake = SimpleNamespace(world_size=size, broadcast=broadcast)
    monkeypatch.setitem(sys.modules, 'sglang.srt.distributed',
                        SimpleNamespace(get_tp_group=lambda: fake))


def equal_bits(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype
    assert torch.equal(x.contiguous().flatten().view(torch.uint8),
                       y.contiguous().flatten().view(torch.uint8))


@pytest.mark.parametrize('n', [0, 1, 17])
def test_identical_rank_selection_preserves_every_payload_and_output(monkeypatch, n):
    off, on = make(monkeypatch, False), make(monkeypatch, True)
    calls = []
    def broadcast(x, src):
        assert src == 0 and x.dtype == torch.int32
        calls.append(x.clone())
        return x
    group(monkeypatch, broadcast)
    h = torch.randn(n, 128).bfloat16()
    base = torch.randn_like(h)
    pos = torch.arange(n)
    before, expected = off.encode_and_decode(h, pos, base)
    after, actual = on.encode_and_decode(h, pos, base)
    for name in vars(before):
        equal_bits(getattr(before, name), getattr(after, name))
    equal_bits(expected, actual)
    equal_bits(actual, on.decode(after, base))
    assert len(calls) == bool(n)
    assert on.tp_index_statistics()['different_coordinate_rows'] == 0


def test_rank_zero_coordinates_gather_local_corresponding_values(monkeypatch):
    on = make(monkeypatch, True)
    canonical = torch.arange(16).reshape(1, -1).int()
    def broadcast(x, src):
        x.copy_(canonical)
        return x
    group(monkeypatch, broadcast)
    h = torch.randn(1, 128).bfloat16()
    base = torch.randn_like(h)
    pos = torch.tensor([8192])
    batch, actual = on.encode_and_decode(h, pos, base)
    residual = h.float() - base.float()
    rms = (residual.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    reconstructed = on.mean + on.project(codec.unpack_nvfp4(batch.z, batch.z_block_scale, batch.z_scale), 'D')
    values = (residual / rms - reconstructed).gather(-1, canonical.long()).bfloat16()
    equal_bits(batch.spike_values, values)
    equal_bits(codec.unpack_gap8(batch.spike_indices, batch.spike_lengths, sparse=16, width=128), canonical.long())
    equal_bits(actual, on.decode(batch, base))
    assert on.tp_index_statistics() == dict(enabled=True, calls=1, rows=1, different_coordinate_rows=1)


def test_flag_off_never_calls_collective(monkeypatch):
    off = make(monkeypatch, False)
    def fail(*args, **kwargs):
        raise AssertionError('flag-off collective')
    group(monkeypatch, fail)
    indices = torch.tensor([[9, 2]])
    assert off.align_tp_indices(indices) is indices


def test_tp1_preserves_order_and_skips_collective(monkeypatch):
    on = make(monkeypatch, True)
    def fail(*args, **kwargs):
        raise AssertionError('TP1 collective')
    group(monkeypatch, fail, size=1)
    indices = torch.tensor([[9, 2]])
    assert on.align_tp_indices(indices) is indices


def test_mismatch_counts_sets_not_topk_order(monkeypatch):
    on = make(monkeypatch, True)
    def broadcast(x, src):
        x.copy_(torch.tensor([[1, 3, 5], [1, 3, 5]], dtype=torch.int32))
        return x
    group(monkeypatch, broadcast)
    result = on.align_tp_indices(torch.tensor([[5, 1, 3], [1, 3, 6]]))
    assert result.tolist() == [[1, 3, 5], [1, 3, 5]]
    assert on.tp_index_statistics()['different_coordinate_rows'] == 1
