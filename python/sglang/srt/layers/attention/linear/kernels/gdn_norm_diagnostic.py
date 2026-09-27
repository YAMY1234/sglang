"""Opt-in full-model shadow comparison; never used for admission or timing."""
import json
from pathlib import Path

import torch

_records = {}
_saved = set()


def before(layer, mixed, a, b, kwargs):
    from .gdn_factored import factored_packed_decode
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

    assert kwargs['norm_context'] is not None
    assert kwargs['conv_context'] is None and kwargs['async_stream'] is None
    indices = kwargs['ssm_state_indices']
    assert indices.numel() == 1
    gather = indices.clamp_min(0).long()
    shadow = dict(kwargs)
    captured = {}
    for name in ('fa', 'fu', 'fw', 'fcount', 'stale', 'prefix_valid'):
        value = kwargs.get(name)
        if value is not None:
            captured[name] = value.index_select(0, gather)
            shadow[name] = captured[name].clone()
    shadow['ssm_state_indices'] = torch.where(indices < 0, -1, 0)
    shadow['norm_context'] = None
    raw = factored_packed_decode(mixed, a, b, **shadow)
    z, weight, eps, rows, activation = kwargs['norm_context']
    reference = rms_norm_gated(x=raw.reshape(-1, raw.shape[-1]), weight=weight,
        bias=None, z=z, eps=eps, norm_before_gate=True, is_rms_norm=True,
        activation=activation).reshape_as(raw)
    record = dict(layer=layer, gather=gather, captured=captured, shadow=shadow,
        mixed=mixed.clone(), a=a.clone(), b=b.clone(), raw=raw,
        reference=reference, z=z.clone(), weight=weight.clone(), eps=eps,
        rows=rows, activation=activation)
    _records[layer] = record
    return record


def after(record, output, kwargs):
    record['actual'] = output.clone()
    expected = [record['reference']]
    actual = [record['actual']]
    record['actual_state'] = {}
    for name in record['captured']:
        value = kwargs[name].index_select(0, record['gather'])
        record['actual_state'][name] = value
        expected.append(record['shadow'][name])
        actual.append(value)
    record['names'] = ['output'] + list(record['captured'])
    record['differences'] = torch.stack([
        (x.contiguous().view(torch.uint8) != y.contiguous().view(torch.uint8)).sum()
        for x, y in zip(expected, actual)])


def save_step(directory, window, step):
    """Called outside CUDA capture, after each forced full-model forward."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer, record in sorted(_records.items()):
        counts = record['differences'].tolist()
        row = dict(rank=rank, layer=layer, window=window, step=step,
                   differing_bytes=dict(zip(record['names'], counts)))
        if any(counts) and layer not in _saved:
            _saved.add(layer)
            def cpu(value):
                if isinstance(value, torch.Tensor): return value.detach().cpu()
                if isinstance(value, dict): return {k: cpu(v) for k,v in value.items()}
                if isinstance(value, tuple): return tuple(cpu(v) for v in value)
                return value
            filename = f'rank{rank}-layer{layer}-window{window}-step{step}.pt'
            torch.save(cpu(record), root / filename)
            row['counterexample'] = filename
            print('SSMOFF_NORM_COUNTEREXAMPLE ' + json.dumps(row), flush=True)
        rows.append(row)
    with (root / f'rank{rank}.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(window=window, step=step, layers=rows)) + '\n')
