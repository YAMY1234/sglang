"""Opt-in, unscored I4 causal diagnostics. Never enabled by qualification runs."""

import contextlib
import json
from collections import defaultdict

import torch


def delta(x, y):
    x, y = x.double(), y.double()
    return dict(rel_f=float((x-y).norm()/y.norm().clamp_min(1e-30)),
                max_abs=float((x-y).abs().max()), equal=bool(torch.equal(x, y)))


@contextlib.contextmanager
def instrument(runtime, fb, config, root):
    """Replace only this uncached prefill; subsequent decode stays production."""
    body_forward = runtime.body.forward
    decode = runtime._decode_forward
    batched = runtime.batched_extend
    mode = config['mode']
    ablation_saved = []

    def serial_tail(ids, positions, batch):
        from sglang.srt.model_executor.forward_context import get_attn_backend
        outputs = []
        for row, length in enumerate(batch.seq_lens_cpu.tolist()):
            one = runtime.slice_batch(batch, row, row, row+1, length, decode=True)
            get_attn_backend().init_forward_metadata(one)
            outputs.append(decode(one.input_ids, one.positions, one))
        return torch.cat(outputs)

    def reference_prefix(ids, positions, batch, *a, **kw):
        from sglang.srt.model_executor.forward_context import get_attn_backend
        runtime.batched_extend = False
        runtime._decode_forward = lambda ids, pos, fb: runtime.body.embed_tokens.weight.new_zeros(
            (len(ids), runtime.body.config.hidden_size))
        try:
            out = body_forward(ids, positions, batch, *a, **kw)
        finally:
            runtime._decode_forward = decode
            runtime.batched_extend = batched
        rows, indices, offset = [], [], 0
        for row, n in enumerate(batch.extend_seq_lens_cpu):
            if n > 1:
                rows.append(row)
                indices.append(offset+n-1)
            offset += n
        tail = runtime.packed_batch(batch, rows, indices,
            [batch.extend_seq_lens_cpu[r] for r in rows], decode=True)
        get_attn_backend().init_forward_metadata(tail)
        out[indices] = decode(tail.input_ids, tail.positions, tail)
        return out

    if mode == 'serial_tail':
        runtime._decode_forward = serial_tail
    elif mode == 'reference_prefix':
        runtime.body.forward = reference_prefix
    elif mode == 'current':
        runtime.batched_extend = False
    elif mode in ('serial_factor', 'reference_emitters', 'reference_emitters_serial_factor'):
        pool = runtime.pool
        if 'serial_factor' in mode:
            prune = pool._prune

            def serial_factor(layers, slots, state, **kw):
                for row in range(len(layers)):
                    prune(layers[row:row+1], slots[row:row+1], state[row:row+1], **kw)
            ablation_saved.append((pool, '_prune', prune))
            pool._prune = serial_factor
        if 'reference_emitters' in mode:
            import sglang.srt.layers.attention.linear.mamba2_emitter_extend as emit
            original_emitter = emit.emitter_extend

            def reference_emitter(components, lid, residual, metadata, pool, plan):
                outputs, offset = [], 0
                for row, length in enumerate(metadata.mixed_metadata.extend_seq_lens_cpu):
                    out = components.emitter(lid, residual[offset:offset+length])
                    outputs.append(out)
                    if 'conv' in out:
                        pool.conv[pool.layer_map[lid], plan.slots[row]].copy_(out['conv'])
                    offset += length
                if lid in components.attention_emitters:
                    return {key: torch.cat([out[key] for out in outputs]) for key in ('k', 'v')}
                pool.commit_extend(lid, plan, torch.stack([out['state'] for out in outputs]))
                return None
            ablation_saved.append((emit, 'emitter_extend', original_emitter))
            emit.emitter_extend = reference_emitter
    elif mode == 'shadow':
        import sglang.srt.models.lightning_duet.engine as engine
        pool = runtime.pool
        # The runner enters forward_context inside original.forward. Warmup has
        # already constructed the pool; do not access backend globals here.
        assert pool is not None, 'run normal warmup before the shadow probe'
        snapshots = defaultdict(list)
        candidates = defaultdict(list)
        report = dict(tag=config['tag'], lengths=fb.extend_seq_lens_cpu, comparisons=[])
        phase = ['reference']
        pending = {}
        saved = []

        def watch(key, value):
            value = value.detach().cpu().clone()
            if phase[0] == 'reference':
                snapshots[key].append(value)
            else:
                candidates[key].append(value)

        def patch(obj, key, value):
            saved.append((obj, key, getattr(obj, key)))
            setattr(obj, key, value)

        for lid, layer in enumerate(runtime.body.layers):
            original = layer.forward

            def layer_forward(*a, _original=original, _lid=lid, **kw):
                out = _original(*a, **kw)
                batch = kw.get('forward_batch')
                kind = 'tail' if batch.forward_mode.is_decode() else 'shallow'
                h, r = out
                watch(f'{kind}/layer{_lid}', h if r is None else h+r)
                return out
            patch(layer, 'forward', layer_forward)
        boundary = engine.reconstruct_boundary

        def reconstruct(*a, **kw):
            record, restored = boundary(*a, **kw)
            watch('boundary/input', a[1])
            watch('boundary/restored', restored)
            return record, restored
        patch(engine, 'reconstruct_boundary', reconstruct)
        initialize = pool.initialize

        def init(lid, slot, state, conv):
            watch(f'raw/layer{lid}', state[None])
            watch(f'conv/layer{lid}', conv[None])
            return initialize(lid, slot, state, conv)
        patch(pool, 'initialize', init)
        commit = pool.commit_extend

        def collect(lid, plan, state, *, rows=None):
            rows = tuple(range(plan.slots.numel())) if rows is None else tuple(rows)
            slots = plan.slots[list(rows)]
            watch(f'raw/layer{lid}', state)
            watch(f'conv/layer{lid}', pool.conv[pool.layer_map[lid], slots])
            for slot, row in zip(slots.tolist(), state.detach().cpu()):
                pending[pool.layer_map[lid], slot] = row.clone()
            return commit(lid, plan, state, rows=rows)
        patch(pool, 'commit_extend', collect)
        prune = pool._prune

        def factor(layers, slots, states, **kw):
            if phase[0] == 'candidate':
                expected = torch.stack([pending[i, s] for i, s in zip(layers.tolist(), slots.tolist())])
                report['comparisons'].append(dict(stage='pending_lifetime',
                    **delta(states.detach().cpu(), expected)))
            result = prune(layers, slots, states, **kw)
            # Materialized factors avoid treating QR sign choices as state errors.
            material = (pool.directions[layers, :, :, None] * pool.coeff[layers, slots, :, None, :]
                        + pool.u[layers, slots, :, :pool.rank].transpose(-1, -2)
                        @ pool.v[layers, slots, :, :pool.rank])
            for layer in dict.fromkeys(layers.tolist()):
                lid = pool.layer_ids[layer]
                watch(f'factor/layer{lid}', material[layers == layer])
            return result
        patch(pool, '_prune', factor)

        def shadow(*a, **kw):
            runtime.batched_extend = False
            body_forward(*a, **kw)
            runtime.batched_extend = True
            phase[0] = 'candidate'
            result = body_forward(*a, **kw)
            for key, values in candidates.items():
                report['comparisons'].append(dict(stage=key,
                    **delta(torch.cat(values), torch.cat(snapshots[key]))))
            (root / (config['tag']+'-stages.json')).write_text(json.dumps(report, indent=2)+'\n')
            return result
        runtime.body.forward = shadow
    elif mode != 'batched':
        raise ValueError(mode)
    try:
        yield
    finally:
        runtime.body.forward = body_forward
        runtime._decode_forward = decode
        runtime.batched_extend = batched
        for obj, key, original in reversed(ablation_saved):
            setattr(obj, key, original)
        if mode == 'shadow':
            for obj, key, original in reversed(saved):
                setattr(obj, key, original)
