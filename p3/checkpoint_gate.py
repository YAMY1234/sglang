"""Real P31 TP2: old/full-N raw checkpoint bytes and production radix reachability.

Not a performance run. Capture the exact dense checkpoint entering the factor
publication, insert with MambaRadixCache.cache_finished_req, then match the
scheduler's prompt-minus-one key. No manually supplied prefix indices count as
evidence for this gate.
"""
import json
import os
from pathlib import Path


def factor_diagnostic(raw, old, actual, pool):
    """Same checkpoint/rank, stored factors; never mutate service route flags.

    Content and fp64-control residual use module 1's stored reconstruction.
    The lead's 22:21 cache ruling requires every residual ratio <= 1.01.
    Cross-route a equality is reported, never made a bitwise admission gate.
    """
    import torch
    from sglang.srt.layers.attention.linear.kernels import gdn_prefill_reference as ref

    device = pool.vbar.device
    dense = raw.to(device).float()
    layers, heads, width, key = dense.shape
    vb = pool.vbar.float()
    omega = pool.k31_omega
    if omega.ndim == 3:
        omega = omega.unsqueeze(0)
    directions = omega[:, None].expand(1, layers, heads, width, omega.shape[-1])
    control = ref.factorize_prefill_k31(
        dense.reshape(1, layers * heads, width, key), vb.reshape(layers * heads, width),
        pool.cfg.r, pool.cfg.rmax, pool.cfg.dtype,
        directions.reshape(1, layers * heads, width, omega.shape[-1]),
        mixed_eigh=False, mixed_cholqr=False)

    def reconstruct(factors):
        a, u, w = [factors[name].to(device).float() for name in ('a', 'U', 'W')]
        return vb[..., None] * a[..., None, :] + torch.einsum('lhrv,lhrk->lhvk', w, u)

    control_factors = {name: value.reshape(layers, heads, *value.shape[2:])
                       for name, value in zip(('a', 'U', 'W'), control)}
    before, after, full64 = [reconstruct(f) for f in (old, actual, control_factors)]
    norm = dense.flatten(2).norm(dim=-1).clamp_min(1e-30)
    content = (after - before).flatten(2).norm(dim=-1) / norm
    res64 = (dense - full64).flatten(2).norm(dim=-1).clamp_min(1e-30)
    ratio = (dense - after).flatten(2).norm(dim=-1) / res64
    old_ratio = (dense - before).flatten(2).norm(dim=-1) / res64
    da = (actual['a'] - old['a']).to(device).float()
    sink = (vb[..., None] * da[..., None, :]).flatten(2).norm(dim=-1) / norm
    nonfinite = sum(int((~torch.isfinite(value)).sum())
                    for name in ('a', 'U', 'W') for value in (old[name], actual[name]))
    nonfinite += int((~torch.isfinite(dense)).sum())
    row = dict(a_max_abs=float(da.abs().max()), a_sink_content_max=float(sink.max()),
               content_median=float(content.median()), content_max=float(content.max()),
               residual_ratio_max=float(ratio.max()),
               residual_fraction_le_1_01=float((ratio <= 1.01).double().mean()),
               old_residual_ratio_max=float(old_ratio.max()), nonfinite=nonfinite,
               reference='module-1 fp64 CholQR/eigh control with production pool storage dtype',
               matrices=layers * heads)
    row['gate'] = dict(content=row['content_median'] <= 3e-5 and row['content_max'] <= 2.5e-4,
                       a_sink_content=row['a_sink_content_max'] <= 2.5e-4,
                       residual=row['residual_ratio_max'] <= 1.01,
                       finite=nonfinite == 0 and bool(torch.isfinite(content).all())
                              and bool(torch.isfinite(ratio).all()))
    row['passed'] = all(row['gate'].values())
    return row


def main():
    import torch
    from p3_factory import build, token_ids
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.gdn_prefill_agg_contract import set_pd_micro_route
    from sglang.srt.mem_cache.gdn_prefill_batch_graph import PrefillBatchGraph
    from sglang.srt.runtime_context import get_schedule, get_exec, mamba_checkpoint_grid

    torch.set_grad_enabled(False)
    w = build()
    rank = int(os.environ['RANK'])
    if os.environ.get('P3_CHECKPOINT_DEBUG') == '1':
        # Diagnostic-only barriers locate the first failing layer; this mode
        # is never a timing or complete cache-admission result.
        def after_layer(label):
            def hook(*unused):
                torch.cuda.synchronize()
                print('P3_LAYER_DONE', rank, label, flush=True)
            return hook
        for layer_id in w.r.model.p_layer_ids:
            w.r.model.model.model.layers[layer_id].register_forward_hook(after_layer(layer_id))
        for layer_id in w.r.model.emitter_ids:
            emitter = w.r.model.emitters[str(layer_id)]
            emit = emitter.emit
            def traced(*a, _emit=emit, _id=layer_id, **k):
                print('P3_EMITTER_BEGIN', rank, _id, flush=True)
                if _id == 32:
                    from sglang.srt.model_executor.forward_context import get_attn_backend
                    meta = get_attn_backend().linear_attn_backend.forward_metadata
                    for name in ('query_start_loc', 'mamba_cache_indices', 'track_ssm_h_src',
                                 'track_ssm_h_dst', 'track_ssm_final_src', 'track_ssm_final_dst'):
                        value = getattr(meta, name, None)
                        print('P3_DEEP_METADATA', rank, name,
                              None if value is None else value.cpu().tolist(), flush=True)
                    print('P3_DEEP_BATCH', rank, a[1].mamba_track_seqlens.cpu().tolist(),
                          a[1].extend_seq_lens.cpu().tolist(), flush=True)
                value = _emit(*a, **k)
                torch.cuda.synchronize()
                print('P3_EMITTER_DONE', rank, _id, flush=True)
                return value
            emitter.emit = traced
    assert get_exec().mamba.enable_mamba_extra_buffer
    captured = {}
    original = PrefillBatchGraph.run

    def capture(graph, pool, plan, states, track_slots, final_src, final_dst, **kwargs):
        # Copy on the producer stream before static publication inputs can be
        # rebound. This gate intentionally synchronizes, outside any timing.
        for slot, row in ([] if track_slots is None else
                          zip(track_slots.tolist(), range(track_slots.numel()))):
            captured[int(slot)] = torch.stack([pair[1][row] for pair in states]).cpu()
        if final_src is not None:
            for src, dst in zip(final_src.tolist(), final_dst.tolist()):
                row = plan.slots.tolist().index(src)
                captured[int(dst)] = torch.stack([pair[0][row] for pair in states]).cpu()
        return original(graph, pool, plan, states, track_slots, final_src, final_dst, **kwargs)

    PrefillBatchGraph.run = capture
    results, trees = [], []
    out = Path(os.environ['P3_OUT']) / f'checkpoint-r{rank}.json'
    # P-1 aligned, P aligned, and a non-final chunk. Larger cases also exercise
    # more than one GDN chunk, with the exact same P input across routes.
    cases = ((65, True), (128, True), (129, True), (4096, True),
             (4097, True), (128, False))
    if os.environ.get('P3_CHECKPOINT_DEBUG') == '1':
        cases = ((65, True),)
    for length, final in cases:
        reference = None
        for route in ('0n', '0', '1'):
            set_pd_micro_route(route)
            captured.clear()
            rp, alloc = w.r.req_to_token_pool, w.r.token_to_kv_pool_allocator
            tree = MambaRadixCache(CacheInitParams(
                disable=False, req_to_token_pool=rp, token_to_kv_pool_allocator=alloc,
                page_size=get_schedule().page_size, enable_mamba_extra_buffer=True,
                enable_mamba_extra_buffer_lazy=get_exec().mamba.enable_mamba_extra_buffer_lazy))
            trees.append(tree)  # retain ownership of all cached slots until exit
            ids = token_ids(length + (0 if final else 64), 7000 + length, w.vocab)
            req = w._req(ids)
            req.set_extend_range(0, length)
            req.last_node = tree.root_node
            tree.inc_lock_ref(req.last_node)
            fb = w._forward_batch(req, tree=tree)
            before = w.counters()
            w._run(fb)
            w.join()
            torch.cuda.synchronize()
            after = w.counters()
            assert after['fallbacks'] == before['fallbacks'], (before, after)
            depth = req.kv.mamba_last_track_seqlen
            keep = rp.get_mamba_ping_pong_keep_idx(req)
            slot = int(req.kv.mamba_ping_pong_track_buffer[keep])
            assert slot in captured, (route, length, slot, captured.keys())
            raw = captured[slot]
            factors = {name: getattr(w.pool, name)[:, slot].cpu().clone()
                       for name in ('a', 'U', 'W', 'count')}
            tree.cache_finished_req(req, kv_len_to_handle=length)
            rp.free(req)
            # Native requests carry array('q'); RadixKey.match requires the
            # same sequence type as the key stored by cache_finished_req.
            match = tree.match_prefix(MatchPrefixParams(key=RadixKey(
                req.origin_input_ids[:-1], req.extra_key, cache_salt=req.cache_salt)))
            hit = len(match.device_indices)
            grid = mamba_checkpoint_grid(tree.page_size)
            expected = ((length - int(final)) // grid) * grid
            assert depth == expected and hit == expected, (route, length, depth, hit, expected)
            cached_slot = int(match.last_device_node.mamba_value.item())
            stored_equal = {name: torch.equal(value, getattr(w.pool, name)[:, cached_slot].cpu())
                            for name, value in factors.items()}
            assert all(stored_equal.values()), stored_equal
            if reference is None:
                reference = (raw, factors, hit)
            raw_equal = torch.equal(raw, reference[0])
            counts_equal = torch.equal(factors['count'], reference[1]['count'])
            # Across grouping routes, a/U/W are rounding-level diagnostics.
            # Raw checkpoint, count and cached-store bytes remain strict gates.
            a_equal = torch.equal(factors['a'], reference[1]['a'])
            numerical = factor_diagnostic(raw, reference[1], factors, w.pool)
            deep_encoding = getattr(fb, '_pd_fulln_emitter_track_seqlens', None)
            if deep_encoding is None:
                deep_encoding = fb.mamba_track_seqlens
            row = dict(rank=rank, length=length, final=final, route=route,
                       cache_depth=depth, hit=hit, expected=expected,
                       old_checkpoint_bitwise=raw_equal, a_bitwise=a_equal,
                       count_bitwise=counts_equal, cache_store_bitwise=stored_equal,
                       factor_diagnostic=numerical,
                       fallback_delta=after['fallbacks'] - before['fallbacks'],
                       shallow_encoding=fb.mamba_track_seqlens.cpu().tolist(),
                       deep_encoding=deep_encoding.cpu().tolist())
            if not raw_equal:
                row['raw_difference'] = dict(
                    max=float((raw.float() - reference[0].float()).abs().max()),
                    layers=[layer for layer, (actual, old) in enumerate(zip(raw, reference[0]))
                            if not torch.equal(actual, old)])
            results.append(row)
            print('P3_CHECKPOINT_GATE', json.dumps(row), flush=True)
            assert raw_equal and counts_equal, row
            # Persist completed rows even if a later forward fails. Numerical
            # failure is collected across all cases, separately for each route.
            out.write_text(json.dumps(dict(complete=False, cases=results), indent=2) + '\n')
    per_route = {route: all(row['factor_diagnostic']['passed'] for row in results if row['route'] == route)
                 for route in ('0n', '0', '1')}
    summary = dict(complete=True, required_bitwise_pass=True,
                   factor_bounds_pass_by_route=per_route,
                   routing_only_pass=per_route['0n'] and per_route['0'],
                   combined_pass=per_route['0n'] and per_route['1'],
                   passed=all(per_route.values()), cases=results,
                   diagnostic_only=os.environ.get('P3_CHECKPOINT_DEBUG') == '1')
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print('P3_CHECKPOINT_SUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'cases'}), flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
