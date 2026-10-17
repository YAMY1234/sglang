"""Real P31 TP2: old/full-N raw checkpoint bytes and production radix reachability.

Not a performance run. Capture the exact dense checkpoint entering the factor
publication, insert with MambaRadixCache.cache_finished_req, then match the
scheduler's prompt-minus-one key. No manually supplied prefix indices count as
evidence for this gate.
"""
import json
import os
from pathlib import Path


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
    # P-1 aligned, P aligned, and a non-final chunk. Larger cases also exercise
    # more than one GDN chunk, with the exact same P input across routes.
    cases = ((65, True), (128, True), (129, True), (4096, True),
             (4097, True), (128, False))
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
            w._run(fb)
            w.join()
            torch.cuda.synchronize()
            depth = req.kv.mamba_last_track_seqlen
            keep = rp.get_mamba_ping_pong_keep_idx(req)
            slot = int(req.kv.mamba_ping_pong_track_buffer[keep])
            assert slot in captured, (route, length, slot, captured.keys())
            raw = captured[slot]
            factors = {name: getattr(w.pool, name)[:, slot].cpu().clone()
                       for name in ('a', 'U', 'W', 'count')}
            tree.cache_finished_req(req, kv_len_to_handle=length)
            rp.free(req)
            match = tree.match_prefix(MatchPrefixParams(key=RadixKey(ids[:-1], req.extra_key)))
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
            # Fusion/mixed may alter U/W at the registered rounding level;
            # the raw donated S checkpoint and a/count must remain exact.
            a_equal = torch.equal(factors['a'], reference[1]['a'])
            assert raw_equal and counts_equal and a_equal
            row = dict(rank=rank, length=length, final=final, route=route,
                       cache_depth=depth, hit=hit, expected=expected,
                       old_checkpoint_bitwise=raw_equal, a_bitwise=a_equal,
                       count_bitwise=counts_equal, cache_store_bitwise=stored_equal,
                       shallow_encoding=fb.mamba_track_seqlens.cpu().tolist(),
                       deep_encoding=getattr(fb, '_pd_fulln_emitter_track_seqlens',
                                             fb.mamba_track_seqlens).cpu().tolist())
            results.append(row)
            print('P3_CHECKPOINT_GATE', json.dumps(row), flush=True)
    out = Path(os.environ['P3_OUT']) / f'checkpoint-r{rank}.json'
    out.write_text(json.dumps(dict(passed=True, cases=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
