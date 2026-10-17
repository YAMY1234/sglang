"""CUDA module gate (3): real graph stores, deferred join/fence, early reader refusal."""
import argparse
import glob
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dumps', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, factorize_layers, factorize_dense, ORTH_METHOD, ORTH_WARPS_OVERRIDE
    from sglang.srt.mem_cache.gdn_prefill_batch_graph import PrefillBatchGraph
    from sglang.srt.mem_cache.gdn_pd_publication import PDBatchPublication
    from sglang.srt.disaggregation.state_handoff import FactorStateHandoff, FactorTransferFence

    paths = sorted(glob.glob(args.dumps))
    if not paths or not torch.cuda.is_available():
        parser.error('requires P dump files and the target CUDA device')
    os.environ.update(SGLANG_GDN_K31_FUSED='1', SGLANG_GDN_K31_CHOLQR_MIXED='1')
    results = []
    for path in paths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        assert data['meta']['role'] == 'prefill'
        vb, omega = data['vbar'].cuda(), data['omega'].cuda()
        layers, heads, width = vb.shape
        with tempfile.TemporaryDirectory() as temporary:
            const = Path(temporary) / 'vbar.pt'
            torch.save({'vbar': {i: data['vbar'][i] for i in range(layers)}}, const)
            cfg = FactoredGDNConfig.parse(f'r=16,m=16,dtype=fp16,ring=16,init_method=k31,decode_method=iter,strict_chunk=1,factored_prefix=1,vbar={const}')

            def pool():
                shape = (layers, 4, heads)
                return NS(cfg=cfg, layer_ids=list(range(layers)), hv=heads, v=width, k=width,
                          vbar=vb, init_omega=lambda b: omega[:1].expand(b, -1, -1, -1),
                          a=torch.zeros(*shape, width, device='cuda'),
                          U=torch.zeros(*shape, cfg.rmax, width, dtype=cfg.dtype, device='cuda'),
                          W=torch.zeros(*shape, cfg.rmax, width, dtype=cfg.dtype, device='cuda'),
                          count=torch.zeros(*shape, dtype=torch.int32, device='cuda'),
                          stale=torch.zeros(4, dtype=torch.int32, device='cuda'),
                          dense_of=torch.full((4,), -1, dtype=torch.int32, device='cuda'),
                          dense_required=torch.zeros(4, dtype=torch.int32, device='cuda'),
                          prefix_valid=torch.zeros(4, dtype=torch.int32, device='cuda'),
                          dense_ring=torch.zeros(layers, 1, heads, width, width, device='cuda'))

            for tracked in (False, True):
                normal = data['S'][:1].cuda()
                t = data.get('S_tracked')
                track = (t[:1].cuda() if t is not None else normal.clone()) if tracked else None
                states = [(normal[:, i], None if track is None else track[:, i]) for i in range(layers)]
                controls = (torch.tensor([2], device='cuda') if tracked else None, None, None)
                plan = NS(slots=torch.tensor([1], device='cuda'), ring_dst=torch.tensor([0], device='cuda'),
                          dense_required_after_commit=torch.zeros(1, dtype=torch.int32, device='cuda'))
                ref, side = pool(), pool()
                graphs = [PrefillBatchGraph(include_tail=False), PrefillBatchGraph(include_tail=False)]
                policy = (ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense)
                for p, graph in zip((ref, side), graphs):
                    graph.run(p, plan, states, *controls, eager=factorize_layers, policy=policy)
                    graph.warmed = True  # exactly this fixed B1 / tracked signature is exercised
                torch.cuda.synchronize()
                fields = ('a', 'U', 'W', 'count', 'stale', 'dense_of', 'dense_required', 'prefix_valid', 'dense_ring')
                for name in fields:
                    getattr(ref, name).zero_()
                    getattr(side, name).zero_()
                ref.count.fill_(-999)
                side.count.fill_(-999)
                graphs[0].run(ref, plan, states, *controls, eager=factorize_layers, policy=policy)
                torch.cuda.synchronize()
                producer = torch.cuda.Event()
                producer.record()
                rejected = False
                try:
                    FactorTransferFence(None, side.count, torch.tensor([1], device='cuda'), cfg.r).wait(producer)
                except RuntimeError as error:
                    rejected = 'before final prefill' in str(error)
                assert rejected, 'early reader negative control did not reject unpublished count'

                publication = PDBatchPublication(side, offload_join=True)
                publication.forward_slots = frozenset((1, 2))
                side._pd_batch_publication = publication
                side.pside_join = publication.join
                publication.submit(graphs[1], plan, states, controls, eager=factorize_layers, policy=policy)
                publication.start_after_forward()
                producer = torch.cuda.Event()
                producer.record()
                fences = []
                sender = NS(set_state_handoff_fence=fences.append)
                req = NS(kv=NS(mamba_pool_idx=torch.tensor([1], device='cuda')),
                         _pfactor_agg_contract=True, factored_prefill_boundary_steps=0,
                         _pd_fulln_boundary_valid=torch.ones(4, dtype=torch.int32, device='cuda'),
                         disagg_kv_sender=sender)
                FactorStateHandoff(side).before_send(req)
                assert len(fences) == 1
                fences[0].wait(producer)
                publication.join()
                torch.cuda.synchronize()
                equal = {name: torch.equal(getattr(ref, name), getattr(side, name)) for name in fields}
                assert all(equal.values()), equal
                results.append(dict(dump=path, tracked=tracked, bitwise=equal,
                                    early_reader_rejected=rejected, fence_validated=fences[0].validated,
                                    publication_stats=publication.stats.copy()))
                print(json.dumps(results[-1]), flush=True)
                del graphs, ref, side, publication
    result = dict(passed=True, cases=results, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, scope='real PrefillBatchGraph + PDBatchPublication + FactorTransferFence; no RDMA')
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
