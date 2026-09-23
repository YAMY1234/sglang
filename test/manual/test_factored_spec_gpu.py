"""Production r8/W8 kernel and eager/graph transaction guard, no model weights.

Run in the pinned serving container. This is a state guard, not an end-to-end
token parity claim. No acceptance simulation is installed in a serving engine.
"""
import json
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels.gdn_factored import (
    factored_expiry_truncate_layers,
    factored_packed_decode,
)
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool


def same(a, b, context):
    if not torch.equal(a, b):
        raise AssertionError(f"{context}: {(a != b).sum().item()} unequal elements")


def run():
    torch.manual_seed(427)
    device = "cuda"
    layers, batch, padded, hv, h, k, v = 2, 2, 3, 24, 8, 128, 128
    cfg = FactoredGDNConfig(r=8, m=8, ring=2, strict_chunk=1, factored_prefix=1)
    pool = FactoredGDNPool(size=8, cache_params=SimpleNamespace(shape=SimpleNamespace(temporal=(hv,v,k))),
                          mamba_layer_ids=[0, 1], device=device, cfg=cfg,
                          spec_max_batch_size=padded, speculative_num_draft_tokens=4)
    for tensor in (pool.a, pool.U, pool.W, pool.vbar):
        tensor.normal_(0, 0.03)
    # Counts 14 and 15 force cuts at input 1 and 0, strictly inside verify.
    pool.count[:, 2].fill_(14)
    pool.count[:, 5].fill_(15)
    slots = torch.tensor([2, 5], device=device)
    spec = pool.spec_state
    backend = GDNAttnBackend.__new__(GDNAttnBackend)
    backend.factored, backend.topk = pool, 1
    descriptors = [SimpleNamespace(layer_id=li, num_q_heads=h, num_v_heads=hv,
                       head_k_dim=k, head_v_dim=v,
                       A_log=torch.randn(hv, device=device), dt_bias=torch.randn(hv, device=device))
                   for li in range(layers)]
    mixed = torch.randn(layers, padded, 4, 2*h*k+hv*v, dtype=torch.bfloat16, device=device)
    aa = torch.randn(layers, padded, 4, hv, dtype=torch.bfloat16, device=device)
    bb = torch.randn_like(aa)

    def verify():
        return [backend._forward_verify_factored(desc, mixed[li].flatten(0,1),
                    aa[li].flatten(0,1), bb[li].flatten(0,1)).view(padded,4,hv,v)
                for li, desc in enumerate(descriptors)]

    # Compile first, then capture only the body. Real snapshots remain outside.
    warm = spec.snapshot_commit(slots)
    verify()
    spec.rollback(warm)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = verify()
    torch.cuda.synchronize()
    evidence = []
    for graph_mode in (False, True):
        # Mixed accepted prefixes, followed by 17 consecutive zero-draft steps.
        for iteration in range(18):
            before = {name: getattr(pool,name).clone() for name in spec.names}
            ref = {name: value.clone() for name,value in before.items()}
            checkpoints, outputs = [], []
            for step in range(4):
                outs = []
                for li, desc in enumerate(descriptors):
                    outs.append(factored_packed_decode(
                        mixed[li,:batch,step], aa[li,:batch,step], bb[li,:batch,step],
                        A_log=desc.A_log, dt_bias=desc.dt_bias, scale=k**-0.5, vbar=pool.vbar[li],
                        fa=ref['a'][li], fu=ref['U'][li], fw=ref['W'][li], fcount=ref['count'][li],
                        stale=pool.stale.clone(), ssm_state_indices=slots,
                        num_q_heads=h, num_v_heads=hv, head_k_dim=k, head_v_dim=v,
                        r=8, rfull=16, truncate=False, **cfg.kernel_kwargs()))
                # Exactly the ordinary final-arm batched-layer W8 decode path.
                factored_expiry_truncate_layers(ref['U'], ref['W'], ref['count'], slots, 8, 16)
                checkpoints.append({name: value[:,slots].clone() for name,value in ref.items()})
                outputs.append(outs)
            ticket = spec.snapshot_commit(slots)
            if graph_mode:
                graph.replay()
                actual = graph_outputs
            else:
                actual = verify()
            for name in spec.names:
                same(getattr(pool,name), before[name], f"unpublished {name}")
                for step in range(4):
                    same(spec.checkpoints[name][:,:batch,step], checkpoints[step][name],
                         f"graph={graph_mode} round={iteration} step={step} {name}")
            for li in range(layers):
                for step in range(4):
                    same(actual[li][:batch,step], outputs[step][li][:,0], "kernel output")
            selected = [0, 2] if iteration == 0 else [0, 0]
            spec.commit(ticket, torch.tensor(selected, device=device),
                        track_slots=torch.tensor([-1,7], device=device),
                        track_steps=torch.tensor([-1,selected[1]], device=device))
            for name in spec.names:
                expected = before[name]
                for row,step in enumerate(selected):
                    expected[:,slots[row]] = checkpoints[step][name][:,row]
                expected[:,7] = checkpoints[selected[1]][name][:,1]
                same(getattr(pool,name), expected, f"published {name}")
            evidence.append(dict(graph=graph_mode, iteration=iteration, consumed_indices=selected,
                                 exact_states=True, exact_output=True))
    torch.cuda.synchronize()
    print(json.dumps(dict(passed=True, gpu=torch.cuda.get_device_name(),
                         factor_dtype=str(cfg.dtype), scratch_bytes=spec.bytes(),
                         cases=evidence)), flush=True)


if __name__ == "__main__":
    run()
