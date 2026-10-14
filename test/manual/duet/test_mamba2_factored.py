"""Standalone CUDA/interpreter qualification; writes all numerical observations.

GDN convention: compare to FP64 dense recurrence with the SAME W-step best
rank-r content cut, separately from the untruncated dense trajectory. A rank
cut's residual is compared with its own optimal residual, not zero.
"""
import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
pkg = types.ModuleType("ltnkernels")
pkg.__path__ = [str(ROOT / "python/sglang/srt/layers/attention/linear/kernels")]
sys.modules[pkg.__name__] = pkg
kernel = importlib.import_module("ltnkernels.mamba2_factored")


def rel(a, b, *, vector=False):
    dim = -1 if vector else -2
    return float(((a.double()-b.double()).flatten(dim).norm(dim=-1) /
                  b.double().flatten(dim).norm(dim=-1).clamp_min(1e-30)).max())


def materialize(a, u, v, count, direction):
    mask = torch.arange(u.shape[-2], device=u.device)[None, None] < count[..., None]
    return direction[None, :, :, None] * a[..., None, :] + u.transpose(-1, -2) @ (v * mask[..., None])


def cut(state, direction, rank=16):
    coeff = (direction[None, :, None, :] @ state).squeeze(-2)
    content = state-direction[None, :, :, None]*coeff[..., None, :]
    u, s, vh = torch.linalg.svd(content, full_matrices=False)
    best = (u[..., :rank]*s[..., None, :rank]) @ vh[..., :rank, :]
    return direction[None, :, :, None]*coeff[..., None, :] + best


def case(batch, device, heads=8, steps=64, seed=13):
    torch.manual_seed(seed)
    p, n, groups, r, w = 64, 128, 2, 16, 16
    direction = torch.nn.functional.normalize(torch.randn(heads, p), dim=-1)
    dd = direction.double()
    # Exact zero-state initialization; reserved zero rows count toward W phase.
    a = torch.zeros(batch+1, heads, n, device=device)
    u = torch.zeros(batch+1, heads, r+w, p, device=device)
    v = torch.zeros(batch+1, heads, r+w, n, device=device)
    count = torch.full((batch+1, heads), r, dtype=torch.int32, device=device)
    # Reverse, nonzero, non-contiguous slot vector exercises physical addressing.
    indices = torch.arange(batch, 0, -1, dtype=torch.int32, device=device)
    slots = torch.zeros(batch*2, dtype=torch.int32, device=device)
    slots[::2] = indices
    slots = slots[::2]
    dense = torch.zeros(batch, heads, p, n, dtype=torch.float64)
    uncut = dense.clone()
    direction_dev = direction.to(device)
    d = torch.randn(heads)
    ddev = d.to(device)
    records, cuts = [], []
    for t in range(steps):
        # Packed views deliberately preserve nontrivial row strides.
        packed = torch.randn(batch, heads*p+2*groups*n)
        xx, bb, cc = packed.split([heads*p, groups*n, groups*n], -1)
        x, b, c = xx.reshape(batch, heads, p), bb.reshape(batch, groups, n), cc.reshape(batch, groups, n)
        dt = torch.nn.functional.softplus(torch.randn(batch, heads))
        decay = (dt * -0.15).exp()
        pd = packed.to(device)
        xd, bd, cd = pd.split([heads*p, groups*n, groups*n], -1)
        out = kernel.step(xd.reshape_as(x), bd.reshape_as(b), cd.reshape_as(c), dt.to(device), decay.to(device), ddev,
                          direction_dev, a, u, v, count, slots, do_truncate=False)
        bh = b.double().repeat_interleave(heads//groups, 1)
        ch = c.double().repeat_interleave(heads//groups, 1)
        sx = dt.double()[..., None]*x.double()
        dense = decay.double()[..., None, None]*dense + sx[..., None]*bh[..., None, :]
        uncut = decay.double()[..., None, None]*uncut + sx[..., None]*bh[..., None, :]
        expected = (dense @ ch[..., None]).squeeze(-1) + d.double()[None, :, None]*x.double()
        full_y = (uncut @ ch[..., None]).squeeze(-1) + d.double()[None, :, None]*x.double()
        got = out.cpu()
        records.append(dict(step=t+1, output_rel_f=rel(got, expected, vector=True),
                            bf16_rounding_rel_f=rel(expected.bfloat16().float(), expected, vector=True),
                            untruncated_rel_f=rel(got, full_y, vector=True)))
        if (t+1) % w == 0:
            # The unit direction is FP32: normalize in double for SVD oracle.
            # Its 1e-7 norm error is part of the recorded FP32 envelope.
            before = materialize(a, u, v, count, direction_dev)[indices.long()].cpu().double()
            optimal = cut(before, dd)
            kernel.truncate(u, v, count, slots)
            after = materialize(a, u, v, count, direction_dev)[indices.long()].cpu().double()
            best_residual = (before-optimal).flatten(-2).norm(dim=-1)
            actual_residual = (before-after).flatten(-2).norm(dim=-1)
            cuts.append(dict(step=t+1, state_vs_best_rel_f=rel(after, optimal),
                             excess_residual=float((actual_residual/(best_residual+1e-20)).max())))
            dense = cut(dense, dd)
        # Unaddressed slot must remain untouched.
        assert not bool(a[0].any()) and not bool(u[0].any()) and not bool(v[0].any())
    # Padding must neither access nor mutate the pool.
    saved = [z.clone() for z in (a, u, v, count)]
    pad = torch.full_like(slots, -1)
    ypad = kernel.step(xd.reshape_as(x), bd.reshape_as(b), cd.reshape_as(c), dt.to(device), decay.to(device), ddev,
                       direction_dev, a, u, v, count, pad)
    assert not bool(ypad.any())
    assert all(torch.equal(old, now) for old, now in zip(saved, (a, u, v, count)))
    passed = all(row['output_rel_f'] <= max(row['bf16_rounding_rel_f'], 1e-5) for row in records)
    result = dict(batch=batch, heads=heads, steps=steps, numerical_pass=passed, rows=records, cuts=cuts)
    if device == 'cuda':
        # Capturing mutates buffers; restore BOTH arms to the same post-capture
        # snapshot. Change x/dt/decay and slot mapping between each replay.
        stable_x, stable_b, stable_c = x.to(device), b.to(device), c.to(device)
        stable_dt, stable_decay = dt.to(device), decay.to(device)
        graph_slots = indices.clone()
        gout = torch.empty_like(stable_x)
        def run(buffers, output):
            return kernel.step(stable_x, stable_b, stable_c, stable_dt, stable_decay, ddev,
                               direction_dev, *buffers, graph_slots, out=output)
        run((a,u,v,count), gout)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run((a,u,v,count), gout)
        eager = [z.clone() for z in (a,u,v,count)]
        eout = torch.empty_like(gout)
        for i in range(64):
            stable_x.mul_(0.99).add_(0.001)
            stable_dt.mul_(0.999)
            stable_decay.copy_((stable_dt*-0.15).exp())
            graph_slots.copy_(indices.roll(i))
            run(eager, eout)
            graph.replay()
            torch.testing.assert_close(gout, eout, rtol=0, atol=0)
        assert all(torch.equal(old, now) for old,now in zip(eager,(a,u,v,count)))
        result['graph_replay_64_changing_inputs_slots'] = 'PASS'
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batches', default='1,8,32')
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(torch=torch.__version__, device=args.device, rows=[])
    if args.device == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    for batch in map(int,args.batches.split(',')):
        row = case(batch,args.device,args.heads,args.steps)
        result['rows'].append(row)
        Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(batch=batch, numerical_pass=row['numerical_pass'],
                              max_output_rel_f=max(r['output_rel_f'] for r in row['rows']), cuts=row['cuts'])),flush=True)
    if not all(r['numerical_pass'] for r in result['rows']):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
