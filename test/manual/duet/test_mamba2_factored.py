"""Standalone CUDA/interpreter qualification; writes all numerical observations.

GDN convention: compare to FP64 dense recurrence with the SAME W-step MGS2
content cut. Exact rank-r SVD and untruncated trajectories are separate diagnostics. A rank
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
    coeff = coeff / direction.square().sum(-1).clamp_min(1e-12)[None, :, None]
    content = state-direction[None, :, :, None]*coeff[..., None, :]
    u, s, vh = torch.linalg.svd(content, full_matrices=False)
    best = (u[..., :rank]*s[..., None, :rank]) @ vh[..., :rank, :]
    return direction[None, :, :, None]*coeff[..., None, :] + best


def mgs_cut(u, v, rank=16):
    """Independent FP64 GDN MGS2 oracle (same diagonal seed and 3 rounds)."""
    gram = v @ v.transpose(-1, -2)
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    order = diagonal.argsort(dim=-1, descending=True, stable=True)[..., :rank]
    z = torch.zeros_like(gram)
    z[..., :rank] = torch.nn.functional.one_hot(order, u.shape[-2]).transpose(-1, -2).double()
    for _ in range(3):
        z = gram @ z
        norm0 = z.norm(dim=-2)
        for sweep in range(2):
            for j in range(rank):
                y = z[..., j].clone()
                prev = z[..., :j]
                y -= (prev * (prev * y[..., None]).sum(-2)[..., None, :]).sum(-1)
                norm = y.norm(dim=-1)
                ok = norm > 1e-12
                if sweep == 0:
                    ok &= norm > 1e-4 * norm0[..., j]
                z[..., j] = torch.where(ok[..., None], y / norm.clamp_min(1e-30)[..., None], 0.)
    return z.transpose(-1, -2)[..., :rank, :] @ u, z.transpose(-1, -2)[..., :rank, :] @ v


def reference_append(a, u, v, count, x, b, dt, decay, direction):
    sx = dt[..., None] * x
    coeff = (sx * direction).sum(-1) / direction.square().sum(-1)
    tail = sx - coeff[..., None] * direction
    a = decay[..., None] * a + coeff[..., None] * b
    active = u[..., :count, :]
    proj = (active * tail[..., None, :]).sum(-1)
    tail -= (active * proj[..., None]).sum(-2)
    correction = (active * tail[..., None, :]).sum(-1)
    tail -= (active * correction[..., None]).sum(-2)
    proj += correction
    norm = tail.norm(dim=-1)
    base = (sx - coeff[..., None] * direction).norm(dim=-1)
    keep = norm > 1e-6 * base
    u[..., count, :] = torch.where(keep[..., None], tail / norm.clamp_min(1e-30)[..., None], 0.)
    v[..., :count, :] = decay[..., None, None] * v[..., :count, :] + proj[..., None] * b[..., None, :]
    v[..., count, :] = torch.where(keep, norm, 0.)[..., None] * b
    return a, u, v


def case(batch, device, heads=8, steps=64, seed=13, native=False):
    torch.manual_seed(seed)
    p, n, groups, r, w = 64, 128, 2, 16, 16
    direction = torch.nn.functional.normalize(torch.randn(heads, p), dim=-1)
    dd = direction.double()
    # Exact zero-state initialization; reserved zero rows count toward W phase.
    a = torch.zeros(batch+1, heads, n, device=device)
    u = torch.zeros(batch+1, heads, r+w, p, device=device)
    v = torch.zeros(batch+1, heads, r+w, n, device=device)
    count = torch.full((batch+1, heads), r, dtype=torch.int32, device=device)
    fused_buffers = [z.clone() for z in (a, u, v, count)]
    # Reverse, nonzero, non-contiguous slot vector exercises physical addressing.
    indices = torch.arange(batch, 0, -1, dtype=torch.int32, device=device)
    slots = torch.zeros(batch*2, dtype=torch.int32, device=device)
    slots[::2] = indices
    slots = slots[::2]
    dense = torch.zeros(batch, heads, p, n, dtype=torch.float64)
    uncut = dense.clone()
    best_dense = dense.clone()
    ra = torch.zeros(batch, heads, n, dtype=torch.float64)
    ru = torch.zeros(batch, heads, r+w, p, dtype=torch.float64)
    rv = torch.zeros(batch, heads, r+w, n, dtype=torch.float64)
    rc = r
    direction_dev = direction.to(device)
    d = torch.randn(heads)
    bias = torch.randn(heads).bfloat16() if native else None
    aval = torch.full((heads,), -0.15, dtype=torch.float32)
    if native:
        d = d.bfloat16()
    biasdev = bias.to(device) if native else None
    ddev = d.to(device)
    records, cuts = [], []
    for t in range(steps):
        # Packed views deliberately preserve nontrivial row strides.
        packed = torch.randn(batch, heads*p+2*groups*n)
        if native:
            packed = packed.bfloat16()
        xx, bb, cc = packed.split([heads*p, groups*n, groups*n], -1)
        x, b, c = xx.reshape(batch, heads, p), bb.reshape(batch, groups, n), cc.reshape(batch, groups, n)
        dt = torch.nn.functional.softplus(torch.randn(batch, heads))
        if native:
            raw_dt = torch.randn(batch, heads).bfloat16()
            biased = raw_dt.float() + bias.float()
            dt = torch.where(biased <= 20., (1. + biased.exp()).log(), biased)
        decay = (dt * aval).exp()
        step_dt = raw_dt.to(device) if native else dt.to(device)
        step_decay = aval.to(device) if native else decay.to(device)
        pd = packed.to(device)
        xd, bd, cd = pd.split([heads*p, groups*n, groups*n], -1)
        out = kernel.step(xd.reshape_as(x), bd.reshape_as(b), cd.reshape_as(c), step_dt, step_decay, ddev,
                          direction_dev, a, u, v, count, slots, do_truncate=False, dt_bias=biasdev)
        fused_out = kernel.step(
            xd.reshape_as(x), bd.reshape_as(b), cd.reshape_as(c), step_dt, step_decay, ddev,
            direction_dev, *fused_buffers, slots, dt_bias=biasdev,
        )
        bh = b.double().repeat_interleave(heads//groups, 1)
        ch = c.double().repeat_interleave(heads//groups, 1)
        sx = dt.double()[..., None]*x.double()
        dense = decay.double()[..., None, None]*dense + sx[..., None]*bh[..., None, :]
        uncut = decay.double()[..., None, None]*uncut + sx[..., None]*bh[..., None, :]
        best_dense = decay.double()[..., None, None]*best_dense + sx[..., None]*bh[..., None, :]
        ra, ru, rv = reference_append(ra, ru, rv, rc, x.double(), bh, dt.double(), decay.double(), dd)
        rc += 1
        expected = (dense @ ch[..., None]).squeeze(-1) + d.double()[None, :, None]*x.double()
        full_y = (uncut @ ch[..., None]).squeeze(-1) + d.double()[None, :, None]*x.double()
        got = out.cpu()
        best_expected = (best_dense @ ch[..., None]).squeeze(-1) + d.double()[None, :, None]*x.double()
        records.append(dict(step=t+1, output_rel_f=rel(got, expected, vector=True),
                            fused_output_rel_f=rel(fused_out.cpu(), expected, vector=True),
                            bf16_rounding_rel_f=rel(expected.bfloat16().float(), expected, vector=True),
                            untruncated_rel_f=rel(got, full_y, vector=True),
                            best_rank16_rel_f=rel(got, best_expected, vector=True),
                            best_rank16_fused_rel_f=rel(fused_out.cpu(), best_expected, vector=True),
                            best_rank16_bf16_rounding_rel_f=rel(best_expected.bfloat16().float(), best_expected, vector=True)))
        if (t+1) % w == 0:
            # Recompute the sink projection in FP64, including the norm of
            # the FP32-stored direction (not assumed to be exactly one).
            before = materialize(a, u, v, count, direction_dev)[indices.long()].cpu().double()
            optimal = cut(before, dd)
            input_u = u[indices.long()].cpu().double()
            input_v = v[indices.long()].cpu().double()
            oracle_u, oracle_v = mgs_cut(input_u, input_v)
            local_mgs = dd[None, :, :, None] * a[indices.long()].cpu().double()[..., None, :] + oracle_u.transpose(-1, -2) @ oracle_v
            kernel.truncate(u, v, count, slots)
            after = materialize(a, u, v, count, direction_dev)[indices.long()].cpu().double()
            best_residual = (before-optimal).flatten(-2).norm(dim=-1)
            actual_residual = (before-after).flatten(-2).norm(dim=-1)
            cuts.append(dict(step=t+1, state_vs_best_rel_f=rel(after, optimal),
                             cut_vs_fp64_same_inputs_rel_f=rel(after, local_mgs),
                             excess_residual=float((actual_residual/(best_residual+1e-20)).max())))
            ruc, rvc = mgs_cut(ru, rv)
            ru.zero_(); rv.zero_()
            ru[..., :r, :] = ruc; rv[..., :r, :] = rvc
            rc = r
            dense = dd[None, :, :, None]*ra[..., None, :] + ru.transpose(-1, -2) @ rv
            best_dense = cut(best_dense, dd)
            cuts[-1]['state_vs_fp64_mgs_rel_f'] = rel(after, dense)
            cuts[-1]['fused_state_vs_fp64_mgs_rel_f'] = rel(materialize(*fused_buffers, direction_dev)[indices.long()].cpu(), dense)
        # Unaddressed slot must remain untouched.
        assert not bool(a[0].any()) and not bool(u[0].any()) and not bool(v[0].any())
    # Padding must neither access nor mutate the pool.
    saved = [z.clone() for z in (a, u, v, count)]
    pad = torch.full_like(slots, -1)
    ypad = kernel.step(xd.reshape_as(x), bd.reshape_as(b), cd.reshape_as(c), step_dt, step_decay, ddev,
                       direction_dev, a, u, v, count, pad, dt_bias=biasdev)
    assert not bool(ypad.any())
    assert all(torch.equal(old, now) for old, now in zip(saved, (a, u, v, count)))
    passed = all(max(row['output_rel_f'], row['fused_output_rel_f']) <= max(row['bf16_rounding_rel_f'], 1e-5) for row in records)
    if native:
        passed = all(max(row['output_rel_f'], row['fused_output_rel_f']) <= row['bf16_rounding_rel_f'] + 2e-5 for row in records)
    best_passed = all(
        max(row['best_rank16_rel_f'], row['best_rank16_fused_rel_f']) <= (
            row['best_rank16_bf16_rounding_rel_f'] + 2e-5 if native
            else max(row['best_rank16_bf16_rounding_rel_f'], 1e-5)
        ) for row in records
    )
    result = dict(batch=batch, native_dtype=native, heads=heads, steps=steps,
                  reference='FP64 dense recurrence with optimal rank16 content SVD every W16 steps (sink retained)',
                  numerical_pass=best_passed, same_mgs_numerical_pass=passed, rows=records, cuts=cuts)
    if device == 'cuda':
        # Capturing mutates buffers; restore BOTH arms to the same post-capture
        # snapshot. Change x/dt/decay and slot mapping between each replay.
        stable_x, stable_b, stable_c = x.to(device), b.to(device), c.to(device)
        stable_dt, stable_decay = step_dt.clone(), step_decay.clone()
        graph_slots = indices.clone()
        gout = torch.empty_like(stable_x)
        def run(buffers, output):
            return kernel.step(stable_x, stable_b, stable_c, stable_dt, stable_decay, ddev,
                               direction_dev, *buffers, graph_slots, out=output, dt_bias=biasdev)
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
            if not native:
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
    variants = (False, True) if args.device == 'cuda' else (False,)
    for batch, native in ((b, n) for n in variants for b in map(int, args.batches.split(','))):
        row = case(batch,args.device,args.heads,args.steps,native=native)
        result['rows'].append(row)
        Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(batch=batch, native_dtype=native, numerical_pass=row['numerical_pass'],
                              max_output_rel_f=max(r['output_rel_f'] for r in row['rows']), cuts=row['cuts'])),flush=True)
    if not all(r['numerical_pass'] for r in result['rows']):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
