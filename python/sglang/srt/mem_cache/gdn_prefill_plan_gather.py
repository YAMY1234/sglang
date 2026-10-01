"""Opt-in integer-only prefill metadata gather; one device-to-host read."""
import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(SLOTS, STALE, DENSE, REQUIRED, VALID, OWNERS, OUT,
            B: tl.constexpr, R: tl.constexpr, HAS_REQUIRED: tl.constexpr,
            HAS_VALID: tl.constexpr, BLOCK: tl.constexpr):
    rows: tl.constexpr = 3 + HAS_REQUIRED + HAS_VALID
    size: tl.constexpr = rows * B + (1 + HAS_REQUIRED) * R
    i = tl.arange(0, BLOCK)
    main = i < rows * B
    field = i // B
    slot = tl.load(SLOTS + i % B, mask=main, other=0).to(tl.int64)
    safe = tl.maximum(slot, 0)
    value = tl.where(field == 0, slot, 0)
    value += tl.load(STALE + safe, mask=main & (field == 1), other=0).to(tl.int64)
    value += tl.load(DENSE + safe, mask=main & (field == 2), other=0).to(tl.int64)
    if HAS_REQUIRED:
        value += tl.load(REQUIRED + safe, mask=main & (field == 3), other=0).to(tl.int64)
    if HAS_VALID:
        value += tl.load(VALID + safe, mask=main & (field == 3 + HAS_REQUIRED), other=0).to(tl.int64)
    if R > 0:
        owner_part = (i >= rows * B) & (i < size)
        offset = i - rows * B
        owner = tl.load(OWNERS + offset % R, mask=owner_part, other=0)
        value += tl.load(STALE + owner, mask=owner_part & (offset < R), other=0).to(tl.int64)
        if HAS_REQUIRED:
            value += tl.load(REQUIRED + owner, mask=owner_part & (offset >= R), other=0).to(tl.int64)
    tl.store(OUT + i, value, mask=i < size)


def reference(pool, slots, owners, first):
    safe = slots.long().clamp(min=0)
    rows = [slots.long(), pool.stale[safe].long(), pool.dense_of[safe].long()]
    if pool.dense_required is not None: rows.append(pool.dense_required[safe].long())
    if pool.prefix_valid is not None and first == 0: rows.append(pool.prefix_valid[safe].long())
    tail = [pool.stale[owners].long()]
    if pool.dense_required is not None: tail.append(pool.dense_required[owners].long())
    return torch.cat([torch.stack(rows).view(-1), torch.stack(tail).view(-1)])


def gather(pool, slots, owners, first):
    # The original kernel addresses a linear index array; preserve semantics
    # when a caller passes a strided view instead of a packed slot vector.
    slots, owners = slots.contiguous(), owners.contiguous()
    b, r = slots.numel(), owners.numel()
    if b == 0: raise ValueError('prefill metadata requires at least one row')
    required = pool.dense_required is not None
    valid = pool.prefix_valid is not None and first == 0
    size = (3 + required + valid)*b + (1 + required)*r
    out = torch.empty(size, dtype=torch.long, device=slots.device)
    _gather[(1,)](slots, pool.stale, pool.dense_of,
        pool.dense_required if required else pool.stale,
        pool.prefix_valid if valid else pool.stale, owners, out,
        b, r, required, valid, triton.next_power_of_2(size), num_warps=4)
    return out


def read_metadata(pool, slots, first):
    device = torch.device(pool.device)
    owners = torch.tensor([max(o,0) for o in pool.ring_owner], dtype=torch.long,
                          pin_memory=device.type == 'cuda').to(device, non_blocking=True)
    out = gather(pool, slots, owners, first)
    if os.environ.get('SGLANG_GDN_PREFILL_PLAN_GATHER_CHECK','0') == '1':
        expected = reference(pool, slots, owners, first)
        if not torch.equal(out, expected): raise RuntimeError('prefill metadata gather differs')
        if not getattr(pool, '_plan_gather_checked', False):
            from sglang.srt.distributed import get_tensor_model_parallel_rank
            print('SSMOFF_PLAN_GATHER_CHECK '+json.dumps(dict(
                rank=get_tensor_model_parallel_rank(), batch=slots.numel(),
                ring=len(pool.ring_owner), passed=True)), flush=True)
            pool._plan_gather_checked = True
    flat = out.tolist()
    pool._plan_gather_calls = getattr(pool, '_plan_gather_calls', 0) + 1
    if pool._plan_gather_calls == 1:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        print('PFACTOR4_PLAN_GATHER '+json.dumps(dict(
            rank=get_tensor_model_parallel_rank(), batch=slots.numel(),
            ring=len(pool.ring_owner), first_layer=first, integer_only=True,
            readback_elements=len(flat))), flush=True)
    b, r = slots.numel(), owners.numel()
    n = 3; extra = {}
    if pool.dense_required is not None: extra['required'] = n; n += 1
    if pool.prefix_valid is not None and first == 0: extra['valid'] = n; n += 1
    rows = [flat[i*b:(i+1)*b] for i in range(n)]
    tail = flat[n*b:]
    owners_rows = [tail[i*r:(i+1)*r] for i in range(1+int(pool.dense_required is not None))]
    return (*rows[:3], {k:rows[i] for k,i in extra.items()}, owners_rows)
