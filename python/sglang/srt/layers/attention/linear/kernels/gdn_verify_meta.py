"""Default-off transaction metadata kernels; retain asynchronous validation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _snapshot_meta(slots, generations, saved_slots, saved_generations,
                   work_indices, written, valid, N: tl.constexpr, S: tl.constexpr,
                   CAP: tl.constexpr, WC: tl.constexpr, BN: tl.constexpr,
                   BC: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(written + x, False, x < WC)
    if pid == 0:
        r = tl.arange(0, BN)
        s = tl.load(slots + r, r < N, other=-1)
        good = (s >= 0) & (s < S)
        duplicate = ((s[:, None] == s[None, :]) & (r[:, None] != r[None, :]) &
                     (r[:, None] < N) & (r[None, :] < N))
        okay = (tl.sum(((r < N) & ~good).to(tl.int32), 0) == 0) & (tl.sum(tl.sum(duplicate.to(tl.int32), 0), 0) == 0)
        generation = tl.load(generations + s, (r < N) & good, other=0)
        tl.store(saved_slots + r, s, r < N)
        tl.store(saved_generations + r, generation, r < N)
        c = tl.arange(0, BC)
        tl.store(work_indices + c, tl.where(c < N, c, -1), c < CAP)
        tl.store(valid, okay)


@triton.jit
def _validate_meta(slots, expected_gen, generations, steps, written, valid,
                   N: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr,
                   L: tl.constexpr, T: tl.constexpr, BN: tl.constexpr, BL: tl.constexpr):
    r = tl.arange(0, BN)
    s = tl.load(slots + r, r < N, other=-1)
    step = tl.load(steps + r, r < N, other=-1)
    good_slot = (s >= 0) & (s < S)
    generation = tl.load(generations + s, (r < N) & good_slot, other=0)
    expected = tl.load(expected_gen + r, r < N, other=0)
    good = good_slot & (generation == expected) & (step >= 0) & (step < T)
    l = tl.arange(0, BL)
    recorded = tl.load(written + l[:, None] * CAP * T + r[None, :] * T + tl.maximum(step, 0)[None, :],
                       (l[:, None] < L) & (r[None, :] < N) & good[None, :], other=False)
    bad = (l[:, None] < L) & (r[None, :] < N) & ~recorded
    okay = (tl.sum(((r < N) & ~good).to(tl.int32), 0) == 0) & (tl.sum(tl.sum(bad.to(tl.int32), 0), 0) == 0)
    tl.store(valid, okay)


@triton.jit
def _invalidate_meta(slots, generations, N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    s = tl.load(slots + r, r < N, other=-1)
    tl.atomic_add(generations + s, 1, (r < N) & (s >= 0) & (s < S), sem='relaxed')


@triton.jit
def _publish_meta(slots, valid, stale, dense_of, required, prefix,
                  N: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr,
                  HAS_STALE: tl.constexpr, HAS_DENSE: tl.constexpr,
                  HAS_REQUIRED: tl.constexpr, HAS_PREFIX: tl.constexpr):
    r = tl.arange(0, BLOCK)
    s = tl.load(slots + r, r < N, other=-1)
    mask = (r < N) & (s >= 0) & (s < S) & tl.load(valid + r, r < N, other=False)
    if HAS_STALE: tl.store(stale + s, 1, mask)
    if HAS_DENSE: tl.store(dense_of + s, -1, mask)
    if HAS_REQUIRED: tl.store(required + s, 0, mask)
    if HAS_PREFIX: tl.store(prefix + s, 0, mask)


def snapshot_metadata(owner, slots):
    saved, gen = torch.empty_like(slots), torch.empty_like(slots)
    valid = torch.empty((), dtype=torch.bool, device=slots.device)
    _snapshot_meta[(triton.cdiv(owner.written.numel(), 256),)](
        slots, owner.generations, saved, gen, owner.work_indices, owner.written, valid,
        slots.numel(), owner.generations.numel(), owner.capacity, owner.written.numel(),
        triton.next_power_of_2(slots.numel()), triton.next_power_of_2(owner.capacity), 256,
        num_warps=4)
    torch._assert_async(valid, 'invalid or duplicate factor slots')
    return saved, gen


def validate_metadata(owner, ticket, steps):
    valid = torch.empty((), dtype=torch.bool, device=steps.device)
    _validate_meta[(1,)](ticket.slots, ticket.generations, owner.generations, steps,
        owner.written, valid, steps.numel(), owner.generations.numel(), owner.capacity,
        owner.written.shape[0], owner.draft_tokens, triton.next_power_of_2(steps.numel()),
        triton.next_power_of_2(owner.written.shape[0]), num_warps=4)
    torch._assert_async(valid, 'reused factor slot, invalid accepted index, or unrecorded candidate')


def invalidate_metadata(owner, slots):
    if slots.numel():
        _invalidate_meta[(1,)](slots, owner.generations, slots.numel(), owner.generations.numel(),
            triton.next_power_of_2(slots.numel()), num_warps=4)


def publish_metadata(owner, slots, valid):
    fields=[getattr(owner.pool, k) for k in ('stale','dense_of','dense_required','prefix_valid')]
    _publish_meta[(1,)](slots, valid, *(t if t is not None else owner.generations for t in fields),
        slots.numel(), owner.generations.numel(), triton.next_power_of_2(slots.numel()),
        *(t is not None for t in fields), num_warps=4)
