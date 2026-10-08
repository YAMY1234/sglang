"""Index layout of the DeepSeek-V4.1 per-request SWA window, built in two launches.

Rows of one request are contiguous. A one-program scan finds each row's group
and each group's first row; a row/group-parallel pass then writes every field
``window_layout`` returns (see ``dsv41_request_window``), with no host sync.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _window_groups_kernel(
    req,
    pos,
    floor,
    n,
    row_group,
    group_start,
    group_req,
    group_first_pos,
    group_floor,
    num_groups,
    HAS_FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    carry = 0
    for block in range(0, tl.cdiv(n, BLOCK)):
        offs = block * BLOCK + tl.arange(0, BLOCK)
        live = offs < n
        r = tl.load(req + offs, mask=live, other=0)
        prev = tl.load(req + offs - 1, mask=live & (offs > 0), other=-1)
        start = live & ((offs == 0) | (r != prev))
        group = carry + tl.cumsum(start.to(tl.int32), axis=0) - 1
        tl.store(row_group + offs, group, mask=live)
        tl.store(group_start + group, offs.to(tl.int64), mask=start)
        tl.store(group_req + group, r, mask=start)
        tl.store(group_first_pos + group, tl.load(pos + offs, mask=live, other=0), mask=start)
        if HAS_FLOOR:
            tl.store(group_floor + group, tl.load(floor + offs, mask=live, other=0), mask=start)
        carry += tl.sum(start.to(tl.int32), axis=0)
    tl.store(num_groups, carry)


@triton.jit
def _window_layout_kernel(
    pos,
    floor,
    row_group,
    group_start,
    group_req,
    group_first_pos,
    group_floor,
    num_groups,
    n,
    groups,
    capacity,
    write_loc,
    indices,
    lengths,
    commit_mask,
    history_req,
    history_pos,
    history_loc,
    history_valid,
    WINDOW: tl.constexpr,
    HAS_FLOOR: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.arange(0, WINDOW)
    real_groups = tl.load(num_groups)
    history_rows = groups * WINDOW
    if pid < n:
        g = tl.load(row_group + pid)
        first_row = tl.load(group_start + g)
        has_next = g + 1 < real_groups
        last_row = tl.where(
            has_next, tl.load(group_start + g + 1, mask=has_next, other=0), n
        ) - 1
        p = tl.load(pos + pid)
        first_pos = p - (pid - first_row)
        seen = p - k
        old_loc = g * WINDOW + seen - (first_pos - WINDOW)
        new_loc = history_rows + first_row + seen - first_pos
        valid = seen >= 0
        if HAS_FLOOR:
            valid = valid & (seen >= tl.load(floor + pid))
        idx = tl.where(valid, tl.where(seen < first_pos, old_loc, new_loc), -1)
        tl.store(indices + pid * WINDOW + k, idx.to(tl.int32))
        tl.store(lengths + pid, tl.sum(valid.to(tl.int32), axis=0))
        tl.store(write_loc + pid, (history_rows + pid).to(tl.int32))
        tl.store(commit_mask + pid, (last_row - pid) < capacity)
    else:
        g = pid - n
        live = g < real_groups
        hp = tl.load(group_first_pos + g, mask=live, other=0) - WINDOW + k
        hv = (hp >= 0) & live
        if HAS_FLOOR:
            hv = hv & (hp >= tl.load(group_floor + g, mask=live, other=0))
        base = g * WINDOW + k
        tl.store(history_req + base, tl.zeros_like(k).to(tl.int64) + tl.load(group_req + g, mask=live, other=0))
        tl.store(history_pos + base, hp)
        tl.store(history_loc + base, base.to(tl.int64))
        tl.store(history_valid + base, hv)


def build_window_layout(
    req: torch.Tensor,
    pos: torch.Tensor,
    *,
    window: int,
    capacity: int,
    floor: Optional[torch.Tensor],
    groups: int,
):
    """Fields of ``WindowLayout`` (without ``size``) for contiguous per-request rows."""
    n = pos.numel()
    device = pos.device
    req = req.to(torch.int64).contiguous()
    pos = pos.to(torch.int64).contiguous()
    has_floor = floor is not None
    floor = floor.to(torch.int64).contiguous() if has_floor else pos
    scratch = max(groups, n) + 1
    row_group = torch.empty(n, dtype=torch.int32, device=device)
    group_start = torch.empty(scratch, dtype=torch.int64, device=device)
    group_req = torch.empty(scratch, dtype=torch.int64, device=device)
    group_first_pos = torch.empty(scratch, dtype=torch.int64, device=device)
    group_floor = torch.empty(scratch, dtype=torch.int64, device=device)
    num_groups = torch.empty(1, dtype=torch.int32, device=device)
    _window_groups_kernel[(1,)](
        req, pos, floor, n, row_group, group_start, group_req, group_first_pos,
        group_floor, num_groups, HAS_FLOOR=has_floor, BLOCK=1024,
    )
    write_loc = torch.empty(n, dtype=torch.int32, device=device)
    indices = torch.empty((n, window), dtype=torch.int32, device=device)
    lengths = torch.empty(n, dtype=torch.int32, device=device)
    commit_mask = torch.empty(n, dtype=torch.bool, device=device)
    history_req = torch.empty(groups * window, dtype=torch.int64, device=device)
    history_pos = torch.empty_like(history_req)
    history_loc = torch.empty_like(history_req)
    history_valid = torch.empty(groups * window, dtype=torch.bool, device=device)
    _window_layout_kernel[(n + groups,)](
        pos, floor, row_group, group_start, group_req, group_first_pos, group_floor,
        num_groups, n, groups, capacity, write_loc, indices, lengths, commit_mask,
        history_req, history_pos, history_loc, history_valid,
        WINDOW=window, HAS_FLOOR=has_floor,
    )
    return (req, pos, write_loc, indices, lengths, history_req, history_pos,
            history_loc, history_valid, commit_mask)
