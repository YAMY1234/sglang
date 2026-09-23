"""Causal QSA pending-ring candidates for fixed-width chain verification."""
from types import SimpleNamespace

import torch


def verify_write_locations(metadata, positions, ratio):
    if metadata.is_cuda_graph:
        # Graph metadata intentionally uses a one-column dummy token table;
        # its device planner has already resolved the real request page slots.
        if metadata.graph_write_locs is None:
            raise RuntimeError("QSA verify graph has no compressed write plan")
        return metadata.graph_write_locs
    rows = torch.arange(positions.numel(), device=positions.device)
    last_locs = metadata.token_slot_table[rows, positions.long()]
    return torch.where((positions + 1) % ratio == 0, last_locs // ratio, 0)


def causal_groups(old_keys, old_rope, keys, rope, requests, positions, width, ratio):
    """Gather a query's completed group before future inputs overwrite the ring.

    A width <= ratio does not by itself prevent aliasing: an unaligned window
    overwrites the *previous* group's early members before it is compressed.
    Members from before this window read the committed ring; current members
    read their exact token rows. No query can read a future candidate.
    """
    n = positions.numel()
    row = torch.arange(n, device=positions.device)
    base = row // width * width
    members = positions[:, None] - ratio + 1 + torch.arange(ratio, device=positions.device)
    first = positions[base]
    candidate_rows = base[:, None] + members - first[:, None]
    from_candidate = (members >= first[:, None]) & (members <= positions[:, None])
    safe = candidate_rows.clamp(0, max(n-1, 0)).long()
    ring_rows = requests[:, None].long() * ratio + members.remainder(ratio)
    key_groups = torch.where(from_candidate[:, :, None, None], keys[safe], old_keys[ring_rows])
    rope_groups = torch.where(from_candidate[:, :, None], rope[safe], old_rope[ring_rows])
    return key_groups, rope_groups


class QSAVerifyState:
    def __init__(self, pool, capacity, width):
        if not 1 <= width <= pool.qsa_compress_ratio:
            raise ValueError("QSA verify width must fit one pending ring")
        self.pool, self.capacity, self.width = pool, capacity, width
        self.ratio = pool.qsa_compress_ratio
        self.working = [torch.zeros_like(t) for t in pool.qsa_key_state_buffer_pool]
        self.rope_working = torch.zeros_like(pool.qsa_rope_position_buffer)
        head_shape = self.working[0].shape[1:]
        self.keys = torch.zeros((len(self.working), capacity, width, *head_shape),
                                dtype=self.working[0].dtype, device=self.working[0].device)
        self.rope = torch.zeros((capacity, width, 3), dtype=torch.int64, device=self.keys.device)
        self.positions = torch.zeros((capacity, width), dtype=torch.int64, device=self.keys.device)
        self.requests = None
        self.closed = True

    def bytes(self):
        return sum(t.numel()*t.element_size() for t in
                   (*self.working, self.rope_working, self.keys, self.rope, self.positions))

    def begin(self, requests):
        if not self.closed:
            raise RuntimeError("uncommitted QSA verify transaction")
        if not 0 < requests.numel() <= self.capacity:
            raise ValueError("invalid QSA verify batch size")
        self.requests = requests.clone().long()
        self.closed = False

    def shadow(self, layer_id):
        li = self.pool._transfer_full_attention_id(layer_id)
        self.working[li].copy_(self.pool.get_qsa_key_state_buffer(layer_id))
        self.rope_working.copy_(self.pool.qsa_rope_position_buffer)
        return SimpleNamespace(get_qsa_key_state_buffer=lambda _: self.working[li],
                               qsa_rope_position_buffer=self.rope_working)

    def record(self, layer_id, keys, rope, positions):
        n = keys.shape[0] // self.width
        if keys.shape[0] != n * self.width or n > self.capacity:
            raise ValueError("QSA verify must have a fixed number of rows per request")
        li = self.pool._transfer_full_attention_id(layer_id)
        self.keys[li, :n].copy_(keys.reshape(n,self.width,*keys.shape[1:]))
        self.rope[:n].copy_(rope.reshape(n,self.width,3))
        self.positions[:n].copy_(positions.reshape(n,self.width))

    def commit(self, steps):
        if self.closed or steps.shape != self.requests.shape:
            raise RuntimeError("QSA commit without matching live verify")
        torch._assert_async(torch.all((steps >= 0) & (steps < self.width)), "invalid QSA commit position")
        n = steps.numel()
        offsets = torch.arange(self.width, device=steps.device)[None,:]
        valid = offsets <= steps[:,None]
        slots = self.requests[:,None] * self.ratio + self.positions[:n].remainder(self.ratio)
        slots = torch.where(valid, slots, -1).flatten()
        source_steps = torch.zeros_like(slots)
        from sglang.srt.mem_cache.gdn_factored_spec import FactoredGDNVerifyState
        for li, dst in enumerate(self.pool.qsa_key_state_buffer_pool):
            src = self.keys[li,:n].reshape(1,n*self.width,1,*dst.shape[1:])
            FactoredGDNVerifyState._scatter(dst[None], src, slots, source_steps)
        src = self.rope[:n].reshape(1,n*self.width,1,3)
        FactoredGDNVerifyState._scatter(self.pool.qsa_rope_position_buffer[None], src, slots, source_steps)
        self.closed = True
