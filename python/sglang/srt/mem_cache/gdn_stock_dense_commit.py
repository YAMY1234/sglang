"""P-only dense checkpoint storage and one all-layer factor publication.

The serving adapter selects this implementation explicitly. It never changes
the decode factor arithmetic, and the local dense sibling is not transferred.
"""
from contextlib import contextmanager
from dataclasses import replace

import torch
import triton
import triton.language as tl


FLAG = "SGLANG_GDN_PREFILL_STOCK_DENSE_COMMIT"


def checkpoint_plan(live, prefix, extend, track_mask, track_slots, track_lengths,
                    *, size, chunk=64):
    """Host mirror of native final/checkpoint selection, including force-h +1.

    Never infer a saved checkpoint from the last token: the scheduler's track
    length may encode a branching checkpoint with the native +1 sentinel.
    """
    if len(live) != len(set(live)):
        raise ValueError("duplicate live dense slots")
    saved = {}
    for slot, start, length in zip(live, prefix, extend, strict=True):
        if not 0 < slot <= size or start < 0 or length < 1:
            raise ValueError("invalid dense slot or extent")
        saved[slot] = start + length
    if track_mask is not None:
        if not (len(track_mask) == len(track_slots) == len(track_lengths) == len(live)):
            raise ValueError("invalid checkpoint metadata length")
        for row, active in enumerate(track_mask):
            if not active:
                continue
            slot, start, length = track_slots[row], prefix[row], extend[row]
            relative = track_lengths[row] - start
            depth = start + (relative // chunk) * chunk
            if (not 0 < slot <= size or relative < chunk or depth > start + length
                    or (relative % chunk == 0 and relative != length)):
                raise ValueError("checkpoint does not match native dense tracking")
            if slot in saved:
                # Even equal-depth aliasing could overwrite another request.
                if slot != live[row] or saved[slot] != depth:
                    raise ValueError("checkpoint aliases a different live state")
            saved[slot] = depth
    return tuple(saved), tuple(saved.values())


def scratch_bound_bytes(layers, heads, value, key, cfg, checkpoints=2):
    """Conservative live tensors for iter factorization, excluding BLAS workspace.

    Six FP32 full-state equivalents cover input conversion/content/expressions;
    probes, bases and products are charged separately, with 64 MiB workspace.
    """
    n = layers * heads * checkpoints
    width = cfg.r + cfg.init_oversample
    state = n * value * key
    return (state * (2 * 2 + 6 * 4)
            + n * 4 * (8 * max(value, key) * width + 4 * width * width)
            + n * (key * 4 + cfg.rmax * (key + value) * 2)
            + (64 << 20))


def factorize_checkpoints(states, vbar, cfg):
    """[layer, checkpoint, head, V, K] -> native factors [checkpoint, L*H, ...].

    Match factorize_layers' seed-0 probe per layer, without a Python per-layer
    contiguous/split/store chain. Checkpoint membership/order is fixed by caller.
    """
    from .gdn_factored_pool import factorize_dense

    layers, count, heads, value, key = states.shape
    if count == 0 or tuple(vbar.shape) != (layers, heads, value):
        raise ValueError("invalid all-layer checkpoint tensor")
    dense = states.permute(1, 0, 2, 3, 4).reshape(count, layers * heads, value, key)
    generator = torch.Generator(device=states.device).manual_seed(0)
    probe = torch.randn(count, heads, value, cfg.r + cfg.init_oversample,
                        generator=generator, device=states.device, dtype=torch.float32)
    probe = probe[:, None].expand(count, layers, heads, value, -1).reshape(
        count, layers * heads, value, -1)
    return factorize_dense(dense, vbar.reshape(layers * heads, value),
                           cfg.r, cfg.rmax, cfg.dtype, iters=cfg.init_iters,
                           oversample=cfg.init_oversample, method=cfg.init_method,
                           omega=probe)


@triton.jit
def _publish(A, U, W, Slots, OutA, OutU, OutW, Count, Stale, DenseOf,
             Required, Valid, C: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
             S: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
             RANK: tl.constexpr, BLOCK: tl.constexpr,
             HAS_REQUIRED: tl.constexpr, HAS_VALID: tl.constexpr,
             A_C: tl.constexpr, A_H: tl.constexpr, A_K: tl.constexpr):
    lh = tl.program_id(0) // C
    checkpoint = tl.program_id(0) % C
    layer, head = lh // H, lh % H
    slot = tl.load(Slots + checkpoint)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = checkpoint * L * H + lh
    destination = (layer * S + slot) * H + head
    a = tl.load(A + checkpoint * A_C + lh * A_H + x * A_K, x < K, other=0)
    u = tl.load(U + source * R * K + x, x < R * K, other=0)
    w = tl.load(W + source * R * V + x, x < R * V, other=0)
    tl.store(OutA + destination * K + x, a, x < K)
    tl.store(OutU + destination * R * K + x, u, x < R * K)
    tl.store(OutW + destination * R * V + x, w, x < R * V)
    if tl.program_id(1) == 0:
        tl.store(Count + destination, RANK)
        if lh == 0:
            # No dense ring owner is needed: the local exact sibling owns
            # continuation, while the handoff payload is compact rank RANK.
            tl.store(Stale + slot, 1)
            tl.store(DenseOf + slot, -1)
            if HAS_REQUIRED:
                tl.store(Required + slot, 0)
            if HAS_VALID:
                tl.store(Valid + slot, 1)


def publish_checkpoints(pool, slots, factors):
    """One kernel publishes every layer/slot/head; consumers share its stream."""
    a, u, w = factors
    layers, _, heads, key = pool.a.shape
    count = slots.numel()
    if (not count or a.shape != (count, layers * heads, key)
            or u.shape != (count, layers * heads, pool.cfg.rmax, key)
            or w.shape != (count, layers * heads, pool.cfg.rmax, pool.v)
            or not all(t.is_contiguous() for t in (u, w, slots))):
        raise ValueError("publication tensors have an invalid layout")
    # Slot uniqueness and bounds are validated by the host checkpoint plan;
    # do not introduce a GPU-to-host wait for every publication.
    _publish[(layers * heads * count,
              triton.cdiv(pool.cfg.rmax * max(key, pool.v), 256))](
        a, u, w, slots, pool.a, pool.U, pool.W, pool.count,
        pool.stale, pool.dense_of, pool.dense_required, pool.prefix_valid,
        count, heads, layers, pool.size + 1, key, pool.v, pool.cfg.rmax,
        pool.cfg.r, 256, pool.dense_required is not None,
        pool.prefix_valid is not None, *a.stride(), num_warps=4)


class DenseCheckpoints:
    """Exact P-local state follows the existing slot clear/copy/host lifecycle."""
    def __init__(self, pool, *, dtype=torch.bfloat16):
        self.states = torch.zeros(len(pool.layer_ids), pool.size + 1,
                                  pool.hv, pool.v, pool.k,
                                  dtype=dtype, device=pool.device)
        self.lengths = torch.zeros(pool.size + 1, dtype=torch.int64, device=pool.device)

    def reset_slots(self, slots):
        self.states[:, slots] = 0
        self.lengths[slots] = 0

    def copy_slots(self, source, destination):
        self.states[:, destination] = self.states[:, source]
        self.lengths[destination] = self.lengths[source]

    def get_cpu_slots(self, slots):
        return (self.states[:, slots].to("cpu", non_blocking=True),
                self.lengths[slots].to("cpu", non_blocking=True))

    def load_cpu_slots(self, data, slots):
        expected = (self.states[:, slots], self.lengths[slots])
        if (len(data) != 2 or any(a.shape != b.shape or a.dtype != b.dtype
                                  for a, b in zip(data, expected))):
            raise ValueError("invalid exact dense checkpoint host payload")
        self.states[:, slots] = data[0].to(self.states.device, non_blocking=True)
        self.lengths[slots] = data[1].to(self.lengths.device, non_blocking=True)

    def mem_usage_bytes(self):
        return self.states.nbytes + self.lengths.nbytes

    def iter_transfer_state_entries(self):
        """Local scan checkpoints never enter the PD factor handoff."""
        return iter(())

    @contextmanager
    def stock_scan(self, backend, kernel):
        """Use the unchanged dense backend body and its native checkpoint plan.

        The ordinary temporal tensor stays empty outside this scope, so the
        transport registration still sees only the original factor payload.
        No scheduler or copy callback runs inside a model forward.
        """
        mamba = backend.req_to_token_pool.mamba_pool
        old_cache = mamba.mamba_cache
        old_factors = backend.factored
        old_kernel = backend.kernel_dispatcher.extend_kernel
        old_metadata = backend.forward_metadata
        mamba.mamba_cache = replace(old_cache, temporal=self.states)
        backend.factored = None
        backend.kernel_dispatcher.extend_kernel = kernel
        try:
            yield
        finally:
            backend.forward_metadata = old_metadata
            backend.kernel_dispatcher.extend_kernel = old_kernel
            backend.factored = old_factors
            mamba.mamba_cache = old_cache
