"""Opt-in low-batch V tiling: immutable metadata before in-place CTA updates.

The head's old a/count must be snapshotted before any V tile updates them.
U rows below old count are read-only; only tile zero appends the masked row.
Each tile owns disjoint W columns and output elements. Scratch is not shared
across layers, graph shapes or execution streams.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _step_metadata_snapshot_kernel(a, count, indices, saved_a, saved_count,
                                   HV: tl.constexpr, K: tl.constexpr, STRIDE_IDX: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.load(indices + pid // HV * STRIDE_IDX).to(tl.int64)
    kk = tl.arange(0, K)
    valid = slot >= 0
    head = slot * HV + pid % HV
    old_a = tl.load(a + head * K + kk, mask=valid, other=0.0)
    old_count = tl.load(count + head, mask=valid, other=-1)
    tl.store(saved_a + pid * K + kk, old_a)
    tl.store(saved_count + pid, old_count)


class StepWorkspaceBank:
    def __init__(self):
        self.buffers = {}

    def get(self, layer, batch, heads, device, stream_id, *, capturing):
        if not 1 <= batch <= 8:
            raise ValueError("V tiling is only enabled for physical B<=8")
        key = (layer, batch, heads, str(device), stream_id)
        if key not in self.buffers:
            if capturing:
                raise RuntimeError("V-tiled step scratch not warmed for layer/shape/stream")
            self.buffers[key] = (
                torch.empty((batch, heads, 128), dtype=torch.float32, device=device),
                torch.empty((batch, heads), dtype=torch.int32, device=device),
            )
        return self.buffers[key]


def snapshot_step_metadata(fa, count, indices, workspace):
    batch = indices.numel(); heads = fa.shape[1]
    saved_a, saved_count = workspace
    if (fa.shape[-1] != 128 or fa.dtype != torch.float32
            or saved_a.shape != (batch, heads, 128) or saved_a.dtype != fa.dtype
            or saved_count.shape != (batch, heads) or saved_count.dtype != torch.int32
            or count.dtype != torch.int32
            or not all(x.is_contiguous() for x in (fa, count, saved_a, saved_count))):
        raise ValueError("invalid V-tiled recurrence state/snapshot layout")
    _step_metadata_snapshot_kernel[(batch * heads,)](
        fa, count, indices, saved_a, saved_count, HV=heads, K=128,
        STRIDE_IDX=indices.stride(0), num_warps=1)
