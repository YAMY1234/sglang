"""Opt-in r16 expiry: separate square MGS from tiled factor projection.

No rank/iteration/threshold changes. Scratch belongs to one backend, layer,
batch shape and execution stream; allocation is forbidden inside capture.
The caller retains the existing post-order side-stream fork/join protocol.
"""
import torch
import triton
import triton.language as tl

from .gdn_factored import _mgs


@triton.jit
def _expiry_directions_kernel(w_ptr, cnt_ptr, indices, z_ptr, active_ptr,
                              HV: tl.constexpr, STRIDE_IDX: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.load(indices + (pid // HV) * STRIDE_IDX).to(tl.int64)
    # Clear every lane, including padding and non-expiry; never reuse old Z.
    tl.store(active_ptr + pid, 0)
    if slot < 0:
        return
    head = slot * HV + pid % HV
    if tl.load(cnt_ptr + head) < 32:
        return
    rr = tl.arange(0, 32)
    vv = tl.arange(0, 128)
    W = tl.load(w_ptr + head * 32 * 128 + rr[:, None] * 128 + vv[None, :])
    G = tl.dot(W, tl.trans(W), input_precision="ieee")
    # W's lifetime ends here. Keep the original 32x32 MGS workspace and ties.
    d = tl.sum(tl.where(rr[:, None] == rr[None, :], G, 0.), axis=1)
    better = (d[None, :] > d[:, None]) | ((d[None, :] == d[:, None]) & (rr[None, :] < rr[:, None]))
    rank = tl.sum(better.to(tl.int32), axis=1)
    Z = tl.where((rank[:, None] == rr[None, :]) & (rr[None, :] < 16), 1., 0.)
    for _ in range(3):
        Z = tl.dot(G, Z, input_precision="ieee")
        Z = _mgs(Z, rr, 16, 2, 1e-4)
    tl.store(z_ptr + pid * 32 * 16 + rr[:, None] * 16 + rr[None, :], Z,
             mask=rr[None, :] < 16)
    tl.store(active_ptr + pid, slot + 1)


@triton.jit
def _expiry_project_kernel(u_ptr, w_ptr, cnt_ptr, z_ptr, active_ptr,
                          HV: tl.constexpr):
    pid = tl.program_id(0)
    encoded_slot = tl.load(active_ptr + pid)
    if encoded_slot == 0:
        return
    head = (encoded_slot.to(tl.int64) - 1) * HV + pid % HV
    rr = tl.arange(0, 32)
    keep = tl.arange(0, 16)
    feature = tl.arange(0, 32)
    Z = tl.load(z_ptr + pid * 32 * 16 + rr[:, None] * 16 + keep[None, :])
    # One CTA owns all features of a head: no cross-CTA in-place alias hazard.
    # Dynamic loop prevents keeping all four U/W feature tiles live together.
    for block in range(4):
        kk = block * 32 + feature
        src = head * 32 * 128 + rr[:, None] * 128 + kk[None, :]
        dst = head * 32 * 128 + keep[:, None] * 128 + kk[None, :]
        U = tl.load(u_ptr + src)
        Un = tl.dot(tl.trans(Z), U, input_precision="ieee")
        tl.store(u_ptr + dst, Un)
        W = tl.load(w_ptr + src)
        Wn = tl.dot(tl.trans(Z), W, input_precision="ieee")
        tl.store(w_ptr + dst, Wn)
    # Publish count only after every projected factor tile has been written.
    tl.store(cnt_ptr + head, 16)


class ExpiryWorkspaceBank:
    """Separate live buffers per layer/shape/execution-stream, never a global slab."""
    def __init__(self):
        self.buffers = {}

    def get(self, layer, batch, heads, device, stream_id, *, capturing):
        key = (layer, batch, heads, str(device), stream_id)
        if key not in self.buffers:
            if capturing:
                raise RuntimeError("R6 expiry scratch was not warmed for this layer/shape/stream")
            self.buffers[key] = (
                torch.empty((batch, heads, 32, 16), dtype=torch.float32, device=device),
                torch.empty((batch, heads), dtype=torch.int32, device=device),
            )
        return self.buffers[key]


def truncate_two_stage(fu, fw, count, indices, workspace):
    batch = indices.numel(); heads = fu.shape[1]
    z, active = workspace
    if fu.shape[-2:] != (32, 128) or fw.shape != fu.shape:
        raise ValueError("R6 expiry requires r16/W16, K=V=128")
    if any(x.dtype != torch.float32 for x in (fu, fw, z)):
        raise ValueError("R6 expiry preserves fp32 factors")
    if z.shape != (batch, heads, 32, 16) or active.shape != (batch, heads):
        raise ValueError("R6 scratch shape differs from captured batch")
    if active.dtype != torch.int32 or not all(x.is_contiguous() for x in (fu, fw, count, z, active)):
        raise ValueError("R6 requires contiguous state/scratch and int32 activity")
    if batch == 0:
        return
    _expiry_directions_kernel[(batch * heads,)](
        fw, count, indices, z, active, HV=heads, STRIDE_IDX=indices.stride(0), num_warps=2)
    _expiry_project_kernel[(batch * heads,)](
        fu, fw, count, z, active, HV=heads, num_warps=4)
