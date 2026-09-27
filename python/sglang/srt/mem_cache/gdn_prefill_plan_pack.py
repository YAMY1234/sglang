"""Integer-only materialization of the already-decided host prefill plan."""
import torch
import triton
import triton.language as tl


@triton.jit
def _materialize(PACKED, FLAGS, INTS, B, REQUIRED_OFFSET, PREFIX_OFFSET,
                 HAS_REQUIRED: tl.constexpr, HAS_PREFIX: tl.constexpr,
                 BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    mask = i < B
    ring = tl.load(PACKED + i, mask=mask, other=0)
    dst = tl.load(PACKED + 2 * B + i, mask=mask, other=-1)
    tl.store(FLAGS + i, ring != 0, mask=mask)
    tl.store(INTS + B + i, dst.to(tl.int32), mask=mask)
    if HAS_REQUIRED:
        required = tl.load(PACKED + REQUIRED_OFFSET + i, mask=mask, other=0)
        tl.store(INTS + i, required.to(tl.int32), mask=mask)
    if HAS_PREFIX:
        prefix = tl.load(PACKED + PREFIX_OFFSET + i, mask=mask, other=0)
        tl.store(FLAGS + B + i, prefix != 0, mask=mask)


def materialize(packed, offsets, batch):
    """Return the original bool/int32 fields; keep index views and writes unchanged."""
    if batch <= 0 or packed.dtype != torch.int64 or not packed.is_contiguous():
        raise ValueError('plan packing requires contiguous int64 metadata and a nonempty batch')
    required = offsets['required'][1] > offsets['required'][0]
    prefix = offsets['use_prefix'][1] > offsets['use_prefix'][0]
    flags = torch.empty((2, batch), dtype=torch.bool, device=packed.device)
    integers = torch.empty((2, batch), dtype=torch.int32, device=packed.device)
    _materialize[(1,)](packed, flags, integers, batch, offsets['required'][0],
                       offsets['use_prefix'][0], required, prefix,
                       triton.next_power_of_2(batch), num_warps=4)
    return dict(use_ring=flags[0], ring_dst_i32=integers[1],
                required=integers[0] if required else None,
                use_prefix=flags[1] if prefix else None)
