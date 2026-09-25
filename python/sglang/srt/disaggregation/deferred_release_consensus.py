"""Keep aborted-transfer destination ownership identical across TP ranks.

These collectives run only for failed transfers or outstanding deferred
releases. The ordinary successful transfer/decode path gains no collective.
"""

import torch
import torch.distributed as dist


def agree_deferred_holds(local_holds, group):
    """If any rank may still receive writes, every rank retains its slot."""
    states = torch.tensor(local_holds, dtype=torch.uint8, device="cpu")
    dist.all_reduce(states, op=dist.ReduceOp.MAX, group=group)
    return states.tolist()


def agree_deferred_releases(local_ready, group):
    """Release the same slots in the same iteration, after all ranks are safe."""
    states = torch.tensor(local_ready, dtype=torch.uint8, device="cpu")
    dist.all_reduce(states, op=dist.ReduceOp.MIN, group=group)
    return states.tolist()
