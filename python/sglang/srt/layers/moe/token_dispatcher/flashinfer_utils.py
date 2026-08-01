from typing import Any

import torch.distributed as dist

from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    from flashinfer.comm.mnnvl import CommBackend
else:

    class CommBackend:
        """
        Placeholder base class when flashinfer is not available
        """

        pass


class TorchDistributedCommBackend(CommBackend):
    """
    Use torch distributed instead of MPI to set up flashinfer MNNVL workspaces during initialization
    """

    def __init__(self, group: dist.ProcessGroup):
        self._group = group

    def Get_rank(self) -> int:
        return self._group.rank()

    def Get_size(self) -> int:
        return self._group.size()

    def allgather(self, data: Any):
        gathered = [None] * self.Get_size()
        dist.all_gather_object(gathered, data, group=self._group)
        return gathered

    def bcast(self, data, root: int = 0):
        obj_list = [data]
        # FlashInfer's CommBackend contract follows MPI and expresses root as
        # a rank local to this communicator.  torch.distributed's ``src`` is a
        # global rank even when ``group`` is provided, so translate it before
        # exchanging MNNVL handles.  Passing ``root`` through directly only
        # works accidentally for the first global process group.
        global_root = dist.get_global_rank(self._group, root)
        # broadcast_object_list mutates obj_list in-place
        dist.broadcast_object_list(obj_list, src=global_root, group=self._group)
        return obj_list[0]

    def Split(self, color: int, key: int):
        # No need to split, we already use the proper group
        return self

    def barrier(self):
        dist.barrier(group=self._group)
