"""Own and validate the communication storage referenced by a PD tail graph.

This validates addresses and ownership, not Lamport counter contents (which
must advance across eager calls and graph replays). It does not change fusion.
"""
from sglang.srt.runtime_context import get_exec, get_resources


NAMES = ("flashinfer_fusion_attn_tp_workspace", "flashinfer_fusion_moe_tp_workspace")


def addresses(workspace):
    if workspace.backend != "mnnvl":
        raise ValueError("PD tail communication guard requires the production MNNVL backend")
    # These are the actual arguments of the image's trtllm_mnnvl_allreduce_fusion.
    return (id(workspace.handle), int(workspace.mc_ptr),
            int(workspace.uc_ptrs_dev), int(workspace.uc_ptr_local),
            workspace.buffer_flags.data_ptr(), workspace.buffer_flags.numel())


class TailCommunicationLease:
    def __init__(self):
        self.enabled = get_exec().comm.flashinfer_allreduce_fusion_backend is not None
        self.entries = []
        if not self.enabled:
            return  # Deterministic recipe disables fusion; record this distinction.
        buffers = get_resources().buffers
        for name in NAMES:
            manager = buffers.get(name)
            if manager is None or not manager.initialized or manager.workspace is None:
                raise RuntimeError("PD tail capture requires pre-initialized communication workspace: " + name)
            workspace = manager.workspace
            self.entries.append((name, manager, workspace, addresses(workspace)))
        self.check()

    def check(self):
        buffers = get_resources().buffers
        for name, manager, workspace, expected in self.entries:
            if (buffers.get(name) is not manager or not manager.initialized
                    or manager.workspace is not workspace
                    or addresses(workspace) != expected):
                raise RuntimeError("PD tail graph communication storage changed: " + name)

    def receipt(self):
        self.check()
        return dict(fusion_enabled=self.enabled, workspaces=[dict(name=name,
            backend=workspace.backend, addresses=list(expected),
            world_size=manager.world_size, rank=manager.rank,
            max_token_num=manager.max_token_num, hidden_dim=manager.hidden_dim)
            for name, manager, workspace, expected in self.entries])
