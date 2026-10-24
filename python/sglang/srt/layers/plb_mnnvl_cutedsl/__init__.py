"""PERF-LB re-compiles of FlashInfer's MNNVL CuTe DSL BT tail (bt_tail.py).

A ``BTTailPatch`` swaps the reduce and MaterializeRMSNorm kernels of a built
workspace's BT protocol before graph capture (SGLANG_PLB_MAT_LAMPORT) and compiles
the E4M3-writing norm for the MoE finalize (SGLANG_PLB_AR_FP8_OUT). FlashInfer's
publish kernels and the protocol state are used as they are.
"""

from __future__ import annotations

import torch

# SGLANG_PLB_MAT_LAMPORT level -> (reduce: early trigger; norm: early dependents,
# poll before the grid-dependency wait)
LAMPORT_LEVELS = {
    0: (False, False, False),
    1: (False, True, True),
    2: (True, True, True),
}


def _quantizes_e4m3fn() -> bool:
    """Whether static_quant_fp8 makes float8_e4m3fn with the +-448 clamp the kernel hard-codes."""
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max
    from sglang.srt.layers.plb_mnnvl_cutedsl.bt_tail import E4M3_MAX

    return fp8_dtype == torch.float8_e4m3fn and fp8_max == E4M3_MAX


def _compile_reduce(protocol, collective):
    """``BTProtocol._compile_tail``'s reduce compile, with the early dependent launch."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Int32, Int64
    from cutlass.cute.runtime import make_fake_compact_tensor
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
        LAMPORT_GENERATIONS,
    )
    from flashinfer.comm.mnnvl_cutedsl.runtime import (
        current_cu_stream,
        make_fake_dynamic_compact_tensor,
    )

    from sglang.srt.layers.plb_mnnvl_cutedsl.bt_tail import (
        PlbOwnerReduceMulticastDeviceKernel,
    )

    kernel = PlbOwnerReduceMulticastDeviceKernel(
        hidden_size=protocol.hidden_size,
        tp_size=protocol.tp_size,
        rank=protocol.rank,
        capacity_m=protocol.capacity_m,
        local_capacity=protocol.local_capacity,
        threads=collective.reduction_threads,
        add_residual=protocol.add_residual,
        enable_pdl=collective.enable_pdl,
        early_trigger=True,
    )
    elements = (
        LAMPORT_GENERATIONS
        * protocol.tp_size
        * protocol.local_capacity
        * protocol.hidden_size
    )
    return cute.compile(
        kernel,
        make_fake_compact_tensor(BFloat16, (elements,), assumed_align=16),
        make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=protocol.hidden_size
        ),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int64(0),
        Int32(protocol.capacity_m),
        current_cu_stream(),
    )


def _compile_materialize(protocol, collective, *, level: int, fp8_out: bool):
    """``BTProtocol._compile_tail``'s rms_norm compile with this module's kernel."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Int32, Int64
    from cutlass.cute.runtime import make_fake_compact_tensor
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
        LAMPORT_GENERATIONS,
    )
    from flashinfer.comm.mnnvl_cutedsl.runtime import (
        current_cu_stream,
        make_fake_dynamic_compact_tensor,
    )

    from sglang.srt.layers.plb_mnnvl_cutedsl.bt_tail import (
        PlbMaterializeRMSNormDeviceKernel,
        PlbMaterializeRMSNormFp8DeviceKernel,
    )

    hidden = protocol.hidden_size
    _, early_dependents, poll_before_wait = LAMPORT_LEVELS[level]
    kwargs = dict(
        hidden_size=hidden,
        capacity_m=protocol.capacity_m,
        threads=collective.rms_threads,
        rms_epsilon=protocol.rms_epsilon,
        weight_bias=protocol.weight_bias,
        write_residual_output=protocol.write_residual_output,
        enable_pdl=collective.enable_pdl,
        apply_rms_norm=protocol.apply_rms_norm,
        early_dependents=early_dependents,
        poll_before_wait=poll_before_wait,
    )
    kernel = (
        PlbMaterializeRMSNormFp8DeviceKernel(**kwargs)
        if fp8_out
        else PlbMaterializeRMSNormDeviceKernel(**kwargs)
    )
    rows = make_fake_dynamic_compact_tensor(
        BFloat16, alignment=16, divisibility=hidden
    )
    addresses = (Int64(0), Int64(0)) if fp8_out else ()
    return cute.compile(
        kernel,
        make_fake_compact_tensor(
            BFloat16,
            (LAMPORT_GENERATIONS * protocol.capacity_m * hidden,),
            assumed_align=16,
        ),
        rows,
        rows,
        make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        *addresses,
        Int32(protocol.capacity_m),
        current_cu_stream(),
    )


class BTTailPatch:
    """The BT kernels of ``workspace`` with this module's tail.

    ``lamport``: a LAMPORT_LEVELS key (0: FlashInfer's order). ``install()`` puts the
    re-compiled finalize and all-reduce kernels in the protocol's tables;
    ``finalize_fp8`` runs the finalize with the E4M3 norm (``fp8_out``).
    """

    def __init__(self, workspace, *, lamport: int, fp8_out: bool) -> None:
        from flashinfer.comm.mnnvl_cutedsl import ProtocolKind
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt import protocol as bt

        self.workspace = workspace
        self.protocol = workspace._protocols.get(ProtocolKind.BT)
        self._bt_kind = ProtocolKind.BT
        self.finalize_kernels = {}
        self.all_reduce_kernels = {}
        # collective tuning -> (reduce, E4M3 norm) of the MoE finalize
        self.fp8_tails = {}
        if self.protocol is None:
            return
        protocol = self.protocol
        tails = {}

        def tail(old_tail, collective):
            if collective not in tails:
                tails[collective] = bt._CompiledTail(
                    reduce=(
                        _compile_reduce(protocol, collective)
                        if LAMPORT_LEVELS[lamport][0]
                        else old_tail.reduce
                    ),
                    rms_norm=_compile_materialize(
                        protocol, collective, level=lamport, fp8_out=False
                    ),
                )
            return tails[collective]

        if lamport:
            for tuning, kernel in protocol.finalize_kernels.items():
                self.finalize_kernels[tuning] = bt.FinalizeAllReduceRMSNormBTKernel(
                    compiled=bt._CompiledFinalize(
                        publish=kernel._compiled.publish,
                        tail=tail(kernel._compiled.tail, tuning.collective),
                    ),
                    **protocol._path_kwargs(),
                )
            for tuning, kernel in protocol.all_reduce_kernels.items():
                self.all_reduce_kernels[tuning] = bt.AllReduceRMSNormBTKernel(
                    compiled=bt._CompiledAllReduce(
                        publish=kernel._compiled.publish,
                        tail=tail(kernel._compiled.tail, tuning.collective),
                    ),
                    **protocol._path_kwargs(),
                )
        if fp8_out and protocol.apply_rms_norm and _quantizes_e4m3fn():
            for tuning, kernel in protocol.finalize_kernels.items():
                collective = tuning.collective
                if collective not in self.fp8_tails:
                    reduce = (
                        tail(kernel._compiled.tail, collective).reduce
                        if lamport
                        else kernel._compiled.tail.reduce
                    )
                    self.fp8_tails[collective] = (
                        reduce,
                        _compile_materialize(
                            protocol, collective, level=lamport, fp8_out=True
                        ),
                    )

    def install(self) -> None:
        if self.protocol is None:
            return
        self.protocol.finalize_kernels.update(self.finalize_kernels)
        self.protocol.all_reduce_kernels.update(self.all_reduce_kernels)

    def takes_fp8(self, m: int) -> bool:
        """Whether the MoE finalize of m rows runs on BT, which has the E4M3 tail."""
        return (
            bool(self.fp8_tails)
            and self.workspace.profile.finalize_routes.select(m).protocol
            is self._bt_kind
        )

    def finalize_fp8(
        self,
        *,
        routed_output: torch.Tensor,
        expert_weights: torch.Tensor,
        permuted_indices: torch.Tensor,
        shared_output: torch.Tensor | None,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor,
        fp8_output: torch.Tensor,
        input_scale: torch.Tensor,
        m: int,
    ) -> None:
        """``FinalizeAllReduceRMSNormBTKernel.__call__`` with the E4M3 tail: also writes
        ``fp8_output`` (m x hidden E4M3 bytes) under ``input_scale`` (an fp32 scalar)."""
        from cutlass import Int32, Int64
        from flashinfer.comm.mnnvl_cutedsl.runtime import (
            current_cu_stream,
            to_cute,
            to_cute_dynamic,
        )

        # As _mnnvl_cutedsl_allreduce_fusion checks before FlashInfer's own launch.
        if self.protocol.include_shared_expert != (shared_output is not None):
            raise ValueError("shared_output does not match the compiled workspace")
        if self.protocol.add_residual != (residual_source is not None):
            raise ValueError("residual_source does not match the compiled workspace")
        target = self.workspace.profile.finalize_routes.select(m)
        kernel = self.protocol.finalize_kernels[target.preset]
        reduce, rms_norm = self.fp8_tails[target.preset.collective]
        state = self.protocol.state
        kernel._validate_state(state, m)
        hidden = self.protocol.hidden_size
        top_k = self.protocol.top_k
        shared_arg = shared_output if shared_output is not None else norm_output
        residual_arg = residual_source if residual_source is not None else norm_output
        stream = current_cu_stream()
        kernel._compiled.publish(
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=hidden),
            to_cute(state.stage_state, 4),
            to_cute(state.contribution_mailbox.peer_addresses, 8),
            Int32(m),
            stream,
        )
        reduce(
            to_cute(state.contribution_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=hidden),
            to_cute(state.stage_state, 4),
            Int64(state.prenorm_mailbox.multicast_address),
            Int32(m),
            stream,
        )
        rms_norm(
            to_cute(state.prenorm_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_output.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=hidden),
            to_cute(gamma, 16),
            to_cute(state.stage_state, 4),
            Int64(fp8_output.data_ptr()),
            Int64(input_scale.data_ptr()),
            Int32(m),
            stream,
        )
