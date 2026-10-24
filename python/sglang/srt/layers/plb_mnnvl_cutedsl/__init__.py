"""PERF-LB re-compiles of FlashInfer's MNNVL CuTe DSL BT tail (bt_tail.py).

A ``BTTailPatch`` builds, for each collective tuning of a built workspace's BT
protocol, a ``PlbTail`` (this module's reduce + MaterializeRMSNorm) and, before
graph capture, swaps it into the protocol's finalize and all-reduce kernels
(SGLANG_PLB_MAT_LAMPORT, SGLANG_PLB_MAT_RESET_IN_OR, SGLANG_PLB_MAT_RESET_EARLY,
SGLANG_PLB_MAT_ROWS2_MIN); the
MoE finalize can also write E4M3 rows through it (SGLANG_PLB_AR_FP8_OUT).
FlashInfer's publish kernels and the protocol state are used as they are.
"""

from __future__ import annotations

import functools

import torch

# SGLANG_PLB_MAT_LAMPORT level -> (reduce: early trigger; norm: early dependents,
# poll before the grid-dependency wait)
LAMPORT_LEVELS = {
    0: (False, False, False),
    1: (False, True, True),
    2: (True, True, True),
}

# protocol id -> int32[LAMPORT_GENERATIONS]: rows each generation's last call wrote,
# shared by every tail on that protocol (a reset of the previous generation reads it).
_ROWS_STATE: dict[int, torch.Tensor] = {}


def _quantizes_e4m3fn() -> bool:
    """Whether static_quant_fp8 makes float8_e4m3fn with the +-448 clamp the kernel hard-codes."""
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max
    from sglang.srt.layers.plb_mnnvl_cutedsl.bt_tail import E4M3_MAX

    return fp8_dtype == torch.float8_e4m3fn and fp8_max == E4M3_MAX


def rows_state_of(protocol) -> torch.Tensor:
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
        LAMPORT_GENERATIONS,
    )

    if id(protocol) not in _ROWS_STATE:
        _ROWS_STATE[id(protocol)] = torch.zeros(
            LAMPORT_GENERATIONS,
            dtype=torch.int32,
            device=protocol.state.stage_state.device,
        )
    return _ROWS_STATE[id(protocol)]


def _compile_reduce(
    protocol,
    collective,
    *,
    early_trigger: bool,
    reset_previous: bool,
    cta_stage: bool = False,
):
    """``BTProtocol._compile_tail``'s reduce compile with this module's kernel."""
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
        early_trigger=early_trigger,
        reset_previous=reset_previous,
        cta_stage=cta_stage,
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
        Int64(0),
        Int64(0),
        Int32(protocol.capacity_m),
        current_cu_stream(),
    )


def _compile_materialize(
    protocol,
    collective,
    *,
    level: int,
    reset_own: bool = True,
    reset_previous: bool = False,
    rows: int = 1,
    fp8_out: bool = False,
    cta_stage: bool = False,
):
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
    )

    hidden = protocol.hidden_size
    _, early_dependents, poll_before_wait = LAMPORT_LEVELS[level]
    kernel = PlbMaterializeRMSNormDeviceKernel(
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
        reset_own=reset_own,
        reset_previous=reset_previous,
        rows=rows,
        fp8_out=fp8_out,
        cta_stage=cta_stage,
    )
    rows_tensor = make_fake_dynamic_compact_tensor(
        BFloat16, alignment=16, divisibility=hidden
    )
    return cute.compile(
        kernel,
        make_fake_compact_tensor(
            BFloat16,
            (LAMPORT_GENERATIONS * protocol.capacity_m * hidden,),
            assumed_align=16,
        ),
        rows_tensor,
        rows_tensor,
        make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(protocol.capacity_m),
        current_cu_stream(),
    )


class PlbTail:
    """The reduce and norm launches of one collective tuning (``_BTPath._launch_tail``)."""

    def __init__(
        self,
        protocol,
        collective,
        *,
        level: int,
        reset_in_or: bool,
        reset_early: bool,
        rows2_min: int,
        fp8_out: bool,
    ) -> None:
        if reset_in_or and reset_early:
            raise ValueError("the previous prenorm rows are reset by the reduce or the norm kernel")
        # Their extra stage / row words would otherwise be loaded by every warp (one L2 line).
        cta_stage = reset_in_or or reset_early
        self.protocol = protocol
        self.rows2_min = rows2_min
        self.rows_state = rows_state_of(protocol)
        self.reduce = _compile_reduce(
            protocol,
            collective,
            early_trigger=LAMPORT_LEVELS[level][0],
            reset_previous=reset_in_or,
            cta_stage=cta_stage,
        )
        widths = (1, 2) if rows2_min else (1,)
        self.rms_norm = {
            rows: _compile_materialize(
                protocol,
                collective,
                level=level,
                reset_own=not (reset_in_or or reset_early),
                reset_previous=reset_early,
                rows=rows,
                fp8_out=fp8_out,
                cta_stage=cta_stage,
            )
            for rows in widths
        }

    def launch(
        self,
        *,
        state,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor | None,
        m: int,
        fp8_output: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        from cutlass import Int32, Int64
        from flashinfer.comm.mnnvl_cutedsl.runtime import (
            current_cu_stream,
            to_cute,
            to_cute_dynamic,
        )

        hidden = self.protocol.hidden_size
        residual_arg = residual_source if residual_source is not None else norm_output
        residual_output_arg = (
            residual_output if residual_output is not None else norm_output
        )
        stream = current_cu_stream()
        self.reduce(
            to_cute(state.contribution_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=hidden),
            to_cute(state.stage_state, 4),
            Int64(state.prenorm_mailbox.multicast_address),
            Int64(state.prenorm_mailbox.tensor.data_ptr()),
            Int64(self.rows_state.data_ptr()),
            Int32(m),
            stream,
        )
        rows = 2 if self.rows2_min and m > self.rows2_min else 1
        self.rms_norm[rows](
            to_cute(state.prenorm_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_output_arg.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=hidden),
            to_cute(gamma, 16),
            to_cute(state.stage_state, 4),
            Int64(0 if fp8_output is None else fp8_output.data_ptr()),
            Int64(0 if input_scale is None else input_scale.data_ptr()),
            Int64(self.rows_state.data_ptr()),
            Int32(m),
            stream,
        )


def _launch_plb_tail(
    self, tail, residual_source, gamma, state, norm_output, residual_output, m
):
    tail.launch(
        state=state,
        residual_source=residual_source,
        gamma=gamma,
        norm_output=norm_output,
        residual_output=residual_output,
        m=m,
    )


@functools.cache
def _kernel_classes():
    """FlashInfer's BT finalize / all-reduce kernels launching a ``PlbTail``."""
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt import protocol as bt

    return (
        type(
            "PlbFinalizeAllReduceRMSNormBTKernel",
            (bt.FinalizeAllReduceRMSNormBTKernel,),
            {"_launch_tail": _launch_plb_tail},
        ),
        type(
            "PlbAllReduceRMSNormBTKernel",
            (bt.AllReduceRMSNormBTKernel,),
            {"_launch_tail": _launch_plb_tail},
        ),
    )


class BTTailPatch:
    """The BT kernels of ``workspace`` on this module's tails.

    ``lamport``: a LAMPORT_LEVELS key. ``reset_in_or``: the reduce kernel resets the
    previous call's prenorm rows instead of the norm kernel its own; ``reset_early``:
    the norm kernel resets them before its grid-dependency wait. ``rows2_min``:
    the norm kernel takes 2 rows a CTA above this many rows (0: never). ``install()``
    puts the finalize and all-reduce kernels in the protocol's tables (all of them
    for a reset of the previous rows, since every BT call must then reset that way);
    ``finalize_fp8`` runs the finalize with the E4M3 norm (``fp8_out``).
    """

    def __init__(
        self,
        workspace,
        *,
        lamport: int,
        fp8_out: bool,
        reset_in_or: bool = False,
        reset_early: bool = False,
        rows2_min: int = 0,
    ) -> None:
        from flashinfer.comm.mnnvl_cutedsl import ProtocolKind
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt import protocol as bt

        self.workspace = workspace
        self.protocol = workspace._protocols.get(ProtocolKind.BT)
        self._bt_kind = ProtocolKind.BT
        self.finalize_kernels = {}
        self.all_reduce_kernels = {}
        # collective tuning -> PlbTail with the E4M3 norm (the MoE finalize's)
        self.fp8_tails = {}
        if self.protocol is None:
            return
        protocol = self.protocol
        options = dict(
            level=lamport,
            reset_in_or=reset_in_or,
            reset_early=reset_early,
            rows2_min=rows2_min,
        )
        tails = {}

        def tail(collective):
            if collective not in tails:
                tails[collective] = PlbTail(
                    protocol, collective, **options, fp8_out=False
                )
            return tails[collective]

        if lamport or reset_in_or or reset_early or rows2_min:
            finalize_type, all_reduce_type = _kernel_classes()
            for tuning, kernel in protocol.finalize_kernels.items():
                self.finalize_kernels[tuning] = finalize_type(
                    compiled=bt._CompiledFinalize(
                        publish=kernel._compiled.publish, tail=tail(tuning.collective)
                    ),
                    **protocol._path_kwargs(),
                )
            for tuning, kernel in protocol.all_reduce_kernels.items():
                self.all_reduce_kernels[tuning] = all_reduce_type(
                    compiled=bt._CompiledAllReduce(
                        publish=kernel._compiled.publish, tail=tail(tuning.collective)
                    ),
                    **protocol._path_kwargs(),
                )
        if fp8_out and protocol.apply_rms_norm and _quantizes_e4m3fn():
            for tuning in protocol.finalize_kernels:
                if tuning.collective not in self.fp8_tails:
                    self.fp8_tails[tuning.collective] = PlbTail(
                        protocol, tuning.collective, **options, fp8_out=True
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
        from cutlass import Int32
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
        state = self.protocol.state
        kernel._validate_state(state, m)
        hidden = self.protocol.hidden_size
        top_k = self.protocol.top_k
        shared_arg = shared_output if shared_output is not None else norm_output
        kernel._compiled.publish(
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=hidden),
            to_cute(state.stage_state, 4),
            to_cute(state.contribution_mailbox.peer_addresses, 8),
            Int32(m),
            current_cu_stream(),
        )
        self.fp8_tails[target.preset.collective].launch(
            state=state,
            residual_source=residual_source,
            gamma=gamma,
            norm_output=norm_output,
            residual_output=residual_output,
            m=m,
            fp8_output=fp8_output,
            input_scale=input_scale,
        )
