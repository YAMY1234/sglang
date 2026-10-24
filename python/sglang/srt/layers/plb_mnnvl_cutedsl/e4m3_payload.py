"""PERF-LP: the BT route of FlashInfer's MNNVL CuTe DSL fusion with an E4M3 payload
(SGLANG_PERFLP_AR_FP8_PAYLOAD; device kernels in e4m3_kernels.py).

TensorRT-LLM's TRTLLM_MNNVL_TWOSHOT_FP8 on BT's two Lamport stages, for calls on the BT
route of at most ``max_tokens`` rows; other calls keep FlashInfer's BF16 kernels. The
payload has its own mailboxes (E4M3 rows, the FP32 -0 sentinel) and shares the BT
protocol's stage state, so the generations rotate once a call whichever path it takes.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from sglang.srt.layers.plb_mnnvl_cutedsl import LAMPORT_LEVELS

# TensorRT-LLM's conditions besides BF16 and the residual RMSNorm fusion.
E4M3_TP_SIZES = (2, 4, 8)
E4M3_HIDDEN_MULTIPLE = 1024


def e4m3_payload_unsupported(workspace) -> str | None:
    """Why the workspace's BT route cannot carry the E4M3 payload, or None."""
    from flashinfer.comm.mnnvl_cutedsl import ProtocolKind

    protocol = workspace._protocols.get(ProtocolKind.BT)
    if protocol is None:
        return "no BT route"
    if protocol.tp_size not in E4M3_TP_SIZES:
        return f"tp {protocol.tp_size} not in {E4M3_TP_SIZES}"
    if protocol.hidden_size % E4M3_HIDDEN_MULTIPLE:
        return f"hidden {protocol.hidden_size} not a multiple of {E4M3_HIDDEN_MULTIPLE}"
    if not (
        protocol.add_residual
        and protocol.write_residual_output
        and protocol.apply_rms_norm
    ):
        return "no residual RMSNorm fusion"
    return None


def compile_shared_publish(protocol, tuning, stream):
    """The all-reduce publish of ``tuning`` (a BTAllReduceTuning)."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Int32, Int64
    from cutlass.cute.runtime import make_fake_compact_tensor
    from flashinfer.comm.mnnvl_cutedsl.runtime import make_fake_dynamic_compact_tensor

    from sglang.srt.layers.plb_mnnvl_cutedsl.e4m3_kernels import (
        PlpSharedOnlyPublishE4M3DeviceKernel,
    )

    kernel = PlpSharedOnlyPublishE4M3DeviceKernel(
        hidden_size=protocol.hidden_size,
        tp_size=protocol.tp_size,
        rank=protocol.rank,
        local_capacity=protocol.local_capacity,
        threads=tuning.publish_threads,
        vectors_per_thread=tuning.publish_vectors_per_thread,
        enable_pdl=tuning.collective.enable_pdl,
    )
    return cute.compile(
        kernel,
        make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=protocol.hidden_size
        ),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int64, (protocol.tp_size,), assumed_align=8),
        Int64(0),
        Int32(protocol.capacity_m),
        stream,
    )


def compile_finalize_publish(protocol, tuning, stream):
    """The MoE finalize publish of ``tuning`` (a BTFinalizeTuning): 8 elements a thread
    whatever its elements_per_thread, at its thread count."""
    import cutlass.cute as cute
    from cutlass import BFloat16, Int32, Int64
    from cutlass.cute.runtime import make_fake_compact_tensor
    from flashinfer.comm.mnnvl_cutedsl.runtime import make_fake_dynamic_compact_tensor

    from sglang.srt.layers.plb_mnnvl_cutedsl.e4m3_kernels import (
        PlpVectorFinalizeUnicastE4M3DeviceKernel,
    )

    kernel = PlpVectorFinalizeUnicastE4M3DeviceKernel(
        hidden_size=protocol.hidden_size,
        top_k=protocol.top_k,
        tp_size=protocol.tp_size,
        rank=protocol.rank,
        local_capacity=protocol.local_capacity,
        threads=tuning.threads,
        routed_scaling_factor=protocol.routed_scaling_factor,
        include_shared_expert=protocol.include_shared_expert,
        load_shared_expert_before_pdl=tuning.load_shared_expert_before_pdl,
        enable_pdl=tuning.collective.enable_pdl,
        prefetch_group=tuning.prefetch_group,
    )
    activation = make_fake_dynamic_compact_tensor(
        BFloat16, alignment=16, divisibility=protocol.hidden_size
    )
    return cute.compile(
        kernel,
        activation,
        make_fake_dynamic_compact_tensor(
            BFloat16, alignment=2, divisibility=protocol.top_k
        ),
        make_fake_dynamic_compact_tensor(
            Int32, alignment=4, divisibility=protocol.top_k
        ),
        activation,
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int64, (protocol.tp_size,), assumed_align=8),
        Int64(0),
        Int32(protocol.capacity_m),
        stream,
    )


def compile_reduce(protocol, collective, stream, *, level: int):
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_compact_tensor

    from sglang.srt.layers.plb_mnnvl_cutedsl.e4m3_kernels import (
        PlpOwnerReduceMulticastE4M3DeviceKernel,
    )

    kernel = PlpOwnerReduceMulticastE4M3DeviceKernel(
        hidden_size=protocol.hidden_size,
        tp_size=protocol.tp_size,
        rank=protocol.rank,
        capacity_m=protocol.capacity_m,
        local_capacity=protocol.local_capacity,
        threads=collective.reduction_threads,
        enable_pdl=collective.enable_pdl,
        early_trigger=LAMPORT_LEVELS[level][0],
    )
    return cute.compile(
        kernel,
        Int64(0),
        Int64(0),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int64(0),
        Int32(protocol.capacity_m),
        stream,
    )


def compile_norm(protocol, collective, stream, *, level: int, fp8_out: bool):
    import cutlass.cute as cute
    from cutlass import BFloat16, Int32, Int64
    from cutlass.cute.runtime import make_fake_compact_tensor
    from flashinfer.comm.mnnvl_cutedsl.runtime import make_fake_dynamic_compact_tensor

    from sglang.srt.layers.plb_mnnvl_cutedsl.e4m3_kernels import (
        PlpMaterializeRMSNormE4M3DeviceKernel,
        PlpMaterializeRMSNormE4M3Fp8DeviceKernel,
    )

    _, early_dependents, poll_before_wait = LAMPORT_LEVELS[level]
    kernel_type = (
        PlpMaterializeRMSNormE4M3Fp8DeviceKernel
        if fp8_out
        else PlpMaterializeRMSNormE4M3DeviceKernel
    )
    kernel = kernel_type(
        hidden_size=protocol.hidden_size,
        capacity_m=protocol.capacity_m,
        threads=collective.rms_threads,
        rms_epsilon=protocol.rms_epsilon,
        weight_bias=protocol.weight_bias,
        write_residual_output=protocol.write_residual_output,
        enable_pdl=collective.enable_pdl,
        early_dependents=early_dependents,
        poll_before_wait=poll_before_wait,
    )
    rows = make_fake_dynamic_compact_tensor(
        BFloat16, alignment=16, divisibility=protocol.hidden_size
    )
    addresses = (Int64(0), Int64(0)) if fp8_out else ()
    return cute.compile(
        kernel,
        Int64(0),
        rows,
        rows,
        rows,
        make_fake_compact_tensor(BFloat16, (protocol.hidden_size,), assumed_align=16),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        *addresses,
        Int32(protocol.capacity_m),
        stream,
    )


class E4M3Payload:
    """The BT route's E4M3 kernels and mailboxes for ``workspace`` (an
    MNNVLCuteDSLAllReduceFusionWorkspace). Collective: every rank constructs it.

    ``lamport``: a LAMPORT_LEVELS key, as the BF16 tail's. ``fp8_out``: also the MoE
    finalize norm that writes the next projection's static E4M3 input.
    """

    def __init__(
        self, workspace, *, max_tokens: int, lamport: int, fp8_out: bool
    ) -> None:
        from flashinfer.comm.mnnvl_cutedsl import ProtocolKind
        from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
            LAMPORT_GENERATIONS,
        )
        from flashinfer.comm.mnnvl_cutedsl.runtime import current_cu_stream
        from flashinfer.comm.mnnvl_cutedsl.symmetric_buffer import SymmetricBuffer

        from sglang.srt.layers.plb_mnnvl_cutedsl.e4m3_kernels import (
            E4M3_SENTINEL,
            e4m3_row_bytes,
        )

        self._bt_kind = ProtocolKind.BT
        self.protocol = protocol = workspace._protocols[ProtocolKind.BT]
        self.profile = workspace.profile
        self.max_tokens = int(max_tokens)
        self.lamport = lamport
        self.hidden_size = protocol.hidden_size

        stream = current_cu_stream()
        self.all_reduce_publish = {
            tuning: compile_shared_publish(protocol, tuning, stream)
            for tuning in protocol.all_reduce_kernels
        }
        self.finalize_publish = {
            tuning: compile_finalize_publish(protocol, tuning, stream)
            for tuning in protocol.finalize_kernels
        }
        collectives = dict.fromkeys(
            tuning.collective
            for tuning in (*protocol.all_reduce_kernels, *protocol.finalize_kernels)
        )
        self.tails = {
            collective: (
                compile_reduce(protocol, collective, stream, level=lamport),
                compile_norm(
                    protocol, collective, stream, level=lamport, fp8_out=False
                ),
            )
            for collective in collectives
        }
        self.fp8_norms = (
            {
                collective: compile_norm(
                    protocol, collective, stream, level=lamport, fp8_out=True
                )
                for collective in dict.fromkeys(
                    tuning.collective for tuning in protocol.finalize_kernels
                )
            }
            if fp8_out
            else {}
        )

        group = workspace.group
        device = torch.device("cuda", torch.cuda.current_device())
        row_bytes = e4m3_row_bytes(protocol.hidden_size)
        # [generation][source rank][local token] rows; this rank's own rows stay BF16
        # in own_rows [generation][local token].
        self.contribution = SymmetricBuffer.allocate(
            (LAMPORT_GENERATIONS, protocol.tp_size, protocol.local_capacity, row_bytes),
            torch.uint8,
            device,
            group,
            materialize_peer_addresses=True,
        )
        self.prenorm = SymmetricBuffer.allocate(
            (LAMPORT_GENERATIONS, protocol.capacity_m, row_bytes),
            torch.uint8,
            device,
            group,
            require_multicast=True,
        )
        sentinel = E4M3_SENTINEL - (1 << 32)
        self.contribution.tensor.view(torch.int32).fill_(sentinel)
        self.prenorm.tensor.view(torch.int32).fill_(sentinel)
        self.own_rows = torch.empty(
            (LAMPORT_GENERATIONS, protocol.local_capacity, protocol.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        # No peer may write a mailbox before every rank filled its sentinels.
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def takes_all_reduce(self, m: int) -> bool:
        return (
            m <= self.max_tokens
            and self.profile.all_reduce_routes.select(m).protocol is self._bt_kind
        )

    def takes_finalize(self, m: int) -> bool:
        return (
            m <= self.max_tokens
            and self.profile.finalize_routes.select(m).protocol is self._bt_kind
        )

    def describe(self) -> str:
        def rows(takes) -> str:
            ms = [m for m in range(1, self.protocol.capacity_m + 1) if takes(m)]
            return f"{ms[0]}..{ms[-1]}" if ms else "none"

        return (
            f"all-reduce m {rows(self.takes_all_reduce)}, "
            f"MoE finalize m {rows(self.takes_finalize)}"
        )

    def all_reduce(
        self,
        *,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor,
        gamma: torch.Tensor,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor,
        m: int,
    ) -> None:
        from cutlass import Int32, Int64
        from flashinfer.comm.mnnvl_cutedsl.runtime import (
            current_cu_stream,
            to_cute,
            to_cute_dynamic,
        )

        tuning = self.profile.all_reduce_routes.select(m).preset
        self._validate(m)
        stream = current_cu_stream()
        self.all_reduce_publish[tuning](
            to_cute_dynamic(
                local_contribution.flatten(), 16, divisibility=self.hidden_size
            ),
            to_cute(self.protocol.state.stage_state, 4),
            to_cute(self.contribution.peer_addresses, 8),
            Int64(self.own_rows.data_ptr()),
            Int32(m),
            stream,
        )
        self._tail(
            tuning.collective,
            residual_source=residual_source,
            gamma=gamma,
            norm_output=norm_output,
            residual_output=residual_output,
            m=m,
            stream=stream,
        )

    def finalize(
        self,
        *,
        routed_output: torch.Tensor,
        expert_weights: torch.Tensor,
        permuted_indices: torch.Tensor,
        shared_output: torch.Tensor | None,
        residual_source: torch.Tensor,
        gamma: torch.Tensor,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor,
        m: int,
        fp8_output: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        """The MoE finalize; with ``fp8_output`` the norm also writes its rows as E4M3
        under the static ``input_scale`` (SGLANG_PLB_AR_FP8_OUT)."""
        from cutlass import Int32, Int64
        from flashinfer.comm.mnnvl_cutedsl.runtime import (
            current_cu_stream,
            to_cute,
            to_cute_dynamic,
        )

        if self.protocol.include_shared_expert != (shared_output is not None):
            raise ValueError("shared_output does not match the compiled workspace")
        tuning = self.profile.finalize_routes.select(m).preset
        self._validate(m)
        hidden = self.hidden_size
        top_k = self.protocol.top_k
        shared_arg = shared_output if shared_output is not None else norm_output
        stream = current_cu_stream()
        self.finalize_publish[tuning](
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=hidden),
            to_cute(self.protocol.state.stage_state, 4),
            to_cute(self.contribution.peer_addresses, 8),
            Int64(self.own_rows.data_ptr()),
            Int32(m),
            stream,
        )
        self._tail(
            tuning.collective,
            residual_source=residual_source,
            gamma=gamma,
            norm_output=norm_output,
            residual_output=residual_output,
            m=m,
            stream=stream,
            fp8_output=fp8_output,
            input_scale=input_scale,
        )

    def _validate(self, m: int) -> None:
        if not 1 <= m <= self.protocol.capacity_m:
            raise ValueError(f"m must be in [1, {self.protocol.capacity_m}]")

    def _tail(
        self,
        collective,
        *,
        residual_source: torch.Tensor,
        gamma: torch.Tensor,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor,
        m: int,
        stream,
        fp8_output: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        from cutlass import Int32, Int64
        from flashinfer.comm.mnnvl_cutedsl.runtime import to_cute, to_cute_dynamic

        hidden = self.hidden_size
        reduce, norm = self.tails[collective]
        reduce(
            Int64(self.contribution.tensor.data_ptr()),
            Int64(self.own_rows.data_ptr()),
            to_cute(self.protocol.state.stage_state, 4),
            Int64(self.prenorm.multicast_address),
            Int32(m),
            stream,
        )
        args = (
            Int64(self.prenorm.tensor.data_ptr()),
            to_cute_dynamic(residual_source.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(residual_output.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=hidden),
            to_cute(gamma, 16),
            to_cute(self.protocol.state.stage_state, 4),
        )
        if fp8_output is None:
            norm(*args, Int32(m), stream)
        else:
            self.fp8_norms[collective](
                *args,
                Int64(fp8_output.data_ptr()),
                Int64(input_scale.data_ptr()),
                Int32(m),
                stream,
            )
