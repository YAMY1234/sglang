"""K3 SP-MoE: o_proj GEMM overlapped with its attention-TP reduce-scatter.

Under SP-MoE the attention output projection's TP-partial [T, H] output is
reduce-scattered onto each rank's T / attn_tp rows at the next stage boundary,
after the whole GEMM. Here o_proj computes its output in column chunks instead;
each chunk's reduce-scatter runs on a side stream while the next chunk's GEMM
runs, and o_proj returns this rank's reduced rows. The FFN read's fusion
(``presummed_fusion``) recognizes that output and skips the boundary's
reduce-scatter. Every output element is still the sum of the same TP partials,
each computed with the same K reduction; only NCCL's choice of algorithm for the
smaller messages can reorder that sum.

SGLANG_K3_SP_RS_OVERLAP_CHUNKS (0: off) and _MIN_TOKENS gate it; it runs only
eagerly and only where NCCL, not the K3 SP collective, would reduce the batch.

With SGLANG_K3_SP_GEMM_RS_MIN_TOKENS (0: off), eager batches of at least that
many rows instead run o_proj as one CUTLASS GEMM whose epilogue reduce-scatters
through NVLink multicast (``sp_collective_gemm.matmul_reduce_scatter``), ahead
of both the chunked overlap and the K3 SP collective's reduce-scatter. The sum
is taken in fp32 by the switch, so its bf16 rounding can differ from NCCL's.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.contracts import ReadoutFusion
from sglang.srt.layers.layer_boundary.layout import SumGroup

logger = logging.getLogger(__name__)

_LOGGED = False
_GEMM_RS_LOGGED = False
# The o_proj output that is already this rank's reduced rows, until the FFN
# read takes it: (data_ptr, shape).
_PENDING: Optional[tuple[int, tuple]] = None
_STREAMS: dict[int, torch.cuda.Stream] = {}


def num_chunks(hidden_size: int) -> int:
    """The configured column chunk count, or 0 when off or not a clean split."""
    chunks = envs.SGLANG_K3_SP_RS_OVERLAP_CHUNKS.get()
    if chunks < 2 or hidden_size % chunks or (hidden_size // chunks) % 64:
        return 0
    return chunks


def installs(hidden_size: int) -> bool:
    """Whether SP-MoE o_proj takes this module's wrapper (either mode)."""
    return num_chunks(hidden_size) > 0 or envs.SGLANG_K3_SP_GEMM_RS_MIN_TOKENS.get() > 0


def _eager_bf16(o_proj, x) -> bool:
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.runtime_context import get_forward

    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and o_proj.weight.dtype == x.dtype
        and o_proj.bias is None
        and getattr(o_proj.quant_method, "apply_into", None) is not None
        and not get_forward().sp_active
        and not get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
        and not torch.cuda.is_current_stream_capturing()
    )


def takes_gemm_rs(o_proj, x) -> bool:
    """Whether o_proj's GEMM and attention-TP reduce-scatter of ``x``'s rows
    run as one fused GEMM-RS kernel (SGLANG_K3_SP_GEMM_RS_MIN_TOKENS)."""
    min_tokens = envs.SGLANG_K3_SP_GEMM_RS_MIN_TOKENS.get()
    if min_tokens <= 0 or not _eager_bf16(o_proj, x):
        return False
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world = parallel.attn_tp_size
    rows = x.shape[0]
    return (
        world in (2, 4, 8)
        and parallel.attn_cp_size == 1
        and rows >= min_tokens
        and rows % world == 0
        and x.shape[1] % 8 == 0
        and o_proj.weight.shape[0] % 128 == 0
        and o_proj.weight.is_contiguous()
        and torch.cuda.get_device_capability(x.device)[0] == 10
    )


def _gemm_rs(o_proj, x: torch.Tensor) -> torch.Tensor:
    global _GEMM_RS_LOGGED
    from sglang.kernels.ops.communication.sp_collective_gemm import (
        matmul_reduce_scatter,
    )
    from sglang.srt.layers.communication.sp_collective_gemm_workspace import (
        sp_collective_gemm_workspace,
    )
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().attn_tp_group
    out = matmul_reduce_scatter(
        x.contiguous(),
        o_proj.weight,
        group.device_group,
        workspace=sp_collective_gemm_workspace,
    )
    if not _GEMM_RS_LOGGED:
        _GEMM_RS_LOGGED = True
        logger.info(
            "K3 SP o_proj GEMM + reduce-scatter fused (CUTLASS, multicast "
            "epilogue): %d rows, attn TP%d (SGLANG_K3_SP_GEMM_RS_MIN_TOKENS=%d)",
            x.shape[0],
            group.world_size,
            envs.SGLANG_K3_SP_GEMM_RS_MIN_TOKENS.get(),
        )
    return out


def _comm_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _STREAMS.get(index)
    if stream is None:
        stream = _STREAMS[index] = torch.cuda.Stream(device=index, priority=-1)
    return stream


def overlapped_matmul_reduce_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    chunks: int,
    world_size: int,
    rank: int,
    reduce_scatter: Callable[[torch.Tensor, torch.Tensor], None],
    *,
    alloc: Callable[[tuple], torch.Tensor],
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """This rank's rows of reduce_scatter(x @ weight.T) over ``world_size``
    ranks, in ``chunks`` column chunks. With ``comm_stream`` each chunk's
    reduce-scatter (``reduce_scatter(output, input)``, an in-place one on the
    chunk's own rows) runs there while the next chunk's GEMM runs; without one
    (CPU tests) they alternate on the current stream."""
    x = x.contiguous()
    rows, width = x.shape[0], weight.shape[0]
    local = rows // world_size
    step = width // chunks
    # [chunk, T, H / chunks]: each chunk's partial is contiguous, as the
    # reduce-scatter needs, and its reduced rows land in its own slot.
    partial = alloc((chunks, rows, step))
    main = torch.cuda.current_stream() if comm_stream is not None else None
    for j in range(chunks):
        torch.mm(x, weight[j * step : (j + 1) * step].t(), out=partial[j])
        shard = partial[j][rank * local : (rank + 1) * local]
        if comm_stream is None:
            reduce_scatter(shard, partial[j])
            continue
        comm_stream.wait_stream(main)
        with torch.cuda.stream(comm_stream):
            reduce_scatter(shard, partial[j])
    if comm_stream is not None:
        main.wait_stream(comm_stream)
    # [T/w, chunk, H/chunk] -> [T/w, H]
    return (
        partial[:, rank * local : (rank + 1) * local]
        .permute(1, 0, 2)
        .reshape(local, width)
    )


def wrap_o_proj(o_proj) -> None:
    """Route an SP-MoE o_proj's large eager batches through the fused GEMM-RS
    or the overlapped GEMM + reduce-scatter. Installed under any wrapper that
    edits o_proj's input (the MLA output gate), so it sees the final input."""
    from sglang.srt.layers.communication import k3_sp_collective
    from sglang.srt.runtime_context import get_parallel

    inner = o_proj.forward

    def _forward(x, *args, **kwargs):
        global _PENDING, _LOGGED
        if not args and not kwargs and takes_gemm_rs(o_proj, x):
            if _PENDING is not None:
                raise RuntimeError(
                    "K3 SP o_proj GEMM-RS: the previous layer's reduced o_proj "
                    "output never reached its FFN read"
                )
            out = _gemm_rs(o_proj, x)
            _PENDING = (out.data_ptr(), tuple(out.shape))
            return out, None
        chunks = num_chunks(o_proj.output_size)
        if not chunks or args or kwargs or not _eager_bf16(o_proj, x):
            return inner(x, *args, **kwargs)
        parallel = get_parallel()
        world = parallel.attn_tp_size
        rows = x.shape[0]
        if (
            world < 2
            or parallel.attn_cp_size != 1
            or rows < envs.SGLANG_K3_SP_RS_OVERLAP_MIN_TOKENS.get()
            or rows % world
            or k3_sp_collective.takes_reduce_scatter(rows, x.device, x.element_size())
        ):
            return inner(x, *args, **kwargs)
        if _PENDING is not None:
            raise RuntimeError(
                "K3 SP o_proj/RS overlap: the previous layer's reduced o_proj "
                "output never reached its FFN read"
            )
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )

        group = parallel.attn_tp_group

        def alloc(shape):
            # The reduce-scatter's buffer, where the plain o_proj puts it too.
            with use_symmetric_memory(group):
                return torch.empty(shape, dtype=x.dtype, device=x.device)

        out = overlapped_matmul_reduce_scatter(
            x,
            o_proj.weight,
            chunks,
            world,
            parallel.attn_tp_rank,
            group.reduce_scatter_tensor,
            alloc=alloc,
            comm_stream=_comm_stream(x.device),
        )
        _PENDING = (out.data_ptr(), tuple(out.shape))
        if not _LOGGED:
            _LOGGED = True
            logger.info(
                "K3 SP o_proj/RS overlap: %d column chunks, reduce-scatter on a "
                "side stream (%d rows, attn TP%d; SGLANG_K3_SP_RS_OVERLAP_CHUNKS)",
                chunks,
                rows,
                world,
            )
        return out, None

    o_proj.forward = _forward


def _take(hidden_states) -> bool:
    global _PENDING
    if _PENDING is None or not isinstance(hidden_states, torch.Tensor):
        return False
    if _PENDING != (hidden_states.data_ptr(), tuple(hidden_states.shape)):
        raise RuntimeError(
            "K3 SP o_proj/RS overlap: the FFN read got another tensor than the "
            "reduced o_proj output"
        )
    _PENDING = None
    return True


def _presummed_add(hidden_states, residual, forward_batch):
    """The written stream for an o_proj output already reduced onto this rank's
    rows: plus the residual (its rows on this rank), as the read's own add."""
    if not _take(hidden_states):
        return None
    if residual is None:
        return hidden_states
    if residual.shape[0] != hidden_states.shape[0]:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        residual = residual.tensor_split(parallel.attn_tp_size)[parallel.attn_tp_rank]
    return residual + hidden_states


# First among the FFN read's fusions: a scattering one must never see these rows.
presummed_fusion = ReadoutFusion(SumGroup.ATTN_TP, _presummed_add, scatters=True)


def check_consumed() -> None:
    """At the end of a forward: every reduced o_proj output was read."""
    global _PENDING
    if _PENDING is not None:
        _PENDING = None
        raise RuntimeError(
            "K3 SP o_proj/RS overlap: a reduced o_proj output was never read"
        )
