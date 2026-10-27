"""HC N32/K640 warp-MMA partials with a native up/gate consumer."""

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute import experimental as cute_ext

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs
from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
    SplitKDenseGemmKernel,
    _from_dlpack_dynamic,
    _sigmoid_f32,
    _to_cute_swap,
    default_tactic,
)


@cache_once
def _down_module():
    return load_jit(
        "hc_splitup_down_v1",
        cuda_files=["elementwise/hc_splitup.cuh"],
        cuda_wrappers=[
            ("down", "hc_splitup::HcSplitUp::down"),
            ("down_trace", "hc_splitup::HcSplitUp::down_trace"),
        ],
    )


class HcSplitUpKernel(SplitKDenseGemmKernel):
    def __init__(self, rows):
        super().__init__(
            tactic=default_tactic(rows, 10240, 384),
            use_pdl=True,
            has_bias=False,
            epilogue_mode="gate",
            epilogue_scale=0.25,
            epilogue_group=4,
        )
        self.hc_splitup = True
        assert self.num_ab_stage >= 3

    @cute.experimental.jit
    def reduce_partial_activation(self, partial, sb, bar_full, n_idx):
        # The weight DMA warp remains independent of this wait. The ordinary
        # B-DMA warp is disabled, and epilogue threads publish its three stages.
        cute.arch.griddepcontrol_wait()
        tid = cute.arch.thread_idx()[0] - 128
        rows = cute.size(partial, mode=[1])
        for it in cutlass.range_constexpr(self.cta_n * 384 // 128):
            index = it * 128 + tid
            row = index // 384
            k = index % 384
            value = cutlass.Float32(0.0)
            if row + n_idx * self.cta_n < rows and k < 320:
                value = partial[0, row + n_idx * self.cta_n, k]
                for split in cutlass.range_constexpr(1, 16):
                    value = value + partial[split, row + n_idx * self.cta_n, k]
                scaled = value * 0.25
                value = scaled * _sigmoid_f32(scaled)
            sb[(row, k % 16), 0, ((k % 128) // 16 % 4, (k % 128) // 64), k // 128] = value.to(
                cutlass.BFloat16
            )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=6, number_of_threads=128)
        if tid == 0:
            for stage in cutlass.range_constexpr(3):
                cute.arch.mbarrier_arrive(bar_full + stage)


@cute.experimental.jit
def _bmm_splitup(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    partial: cute.Tensor,
    x: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        partial,
        x,
        out,
        stream,
    )


_up_cache = {}
_trace_buffers = {}


def dump_splitup_trace(path):
    """Called after cudaProfilerStop, never during the measured Static pass."""
    import json
    import os
    from pathlib import Path

    if not _trace_buffers:
        return
    records = [
        {"weight_ptr": ptr, "rows": rows, "cta": buf.cpu().tolist()}
        for (ptr, rows), buf in _trace_buffers.items()
    ]
    target = Path(f"{path}.{os.getpid()}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "columns": ["entry_ns", "weights_ready_before_wait_ns", "after_wait_ns", "end_ns", "smid"],
        "note": "Instrumented M4 kernel only; align globaltimer to its matching nsys kernel; not a speed result.",
        "records": records,
    }) + "\n")


def hc_splitup(x, down_weight, up_weight, out, shape_anchor):
    rows = x.shape[0]
    partial = torch.empty((16, rows, 320), dtype=torch.float32, device=x.device)
    if rows == 4 and envs.SGLANG_Q38C_HC_SPLITUP_TRACE_FILE.get():
        trace_key = (down_weight.data_ptr(), rows)
        trace = _trace_buffers.get(trace_key)
        if trace is None:
            trace = torch.zeros((160, 5), dtype=torch.int64, device=x.device)
            _trace_buffers[trace_key] = trace
        _down_module().down_trace(x, down_weight, partial, trace)
    else:
        _down_module().down(x, down_weight, partial)
    # The anchor carries only the native up MMA shape/strides. Its values are
    # never loaded; the epilogue threads create BF16 activations in shared memory.
    args = _to_cute_swap(shape_anchor[:rows], up_weight.T, x, None)
    tensors = (
        *args[:3],
        _from_dlpack_dynamic(partial, 2),
        _from_dlpack_dynamic(x, 1),
        _from_dlpack_dynamic(out, 1),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key = (x.device, 8 if rows <= 8 else 16)
    compiled = _up_cache.get(key)
    if compiled is None:
        compiled = cute_ext.compile(_bmm_splitup, HcSplitUpKernel(rows), *tensors, stream)
        _up_cache[key] = compiled
    compiled(*tensors, stream)
    return out
