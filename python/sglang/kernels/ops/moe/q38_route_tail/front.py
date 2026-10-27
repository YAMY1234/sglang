"""Native router MMA/rounding plus a last-owner native metadata tail."""
import ctypes
import importlib.util
import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from ..q38_router_native_kernel import fused as native

HERE = Path(__file__).resolve().parent
_TAIL = (HERE / "tail_inline.ptx").read_text()


@dsl_user_op
def owner_arrive(pointer, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), [pointer.toint(loc=loc, ip=ip).ir_value()],
        "atom.add.acq_rel.gpu.u32 $0, [$1], 1;", "=r,l,~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def route_tail(meta, shared, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [meta.toint(loc=loc, ip=ip).ir_value(),
               shared.toint(loc=loc, ip=ip).ir_value()],
        _TAIL, "l,r,~{memory}", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


spec = importlib.util.spec_from_file_location(__package__ + ".splitk", HERE / "splitk.py")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
core._q38_quant16 = native.quant16
core._q38_owner_arrive = owner_arrive
core._q38_route_tail = route_tail
spec.loader.exec_module(core)


class Params(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "logits", "topk_ids", "padded_count", "expanded_to_permuted",
        "permuted_to_token", "expert_weights", "histogram", "tokens_per_expert",
        "cta_expert", "cta_mn_limit", "live_ctas",
    )] + [("num_tokens", ctypes.c_uint32), ("tile_n", ctypes.c_uint32)]


@ext.jit
def invoke(op: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
           gs: cute.Tensor, q: cute.Tensor, sf: cute.Tensor, meta: cute.Tensor,
           arrival: cute.Tensor, stream: cuda.CUstream):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    op(cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
       cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
       c, c, gs, q, sf, meta, arrival, stream)


class Front:
    def __init__(self, x, weight, gs):
        self.m = m = x.shape[0]
        if m not in (4, 8, 12) or tuple(x.shape) != (m, 2560):
            raise ValueError("ROUTE_TAIL requires M4/M8/M12 H2560")
        self.logits = torch.empty((m, 512), device=x.device, dtype=torch.bfloat16)
        self.packed = torch.empty((m, 1280), device=x.device, dtype=torch.uint8)
        self.sf = torch.empty((m, 160), device=x.device, dtype=torch.uint8)
        self.ids = torch.empty((m, 10), device=x.device, dtype=torch.int32)
        self.weights = torch.empty((m, 10), device=x.device, dtype=torch.bfloat16)
        self.arrival = torch.zeros((1,), device=x.device, dtype=torch.int32)
        self.tactic = core.SplitKTactic(64, 8 if m <= 8 else 16, 4, 10)
        core.validate_tactic(self.tactic, m, 512, 2560)
        # One eager native reference call supplies real logits to the original
        # native tactic resolver. It is never present in the performance graph.
        ref = native.core()
        op = ref.SplitKDenseGemmKernel(
            tactic=ref.SplitKTactic(64, 8 if m <= 8 else 16, 4, 10),
            use_pdl=True, has_bias=False,
        )
        args = self._args(x, weight, gs)
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        compiled = ext.compile(native.invoke, op, *args, stream)
        compiled(*args, stream)

    def _args(self, x, weight, gs):
        tensors = core._to_cute_swap(x, weight.T, self.logits, None)
        return (*tensors[:3], from_dlpack(gs), from_dlpack(self.packed.flatten()),
                from_dlpack(self.sf.flatten()))

    def prepare(self, metadata, tile_n, x, weight, gs):
        self.metadata = tuple(metadata)
        p = Params(self.logits.data_ptr(), self.ids.data_ptr(),
                   *(t.data_ptr() for t in metadata), self.m, tile_n)
        self.params = torch.tensor(list(bytes(p)), device=x.device, dtype=torch.uint8)
        args = (*self._args(x, weight, gs), from_dlpack(self.params), from_dlpack(self.arrival))
        op = core.SplitKDenseGemmKernel(tactic=self.tactic, use_pdl=True, has_bias=False)
        self.compiled = ext.compile(
            invoke, op, *args, cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
        )

    def run(self, x, weight, gs):
        # Rebind caller activation pointers in eager/warmup/capture; replay keeps
        # its own captured pointers. All produced buffers remain strongly owned.
        self.compiled(
            *self._args(x, weight, gs), from_dlpack(self.params), from_dlpack(self.arrival),
            cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
        )
