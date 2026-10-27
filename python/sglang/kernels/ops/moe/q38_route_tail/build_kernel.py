"""Generate the isolated router from the unchanged native K4+quant source."""
from pathlib import Path

here = Path(__file__).resolve().parent
source = (here.parent / "q38_router_native_kernel/source/native_splitk_quant_generated.py").read_text()
source = source.replace(
    "        q38_sf: cute.Tensor,\n",
    "        q38_sf: cute.Tensor,\n        q38_meta: cute.Tensor,\n        q38_arrival: cute.Tensor,\n",
)
source = source.replace(
    "self.kernel(a, b, c, bias, q38_gs, q38_q, q38_sf).launch(",
    "self.kernel(a, b, c, bias, q38_gs, q38_q, q38_sf, q38_meta, q38_arrival).launch(",
)
# The original trigger permits a routing successor. Our successor is FC1 and
# must only be released after the complete nine-field record has been written.
source = source.replace(
    "        if cutlass.const_expr(is_a and self.use_pdl):\n"
    "            cute.arch.griddepcontrol_launch_dependents()\n",
    "",
)
anchor = "    @cute.experimental.jit\n    def dma_warp("
assert source.count(anchor) == 1
tail = """        # All native warps complete before the owner publishes its logits.
        route_arena = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1609), alignment=16)
        cute.arch.barrier()
        if split_rank == OWNER_RANK:
            if cute.arch.thread_idx()[0] == 0:
                owners = (cute.ceil_div(mC.layout.shape[0], self.cta_m)
                          * cute.ceil_div(mC.layout.shape[1], self.cta_n))
                previous = _q38_owner_arrive(q38_arrival.iterator)
                route_arena[1608] = cutlass.Int32(previous % owners == owners - 1)
            cute.arch.barrier()
            if route_arena[1608] != 0:
                _q38_route_tail(q38_meta.iterator, route_arena.iterator)
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_launch_dependents()

"""
source = source.replace(anchor, tail + anchor)
(here / "splitk.py").write_text(source)
