"""MXFP4 x MXFP8 routed experts on SM107 with CuTe DSL grouped GEMMs (SGLANG_K31_PREFILL_MOE_CUTE_FC1).

Replaces the trtllm-gen FC1 -> FC2 body for prefill, the way FlashInfer's staged MXFP4 x MXFP8 runner does:
FlashInfer's staged routing (tile 128; it also allocates the FC1 / FC2 output buffers) -> one Triton pass that writes
each expert's routed token rows and their ue8m0 scales in permuted order -> FC1 (swap-AB grouped GEMM with the SiTU +
MXFP8 requant epilogue, bit-exact with trtllm-gen's fused FC1) -> FC2 (same kernel, BF16 rows) -> FlashInfer finalize,
or SGLang's deferred finalize. Every buffer is in FlashInfer's permuted-row layout (experts in order, each padded to
128 rows), so finalize consumes FlashInfer's own expanded->permuted map.

Weights take this path's layout at load (one copy per layer): FC1 rows in 256-row blocks [gate 128 | up 128] of the
same 128 outputs, FC2 rows as in the checkpoint, both with ue8m0 scales in 128x4 atoms.
"""

from __future__ import annotations

import threading
from typing import Tuple, Union

import msgspec
import torch
import triton
import triton.language as tl

_ROWS = 128  # routing padding per expert segment = scale-atom height
_CFG = ((512, 192, 128), (256, 192, 64), (2, 1))  # mma tiler, mma instruction, cluster
# Test hook: when a list, run() appends (routing slot, FC1 rows, FC1 scale atoms) of each call.
_capture = None


class CutePrefillMoeWeights(msgspec.Struct, frozen=True, kw_only=True):
    w13: torch.Tensor  # [E, 2I, H/2] packed FP4; 256-row block b = gate[128b:+128], up[128b:+128]
    s13: torch.Tensor  # [E, 2I, H/32] ue8m0 bytes (float8_e4m3fn view) in 128x4 atoms, K atoms fastest
    w2: torch.Tensor  # [E, H, I/2] packed FP4
    s2: torch.Tensor  # [E, H, I/32] like s13
    hidden: int
    inter: int
    num_experts: int
    beta: float  # SiTU gate tanh scale (gemm1_alpha)
    lbeta: float  # SiTU up tanh scale (gemm1_clamp_limit)


def _swizzle_sf(s: torch.Tensor) -> torch.Tensor:
    """[E, rows, kb] linear ue8m0 -> same shape holding 128x4 atoms, K atoms fastest."""
    n, r, kb = s.shape
    return (
        s.reshape(n, r // 128, 4, 32, kb // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(n, r, kb)
        .view(torch.float8_e4m3fn)
    )


def prepare_weights(
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    *,
    beta: float,
    lbeta: float,
) -> CutePrefillMoeWeights:
    """Checkpoint-order weights (w13 rows: gate [0, I), up [I, 2I)) -> this path's layout."""
    e, two_i, half_h = w13.shape
    inter, hidden = two_i // 2, half_h * 2
    if hidden % 128 or inter % 128 or tuple(w2.shape) != (e, hidden, inter // 2):
        raise ValueError(f"unsupported MoE shape: w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)}")

    def blocks(t):  # [E, 2I, X] -> 256-row blocks of gate 128 then up 128
        g, u = t[:, :inter], t[:, inter:]
        return torch.stack(
            (g.reshape(e, inter // 128, 128, -1), u.reshape(e, inter // 128, 128, -1)), dim=2
        ).reshape(e, two_i, -1)

    return CutePrefillMoeWeights(
        w13=blocks(w13.view(torch.uint8)).contiguous(),
        s13=_swizzle_sf(blocks(s13.view(torch.uint8))),
        w2=w2.view(torch.uint8).contiguous(),
        s2=_swizzle_sf(s2.view(torch.uint8)),
        hidden=hidden,
        inter=inter,
        num_experts=e,
        beta=float(beta),
        lbeta=float(lbeta),
    )


@triton.jit
def _plan_kernel(
    counts_ptr, n_groups,
    s1_ptr, p1_ptr, f1_ptr, s2_ptr, p2_ptr, f2_ptr,
    w13, w13_stride, s13, s13_stride, w2, w2_stride, s2, s2_stride,
    xp, sfb, o1, o1sf, o2,
    H: tl.constexpr, I: tl.constexpr, BLOCK: tl.constexpr,
):
    # one program: per-expert 128-padded segment offsets and the two GEMMs' group arrays, all on device
    e = tl.arange(0, BLOCK)
    m = e < n_groups
    c = tl.load(counts_ptr + e, mask=m, other=0)
    pad = (c + 127) // 128 * 128
    off = (tl.cumsum(pad, 0) - pad).to(tl.int64)
    e64 = e.to(tl.int64)
    tl.store(s1_ptr + e * 4 + 0, 2 * I + 0 * c, mask=m)
    tl.store(s1_ptr + e * 4 + 1, c, mask=m)
    tl.store(s1_ptr + e * 4 + 2, H + 0 * c, mask=m)
    tl.store(s1_ptr + e * 4 + 3, 1 + 0 * c, mask=m)
    tl.store(p1_ptr + e * 4 + 0, w13 + e64 * w13_stride, mask=m)
    tl.store(p1_ptr + e * 4 + 1, xp + off * H, mask=m)
    tl.store(p1_ptr + e * 4 + 2, o1 + off * I, mask=m)
    tl.store(p1_ptr + e * 4 + 3, 0 * off, mask=m)
    tl.store(f1_ptr + e * 3 + 0, s13 + e64 * s13_stride, mask=m)
    tl.store(f1_ptr + e * 3 + 1, sfb + off // 128 * ((H // 128) * 512), mask=m)
    tl.store(f1_ptr + e * 3 + 2, o1sf + off * (I // 32), mask=m)
    tl.store(s2_ptr + e * 4 + 0, H + 0 * c, mask=m)
    tl.store(s2_ptr + e * 4 + 1, c, mask=m)
    tl.store(s2_ptr + e * 4 + 2, I + 0 * c, mask=m)
    tl.store(s2_ptr + e * 4 + 3, 1 + 0 * c, mask=m)
    tl.store(p2_ptr + e * 4 + 0, w2 + e64 * w2_stride, mask=m)
    tl.store(p2_ptr + e * 4 + 1, o1 + off * I, mask=m)
    tl.store(p2_ptr + e * 4 + 2, o2 + off * (H * 2), mask=m)
    tl.store(p2_ptr + e * 4 + 3, 0 * off, mask=m)
    tl.store(f2_ptr + e * 3 + 0, s2 + e64 * s2_stride, mask=m)
    tl.store(f2_ptr + e * 3 + 1, o1sf + off * (I // 32), mask=m)
    tl.store(f2_ptr + e * 3 + 2, 0 * off, mask=m)


@triton.jit
def _count_kernel(ids_ptr, n, counts_ptr, n_experts, SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    # tokens per expert from the routed ids (SHIFT 16: packed id << 16 | weight)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    e = tl.load(ids_ptr + i, mask=m, other=-1) >> SHIFT
    m = m & (e >= 0) & (e < n_experts)
    tl.atomic_add(counts_ptr + e, 1 + 0 * e, mask=m)


@triton.jit
def _permute_kernel(
    p2t_ptr, total_ptr, xs_ptr, sfb_ptr, xq_ptr, xp_ptr, n_rows, n_tokens,
    KB: tl.constexpr, HW: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # permuted row p <- token p2t[p]: its 8-byte words and its ue8m0 scales into the 128x4 atoms
    # (rows with no token: scales 0, data untouched; rows past the live padded count: untouched)
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = (p < n_rows) & (p < tl.load(total_ptr))
    t = tl.load(p2t_ptr + p, mask=pm, other=-1)
    ok = pm & (t >= 0) & (t < n_tokens)
    c = tl.arange(0, 128)
    cm = c < KB
    v = tl.load(xs_ptr + t[:, None] * KB + c[None, :], mask=ok[:, None] & cm[None, :], other=0)
    r = p[:, None]
    off = ((r // 128) * (KB // 4) + c[None, :] // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + c[None, :] % 4
    tl.store(sfb_ptr + off, v, mask=pm[:, None] & cm[None, :])
    for h0 in tl.static_range(0, HW, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        mm = ok[:, None] & (h[None, :] < HW)
        x = tl.load(xq_ptr + t[:, None].to(tl.int64) * HW + h[None, :], mask=mm)
        tl.store(xp_ptr + r.to(tl.int64) * HW + h[None, :], x, mask=mm)


class _Engine:
    """Compiled FC1 / FC2 kernels and their device-resident group arrays for one (E, H, I, SiTU) shape."""

    def __init__(self, num_groups: int, hidden: int, inter: int, beta: float, lbeta: float, device):
        import cuda.bindings.driver as cuda  # noqa: F401
        import cutlass
        from cutlass.cute.runtime import from_dlpack

        from . import sm107_grouped_moe_gemm as k

        self.k, self.G, self.H, self.I = k, num_groups, hidden, inter
        tiler, inst, cluster = _CFG

        def kernel(fc2: bool):
            kern = k.Sm107GroupedBlockScaledGemmKernel(32, inst, tiler, cluster, True)
            kern.fc1_inter = inter
            kern.fc2_mode = fc2
            kern.fc2_cols = hidden
            # SiTU constants as the trtllm-gen epilogue rounds them (fp32 1/beta)
            kern.situ_beta, kern.situ_lbeta = beta, lbeta
            kern.situ_inv_beta = float(torch.tensor(1.0 / beta, dtype=torch.float32))
            kern.situ_inv_lbeta = float(torch.tensor(1.0 / lbeta, dtype=torch.float32))
            return kern

        self.kern = (kernel(False), kernel(True))
        cts = k.create_tensor_and_stride
        self.init = (
            cts(1, 32, 32, False, cutlass.Float4E2M1FN)[2],
            cts(1, 32, 32, False, cutlass.Float8E4M3FN)[2],
            cts(1, 16, 16, True, cutlass.Float8E4M3FN)[2],
            cts(1, 16, 16, False, cutlass.Float8E8M0FNU)[2],
            cts(1, 16, 16, False, cutlass.Float8E8M0FNU)[2],
        )
        hw = cutlass.utils.HardwareInfo()
        self.mac = hw.get_max_active_clusters(cluster[0] * cluster[1])
        ctas = max(hw.get_max_active_clusters(1), self.mac * cluster[0] * cluster[1])

        def dev(*shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        cls = k.Sm107GroupedBlockScaledGemmKernel
        # per GEMM: shapes [G,4], strides [G,3,2], ptrs [G,4], scale ptrs [G,3], tensormaps [CTAs,5,16]
        self.t = []
        for m_rows, k_cols in ((2 * inter, hidden), (hidden, inter)):
            strides = dev(num_groups, 3, 2, dtype=torch.int32)
            strides[:, 0, 0] = k_cols
            strides[:, 0, 1] = 1
            strides[:, 1, 0] = k_cols
            strides[:, 1, 1] = 1
            strides[:, 2, 0] = 1
            strides[:, 2, 1] = m_rows
            ts = (
                dev(num_groups, 4, dtype=torch.int32),
                strides,
                dev(num_groups, 4, dtype=torch.int64),
                dev(num_groups, 3, dtype=torch.int64),
                dev(ctas, cls.num_tensormaps, cls.bytes_per_tensormap // 8, dtype=torch.int64),
            )
            self.t.append((ts, tuple(from_dlpack(x, assumed_align=16) for x in ts)))
        self.comp = [None, None]
        self.lock = threading.Lock()
        # FlashInfer staged-routing operands that this path does not use
        self.unit_scales = torch.ones(num_groups, dtype=torch.float32, device=device)
        self.no_weights = torch.empty((0,), dtype=torch.bfloat16, device=device)

    def launch(self, which: int, stream) -> None:
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute

        a0, b0, c0, sfa0, sfb0 = self.init
        _, (shapes, strides, ptrs, sfptrs, tmap) = self.t[which]
        cu_stream = cuda.CUstream(stream.cuda_stream)
        if self.comp[which] is None:
            with self.lock:
                if self.comp[which] is None:
                    self.comp[which] = cute.compile(
                        self.kern[which], a0, b0, c0, sfa0, sfb0, self.G, shapes, strides, ptrs, sfptrs,
                        self.mac, tmap, self.mac, cu_stream,
                    )
        self.comp[which](a0, b0, c0, sfa0, sfb0, shapes, strides, ptrs, sfptrs, tmap, cu_stream)


_engines: dict = {}


def _engine(w: CutePrefillMoeWeights, device) -> _Engine:
    key = (w.num_experts, w.hidden, w.inter, w.beta, w.lbeta, torch.device(device).index)
    eng = _engines.get(key)
    if eng is None:
        eng = _engines[key] = _Engine(w.num_experts, w.hidden, w.inter, w.beta, w.lbeta, device)
    return eng


def run(
    w: CutePrefillMoeWeights,
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    routing: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    *,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    output: torch.Tensor,
    defer_finalize: bool,
    enable_pdl: bool,
):
    """Routed experts for MXFP8 tokens x_quant [T, H] (scales x_scale [T, H/32], linear ue8m0).

    routing: packed (id << 16 | bf16 weight) ids [T, K], or (ids, weights). Returns None after finalizing into
    output, or (gemm2_out, expert_weights, expanded_idx_to_permuted_idx) for the deferred finalize.
    """
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType, WeightLayout

    if local_num_experts != w.num_experts or local_expert_offset != 0 or num_experts != w.num_experts:
        raise NotImplementedError("the CuTe prefill MoE path supports EP size 1 only")
    packed = not isinstance(routing, tuple)
    ids, weights = (routing, None) if packed else routing
    num_tokens, top_k = ids.shape
    H, I, E = w.hidden, w.inter, w.num_experts
    device = x_quant.device
    eng = _engine(w, device)
    module = get_trtllm_moe_sm100_module()
    fp8 = torch.float8_e4m3fn

    # The multi-tile routing-metadata API takes top_k <= 8; the staged routing takes any top_k.
    raw = module.moe_op.trtllm_moe_run_routing_fp4_mxfp4_mxfp8(
        None, None, ids, eng.no_weights if packed else weights, x_quant.view(fp8), x_scale.view(fp8),
        w.w13, w.s13, w.w2, w.s2, eng.unit_scales, eng.unit_scales, eng.unit_scales,
        num_experts, top_k, None, None, I, local_expert_offset, local_num_experts, None,
        RoutingMethodType.TopK.value, enable_pdl, [_ROWS, -1], WeightLayout.MajorK.value,
        ActivationType.Situ.value, True, None,
    )
    # Slot 0 is the caller's weights in the unpacked form (an undefined tensor there: never convert it).
    assert len(raw) == 10, len(raw)
    t = [_torch_view(r) for r in raw[1:]]
    expanded, p2t, total, o1, o1sf, o2 = t[0], t[1], t[5], t[6], t[7], t[8]
    expert_weights = _torch_view(raw[0]) if packed else weights

    counts = torch.zeros((E,), dtype=torch.int32, device=device)
    n_ids = num_tokens * top_k
    _count_kernel[(triton.cdiv(n_ids, 1024),)](
        ids, n_ids, counts, E, SHIFT=16 if packed else 0, BLOCK=1024, num_warps=4,
    )
    n_map = p2t.numel() - 1  # FlashInfer keeps one guard element past the map
    rows = -(-n_map // _ROWS) * _ROWS
    u8 = torch.uint8
    xp = torch.empty((rows, H), dtype=u8, device=device)
    sfb = torch.empty((rows // 128 * (H // 128) * 512,), dtype=u8, device=device)
    (s1, _, p1, f1, _), _ = eng.t[0]
    (s2t, _, p2, f2, _), _ = eng.t[1]
    _plan_kernel[(1,)](
        counts, E,
        s1, p1, f1, s2t, p2, f2,
        w.w13.data_ptr(), w.w13[0].numel(), w.s13.data_ptr(), w.s13[0].numel(),
        w.w2.data_ptr(), w.w2[0].numel(), w.s2.data_ptr(), w.s2[0].numel(),
        xp.data_ptr(), sfb.data_ptr(), o1.data_ptr(), o1sf.data_ptr(), o2.data_ptr(),
        H=H, I=I, BLOCK=triton.next_power_of_2(E),
    )
    _permute_kernel[(triton.cdiv(n_map, 64),)](
        p2t, total, x_scale.view(u8), sfb, x_quant.view(torch.int64), xp.view(torch.int64),
        n_map, num_tokens, KB=H // 32, HW=H // 8, BLOCK_P=64, BLOCK_H=64, num_warps=4,
    )
    stream = torch.cuda.current_stream(device)
    eng.launch(0, stream)  # FC1: SiTU + MXFP8 rows o1, scale atoms o1sf
    eng.launch(1, stream)  # FC2: BF16 rows o2
    if _capture is not None:
        _capture.append((expanded, o1.view(u8), o1sf.view(u8)))
    if defer_finalize:
        return o2, expert_weights, expanded
    module.trtllm_moe_run_finalize(
        o2, output, expert_weights, expanded, total, num_tokens, num_experts, top_k, H, enable_pdl, False,
    )
    return None


def _torch_view(value) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.from_dlpack(value)
