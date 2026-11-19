"""SGLang DSv4.1 prefill integration of Mega mHC.

Credit: DeepSeek's DeepGEMM project provides the fused Mega mHC kernel.
https://github.com/deepseek-ai/DeepGEMM
"""

import functools
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda

if is_cuda():
    import deep_gemm

logger = logging.getLogger(__name__)

# Locality-striped residual pool (SGLANG_OPT_DSV41_MEGA_MHC_LOCALITY): rows alternate between the two dies' memory
# every _STRIPE_TOKENS tokens and each die's SMs claim the tasks of their own rows, so the residual stream stays in
# local HBM. Smaller batches and graph capture keep plain outputs.
_STRIPE_TOKENS = 1024
_LOCALITY_MIN_TOKENS = 5000
_POOL_SLOTS = 2


@functools.cache
def _task_ring_dg():
    """The DeepGEMM build whose Mega mHC host runs the task-ring kernel, or None (switch off)."""
    if not envs.SGLANG_OPT_DSV41_MEGA_MHC_TASK_RING.get():
        return None
    import deep_gemm_fp4acts

    logger.info(
        "Mega mHC runs the task-ring kernel through %s (%s), locality %s",
        deep_gemm_fp4acts.__name__,
        deep_gemm_fp4acts.__file__,
        envs.SGLANG_OPT_DSV41_MEGA_MHC_LOCALITY.get(),
    )
    return deep_gemm_fp4acts


def _num_striped_below(x: int, domain: int) -> int:
    stripe = _STRIPE_TOKENS
    return x // (2 * stripe) * stripe + min(stripe, max(0, x % (2 * stripe) - domain * stripe))


@functools.cache
def _stripe_offset(rows: int) -> int:
    """The first row (a multiple of the kernel's 64-row block) at which ``rows`` rows split most evenly over the
    two dies' stripes."""
    return min(
        range(0, 2 * _STRIPE_TOKENS, 64),
        key=lambda o: abs(2 * (_num_striped_below(o + rows, 0) - _num_striped_below(o, 0)) - rows),
    )


class _LocalityPool:
    """Rotating locality-striped residual and normed-output buffers."""

    def __init__(self, dg, hc: int, hidden: int, max_rows: int, device) -> None:
        domains = dg.locality_domain
        self.rows = -(-max_rows // _STRIPE_TOKENS) * _STRIPE_TOKENS + 2 * _STRIPE_TOKENS
        b = torch.bfloat16.itemsize
        self.residuals = [
            domains.empty_striped([self.rows, hc, hidden], torch.bfloat16, _STRIPE_TOKENS * hc * hidden * b)
            for _ in range(_POOL_SLOTS)
        ]
        self.ys = [
            domains.empty_striped([self.rows, hidden], torch.bfloat16, _STRIPE_TOKENS * hidden * b)
            for _ in range(_POOL_SLOTS)
        ]
        self.sm_domains = domains.get_sm_locality_domains().to(device=device, dtype=torch.uint8).contiguous()
        self.next_slot = 0
        logger.info("Mega mHC locality pool: %d rows x %d slots (residual and normed output)", self.rows, _POOL_SLOTS)

    def take(self, residual: torch.Tensor, rows: int):
        """The next slot not holding this call's input residual: its residual and output rows, and their offset."""
        offset = _stripe_offset(rows)
        while True:
            slot = self.next_slot
            self.next_slot = (slot + 1) % _POOL_SLOTS
            buf = self.residuals[slot]
            begin = buf.data_ptr()
            if not begin <= residual.data_ptr() < begin + buf.numel() * buf.element_size():
                sl = slice(offset, offset + rows)
                return buf[sl], self.ys[slot][sl], offset


@functools.cache
def _locality_pool(hc: int, hidden: int, device):
    dg = _task_ring_dg()
    if dg is None or not envs.SGLANG_OPT_DSV41_MEGA_MHC_LOCALITY.get():
        return None
    ld = dg.locality_domain
    if ld.get_num_locality_domains() != 2 or not dg.is_localization_available():
        logger.warning("Mega mHC locality pool unavailable (no 2-domain MLOPart localization); plain outputs")
        return None
    return _LocalityPool(dg, hc, hidden, envs.SGLANG_OPT_DSV41_MEGA_MHC_LOCALITY_MAX_TOKENS.get(), device)


def mhc_mega_boundary(
    x,
    residual,
    pre,
    post,
    comb,
    fn,
    scale,
    base,
    weight,
    rms_eps,
    hc_eps,
    norm_eps,
    sinkhorn_iters,
    y_fp8=None,
    y_gemm_sf=None,
    moe_input=None,
):
    """Update the residual, collapse with carried pre, and compute next stats.

    The collapse uses ``pre`` from the preceding sublayer, not the new pre
    produced from the updated residual. Preserve this DSv4.1 shifted ordering.
    Used by the opt-in SM10x prefill path. Before CUDA graph capture, warm up
    on the capture stream to initialize DeepGEMM's per-stream barriers.
    ``y_fp8`` / ``y_gemm_sf`` also receive the normed rows as a DeepGEMM MXFP8 pair.
    """
    rows, hidden = x.shape
    hc = residual.shape[1]
    dg = _task_ring_dg()
    locality = {}
    pool = None
    if dg is not None and _LOCALITY_MIN_TOKENS <= rows and not torch.cuda.is_current_stream_capturing():
        pool = _locality_pool(hc, hidden, x.device)
    if pool is not None and rows + 2 * _STRIPE_TOKENS <= pool.rows:
        updated, normalized, offset = pool.take(residual, rows)
        if moe_input is not None:
            # The MoE gate reads the normed rows in token order
            normalized = torch.empty_like(x)
        locality = dict(
            sm_locality_domains=pool.sm_domains,
            domain_stripe_tokens=_STRIPE_TOKENS,
            domain_stripe_offset_tokens=offset,
        )
    else:
        updated = torch.empty_like(residual)
        normalized = torch.empty_like(x)
    next_pre = torch.empty((rows, hc, 1), device=x.device, dtype=torch.float32)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    y_routed_sf = y_shared_sf = None
    shared_sf_block_m = 0
    shared_sf_page_stride = 0
    if moe_input is not None:
        assert y_fp8 is None
        y_fp8, y_routed_sf, y_shared_sf, shared_sf_block_m = moe_input
    if dg is not None:
        if y_gemm_sf is not None and y_gemm_sf.dim() == 4:
            # FlashInfer's 128x4 scales are the kernel's paged shared-SF layout with 128-row pages of every word
            words = y_gemm_sf.shape[1]
            y_shared_sf = y_gemm_sf.view(-1).as_strided((rows, words), (1, 128))
            shared_sf_block_m, shared_sf_page_stride, y_gemm_sf = 128, words * 128, None
        dg.mega_mhc(
            x=x,
            residual=residual,
            shifted_prev_mix=pre.unsqueeze(-1),
            post_mix=post.unsqueeze(-1),
            comb_res_mix=comb,
            fn=fn,
            mix_scales=scale,
            mix_bases=base,
            hc_mult=hc,
            hc_norm_eps=rms_eps,
            hc_pre_eps=hc_eps,
            hc_post_scale=2.0,
            sinkhorn_eps=hc_eps,
            num_sinkhorn_iters=sinkhorn_iters,
            rmsnorm_weight=weight,
            rmsnorm_eps=norm_eps,
            rmsnorm_scale=1.0,
            new_residual=updated,
            new_prev_mix=next_pre,
            new_post_mix=next_post,
            new_comb_res_mix=next_comb,
            y_bf16=normalized,
            y_fp8=y_fp8,
            y_gemm_sf=y_gemm_sf,
            y_routed_sf=y_routed_sf,
            y_shared_sf=y_shared_sf,
            shared_sf_block_m=shared_sf_block_m,
            shared_sf_page_stride=shared_sf_page_stride,
            **locality,
        )
        return updated, normalized, (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)
    deep_gemm.mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=pre.unsqueeze(-1),
        post_mix=post.unsqueeze(-1),
        comb_res_mix=comb,
        fn=fn,
        mix_scales=scale,
        mix_bases=base,
        hc_mult=hc,
        hc_norm_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_post_scale=2.0,
        sinkhorn_eps=hc_eps,
        num_sinkhorn_iters=sinkhorn_iters,
        rmsnorm_weight=weight,
        rmsnorm_eps=norm_eps,
        rmsnorm_scale=1.0,
        new_residual=updated,
        new_prev_mix=next_pre,
        new_post_mix=next_post,
        new_comb_res_mix=next_comb,
        y_bf16=normalized,
        y_fp8=y_fp8,
        y_gemm_sf=y_gemm_sf,
        y_routed_sf=y_routed_sf,
        y_shared_sf=y_shared_sf,
        shared_sf_block_m=shared_sf_block_m,
    )
    return updated, normalized, (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)
