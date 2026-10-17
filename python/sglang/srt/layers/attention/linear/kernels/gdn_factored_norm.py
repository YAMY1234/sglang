"""Default-off P4 gate shared by model dispatch and the factored backend."""
import logging
import math
import os


# docs/231: only timed buckets meeting the unchanged >=1 us/layer gate.
# This filters norm fusion, independently of the wider K1 CUDA whitelist.
# docs/219 lead 01:14 / 01:27 (v10): HV24 admitted buckets {1, 4} by the served A/B (np4ab-043542 bucket 1,
# np4ab-051459 bucket 4); 2 / 8 not admitted (no certificate in two loads).
QUALIFIED_NORM_BATCHES = {24: frozenset((1, 4)), 48: frozenset((1, 2, 3))}
# Per factor dtype (lead 12:5x): fp32 factors (SGLANG_GDN_FACTORED_CUDA_FP32=1) start from the fp16 buckets; a bucket whose
# fp32 fused-vs-unfused micro is not faster is removed here only.  The fp16 whitelist above is unchanged.
QUALIFIED_NORM_BATCHES_FP32 = {24: frozenset((1, 4)), 48: frozenset((1, 2, 3))}


def qualified_norm_batches(factor_dtype):
    import torch

    return QUALIFIED_NORM_BATCHES_FP32 if factor_dtype == torch.float32 else QUALIFIED_NORM_BATCHES


# docs/219 fp32-factor arm (C+f32x): admit fp32 U / W to the CUDA step (and so P4) and the CUDA expiry.  Default off;
# fp16 factors take the unchanged kernels either way.
CUDA_FP32 = os.environ.get("SGLANG_GDN_FACTORED_CUDA_FP32", "0") == "1"
_FP32_LOGGED = set()


def factor_dtype_ok(fu, fw, what):
    import torch

    if fu.dtype == fw.dtype == torch.float16:
        return True
    if CUDA_FP32 and fu.dtype == fw.dtype == torch.float32:
        if what not in _FP32_LOGGED:
            _FP32_LOGGED.add(what)
            logging.getLogger(__name__).info("FACTORED_CUDA_FP32_ACTIVE %s (fp32 factors)", what)
        return True
    return False


def enabled():
    # Read at capture/dispatch time so in-process A/B can capture both arms.
    return os.environ.get("SGLANG_GDN_FACTORED_STEP_FUSE_NORM", "0") == "1"


def cuda_step_eligible(gf, mixed, a, b, fu, fw, *, invalidate=False,
                       prefix_valid=None):
    B = mixed.shape[0]
    _, HV, RMAX, K = fu.shape
    import torch

    return (
        RMAX == 32
        and gf.STEP_IMPL_BY_RMAX.get(RMAX) == "cuda"
        and B in gf.STEP_CUDA_BATCHES
        and B <= (3 if HV == 48 else 8 if HV == 24 else 0)
        and K == fw.shape[-1] == 128
        and not gf.STEP_ROWS_BY_RMAX.get(RMAX, 0)
        and gf.STEP_VSPLIT_BY_RMAX.get(RMAX, 1) == 1
        and (not gf.STEP_CUDA_MAX_BATCH or B <= gf.STEP_CUDA_MAX_BATCH)
        and mixed.dtype == a.dtype == b.dtype == torch.bfloat16
        and factor_dtype_ok(fu, fw, "step")
        and (not invalidate or (prefix_valid is not None
                               and prefix_valid.dtype == torch.int32))
    )


def decode_backend(backend):
    # HybridAttnBackend selects decode_backend for DECODE; its child can in
    # turn be HybridLinearAttnBackend. Follow only these audited dispatches.
    seen = set()
    while id(backend) not in seen:
        seen.add(id(backend))
        child = getattr(backend, "decode_backend", None)
        if child is None:
            child = getattr(backend, "linear_attn_backend", None)
        if child is None:
            break
        backend = child
    return backend


def _can_fuse_norm(backend, forward_batch, mixed, z, norm):
    """Only promise fusion where the exact CUDA route and norm contract apply.

    No GPU reads or allocations. The extension is preloaded by the pool before
    graph capture. Returning True commits the backend to fusion (fail closed).
    """
    if not enabled() or not forward_batch.forward_mode.is_decode():
        return False
    backend = decode_backend(backend)
    pool = getattr(backend, "factored", None)
    if pool is None or getattr(pool, "_exact_tail_transaction", None) is not None:
        return False
    # The default Qwen3.5 decode producer generates z inside GDNBackend,
    # after the model has handed it (projected_qkvz, projected_ba). Both its
    # fused Conv1D path and correctness fallback produce the same BF16 z shape.
    deferred_z = isinstance(mixed, tuple) and len(mixed) == 2 and z is None
    projected_ba = mixed[1] if deferred_z else None
    if deferred_z:
        mixed = mixed[0]
    if (z is None and not deferred_z) or not hasattr(mixed, "dtype"):
        return False
    from . import gdn_factored as gf
    from .gdn_factored_cuda import load_extension
    import torch

    if (pool.cfg.kernel or gf.DEFAULT_KERNEL) != "split":
        return False
    # These checks mirror factored_packed_decode's CUDA selection. Pool tensors
    # have one leading layer dimension; select metadata-only views here.
    if not cuda_step_eligible(gf, mixed, mixed, mixed, pool.U[0], pool.W[0]):
        return False
    B, HV = mixed.shape[0], pool.U.shape[2]
    if B not in qualified_norm_batches(pool.U.dtype).get(HV, ()):
        return False
    weight = getattr(norm, "weight", None)
    if (getattr(norm, "activation", None) != "sigmoid"
            or not getattr(norm, "norm_before_gate", False)
            or getattr(norm, "group_size", None) not in (None, 128)
            or getattr(norm, "bias", None) is not None
            or weight is None or weight.dtype != torch.bfloat16
            or weight.shape != (128,) or not weight.is_contiguous()
            or weight.device != mixed.device
            or not math.isfinite(norm.eps) or norm.eps <= 0):
        return False
    if deferred_z:
        if (not hasattr(projected_ba, "dtype")
                or len(mixed.shape) != 2 or len(projected_ba.shape) != 2
                or projected_ba.shape != (B, 2*HV)
                or projected_ba.dtype != torch.bfloat16
                or projected_ba.device != mixed.device
                or mixed.shape[1] <= 2*HV*128 or mixed.shape[1] % 128):
            return False
    elif (z.dtype != torch.bfloat16 or z.shape != (B, HV, 128)
            or z.stride(-1) != 1 or z.stride(0) <= 0 or z.stride(1) <= 0
            or z.stride(0) % 8 or z.stride(1) % 8 or z.storage_offset() % 8
            or z.stride(0) > 2**31-1 or z.stride(1) > 2**31-1
            or (B-1)*z.stride(0)+(HV-1)*z.stride(1)+127 > 2**31-1
            or z.device != mixed.device):
        return False
    return load_extension() is not None


_REJECTIONS = set()


def can_fuse_norm(backend, forward_batch, mixed, z, norm):
    result = _can_fuse_norm(backend, forward_batch, mixed, z, norm)
    metadata_input = mixed[0] if isinstance(mixed, tuple) and mixed else mixed
    if (not result and enabled() and os.environ.get("SGLANG_GDN_P4_CAPTURE_DIR")
            and forward_batch.forward_mode.is_decode()
            and getattr(metadata_input, "shape", (0,))[0] in range(1, 9)):
        leaf = decode_backend(backend)
        key = (type(backend).__name__, type(leaf).__name__, metadata_input.shape[0])
        if key not in _REJECTIONS:
            _REJECTIONS.add(key)
            pool = getattr(leaf, "factored", None)
            def tensor_meta(x):
                if x is None: return None
                if isinstance(x, tuple): return tuple(tensor_meta(t) for t in x)
                return dict(shape=tuple(x.shape), dtype=str(x.dtype), device=str(x.device),
                            stride=tuple(x.stride()), storage_offset=x.storage_offset(),
                            contiguous=x.is_contiguous())
            logging.getLogger(__name__).warning(
                "P4_NORM_FUSION_REJECTED backend=%s leaf=%s mixed=%s z=%s norm=%s "
                "activation=%s before_gate=%s group=%s weight=%s pool_U=%s exact_tail=%s",
                type(backend).__name__, type(leaf).__name__, tensor_meta(mixed), tensor_meta(z),
                type(norm).__name__, getattr(norm, "activation", None),
                getattr(norm, "norm_before_gate", None), getattr(norm, "group_size", None),
                tensor_meta(getattr(norm, "weight", None)), tensor_meta(getattr(pool, "U", None)),
                getattr(pool, "_exact_tail_transaction", None) is not None)
    return result
