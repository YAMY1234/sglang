"""Low-batch routed-expert GEMMs for DeepSeek-V4.1 decode (1-8 rows), opt-in.

Ported from the muLLM vLLM branch (mullm/vllm@b5a4d16f,
vllm/models/deepseek_v41/nvidia/ops/routed_expert_gemm.py and routed_expert_gemm_src/,
commits 885aec78 / 42636ce2 / 31bd8785 / 4b78e351). With
SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM=1 a TP MoE layer (all experts local, intermediate
576 padded to 640) replaces FlashInfer's TRT-LLM-gen routing + two batched GEMMs by a
route kernel and a GEMM pair tuned for at most eight rows (128 x 8 tcgen05 block-scaled
tiles, persistent schedule, PDL), compiled with NVRTC on first use. The kernels read the
same shuffled MXFP4 weights and linear-layout MXFP8 activations as FlashInfer and return
its deferred-finalize triple (permuted GEMM2 rows, routing weights, slot -> row map), so
the fused finalize + all-reduce + mHC consumer is unchanged.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
from functools import cache
from pathlib import Path
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)

SOURCE_DIR = Path(__file__).resolve().parent / "dsv41_routed_expert_gemm_src"

HIDDEN = 5120
SLICE = 576
PADDED = 640
MAX_EXPERTS = 384
MAX_TOP_K = 8
TILE = 8
MAX_ROWS = 8
GEMM1_UNITS = 9  # 128-row tiles of w13 holding the 576-wide slice (10th is padding)
GEMM2_UNITS = HIDDEN // 128
ROUTE_THREADS = 64
FINALIZE_THREADS = 256  # one thread per output value, HIDDEN / 256 CTAs per row
NVRTC_OPTIONS = ("-std=c++17", "-default-device", "--use_fast_math")
HEADERS = ("route.cuh",)
GEMM1_PARAMETERS = ("A", "SFA", "X", "XS", "route", "clamp", "H", "HS", "cta_words")
GEMM2_PARAMETERS = ("A", "SFA", "H", "HS", "route", "Y", "cta_words")
ADMISSION_PARAMETERS = ("barrier", "ready")
PARAMETERS = {
    "gemm1": GEMM1_PARAMETERS,
    "gemm2": GEMM2_PARAMETERS,
    "gemm1_admit": GEMM1_PARAMETERS + ADMISSION_PARAMETERS,
    "gemm2_admit": GEMM2_PARAMETERS + ADMISSION_PARAMETERS,
}
ADMIT_SLOTS = 64


def max_ctas(rows: int, top_k: int, experts: int) -> int:
    """The most 8-row token tiles any routing of ``rows * top_k`` slots needs."""
    slots = rows * top_k
    filled = min(experts, slots)
    return filled + (slots - filled) // TILE


def cta_words(ctas: int) -> int:
    return (ctas + 3) // 4 * 4


def route_words(ctas: int) -> int:
    """int32 length of the route record for ``ctas`` token tiles."""
    return 4 + 2 * cta_words(ctas) + TILE * ctas


def _driver():
    from cuda.bindings import driver

    return driver


def _check(result: tuple, what: str) -> tuple:
    if int(result[0]) != 0:
        raise RuntimeError(f"{what} failed: {result[0]!r}")
    return result[1:]


def _include_dirs() -> list[str]:
    candidates: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "include")
    candidates.append(Path("/usr/local/cuda/include"))
    try:
        import nvidia

        for base in nvidia.__path__:
            candidates.extend(sorted(Path(base).glob("*/include")))
    except ImportError:
        pass
    found: list[str] = []
    for directory in candidates:
        if (directory / "cuda.h").is_file() and str(directory) not in found:
            found.append(str(directory))
            if (directory / "cccl" / "cuda" / "std").is_dir():
                found.append(str(directory / "cccl"))
    return found


def nvrtc_version() -> tuple[int, int]:
    from cuda.bindings import nvrtc

    major, minor = _check(nvrtc.nvrtcVersion(), "nvrtcVersion")
    return int(major), int(minor)


def _cache_dir() -> Path:
    root = os.environ.get("SGLANG_CACHE_DIR") or os.path.expanduser("~/.cache/sglang")
    return Path(root) / "dsv41_routed_expert_gemm"


def compile_source(
    source: str,
    name: str,
    arch: str,
    options: tuple[str, ...] = NVRTC_OPTIONS,
    headers: Optional[dict[str, str]] = None,
) -> bytes:
    """Compile ``source`` to a cubin for ``arch`` with NVRTC, cached on disk by the
    hash of source, headers, options and NVRTC version."""
    from cuda.bindings import nvrtc

    headers = dict(headers or {})
    options = [f"--gpu-architecture={arch}", *options]
    options += [f"-I{d}" for d in _include_dirs()]
    if nvrtc_version() >= (12, 9):
        options.append("--no-cache")
    header_text = [f"{k}\0{v}" for k, v in sorted(headers.items())]
    key = hashlib.sha256(
        "\0".join([source, *header_text, *options, repr(nvrtc_version())]).encode()
    ).hexdigest()
    cached = _cache_dir() / f"{name}-{key[:24]}.cubin"
    if cached.is_file():
        return cached.read_bytes()
    (program,) = _check(
        nvrtc.nvrtcCreateProgram(
            source.encode(),
            f"{name}.cu".encode(),
            len(headers),
            [text.encode() for text in headers.values()],
            [header.encode() for header in headers],
        ),
        "nvrtcCreateProgram",
    )
    try:
        encoded = [option.encode() for option in options]
        (status,) = nvrtc.nvrtcCompileProgram(program, len(encoded), encoded)
        if int(status) != 0:
            (size,) = _check(nvrtc.nvrtcGetProgramLogSize(program), "log size")
            log = b" " * size
            _check(nvrtc.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
            raise RuntimeError(f"NVRTC failed to compile {name}: {log.decode(errors='replace')}")
        (size,) = _check(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
        cubin = b" " * size
        _check(nvrtc.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(program)
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(cubin)
    temporary.replace(cached)
    return cubin


class _Kernel(NamedTuple):
    function: object
    block: tuple[int, int, int]
    shared_memory_bytes: int
    cubin: bytes


def _load(cubin: bytes, symbol: str, block, shared: int) -> _Kernel:
    cu = _driver()
    (module,) = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
    (function,) = _check(cu.cuModuleGetFunction(module, symbol.encode()), symbol)
    if shared > 48 * 1024:
        _check(
            cu.cuFuncSetAttribute(
                function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            ),
            "cuFuncSetAttribute",
        )
    return _Kernel(function, tuple(block), shared, cubin)


@cache
def load_kernels(device_index: int) -> dict[str, _Kernel]:
    """Compile (or read from the cache) and load the route kernel and GEMMs."""
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        major, minor = torch.cuda.get_device_capability(device)
        arch = f"sm_{major}{minor}a"
        facts = json.loads((SOURCE_DIR / "launch.json").read_text())
        headers = {name: (SOURCE_DIR / name).read_text() for name in HEADERS}
        kernels = {
            "route": _load(
                compile_source((SOURCE_DIR / "route.cu").read_text(), "route", arch, headers=headers),
                "dsv41_routed_expert_route",
                (ROUTE_THREADS, 1, 1),
                0,
            )
        }
        for name, fact in facts.items():
            if tuple(fact["parameters"]) != PARAMETERS.get(name):
                raise RuntimeError(f"{name}: unexpected launch.json parameters {fact['parameters']}")
            source = (SOURCE_DIR / f"{name}.cu").read_text()
            kernels[name] = _load(
                compile_source(source, name, arch),
                fact["symbol"],
                fact["block"],
                int(fact["shared_memory_bytes"]),
            )
        return kernels


@cache
def load_finalize(device_index: int) -> _Kernel:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        torch.zeros(1, device=device)
        major, minor = torch.cuda.get_device_capability(device)
        source = (SOURCE_DIR / "finalize.cu").read_text()
        cubin = compile_source(source, "finalize", f"sm_{major}{minor}a")
        return _load(cubin, "dsv41_routed_expert_finalize", (FINALIZE_THREADS, 1, 1), 0)


def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _launch(kernel: _Kernel, grid: tuple[int, int, int], arguments: list, stream: int) -> None:
    """Launch with programmatic stream serialization (PDL); graph-capturable."""
    cu = _driver()
    addresses = [
        argument.getPtr() if isinstance(argument, cu.CUtensorMap) else ctypes.addressof(argument)
        for argument in arguments
    ]
    pointer_array = (ctypes.c_void_p * len(addresses))(*addresses)
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = grid
    config.blockDimX, config.blockDimY, config.blockDimZ = kernel.block
    config.sharedMemBytes = kernel.shared_memory_bytes
    config.hStream = cu.CUstream(stream)
    attribute = cu.CUlaunchAttribute()
    attribute.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    attribute.value.programmaticStreamSerializationAllowed = 1
    config.attrs = [attribute]
    config.numAttrs = 1
    _check(
        cu.cuLaunchKernelEx(config, kernel.function, ctypes.addressof(pointer_array), 0),
        "cuLaunchKernelEx",
    )


def _weight_map(weights: torch.Tensor, m: int, k: int, chunks: int):
    """TMA map of MXFP4 weights ``[E, m, k / 2]`` (16U4, 128B swizzle)."""
    cu = _driver()
    if weights.data_ptr() % 32:
        raise ValueError("packed expert weights must be 32-byte aligned")
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
            4,
            weights.data_ptr(),
            [u64(128), u64(m), u64(k // 128), u64(weights.size(0))],
            [u64(k // 2), u64(64), u64(m * k // 2)],
            [u32(128), u32(128), u32(chunks), u32(1)],
            [u32(1)] * 4,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(weights)",
    )
    return tensor_map


def _scale_map(scales: torch.Tensor, groups: int):
    """TMA map of 128x4-layout scale bytes viewed as ``[bytes/256, 256]``."""
    cu = _driver()
    total = scales.numel() * scales.element_size()
    if total % 256 or scales.data_ptr() % 16:
        raise ValueError("weight scales must be whole 256-byte rows, 16-byte aligned")
    u64, u32 = cu.cuuint64_t, cu.cuuint32_t
    (tensor_map,) = _check(
        cu.cuTensorMapEncodeTiled(
            cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            scales.data_ptr(),
            [u64(256), u64(total // 256)],
            [u64(256)],
            [u32(256), u32(2 * groups)],
            [u32(1)] * 2,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        ),
        "cuTensorMapEncodeTiled(scales)",
    )
    return tensor_map


def check_weights(w13, w13_scale, w2, w2_scale, clamp, top_k: int) -> None:
    """Raise ValueError unless the weights have the layout the kernels read."""
    experts = w13.size(0)
    expected = (
        (w13, (experts, 2 * PADDED, HIDDEN // 2)),
        (w13_scale, (experts, 2 * PADDED, HIDDEN // 32)),
        (w2, (experts, HIDDEN, PADDED // 2)),
        (w2_scale, (experts, HIDDEN, PADDED // 32)),
    )
    for tensor, shape in expected:
        if tuple(tensor.shape) != shape or tensor.element_size() != 1 or not tensor.is_contiguous():
            raise ValueError(
                f"expected a contiguous {shape} 1-byte tensor, got {tuple(tensor.shape)} {tensor.dtype}"
            )
    if experts % 32 or not 0 < experts <= MAX_EXPERTS:
        raise ValueError(f"expected a multiple of 32 local experts <= {MAX_EXPERTS}, got {experts}")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"expected a top-k of 1-{MAX_TOP_K}, got {top_k}")
    if clamp is None or clamp.dtype != torch.float32 or tuple(clamp.shape) != (experts,):
        raise ValueError("expected an FP32 [E] SwiGLU clamp limit")
    # GEMM1 never reads w13's padding tile, so the padding must be zero, as must the
    # matching K columns of w2.
    if (
        torch.count_nonzero(w13.view(torch.uint8)[:, 2 * SLICE :]).item()
        or torch.count_nonzero(w2.view(torch.uint8)[:, :, SLICE // 2 :]).item()
    ):
        raise ValueError("the intermediate padding of w13/w2 is not zero")


class RoutedExpertGemm:
    """The route kernel and GEMM pair bound to one MoE layer's weights."""

    def __init__(
        self, w13, w13_scale, w2, w2_scale, clamp, admission: bool, top_k: int, finalize: bool
    ):
        w13, w13_scale = w13.view(torch.uint8), w13_scale.view(torch.uint8)
        w2, w2_scale = w2.view(torch.uint8), w2_scale.view(torch.uint8)
        check_weights(w13, w13_scale, w2, w2_scale, clamp, top_k)
        self.experts = int(w13.size(0))
        self.top_k = top_k
        self.kernels = load_kernels(w13.device.index)
        suffix = "_admit" if admission else ""
        self.admission = admission
        self.gemm1 = self.kernels[f"gemm1{suffix}"]
        self.gemm2 = self.kernels[f"gemm2{suffix}"]
        # Non-deferred calls (local finalize) take the pair + finalize.cu only with this set.
        self.finalize_kernel = load_finalize(w13.device.index) if finalize else None
        # Bit 31 of `barrier` is flipped by every route launch; `ready` holds the
        # admission counters of both parities (one route -> one GEMM1, in order).
        self.barrier = torch.zeros(1, dtype=torch.int32, device=w13.device)
        self.ready = torch.zeros(2 * ADMIT_SLOTS, dtype=torch.int32, device=w13.device)
        self._tensors = (w13, w13_scale, w2, w2_scale, clamp)  # the maps hold their addresses
        self.clamp = clamp
        self.gemm1_maps = (_weight_map(w13, 2 * PADDED, HIDDEN, 2), _scale_map(w13_scale, 4))
        self.gemm2_maps = (_weight_map(w2, HIDDEN, PADDED, 2), _scale_map(w2_scale, 1))

    def supports(self, x_quant, x_scale, topk_ids, topk_weights) -> bool:
        rows = x_quant.size(0)
        return (
            1 <= rows <= MAX_ROWS
            and topk_ids.dim() == 2
            and topk_ids.size(1) == self.top_k
            and tuple(x_quant.shape) == (rows, HIDDEN)
            and x_quant.element_size() == 1
            and x_quant.is_contiguous()
            and x_scale is not None
            and x_scale.element_size() == 1
            and x_scale.is_contiguous()
            and x_scale.numel() >= rows * HIDDEN // 32
            and topk_weights.dtype == torch.float32
        )

    def __call__(self, x_quant, x_scale, topk_ids, topk_weights):
        """Route + GEMM1 + GEMM2 on the current stream; returns FlashInfer's
        deferred-finalize triple (gemm2 rows, weights, slot -> permuted row)."""
        rows, top_k = topk_ids.shape
        device = x_quant.device
        stream = torch.cuda.current_stream(device).cuda_stream
        ids = topk_ids.to(torch.int32).contiguous()
        ctas = max_ctas(rows, top_k, self.experts)
        permuted = torch.empty((rows, top_k), dtype=torch.int32, device=device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=device)
        _launch(
            self.kernels["route"],
            (1, 1, 1),
            [
                _pointer(ids),
                _pointer(permuted),
                _pointer(record),
                _pointer(self.barrier),
                ctypes.c_int32(rows),
                ctypes.c_int32(top_k),
                ctypes.c_int32(self.experts),
            ],
            stream,
        )
        padded_rows = TILE * ctas
        h_values = torch.empty((padded_rows, PADDED), dtype=torch.uint8, device=device)
        h_scales = torch.empty(
            (padded_rows + 127) // 128 * 128 * PADDED // 32, dtype=torch.uint8, device=device
        )
        gemm2_rows = torch.empty((rows * top_k * TILE, HIDDEN), dtype=torch.bfloat16, device=device)
        words = ctypes.c_int32(cta_words(ctas))
        admission = [_pointer(self.barrier), _pointer(self.ready)] if self.admission else []
        _launch(
            self.gemm1,
            (GEMM1_UNITS, ctas, 1),
            [
                *self.gemm1_maps,
                _pointer(x_quant),
                _pointer(x_scale),
                _pointer(record),
                _pointer(self.clamp),
                _pointer(h_values),
                _pointer(h_scales),
                words,
                *admission,
            ],
            stream,
        )
        _launch(
            self.gemm2,
            (GEMM2_UNITS, ctas, 1),
            [
                *self.gemm2_maps,
                _pointer(h_values),
                _pointer(h_scales),
                _pointer(record),
                _pointer(gemm2_rows),
                words,
                *admission,
            ],
            stream,
        )
        return gemm2_rows, topk_weights.contiguous().view(rows, top_k), permuted

    def finalize(self, gemm2_rows, expert_weights, permuted, output) -> None:
        """output[t] = BF16(sum_k w[t, k] * gemm2_rows[permuted[t, k]]), FlashInfer's
        small-batch finalize arithmetic; output is BF16 [rows, 5120], contiguous."""
        rows, top_k = expert_weights.shape
        _launch(
            self.finalize_kernel,
            (HIDDEN // FINALIZE_THREADS, rows, 1),
            [
                _pointer(gemm2_rows),
                _pointer(permuted),
                _pointer(expert_weights),
                _pointer(output),
                ctypes.c_int32(top_k),
            ],
            torch.cuda.current_stream(output.device).cuda_stream,
        )


def draft_graph_rows(capture_bs: list[int], rows_per_request: int) -> Optional[int]:
    """Rows of the smallest draft CUDA graph (``None`` without graphs)."""
    return min(capture_bs) * rows_per_request if capture_bs else None


def check_draft_graph_rows(capture_bs: list[int], rows_per_request: int) -> bool:
    """Warn when SGLANG_OPT_DSV41_DRAFT_FAST_PATHS cannot serve one request: the
    draft kernels take at most MAX_ROWS rows (muLLM df8786f6 check_draft_graph_rows)."""
    smallest = draft_graph_rows(capture_bs, rows_per_request)
    if smallest is None or smallest <= MAX_ROWS:
        return True
    logger.warning(
        f"SGLANG_OPT_DSV41_DRAFT_FAST_PATHS is set, but the smallest DSpark draft CUDA graph "
        f"has {smallest} rows ({rows_per_request} per request), above the {MAX_ROWS} the "
        f"draft fast paths take; add batch size 1 to the decode CUDA graph sizes."
    )
    return False


def maybe_install(layer, w13, w13_scale, w2, w2_scale, clamp, top_k: int) -> bool:
    """Bind a RoutedExpertGemm to ``layer`` (attribute ``dsv41_routed_expert_gemm``)
    when the flag is on and the layer is a TP MoE the kernels take; log why not.

    DSpark draft layers (``dsv41_draft_layer``, set by the draft stage) follow
    SGLANG_OPT_DSV41_DRAFT_FAST_PATHS instead: they always finalize locally, so they take
    the pair and the finalize kernel (muLLM df8786f6, maybe_enable_draft_routed_expert_gemm).
    """
    from sglang.srt.environ import envs

    draft = getattr(layer, "dsv41_draft_layer", False)
    if draft:
        finalize = True
        if not envs.SGLANG_OPT_DSV41_DRAFT_FAST_PATHS.get():
            return False
    else:
        finalize = envs.SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM_FINALIZE.get()
        if not envs.SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM.get():
            return False
    if layer.num_local_experts != layer.num_experts:
        reason = "expert parallelism (needs --ep-size 1)"
    else:
        try:
            layer.dsv41_routed_expert_gemm = RoutedExpertGemm(
                w13,
                w13_scale,
                w2,
                w2_scale,
                clamp,
                admission=envs.SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM_ADMIT.get(),
                top_k=top_k,
                finalize=finalize,
            )
            return True
        except (ValueError, ImportError, OSError, RuntimeError) as error:
            reason = f"{type(error).__name__}: {error}"
    where = " in the DSpark draft" if draft else ""
    logger.warning(f"DSV4.1 routed-expert GEMMs not installed{where}, FlashInfer kept: {reason}")
    return False
