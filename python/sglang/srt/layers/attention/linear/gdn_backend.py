from typing import Optional, Tuple, Union

import msgspec
import torch

from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnBackends,
    LinearAttnKernelBackend,
    build_verify_intermediate_state_indices,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_exec, get_memory, get_schedule
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import rank0_log

_is_hip = is_hip()

if not is_cpu():
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda() or is_hip() or is_xpu():
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        can_use_fused_qkvzba_causal_conv1d_update_contiguous,
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_causal_conv1d_update_contiguous,
        fused_qkvzba_split_reshape_cat_contiguous,
    )

MAX_FUSED_QKV_SPLIT_DIM = 8192
# TwinStar K1 debug hooks (docs/62 §2): teacher-forced "decode via extend" and state dumps.
#   SGLANG_GDN_EXTEND_STEPWISE_MIN_PREFIX=N  extend batches whose every row has prefix >= N run the recurrence token by
#       token (flag off: the stock recurrent decode kernel; flag on: the factored step kernel on the pool slots) instead
#       of the chunk kernel -- the served decode path's maths, driven by the engine's prefix-hit extend.
#   SGLANG_GDN_EXTEND_STEPWISE_FLAGFILE=path  the hook is active only while this file exists (arm switch at run time).
#   SGLANG_GDN_FACTORED_DUMP=dir             dump the per-layer final states of every extend (dense or factored).
import os as _os

_STEPWISE_MIN_PREFIX = int(_os.environ.get("SGLANG_GDN_EXTEND_STEPWISE_MIN_PREFIX", "0") or 0)
_STEPWISE_FLAGFILE = _os.environ.get("SGLANG_GDN_EXTEND_STEPWISE_FLAGFILE") or None
_FACTORED_DUMP_DIR = _os.environ.get("SGLANG_GDN_FACTORED_DUMP") or None
_DUET_CAPTURE_DIR = _os.environ.get("SGLANG_GDN_DUET_CAPTURE") or None
_stepwise_logged = set()
_fused_decode_proj_conv_logged = False
_fused_decode_proj_conv_fallback_logged = False
_fused_decode_proj_conv_layers_logged: set[int] = set()
_fused_decode_real_tensor_verified_layers: set[int] = set()
_fused_decode_log_layer_hits = envs.SGLANG_GDN_DECODE_FUSION_LOG_LAYER_HITS.get()
_fused_decode_verify_real_tensors = (
    envs.SGLANG_GDN_DECODE_FUSION_VERIFY_REAL_TENSORS.get()
)


class GDNMISMetadata(msgspec.Struct, frozen=True):
    query_token_indices: torch.Tensor
    query_cu_seqlens: torch.Tensor
    query_seq_lens_cpu: list[int]
    query_request_indices: torch.Tensor
    item_token_indices: torch.Tensor
    item_cu_seqlens: torch.Tensor
    item_seq_lens_cpu: list[int]
    item_request_indices: torch.Tensor


def build_gdn_mis_metadata(forward_batch: ForwardBatch) -> GDNMISMetadata:
    """Build compact query/item segments from request-local MIS delimiters."""
    if not forward_batch.is_prefill_only:
        raise ValueError("GDN MIS is only supported for prefill-only requests")

    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if isinstance(prefix_lens, torch.Tensor):
        prefix_lens = prefix_lens.tolist()
    if any(int(prefix_len) != 0 for prefix_len in prefix_lens):
        raise ValueError("GDN MIS does not support cached prefixes")

    seq_lens = forward_batch.extend_seq_lens_cpu
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.tolist()
    seq_lens = [int(seq_len) for seq_len in seq_lens]
    delimiter_indices = forward_batch.multi_item_delimiter_indices
    if delimiter_indices is None or len(delimiter_indices) != len(seq_lens):
        raise ValueError("GDN MIS requires delimiter indices for every request")
    if sum(seq_lens) > forward_batch.input_ids.numel():
        raise ValueError("GDN MIS sequence lengths exceed the input tokens")

    query_token_indices: list[int] = []
    query_seq_lens_cpu: list[int] = []
    query_request_indices: list[int] = []
    item_token_indices: list[int] = []
    item_seq_lens_cpu: list[int] = []
    item_request_indices: list[int] = []

    request_start = 0
    for request_idx, (seq_len, request_delimiters) in enumerate(
        zip(seq_lens, delimiter_indices)
    ):
        delimiters = [int(index) for index in request_delimiters.tolist()]
        if len(delimiters) < 2:
            raise ValueError("GDN MIS requires at least two delimiters per request")
        if any(
            current >= following
            for current, following in zip(delimiters, delimiters[1:])
        ):
            raise ValueError("GDN MIS delimiter indices must be strictly increasing")
        if delimiters[0] < 0 or delimiters[-1] >= seq_len:
            raise ValueError("GDN MIS delimiter index is outside the request")
        if delimiters[-1] != seq_len - 1:
            raise ValueError("GDN MIS final delimiter must be the last request token")

        query_len = delimiters[0]
        if query_len > 0:
            query_token_indices.extend(range(request_start, request_start + query_len))
            query_seq_lens_cpu.append(query_len)
            query_request_indices.append(request_idx)

        branch_ends = delimiters[1:] + [seq_len]
        for branch_start, branch_end in zip(delimiters, branch_ends):
            branch_len = branch_end - branch_start
            item_token_indices.extend(
                range(request_start + branch_start, request_start + branch_end)
            )
            item_seq_lens_cpu.append(branch_len)
            item_request_indices.append(request_idx)

        request_start += seq_len

    device = forward_batch.input_ids.device

    def _indices(values: list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
        result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
        if lengths:
            result[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(
                dim=0
            )
        return result

    return GDNMISMetadata(
        query_token_indices=_indices(query_token_indices, torch.int64),
        query_cu_seqlens=_cu_seqlens(query_seq_lens_cpu),
        query_seq_lens_cpu=query_seq_lens_cpu,
        query_request_indices=_indices(query_request_indices, torch.int64),
        item_token_indices=_indices(item_token_indices, torch.int64),
        item_cu_seqlens=_cu_seqlens(item_seq_lens_cpu),
        item_seq_lens_cpu=item_seq_lens_cpu,
        item_request_indices=_indices(item_request_indices, torch.int64),
    )


def validate_gdn_mis_backend(prefill_backend: LinearAttnKernelBackend) -> None:
    if not get_exec().features.enable_mis:
        return
    if not prefill_backend.is_triton():
        raise ValueError(
            "GDN multi-item scoring requires the Triton linear-attention prefill "
            "backend. Set --linear-attn-prefill-backend triton."
        )
    if get_memory().enable_page_major_kv_layout:
        raise ValueError("GDN multi-item scoring does not support page-major layout")


if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


def flashinfer_gdn_prefill_default(model_runner: ModelRunner) -> Optional[str]:
    """FlashInfer for the narrow SM90/SM100 GDN prefill domains we validated, else None."""
    sm_major = torch.cuda.get_device_capability()[0] if is_cuda() else 0
    if (
        get_exec().mamba.linear_attn_prefill_backend is not None
        or get_exec().mamba.linear_attn_backend != "triton"
        or get_exec().deterministic.enable_deterministic_inference
        or get_memory().enable_page_major_kv_layout
        or sm_major not in (9, 10)
    ):
        return None

    # SM100 runs the CUDA>=13 CuTe-DSL chunk kernel on a bf16 state pool;
    # SM90 runs the fused Hopper kernel on an fp32 state pool and tolerates
    # larger chunks. Everything outside these validated domains keeps Triton.
    cuda_version = torch.version.cuda
    if sm_major == 10:
        if cuda_version is None or int(cuda_version.split(".", 1)[0]) < 13:
            return None
        max_chunk = 8192
        expected_state_dtype = torch.bfloat16
    else:
        max_chunk = 32768
        expected_state_dtype = torch.float32

    chunk_size = get_schedule().chunked_prefill_size
    config = hybrid_gdn_config(model_runner.model_config)
    if (
        get_schedule().enable_dynamic_chunking
        or chunk_size is None
        or not 1 <= chunk_size <= max_chunk
        or getattr(config, "linear_key_head_dim", None) != 128
        or getattr(config, "linear_value_head_dim", None) != 128
        or model_runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype
        != expected_state_dtype
    ):
        return None

    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )

    if not is_flashinfer_gdn_prefill_available():
        return None

    rank0_log(f"Defaulting SM{sm_major}0 GDN prefill backend to FlashInfer.")
    return "flashinfer"


def _validate_gdn_linear_attn_backends(backends: LinearAttnBackends) -> None:
    if (
        get_exec().deterministic.enable_deterministic_inference
        and backends.prefill.is_flashinfer()
    ):
        raise ValueError(
            "FlashInfer GDN prefill is not supported with "
            "--enable-deterministic-inference. Use "
            "--linear-attn-prefill-backend triton."
        )


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
        verify_backend: Optional[LinearAttnKernelBackend] = None,
    ):
        triton_kernel = TritonGDNKernel()
        self.tree_verify_kernel = triton_kernel

        cutedsl_kernel = None
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_intel_xpu():
            if not is_xpu():
                raise ValueError("--linear-attn-backend intel_xpu requires Intel XPU")
            # The fused SYCL kernel is dispatched via XpuGDNAttnBackend.forward_fused_gdn,
            # outside this dispatcher; Triton is the dispatcher-level kernel for requests
            # that hook doesn't handle (e.g. verify).
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            cutedsl_kernel = CuteDSLGDNKernel()
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        elif decode_backend.is_helion():
            raise ValueError(
                "The Helion linear-attention backend supports KDA only, not GDN."
            )
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_intel_xpu():
            if not is_xpu():
                raise ValueError("--linear-attn-backend intel_xpu requires Intel XPU")
            # See the decode branch above: intel_xpu uses Triton as its
            # dispatcher-level fallback kernel.
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            # Reuse the CuteDSL kernel if already created for decode
            if cutedsl_kernel is None:
                from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                    CuteDSLGDNKernel,
                )

                cutedsl_kernel = CuteDSLGDNKernel()
            # The CuteDSL prefill kernel only exists on SM100+ (Blackwell).
            # On SM90 (Hopper) fall back to Triton so users can pick
            # `cutedsl` uniformly across hardware.
            if cutedsl_kernel.supports_prefill:
                self.extend_kernel = cutedsl_kernel
            else:
                rank0_log(
                    "CuTe DSL GDN prefill is not supported on this GPU "
                    "(requires SM100+). Falling back to Triton for prefill."
                )
                self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        elif prefill_backend.is_helion():
            raise ValueError(
                "The Helion linear-attention backend supports KDA only, not GDN."
            )
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel. An explicitly configured verify backend wins; the
        # historical auto rule (FlashInfer when the selected FlashInfer kernel
        # supports MTP verify) only applies when no explicit choice was made.
        # SM90 FlashInfer verify requires a fp32 SSM state, so e.g.
        # --mamba-ssm-dtype bfloat16 setups must be able to force Triton here.
        if verify_backend is not None and verify_backend.is_triton():
            self.verify_kernel = triton_kernel
            self.verify_kernel_is_flashinfer = False
        elif (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
            self.verify_kernel_is_flashinfer = True
        else:
            self.verify_kernel = triton_kernel
            self.verify_kernel_is_flashinfer = False

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    @property
    def extend_uses_state_checkpoints(self) -> bool:
        return self.extend_kernel.uses_state_checkpoints

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # FlashInfer verify supports a linear MTP chain. Tree-shaped drafts
        # carry parent indices and must use Triton even when decode/prefill use
        # FlashInfer.
        verify_kernel = (
            self.tree_verify_kernel
            if kwargs.get("retrieve_parent_token") is not None
            else self.verify_kernel
        )
        return verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    needs_cpu_seq_lens: bool = False
    supports_mis: bool = True

    def __init__(self, model_runner: ModelRunner):
        _validate_gdn_linear_attn_backends(model_runner.linear_attn_backends)
        super().__init__(model_runner)
        self.enable_mis = get_exec().features.enable_mis
        self.mis_metadata: Optional[GDNMISMetadata] = None
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert self.conv_states_shape[-1] < FLA_CHUNK_SIZE, (
                f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"
            )

        backends = model_runner.linear_attn_backends
        validate_gdn_mis_backend(backends.prefill)
        self.linear_attn_backends = backends
        self.kernel_dispatcher = GDNKernelDispatcher(
            backends.decode, backends.prefill, backends.verify
        )
        # Sized past the pool for attn_tp-padded warmup/MLP-sync batches (see helper).
        self.verify_intermediate_state_indices = (
            build_verify_intermediate_state_indices(
                self.req_to_token_pool.size,
                model_runner.device,
            )
        )
        # TwinStar factored GDN state (docs/62): the FactoredGDNPool sibling of the
        # mamba pool, or None (stock dense path, byte-identical).
        self.factored = getattr(self.req_to_token_pool, "factored_gdn_pool", None)
        self._factored_side_stream = None
        self._factored_batch_trunc = (
            self.factored is not None
            and self.factored.cfg.r in (8, 16)
            and (self.factored.cfg.strict_chunk or _os.environ.get("SGLANG_GDN_FACTORED_BATCH_LAYERS", "0") == "1")
        )
        if self._factored_batch_trunc:
            if not self.factored.cfg.use_async_trunc:
                raise ValueError("batched layer expiry requires the split post-order path")
            if self.factored.cfg.r == 16 and (_os.environ.get("SGLANG_GDN_FACTORED_TRUNC_METHOD") != "tensor"
                or _os.environ.get("SGLANG_GDN_FACTORED_TENSOR_WHOLE") != "1"
                or _os.environ.get("SGLANG_GDN_FACTORED_LU") != "1"):
                raise ValueError("batched layer expiry requires the validated whole LU kernel")
            if self.factored.cfg.r == 8 and _os.environ.get("SGLANG_GDN_FACTORED_TRUNC_METHOD", "mgs") not in ("mgs", "tensor"):
                raise ValueError("r8 batched layer expiry requires the MGS path")
        if self.factored is not None:
            if self.factored.cfg.use_async_trunc and not self._factored_batch_trunc:
                # K2 (docs/63 §4): the slot-expiry truncation runs on this stream after each layer's step and is joined
                # at the last GDN layer of the forward (inside CUDA-graph captures)
                self._factored_side_stream = torch.cuda.Stream()
            rank0_log(
                "GDN backend: factored decode state ON "
                f"(r={self.factored.cfg.r}, m={self.factored.cfg.m}, "
                f"RMAX={self.factored.cfg.rmax}, factors={self.factored.cfg.dtype}, "
                f"ring={self.factored.cfg.ring}, kernel={self.factored.cfg.kernel or 'default'}, "
                f"async_trunc={self.factored.cfg.use_async_trunc}, "
                f"batch_layer_trunc={self._factored_batch_trunc}, "
                f"trunc_warps={self.factored.cfg.trunc_warps}, trunc_iters={self.factored.cfg.trunc_iters}, "
                f"fused_warps={self.factored.cfg.fused_warps})"
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if (
            self.factored is not None
            and forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.extend_seq_lens_cpu is not None
        ):
            # Host-side dense-ring plan (one D2H sync per extend forward), shared by
            # every GDN layer of this forward (docs/62 §1.3).
            self.forward_metadata.factored_extend = self.factored.plan_extend(
                self.forward_metadata.mamba_cache_indices,
                forward_batch.extend_seq_lens_cpu,
                prefix_lens=forward_batch.extend_prefix_lens_cpu,
                prompt_final=getattr(forward_batch, "twinstar_prompt_final", None),
                layer_range=getattr(forward_batch, "flashnext_gdn_layer_range", None),
            )
        self.mis_metadata = None
        if forward_batch.multi_item_delimiter_indices is not None:
            if not self.enable_mis:
                raise ValueError("GDN MIS metadata requires --enable-mis")
            self.mis_metadata = build_gdn_mis_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )
            if self.kernel_dispatcher.extend_uses_state_checkpoints:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    maybe_build_flashinfer_checkpoint_plan,
                )

                maybe_build_flashinfer_checkpoint_plan(
                    forward_batch, self.forward_metadata, self.device
                )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        global _fused_decode_proj_conv_fallback_logged
        global _fused_decode_proj_conv_logged
        global _fused_decode_proj_conv_layers_logged
        global _fused_decode_real_tensor_verified_layers

        if _is_hip and isinstance(mixed_qkv, torch.Tensor) and mixed_qkv.shape[0] == 0:
            return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
        # per-row write cursor. All None unless --enable-linear-replayssm, so the
        # legacy dispatch below is byte-identical when the flag is off.
        replayssm_write_pos = self.forward_metadata.replayssm_write_pos
        # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
        # boundaries (None unless --enable-linear-replayssm). When present the
        # kernel folds the ring into temporal[slot] on the snapshot steps.
        replayssm_force_flush = self.forward_metadata.replayssm_force_flush
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        return_z = False
        conv_already_applied = False
        if isinstance(mixed_qkv, tuple):
            if len(mixed_qkv) != 2:
                raise ValueError(
                    "Fused GDN decode projection input must be "
                    "(projected_qkvz, projected_ba)"
                )
            projected_qkvz, projected_ba = mixed_qkv
            eligible, eligibility_reason = (
                can_use_fused_qkvzba_causal_conv1d_update_contiguous(
                    projected_qkvz,
                    projected_ba,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    cache_indices,
                    qkv_dim=layer.q_dim + layer.k_dim + layer.v_dim,
                    v_dim=layer.v_dim,
                    num_v_heads=layer.num_v_heads,
                    activation=layer.activation,
                )
            )
            if eligible:
                qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
                fused_backend = "triton_direct_oracle_exact"
                if not _fused_decode_proj_conv_logged:
                    rank0_log("Using fused GDN decode QKVZ/BA unpack + indexed Conv1D.")
                    _fused_decode_proj_conv_logged = True
                if (
                    _fused_decode_log_layer_hits or _fused_decode_verify_real_tensors
                ) and layer.layer_id not in _fused_decode_proj_conv_layers_logged:
                    rank0_log(
                        "GDN_FUSED_DECODE_BACKEND "
                        f"layer_id={layer.layer_id} backend={fused_backend} "
                        f"batch={projected_qkvz.shape[0]} "
                        f"qkv_dim={qkv_dim} state_shape={tuple(conv_states.shape)} "
                        f"state_indices_dtype={cache_indices.dtype}"
                    )
                    _fused_decode_proj_conv_layers_logged.add(layer.layer_id)

                # Compare real activations against the direct-Triton update on a
                # compact state copy, leaving the live cache to the candidate.
                verify_real_tensors = (
                    _fused_decode_verify_real_tensors
                    and layer.layer_id not in _fused_decode_real_tensor_verified_layers
                )
                if verify_real_tensors:
                    if bool(torch.any(cache_indices < 0).item()):
                        raise AssertionError(
                            "Real-tensor GDN fusion verification requires "
                            "non-padding cache indices"
                        )
                    ref_indices = torch.arange(
                        cache_indices.numel(),
                        device=cache_indices.device,
                        dtype=torch.int32,
                    )
                    ref_state = torch.index_select(
                        conv_states, 0, cache_indices.to(torch.int64)
                    )
                    ref_mixed_qkv, ref_z, ref_b, ref_a = (
                        fused_qkvzba_split_reshape_cat_contiguous(
                            projected_qkvz,
                            projected_ba,
                            layer.num_q_heads,
                            layer.num_v_heads,
                            layer.head_q_dim,
                            layer.head_v_dim,
                        )
                    )
                    ref_mixed_qkv = causal_conv1d_update(
                        ref_mixed_qkv,
                        ref_state,
                        layer.conv_weights,
                        layer.bias,
                        layer.activation,
                        conv_state_indices=ref_indices,
                    )

                mixed_qkv, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
                    projected_qkvz,
                    projected_ba,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    cache_indices,
                    qkv_dim=qkv_dim,
                    v_dim=layer.v_dim,
                    num_v_heads=layer.num_v_heads,
                    head_v_dim=layer.head_v_dim,
                    activation=layer.activation,
                )
                if verify_real_tensors:
                    candidate_state = torch.index_select(
                        conv_states, 0, cache_indices.to(torch.int64)
                    )
                    named_pairs = (
                        ("qkv", mixed_qkv, ref_mixed_qkv),
                        ("z", z, ref_z),
                        ("b", b, ref_b),
                        ("a", a, ref_a),
                        ("state", candidate_state, ref_state),
                    )
                    report = []
                    mismatch = False
                    for tensor_name, candidate, reference in named_pairs:
                        diff = (candidate.float() - reference.float()).abs()
                        nonzero = int(torch.count_nonzero(diff).item())
                        mismatch |= nonzero != 0
                        report.append(
                            f"{tensor_name}_nonzero={nonzero}/"
                            f"{diff.numel()} {tensor_name}_max="
                            f"{diff.max().item()}"
                        )
                    rank0_log(
                        "GDN_FUSED_REAL_TENSOR_PARITY "
                        f"layer_id={layer.layer_id} backend={fused_backend} "
                        + " ".join(report)
                    )
                    _fused_decode_real_tensor_verified_layers.add(layer.layer_id)
                    if mismatch:
                        raise AssertionError(
                            "GDN fused real-tensor parity failed at "
                            f"layer_id={layer.layer_id}; " + " ".join(report)
                        )
                conv_already_applied = True
            else:
                # Explicit correctness fallback for an unexpected runtime
                # tensor/state contract. This still returns Z to the model.
                if not _fused_decode_proj_conv_fallback_logged:
                    rank0_log(
                        "Falling back from fused GDN decode projection/Conv1D: "
                        f"{eligibility_reason}"
                    )
                    _fused_decode_proj_conv_fallback_logged = True
                mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                    projected_qkvz,
                    projected_ba,
                    layer.num_q_heads,
                    layer.num_v_heads,
                    layer.head_q_dim,
                    layer.head_v_dim,
                )
            return_z = True
        else:
            assert isinstance(mixed_qkv, torch.Tensor)

        if not conv_already_applied:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices,
            )

        # TwinStar factored GDN state (docs/62 §1.4): one fused launch per layer =
        # sink recurrence + basis append + slot-expiry truncation, reading the same
        # packed mixed_qkv / a / b and the same static cache_indices as the stock
        # packed kernel (CUDA-graph safe).  Stock path below is untouched when off.
        if self.factored is not None:
            core_attn_out = self._forward_decode_factored(
                layer, forward_batch, mixed_qkv, a, b, conv_states, ssm_states, cache_indices
            )
            return (core_attn_out, z) if return_z else core_attn_out

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
            )
            return (core_attn_out, z) if return_z else core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )

        return (core_attn_out, z) if return_z else core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        if _is_hip and seq_len == 0:
            return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if self.mis_metadata is not None:
            if is_target_verify:
                raise ValueError("GDN MIS does not support target verify")
            if self.factored is not None:
                raise ValueError("--linear-attn-factored-state does not support MIS (K1)")
            return self._forward_extend_mis(
                layer=layer,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
            )
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        # Page-major envelope: the prefill kernels (CUDA causal_conv1d_fwd,
        # chunk_gated_delta_rule) write state back in place assuming a contiguous
        # slot layout, so they silently drop the write to the strided envelope
        # pool. Run them on contiguous per-sequence copies (identity-indexed) and
        # scatter the result back. No-op for the default contiguous pool.
        # CPU kernels (causal_conv1d_fwd_cpu, chunk_gated_delta_rule_cpu) use
        # proper indexed writes and handle non-contiguous pools directly via
        # cache_indices, so the gather/scatter round-trip is unnecessary on CPU.
        # TODO(ch-wan): drop these .contiguous() copies by making the prefill conv
        # and chunk_gated_delta_rule kernels honor the pool's real slot stride +
        # int64 indexing, like packed_decode / causal_conv1d_update already do.
        needs_state_gather = (
            (not is_target_verify)
            and (not is_cpu())
            and (not conv_states.is_contiguous() or not ssm_states.is_contiguous())
        )
        if needs_state_gather:
            conv_states_contig = conv_states[cache_indices].contiguous()
            ssm_states_contig = ssm_states[cache_indices].contiguous()
            state_cache_indices = torch.arange(
                cache_indices.shape[0],
                device=cache_indices.device,
                dtype=cache_indices.dtype,
            )
        else:
            conv_states_contig = conv_states
            ssm_states_contig = ssm_states
            state_cache_indices = cache_indices

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states_contig,
                has_initial_state=has_initial_states,
                cache_indices=state_cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip() or is_xpu()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            # ReplaySSM verify protocols: fold-every-commit (ring-write during
            # verify, fold on commit), circular ring, or the snapshotting
            # fallback when neither ring is allocated.
            mamba_pool = self.req_to_token_pool.mamba_pool
            use_replayssm_fold = (
                mamba_cache_params.replayssm_rawv is not None
                and getattr(mamba_pool, "replayssm_spec_fold", False)
                and not getattr(mamba_pool, "replayssm_is_kda", False)
            )
            use_replayssm_spec = (
                mamba_cache_params.replayssm_d is not None
                and getattr(mamba_pool, "replayssm_cache_base", None) is not None
                and not getattr(mamba_pool, "replayssm_is_kda", False)
            )
            if use_replayssm_fold:
                core_attn_out = self._replayssm_fold_target_verify(
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    layer_cache=mamba_cache_params,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    retrieve_parent_token=retrieve_parent_token,
                )
            elif use_replayssm_spec:
                core_attn_out = self._replayssm_target_verify(
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    mamba_pool=mamba_pool,
                    layer_cache=mamba_cache_params,
                    cache_indices=cache_indices,
                    replay_indices=forward_batch.req_pool_indices,
                    query_start_loc=query_start_loc,
                    draft_token_num=forward_batch.spec_info.draft_token_num,
                )
            else:
                # The recurrent fallback needs the per-draft snapshots, which
                # the pool gates OFF under --enable-linear-replayssm-spec (the
                # same flag that makes `use_replayssm_spec` true above), so
                # this branch is unreachable with a None buffer by
                # construction -- keep it loud rather than silently frozen.
                assert intermediate_state_cache is not None, (
                    "recurrent target_verify fallback requires intermediate_ssm, "
                    "which is not allocated under --enable-linear-replayssm-spec"
                )
                core_attn_out = self.kernel_dispatcher.target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    intermediate_states_buffer=intermediate_state_cache,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=forward_batch.spec_info.draft_token_num,
                    retrieve_parent_token=retrieve_parent_token,
                )
        else:
            if self.factored is None and self._stepwise_active(forward_batch):
                # debug: stock recurrent decode kernel over the extend tokens (same maths as the served
                # dense decode step, token by token) instead of the chunk kernel
                from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
                    fused_sigmoid_gating_delta_rule_update,
                )

                self._stepwise_log("dense", forward_batch)
                core_attn_out = fused_sigmoid_gating_delta_rule_update(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    initial_state_source=ssm_states_contig,
                    initial_state_indices=state_cache_indices,
                    cu_seqlens=query_start_loc,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                )
                if needs_state_gather:
                    conv_states[cache_indices] = conv_states_contig
                    ssm_states[cache_indices] = ssm_states_contig
                self._maybe_dump_dense(layer, forward_batch, ssm_states, cache_indices)
                return core_attn_out
            if self.factored is not None:
                # TwinStar factored GDN state (docs/62 §1.3): dense only inside this
                # call (ring / densified initial state -> chunk kernel -> factorise).
                return self._forward_extend_factored(
                    layer=layer,
                    forward_batch=forward_batch,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    query_start_loc=query_start_loc,
                    forward_metadata=forward_metadata,
                    output=kwargs.get("linear_attn_output"),
                )
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states_contig,
                cache_indices=state_cache_indices,
                query_start_loc=query_start_loc,
                state_checkpoint_cu_starts=(
                    forward_metadata.state_checkpoint_cu_starts
                ),
                num_state_checkpoints=forward_metadata.num_state_checkpoints,
                state_checkpoint_every_n_tokens=(
                    forward_metadata.state_checkpoint_every_n_tokens
                ),
                output=kwargs.get("linear_attn_output"),
            )

            if is_npu() and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if needs_state_gather:
                # Scatter the in-place-updated contiguous copies back to the
                # strided envelope pool (advanced indexing handles the strides).
                conv_states[cache_indices] = conv_states_contig
                ssm_states[cache_indices] = ssm_states_contig

            if forward_metadata.has_mamba_track_mask:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )
            self._maybe_dump_dense(layer, forward_batch, ssm_states, cache_indices)

            if _DUET_CAPTURE_DIR is not None:
                self._capture_duet_blackboard(
                    layer, forward_batch, ssm_states, cache_indices,
                    query, key, value, g, beta,
                )

        return core_attn_out

    def _capture_duet_blackboard(self, layer, batch, states, indices, q, k, v, g, beta):
        """Diagnostic only: K1's 32 prefix states and first 64 continuation inputs.

        Never enabled in timing/guard runs. The chunk arm precedes the stepwise
        arm; cap each layer at 32 windows so the latter cannot overwrite it.
        """
        prefix = list(batch.extend_prefix_lens_cpu)
        lens = list(batch.extend_seq_lens_cpu)
        if len(prefix) != 1 or (prefix, lens) not in [([0], [3584]), ([3584], [512])]:
            return
        from sglang.srt.runtime_context import get_parallel

        lid = layer.layer_id
        counts = getattr(self, "_duet_capture_counts", {})
        self._duet_capture_counts = counts
        phase = "prefix" if prefix[0] == 0 else "continuation"
        window = counts.get((lid, phase), 0)
        if window >= 32:
            return
        counts[lid, phase] = window + 1
        rank = get_parallel().attn_tp_rank
        dest = _os.path.join(_DUET_CAPTURE_DIR, f"rank{rank}")
        _os.makedirs(dest, exist_ok=True)
        data = {"layer": lid, "window": window, "tp_rank": rank,
                "prefix": prefix, "lens": lens, "phase": phase}
        if phase == "prefix":
            data["S"] = states[indices.to(torch.long)].detach().cpu()
        else:
            for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
                # FLA inputs have [1, tokens, heads, ...] layout.
                data[name] = tensor[:, :64].detach().cpu()
        torch.save(data, _os.path.join(dest, f"w{window:02d}_L{lid:02d}_{phase}.pt"))

    # ------------------------------------------------------------------ TwinStar K1 debug hooks (docs/62 §2)
    def _stepwise_active(self, forward_batch: ForwardBatch) -> bool:
        if _STEPWISE_MIN_PREFIX <= 0:
            return False
        if _STEPWISE_FLAGFILE is not None and not _os.path.exists(_STEPWISE_FLAGFILE):
            return False
        pl = forward_batch.extend_prefix_lens_cpu
        return pl is not None and len(pl) > 0 and all(int(p) >= _STEPWISE_MIN_PREFIX for p in pl)

    def _stepwise_log(self, kind: str, forward_batch: ForwardBatch) -> None:
        key = (kind, tuple(int(x) for x in forward_batch.extend_prefix_lens_cpu[:1]))
        if key not in _stepwise_logged and len(_stepwise_logged) < 8:
            _stepwise_logged.add(key)
            rank0_log(
                f"GDN stepwise-extend debug hook ({kind}): prefix {list(forward_batch.extend_prefix_lens_cpu)[:4]} "
                f"extend {list(forward_batch.extend_seq_lens_cpu)[:4]}"
            )

    def _maybe_dump_dense(self, layer, forward_batch, ssm_states, cache_indices) -> None:
        if _FACTORED_DUMP_DIR is None or ssm_states.numel() == 0:
            return
        from sglang.srt.runtime_context import get_parallel

        rank = get_parallel().attn_tp_rank
        d = _os.path.join(_FACTORED_DUMP_DIR, f"rank{rank}")
        _os.makedirs(d, exist_ok=True)
        n = self._dump_n = getattr(self, "_dump_n", 0) + 1
        torch.save(
            {"kind": "dense", "layer": layer.layer_id, "slots": cache_indices.cpu(),
             "S": ssm_states[cache_indices.to(torch.long)].cpu(),
             "prefix": [int(x) for x in forward_batch.extend_prefix_lens_cpu],
             "lens": [int(x) for x in forward_batch.extend_seq_lens_cpu]},
            _os.path.join(d, f"dense_{n:05d}_L{layer.layer_id:02d}.pt"),
        )

    def _maybe_dump_factored(self, layer, forward_batch, plan) -> None:
        if _FACTORED_DUMP_DIR is None:
            return
        from sglang.srt.runtime_context import get_parallel

        rank = get_parallel().attn_tp_rank
        self.factored.dump_slots(
            layer.layer_id, plan.slots, {"prefix": [int(x) for x in forward_batch.extend_prefix_lens_cpu],
                                         "lens": [int(x) for x in forward_batch.extend_seq_lens_cpu]},
            _os.path.join(_FACTORED_DUMP_DIR, f"rank{rank}"), "factored",
        )

    def _forward_extend_factored_stepwise(self, *, layer, forward_batch, query, key, value, a, b, plan, output):
        """Debug: run the extend tokens through the factored step kernel on the pool slots, one token at a time
        (the served decode path's maths), instead of densify -> chunk kernel -> factorise."""
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import (
            factored_packed_decode,
        )

        pool = self.factored
        fa, fu, fw, fcount, vbar = pool.layer_tensors(layer.layer_id)
        lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
        starts = [0]
        for l_ in lens[:-1]:
            starts.append(starts[-1] + l_)
        T = query.shape[1]
        HV, V = layer.num_v_heads, layer.head_v_dim
        core = torch.empty(1, T, HV, V, dtype=value.dtype, device=value.device) if output is None else output
        slots_all = plan.slots.to(torch.int32)
        dev = value.device
        for t in range(max(lens)):
            rows = [i for i, l_ in enumerate(lens) if l_ > t]
            tok = torch.tensor([starts[i] + t for i in rows], device=dev, dtype=torch.long)
            rows_t = torch.tensor(rows, device=dev, dtype=torch.long)
            mixed = torch.cat([query[0, tok].reshape(len(rows), -1), key[0, tok].reshape(len(rows), -1),
                               value[0, tok].reshape(len(rows), -1)], dim=-1).contiguous()
            out_t = factored_packed_decode(
                mixed, a[tok].contiguous(), b[tok].contiguous(), A_log=layer.A_log, dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5, vbar=vbar, fa=fa, fu=fu, fw=fw, fcount=fcount, stale=pool.stale,
                ssm_state_indices=slots_all[rows_t], num_q_heads=layer.num_q_heads, num_v_heads=HV,
                head_k_dim=layer.head_k_dim, head_v_dim=V, r=pool.cfg.r, rfull=pool.cfg.rfull,
                **pool.cfg.kernel_kwargs(),
            )
            core[0, tok] = out_t[:, 0].to(core.dtype)
        pool.abandon_ring(plan)
        self._maybe_dump_factored(layer, forward_batch, plan)
        return core

    # ------------------------------------------------------------------ TwinStar factored GDN state (docs/62)
    def _forward_decode_factored(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import (
            factored_packed_decode,
            factored_expiry_truncate_layers,
        )

        pool = self.factored
        fa, fu, fw, fcount, vbar = pool.layer_tensors(layer.layer_id)
        if pool.layer_index(layer.layer_id) == 0:
            pool.invalidate_prefix_dense(cache_indices)
        out = factored_packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=layer.head_k_dim**-0.5,
            vbar=vbar,
            fa=fa,
            fu=fu,
            fw=fw,
            fcount=fcount,
            stale=pool.stale,
            ssm_state_indices=cache_indices,
            num_q_heads=layer.num_q_heads,
            num_v_heads=layer.num_v_heads,
            head_k_dim=layer.head_k_dim,
            head_v_dim=layer.head_v_dim,
            r=pool.cfg.r,
            rfull=pool.cfg.rfull,
            async_stream=self._factored_side_stream,
            truncate=not self._factored_batch_trunc,
            **pool.cfg.kernel_kwargs(),
        )
        if self._factored_batch_trunc and pool.is_last_layer(layer.layer_id):
            # All these layers are independent until the next token. Consolidate
            # their due heads without changing any request's r+m expiry count.
            # This launch is inside decode-graph capture and precedes track_copy.
            factored_expiry_truncate_layers(
                pool.U, pool.W, pool.count, cache_indices, pool.cfg.r, pool.cfg.rfull
            )
        if self._factored_side_stream is not None and pool.is_last_layer(layer.layer_id):
            # join the side stream: every layer's expiry truncation of this step is done before the track copy below,
            # before sampling, and before the next forward / COW copy / host offload touch the pool
            torch.cuda.current_stream().wait_stream(self._factored_side_stream)
        # radix tracking: conv windows through the stock kernels (ssm buffer is empty),
        # the factored state through one all-layers masked copy at the last GDN layer
        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )
        if forward_batch.mamba_track_mask is not None and pool.is_last_layer(layer.layer_id):
            pool.track_copy(
                cache_indices,
                forward_batch.mamba_track_mask,
                self.forward_metadata.mamba_track_indices,
            )
        return out.transpose(0, 1)  # [1, B, HV, V]

    def _forward_extend_factored(
        self,
        *,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        query_start_loc: torch.Tensor,
        forward_metadata,
        output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pool = self.factored
        plan = forward_metadata.factored_extend
        assert plan is not None, "factored extend plan missing (init_forward_metadata)"
        if self._stepwise_active(forward_batch):
            self._stepwise_log("factored", forward_batch)
            return self._forward_extend_factored_stepwise(
                layer=layer, forward_batch=forward_batch, query=query, key=key, value=value, a=a, b=b, plan=plan,
                output=output,
            )
        B = plan.slots.shape[0]
        # dense initial states for the chunk kernel: exact ring copies where the slot
        # still owns one, else densified from the factored form (zeros for fresh slots)
        S0 = pool.initial_dense(layer.layer_id, plan)  # (B, HV, V, K) fp32, contiguous
        row_indices = torch.arange(B, device=S0.device, dtype=torch.int32)
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        extend = self.kernel_dispatcher.extend
        if (_os.environ.get('SGLANG_GDN_PREFILL_DENSE_GRAPH', '0') == '1'
                and isinstance(self.kernel_dispatcher.extend_kernel, TritonGDNKernel)):
            from sglang.srt.mem_cache.gdn_prefill_dense_graph import DenseBuffers, PrefillDenseGraph
            graph = getattr(self, '_pdfix_dense_graph', None)
            if graph is None:
                graph = self._pdfix_dense_graph = PrefillDenseGraph()
            # Checkpoint kwargs are ignored by the original Triton extend.
            # Passing only its consumed operands avoids graph identities tied
            # to irrelevant per-request metadata tensor objects.
            extend = lambda **kw: graph.run(eager=self.kernel_dispatcher.extend_kernel.extend,
                **{name: value for name, value in kw.items()
                   if name in (*DenseBuffers.names, 'output')})
        core_attn_out, last_recurrent_state, h = extend(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ssm_states=S0,
            cache_indices=row_indices,
            query_start_loc=query_start_loc,
            state_checkpoint_cu_starts=forward_metadata.state_checkpoint_cu_starts,
            num_state_checkpoints=forward_metadata.num_state_checkpoints,
            state_checkpoint_every_n_tokens=forward_metadata.state_checkpoint_every_n_tokens,
            output=output,
        )
        if last_recurrent_state is not None and last_recurrent_state.data_ptr() != S0.data_ptr():
            S0 = last_recurrent_state.to(torch.float32)
        if pool.batch_prefill and _FACTORED_DUMP_DIR is None:
            hs = track_slots = final_src = final_dst = None
            if forward_metadata.has_mamba_track_mask:
                assert (
                    forward_metadata.track_ssm_recompute_dst is None
                    or forward_metadata.track_ssm_recompute_dst.numel() == 0
                ), "factored extend: checkpoint recompute tracking is not supported"
                if forward_metadata.track_ssm_h_src.numel():
                    assert h is not None
                    hs = h.squeeze(0)[forward_metadata.track_ssm_h_src]
                    track_slots = forward_metadata.track_ssm_h_dst
                final_src = forward_metadata.track_ssm_final_src
                final_dst = forward_metadata.track_ssm_final_dst
            pool.commit_extend_batched(layer.layer_id, plan, S0, hs, track_slots, final_src, final_dst)
            return core_attn_out
        # final dense -> factored (count = r, stale = 0) + exact copy into the ring
        pool.commit_extend(layer.layer_id, plan, S0)
        self._maybe_dump_factored(layer, forward_batch, plan)
        if forward_metadata.has_mamba_track_mask:
            if forward_metadata.track_ssm_h_src.numel() > 0:
                assert h is not None
                hs = h.squeeze(0)[forward_metadata.track_ssm_h_src]
                pool.write_factored_dense(layer.layer_id, forward_metadata.track_ssm_h_dst, hs)
            assert (
                forward_metadata.track_ssm_recompute_dst is None
                or forward_metadata.track_ssm_recompute_dst.numel() == 0
            ), "factored extend: FlashInfer checkpoint recompute tracking is not supported (Triton only)"
            if forward_metadata.track_ssm_final_src.numel() > 0:
                pool.copy_slots_layer(
                    layer.layer_id,
                    forward_metadata.track_ssm_final_src,
                    forward_metadata.track_ssm_final_dst,
                )
        return core_attn_out

    def _forward_extend_mis(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        metadata = self.mis_metadata
        assert metadata is not None

        conv_states[cache_indices] = 0
        ssm_states[cache_indices] = 0
        output = mixed_qkv.new_zeros(
            1, mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
        )

        if metadata.query_token_indices.numel() > 0:
            query_cache_indices = cache_indices[metadata.query_request_indices]
            query_output = self._forward_mis_segments(
                layer=layer,
                mixed_qkv=mixed_qkv[metadata.query_token_indices],
                a=a[metadata.query_token_indices],
                b=b[metadata.query_token_indices],
                conv_states=conv_states,
                conv_cache_indices=query_cache_indices,
                has_initial_states=torch.zeros(
                    len(metadata.query_seq_lens_cpu),
                    dtype=torch.bool,
                    device=mixed_qkv.device,
                ),
                query_start_loc=metadata.query_cu_seqlens,
                seq_lens_cpu=metadata.query_seq_lens_cpu,
                ssm_states=ssm_states,
                ssm_cache_indices=query_cache_indices,
                inplace_update=True,
            )
            output[:, metadata.query_token_indices] = query_output

        item_ssm_indices = cache_indices[metadata.item_request_indices]
        item_conv_states = conv_states[item_ssm_indices]
        item_conv_indices = torch.arange(
            item_ssm_indices.shape[0],
            dtype=cache_indices.dtype,
            device=cache_indices.device,
        )
        item_output = self._forward_mis_segments(
            layer=layer,
            mixed_qkv=mixed_qkv[metadata.item_token_indices],
            a=a[metadata.item_token_indices],
            b=b[metadata.item_token_indices],
            conv_states=item_conv_states,
            conv_cache_indices=item_conv_indices,
            has_initial_states=torch.ones(
                len(metadata.item_seq_lens_cpu),
                dtype=torch.bool,
                device=mixed_qkv.device,
            ),
            query_start_loc=metadata.item_cu_seqlens,
            seq_lens_cpu=metadata.item_seq_lens_cpu,
            ssm_states=ssm_states,
            ssm_cache_indices=item_ssm_indices,
            inplace_update=False,
        )
        output[:, metadata.item_token_indices] = item_output
        return output

    def _forward_mis_segments(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        conv_cache_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_cpu: list[int],
        ssm_states: torch.Tensor,
        ssm_cache_indices: torch.Tensor,
        inplace_update: bool,
    ) -> torch.Tensor:
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=has_initial_states,
            cache_indices=conv_cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
        ).transpose(0, 1)

        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
            )
            num_tokens = mixed_qkv.shape[0]
            query = query.view(1, num_tokens, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, num_tokens, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, num_tokens, layer.num_v_heads, layer.head_v_dim)

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        output, _, _ = self.kernel_dispatcher.extend(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=ssm_cache_indices,
            query_start_loc=query_start_loc,
            inplace_update=inplace_update,
        )
        return output

    def _replayssm_fold_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        layer_cache: "MambaPool.SpeculativeState",
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        retrieve_parent_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ring-writing verify; the commit fold replays the accepted prefix
        into ``temporal``. Uses the vendored CuTe DSL MTP kernel when the
        dispatcher selected the FlashInfer bf16-state verify, else the Triton
        recurrent kernel (both store the same raw window)."""
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )

        assert retrieve_parent_token is None, (
            "ReplaySSM fold-every-commit supports a linear draft chain only "
            "(topk <= 1); EAGLE tree verify must use the recurrent verify."
        )
        seq_len = query.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size
        if (
            self.kernel_dispatcher.verify_kernel_is_flashinfer
            and ssm_states.dtype == torch.bfloat16
            and draft_token_num >= 3
        ):
            from sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring import (
                gated_delta_rule_mtp,
            )

            num_v_heads = value.shape[2]
            head_v_dim = value.shape[3]
            out = gated_delta_rule_mtp(
                A_log=layer.A_log.detach(),
                a=a.view(batch_size, draft_token_num, num_v_heads),
                dt_bias=layer.dt_bias.detach(),
                q=query.view(batch_size, draft_token_num, *query.shape[2:]),
                k=key.view(batch_size, draft_token_num, *key.shape[2:]),
                v=value.view(batch_size, draft_token_num, num_v_heads, head_v_dim),
                b=b.view(batch_size, draft_token_num, num_v_heads),
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                cache_ring=True,
                replayssm_rawv=layer_cache.replayssm_rawv,
                replayssm_rawk=layer_cache.replayssm_rawk,
                replayssm_g=layer_cache.replayssm_g,
                replayssm_beta=layer_cache.replayssm_beta,
            )
            return out.view(1, seq_len, num_v_heads, head_v_dim)
        return fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            disable_state_update=True,
            cache_ring=True,
            replayssm_rawv=layer_cache.replayssm_rawv,
            replayssm_rawk=layer_cache.replayssm_rawk,
            replayssm_g=layer_cache.replayssm_g,
            replayssm_beta=layer_cache.replayssm_beta,
        )

    def _replayssm_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        mamba_pool: MambaPool,
        layer_cache: "MambaPool.SpeculativeState",
        cache_indices: torch.Tensor,
        replay_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        """ReplaySSM GDN spec-verify (Part B of #28511).

        Reconstructs the verify output for the whole draft window from the frozen
        checkpoint (``temporal``) + the per-slot circular ``(d, k, g)`` ring, and
        appends this window's drafts to the compact ``(d, k, g)`` rings. BF16
        checkpoints also keep low parts for compensated materialization. The rings
        are per-layer (sliced via ``mamba2_layer_cache``), while the cursors
        (write_pos, cache_base, is_flush) are request-slot pool attributes shared
        by all GDN layers of the step. The cursors advance once per accepted step;
        here we only read them and write this step's ring entries. GDN has K == V,
        so ``temporal``
        ([slots, HV, K, V]) is consumed directly as the kernel's [slots, HV, V, K]
        checkpoint.
        """
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

        H, K = layer.num_k_heads, layer.head_k_dim
        HV, V = layer.num_v_heads, layer.head_v_dim
        # q/k/v may be [1, seq, *] (fallback split) or [seq, *] (fused split);
        # derive the packed token count from numel so both layouts flatten.
        seq_len = query.numel() // (H * K)
        q = query.reshape(seq_len, H, K)
        k = key.reshape(seq_len, H, K)
        v = value.reshape(seq_len, HV, V)
        a = a.reshape(seq_len, HV)
        b = b.reshape(seq_len, HV)
        d_cache = layer_cache.replayssm_d  # [slots, HV, L, V]
        max_cache_len = d_cache.shape[-2]  # ring length L
        out = q.new_empty(seq_len, HV, V)
        gdn_replayssm_spec_decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            checkpoint_state=layer_cache.temporal,
            d_cache=d_cache,
            k_cache=layer_cache.replayssm_k,
            g_cache=layer_cache.replayssm_g,
            # BF16 compact D/K low parts; None for fp32 checkpoints.
            rawv_cache=layer_cache.replayssm_rawv,
            rawk_cache=layer_cache.replayssm_rawk,
            beta_cache=layer_cache.replayssm_beta,
            out=out,
            query_start_loc=query_start_loc,
            ssm_state_indices=cache_indices,
            replay_indices=replay_indices,
            # Request-slot cursors live on the pool (shared across all GDN layers)
            # and advance only after acceptance, so the decode-path
            # forward_metadata.replayssm_write_pos snapshot is not used here.
            write_pos=mamba_pool.replayssm_spec_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            max_cache_len=max_cache_len,
            max_spec_len=draft_token_num,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            # SGLang marks invalid/padding requests with a negative mamba slot
            # index (valid slots start at 0), so the kernel's "null block"
            # sentinel is -1, not the vLLM default of 0.
            null_block_id=-1,
            # Capacity folds are committed once across every GDN layer after
            # acceptance; the active path is therefore a single launch/layer.
            launch_mode="verify",
        )
        # Match the recurrent target_verify output shape (== value.shape).
        return out.reshape(value.shape)
