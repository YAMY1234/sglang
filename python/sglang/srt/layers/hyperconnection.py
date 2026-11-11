import logging
from functools import lru_cache
from typing import Optional

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.gemm.hc_mix import fused_hc_mix, fused_hc_mix_supported
from sglang.srt.environ import envs
from sglang.srt.layers.hc_mix_flydsl import (
    flydsl_hc_mix_supported,
    flydsl_hc_mix_weights_supported,
    weight_cache_key,
)

logger = logging.getLogger(__name__)

# Extend boundary (SGLANG_OPT_HC_EXTEND_BOUNDARY) build outcome per (hc, hidden, dtype),
# decided once per process, so every layer agrees on whether its [rows, hc] gate
# totals are in use; and the row bands that have logged their first use.
_HC_EXT_BUILT: dict = {}
_HC_EXT_LOGGED: set = set()
# Below this the decode paths (side-stream gate, HC macro) own the boundary.
_HC_EXT_FLOOR_ROWS = 33


@lru_cache(None)
def _hc_gate_stream(device: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device)


def _hc_mix_wide_max_rows() -> int:
    """Largest row count hc_mix_wide serves (SGLANG_HC_MIX_SPLITK_MAX_ROWS); 0 = off."""
    if not envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get():
        return 0
    from sglang.kernels.ops.gemm.hc_mix_splitk import WIDE_MAX_ROWS

    return min(envs.SGLANG_HC_MIX_SPLITK_MAX_ROWS.get(), WIDE_MAX_ROWS)


def _hc_ext_min_rows() -> int:
    """Smallest write-back the extend boundary serves; 0 = off."""
    if not envs.SGLANG_OPT_HC_EXTEND_BOUNDARY.get():
        return 0
    return max(_HC_EXT_FLOOR_ROWS, envs.SGLANG_X_HC_EXTEND_MIN_ROWS.get())


def _hc_ext_ready(hc_count: int, hidden_size: int, dtype: torch.dtype) -> bool:
    key = (hc_count, hidden_size, dtype)
    ready = _HC_EXT_BUILT.get(key)
    if ready is None:
        try:
            from sglang.kernels.ops.elementwise.hc_combine_extend import warmup

            warmup(hc_count, hidden_size, dtype)
            ready = True
        except Exception as e:  # keep the stock boundary
            logger.warning("hc_combine_extend unavailable (%s); stock HC", e)
            ready = False
        _HC_EXT_BUILT[key] = ready
    return ready


def _log_hc_ext_once(
    rows: int, min_rows: int, has_partials: bool, norm: bool, gate: bool
):
    band = "prefill" if rows >= 1024 else "verify"
    if band in _HC_EXT_LOGGED:
        return
    _HC_EXT_LOGGED.add(band)
    logger.info(
        "HC extend boundary engaged (%s rows): rows=%d min_rows=%d partials=%s "
        "norm=%s gate=%s (SGLANG_OPT_HC_EXTEND_BOUNDARY)",
        band,
        rows,
        min_rows,
        has_partials,
        norm,
        gate,
    )


class HyperConnectionConfig(msgspec.Struct, frozen=True):
    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, group_size: Optional[int] = None
    ):
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight.weight_loader = self._weight_loader
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(
                x, self.weight, self._jit_group_size, self.variance_epsilon
            )
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            x_grouped = x_float.reshape(
                *x_float.shape[:-1],
                x_float.shape[-1] // self.group_size,
                self.group_size,
            )
            variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
            x_norm = (
                x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (x_norm * (1.0 + self.weight.float())).to(input_dtype)


class HyperConnectionBase(nn.Module):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        self.hc_count = config.hc_count
        if config.mtp_hc and role is not None and "mtp" in role:
            self.hc_count = self.hc_count + 1
        self.hidden_size = config.hidden_size
        self.params_dtype = config.params_dtype

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        mixed_input = hyper_input.view(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        ).mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.view(
            *residual.shape[:-1], self.hc_count, self.hidden_size
        )
        combined_output = residual_reshaped + block_output.unsqueeze(-2)
        combined_output = combined_output.view(
            *residual.shape[:-1], self.hc_count * self.hidden_size
        )
        return combined_output


class GatedResidual(HyperConnectionBase):
    def numeric_signatures(self, widths, tokens_per_request):
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            HcSignature,
            NumericOperation,
            ScalarType,
        )

        if self.params_dtype != torch.bfloat16:
            return ()
        operations = (
            NumericOperation.HC_GATE,
            NumericOperation.HC_APPLY_NORM,
            NumericOperation.HC_APPLY_NORM_GATE,
        )
        signatures = [
            HcSignature(op, rows, self.hc_count, self.hidden_size, ScalarType.BF16)
            for rows in widths
            if type(rows) is int and rows > 0
            for op in operations
        ]
        # The tuning role is resolved here; a provider never queries the runtime.
        mix_variant = self._numeric_mix_variant()
        signatures.extend(
            HcSignature(
                NumericOperation.HC_MIX,
                rows,
                self.hc_count,
                self.hidden_size,
                ScalarType.BF16,
                lowrank=self.config.hc_lowrank,
                variant=mix_variant,
            )
            for rows in widths
            if type(rows) is int and rows > 0
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import (
            signature_capability_supported,
        )

        declared = tuple(
            signature
            for signature in signatures
            if signature_capability_supported(
                signature,
                spec=getattr(self, "_numeric_capability_spec", None),
            )
        )
        if (
            self._gate_stream is None
            and hasattr(self, "block_inject_weight")
            and self.config.hc_per_branch_norm
            and any(s.operation == NumericOperation.HC_GATE for s in declared)
            and torch.cuda.is_available()
        ):
            # Existing startup lifecycle, after capability admission and before
            # graph warmup; no stream is created for an unsupported shape.
            self._gate_stream = _hc_gate_stream(torch.cuda.current_device())
        return declared

    def _numeric_hc_partials_supported(self, rows):
        if rows == 1:
            return True
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            HcSignature,
            NumericOperation,
            ScalarType,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

        return (
            numeric_operator(
                HcSignature(
                    NumericOperation.HC_APPLY_NORM,
                    rows,
                    self.hc_count,
                    self.hidden_size,
                    ScalarType.BF16,
                ),
                prepared=True,
            )
            is not None
        )

    def _numeric_hc_macro(self, rows):
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            HcSignature,
            NumericOperation,
            ScalarType,
        )
        from sglang.srt.environ import envs
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

        return numeric_operator(
            HcSignature(
                NumericOperation.HC_APPLY_NORM_GATE,
                rows,
                self.hc_count,
                self.hidden_size,
                ScalarType.BF16,
            ),
            prepared=(
                envs.SGLANG_HC_MACRO.get()
                and max(1, envs.SGLANG_HC_MACRO_MIN_ROWS.get()) <= rows
                and self.config.hc_per_branch_norm
                and hasattr(self, "block_inject_weight")
                and self._numeric_mix_variant() == 1
                and not torch.compiler.is_compiling()
                and self._numeric_hc_partials_supported(rows)
            ),
        )

    def _numeric_hc_mix_supported(self, rows):
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            HcSignature,
            NumericOperation,
            ScalarType,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

        return (
            numeric_operator(
                HcSignature(
                    NumericOperation.HC_MIX,
                    rows,
                    self.hc_count,
                    self.hidden_size,
                    ScalarType.BF16,
                    lowrank=self.config.hc_lowrank,
                    variant=self._numeric_mix_variant(),
                ),
                prepared=True,
            )
            is not None
        )

    @staticmethod
    def _numeric_mix_variant():
        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_size == 1 and envs.SGLANG_HC_MACRO.get())

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__(config, use_mix, use_combine, role)

        norm_dim = (
            self.config.hidden_size * self.hc_count
            if self.config.hc_per_branch_norm
            else self.config.hidden_size
        )
        norm_group_size = (
            self.config.hidden_size if self.config.hc_per_branch_norm else None
        )
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_dim, eps=self.config.rms_norm_eps, group_size=norm_group_size
        )

        if use_mix:
            self.input_mix_weight_down = nn.Linear(
                self.hidden_size * self.hc_count,
                self.config.hc_lowrank,
                bias=False,
                dtype=config.params_dtype,
            )
            self.input_mix_weight_up = nn.Linear(
                self.config.hc_lowrank,
                self.hc_count * self.hidden_size,
                bias=False,
                dtype=config.params_dtype,
            )
            lowrank = self.config.hc_lowrank
            self._jit_mix_ok = (
                torch.cuda.is_available()
                # The CuTe split-K pair is tcgen05 (sm_100 family) only.
                and torch.cuda.get_device_capability()[0] == 10
                and (self.hc_count * self.hidden_size) % 2048 == 0
                and self.hidden_size % 8 == 0
                and lowrank > 0
                and lowrank % 8 == 0
            )
            self._mix_up_weight_padded = None
            # hc_mix_wide's up GEMM tiles: 32 hidden columns, 64 low-rank columns.
            self._mix_wide_max_rows = (
                _hc_mix_wide_max_rows()
                if self._jit_mix_ok
                and self.hc_count == 4
                and self.hidden_size % 32 == 0
                and lowrank % 64 == 0
                else 0
            )
            # Resolved on the first mix(), once the weights are final.
            self._mix_flydsl_ok = None
            self._mix_flydsl_weights = None
            self._mix_flydsl_weight_key = None

        self._gate_stream = None
        self._ext_min_rows = 0
        self._gate_max_rows = envs.SGLANG_HC_GATE_MAX_ROWS.get()
        if self._gate_max_rows not in (1, 4, 16, 32):
            raise ValueError("HC gate max rows must be 1, 4, 16 or 32")
        if use_combine:
            self.block_inject_weight = nn.Linear(
                self.hidden_size * self.hc_count,
                self.hc_count,
                bias=False,
                dtype=config.params_dtype,
            )
            # hc_combine rejects other shapes; device and dtype are checked per call.
            self._jit_combine_ok = (
                self.hidden_size % 8 == 0
                and (self.hc_count * self.hidden_size) % 2048 == 0
            )
            vecs = self.hc_count * self.hidden_size // 8
            self._split_combine_ok = (
                self._jit_combine_ok
                and vecs % (8 * 160) == 0
                and (self.hidden_size // 8) % (vecs // 8) == 0
            )
            # The extend boundary replays grouped_gemma_rmsnorm's Blackwell (32-byte
            # vector) reduction order; its source refuses to build for other targets.
            # Built here, before graph capture; a failed build keeps the stock boundary.
            self._ext_min_rows = _hc_ext_min_rows()
            if not (
                self._ext_min_rows
                and config.hc_per_branch_norm
                and self.hc_norm._jit_group_size == self.hidden_size
                and self.hidden_size % 512 == 0
                and (self.hc_count * self.hidden_size) % 2048 == 0
                and self.hc_count <= 8
                and config.params_dtype in (torch.bfloat16, torch.float16)
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 10
                and _hc_ext_ready(self.hc_count, self.hidden_size, config.params_dtype)
            ):
                self._ext_min_rows = 0

        def _mix_compute(
            hyper_input_normed: torch.Tensor,
            input_mix_weight_down: torch.Tensor,
            input_mix_weight_up: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            input_mix_weight = F.silu(
                F.linear(hyper_input_normed, input_mix_weight_down) / hc
            )
            input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
            input_mix_weight = torch.sigmoid(input_mix_weight)
            input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
            output = (
                input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
            ).mean(dim=-2)
            return output

        def _combine_compute(
            block_output: torch.Tensor,
            residual: torch.Tensor,
            normed_residual: torch.Tensor,
            block_inject_weight: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            R = residual.unflatten(-1, (hc, hs))
            block_inject_weight_out = 2 * torch.sigmoid(
                F.linear(normed_residual, block_inject_weight) / hc
            )
            injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(
                -1
            )
            return (R + injection).flatten(-2)

        self._mix_compute = torch.compile(_mix_compute)
        self._combine_compute = torch.compile(_combine_compute)

    def _flydsl_mix_ok(self) -> bool:
        if self._mix_flydsl_ok is None:
            self._mix_flydsl_ok = flydsl_hc_mix_weights_supported(
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
            )
        return self._mix_flydsl_ok

    def mix(
        self,
        hyper_input: torch.Tensor,
        normalized_input: Optional[torch.Tensor] = None,
        partials: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype
            )
            return mixed_input, (hyper_input, hyper_input, None)

        if partials is not None:
            assert normalized_input is not None
        if normalized_input is not None:
            assert self.config.hc_per_branch_norm
            assert normalized_input.shape == hyper_input.shape
            hyper_input_normed = normalized_input
        elif self.config.hc_per_branch_norm:
            hyper_input_normed = self.hc_norm(hyper_input)
        else:
            hyper_input_normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        if (
            partials is None
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype == torch.bfloat16
            and self._numeric_hc_macro(hyper_input.shape[0]) is not None
        ):
            from sglang.kernels.ops.elementwise.hc_combine_decode import hc_combine_gate

            partials = hc_combine_gate(
                hyper_input_normed, self.block_inject_weight.weight
            )
        if (
            self._jit_mix_ok
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and (
                hyper_input_normed.shape[0] <= 24
                or (
                    self._numeric_mix_variant()
                    and self._numeric_hc_mix_supported(hyper_input_normed.shape[0])
                )
            )
        ):
            from sglang.kernels.ops.gemm.hc_mix_splitk import (
                hc_mix,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
                numeric_variant=self._numeric_mix_variant(),
            ).to(self.params_dtype)
        elif (
            32 < hyper_input_normed.shape[0] <= self._mix_wide_max_rows
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.is_contiguous()
            and self._numeric_mix_variant()
            and not torch.compiler.is_compiling()
        ):
            from sglang.kernels.ops.gemm.hc_mix_splitk import (
                hc_mix_wide,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix_wide(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif flydsl_hc_mix_supported(hyper_input_normed, self._flydsl_mix_ok()):
            from aiter.ops.flydsl.hc_mix import hc_mix, pack_hc_weights

            down = self.input_mix_weight_down.weight
            up = self.input_mix_weight_up.weight
            key = weight_cache_key(down, up)
            if self._mix_flydsl_weight_key != key:
                self._mix_flydsl_weights = pack_hc_weights(down, up, self.hc_count)
                self._mix_flydsl_weight_key = key
            mixed_input = hc_mix(hyper_input_normed, *self._mix_flydsl_weights).to(
                self.params_dtype
            )
        elif fused_hc_mix_supported(
            hyper_input_normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        ):
            mixed_input = fused_hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        else:
            mixed_input = self._mix_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        stream = self._gate_stream
        if (
            stream is not None
            and partials is None
            and 0 < hyper_input.shape[0] <= self._gate_max_rows
            and self._numeric_hc_partials_supported(hyper_input.shape[0])
            and hyper_input.dtype == torch.bfloat16
            and not torch.compiler.is_compiling()
        ):
            from sglang.kernels.ops.elementwise.hc_combine_decode import hc_combine_gate

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                partials = hc_combine_gate(
                    hyper_input_normed, self.block_inject_weight.weight
                )
            hyper_input_normed.record_stream(stream)
        return mixed_input, (hyper_input, hyper_input_normed, partials)

    def _ext_rows(
        self,
        block_output: torch.Tensor,
        hyper_input: torch.Tensor,
        hyper_input_normed: torch.Tensor,
        partials: Optional[torch.Tensor],
    ) -> bool:
        """Whether the extend boundary serves this write-back. Its partials are
        [rows, hc] fp32 gate dot totals, made only by itself."""
        rows = block_output.shape[0]
        if not (self._ext_min_rows and rows >= self._ext_min_rows):
            return False
        dtype = block_output.dtype
        if partials is None:
            partials_ok = (
                hyper_input_normed.dtype == dtype
                and hyper_input_normed.shape == hyper_input.shape
            )
        else:
            partials_ok = partials.dtype == torch.float32 and tuple(partials.shape) == (
                rows,
                self.hc_count,
            )
        return (
            partials_ok
            and block_output.dim() == 2
            and hyper_input.dim() == 2
            and hyper_input.dtype == dtype
            and self.block_inject_weight.weight.dtype == dtype
            and block_output.is_cuda
            and not torch.compiler.is_compiling()
        )

    def _combine_extend(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        next_norm: Optional[GroupedGemmaRMSNorm],
        next_inject_weight: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        from sglang.kernels.ops.elementwise.hc_combine_extend import (
            hc_combine_extend,
        )

        hyper_input, hyper_input_normed, partials = residuals
        dtype = block_output.dtype
        width = self.hc_count * self.hidden_size
        if next_norm is not None and not (
            next_norm.group_size == self.hidden_size
            and next_norm._jit_group_size == self.hidden_size
            and next_norm.weight.dtype == dtype
            and next_norm.weight.numel() == width
        ):
            next_norm = None
        if next_norm is None or (
            next_inject_weight is not None
            and (
                next_inject_weight.dtype != dtype
                or tuple(next_inject_weight.shape) != (self.hc_count, width)
            )
        ):
            next_inject_weight = None
        _log_hc_ext_once(
            block_output.shape[0],
            self._ext_min_rows,
            partials is not None,
            next_norm is not None,
            next_inject_weight is not None,
        )
        return hc_combine_extend(
            block_output,
            hyper_input,
            self.hc_count,
            self.hidden_size,
            partials=partials,
            normed_residual=hyper_input_normed if partials is None else None,
            inject_weight=(
                self.block_inject_weight.weight if partials is None else None
            ),
            norm_weight=next_norm.weight if next_norm is not None else None,
            eps=next_norm.variance_epsilon if next_norm is not None else 0.0,
            next_inject_weight=next_inject_weight,
        )

    def combine_normalize_and_gate(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        next_norm: Optional[GroupedGemmaRMSNorm],
        next_inject_weight: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Publish a local HC boundary; the caller carries the next read's gate.

        The two-result stock API remains unchanged. No prepared values cross a
        history update, row redistribution, or pipeline boundary.
        """
        residual, normed, partials = residuals
        if self._ext_rows(block_output, residual, normed, partials):
            return self._combine_extend(
                block_output, residuals, next_norm, next_inject_weight
            )
        if partials is not None and partials.dim() == 2:
            # Extend-boundary gate totals, which the decode kernels ([rows, 8, hc]
            # partials) cannot read; the gate recomputed from the normalized streams
            # is the same.
            partials = None
            residuals = (residual, normed, None)
        rows = block_output.shape[0]
        if (
            partials is not None
            and next_norm is not None
            and next_inject_weight is not None
            and next_norm.group_size == self.hidden_size
            and block_output.is_cuda
            and block_output.dtype
            == residual.dtype
            == next_norm.weight.dtype
            == next_inject_weight.dtype
            == torch.bfloat16
            and block_output.shape == (rows, self.hidden_size)
            and residual.shape == (rows, self.hc_count * self.hidden_size)
            and partials.shape == (rows, 8, self.hc_count)
            and partials.dtype == torch.float32
            and next_norm.weight.numel() == self.hc_count * self.hidden_size
            and next_inject_weight.shape
            == (self.hc_count, self.hc_count * self.hidden_size)
            and all(
                t.device == block_output.device
                for t in (residual, partials, next_norm.weight, next_inject_weight)
            )
            and all(
                t.is_contiguous()
                for t in (
                    block_output,
                    residual,
                    partials,
                    next_norm.weight,
                    next_inject_weight,
                )
            )
        ):
            run = self._numeric_hc_macro(rows)
            if run is not None:
                from sglang.kernels.plugin_kernels.nvidia.numeric import (
                    HcApplyNormGateIO,
                )

                combined = torch.empty_like(residual)
                normalized = torch.empty_like(residual)
                next_partials = torch.empty_like(partials)
                run(
                    HcApplyNormGateIO(
                        block_output,
                        residual,
                        partials,
                        next_norm.weight,
                        next_inject_weight,
                        next_norm.variance_epsilon,
                        2,
                        combined,
                        normalized,
                        next_partials,
                    )
                )
                return combined, normalized, next_partials
        combined, normalized = self.combine_and_normalize(
            block_output,
            residuals,
            next_norm,
        )
        return combined, normalized, None

    def combine_and_normalize(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        next_norm: Optional[GroupedGemmaRMSNorm],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return the residual and its next HC normalization explicitly.

        The BF16 combine result is retained before calculating the next norm.
        Callers must only pass the norm when no PLE or other update intervenes.
        """
        hyper_input, hyper_input_normed, partials = residuals
        if (
            partials is not None
            and next_norm is not None
            and next_norm.group_size == self.hidden_size
            and next_norm.weight.dtype == torch.bfloat16
            and block_output.dtype == torch.bfloat16
        ):
            from sglang.kernels.ops.elementwise.hc_combine_decode import (
                _numeric_hc_apply_norm,
                hc_combine_apply_norm,
            )

            if (
                block_output.shape[0] > 4
                and _numeric_hc_apply_norm(
                    block_output,
                    hyper_input,
                    partials,
                    next_norm.weight,
                )
                is None
            ):
                return self.combine(
                    block_output,
                    (hyper_input, hyper_input_normed, None),
                ), None
            if (
                self._gate_stream is not None
                and self._numeric_hc_macro(block_output.shape[0]) is None
            ):
                current = torch.cuda.current_stream()
                current.wait_stream(self._gate_stream)
                partials.record_stream(current)
            return hc_combine_apply_norm(
                block_output,
                hyper_input,
                partials,
                next_norm.weight,
                next_norm.variance_epsilon,
            )
        return self.combine(block_output, residuals), None

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        hyper_input, hyper_input_normed, partials = residuals
        if self._ext_rows(block_output, hyper_input, hyper_input_normed, partials):
            return self._combine_extend(block_output, residuals, None, None)[0]
        if partials is not None and partials.dim() == 2:
            # Extend-boundary gate totals: recompute from the normalized streams.
            partials = None
        if partials is not None:
            from sglang.kernels.ops.elementwise.hc_combine_decode import (
                hc_combine_apply,
            )

            if (
                self._gate_stream is not None
                and self._numeric_hc_macro(block_output.shape[0]) is None
            ):
                current = torch.cuda.current_stream()
                current.wait_stream(self._gate_stream)
                partials.record_stream(current)
            return hc_combine_apply(block_output, hyper_input, partials)
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        if block_output.shape[0] == 0:
            return hyper_input.to(self.params_dtype)

        if (
            self._jit_combine_ok
            and block_output.is_cuda
            and block_output.dtype in (torch.bfloat16, torch.float16)
            and hyper_input.dtype == block_output.dtype
            and hyper_input_normed.dtype == block_output.dtype
            and self.block_inject_weight.weight.dtype == block_output.dtype
        ):
            if self._split_combine_ok and block_output.shape[0] <= 32:
                from sglang.kernels.ops.elementwise.hc_combine import (
                    hc_combine_split,
                )

                return hc_combine_split(
                    block_output,
                    hyper_input,
                    hyper_input_normed,
                    self.block_inject_weight.weight.data,
                    self.hc_count,
                    self.hidden_size,
                )
            from sglang.kernels.ops.elementwise.hc_combine import hc_combine

            return hc_combine(
                block_output,
                hyper_input,
                hyper_input_normed,
                self.block_inject_weight.weight,
                self.hc_count,
                self.hidden_size,
            )

        updated_residuals = self._combine_compute(
            block_output,
            hyper_input,
            hyper_input_normed,
            self.block_inject_weight.weight,
            self.hc_count,
            self.hidden_size,
        ).to(self.params_dtype)
        return updated_residuals


HYPERCONNECTION_CLASS_DICT = {
    "hyperconnection_average": HyperConnectionBase,
    "gated_residual_simple": GatedResidual,
}
