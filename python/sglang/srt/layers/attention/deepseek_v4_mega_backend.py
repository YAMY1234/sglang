from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.dp_attention import get_eager_max_batch_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_disagg, get_exec, max_prefill_buffer_tokens

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

HEAD_DIM = 512


def interleave_query_rows(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    return x.view(num_heads, 32, 16, -1).transpose(0, 1).reshape(x.shape)


def pack_query(
    q: torch.Tensor, padded_heads: int, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    tokens, heads, _ = q.shape
    if out is None:
        out = q.new_zeros(tokens, padded_heads, HEAD_DIM)
    out.view(tokens, 32, padded_heads, 16)[:, :, :heads] = q.view(
        tokens, heads, 32, 16
    ).transpose(1, 2)
    return out


def max_packed_query_rows(*, max_running_requests: int, rows_per_request: int) -> int:
    graphs = get_exec().graph.cuda_graph_config
    max_bs = max(max_running_requests, graphs.decode.max_bs or 0)
    rows = get_eager_max_batch_size(max_bs) * rows_per_request
    if get_disagg().disaggregation_mode == "decode":
        return rows
    return max(
        rows,
        max_prefill_buffer_tokens(),
        graphs.prefill.max_bs or 0,
        envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.get(),
    )


def static_fp8_q_head_scale(
    weight: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
    q_norm_weight: torch.Tensor,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-head power-of-two scales for a static E4M3 Q, bounded from the weights as
    vLLM's ``_init_fp8_q``: wq_b reads RMSNorm(x) * w with ||RMSNorm(x)|| = sqrt(K), so
    |Q_n| <= sqrt(K) * ||w * W_n||; twice that, rounded up to a power of two, also bounds
    the rotated RoPE pairs. ``weight`` [N, K] (MXFP8, scales [N, K / 32]) has its rows in
    the MegaAttention query layout: row r feeds head r % (num_heads * 16) // 16.
    Returns (scale [num_heads] fp32, log2 of each row's head scale [N] int32)."""
    n, k = weight.shape
    qn = q_norm_weight.float()
    row_bound = torch.empty(n, dtype=torch.float32, device=weight.device)
    step = 4096
    for r0 in range(0, n, step):
        w = weight[r0 : r0 + step].float() * torch.exp2(
            weight_scale_ue8m0[r0 : r0 + step].view(torch.uint8).float() - 127
        ).repeat_interleave(32, dim=1)
        row_bound[r0 : r0 + step] = (w * qn).norm(dim=1)
    row_bound *= k**0.5
    head = torch.arange(n, device=weight.device) % (num_heads * 16) // 16
    bound = row_bound.new_zeros(num_heads).scatter_reduce(0, head, row_bound, "amax")
    scale = torch.exp2(torch.ceil(torch.log2(2 * bound / 448.0)))
    return scale.contiguous(), torch.log2(scale).round().int()[head]


def interleave_output_columns(w: torch.Tensor) -> torch.Tensor:
    return w.view(*w.shape[:-1], 8, 16, 32).transpose(-3, -2).reshape(w.shape)


class MegaPrefillSplit(NamedTuple):
    """MegaAttention prefill output with per-group 128x4 scales for the 128-aligned head
    rows and the packed scales for a tail of < 128 rows (None when there is none)."""

    head_data: torch.Tensor
    head_sf: torch.Tensor
    tail_data: Optional[torch.Tensor]
    tail_sf: Optional[torch.Tensor]


def quantize_wo_a(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rank, _ = weight.shape
    return sglang_per_token_group_quant_fp8(
        weight.contiguous(),
        group_size=32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        masked_m=torch.full((groups,), rank, dtype=torch.int32, device=weight.device),
    )


class DeepseekV4MegaAttnBackend(DeepseekV4AttnBackend):
    mega_attn: bool = True

    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__(model_runner, **kwargs)
        self.max_packed_query_rows = max_packed_query_rows(
            max_running_requests=model_runner.max_running_requests,
            rows_per_request=self.speculative_num_draft_tokens or 1,
        )
        self._packed_query: Optional[torch.Tensor] = None

    def packed_query_buffer(
        self, rows: int, padded_heads: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        # Never reallocated: captured graphs keep its address.
        if self._packed_query is None:
            assert not torch.cuda.is_current_stream_capturing()
            self._packed_query = torch.zeros(
                self.max_packed_query_rows,
                padded_heads,
                HEAD_DIM,
                dtype=dtype,
                device=self.device,
            )
        buf = self._packed_query
        assert buf.shape[1] == padded_heads and buf.dtype == dtype
        if rows > buf.shape[0]:
            return None
        return buf[:rows]

    def _mega_paged_prefill(
        self,
        forward_batch: ForwardBatch,
        *,
        num_qo_tokens: int,
        layer_id: int,
        compress_ratio: int,
        num_heads: int,
    ) -> bool:
        if (
            num_heads != 64
            or compress_ratio not in (1, 2)
            or num_qo_tokens > 1024
            or not forward_batch.forward_mode.is_extend_without_speculative()
            or forward_batch.seq_lens_cpu is None
            or self.token_to_kv_pool.get_extra_key_bytes_per_token(layer_id) != 288
        ):
            return False
        gather_tokens = int(forward_batch.seq_lens_cpu.sum()) // compress_ratio
        return gather_tokens >= max(32768, 192 * num_qo_tokens)

    def _forward_mega_decode(
        self,
        *,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_page_indices: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        extra_k_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
        q_head_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from sglang.kernels.ops.attention import flash_mla_interface

        data, scale, _ = flash_mla_interface.flash_mla_mega_decode(
            q,
            swa_k_cache,
            swa_page_indices,
            positions,
            cos_sin_cache,
            self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=swa_topk_lengths,
            extra_kv=extra_k_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_lengths,
            use_legacy=not self._swa_interleaved,
            q_head_scale=q_head_scale,
        )
        return data, scale

    def _forward_mega_prefill(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_length: torch.Tensor,
        sf_swizzled_128x4: bool = False,
        q_head_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from sglang.kernels.ops.attention import flash_mla_interface

        def run(rows: slice, swizzled: bool):
            data, scale, _, _ = flash_mla_interface.flash_mla_mega_prefill(
                q[rows],
                kv,
                indices[rows],
                positions[rows],
                cos_sin_cache,
                self.softmax_scale,
                attn_sink=attn_sink,
                topk_length=topk_length[rows],
                use_legacy=not self._swa_interleaved,
                out_sf_swizzled_128x4=swizzled,
                q_head_scale=q_head_scale,
            )
            if swizzled:
                # [n_wv_group, 128x4-swizzled words]
                groups = data.shape[1]
                scale = scale.as_strided(
                    (groups, scale.numel() // groups),
                    (scale.numel() // groups, 1),
                    scale.storage_offset(),
                )
            return data, scale

        # The 128x4 scales need whole 128-row tiles and the current kernels: the
        # 128-aligned head gets them, a tail of < 128 rows keeps the packed ones.
        head_rows = q.shape[0] // 128 * 128
        if not (sf_swizzled_128x4 and head_rows and self._swa_interleaved):
            return run(slice(None), False)
        head = run(slice(0, head_rows), True)
        tail = run(slice(head_rows, None), False) if head_rows < q.shape[0] else None
        return MegaPrefillSplit(*head, *(tail or (None, None))), None
