from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import add_prefix, align, is_cuda, is_hip, is_npu

if is_cuda():
    import deep_gemm

from sglang.srt.layers.attention.nsa.utils import NSA_DUAL_STREAM, NSA_USE_REAL_INDEXER
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

DEBUG_NSA = bool(int(os.getenv("SGLANG_DEBUG_NSA", "0")))
DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        Args:
            logits: Input logits tensor
            topk: Number of top elements to select
            lengths: Optional per-query lengths for ragged attention
                    (can override metadata's internal lengths)

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


class V32LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)


class Indexer(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if is_cuda():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.k_norm = V32LayerNorm(self.head_dim)
        # NOTE: weight_proj is not quantized
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    def _forward_fake(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ):
        bs = x.shape[0]
        assert self.index_topk == 2048
        ans = torch.arange(0, self.index_topk, dtype=torch.int32, device=x.device)[
            None, ...
        ].repeat(bs, 1)
        if forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.seq_lens_cpu is not None
            )
            which = 0
            for i, (kv_len, qo_len) in enumerate(
                zip(
                    forward_batch.seq_lens_cpu.tolist(),
                    forward_batch.extend_seq_lens_cpu,
                    strict=True,
                )
            ):
                for j in range(kv_len - qo_len, kv_len):
                    ans[which, j + 1 :] = -1
                    which += 1
            assert which == ans.shape[0]
        else:
            assert forward_batch.seq_lens_cpu is not None
            for i, seq_len in enumerate(forward_batch.seq_lens_cpu.tolist()):
                ans[i, seq_len:] = -1

        return ans

    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x)
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ):

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # TODO we should also put DeepGEMM half SM here?
                key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            query = rotate_activation(query)
            key = rotate_activation(key)

        return query, key

    def _eff_topk(self, lengths: torch.Tensor) -> int:
        """Dynamic topk: does not exceed actual visible length"""
        maxL = int(lengths.max().item()) if lengths.numel() > 0 else 0
        return min(self.index_topk, maxL)

    def _pad_topk(self, topk_idx: torch.Tensor, target_k: int) -> torch.Tensor:
        """Pad (N, k_eff) to (N, target_k); pad with -1 if insufficient"""
        n, k_eff = topk_idx.shape[0], topk_idx.shape[-1]
        if k_eff == target_k:
            return topk_idx
        pad = topk_idx.new_full((n, target_k - k_eff), -1)
        return torch.cat([topk_idx, pad], dim=-1)

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        assert page_size == 64, "only support page size 64"

        # NOTE(dark): this support extend/decode/decode+graph
        block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        seqlens_32 = metadata.get_seqlens_int32()
        
        # Self-check for paged batch dimension consistency with defensive clipping
        bs_seqlens = seqlens_32.numel()
        bs_blocks = block_tables.shape[0]
        bs_q = q_fp8.shape[0]
        bs_weights = weights.shape[0]
        if not (bs_seqlens == bs_blocks == bs_weights == bs_q):
            if DEBUG_NSA:
                print(
                    f"[NSA-PAGED] batch dims mismatch: seqlens={bs_seqlens}, blocks={bs_blocks}, "
                    f"weights={bs_weights}, q={bs_q} -> clipping to min"
                )
            m = min(bs_seqlens, bs_blocks, bs_q, bs_weights)
            seqlens_32 = seqlens_32[:m]
            block_tables = block_tables[:m]
            q_fp8 = q_fp8[:m]
            weights = weights[:m]
        
        # NOTE(dark): 132 is SM count on H200/B200, not magic number
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32, blocksize, self.sm_count
        )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=True,  # Enable kernel-level NaN/Inf cleaning
        )

        # Ensure contiguous for downstream kernel
        logits = logits.contiguous()
        
        # NaN/Inf observation sentinel (before cleaning)
        if DEBUG_NSA or bool(int(os.getenv("SGLANG_NSA_MONITOR_NANINF", "0"))):
            bad_before = (~torch.isfinite(logits)).sum().item()
            bad_ratio = bad_before / max(logits.numel(), 1)
            if bad_ratio > 0.001:  # Warn if > 0.1%
                print(f"[NSA-PAGED:sentinel] NaN/Inf ratio BEFORE cleaning: {bad_before}/{logits.numel()} ({bad_ratio:.2%})")
        
        # First-pass NaN/Inf cleaning (nan_to_num style)
        logits = torch.nan_to_num(logits, nan=-1e30, posinf=-1e30, neginf=-1e30)
        
        # Per-row masking for decode (align with vLLM MTP)
        if forward_batch.forward_mode.is_decode_or_idle():
            # Each query's upper bound is its sequence length
            num_q = logits.shape[0]
            if seqlens_32.numel() == num_q:
                index_end_pos = seqlens_32
            else:
                # Broadcast per-batch seqlens to per-query
                index_end_pos = seqlens_32.repeat_interleave(num_q // seqlens_32.numel())
            
            # Mask positions beyond each query's valid range
            positions = torch.arange(logits.shape[1], device=logits.device, dtype=torch.int32)
            mask = positions.unsqueeze(0) >= index_end_pos.unsqueeze(1)
            logits = logits.masked_fill(mask, float('-inf'))
        
        # Second-pass cleaning (fallback for edge cases)
        _cleaned_once = getattr(self, "_cleaned_once_paged", False)
        bad = ~torch.isfinite(logits)
        if bad.any():
            if DEBUG_NSA and not _cleaned_once:
                print(f"[NSA-PAGED] cleaned NaN/Inf logits: {int(bad.sum())}/{logits.numel()}")
                self._cleaned_once_paged = True
            logits = logits.masked_fill(bad, -1e30)
        
        # Perform top-k
        topk_result = metadata.topk_transform(logits, self.index_topk)
        
        # Second-pass: clamp top-k indices to valid range (insurance against NaN/Inf)
        # This ensures no index exceeds the per-row upper bound
        if forward_batch.forward_mode.is_decode_or_idle():
            # Expand index_end_pos to match topk_result shape
            if index_end_pos.numel() == topk_result.shape[0]:
                end_pos_expanded = index_end_pos.unsqueeze(1)  # [num_q, 1]
            else:
                end_pos_expanded = index_end_pos.unsqueeze(1)
            
            # Mark out-of-bounds indices as -1
            topk_result = torch.where(
                topk_result > end_pos_expanded,
                torch.full_like(topk_result, -1),
                topk_result
            )
        
        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        # Early exit for empty batch to avoid kernel invocation issues
        if forward_batch.batch_size == 0:
            return torch.full(
                (0, self.index_topk), -1, dtype=torch.int, device=q_fp8.device
            )

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        
        # Get seqlens_expanded and page_table early for KV=0 check
        seq_lens_expanded = metadata.get_seqlens_expanded()
        block_tables = metadata.get_page_table_64()
        
        # Early exit for KV=0 case (VERIFY with no context, or empty batch)
        # Check three conditions: page_table columns=0, or seqlens_expanded empty/all-zero
        if (block_tables.shape[1] == 0 or 
            seq_lens_expanded.numel() == 0 or 
            int(seq_lens_expanded.max().item()) == 0):
            num_q_out = q_fp8.shape[0] if q_fp8.shape[0] > 0 else int(seq_lens_expanded.numel())
            if DEBUG_NSA:
                print(f"[NSA-RAGGED:kv0-early-exit] maxK=0 or empty, returning empty topk for num_q={num_q_out}")
            return torch.full((num_q_out, self.index_topk), -1, dtype=torch.int, device=q_fp8.device)
        
        k_fp8_list = []
        k_scale_list = []

        # block_tables already obtained in early-exit check
        seq_lens_cpu_merged = forward_batch.seq_lens_cpu

        # Correct relationship: block_tables rows == batch size; num_q == sum(extend_list)
        B = forward_batch.batch_size
        assert block_tables.shape[0] == B, \
            f"[NSA-RAGGED] page_table rows({block_tables.shape[0]}) != batch_size({B})"
        # num_q consistency is guaranteed by later assertion between seq_lens_expanded and q_fp8
        
        # Single-fragment view: metadata already aligned per fragment, no cross-fragment merging
        eff_ext_list = list(metadata.attn_metadata.nsa_extend_seq_lens_list)
        
        assert len(eff_ext_list) == B, \
            f"extend_list({len(eff_ext_list)}) must equal batch_size({B})"
        
        # Vectorized computation setup
        S = seq_lens_cpu_merged.to(dtype=torch.int32, device=q_fp8.device)  # [B] prefix KV lengths
        counts = torch.tensor(eff_ext_list, dtype=torch.int32, device=q_fp8.device)  # [B] query counts (=qo)
        
        # K buffer total length = prefix + (qo - 1) drafts
        # When verifying the j-th draft, need to see previous (j-1) drafts
        S_total = S + counts - 1  # [B]
        
        # Build K buffer including draft tokens
        for i in range(B):
            seq_len_total = int(S_total[i].item())
            if DEBUG_NSA and i == 0:  # Only print first sample
                print(f"[RAGGED] K buffer: sample_0 seq_len_total={seq_len_total} "
                      f"(prefix={int(S[i].item())} + draft={int(counts[i].item()-1)})")
            
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len_total,  # ✅ Includes draft
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len_total,  # ✅ Includes draft
                block_tables[i],
            )
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        
        # Each query's batch ID
        batch_id = torch.repeat_interleave(
            torch.arange(B, device=q_fp8.device, dtype=torch.int32), counts
        )  # [N] where N = sum(eff_ext_list)
        
        # Position of each query in its request group [0, 1, ..., qo-1]
        # Used to compute how many drafts this query can see
        pos_per_q = torch.cat([
            torch.arange(counts[b].item(), device=q_fp8.device, dtype=torch.int32)
            for b in range(B)
        ])  # [N]
        
        # Base offset for each batch in the concatenated K buffer (prefix sum)
        # Note: base here is cumulative based on S_total
        base_offsets = torch.cumsum(S_total, dim=0, dtype=torch.int32) - S_total  # [B]
        base_per_q = base_offsets[batch_id]  # [N]
        
        S_per_q = S[batch_id]  # [N] Prefix KV length for each query
        
        # Get num_q early for validation
        num_q = q_fp8.shape[0]
        orig_num_q = num_q  # Save original num_q for later restoring filtered rows
        
        if DEBUG_NSA:
            print(
                f"[NSA-RAGGED] num_q={num_q}, seqlens_expanded={seq_lens_expanded.numel()}, "
                f"sum_eff={sum(eff_ext_list)}, md_ptr={seq_lens_expanded.data_ptr()}"
            )
        
        # Defensive length check: ensure seq_lens_expanded dimension matches num_q (before computing ks/ke)
        if seq_lens_expanded.numel() != num_q:
            # AUTOFIX is controllable (default: disabled to avoid masking upstream bugs)
            ENABLE_AUTOFIX = bool(int(os.getenv("SGLANG_NSA_ENABLE_RAGGED_AUTOFIX", "0")))
            
            if not ENABLE_AUTOFIX:
                raise AssertionError(
                    f"[NSA-RAGGED] metadata.nsa_seqlens_expanded has size {seq_lens_expanded.numel()} but "
                    f"num_q={num_q}. sum(eff_ext_list)={sum(eff_ext_list)}, eff_ext_list={eff_ext_list}. "
                    f"Set SGLANG_NSA_ENABLE_RAGGED_AUTOFIX=1 to enable auto-padding (not recommended)."
                )
            
            # Try conservative auto-fix: if every request contributes the same qo,
            # pad missing queries with length=1 to reach bsz*qo.
            qo = None
            if isinstance(eff_ext_list, (list, tuple)) and len(eff_ext_list) > 0:
                qo = eff_ext_list[0]
                for v in eff_ext_list:
                    if v != qo:
                        qo = None
                        break

            missing = int(num_q) - int(seq_lens_expanded.numel())
            if missing > 0 and qo is not None and (missing % qo == 0):
                pad = torch.ones(
                    missing,
                    dtype=seq_lens_expanded.dtype,
                    device=seq_lens_expanded.device,
                )
                old_n = int(seq_lens_expanded.numel())
                seq_lens_expanded = torch.cat([seq_lens_expanded, pad], dim=0)
                print(
                    f"[NSA-RAGGED-AUTOFIX] padded seqlens_expanded {old_n} -> "
                    f"{int(seq_lens_expanded.numel())} using qo={qo}, missing={missing}"
                )
            else:
                raise AssertionError(
                    f"metadata.nsa_seqlens_expanded has size {seq_lens_expanded.numel()} but "
                    f"num_q={num_q}. sum(eff_ext_list)={sum(eff_ext_list)}, eff_ext_list={eff_ext_list}"
                )

        # Now seq_lens_expanded dimension is guaranteed correct, can safely compute ks/ke
        # Right-aligned ks/ke with draft context
        # ke[j] = base + prefix_len + j (j-th query sees prefix + j tokens)
        # ks[j] = ke[j] - seqlens_expanded[j] (right-aligned window)
        ke = base_per_q + S_per_q + pos_per_q  # Includes (j-1) drafts
        lengths = seq_lens_expanded
        ks = ke - lengths  # Right-aligned
        
        # Pre-normalize before DeepGEMM: ensure dtype/contiguous consistency
        ks = ks.to(torch.int32).contiguous()
        ke = ke.to(torch.int32).contiguous()
        weights = weights.contiguous()  # Already squeezed, ensure contiguous
        q_fp8 = q_fp8.contiguous()
        
        # Self-test assertions (aligned to vLLM's correctness checks with draft context)
        if DEBUG_NSA:
            # 1. Right-alignment: ke - ks should equal seqlens_expanded
            lengths_check = ke - ks
            assert (lengths_check == seq_lens_expanded).all(), \
                f"[ASSERT] Right-alignment failed: (ke-ks) != seqlens_expanded"
            
            # 2. ks should not go before batch base offset
            assert (ks >= base_per_q).all(), \
                f"[ASSERT] ks went before batch base: min(ks-base)={int((ks - base_per_q).min().item())}"
            
            # 3. ke should equal base + prefix + pos_per_q (includes draft context)
            expected_ke = base_per_q + S_per_q + pos_per_q
            assert (ke == expected_ke).all(), \
                f"[ASSERT] ke != base + prefix + pos (draft context missing)"
            
            print(f"[NSA-RAGGED:self-test] Right-alignment verified")
            print(f"[RAGGED] ks[:8]={ks[:8].tolist()}, ke[:8]={ke[:8].tolist()}, "
                  f"len[:8]={lengths[:8].tolist()}, pos[:8]={pos_per_q[:8].tolist()}")
        
        # Diagnostic prints and alignment assertions
        if DEBUG_NSA:
            print(f"[NSA-RAGGED:pre] "
                  f"num_q={int(q_fp8.shape[0])}, "
                  f"weights={tuple(weights.shape)}, "
                  f"ks.shape={tuple(ks.shape)}, ke.shape={tuple(ke.shape)}, "
                  f"ks_vals={ks.tolist()[:10]}, ke_vals={ke.tolist()[:10]}, "
                  f"len(expanded)={int(seq_lens_expanded.numel())}, "
                  f"block_tables={tuple(block_tables.shape) if block_tables is not None else 'None'}")
        
        # Consistency checks before entering DeepGEMM
        
        # Core alignment: Q/ks/ke/seqlens_expanded must all be per-query (same length)
        assert q_fp8.shape[0] == seq_lens_expanded.numel(), \
            f"num_q({q_fp8.shape[0]}) != len(expanded)({seq_lens_expanded.numel()})"
        assert ks.numel() == q_fp8.shape[0], \
            f"ks({ks.numel()}) != num_q({q_fp8.shape[0]})"
        assert ke.numel() == q_fp8.shape[0], \
            f"ke({ke.numel()}) != num_q({q_fp8.shape[0]})"
        
        # Dimension consistency
        assert ks.numel() == num_q, \
            f"ks({ks.numel()}) != num_q({num_q}), mode={forward_batch.forward_mode.name}, " \
            f"eff_ext_list={eff_ext_list}, sum={sum(eff_ext_list)}"
        assert ke.numel() == num_q, f"ke({ke.numel()}) != num_q({num_q})"
        assert seq_lens_expanded.numel() == num_q, \
            f"seqlens_expanded({seq_lens_expanded.numel()}) != num_q({num_q})"
        
        # Segment validity with defensive filtering for KV=0 cases
        lengths = ke - ks
        
        # Defense: detect and filter invalid segments (ke <= ks may occur when KV=0)
        valid_mask = lengths > 0
        if not valid_mask.all():
            if DEBUG_NSA:
                num_invalid = int((~valid_mask).sum().item())
                print(f"[NSA-RAGGED:filter] Found {num_invalid} invalid segments (ke<=ks), filtering them out")
                print(f"  ks_vals={ks.tolist()[:20]}")
                print(f"  ke_vals={ke.tolist()[:20]}")
                print(f"  lengths={lengths.tolist()[:20]}")
            
            # If all invalid, return empty result
            if not valid_mask.any():
                if DEBUG_NSA:
                    print(f"[NSA-RAGGED:empty] All segments invalid, returning empty topk result")
                return torch.full((num_q, self.index_topk), -1, dtype=torch.int, device=q_fp8.device)
            
            # Filter out valid segments
            q_fp8 = q_fp8[valid_mask]
            weights = weights[valid_mask]
            ks = ks[valid_mask]
            ke = ke[valid_mask]
            seq_lens_expanded = seq_lens_expanded[valid_mask]
            lengths = lengths[valid_mask]
            num_q = int(q_fp8.shape[0])
        
        # Final check: all segments must be valid
        assert (lengths > 0).all(), \
            f"Invalid segment lengths: some ke[i] <= ks[i]. min(ke-ks)={lengths.min().item()}"
        assert (seq_lens_expanded > 0).all(), "seqlens_expanded must be positive lengths"
        
        # ---- Dynamic topk + small-K fast path ----
        eff_topk = self._eff_topk(seq_lens_expanded)  # min(index_topk, max(lengths))
        
        SMALLK_THRESH = int(os.getenv("SGLANG_NSA_SMALLK_THRESH", "32"))
        use_smallk_fastpath = (int(seq_lens_expanded.max().item()) <= SMALLK_THRESH)
        
        if use_smallk_fastpath:
            # Small-K fast path: construct "fake logits" to let topk select first lengths[i] positions for each query
            maxL = int(seq_lens_expanded.max().item())
            if maxL == 0:
                return torch.full((num_q, self.index_topk), -1, dtype=torch.int, device=q_fp8.device)
            
            # Shape: (num_q, maxL), scores descending
            fake_scores = torch.arange(maxL, 0, -1, device=q_fp8.device, dtype=torch.float32
                                      ).unsqueeze(0).expand(num_q, -1).contiguous()
            
            # Use metadata.topk_transform (AccumulatingIndexerMetadata will choose non-fused version)
            # Explicitly pass current seq_lens_expanded
            smallk_idx = metadata.topk_transform(fake_scores, eff_topk, seq_lens_expanded)
            
            # Pad to index_topk
            if smallk_idx.shape[-1] < self.index_topk:
                smallk_idx = self._pad_topk(smallk_idx, self.index_topk)
            
            if DEBUG_NSA:
                print(f"[NSA-RAGGED:smallk] maxL={maxL}, eff_topk={eff_topk}, skipped DeepGEMM")
            
            return smallk_idx
        else:
            # Regular large-K path: run DeepGEMM, but still use eff_topk for topk, then pad to index_topk for external use
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                kv_fp8,
                weights,
                ks,
                ke,
                clean_logits=True,  # Enable kernel-level NaN/Inf cleaning
            )

            # Ensure contiguous for downstream kernel
            logits = logits.contiguous()
            
            # NaN/Inf observation sentinel (before cleaning)
            if DEBUG_NSA or bool(int(os.getenv("SGLANG_NSA_MONITOR_NANINF", "0"))):
                bad_before = (~torch.isfinite(logits)).sum().item()
                bad_ratio = bad_before / max(logits.numel(), 1)
                if bad_ratio > 0.001:  # Warn if > 0.1%
                    print(f"[NSA-RAGGED:sentinel] NaN/Inf ratio BEFORE cleaning: {bad_before}/{logits.numel()} ({bad_ratio:.2%})")
            
            # First-pass NaN/Inf cleaning (nan_to_num style)
            logits = torch.nan_to_num(logits, nan=-1e30, posinf=-1e30, neginf=-1e30)
            
            # Second-pass cleaning (fallback for edge cases)
            _cleaned_once = getattr(self, "_cleaned_once", False)
            bad = ~torch.isfinite(logits)
            if bad.any():
                if DEBUG_NSA and not _cleaned_once:
                    print(f"[NSA] cleaned NaN/Inf logits: {int(bad.sum())}/{logits.numel()}")
                    self._cleaned_once = True
                logits = logits.masked_fill(bad, -1e30)
            
            # Post-DeepGEMM diagnostic
            if DEBUG_NSA:
                bad = (~torch.isfinite(logits)).sum().item()
                print(f"[NSA-RAGGED:post] logits.shape={tuple(logits.shape)}, naninf={bad}")

            assert logits.shape[0] == len(seq_lens_expanded)
            
            # Use "smaller" eff_topk for selection to reduce meaningless -1; then uniformly pad
            # Explicitly pass "current valid expanded lengths" to avoid getting stale ones from metadata
            topk_idx_eff = metadata.topk_transform(logits, eff_topk, seq_lens_expanded)
            
            # Second-pass: clamp top-k indices to valid range (insurance against NaN/Inf)
            # Ensure no index exceeds per-row upper bound (seq_lens_expanded defines visible length per query)
            topk_idx_eff = torch.where(
                topk_idx_eff >= seq_lens_expanded.unsqueeze(1),
                torch.full_like(topk_idx_eff, -1),
                topk_idx_eff
            )
            
            # Self-test: Top-K should not exceed valid lengths (per-row check)
            if DEBUG_NSA:
                # Stricter per-row validation
                per_row_ok = (topk_idx_eff == -1) | (topk_idx_eff < seq_lens_expanded.view(-1, 1))
                if not per_row_ok.all():
                    bad_rows = (~per_row_ok).any(dim=1)
                    bad_count = int(bad_rows.sum().item())
                    raise AssertionError(
                        f"[ASSERT] Per-row top-k index out of range: {bad_count} rows violated. "
                        f"First bad row: {int(bad_rows.nonzero()[0].item())}"
                    )
                print(f"[NSA-RAGGED:self-test] Per-row Top-K bounds verified")
            
            # If invalid segments were filtered earlier, need to scatter back to original positions (not cat to end)
            if topk_idx_eff.shape[0] < orig_num_q:
                # Create all-(-1) result tensor
                restored = topk_idx_eff.new_full((orig_num_q, topk_idx_eff.shape[1]), -1)
                # Scatter valid results back to original positions
                restored[valid_mask] = topk_idx_eff
                topk_idx_eff = restored
                
                # Self-test: Verify scatter correctness
                if DEBUG_NSA:
                    valid_count = int(valid_mask.sum().item())
                    assert (restored[valid_mask] != -1).any() or valid_count == 0, \
                        "[ASSERT] Scatter failed: no valid results in restored positions"
                    assert (restored[~valid_mask] == -1).all(), \
                        "[ASSERT] Scatter failed: invalid positions should be -1"
                    print(f"[NSA-RAGGED:self-test] Scatter restore verified ({valid_count}/{orig_num_q} valid)")
            
            if topk_idx_eff.shape[-1] < self.index_topk:
                topk_idx_eff = self._pad_topk(topk_idx_eff, self.index_topk)
            
            return topk_idx_eff

    def forward_indexer_bs_1(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        if not is_npu():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import fp8_index

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        # logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
        k_fp8_list = []
        k_scale_list = []

        topk_indices_list = []

        block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided_indices = torch.arange(
            0, block_tables.shape[-1], page_size,
            device=block_tables.device, dtype=torch.int32
        )
        block_tables = block_tables[:, strided_indices] // page_size

        q_len_start = 0

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens[i].item()  # prefix length (S)
            q_len = (
                forward_batch.extend_seq_lens_cpu[i]  # qo in VERIFY
                if forward_batch.forward_mode.is_extend()
                else 1
            )
            # K buffer total length = prefix + (qo - 1) drafts
            seq_len_total = seq_len + q_len - 1  # S_total = S + qo - 1
            
            q_len_end = q_len_start + q_len

            q_fp8_partial = q_fp8[q_len_start:q_len_end]
            q_fp8_partial = q_fp8_partial.unsqueeze(0).contiguous()

            weights_partial = weights[q_len_start:q_len_end]
            weights_partial = weights_partial.squeeze(-1).unsqueeze(0).contiguous()

            # Get continuous K including draft
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len_total,  # Includes draft
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len_total,  # Includes draft
                block_tables[i],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn).unsqueeze(0).contiguous()
            k_scale = k_scale.view(torch.float32).squeeze(-1).unsqueeze(0).contiguous()

            index_score = fp8_index(
                q_fp8_partial,
                weights_partial,
                k_fp8,
                k_scale,
            )
            # end_pos should be K buffer's total length
            end_pos = seq_len_total
            topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

            # Pad to index_topk (aligned to 32) instead of hardcoded 2048
            pad_target = align(self.index_topk, 32)
            pad_len = max(0, pad_target - topk_indices.shape[-1])
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, pad_len), "constant", -1
            )

            topk_indices_list.append(topk_indices)

            q_len_start = q_len_end

        topk_indices = torch.cat(topk_indices_list, dim=0)

        return topk_indices

    def forward_indexer(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        return self.forward_indexer_bs_1(q_fp8, weights, forward_batch, topk, layer_id)

    def _forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        if not is_npu():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        # Directly get batch-local metadata to avoid concurrent batch overwrites
        metadata = getattr(forward_batch, "nsa_indexer_metadata", None)
        if metadata is None:
            # Allow skipping NSA in IDLE / empty batch / no query cases
            if (forward_batch.forward_mode.is_idle()
                or forward_batch.batch_size == 0
                or q_lora.numel() == 0):
                if DEBUG_NSA:
                    print("[NSA] Skip indexer: idle/empty batch, no batch-local metadata")
                return None
            # Non-idle mode must have batch-local metadata (strict enforcement)
            raise AssertionError(
                "[NSA] missing batch-local nsa_indexer_metadata in non-idle mode "
                "(did you forget to call init_forward_metadata?)"
            )

        enable_dual_stream = (
            NSA_DUAL_STREAM
            and self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        if not NSA_USE_REAL_INDEXER:  # temporary
            return self._forward_fake(x, q_lora, positions, forward_batch, layer_id)

        query, key = self._get_q_k_bf16(q_lora, x, positions, enable_dual_stream)

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        # k_fp8: (seq_len, head_dim) fp8_e4m3fn
        # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
        # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
        # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
        
        # --- Ensure loc/k have expected layout and type ---
        loc = forward_batch.out_cache_loc
        if not loc.is_contiguous():
            # Optional: print once for debugging to confirm source
            if DEBUG_NSA:
                print(f"[NSA] out_cache_loc not contiguous; shape={tuple(loc.shape)}, stride={tuple(loc.stride())} -> making contiguous")
            loc = loc.contiguous()
        
        index_k = k_fp8.contiguous()
        index_k_scale = k_scale.contiguous()
        
        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(
            layer_id=layer_id,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )

        weights = self._get_logits_head_gate(x, q_scale)
        
        # Intra-fragment slicing (only for extend/verify when _nsa_acc_range exists)
        if (forward_batch.forward_mode.is_extend() and hasattr(forward_batch, "_nsa_acc_range")
            and not bool(int(os.getenv("SGLANG_NSA_DISABLE_Q_SLICE","0")))):
            s, e = map(int, getattr(forward_batch, "_nsa_acc_range"))
            s = max(0, s)
            e = min(e, int(q_fp8.shape[0]))
            if e > s:
                q_fp8 = q_fp8[s:e].contiguous()
                weights = weights[s:e].contiguous()
            else:
                if DEBUG_NSA:
                    print(f"[NSA] Empty fragment window: range=({s},{e}), skip indexer")
                return torch.full((0, self.index_topk), -1, dtype=torch.int, device=q_fp8.device)

        if is_cuda():
            assert forward_batch.seq_lens_cpu is not None
            if len(forward_batch.seq_lens_cpu) == 0:
                # this seems b/c max-pad, no worries?
                # if x.shape[0] != 0:
                #     print(
                #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
                #     )
                return torch.full(
                    (x.shape[0], self.index_topk), -1, dtype=torch.int, device=q_fp8.device
                )

            if forward_batch.forward_mode.is_decode_or_idle():
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                topk_result = self._get_topk_ragged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
        else:
            topk_result = self.forward_indexer(
                q_fp8.contiguous(),
                weights,
                forward_batch,
                topk=self.index_topk,
                layer_id=layer_id,
            )

        return topk_result

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        return self._forward(x, q_lora, positions, forward_batch, layer_id)

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        import custom_ops
        import torch_npu

        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
        )
        from sglang.srt.utils import get_bool_env_var

        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        enable_index_cp = (
            get_bool_env_var("SGLANG_USE_AG_AFTER_QLORA") and layer_id >= 4
        )
        is_prefill = forward_batch.forward_mode.is_extend()

        attention_tp_rank = get_attention_tp_rank()
        attention_tp_size = get_attention_tp_size()

        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        if is_prefill and enable_index_cp:
            slice_length = cos.shape[0] // attention_tp_size
            cos = cos[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]
            sin = sin[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]

        slot_mapping = forward_batch.out_cache_loc
        block_table = forward_batch.attn_backend.forward_metadata.block_tables

        bs = x.shape[0]

        q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
        q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64, 64 + 64]

        q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(
            bs, self.n_heads, self.rope_head_dim
        )  # [bs, n, d]
        q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
        k = self.k_norm(k_proj)
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64 + 64]

        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(
            bs, 1, self.rope_head_dim
        )  # [bs, 1, d]
        k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        if is_prefill and enable_index_cp:
            k, local_k = (
                torch.empty(
                    (k.shape[0] * attention_tp_size, k.shape[1], k.shape[2]),
                    dtype=k.dtype,
                    device=k.device,
                ),
                k,
            )
            get_attention_tp_group().all_gather_into_tensor(k, local_k)

        forward_batch.token_to_kv_pool.set_index_k_buffer(layer_id, slot_mapping, k)

        indexer_input = {}
        if is_prefill:
            actual_seq_lengths_kv = forward_batch.seq_lens.to(device=q.device)
            actual_seq_lengths_q = forward_batch.seq_lens.cumsum(dim=0).to(
                device=q.device
            )
            if enable_index_cp:
                actual_seq_lengths_q -= bs * attention_tp_rank
                actual_seq_lengths_q = torch.max(
                    actual_seq_lengths_q,
                    torch.zeros_like(actual_seq_lengths_q).to(
                        device=actual_seq_lengths_q.device
                    ),
                )
                actual_seq_lengths_q = torch.min(
                    actual_seq_lengths_q,
                    torch.full(actual_seq_lengths_q.shape, bs).to(
                        device=actual_seq_lengths_q.device
                    ),
                )

        else:
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                actual_seq_lengths_q = torch.tensor(
                    [1 + i * 1 for i in range(bs)], dtype=torch.int32, device=k.device
                )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x)[0]
        block_table = (
            block_table[: actual_seq_lengths_q.size()[0]] if is_prefill else block_table
        )

        topk_indices = torch.ops.custom.npu_lightning_indexer(
            query=q.view(-1, self.n_heads, self.head_dim),
            key=past_key_states,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
            actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(torch.int32),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )

        if is_prefill and enable_index_cp:
            topk_indices, local_topk_indices = (
                torch.empty(
                    (
                        topk_indices.shape[0] * attention_tp_size,
                        topk_indices.shape[1],
                        topk_indices.shape[2],
                    ),
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                ),
                topk_indices,
            )
            get_attention_tp_group().all_gather_into_tensor(
                topk_indices, local_topk_indices
            )

        return topk_indices
