"""QSA indexer for Qwen4-Exp checkpoints."""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    qsa_fast_topk,
)
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
)
from sglang.srt.layers.attention.qsa.mqa import (
    HAS_TILELANG,
    qsa_mqa_decode,
    qsa_mqa_prefill,
    qsa_mqa_prefill_relative,
)
from sglang.srt.layers.cp.utils import (
    ContextParallelStrategyKind,
    cp_materialize_global_token_order,
    cp_shard_hidden_states,
    cp_shard_position_ids,
    get_cp_strategy,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_sm100_supported

logger = logging.getLogger(__name__)

# Cap on the fp32 [query_rows, compressed_keys] prefill logits workspace;
# top-k is per row, so tiling rows does not change the selection.
_QSA_PREFILL_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024


# PREFILL_SELECT variant bits 0-3 per SGLANG_QSA_PREFILL_FAST_SELECT_TOPK.
_FAST_SELECT_TOPK = {"sampled": 0, "onepass": 1}


def _fast_select_logits_budget(width: int, variant: int) -> int:
    """The source budget rule the variant declares: its wide budget (bits 8-19, MiB)
    for calls at least bits 20-30 (x 256 keys) wide, else stock's."""
    wide_mib = variant >> 8 & 0xFFF
    if wide_mib and width >= (variant >> 20 & 0x7FF) * 256:
        return wide_mib << 20
    return _QSA_PREFILL_LOGITS_BUDGET_BYTES


def _fast_select_chunk_rows(
    rows: int, width: int, sms: int, budget: int = _QSA_PREFILL_LOGITS_BUDGET_BYTES
) -> int:
    """The source fast-select plan: the logits budget in 32-row blocks,
    rounded down to whole top-k waves of 2 * sms rows."""
    max_rows = max(32, budget // (width * 4))
    max_rows = max(32, max_rows // 32 * 32)
    if rows <= max_rows:
        return rows
    unit = 2 * sms // 32 * 32
    return max_rows // unit * unit if unit >= 32 and max_rows >= unit else max_rows


def _qsa_prefill_row_chunk_size(rows: int, keys: int, heads: int) -> int:
    if rows <= 0 or keys <= 0:
        return max(rows, 1)
    block_q = max(1, 128 // heads)
    bytes_per_row = keys * torch.float32.itemsize
    max_padded_rows = max(block_q, _QSA_PREFILL_LOGITS_BUDGET_BYTES // bytes_per_row)
    max_padded_rows = max(block_q, max_padded_rows // block_q * block_q)
    return min(rows, max_padded_rows)


# SGLANG_OPT_QSA_PREFILL_TOPK_BRIDGE_DIRECT: the breakable prefill graph's top-k bridge
# (models/qwen4_exp.py _breakable_qsa_indexer) lends its rows to the next prefill selection,
# which writes them in place of a new tensor, so the bridge's full fill and its
# [rows, token_topk + ratio - 1] int32 D2D copy are skipped. One shot: the bridge is taken by
# the first select_prefill_tokens call inside prefill_topk_output() or by none.
_PREFILL_TOPK_OUT: list = [None]


class prefill_topk_output:
    """Offer ``buffer`` (int32 [>= rows, token_topk + compress_ratio - 1], unit last stride) to the
    next select_prefill_tokens call; withdrawn on exit whether taken or not."""

    __slots__ = ("_buffer",)

    def __init__(self, buffer: torch.Tensor):
        self._buffer = buffer

    def __enter__(self):
        _PREFILL_TOPK_OUT[0] = self._buffer
        return self

    def __exit__(self, *exc):
        _PREFILL_TOPK_OUT[0] = None
        return False


def _take_prefill_topk_output(rows: int, width: int, device) -> Optional[torch.Tensor]:
    buffer = _PREFILL_TOPK_OUT[0]
    if buffer is None:
        return None
    _PREFILL_TOPK_OUT[0] = None
    if (
        buffer.dtype != torch.int32
        or buffer.dim() != 2
        or buffer.shape[0] < rows
        or buffer.shape[1] != width
        or buffer.stride(1) != 1
        or buffer.device != device
    ):
        return None
    view = buffer[:rows]
    return view if view.is_contiguous() else None


class QSAIndexer(MultiPlatformOp):
    """Config-driven fused-QK, weight-free sparse-attention indexer."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        rotary_emb=None,
    ) -> None:
        self._validate_config(config)
        super().__init__()
        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.block_topk = self.token_topk // self.compress_ratio
        if rotary_emb is None:
            raise ValueError("QSAIndexer must reuse its Qwen4-Exp attention RoPE")
        self.rotary_emb = rotary_emb
        if not 0 < self.rotary_emb.rotary_dim <= self.index_head_dim:
            raise ValueError(
                "Qwen4-Exp attention RoPE rotary_dim must fit the QSA index head: "
                f"{self.rotary_emb.rotary_dim=} {self.index_head_dim=}"
            )
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self._rope_axis_map_cache = None

    def numeric_signatures(self, widths, tokens_per_request):
        self._numeric_split_select_widths = tuple(
            rows for rows in widths if type(rows) is int and rows > 0
        )
        self._numeric_mtp_tokens_per_request = tokens_per_request
        self._numeric_mtp_deferred_widths = tuple(
            rows for rows in widths if type(rows) is int and rows > 0
        )
        self._numeric_mtp_capture_widths = self._numeric_mtp_deferred_widths
        return ()

    def numeric_state_preparation_signatures(self, model_runner):
        from sglang.srt.kernel_plugins.nvidia.mtp_capture_seed import prepare
        from sglang.srt.kernel_plugins.nvidia.mtp_deferred_kv import (
            prepare as prepare_kv,
        )

        return (
            prepare(self, model_runner)
            + prepare_kv(self, model_runner)
            + self._qpf_signatures(model_runner)
        )

    def _qpf_signatures(self, model_runner) -> tuple:
        """SGLANG_QSA_QPF: the one-launch prologue's exact geometry (any row count)."""
        self._numeric_prepared_qpf = None
        if not envs.SGLANG_QSA_QPF.get():
            return ()
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            NumericOperation,
            QsaPrologueFoldSignature,
            ScalarType,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import (
            signature_capability_supported,
        )

        pool = getattr(model_runner, "token_to_kv_pool", None)
        if not hasattr(pool, "qsa_compressed_dtype"):
            pool = getattr(pool, "full_kv_pool", None)
        scalar = {torch.bfloat16: ScalarType.BF16, torch.float16: ScalarType.FP16}
        out_scalar = {**scalar, torch.float8_e4m3fn: ScalarType.FP8_E4M3}
        if (
            pool is None
            or not hasattr(pool, "qsa_compressed_dtype")
            or not self._use_fused_compress(pool)
            or pool.index_state_dtype not in scalar
            or pool.qsa_compressed_dtype not in out_scalar
            or self.index_kv_heads != 1
            or not 0 < self.compress_ratio <= 16
        ):
            return ()
        signature = QsaPrologueFoldSignature(
            NumericOperation.QSA_PROLOGUE_FOLD,
            0,
            self.index_n_heads,
            ((self.index_n_heads + 7) // 8) * 8,
            self.index_head_dim,
            self.rotary_emb.rotary_dim,
            3 if getattr(self.rotary_emb, "mrope_section", None) else 1,
            self.compress_ratio,
            scalar[pool.index_state_dtype],
            out_scalar[pool.qsa_compressed_dtype],
            int(bool(self.rotary_emb.is_neox_style)),
        )
        if not signature_capability_supported(
            signature, spec=getattr(self, "_numeric_capability_spec", None)
        ):
            return ()
        self._numeric_prepared_qpf = signature
        return (signature,)

    def _try_prologue_fold(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata,
        state_slots: torch.Tensor | None,
        projected_qk: torch.Tensor | None,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """project_qk's fused q_prep and the graph compression (_compress_decode_cuda_graph's
        fused branch) as one launch. Returns (q, qk); q is None when the rows do not
        qualify, and qk (if projected here) is handed to the two-launch path."""
        signature = getattr(self, "_numeric_prepared_qpf", None)
        pool = metadata.token_to_kv_pool
        group_locs = metadata.graph_ring_group_locs
        write_locs = metadata.graph_write_locs
        if (
            signature is None
            or pool is None
            or state_slots is None
            or group_locs is None
            or write_locs is None
            or hidden_states.shape[0] == 0
            or not self._use_fused_compress(pool)
        ):
            return None, projected_qk
        from sglang.kernels.plugin_kernels.nvidia.numeric import QsaPrologueFoldIO
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

        run = numeric_operator(signature, prepared=True)
        if run is None:
            return None, projected_qk
        qk = projected_qk
        if qk is None:
            qk, _ = self.index_qk_proj(hidden_states)
        key_state_buffer = pool.get_qsa_key_state_buffer(self.layer_id)
        ring = key_state_buffer.view(key_state_buffer.shape[0], -1)
        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)
        compressed = compressed_buffer.view(compressed_buffer.shape[0], -1)
        out_dtype = pool.qsa_compressed_dtype
        rope_positions = positions.long()
        if rope_positions.ndim == 1:
            rope_positions = rope_positions.unsqueeze(0)
        if (
            qk.shape[0] == 0
            or not self._use_fused_prep(qk)
            or not qk.is_contiguous()
            or ring.dtype != qk.dtype
            or pool.index_state_dtype != qk.dtype
            or not ring.is_contiguous()
            or compressed.dtype != out_dtype
            or group_locs.dtype != torch.int32
            or write_locs.dtype != torch.int32
            or group_locs.ndim != 2
            or group_locs.shape[1] != self.compress_ratio
            or group_locs.shape[0] == 0
            or write_locs.shape != group_locs.shape[:1]
            or not group_locs.is_contiguous()
            or rope_positions.shape[0] != signature.rope_axes
        ):
            return None, qk
        if not get_is_capture_mode() and hasattr(
            self.rotary_emb, "_ensure_cos_sin_cache_length"
        ):
            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))
        q = torch.empty(
            (qk.shape[0], signature.q_heads_padded, self.index_head_dim),
            dtype=out_dtype,
            device=qk.device,
        )
        run(
            QsaPrologueFoldIO(
                qk=qk,
                positions=rope_positions,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                axis_map=self._rope_axis_map(qk.device),
                q_weight=self.q_layernorm.weight.data,
                cache_loc=state_slots[: qk.shape[0]].long(),
                key_state_buffer=ring,
                rope_position_buffer=pool.qsa_rope_position_buffer,
                group_locs=group_locs,
                k_weight=self.k_layernorm.weight.data,
                write_locs=write_locs,
                compressed_k_buffer=compressed,
                q_out=q,
                q_eps=self.q_layernorm.variance_epsilon,
                k_eps=self.k_layernorm.variance_epsilon,
            )
        )
        if not getattr(self, "_qpf_engaged_logged", False):
            logger.info(
                "Q38_QPF_ACTIVE layer=%s rows=%s groups=%s dtype=%s",
                self.layer_id,
                qk.shape[0],
                group_locs.shape[0],
                qk.dtype,
            )
            self._qpf_engaged_logged = True
        return q, qk

    def numeric_preparation_signatures(self, context_len: int, page_size: int):
        """Describe actual scorer/select geometry; provider admission is separate."""
        self._numeric_score_capacity = 0
        self._numeric_page_size = page_size
        self._numeric_prepared_sparse = frozenset()
        widths = getattr(self, "_numeric_split_select_widths", ())
        grouped4 = self._grouped4_score_variant()
        if not (
            (
                envs.SGLANG_QSA_MK_SCORE.get()
                or envs.SGLANG_QSA_SPLIT_TOPK.get()
                or grouped4 is not None
                or envs.SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT.get()
                or self._verify_select_source_enabled()
            )
            and type(context_len) is int
            and context_len > 0
            and type(page_size) is int
            and page_size > 0
            and self.index_n_heads > 0
            and self.index_kv_heads > 0
            and self.index_head_dim > 0
            and self.compress_ratio > 0
        ):
            return ()
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            NumericOperation,
            NumericSignature,
            ScalarType,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import (
            signature_capability_supported,
        )

        blocks = (context_len + self.compress_ratio - 1) // self.compress_ratio
        capacity = ((blocks + page_size - 1) // page_size) * page_size
        query_heads = ((self.index_n_heads + 7) // 8) * 8
        context = dict(
            tp_size=get_parallel().tp_size,
            sm100_supported=int(is_sm100_supported()),
            page_size=page_size,
            unpadded_query_heads=self.index_n_heads,
        )

        def admitted(operation, rows, scalar_type, variant, ctas_per_group=0):
            signature = NumericSignature(
                operation,
                rows,
                query_heads,
                self.index_kv_heads,
                self.index_head_dim,
                self.block_topk,
                self.compress_ratio,
                scalar_type,
                context_bound=capacity,
                variant=variant,
                ctas_per_group=ctas_per_group,
            )
            return (
                signature
                if signature_capability_supported(
                    signature,
                    spec=getattr(self, "_numeric_capability_spec", None),
                    **context,
                )
                else None
            )

        signatures = []
        split_select = self._split_select_source_enabled()
        verify_select = self._verify_select_source_enabled()
        # The source validated grouped4 bit-identical on SM107 only.
        if grouped4 is not None and torch.cuda.get_device_capability() != (10, 7):
            grouped4 = None
        for rows in widths:
            if envs.SGLANG_QSA_MK_SCORE.get():
                for operation, scalar_type, variant in (
                    (
                        NumericOperation.SPARSE_SCORE,
                        ScalarType.BF16,
                        3 if envs.SGLANG_QSA_MK_SCORE_SHORT32.get() else 0,
                    ),
                    (NumericOperation.SPARSE_SELECT, ScalarType.FP32, 0),
                ):
                    if (
                        signature := admitted(operation, rows, scalar_type, variant)
                    ) is not None:
                        signatures.append(signature)
            if self._split_score_preset(rows, capacity):
                if (
                    signature := admitted(
                        NumericOperation.SPARSE_SELECT, rows, ScalarType.FP32, 1
                    )
                ) is not None:
                    signatures.append(signature)
                for group_rows, variant in ((1, 5), (4, 4)):
                    if self._native_score_preset(rows, group_rows):
                        if (
                            signature := admitted(
                                NumericOperation.SPARSE_SCORE,
                                rows,
                                ScalarType.BF16,
                                variant,
                            )
                        ) is not None:
                            signatures.append(signature)
            if split_select:
                if (
                    signature := admitted(
                        NumericOperation.SPARSE_SELECT, rows, ScalarType.FP32, 1
                    )
                ) is not None:
                    signatures.append(signature)
            if verify_select:
                if (
                    signature := admitted(
                        NumericOperation.SPARSE_SELECT, rows, ScalarType.FP32, 2
                    )
                ) is not None:
                    signatures.append(signature)
            if (
                grouped4 is not None
                and rows % 4 == 0
                and 16 < rows // 4 <= envs.SGLANG_QSA_GROUPED_SCORE_MAX_REQUESTS.get()
                and rows <= 576
            ):
                if (
                    signature := admitted(
                        NumericOperation.SPARSE_SCORE,
                        rows,
                        ScalarType.BF16,
                        grouped4,
                        envs.SGLANG_QSA_GROUPED4_VERIFY_CHAIN_CTAS.get(),
                    )
                ) is not None:
                    signatures.append(signature)
        if envs.SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT.get():
            # One declaration for every prefill length; rows vary per call.
            signature = NumericSignature(
                NumericOperation.PREFILL_SELECT,
                0,
                self.index_n_heads,
                self.index_kv_heads,
                self.index_head_dim,
                self.block_topk,
                self.compress_ratio,
                ScalarType.BF16,
                context_bound=capacity,
                variant=self._prefill_select_variant(),
            )
            if signature_capability_supported(
                signature,
                spec=getattr(self, "_numeric_capability_spec", None),
                **context,
            ):
                signatures.append(signature)
        declared = tuple(dict.fromkeys(signatures))
        if declared:
            self._numeric_score_capacity = capacity
            self._numeric_prepared_sparse = frozenset(declared)
        return declared

    @staticmethod
    def _prefill_select_variant():
        """The source fast select's options: bits 0-3 = the top-k (0 sampled,
        1 one-pass), bit 4 = the tensor-core scorer, bits 8-19 / 20-30 = the wide
        logits budget (MiB) and the width it starts at (256-key units)."""
        topk = envs.SGLANG_QSA_PREFILL_FAST_SELECT_TOPK.get()
        if topk not in _FAST_SELECT_TOPK:
            raise ValueError(
                "SGLANG_QSA_PREFILL_FAST_SELECT_TOPK must be one of "
                f"{tuple(_FAST_SELECT_TOPK)}, got {topk!r}"
            )
        variant = _FAST_SELECT_TOPK[topk]
        if envs.SGLANG_QSA_PREFILL_SCORER_TC.get():
            variant |= 16
        wide_mib = envs.SGLANG_QSA_PREFILL_FAST_SELECT_WIDE_LOGITS_MB.get()
        if wide_mib:
            keys = envs.SGLANG_QSA_PREFILL_FAST_SELECT_WIDE_MIN_KEYS.get()
            if not (0 < wide_mib < 4096 and 0 < keys < 2048 * 256 and keys % 256 == 0):
                raise ValueError(
                    "SGLANG_QSA_PREFILL_FAST_SELECT_WIDE_LOGITS_MB must be in "
                    "[0, 4096) and SGLANG_QSA_PREFILL_FAST_SELECT_WIDE_MIN_KEYS a "
                    f"positive multiple of 256 below 524288; got {wide_mib}, {keys}"
                )
            variant |= wide_mib << 8 | keys // 256 << 20
        return variant

    @staticmethod
    def _grouped4_score_variant():
        """Source grouped4 verify scorer: 6 = chain, 7 = two-stage key pipeline."""
        if not (
            envs.SGLANG_ENABLE_QSA_GROUPED4_VERIFY.get()
            and envs.SGLANG_ENABLE_QSA_GROUPED4_VERIFY_CHAIN.get()
            and envs.SGLANG_QSA_GROUPED_SCORE.get()
        ):
            return None
        return {0: 6, 2: 7}.get(envs.SGLANG_QSA_GROUPED4_VERIFY_PIPELINE_STAGES.get())

    @staticmethod
    def _verify_select_source_enabled():
        """SGLANG_QSA_VERIFY_TOPK=sampled: the source's target-verify top-k (variant 2)."""
        mode = (envs.SGLANG_QSA_VERIFY_TOPK.get() or "stock").strip().lower()
        if mode not in ("stock", "sampled"):
            raise ValueError(
                f"SGLANG_QSA_VERIFY_TOPK must be stock or sampled, got {mode!r}"
            )
        return mode == "sampled" and get_parallel().tp_size == 1

    @staticmethod
    def _split_select_source_enabled():
        return (
            envs.SGLANG_QSA_SPLIT_TOPK.get()
            and get_parallel().tp_size == 1
            and is_sm100_supported()
            and envs.SGLANG_QSA_TOPK_THREADS.get() in (None, 1024)
            and envs.SGLANG_QSA_TOPK_CLUSTER.get() is None
        )

    def _numeric_verify_select(self, scores, lengths):
        """Target-verify rows through the declared variant-2 selector, or None."""
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            NumericOperation,
            NumericSignature,
            ScalarType,
            SparseSelectIO,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

        if not (
            scores.ndim == 2
            and scores.is_cuda
            and scores.dtype == torch.float32
            and scores.shape[0] in getattr(self, "_numeric_split_select_widths", ())
            and scores.shape[1] > 0
            and scores.shape[1] == getattr(self, "_numeric_score_capacity", 0)
            and scores.is_contiguous()
            and lengths.shape == (scores.shape[0],)
            and lengths.dtype in (torch.int32, torch.int64)
            and lengths.device == scores.device
            and lengths.is_contiguous()
        ):
            return None
        rows, capacity = scores.shape
        signature = NumericSignature(
            NumericOperation.SPARSE_SELECT,
            rows,
            ((self.index_n_heads + 7) // 8) * 8,
            self.index_kv_heads,
            self.index_head_dim,
            self.block_topk,
            self.compress_ratio,
            ScalarType.FP32,
            context_bound=capacity,
            variant=2,
        )
        if signature not in getattr(self, "_numeric_prepared_sparse", ()):
            return None
        run = numeric_operator(signature, prepared=True)
        if run is None:
            return None
        output = torch.empty(
            (rows, self.block_topk), dtype=torch.int32, device=scores.device
        )
        # Two int32 [rows] views for the provider: zero row starts and per-row flags.
        scratch = torch.empty(8 * rows, dtype=torch.uint8, device=scores.device)
        run(
            SparseSelectIO(
                scores, lengths.to(torch.int32), self.block_topk, output, None, scratch
            )
        )
        return output

    def _numeric_split_select(self, scores, lengths):
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            NumericOperation,
            NumericSignature,
            ScalarType,
            SparseSelectIO,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

        if not (
            scores.ndim == 2
            and scores.is_cuda
            and scores.dtype == torch.float32
            and scores.shape[0] in getattr(self, "_numeric_split_select_widths", ())
            and scores.shape[1] > 0
            and scores.shape[1] == getattr(self, "_numeric_score_capacity", 0)
            and scores.is_contiguous()
            and lengths.shape == (scores.shape[0],)
            and lengths.dtype in (torch.int32, torch.int64)
            and lengths.device == scores.device
            and lengths.is_contiguous()
            and self._split_select_source_enabled()
        ):
            return None
        rows, capacity = scores.shape
        signature = NumericSignature(
            NumericOperation.SPARSE_SELECT,
            rows,
            ((self.index_n_heads + 7) // 8) * 8,
            self.index_kv_heads,
            self.index_head_dim,
            self.block_topk,
            self.compress_ratio,
            ScalarType.FP32,
            context_bound=capacity,
            variant=1,
        )
        if signature not in getattr(self, "_numeric_prepared_sparse", ()):
            return None
        run = numeric_operator(signature, prepared=True)
        if run is None:
            return None
        output = torch.empty(
            (rows, self.block_topk), dtype=torch.int32, device=scores.device
        )
        run(
            SparseSelectIO(
                scores,
                lengths.to(torch.int32),
                self.block_topk,
                output,
                None,
                torch.empty(0, dtype=torch.uint8, device=scores.device),
            )
        )
        return output

    @staticmethod
    def _split_score_preset(rows, capacity):
        # With the source's default 1024 threads and adaptive 1..16 cluster,
        # every admitted width fits <=8 float4 iterations. Overrides stay stock.
        return (
            rows > 0
            and capacity > 0
            and envs.SGLANG_QSA_SPLIT_TOPK.get()
            and envs.SGLANG_QSA_TOPK_THREADS.get() is None
            and envs.SGLANG_QSA_TOPK_CLUSTER.get() is None
        )

    @staticmethod
    def _native_score_preset(rows, group_rows):
        if group_rows == 1:
            return not (
                envs.SGLANG_QSA_SCORER_V2.get() and envs.SGLANG_QSA_SCORER_V2_ROWS.get()
            )
        return (
            group_rows == 4
            and rows > 0
            and rows % 4 == 0
            and envs.SGLANG_QSA_GROUPED_SCORE.get()
            and envs.SGLANG_QSA_SCORER_V2.get()
            and envs.SGLANG_QSA_SCORER_V2_VER.get() == 7
            and envs.SGLANG_QSA_SCORER_V2_TARGET.get() == 4096
            and envs.SGLANG_QSA_SCORER_V2_C.get() is None
            and envs.SGLANG_QSA_SCORER_V2_NT.get() == 1
            and not envs.SGLANG_QSA_SCORER_V2_GUARD.get()
            and not envs.SGLANG_QSA_SCORER_V2_PEEL.get()
        )

    def _score_requests(self, q, cache, metadata, forward_batch, target, draft):
        # Preserve the source's single-request graph-only M4 entry policy.
        if (
            (envs.SGLANG_QSA_MK_SCORE.get() or envs.SGLANG_QSA_Q4_SCORE.get())
            and (target or draft)
            and (
                draft
                or get_parallel().tp_size != 1
                or not envs.SGLANG_QSA_GROUPED_SCORE.get()
            )
            and metadata.is_cuda_graph
            and q.is_cuda
            and is_sm100_supported()
            and q.ndim == 3
            and q.shape[1:]
            == (((self.index_n_heads + 7) // 8) * 8, self.index_head_dim)
            and q.dtype == torch.bfloat16
            and cache.dtype == torch.bfloat16
            and cache.shape[1:3]
            == (
                getattr(self, "_numeric_page_size", cache.shape[1]),
                self.index_kv_heads,
            )
            and q.shape[0]
            in {
                sig.rows
                for sig in getattr(self, "_numeric_prepared_sparse", ())
                if sig.operation.name == "SPARSE_SCORE" and sig.variant in (0, 3)
            }
            and forward_batch.req_pool_indices.numel() == 1
            and metadata.req_pool_indices is not None
            and metadata.req_pool_indices.numel() == q.shape[0]
        ):
            return metadata.req_pool_indices
        return None

    def _numeric_score_select(
        self, q, cache, table, lengths, capacity, requests, rows_per_group=1
    ):
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            NumericOperation,
            NumericSignature,
            ScalarType,
            SparseScoreIO,
            SparseSelectIO,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

        rows = q.shape[0]
        query_heads = ((self.index_n_heads + 7) // 8) * 8
        if not (
            q.is_cuda
            and is_sm100_supported()
            and not is_batch_invariant_mode_enabled()
            and q.shape == (rows, query_heads, self.index_head_dim)
            and q.dtype == torch.bfloat16
            and cache.ndim == 4
            and cache.shape[1:]
            == (
                getattr(self, "_numeric_page_size", cache.shape[1]),
                self.index_kv_heads,
                self.index_head_dim,
            )
            and cache.dtype == torch.bfloat16
            and table.shape == (rows, (capacity + cache.shape[1] - 1) // cache.shape[1])
            and table.dtype in (torch.int32, torch.int64)
            and lengths.shape == (rows,)
            and lengths.dtype in (torch.int32, torch.int64)
            and q.device == cache.device == table.device == lengths.device
            and capacity == getattr(self, "_numeric_score_capacity", 0)
            and capacity > 0
        ):
            return None
        tp1 = get_parallel().tp_size == 1
        split = tp1 and self._split_score_preset(rows, capacity)
        # A nondefault split policy may still take precedence in the source.
        if tp1 and envs.SGLANG_QSA_SPLIT_TOPK.get() and not split:
            return None
        if requests is not None:
            if not (
                envs.SGLANG_QSA_MK_SCORE.get()
                and requests.shape == (rows,)
                and requests.dtype in (torch.int32, torch.int64)
                and requests.device == q.device
            ):
                return None  # unsupported non-MK Q4/N16 route is not per-row
            score_variant = 3 if envs.SGLANG_QSA_MK_SCORE_SHORT32.get() else 0
        else:
            if not split:
                return None
            if not self._native_score_preset(rows, rows_per_group):
                return None
            score_variant = 4 if rows_per_group == 4 else 5
        if not split and requests is None:
            return None
        score_sig = NumericSignature(
            NumericOperation.SPARSE_SCORE,
            rows,
            query_heads,
            self.index_kv_heads,
            self.index_head_dim,
            self.block_topk,
            self.compress_ratio,
            ScalarType.BF16,
            context_bound=capacity,
            variant=score_variant,
        )
        select_sig = NumericSignature(
            NumericOperation.SPARSE_SELECT,
            rows,
            query_heads,
            self.index_kv_heads,
            self.index_head_dim,
            self.block_topk,
            self.compress_ratio,
            ScalarType.FP32,
            context_bound=capacity,
            variant=1 if split else 0,
        )
        declared = getattr(self, "_numeric_prepared_sparse", ())
        if score_sig not in declared or select_sig not in declared:
            return None
        score = numeric_operator(score_sig, prepared=True)
        select = numeric_operator(select_sig, prepared=True)
        if score is None or select is None:
            return None
        table = table.to(torch.int32).contiguous()
        lengths = lengths.to(torch.int32).contiguous()
        requests = requests.contiguous() if requests is not None else None
        scores = torch.empty((rows, capacity), dtype=torch.float32, device=q.device)
        indices = torch.empty(
            (rows, self.block_topk), dtype=torch.int32, device=q.device
        )
        scratch = torch.empty(0, dtype=torch.uint8, device=q.device)
        score(
            SparseScoreIO(
                q.contiguous(),
                cache.contiguous(),
                table,
                lengths,
                requests,
                None,
                cache.shape[1],
                self.compress_ratio,
                math.sqrt(self.index_head_dim),
                scores,
                scratch,
            )
        )
        select(SparseSelectIO(scores, lengths, self.block_topk, indices, None, scratch))
        return indices

    def _numeric_prefill_select(
        self, q, keys, starts, ends, positions, lengths, output, width
    ):
        """Fill ``output`` as the stock relative prefill path does, or return False."""
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            NumericOperation,
            NumericSignature,
            PrefillSelectIO,
            ScalarType,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator
        from sglang.srt.kernel_plugins.nvidia.pool_attention import pool_device_sms

        capacity = getattr(self, "_numeric_score_capacity", 0)
        if not (
            envs.SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT.get()
            and envs.SGLANG_OPT_QSA_PREFILL_SCALE_FOLD.get()
            and type(width) is int
            and 0 < width <= capacity
            and q.dtype == torch.bfloat16
            and q.shape[1:] == (self.index_n_heads, self.index_head_dim)
            and keys.ndim == 3
            and keys.shape[0] > 0
            and keys.shape[1:] == (self.index_kv_heads, self.index_head_dim)
            and keys.dtype == q.dtype
            and not is_batch_invariant_mode_enabled()
            # Captured prefill keeps stock; per-call workspace is eager only.
            and not torch.cuda.is_current_stream_capturing()
        ):
            return False
        signature = NumericSignature(
            NumericOperation.PREFILL_SELECT,
            0,
            self.index_n_heads,
            self.index_kv_heads,
            self.index_head_dim,
            self.block_topk,
            self.compress_ratio,
            ScalarType.BF16,
            context_bound=capacity,
            variant=self._prefill_select_variant(),
        )
        if signature not in getattr(self, "_numeric_prepared_sparse", ()):
            return False
        budget = _fast_select_logits_budget(width, signature.variant)
        run = numeric_operator(signature, prepared=True)
        if run is None:
            return False
        rows = q.shape[0]
        sms = pool_device_sms(q.device)
        chunk = _fast_select_chunk_rows(rows, width, sms, budget)
        padded = (chunk + 31) // 32 * 32
        starts, ends, positions, lengths = (
            t.to(device=q.device, dtype=torch.int32).contiguous()
            for t in (starts, ends, positions, lengths)
        )
        i32 = dict(device=q.device, dtype=torch.int32)
        run(
            PrefillSelectIO(
                q.contiguous(),
                keys.contiguous(),
                starts,
                ends,
                positions,
                lengths,
                output,
                torch.empty((padded, *q.shape[1:]), device=q.device, dtype=q.dtype),
                torch.empty(padded, **i32),
                torch.empty(padded, **i32),
                ends - starts,
                torch.zeros(chunk, **i32),
                torch.empty((rows, self.block_topk), **i32),
                torch.empty((padded, width), device=q.device, dtype=torch.float32),
                torch.empty(4096, **i32),
                chunk,
                sms,
            )
        )
        return True

    def _numeric_grouped4_score(self, q, cache, table, lengths, capacity):
        """Scores of one verify or draft-extend grouped4 launch, or None for stock."""
        from sglang.kernels.plugin_kernels.nvidia.numeric import (
            NumericOperation,
            NumericSignature,
            ScalarType,
            SparseScoreIO,
        )
        from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

        variant = self._grouped4_score_variant()
        rows = q.shape[0]
        query_heads = ((self.index_n_heads + 7) // 8) * 8
        if not (
            variant is not None
            and q.is_cuda
            and q.shape == (rows, query_heads, self.index_head_dim)
            and q.dtype == torch.bfloat16
            and cache.ndim == 4
            and cache.shape[1:]
            == (
                getattr(self, "_numeric_page_size", cache.shape[1]),
                self.index_kv_heads,
                self.index_head_dim,
            )
            and cache.dtype == torch.bfloat16
            and table.shape == (rows, (capacity + cache.shape[1] - 1) // cache.shape[1])
            and table.dtype in (torch.int32, torch.int64)
            and lengths.shape == (rows,)
            and lengths.dtype in (torch.int32, torch.int64)
            and q.device == cache.device == table.device == lengths.device
            and capacity == getattr(self, "_numeric_score_capacity", 0)
            and capacity > 0
        ):
            return None
        signature = NumericSignature(
            NumericOperation.SPARSE_SCORE,
            rows,
            query_heads,
            self.index_kv_heads,
            self.index_head_dim,
            self.block_topk,
            self.compress_ratio,
            ScalarType.BF16,
            context_bound=capacity,
            variant=variant,
            ctas_per_group=envs.SGLANG_QSA_GROUPED4_VERIFY_CHAIN_CTAS.get(),
        )
        if signature not in getattr(self, "_numeric_prepared_sparse", ()):
            return None
        run = numeric_operator(signature, prepared=True)
        if run is None:
            return None
        scores = torch.empty((rows, capacity), dtype=torch.float32, device=q.device)
        run(
            SparseScoreIO(
                q.contiguous(),
                cache.contiguous(),
                table.to(torch.int32).contiguous(),
                lengths.to(torch.int32).contiguous(),
                None,
                None,
                cache.shape[1],
                self.compress_ratio,
                math.sqrt(self.index_head_dim),
                scores,
                torch.empty(0, dtype=torch.uint8, device=q.device),
            )
        )
        return scores

    @staticmethod
    def _validate_config(config) -> None:
        names = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        missing = [name for name in names if getattr(config, name, None) is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")
        values = {name: int(getattr(config, name)) for name in names}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}")
        if values["indexer_compress_ratio"] < 2:
            # DP token-padding rows carry logical length 1, which must never
            # reach a compression boundary; ratio >= 2 guarantees that.
            raise ValueError(
                "QSA requires indexer_compress_ratio >= 2, got "
                f"{values['indexer_compress_ratio']}"
            )
        if values["indexer_kv_heads"] != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if values["indexer_budget"] % values["indexer_compress_ratio"] != 0:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        block_topk = values["indexer_budget"] // values["indexer_compress_ratio"]
        if block_topk not in (512, 2048):
            raise ValueError(
                "fast_topk_v2 requires indexer_budget / indexer_compress_ratio "
                f"to be 512 or 2048, got {block_topk}"
            )

    def _use_fused_prep(self, tensor: torch.Tensor) -> bool:
        """Whether the fused indexer-prep kernels support this configuration."""
        return (
            tensor.is_cuda
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and self.index_head_dim in (64, 128, 256)
            and self.rotary_emb.rotary_dim % 2 == 0
            and not getattr(self.rotary_emb, "mrope_interleaved_glm", False)
            and len(getattr(self.rotary_emb, "mrope_section", None) or ()) in (0, 3)
            and getattr(self.rotary_emb, "cos_sin_cache", None) is not None
            and self.rotary_emb.cos_sin_cache.is_cuda
            and self.rotary_emb.cos_sin_cache.dtype == torch.float32
        )

    def _rope_axis_map(self, device) -> torch.Tensor:
        """axis_map[i] is the MRoPE position axis whose cos/sin rotary pair i reads."""
        cache = self._rope_axis_map_cache
        if cache is not None and cache.device == device:
            return cache
        half = self.rotary_emb.rotary_dim // 2
        section = getattr(self.rotary_emb, "mrope_section", None) or None
        axis_map = torch.zeros(half, dtype=torch.int32)
        if section is not None:
            s0, s1, s2 = (int(v) for v in section)
            if getattr(self.rotary_emb, "mrope_interleaved", False):
                pair = torch.arange(half, dtype=torch.int32)
                axis_map[((pair % 3) == 1) & (pair < s1 * 3)] = 1
                axis_map[((pair % 3) == 2) & (pair < s2 * 3)] = 2
            else:
                axis_map[s0 : s0 + s1] = 1
                axis_map[s0 + s1 :] = 2
        self._rope_axis_map_cache = axis_map.to(device)
        return self._rope_axis_map_cache

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        pool=None,
        cache_loc: torch.Tensor | None = None,
        q_heads_padded: int | None = None,
        projected_qk: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        if projected_qk is None:
            qk, _ = self.index_qk_proj(hidden_states)
        else:
            qk = projected_qk
        token_k = qk[:, self.index_n_heads * self.index_head_dim :].reshape(
            -1, self.index_kv_heads, self.index_head_dim
        )
        if (
            pool is not None
            and cache_loc is not None
            and qk.shape[0] > 0
            and self._use_fused_prep(qk)
        ):
            from sglang.kernels.ops.attention.qsa_indexer import (
                qsa_index_q_norm_rope_store,
            )

            key_state_buffer = pool.get_qsa_key_state_buffer(self.layer_id)
            q = qsa_index_q_norm_rope_store(
                qk,
                positions.long(),
                self.rotary_emb.cos_sin_cache,
                self._rope_axis_map(qk.device),
                self.q_layernorm.weight.data,
                cache_loc[: qk.shape[0]].long(),
                key_state_buffer.view(key_state_buffer.shape[0], -1),
                pool.qsa_rope_position_buffer,
                self.index_n_heads,
                self.rotary_emb.rotary_dim,
                self.q_layernorm.variance_epsilon,
                self.rotary_emb.is_neox_style,
                q_heads_padded=q_heads_padded,
                out_dtype=pool.qsa_compressed_dtype,
            )
            return q, token_k, True
        q_raw = qk[:, : self.index_n_heads * self.index_head_dim]
        q = self.q_layernorm(q_raw.reshape(-1, self.index_head_dim)).reshape(
            -1, self.index_n_heads, self.index_head_dim
        )
        q = self.apply_rope(positions, q)
        if pool is not None:
            q = q.to(pool.qsa_compressed_dtype)
        return q, token_k, False

    def normalize_compressed_keys(
        self, compressed_keys: torch.Tensor, block_positions: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.k_layernorm(
            compressed_keys.reshape(-1, self.index_head_dim)
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)
        return self.apply_rope(block_positions, normalized)

    def _use_fused_compress(self, pool) -> bool:
        return getattr(
            pool, "qsa_rope_position_buffer", None
        ) is not None and self._use_fused_prep(
            pool.get_qsa_key_state_buffer(self.layer_id)
        )

    def _fused_compress_store(
        self,
        pool,
        group_locs: torch.Tensor,
        write_locs: torch.Tensor,
        source_keys: torch.Tensor | None = None,
        source_rope: torch.Tensor | None = None,
    ) -> None:
        """Fused mean -> gemma norm -> MRoPE -> compressed-cache store;
        a None source_keys/source_rope reads the members from the pending ring."""
        from sglang.kernels.ops.attention.qsa_indexer import (
            qsa_index_k_compress_store,
        )

        if source_keys is None:
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
        if source_rope is None:
            source_rope = pool.qsa_rope_position_buffer
        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)
        qsa_index_k_compress_store(
            source_keys.reshape(source_keys.shape[0], -1)
            .contiguous()
            .to(pool.index_state_dtype),
            group_locs.to(torch.int32),
            source_rope,
            self.rotary_emb.cos_sin_cache,
            self._rope_axis_map(source_keys.device),
            self.k_layernorm.weight.data,
            write_locs.to(torch.int32),
            compressed_buffer.view(compressed_buffer.shape[0], -1),
            self.compress_ratio,
            self.rotary_emb.rotary_dim,
            self.k_layernorm.variance_epsilon,
            self.rotary_emb.is_neox_style,
        )

    def _pending_ring_slots(
        self, metadata, logical_positions: torch.Tensor, is_extend: bool
    ) -> torch.Tensor:
        return build_pending_ring_slots(
            token_to_batch_idx=metadata.token_to_batch_idx,
            req_pool_indices=metadata.req_pool_indices,
            sequence_lengths=metadata.sequence_lengths,
            logical_positions=logical_positions,
            compress_ratio=self.compress_ratio,
            is_extend=is_extend,
        )

    def _group_ring_slots(
        self, metadata, group_end_positions: torch.Tensor, sequence_ids: torch.Tensor
    ) -> torch.Tensor:
        return build_group_ring_slots(
            req_pool_indices=metadata.req_pool_indices,
            group_end_positions=group_end_positions,
            sequence_ids=sequence_ids,
            compress_ratio=self.compress_ratio,
        )

    def update_key_state_and_compress(
        self,
        token_k: torch.Tensor,
        logical_positions: torch.Tensor,
        rope_positions: torch.Tensor,
        metadata,
        state_slots: torch.Tensor | None = None,
        state_stored: bool = False,
    ) -> None:
        """Store the pending-group ring and compress each completed group."""

        pool = metadata.token_to_kv_pool
        is_extend = metadata.compress_member_rows is not None
        if not state_stored:
            if state_slots is None:
                state_slots = self._pending_ring_slots(
                    metadata, logical_positions, is_extend
                )
            pool.set_qsa_key_state_buffer(
                self.layer_id, state_slots[: token_k.shape[0]], token_k
            )
            pool.set_qsa_rope_position_buffer(
                state_slots[: token_k.shape[0]], rope_positions
            )

        if metadata.is_cuda_graph:
            self._compress_decode_cuda_graph(metadata)
            return

        if metadata.write_locs is None:
            raise RuntimeError(
                "QSA metadata is missing the precomputed write plan; the "
                "sparse-attention backend derives it from the batch lengths"
            )
        if metadata.write_locs.numel() == 0:
            return
        group_end_positions = metadata.compress_group_positions.long()
        compressed_locs = metadata.write_locs
        if is_extend:
            # Extend chunks are group-aligned; each planned group lies in this forward,
            # so read its members from the packed chunk tensors.
            member_rows = metadata.compress_member_rows.long()
            group_locs = member_rows[:, None] + torch.arange(
                self.compress_ratio, device=member_rows.device, dtype=torch.long
            )
            source_keys = token_k
            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)
            source_rope = metadata.extend_rope_matrix
            if source_rope is None:
                source_rope = build_rope_position_matrix(
                    rope_positions, token_k.shape[0]
                )
        else:
            # Paged eager rows (speculative fallback) complete at most one
            # group each; its members are exactly the pending ring window.
            group_locs = metadata.compress_group_ring_locs
            if group_locs is None:
                group_locs = self._group_ring_slots(
                    metadata,
                    group_end_positions,
                    metadata.compress_sequence_ids.long(),
                )
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
            source_rope = pool.qsa_rope_position_buffer
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                compressed_locs,
                source_keys=source_keys,
                source_rope=source_rope,
            )
            return
        key_groups = source_keys[group_locs]
        pooled = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            source_rope[group_locs[:, 0]]
        )
        normalized = self.normalize_compressed_keys(pooled, compressed_rope_positions)
        pool.set_qsa_compressed_k_buffer(self.layer_id, compressed_locs, normalized)

    def _compress_decode_cuda_graph(self, metadata) -> None:
        """Fixed-shape graph-replay compression; non-boundary rows write slot 0."""

        if metadata.graph_write_locs is None or metadata.graph_ring_group_locs is None:
            raise RuntimeError("QSA CUDA graph compression metadata is incomplete")
        pool = metadata.token_to_kv_pool
        group_locs = metadata.graph_ring_group_locs
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                metadata.graph_write_locs,
            )
            return
        key_groups = pool.get_qsa_key_state_buffer(self.layer_id)[group_locs]
        compressed = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            pool.qsa_rope_position_buffer[group_locs[:, 0]]
        )
        compressed = self.normalize_compressed_keys(
            compressed, compressed_rope_positions
        )
        pool.set_qsa_compressed_k_buffer(
            self.layer_id, metadata.graph_write_locs, compressed.contiguous()
        )

    def _rope_from_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """[n, 3] slot coordinates -> the layout apply_rope expects."""
        positions = positions.transpose(0, 1)
        if not getattr(self.rotary_emb, "mrope_section", None):
            return positions[0]
        return positions

    def apply_rope(
        self,
        positions: torch.Tensor,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        positions = positions.long()
        num_positions = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if num_positions != tensor.shape[0]:
            raise ValueError("QSA RoPE positions must match the token dimension")
        # position_cos/position_sin repeat cos/sin to the full rotary width;
        # apply_rotary_emb consumes one half.
        self.rotary_emb.get_cos_sin_with_position(positions)
        rotary_dim = self.rotary_emb.rotary_dim
        half_rotary_dim = rotary_dim // 2
        cos = self.rotary_emb.position_cos.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        sin = self.rotary_emb.position_sin.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        rotated = apply_rotary_emb(
            tensor[..., :rotary_dim],
            cos,
            sin,
            self.rotary_emb.is_neox_style,
        )
        return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)

    def select_prefill_tokens(
        self,
        q: torch.Tensor,
        compressed_keys: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths_for_rows: torch.Tensor,
        max_row_keys: Optional[int] = None,
    ) -> torch.Tensor:
        """Top-k token selection for packed prefill rows.

        ``max_row_keys`` (host int, the longest request's compressed-block count)
        switches scoring to request-relative logits: [rows, max_row_keys] instead
        of [rows, all compressed keys of the batch]. The latter caps a chunk at
        a few dozen rows once a batch holds many long prefixes, so the scoring
        kernel ran as thousands of one-CTA launches. Selection is unchanged: the
        top-k reads only each row's own section and returns indices relative to
        the row's start either way."""
        rows = q.shape[0]
        output = (
            _take_prefill_topk_output(
                rows, self.token_topk + self.compress_ratio - 1, q.device
            )
            if _PREFILL_TOPK_OUT[0] is not None
            else None
        )
        if output is None:
            output = torch.empty(
                (rows, self.token_topk + self.compress_ratio - 1),
                dtype=torch.int32,
                device=q.device,
            )
        if rows == 0:
            return output

        relative = (
            max_row_keys is not None and max_row_keys > 0 and q.is_cuda and HAS_TILELANG
        )
        if relative and self._numeric_prefill_select(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            query_positions,
            sequence_lengths_for_rows,
            output,
            max_row_keys,
        ):
            return output
        row_chunk_size = _qsa_prefill_row_chunk_size(
            rows,
            max_row_keys if relative else compressed_keys.shape[0],
            q.shape[1],
        )
        for row_start in range(0, rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, rows)
            chunk_slice = slice(row_start, row_end)
            if compressed_keys.shape[0] == 0:
                block_indices = torch.full(
                    (row_end - row_start, self.block_topk),
                    -1,
                    dtype=torch.int32,
                    device=q.device,
                )
                logits = None
            elif relative:
                chunk_starts = row_starts[chunk_slice]
                chunk_lengths = row_ends[chunk_slice] - chunk_starts
                logits = qsa_mqa_prefill_relative(
                    q[chunk_slice],
                    compressed_keys,
                    chunk_starts,
                    row_ends[chunk_slice],
                    max_row_keys,
                )
                block_indices = qsa_fast_topk(
                    logits,
                    torch.zeros_like(chunk_starts),
                    chunk_lengths,
                    topk=self.block_topk,
                )
            else:
                logits = qsa_mqa_prefill(
                    q[chunk_slice],
                    compressed_keys,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                )
                block_indices = qsa_fast_topk(
                    logits,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                    topk=self.block_topk,
                )
            selected = expand_qsa_block_indices(
                block_indices,
                query_positions[chunk_slice],
                sequence_lengths_for_rows[chunk_slice],
                compress_ratio=self.compress_ratio,
                token_topk=self.token_topk,
            )
            output[chunk_slice].copy_(selected)
            del logits, block_indices, selected
        return output

    def select_decode_tokens(
        self,
        q: torch.Tensor,
        compressed_cache: torch.Tensor,
        compressed_page_table: torch.Tensor,
        compressed_lengths: torch.Tensor,
        max_model_len: int,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        defer_expansion: bool = False,
        score_q4_requests: torch.Tensor | None = None,
        rows_per_group: int = 1,
        grouped4_only: bool = False,
    ) -> torch.Tensor:
        verify_rows_per_group = rows_per_group
        # grouped4_only: rows outside the grouped4 scorer are scored per row.
        if not (
            rows_per_group > 1
            and not grouped4_only
            and get_parallel().tp_size == 1
            and is_sm100_supported()
            and q.shape[0] % rows_per_group == 0
            and 1 <= q.shape[0] // rows_per_group <= 16
            and envs.SGLANG_QSA_GROUPED_SCORE.get()
        ):
            rows_per_group = 1
        selected = (
            self._numeric_score_select(
                q,
                compressed_cache,
                compressed_page_table,
                compressed_lengths,
                max_model_len,
                score_q4_requests,
                rows_per_group,
            )
            if score_q4_requests is not None
            or getattr(self, "_numeric_score_capacity", 0)
            else None
        )
        if selected is not None:
            if defer_expansion:
                return selected
            return expand_qsa_block_indices(
                selected,
                query_positions,
                sequence_lengths,
                compress_ratio=self.compress_ratio,
                token_topk=self.token_topk,
            )
        # fast_topk (k = 512) writes 0..length-1 for rows that fit k without reading
        # their scores, and never reads past a row's length.
        use_fast_topk = (
            q.is_cuda
            and self.block_topk == 512
            and not is_batch_invariant_mode_enabled()
        )
        logits = (
            self._numeric_grouped4_score(
                q,
                compressed_cache,
                compressed_page_table,
                compressed_lengths,
                max_model_len,
            )
            if use_fast_topk and verify_rows_per_group == 4
            else None
        )
        if logits is None:
            logits = qsa_mqa_decode(
                q,
                compressed_cache,
                compressed_page_table,
                compressed_lengths,
                max_model_len,
                min_scored_len=self.block_topk if use_fast_topk else 0,
            )
        if use_fast_topk:
            # Decode rows start at zero, so compressed lengths double as row lengths;
            # skip the generic zero-fill + subtract.
            from sglang.kernels.ops.attention.fast_topk import fast_topk

            block_indices = self._numeric_split_select(logits, compressed_lengths)
            # SGLANG_ENABLE_QSA_DRAFT_EXTEND_UNION takes it for the grouped4 draft
            # extend too, as the source does.
            if (
                block_indices is None
                and verify_rows_per_group > 1
                and (
                    not grouped4_only or envs.SGLANG_ENABLE_QSA_DRAFT_EXTEND_UNION.get()
                )
            ):
                block_indices = self._numeric_verify_select(logits, compressed_lengths)
            if block_indices is None:
                block_indices = fast_topk(
                    logits,
                    compressed_lengths.to(torch.int32),
                    topk=self.block_topk,
                    row_starts=None,
                )
        else:
            row_starts = torch.zeros_like(compressed_lengths, dtype=torch.int32)
            block_indices = qsa_fast_topk(
                logits, row_starts, compressed_lengths, topk=self.block_topk
            )
        if defer_expansion:
            return block_indices
        return expand_qsa_block_indices(
            block_indices,
            query_positions,
            sequence_lengths,
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
        )

    def forward_cuda_cp(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        global_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill CP: score only this rank's zigzag query rows.

        ``hidden_states``/``positions`` are the local (padded) rows and their
        RoPE positions. Index q/k are projected for the local rows only; the raw
        keys are all-gathered into global token order so the pending-group ring
        store and the block compression run on the batch's global write plan
        exactly as without CP (every rank ends up with the complete compressed-K
        cache). Selection then scores the local rows against all compressed
        blocks of their request, position-driven, so the output rows line up
        with the local query rows the sparse attention consumes.
        """
        strategy = get_cp_strategy()
        if strategy is None or strategy.kind != ContextParallelStrategyKind.ZIGZAG:
            strategy_name = strategy.name if strategy is not None else "none"
            raise NotImplementedError(
                "QSA prefill CP only supports the zigzag strategy; "
                f"got {strategy_name}."
            )
        meta = forward_batch.attn_cp_metadata
        num_local = int(meta.total_q_prev_tokens + meta.total_q_next_tokens)
        global_logical = getattr(forward_batch, "positions", None)
        global_logical = (
            global_logical[0] if global_logical.ndim == 2 else global_logical
        ).flatten()
        num_global = indexer_metadata.get_token_to_batch_idx().numel()
        global_logical = global_logical[:num_global]
        local_logical = cp_shard_position_ids(global_logical, forward_batch)[:num_local]
        local_sequence_ids = cp_shard_hidden_states(
            indexer_metadata.get_token_to_batch_idx(), forward_batch
        )[:num_local]
        hidden_states = hidden_states[:num_local]
        positions = (
            positions[:, :num_local] if positions.ndim == 2 else positions[:num_local]
        )
        # Use the compressed-cache dtype for Q. Leaving cache_loc unset defers
        # the ring store until raw local K has been gathered into global order.
        q, token_k, _ = self.project_qk(
            hidden_states,
            positions,
            pool=indexer_metadata.token_to_kv_pool,
        )
        # Raw keys of all new tokens in global packed order: the ring store
        # and compression use the global write plan unchanged. Cached prefix
        # groups already reside in the compressed-K pool on every rank.
        token_k_local = token_k.reshape(num_local, -1).contiguous()
        token_k_full = cp_materialize_global_token_order(
            token_k_local, forward_batch
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)
        global_rope = (
            global_rope_positions[:, :num_global]
            if global_rope_positions.ndim == 2
            else global_rope_positions[:num_global]
        )
        self.update_key_state_and_compress(
            token_k_full,
            global_logical,
            global_rope,
            indexer_metadata,
            state_slots=indexer_metadata.pending_ring_slots,
            state_stored=False,
        )
        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(
                self.layer_id, local_logical, query_sequence_ids=local_sequence_ids
            )
        )
        row_sequence_lengths = sequence_lengths.index_select(
            0, local_sequence_ids.long()
        )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            local_logical,
            row_sequence_lengths,
            max_row_keys=indexer_metadata.prefill_max_row_keys(
                getattr(forward_batch, "seq_lens_cpu", None)
            ),
        )

    @staticmethod
    def _decode_row_groups(
        forward_batch, num_valid_tokens, is_target_verify, is_draft_extend
    ):
        """(rows_per_group, grouped4_only) of a TP1 verify or draft-extend window."""
        if get_parallel().tp_size != 1:
            return 1, False
        if is_target_verify:
            width = int(getattr(forward_batch.spec_info, "draft_token_num", 1) or 1)
            if width > 1 and num_valid_tokens % width == 0:
                return width, False
        if is_draft_extend and envs.SGLANG_ENABLE_QSA_GROUPED4_DRAFT_EXTEND.get():
            # Spec-v2 draft extend packs one equal window per request in position
            # order (graph_metadata mode 2), so four rows group like verify.
            requests = forward_batch.req_pool_indices.numel()
            if requests > 0 and num_valid_tokens == 4 * requests:
                return 4, True
        return 1, False

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        cp_global_rope_positions: torch.Tensor | None = None,
        projected_qk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Portable orchestration shared by different platforms.
        Fast paths are gated per platforms inside kernel calls.
        """
        forward_mode = forward_batch.forward_mode
        is_target_verify = getattr(forward_mode, "is_target_verify", lambda: False)()
        is_draft_extend = getattr(forward_mode, "is_draft_extend_v2", lambda: False)()
        if cp_global_rope_positions is not None and not (
            forward_mode.is_decode() or is_target_verify or is_draft_extend
        ):
            return self.forward_cuda_cp(
                hidden_states,
                positions,
                forward_batch,
                indexer_metadata,
                cp_global_rope_positions,
            )
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            # EAGLE/MTP may advance the model's RoPE coordinate independently
            # from the physical paged-KV position.  Compression and sparse
            # selection must use the latter; otherwise a draft step can index
            # token_slot_table[:, seq_len] one past the valid range.
            logical_positions = indexer_metadata.decode_logical_positions
            if logical_positions is None:
                logical_positions = indexer_metadata.get_seqlens_expanded() - 1
        else:
            logical_positions = getattr(forward_batch, "positions", None)
            if logical_positions is None:
                logical_positions = positions[0] if positions.ndim == 2 else positions
            logical_positions = logical_positions.flatten()
        # DP MAX_LEN padding adds token rows without assigning them to a
        # request. token_to_batch_idx is the source of truth for semantic rows.
        num_valid_tokens = indexer_metadata.get_token_to_batch_idx().numel()
        if logical_positions.numel() < num_valid_tokens:
            raise ValueError(
                "QSA logical positions are shorter than the request mapping: "
                f"positions={logical_positions.numel()}, mapping={num_valid_tokens}"
            )
        if hidden_states.shape[0] < num_valid_tokens:
            raise ValueError(
                "QSA hidden states are shorter than the request mapping: "
                f"hidden={hidden_states.shape[0]}, mapping={num_valid_tokens}"
            )
        position_tokens = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if position_tokens < num_valid_tokens:
            raise ValueError(
                "QSA RoPE positions are shorter than the request mapping: "
                f"positions={position_tokens}, mapping={num_valid_tokens}"
            )

        logical_positions = logical_positions[:num_valid_tokens]
        hidden_states = hidden_states[:num_valid_tokens]
        positions = (
            positions[:, :num_valid_tokens]
            if positions.ndim == 2
            else positions[:num_valid_tokens]
        )
        state_slots = indexer_metadata.pending_ring_slots
        if state_slots is None:
            state_slots = self._pending_ring_slots(
                indexer_metadata,
                logical_positions,
                indexer_metadata.compress_member_rows is not None,
            )
        is_decode_like = forward_mode.is_decode() or is_target_verify or is_draft_extend
        q = None
        if is_decode_like and indexer_metadata.is_cuda_graph:
            q, projected_qk = self._try_prologue_fold(
                hidden_states, positions, indexer_metadata, state_slots, projected_qk
            )
        if q is None:
            q, token_k, state_stored = self.project_qk(
                hidden_states,
                positions,
                pool=indexer_metadata.token_to_kv_pool,
                cache_loc=state_slots,
                q_heads_padded=(
                    # The tilelang decode MQA kernel needs query heads in multiples of 8.
                    ((self.index_n_heads + 7) // 8) * 8 if is_decode_like else None
                ),
                projected_qk=projected_qk,
            )
            self.update_key_state_and_compress(
                token_k,
                logical_positions,
                positions,
                indexer_metadata,
                state_slots=state_slots,
                state_stored=state_stored,
            )
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            compressed_cache, page_table, compressed_lengths, max_model_len = (
                indexer_metadata.get_decode_mqa_inputs(self.layer_id)
            )
            rows_per_group, grouped4_only = self._decode_row_groups(
                forward_batch, num_valid_tokens, is_target_verify, is_draft_extend
            )
            return self.select_decode_tokens(
                q,
                compressed_cache,
                page_table,
                compressed_lengths,
                max_model_len,
                logical_positions,
                indexer_metadata.get_seqlens_int32(),
                defer_expansion=q.is_cuda and indexer_metadata.defer_block_expansion,
                score_q4_requests=self._score_requests(
                    q,
                    compressed_cache,
                    indexer_metadata,
                    forward_batch,
                    is_target_verify,
                    is_draft_extend,
                )
                if (envs.SGLANG_QSA_MK_SCORE.get() or envs.SGLANG_QSA_Q4_SCORE.get())
                else None,
                rows_per_group=rows_per_group,
                grouped4_only=grouped4_only,
            )

        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(self.layer_id, logical_positions)
        )
        query_sequence_ids = indexer_metadata.get_token_to_batch_idx()
        row_sequence_lengths = sequence_lengths.index_select(
            0, query_sequence_ids.long()
        )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            logical_positions,
            row_sequence_lengths,
            max_row_keys=indexer_metadata.prefill_max_row_keys(
                getattr(forward_batch, "seq_lens_cpu", None)
            ),
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
        cp_global_rope_positions: torch.Tensor | None = None,
        projected_qk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states,
            positions,
            forward_batch,
            indexer_metadata,
            cp_global_rope_positions,
            projected_qk,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
    ) -> torch.Tensor:
        return self._forward_impl(
            hidden_states, positions, forward_batch, indexer_metadata
        )


__all__ = [
    "QSAIndexer",
]
