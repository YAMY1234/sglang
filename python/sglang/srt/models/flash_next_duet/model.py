"""In-tree Flash-Next DUET serving for BF16 and NVFP4 bases.

The stock target layers, cache writes, tensor names and emitter arithmetic are
preserved from the qualified TwinStar serving path. A canonical release selects
this adapter; diagnostics remain optional modules in the TwinStar package.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import logging
import os
import time
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.duet.options import resolve_emitter_precision
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.qsa.glue import get_qsa_indexer_metadata
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models import qwen3_5 as _q35
from sglang.srt.models import qwen4_exp as _stock
from sglang.srt.runtime_context import get_parallel

from .diagnostics import optional

logger = logging.getLogger(__name__)

_E_PREFIX = "model.emitters."
_B_PREFIX = "model.bridge."


def _optional_prefill_graph():
    name = "sglang.srt.models.qwen4_exp_prefill_graph"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        # The qualified PD tail fork predates whole-model prefill graphs.
        return None


def _dev(values, dtype, device):
    """Host values -> device tensor. TWINSTAR_PREFILL_ASYNC_H2D=1 (#287 (e)) stages them in pinned memory and copies
    without blocking the host on the stream; the values are identical."""
    from .config import runtime_controls

    if runtime_controls()["async_h2d"] and torch.device(device).type == "cuda":
        host = (
            values
            if isinstance(values, torch.Tensor)
            else torch.tensor(values, dtype=dtype)
        )
        return host.to(dtype).pin_memory().to(device, non_blocking=True)
    if isinstance(values, torch.Tensor):
        return values.to(device, non_blocking=True)
    return torch.tensor(values, dtype=dtype, device=device)


def _is_attn_layer(tcfg, l: int) -> bool:
    return tcfg.layers_block_type[l] == "attention"


# ----------------------------------------------------------------------------- emitters
class _Emitter:
    """Emitter for a skipped D layer l.  Training-free: aliases the stock target layer's modules (read gate, GDN input
    path or QSA k/v + indexer path).  Trained (`model.emitters.{l}.*` tensors present): private copies built by
    `_materialize_private`, fed through the stock TP weight loaders.  Not an nn.Module on purpose: the aliased stock
    modules must not appear twice in the model's parameter tree."""

    # HF-style emitter parameter names (what twinstar/qwen4_exp.py emitters are expected to save) -> loader targets
    HC_NAMES = (
        "attn_hyper_connection.hc_norm.weight",
        "attn_hyper_connection.input_mix_weight_down.weight",
        "attn_hyper_connection.input_mix_weight_up.weight",
    )
    GDN_NAMES = (
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
    )
    QSA_NAMES = (
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.k_norm.weight",
        "self_attn.indexer.index_qk_proj.weight",
        "self_attn.indexer.k_layernorm.weight",
    )
    IGNORED = (
        "attn_hyper_connection.block_inject_weight.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.o_proj.weight",
        "self_attn.indexer.q_layernorm.weight",
    )

    def __init__(
        self, tcfg, quant_config, layer_id: int, layer: nn.Module, *, precision=None
    ):
        self.tcfg = tcfg
        self.quant_config = quant_config
        self.layer_id = layer_id
        self.layer = (
            layer  # the stock Qwen4Exp{Linear,Attention}DecoderLayer of layer l
        )
        self.is_attn = _is_attn_layer(tcfg, layer_id)
        self.hc = layer.attn_hyper_connection  # alias until a trained tensor arrives
        self.gdn = layer.linear_attn if not self.is_attn else None
        self.qsa = None  # SimpleNamespace with the fields forward_prepare_cuda_fused needs (private copy) or None (alias)
        self.private = False
        self._loaded: set = set()
        from .config import runtime_controls

        self.state_only = (
            runtime_controls()["emitter_state_only"] and precision == "bf16"
        )
        # Reference arithmetic is the default, including the release's fp32 dt_bias.
        # Explicit bf16 selects production arithmetic; the old boolean remains an alias.
        self.precision = resolve_emitter_precision(
            SimpleNamespace(duet_emitter_precision=precision)
        )
        self.fp32 = self.precision == "fp32"
        self.dt_bias_fp32 = self.fp32
        if (self.state_only or self.fp32) and quant_config is not None:
            raise ValueError(
                "state-only / fp32 emitters require unquantized emitter weights"
            )
        logger.info(
            "TwinStar emitter %d state_only=%s fp32=%s dt_bias_fp32=%s",
            layer_id,
            self.state_only,
            self.fp32,
            self.dt_bias_fp32,
        )

    # ------------------------------------------------------------------ trained emitters (Phase 2, untested)
    def _materialize_private(self):
        if self.private:
            return
        from sglang.srt.layers.hyperconnection import (
            GatedResidual,
            HyperConnectionConfig,
        )

        # Build every private module UNDER the device context (stock does the same): GroupedGemmaRMSNorm / GemmaRMSNorm
        # allocate plain nn.Parameters without a device argument (AGA 741521: a CPU hc_norm weight reached the JIT grouped
        # RMSNorm kernel), and Qwen3_5GatedDeltaNet hands RadixLinearAttention *views* of conv1d.weight / A_log / dt_bias that
        # a later .to(device) would leave stale.
        dev = torch.device(torch.cuda.current_device())
        hc_cfg = HyperConnectionConfig(
            hc_count=self.tcfg.hc_count,
            hidden_size=self.tcfg.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=self.tcfg.hc_lowrank,
            rms_norm_eps=self.tcfg.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        with torch.device(dev):
            hc = GatedResidual(hc_cfg, use_mix=True, use_combine=False)
        with torch.no_grad():
            hc.hc_norm.weight.copy_(self.layer.attn_hyper_connection.hc_norm.weight)
            hc.input_mix_weight_down.weight.copy_(
                self.layer.attn_hyper_connection.input_mix_weight_down.weight
            )
            hc.input_mix_weight_up.weight.copy_(
                self.layer.attn_hyper_connection.input_mix_weight_up.weight
            )
        self.hc = hc
        if not self.is_attn:
            with torch.device(dev):
                g = _q35.Qwen3_5GatedDeltaNet(
                    self.tcfg,
                    self.layer_id,
                    self.quant_config,
                    None,
                    f"model.emitters.{self.layer_id}.linear_attn",
                )
            with (
                torch.no_grad()
            ):  # start from the target layer (only the loaded tensors then differ)
                for n, p in g.named_parameters():
                    p.copy_(dict(self.layer.linear_attn.named_parameters())[n])
            for n_, p_ in g.named_parameters():
                assert p_.device.type == "cuda", (
                    f"private GDN emitter param {n_} on {p_.device}"
                )
            if (
                self.dt_bias_fp32
            ):  # the gate kernels upcast dt_bias anyway: only its stored value changes
                dt = nn.Parameter(g.dt_bias.detach().float(), requires_grad=False)
                dt.weight_loader = g.dt_bias.weight_loader
                g.dt_bias = dt
                g.attn.dt_bias = dt  # RadixLinearAttention holds its own reference
            self.gdn = g
        else:
            from sglang.srt.layers.attention.qsa.glue import build_qsa_indexer
            from sglang.srt.layers.layernorm import GemmaRMSNorm
            from sglang.srt.layers.linear import QKVParallelLinear

            L = self.layer
            with torch.device(dev):
                qkv = QKVParallelLinear(
                    self.tcfg.hidden_size,
                    L.head_dim,
                    L.total_num_heads * 2,
                    L.total_num_kv_heads,
                    bias=False,
                    quant_config=self.quant_config,
                    tp_rank=L.attn_tp_rank,
                    tp_size=L.attn_tp_size,
                    kv_tp_rank=L.kv_tp_rank,
                    kv_tp_size=L.kv_tp_size,
                    prefix=f"model.emitters.{self.layer_id}.qkv_proj",
                )
                k_norm = GemmaRMSNorm(L.head_dim, eps=self.tcfg.rms_norm_eps)
                idx = build_qsa_indexer(
                    config=self.tcfg,
                    layer_id=self.layer_id,
                    quant_config=self.quant_config,
                    prefix=f"model.emitters.{self.layer_id}.indexer",
                    rotary_emb=L.rotary_emb,
                )
            for m in (qkv, k_norm, idx):
                for n_, p_ in m.named_parameters():
                    assert p_.device.type == "cuda", (
                        f"private QSA emitter param {n_} on {p_.device}"
                    )
            with torch.no_grad():
                qkv.weight.copy_(L.qkv_proj.weight)
                k_norm.weight.copy_(L.k_norm.weight)
                idx.index_qk_proj.weight.copy_(L.indexer.index_qk_proj.weight)
                idx.q_layernorm.weight.copy_(L.indexer.q_layernorm.weight)
                idx.k_layernorm.weight.copy_(L.indexer.k_layernorm.weight)
            self.qsa = SimpleNamespace(
                qkv_proj=qkv,
                k_norm=k_norm,
                indexer=idx,
                q_norm=L.q_norm,
                rotary_emb=L.rotary_emb,
                attn_output_gate=L.attn_output_gate,
                q_size=L.q_size,
                kv_size=L.kv_size,
                num_heads=L.num_heads,
                num_kv_heads=L.num_kv_heads,
                head_dim=L.head_dim,
            )
            if self.fp32:  # fill the host-built RoPE frequency cache now (no host copy inside a graph capture)
                from sglang.srt.layers import twinstar_emitter_fp32

                twinstar_emitter_fp32.rope_tables(
                    L.rotary_emb,
                    torch.zeros(1, dtype=torch.long, device=dev),
                    torch.bfloat16,
                )
        self.private = True

    @torch.no_grad()
    def load(self, pname: str, w: torch.Tensor):
        if pname in self.IGNORED:
            return
        self._materialize_private()

        def put(param, tensor, *shard):
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, tensor, *shard)
            else:
                param.data.copy_(tensor.to(param.dtype))

        if pname == "attn_hyper_connection.hc_norm.weight":
            put(self.hc.hc_norm.weight, w)
        elif pname == "attn_hyper_connection.input_mix_weight_down.weight":
            put(self.hc.input_mix_weight_down.weight, w)
        elif pname == "attn_hyper_connection.input_mix_weight_up.weight":
            put(self.hc.input_mix_weight_up.weight, w)
        elif not self.is_attn and pname in self.GDN_NAMES:
            g = self.gdn
            if pname == "linear_attn.in_proj_qkv.weight":
                put(g.in_proj_qkvz.weight, w, (0, 1, 2))
            elif pname == "linear_attn.in_proj_b.weight":
                put(g.in_proj_ba.weight, w, 0)
            elif pname == "linear_attn.in_proj_a.weight":
                put(g.in_proj_ba.weight, w, 1)
            elif pname == "linear_attn.conv1d.weight":
                put(g.conv1d.weight, w)
            elif pname == "linear_attn.A_log":
                put(g.A_log, w)
            else:
                put(g.dt_bias, w)
        elif self.is_attn and pname in self.QSA_NAMES:
            q = self.qsa
            if pname == "self_attn.k_proj.weight":
                put(q.qkv_proj.weight, w, "k")
            elif pname == "self_attn.v_proj.weight":
                put(q.qkv_proj.weight, w, "v")
            elif pname == "self_attn.k_norm.weight":
                put(q.k_norm.weight, w)
            elif pname == "self_attn.indexer.index_qk_proj.weight":
                put(q.indexer.index_qk_proj.weight, w)
            else:
                put(q.indexer.k_layernorm.weight, w)
        else:
            raise KeyError(
                f"TwinStar Qwen4Exp emitter {self.layer_id}: unexpected tensor {pname}"
            )
        self._loaded.add(pname)

    def finalize(self, *, strict=False, base_filled=()):
        if not self.private and not strict:
            return
        need = set(self.HC_NAMES) | set(
            self.GDN_NAMES if not self.is_attn else self.QSA_NAMES
        )
        need -= set(base_filled)
        miss = sorted(need - self._loaded)
        if miss and strict:
            raise RuntimeError(
                f"x256 emitter {self.layer_id}: missing required weights: {miss}"
            )
        if miss:
            logger.warning(
                "TwinStar Qwen4Exp emitter %d: %d tensors not in the checkpoint, kept from the target layer: %s",
                self.layer_id,
                len(miss),
                miss,
            )
        if not self.is_attn:
            self.gdn.finalize_fused_in_proj()

    # ------------------------------------------------------------------ emission
    def emit(self, streams: torch.Tensor, fb: ForwardBatch) -> None:
        if self.fp32:
            from sglang.srt.layers import twinstar_emitter_fp32 as e32

            x = e32.hc_mix(self.hc, streams)  # (T, D) fp32
        else:
            x, _ = self.hc.mix(
                streams
            )  # (T, D): the target layer's read of the 4 streams
        if not self.is_attn:
            if self.fp32:  # fp32 projections; the backend runs the fp32 conv and the reference's fla chunk kernel
                mixed_qkv, a, b = e32.gdn_inputs(self.gdn, x)
                self.gdn.attn(fb, mixed_qkv=mixed_qkv, a=a, b=b)
            elif self.state_only:
                from .emitters import emit_gdn_state

                emit_gdn_state(self.gdn, x, fb)
            else:
                self.gdn(x, fb)  # conv window + SSM state; output discarded
            return
        if hasattr(fb, "flashnext_arrival_plan"):
            from sglang.srt.mem_cache.flashnext_materialization import emit_kv

            return emit_kv(self, x, fb)
        src = self.qsa if self.qsa is not None else self.layer
        if self.fp32:
            k, v = e32.qsa_kv(src, x, fb.positions, src.qkv_proj.weight.dtype)
        elif self.state_only:
            from .emitters import project_qsa_kv

            k, v = project_qsa_kv(src, x, fb.positions)
        elif self.qsa is not None:
            q, k, v, gate = (
                _q35.Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused(
                    src, fb.positions, x
                )
            )
        else:
            q, k, v, gate = self.layer._prepare_qkv_gate(fb.positions, x, fb)
        attn = (
            self.layer.attn
        )  # RadixAttention(layer_id = l): only its ids / head counts are used here
        pool = get_token_to_kv_pool()
        pool.set_kv_buffer(
            attn,
            fb.out_cache_loc,
            k.view(-1, attn.tp_k_head_num, attn.qk_head_dim),
            v.view(-1, attn.tp_v_head_num, attn.v_head_dim),
        )
        if getattr(self.layer, "is_qsa", False):
            self._emit_indexer(src.indexer, x, fb)

    def _emit_indexer(self, indexer, x: torch.Tensor, fb: ForwardBatch) -> None:
        """The key half of QSAIndexer.forward_cuda (no selection): pending ring + compressed block keys of layer l."""
        prefill_graph = _optional_prefill_graph()

        if prefill_graph is not None and prefill_graph.active(fb):
            # Ring slots and the compression plan follow the live request layout.
            if getattr(self, "_graph_key", None) is None:
                self._graph_key = prefill_graph.register(self)
            return _emitter_index_break(x, self._graph_key)
        self._emit_indexer_now(indexer, x, fb)

    def emit_indexer_for_prefill_graph(self, x: torch.Tensor, fb: ForwardBatch) -> None:
        src = self.qsa if self.qsa is not None else self.layer
        self._emit_indexer_now(src.indexer, x, fb)

    def _emit_indexer_now(self, indexer, x: torch.Tensor, fb: ForwardBatch) -> None:
        meta = get_qsa_indexer_metadata(get_attn_backend(), self.layer_id, fb)
        positions = fb.positions
        logical = (positions[0] if positions.ndim == 2 else positions).flatten()
        n_valid = meta.get_token_to_batch_idx().numel()
        logical = logical[:n_valid]
        hs = x[:n_valid]
        pos = positions[:, :n_valid] if positions.ndim == 2 else positions[:n_valid]
        slots = meta.pending_ring_slots
        if slots is None:
            slots = indexer._pending_ring_slots(
                meta, logical, meta.compress_member_rows is not None
            )
        if self.fp32:
            from sglang.srt.layers import twinstar_emitter_fp32

            token_k, stored = (
                twinstar_emitter_fp32.index_key(
                    indexer, hs, indexer.index_qk_proj.weight.dtype
                ),
                False,
            )
        elif self.state_only:
            from .emitters import project_index_key

            token_k, stored = project_index_key(indexer, hs), False
        else:
            # A hoisted host bound (SGLANG_QSA_PREFILL_HOIST) replaces the per-layer positions.max() readback.
            bound = getattr(meta, "max_position_cpu", None)
            _q, token_k, stored = indexer.project_qk(
                hs,
                pos,
                pool=meta.token_to_kv_pool,
                cache_loc=slots,
                **({"max_position": bound} if bound is not None else {}),
            )
        indexer.update_key_state_and_compress(
            token_k, logical, pos, meta, state_slots=slots, state_stored=stored
        )


def _emitter_index(x, key):
    from sglang.srt.models import qwen4_exp_prefill_graph as prefill_graph

    fb, real = prefill_graph._live_batch()
    prefill_graph._owner(key).emit_indexer_for_prefill_graph(x[:real], fb)


def _emitter_index_stub(x, key):
    pass


def _emitter_index_break(x, key):
    global _EMITTER_INDEX_BREAK
    if _EMITTER_INDEX_BREAK is None:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            eager_on_graph,
        )

        _EMITTER_INDEX_BREAK = eager_on_graph(True, capture_stub=_emitter_index_stub)(
            _emitter_index
        )
    _EMITTER_INDEX_BREAK(x, key)


_EMITTER_INDEX_BREAK = None


# ----------------------------------------------------------------------------- KV-mixer bridge
class _MixerBridge(nn.Module):
    """spec.bridge_kind "mixer" (Phase 2 arms br1 P31 / P35): a trained copy of layer k-1's read gate (`attn_hyper_connection`,
    incl. the write gate `block_inject_weight`) and full GDN mixer (in_proj_qkv/z/a/b, conv, A_log, dt_bias, norm, out_proj),
    run once on P's final streams before the emitters: streams' = streams + mixer(read_gate(streams)) (x) write_gate.
    torch implementation (twinstar.qwen4_exp.GatedResidual / Qwen4GatedDeltaNet, fla chunk kernel) with full heads replicated
    on every TP rank (~58M params); transient state, NO pool slot -> each request's extend runs from a zero state, so only
    whole-prompt extends are served (a prefix batch falls back in _is_twinstar_prefill).  Same structure as
    twinstar_qwen3_5._GdnMixerBridge with the gated residual in place of input_layernorm.  Tensor names
    `model.bridge.{j}.<layer param name>` (attn_hyper_connection.*, linear_attn.*)."""

    def __init__(self, served_dir: str, layer_id: int):
        super().__init__()
        from .base import GatedResidual, Qwen4ExpConfig, Qwen4GatedDeltaNet, _cast_model

        cfg = Qwen4ExpConfig.from_hf(served_dir)
        assert not cfg.is_attn(layer_id), (
            f"mixer bridge is implemented for a GDN layer k-1 (layer {layer_id} is QSA)"
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.attn_hyper_connection = GatedResidual(cfg)
        self.linear_attn = Qwen4GatedDeltaNet(cfg)
        _cast_model(self, torch.bfloat16)  # A_log / dt_bias stay fp32
        self._loaded: set = set()

    @torch.no_grad()
    def load(self, pname: str, w: torch.Tensor):
        params = dict(self.named_parameters())
        if pname not in params:
            raise KeyError(
                f"TwinStar Qwen4Exp bridge {self.layer_id}: unexpected tensor {pname}"
            )
        p = params[pname]
        p.data.copy_(w.reshape(p.shape).to(p.dtype))
        self._loaded.add(pname)

    def finalize(self):
        miss = sorted(set(dict(self.named_parameters())) - self._loaded)
        if miss:
            raise RuntimeError(
                f"TwinStar Qwen4Exp bridge {self.layer_id}: tensors not in the checkpoint: {miss}"
            )

    def forward(self, streams: torch.Tensor, cu_cpu: List[int]) -> torch.Tensor:
        """streams (T, hc*D) flat -> (T, hc*D); per request (cu_cpu = [0, l1, l1+l2, ...]) from a zero GDN state."""
        if self.attn_hyper_connection.weight.device != streams.device:
            self.to(streams.device)
        hc, D = self.cfg.hc_count, self.cfg.hidden_size
        out = streams.clone()
        for a, b in zip(cu_cpu[:-1], cu_cpu[1:]):
            h = streams[a:b].view(1, b - a, hc, D)
            x, inject = self.attn_hyper_connection.mix(h)
            y, _ = self.linear_attn(x, None)
            out[a:b] = (
                self.attn_hyper_connection.combine(h, y, inject)
                .view(b - a, hc * D)
                .to(out.dtype)
            )
        return out


# ----------------------------------------------------------------------------- the model class
class Qwen4ExpForConditionalGeneration(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        from sglang.srt.arg_groups.overrides import resolved_view
        from sglang.srt.duet import numerics
        from sglang.srt.runtime_context import get_server_args

        from .config import PRODUCTION_SUPPORTED, install_config

        args = resolved_view(get_server_args())
        self.numerics_profile = numerics.require_profile(
            "flash-next", args, production_supported=PRODUCTION_SUPPORTED
        )
        install_config(config, args)
        tcfg = config.text_config if hasattr(config, "text_config") else config
        ts = getattr(config, "twinstar", None) or getattr(tcfg, "twinstar", None)
        self.fullstack = ts.get("fullstack") if ts else None
        if self.fullstack and "duet_spec" in self.fullstack:
            from sglang.srt.model_executor.duet_policy import apply_duet_options

            apply_duet_options(SimpleNamespace(hf_config=config), args)
        v3_components = bool(self.fullstack and self.fullstack.get("version") in (2, 3))
        self.fullstack_code = bool(
            v3_components and self.fullstack.get("latent") == "on"
        )
        self.fullstack_v3_latent = bool(
            self.fullstack_code
            and self.fullstack.get("prefill_saving_policy", "latent-and-ssm")
            != "kv-and-ssm"
        )
        self.fullstack_final = bool(
            v3_components and self.fullstack.get("version") == 3
        )
        self.fullstack_contract = None
        if self.fullstack:
            from .release import validate_release

            fs = self.fullstack
            if v3_components:
                # The frozen paper reference computes E/D and GDN projections
                # in fp32 without TF32. This only affects enabled v3 workers.
                torch.backends.cuda.matmul.allow_tf32 = False
                from sglang.srt.model_executor.fullstack_policy import (
                    fullstack_v3_config,
                )

                fullstack_v3_config(SimpleNamespace(hf_config=config))
            elif fs.get("version") != 1 or fs.get("status") != "component-candidate":
                raise ValueError("unsupported x256 serving configuration")
            if fs.get("latent") not in ("on", "off"):
                raise ValueError(
                    "x256 candidate requires an explicit latent on/off choice"
                )
            if fs.get("qsa_code") not in ("on", "off") or not (
                fs.get("gdn_state") == "dense"
                or fs.get("gdn_state") == f"rank:{fs.get('gdn_rank')}"
            ):
                raise ValueError("unsupported candidate QSA code or GDN state mode")
            from sglang.srt.model_executor.fullstack_policy import (
                fullstack_state_config,
            )
            from sglang.srt.runtime_context import get_disagg, get_exec, get_memory

            actual_state = get_exec().mamba.linear_attn_factored_state
            expected_state = fullstack_state_config(
                SimpleNamespace(hf_config=config),
                radix=not get_memory().disable_radix_cache,
                disaggregation_mode=get_disagg().disaggregation_mode,
            )
            if (actual_state or None) != expected_state:
                raise ValueError(
                    "fullstack GDN state was not resolved into the serving pool"
                )
            if get_exec().mamba.qsa_code_prefix != (fs["qsa_code"] == "on"):
                raise ValueError(
                    "fullstack QSA code mode was not resolved into the serving pool"
                )
            if fs["qsa_code"] == "on":
                from pathlib import Path

                if (
                    Path(get_exec().mamba.qsa_code_release).resolve()
                    != Path(fs["release"]).resolve()
                ):
                    raise ValueError("fullstack emitter and QSA code releases differ")
            self.fullstack_contract = getattr(config, "_duet_identity", None) or vars(
                validate_release(fs["release"], config)
            )
            if fs.get("sha256") != self.fullstack_contract["sha256"]:
                raise ValueError("x256 configuration and release hashes differ")
        self.model = _stock.Qwen4ExpForConditionalGeneration(
            config, quant_config, prefix
        )
        self.config = tcfg
        self.cfg_top = config
        self.quant_config = quant_config
        self.twinstar = ts
        self.n_layers = tcfg.num_hidden_layers
        self.emitters: dict = {}
        self.strict = os.environ.get("TWINSTAR_SGL_STRICT", "0") == "1"
        self.dump_dir = os.environ.get("TWINSTAR_SGL_DUMP") or None
        self.state_audit_dir = os.environ.get("TWINSTAR_FULLSTACK_STATE_AUDIT") or None
        self.profile = os.environ.get("TWINSTAR_PROFILE", "0") == "1"
        self.boundary_mode = os.environ.get("TWINSTAR_BOUNDARY", "extend")
        # "none" = BENCH ONLY: shallow prefill + emitters, no boundary chunk at all (logits are zeros).  Emulates the P side
        # of the deferred form (boundary computed on the decode worker, docs/27 s6 / docs/32 s5.1b) for prefill-only timing.
        assert self.boundary_mode in ("extend", "graph", "none"), (
            f"TWINSTAR_BOUNDARY={self.boundary_mode}"
        )
        self.boundary_m = max(
            1, int(os.environ.get("TWINSTAR_BOUNDARY_M", "4"))
        )  # minimum; rounded up to a 4-aligned start
        self.noalign = (
            os.environ.get("TWINSTAR_BOUNDARY_NOALIGN", "0") == "1"
        )  # graph form: exactly M replays (no rounding)
        self.ratio = int(getattr(tcfg, "indexer_compress_ratio", 4) or 4)
        # chunk 2 must start at a multiple of this (default = the QSA compress ratio); TWINSTAR_BOUNDARY_ALIGN_UNIT=1472 makes
        # the split of a 1503-token prompt identical to stock chunked prefill at chunked_prefill_size=1472 (plumbing control)
        self.align_unit = max(
            self.ratio,
            int(os.environ.get("TWINSTAR_BOUNDARY_ALIGN_UNIT", str(self.ratio))),
        )
        assert self.align_unit % self.ratio == 0, (
            "TWINSTAR_BOUNDARY_ALIGN_UNIT must be a multiple of the QSA compress ratio"
        )
        self._model_runner = None
        self._dump_n = 0
        self.n_fallback = self.n_twinstar = self.n_prefix = self.n_graph_fallback = 0
        self.n_graph_trunk = self.n_graph_emitters = 0
        self._prefill_runners = {}
        self._boundary_runner = None
        self.tp_rank = get_parallel().attn_tp_rank
        if ts is None:
            logger.info(
                "TwinStar Qwen4Exp class without a twinstar config section: stock behaviour"
            )
            return
        assert get_pp_group().world_size == 1, (
            "TwinStar: pipeline parallelism not supported"
        )
        n = self.n_layers
        self.p_layer_ids = sorted(int(x) for x in ts["p_layers"])
        self.emitter_ids = sorted(int(x) for x in ts.get("emitters", []))
        assert sorted(int(x) for x in ts["d_layers"]) == list(range(n)), (
            "D must be the full stock layer stack"
        )
        k = len(self.p_layer_ids)
        assert self.p_layer_ids == list(range(k)) and self.emitter_ids == list(
            range(k, n)
        ), "shared form expected: P = layers 0..k-1, emitters for k..n-1"
        assert k >= 2 or not getattr(tcfg, "ple_layer_ids", None), (
            "the PLE layer (layer 1) must be inside P"
        )
        self.k = k
        if self.fullstack:
            spec = self.fullstack.get("duet_spec")
            if spec is not None:
                if k != spec["prefill_depth"] or int(ts.get("bridge", 0)):
                    raise ValueError(
                        "DUET layer cut differs from spec or unsupported bridge"
                    )
            elif k != 31 or n != 48 or int(ts.get("bridge", 0)):
                raise ValueError("legacy x256 requires P31/D48 with bridge=0")
            if not self.fullstack.get("prefill_layer_trim", True):
                self.p_layer_ids = list(range(n))
            self.strict = True
            self.boundary_mode, self.boundary_m, self.noalign = "graph", 1, True
            self.latent_codec = None
            if self.fullstack["latent"] == "on":
                from .latent import FlashNextLatentCodec

                self.latent_codec = FlashNextLatentCodec(
                    device=torch.device(torch.cuda.current_device()),
                    spec=self.fullstack["duet_spec"],
                    width=tcfg.hc_count * tcfg.hidden_size,
                    compute_precision=self.fullstack["latent_compute_precision"],
                )
        if self.fullstack and "duet_spec" in self.fullstack:
            logger.info(
                "DUET controls: trim=%s saving=%s r=%s W=%s P_compute_layers=%d; LinearCode codes every token",
                self.fullstack["prefill_layer_trim"],
                self.fullstack["prefill_saving_policy"],
                self.fullstack["gdn_rank"],
                self.fullstack["gdn_every"],
                len(self.p_layer_ids),
            )
        body = self.model.model
        # Published components are unquantized even when the unchanged routed
        # experts use ModelOpt. Do not apply the base quantizer to emitters.
        emitter_quant = (
            None if self.fullstack and "duet_spec" in self.fullstack else quant_config
        )
        from sglang.srt.runtime_context import get_exec

        emitter_precision = get_exec().mamba.duet_emitter_precision
        for l in self.emitter_ids:
            self.emitters[str(l)] = _Emitter(
                tcfg, emitter_quant, l, body.layers[l], precision=emitter_precision
            )
        if self.fullstack_v3_latent:
            from sglang.srt.runtime_context import get_disagg

            from .serving import capture_boundary

            self.v3_pd_role = get_disagg().disaggregation_mode
            if self.v3_pd_role == "prefill":
                body.layers[self.k].register_forward_pre_hook(
                    capture_boundary, with_kwargs=True
                )
        self.bridge_n = int(ts.get("bridge", 0) or 0)
        self.bridge_kind = ts.get("bridge_kind", "layer") or "layer"
        self.bridges = nn.ModuleList()
        if self.bridge_n:
            assert self.bridge_kind == "mixer", (
                f"bridge_kind {self.bridge_kind} not supported in the Qwen4Exp SGLang class"
            )
            served = getattr(config, "_name_or_path", None) or os.environ.get(
                "TWINSTAR_SERVED_DIR", ""
            )
            assert served and os.path.exists(os.path.join(served, "config.json")), (
                f"TwinStar: cannot locate the served directory for the bridge config (got {served!r}; set TWINSTAR_SERVED_DIR)"
            )
            for _ in range(self.bridge_n):
                self.bridges.append(_MixerBridge(served, k - 1))
        kinds = [
            "QSA" if self.emitters[str(l)].is_attn else "GDN" for l in self.emitter_ids
        ]
        logger.info(
            "TwinStar Qwen4Exp: P layers 0..%d, emitters %d (%d GDN + %d QSA), bridge %d x %s, tp %d, boundary %s (m>=%d, %d-aligned)",
            k - 1,
            len(self.emitter_ids),
            kinds.count("GDN"),
            kinds.count("QSA"),
            self.bridge_n,
            self.bridge_kind,
            get_parallel().attn_tp_size,
            self.boundary_mode,
            self.boundary_m,
            self.align_unit,
        )

    # ------------------------------------------------------------------ delegation to the stock body
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(super().__getattr__("model"), name)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_embed_and_head(self):
        return self.model.get_embed_and_head()

    def set_embed_and_head(self, embed, head):
        return self.model.set_embed_and_head(embed, head)

    @property
    def routed_experts_weights_of_layer(self):
        return self.model.routed_experts_weights_of_layer

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        return _stock.Qwen4ExpForConditionalGeneration.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return _stock.Qwen4ExpForConditionalGeneration.get_model_config_for_expert_location(
            config
        )

    def prepare_before_cuda_graph_capture(self, model_runner):
        self._model_runner = model_runner
        f = getattr(self.model, "prepare_before_cuda_graph_capture", None)
        if f is not None:
            f(model_runner)

    def precompile_kernels_after_loading(self) -> None:
        f = getattr(self.model, "precompile_kernels_after_loading", None)
        if f is not None:
            f()

    def prepare_decode_arrivals(self, forward_batch):
        if self.fullstack_v3_latent and self.v3_pd_role == "decode":
            from .serving import prepare_pd_arrivals

            prepare_pd_arrivals(self, forward_batch)

    def _decode_graph_runner(self):
        mr = self._model_runner
        if mr is None:
            return None
        for name in ("decode_cuda_graph_runner", "cuda_graph_runner"):
            r = getattr(mr, name, None)
            if r is not None and hasattr(r, "can_run_graph") and hasattr(r, "execute"):
                return r
        return None

    # ------------------------------------------------------------------ forward
    def _is_twinstar_prefill(self, fb: ForwardBatch) -> bool:
        if self.twinstar is None or get_is_capture_mode():
            return False
        if (
            self.fullstack
            and not self.fullstack.get("prefill_layer_trim", True)
            and self.fullstack["gdn_rank"] == 0
        ):
            # All-off accuracy uses the base prefill, including its token batch
            # shape. Splitting off a D boundary changes NVFP4 quantization even
            # when P already runs every layer and no emitter/state cut is used.
            return False
        fm = fb.forward_mode
        if (
            not fm.is_extend()
            or fm.is_target_verify()
            or fm.is_draft_extend_v2()
            or fm.is_mixed()
        ):
            return False
        if fb.extend_prefix_lens_cpu is None or fb.extend_seq_lens_cpu is None:
            return False
        if fb.spec_info is not None and not self.fullstack:
            return False
        if (
            getattr(fb, "can_run_tbo", False)
            or fb.input_embeds is not None
            or getattr(fb, "tbo_parent_token_range", None) is not None
        ):
            return False
        if min(fb.extend_seq_lens_cpu) < 1:
            return False
        if getattr(self, "bridge_n", 0) and max(fb.extend_prefix_lens_cpu) > 0:
            msg = (
                f"TwinStar: bridge layers have no cached state; extend batch with prefix {fb.extend_prefix_lens_cpu} "
                f"-> stock forward (disable the radix cache / chunked prefill)"
            )
            if self.strict:
                raise RuntimeError(msg)
            self.n_fallback += 1
            if self.n_fallback <= 3 or self.n_fallback % 100 == 0:
                logger.warning("%s [count %d]", msg, self.n_fallback)
            return False
        if self.fullstack:
            final = getattr(fb, "twinstar_prompt_final", None)
            if final is None or len(final) != len(fb.extend_seq_lens_cpu):
                raise RuntimeError(
                    "x256 requires the fullstack fork's prompt-final metadata"
                )
            # A one-token final extend has no shallow work and uses native DECODE,
            # which supports a partial QSA block.
            unaligned = any(
                int(p) % self.ratio and not (int(l) == 1 and f)
                for p, l, f in zip(
                    fb.extend_prefix_lens_cpu, fb.extend_seq_lens_cpu, final
                )
            )
        else:
            unaligned = any(int(p) % self.ratio for p in fb.extend_prefix_lens_cpu)
        if unaligned:
            msg = f"TwinStar: extend prefix not {self.ratio}-aligned {fb.extend_prefix_lens_cpu} -> stock forward"
            if self.strict:
                raise RuntimeError(msg)
            self.n_fallback += 1
            if self.n_fallback <= 3 or self.n_fallback % 100 == 0:
                logger.warning("%s [count %d]", msg, self.n_fallback)
            return False
        return True

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        if (
            not get_embedding
            and pp_proxy_tensors is None
            and self._is_twinstar_prefill(forward_batch)
        ):
            hidden = self._twinstar_prefill(input_ids, positions, forward_batch)
            if isinstance(
                hidden, LogitsProcessorOutput
            ):  # graph boundary: the decode runner already applied lm_head
                out = hidden
            else:
                out = self.model.logits_processor(
                    input_ids, hidden, self.model.lm_head, forward_batch
                )
            mode = "prefill"
        else:
            out = self.model.forward(
                input_ids,
                positions,
                forward_batch,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            mode = "fallback" if forward_batch.forward_mode.is_extend() else "decode"
        if (
            self.dump_dir is not None
            and getattr(out, "next_token_logits", None) is not None
            and not get_is_capture_mode()
            and self.tp_rank == 0
        ):
            self._dump(mode, input_ids, positions, forward_batch, out.next_token_logits)
        if self.state_audit_dir and not get_is_capture_mode():
            diagnostic = optional("fullstack_state_audit")
            if diagnostic is not None:
                diagnostic.record(self, forward_batch, "output")
        return out

    def _dump(self, mode, input_ids, positions, fb, logits):
        os.makedirs(self.dump_dir, exist_ok=True)
        rec = {
            "mode": mode,
            "input_ids": input_ids.cpu(),
            "positions": positions.cpu(),
            "extend_seq_lens": list(fb.extend_seq_lens_cpu)
            if fb.extend_seq_lens_cpu is not None
            else None,
            "extend_prefix_lens": list(fb.extend_prefix_lens_cpu)
            if fb.extend_prefix_lens_cpu is not None
            else None,
            "seq_lens": fb.seq_lens.cpu(),
            "req_pool_indices": fb.req_pool_indices.cpu(),
            "logits": logits.float().cpu(),
        }
        torch.save(rec, os.path.join(self.dump_dir, f"{self._dump_n:06d}.pt"))
        self._dump_n += 1

    # ------------------------------------------------------------------ batch decomposition
    def _boundary_lens(self, fb: ForwardBatch) -> List[int]:
        """m per request: at least TWINSTAR_BOUNDARY_M, rounded up so that chunk 2 starts compress-ratio-aligned
        (the QSA backend asserts prefix % ratio == 0), capped at the request's extend length."""
        if self.fullstack:
            final = getattr(fb, "twinstar_prompt_final", None)
            if final is None or len(final) != len(fb.extend_seq_lens_cpu):
                raise RuntimeError("x256 missing prompt-final metadata")
            return [1 if f else 0 for f in final]
        r, M = self.align_unit, self.boundary_m
        if self.boundary_mode == "none":
            return [0] * len(fb.extend_seq_lens_cpu)
        out = []
        for l, p in zip(fb.extend_seq_lens_cpu, fb.extend_prefix_lens_cpu):
            l, p = int(l), int(p)
            # graph boundary: the m replays are DECODE steps, which the stock backend runs after any prompt length (the
            # partial last block goes to the tail buffer exactly as after a stock prefill of L-m tokens), so the
            # compress-ratio alignment of chunk 2 is only needed for the eager EXTEND form.  TWINSTAR_BOUNDARY_NOALIGN=1
            # (graph form only) uses m = M exactly: 4 -> 2 replays per request on the served form (docs/32 s5.1).
            if (
                self.boundary_mode == "graph"
                and self.noalign
                and getattr(self, "_graph_ok", False)
                and l - M >= r
            ):
                # only when the graph path will actually run for this batch (_graph_ok: no logprob request, no spec
                # info -- otherwise the boundary falls back to the eager EXTEND, whose QSA write plan asserts on an
                # unaligned prefix: parity harness with top-k logprobs, AGA 744083) and the P stream keeps at least one
                # full compress block (shorter prompts fall through to the aligned rule -> whole-prompt extend)
                out.append(M)
                continue
            m = M + ((p + l - M) % r)
            out.append(min(m, l))
        return out

    def _sub_batch(self, fb: ForwardBatch, input_ids, positions, which: str):
        """Derived extend batch: which='p' = every request's extend tokens except its last m (requests whose whole extend
        is the boundary chunk are dropped); which='b' = every request's last m tokens with prefix = seq_len - m.
        Returns (batch, token idx)."""
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        prefix = [int(x) for x in fb.extend_prefix_lens_cpu]
        ms = self._boundary_lens(fb)
        starts = [0] + list(itertools.accumulate(lens))
        dev = input_ids.device
        if which == "p":
            sel = [r for r, l in enumerate(lens) if l - ms[r] > 0]
            idx_cpu = (
                torch.cat(
                    [torch.arange(starts[r], starts[r] + lens[r] - ms[r]) for r in sel]
                )
                if sel
                else torch.zeros(0, dtype=torch.long)
            )
            new_lens = [lens[r] - ms[r] for r in sel]
            new_prefix = [prefix[r] for r in sel]
        else:
            sel = [r for r in range(len(lens)) if ms[r] > 0]
            idx_cpu = (
                torch.cat(
                    [
                        torch.arange(starts[r] + lens[r] - ms[r], starts[r] + lens[r])
                        for r in sel
                    ]
                )
                if sel
                else torch.zeros(0, dtype=torch.long)
            )
            new_lens = [ms[r] for r in sel]
            new_prefix = [prefix[r] + lens[r] - ms[r] for r in sel]
        idx = _dev(idx_cpu, torch.long, dev)
        sel_t = torch.tensor(sel, dtype=torch.long)
        sel_d = _dev(sel_t, torch.long, dev)
        nb = copy.copy(fb)
        if self.fullstack:
            # P/emitter work is ordinary EXTEND, not a verify candidate batch.
            nb.spec_info = None
        nb.batch_size = len(sel)
        if getattr(fb, "twinstar_prompt_final", None) is not None:
            nb.twinstar_prompt_final = [fb.twinstar_prompt_final[r] for r in sel]
        nb.input_ids = input_ids[idx]
        nb.positions = positions[idx]
        nb.req_pool_indices = fb.req_pool_indices[sel_d]
        if getattr(fb, "req_pool_indices_cpu", None) is not None:
            nb.req_pool_indices_cpu = fb.req_pool_indices_cpu[sel_t]
        seq_new = [p + l for p, l in zip(new_prefix, new_lens)]
        nb.seq_lens = _dev(seq_new, fb.seq_lens.dtype, dev)
        nb.seq_lens_cpu = torch.tensor(
            seq_new,
            dtype=fb.seq_lens_cpu.dtype if fb.seq_lens_cpu is not None else torch.int64,
        )
        nb.seq_lens_sum = int(sum(seq_new))
        if fb.orig_seq_lens is not None:
            nb.orig_seq_lens = fb.orig_seq_lens[sel_d]
        nb.out_cache_loc = fb.out_cache_loc[idx]
        if getattr(fb, "out_cache_loc_virtual", None) is not None:
            nb.out_cache_loc_virtual = fb.out_cache_loc_virtual[idx]
        nb.extend_num_tokens = int(idx_cpu.numel())
        nb.extend_seq_lens = _dev(new_lens, fb.extend_seq_lens.dtype, dev)
        nb.extend_prefix_lens = _dev(new_prefix, fb.extend_prefix_lens.dtype, dev)
        nb.extend_start_loc = (
            _dev(
                [0] + list(itertools.accumulate(new_lens))[:-1],
                fb.extend_start_loc.dtype,
                dev,
            )
            if fb.extend_start_loc is not None
            else None
        )
        nb.extend_seq_lens_cpu = new_lens
        nb.extend_prefix_lens_cpu = new_prefix
        if fb.extend_logprob_start_lens_cpu is not None:
            nb.extend_logprob_start_lens_cpu = [0] * len(sel)
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(
                    nb,
                    name,
                    t[sel_d if t.device == sel_d.device else sel_t.to(t.device)],
                )
        if getattr(fb, "global_num_token_non_padded_cpu", None) is not None:
            nb.global_num_token_non_padded_cpu = nb.extend_num_tokens
        if getattr(fb, "global_num_token_non_padded", None) is not None:
            nb.global_num_token_non_padded = _dev(
                nb.extend_num_tokens, fb.global_num_token_non_padded.dtype, dev
            )
        if getattr(fb, "num_token_non_padded", None) is not None:
            nb.num_token_non_padded = _dev(
                nb.extend_num_tokens, fb.num_token_non_padded.dtype, dev
            )
        for name in ("forward_metadata_ready", "forward_metadata_replan_equivalent"):
            if hasattr(fb, name):
                setattr(nb, name, False)
        for name in (
            "forward_metadata_planned_bs",
            "forward_metadata_planned_num_tokens",
        ):
            if hasattr(fb, name):
                setattr(nb, name, None)
        nb.mm_inputs = None
        nb.input_embeds = None
        nb.mrope_positions = None
        return nb, idx

    def _publish_qsa_prefix(self, fb: ForwardBatch, boundary_lengths):
        """Publish complete P/emitter pages before the final D token reads them."""
        pool = getattr(get_token_to_kv_pool(), "full_kv_pool", None)
        publish = getattr(pool, "publish_before_boundary", None)
        if publish is None:
            if self.fullstack and self.fullstack["qsa_code"] == "on":
                raise RuntimeError("fullstack QSA code pool lacks the P/D handoff hook")
            return
        publish(
            fb.req_pool_indices,
            fb.seq_lens_cpu.tolist(),
            boundary_lengths,
            get_attn_backend().req_to_token_pool,
        )

    def _twinstar_prefill(
        self, input_ids: torch.Tensor, positions: torch.Tensor, fb: ForwardBatch
    ):
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        T = input_ids.shape[0]
        if sum(lens) != T:
            raise RuntimeError(
                f"TwinStar: padded extend batch (sum lens {sum(lens)} != {T} tokens); run the worker with "
                f"--disable-prefill-cuda-graph"
            )
        t0 = time.time() if self.profile else None
        bk = get_attn_backend()
        body = (
            self.model.model
        )  # Qwen4ExpVLModel: embed_tokens, layers, hyper_connection_mixer, PLE plumbing
        rec = get_global_expert_distribution_recorder()
        # the graph boundary is only available without logprob requests / spec info (same condition as below); the
        # exact-M (unaligned) split must not be used when the eager extend will run instead
        self._graph_ok = (
            self.boundary_mode == "graph"
            and fb.spec_info is None
            and not fb.return_logprob
        )
        if (
            self.fullstack
            and fb.return_logprob
            and (
                fb.extend_logprob_start_lens_cpu is None
                or any(
                    int(start) < length
                    for start, length in zip(fb.extend_logprob_start_lens_cpu, lens)
                )
            )
        ):
            raise ValueError(
                "x256 prompt logprobs are not implemented; output-only logprobs are supported"
            )
        capture = None
        if self.fullstack and fb.capture_hidden_mode.is_full():
            diagnostic = optional("mtp_hidden_capture")
            if diagnostic is not None:
                capture = diagnostic.PromptHiddenCapture(
                    lens,
                    self._boundary_lens(fb),
                    self.config.hc_count * self.config.hidden_size,
                )
        fb1, p_idx = self._sub_batch(fb, input_ids, positions, "p")
        if self.fullstack_v3_latent:
            fb1.flashnext_gdn_layer_range = (
                0,
                sum(
                    t == "linear_attention"
                    for t in self.config.layer_types[
                        : self.n_layers if self.fullstack_final else self.k
                    ]
                )
                - 1,
            )
        fb2, b_idx = (
            self._sub_batch(fb, input_ids, positions, "b")
            if self.boundary_mode != "none"
            else (None, None)
        )
        prof = {}

        def tick(name, t_start):
            if self.profile:
                torch.cuda.synchronize()
                now = time.time()
                prof[name] = prof.get(name, 0.0) + (now - t_start) * 1e3
                return now
            return t_start

        n = self.n_layers
        with get_attn_tp_context().maybe_input_scattered(fb1):
            if fb1.batch_size > 0:
                ts = time.time() if self.profile else None
                if self.state_audit_dir:
                    diagnostic = optional("fullstack_gdn_diagnostic")
                    if diagnostic is not None:
                        diagnostic.before_extend(self, fb1)
                bk.init_forward_metadata(fb1)
                ts = tick("meta1", ts)
                runners = getattr(self, "_prefill_runners", None) or {}
                trunk_runner = runners.get("trunk")
                graph_trunk = trunk_runner is not None and trunk_runner.can_run(fb1)
                if graph_trunk:
                    out = trunk_runner.run(fb1)
                    streams, latent_base = (
                        out if isinstance(out, tuple) else (out, None)
                    )
                    self.n_graph_trunk += 1
                else:
                    streams, latent_base = self._p_trunk(fb1)
                ts = tick("P%d" % len(self.p_layer_ids), ts)
                if self.emitter_ids:
                    assert (
                        streams.shape[-1]
                        == self.config.hc_count * self.config.hidden_size
                    ), (
                        f"expected the {self.config.hc_count}-stream residual after layer k-1, got {tuple(streams.shape)}"
                    )
                if len(self.bridges):
                    cu = [0] + list(
                        itertools.accumulate(int(x) for x in fb1.extend_seq_lens_cpu)
                    )
                    for br in self.bridges:
                        streams = br(streams, cu)
                    ts = tick("bridge", ts)
                if self.fullstack_v3_latent:
                    from .serving import encode_prefix

                    decoded = encode_prefix(self, streams, latent_base, fb1)
                    if self.fullstack_final:
                        streams = decoded
                    ts = tick("latent_encode_store", ts)
                elif self.fullstack_code and self.fullstack.get(
                    "prefill_layer_trim", True
                ):
                    from .serving import embedding_streams

                    # The storage policy does not alter quantization-aware emitter inputs.
                    _, streams = self.latent_codec.encode_and_decode(
                        streams, fb1.positions, embedding_streams(self, latent_base)
                    )
                    ts = tick("latent_transient", ts)
                elif (
                    self.fullstack
                    and not self.fullstack_code
                    and self.fullstack["latent"] == "on"
                ):
                    latent = self.latent_codec.encode(streams, fb1.positions)
                    ts = tick("latent_encode", ts)
                    streams = self.latent_codec.decode(latent)
                    ts = tick("latent_decode", ts)
                if capture is not None:
                    # #428: A uses P31 HC; final uses its existing decoded HC.
                    capture.put(capture.prefix_rows, streams)
                emit_runner = runners.get("emitters")
                if emit_runner is not None and emit_runner is trunk_runner:
                    emitted = graph_trunk  # one graph holds the trunk and its emitters
                else:
                    emitted = emit_runner is not None and emit_runner.can_run(fb1)
                    if emitted:
                        emit_runner.run(fb1, streams=streams)
                if emitted:
                    self.n_graph_emitters += 1
                else:
                    for l in self._emit_ids():
                        em = self.emitters[str(l)]
                        em.emit(streams, fb1)
                        if self.profile:
                            ts = tick("emitQSA" if em.is_attn else "emitGDN", ts)
                p287_state_hash = optional("p287_state_hash")
                if p287_state_hash is not None and p287_state_hash.enabled():
                    p287_state_hash.dump(self, fb1, streams)
                if self.state_audit_dir:
                    audit = optional("fullstack_state_audit")
                    diagnostic = optional("fullstack_gdn_diagnostic")
                    if audit is not None:
                        audit.record(self, fb1, "p-handoff")
                    if diagnostic is not None:
                        diagnostic.energy(self, fb1)
        if self.profile:
            torch.cuda.synchronize()
            t1 = time.time()
        ms = self._boundary_lens(fb)
        if self.fullstack_v3_latent and any(ms):
            from .serving import materialize_arrivals

            materialize_arrivals(self, fb, [i for i, length in enumerate(ms) if length])
        if any(ms):
            self._publish_qsa_prefix(fb, ms)
        form = "extend"
        if self.fullstack:
            if any(ms):
                out = self._boundary_graph(input_ids, positions, fb, hc_capture=capture)
                form = "final-token-decode"
            else:
                # Intermediate chunks are ignored by the scheduler's sampler.
                out = LogitsProcessorOutput(
                    next_token_logits=torch.zeros(
                        len(lens),
                        self.config.vocab_size,
                        dtype=torch.float32,
                        device=input_ids.device,
                    )
                )
                form = "intermediate-no-boundary"
        elif (
            self.boundary_mode == "none"
        ):  # bench only (deferred P side): no boundary, zero logits
            out = streams.new_zeros(T, self.config.hidden_size)
            form = "none"
        elif (
            self.boundary_mode == "graph"
            and fb.spec_info is None
            and not fb.return_logprob
            and all(l - m >= 1 for l, m in zip(lens, ms))
        ):
            # m DECODE steps per request (stock decode CUDA graph replay when the runner is available, else the same DECODE
            # steps eagerly -- the parity harness disables the decode graph; either way the boundary tokens take the stock
            # decode path, exactly what every generated token does after a full prefill)
            out = self._boundary_graph(input_ids, positions, fb)
            form = (
                "graph" if self._decode_graph_runner() is not None else "decode-eager"
            )
        else:
            bk.init_forward_metadata(fb2)
            with get_attn_tp_context().maybe_input_scattered(fb2):
                hb = body(fb2.input_ids, fb2.positions, fb2)
            if isinstance(hb, tuple):
                hb = hb[0]
            out = hb.new_zeros(T, hb.shape[-1])
            out[b_idx] = hb
        if capture is not None:
            out.hidden_states = capture.finish()
        # leave the backend planned for the original batch (extend form).  In the graph form this re-plan is SKIPPED by
        # default: with CUDA_LAUNCH_BLOCKING=1 the stock QSA write-plan assert `prefix_lens % 4 == 0` (AGA 743710) is
        # launched from exactly this call after the boundary decode replays whenever m is not a multiple of 4 in the
        # batch (GPQA mixed lengths, parity short prompts); nothing after the boundary reads the extend metadata (the
        # logits are already computed by the decode runner), and the smoke with the skip ran clean (AGA 744008).
        # TWINSTAR_BOUNDARY_REPLAN=1 restores the old behaviour.
        if form == "extend" or os.environ.get("TWINSTAR_BOUNDARY_REPLAN", "0") == "1":
            bk.init_forward_metadata(fb)
        self.n_twinstar += 1
        if any(p > 0 for p in fb.extend_prefix_lens_cpu):
            self.n_prefix += 1
        if self.profile:
            torch.cuda.synchronize()
            parts = " ".join(f"{k} {v:.1f}" for k, v in prof.items())
            logger.info(
                "TwinStar prefill: B=%d T=%d T_p=%d  P+emit %.1f ms [%s]  boundary[%s m=%s] %.1f ms",
                len(lens),
                T,
                int(p_idx.numel()),
                (t1 - t0) * 1e3,
                parts,
                form,
                ms,
                (time.time() - t1) * 1e3,
            )
        if self.n_twinstar % 500 == 1:
            logger.info(
                "TwinStar counters: shallow prefills %d (with prefix %d), stock fallbacks %d, graph-boundary fallbacks %d, "
                "graph trunk %d, graph emitters %d",
                self.n_twinstar,
                self.n_prefix,
                self.n_fallback,
                self.n_graph_fallback,
                self.n_graph_trunk,
                self.n_graph_emitters,
            )
        return out

    def _emit_ids(self):
        if self.fullstack and not self.fullstack.get("prefill_layer_trim", True):
            return []
        if self.fullstack_final and self.fullstack_v3_latent:
            return [l for l in self.emitter_ids if not self.emitters[str(l)].is_attn]
        return [] if self.fullstack_v3_latent else self.emitter_ids

    def _p_trunk(self, fb1: ForwardBatch):
        """Embedding and the P layers of one P sub-batch -> (4-stream residual, embedding for v3 latent)."""
        prefill_graph = _optional_prefill_graph()

        body = self.model.model
        n = self.n_layers
        rec = get_global_expert_distribution_recorder()
        hidden = body.embed_tokens(fb1.input_ids)
        latent_base = hidden if self.fullstack_code else None
        graph_ple = (
            body.has_ple and prefill_graph is not None and prefill_graph.active(fb1)
        )
        if graph_ple:
            ple_batch = prefill_graph.PLE_IN_BREAK
        else:
            ple_batch = (
                _stock._prepare_ple_batch(
                    fb1.input_ids,
                    fb1,
                    ngram_size=body.ple_ngram_size,
                    ngram_eos_token_id=body.ple_ngram_eos_token_id,
                )
                if body.has_ple
                else None
            )
        residual = None
        for l in self.p_layer_ids:
            if l + 1 < n and not graph_ple:
                next_ple = getattr(body.layers[l + 1], "ple", None)
                if next_ple is not None:
                    next_ple.start_prefetch(ple_batch, fb1)
            with rec.with_current_layer(l):
                hidden, residual = body.layers[l](
                    positions=fb1.positions,
                    hidden_states=hidden,
                    residual=residual,
                    forward_batch=fb1,
                    ple_batch=ple_batch,
                )
        if not graph_ple:
            _stock._commit_ple_batch(ple_batch, fb1)
        return (hidden if residual is None else hidden + residual), latent_base

    def capture_model_owned_graphs(self, model_runner):
        """#287: breakable graphs for the P sub-batch (TWINSTAR_PREFILL_GRAPH_TRUNK / _EMITTERS)."""
        self._model_runner = model_runner
        if self.twinstar is None or model_runner.is_draft_worker:
            return
        boundary_graph = optional("boundary_graph")
        from .prefill_graph import capture

        self._prefill_runners = capture(self, model_runner)
        if boundary_graph is not None and boundary_graph.enabled() and self.fullstack:
            self._boundary_runner = boundary_graph.capture(model_runner)

    # ------------------------------------------------------------------ graph form of chunk 2 (docs/31 s4.2)
    def _decode_batch(
        self,
        fb: ForwardBatch,
        input_ids,
        positions,
        tok_idx_cpu: torch.Tensor,
        sel: List[int],
        seq_new: List[int],
    ) -> ForwardBatch:
        dev = input_ids.device
        idx = _dev(tok_idx_cpu, torch.long, dev)
        sel_t = torch.tensor(sel, dtype=torch.long)
        sel_d = _dev(sel_t, torch.long, dev)
        nb = copy.copy(fb)
        nb.forward_mode = ForwardMode.DECODE
        if self.fullstack:
            nb.spec_info = None
        nb.batch_size = len(sel)
        nb.input_ids = input_ids[idx]
        nb.positions = positions[idx]
        nb.req_pool_indices = fb.req_pool_indices[sel_d]
        if getattr(fb, "req_pool_indices_cpu", None) is not None:
            nb.req_pool_indices_cpu = fb.req_pool_indices_cpu[sel_t]
        nb.seq_lens = _dev(seq_new, fb.seq_lens.dtype, dev)
        nb.seq_lens_cpu = torch.tensor(
            seq_new,
            dtype=fb.seq_lens_cpu.dtype if fb.seq_lens_cpu is not None else torch.int64,
        )
        nb.seq_lens_sum = int(sum(seq_new))
        if fb.orig_seq_lens is not None:
            nb.orig_seq_lens = fb.orig_seq_lens[sel_d]
        nb.out_cache_loc = fb.out_cache_loc[idx]
        if getattr(fb, "out_cache_loc_virtual", None) is not None:
            nb.out_cache_loc_virtual = fb.out_cache_loc_virtual[idx]
        nb.extend_num_tokens = None
        nb.extend_seq_lens = nb.extend_prefix_lens = nb.extend_start_loc = None
        nb.extend_seq_lens_cpu = nb.extend_prefix_lens_cpu = (
            nb.extend_logprob_start_lens_cpu
        ) = None
        nb.is_extend_in_batch = False
        nb.can_run_decode_cuda_graph = True
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(
                    nb,
                    name,
                    t[sel_d if t.device == sel_d.device else sel_t.to(t.device)],
                )
        if self.fullstack:
            nb.twinstar_prompt_final = None
            if nb.mamba_track_mask is not None:
                # The P sub-batch already saved the reusable prefix checkpoint.
                # A D boundary must not overwrite it with a different history.
                nb.mamba_track_mask = torch.zeros_like(nb.mamba_track_mask)
        if getattr(fb, "global_num_token_non_padded_cpu", None) is not None:
            nb.global_num_token_non_padded_cpu = len(sel)
        for name in ("global_num_token_non_padded", "num_token_non_padded"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, _dev(len(sel), t.dtype, dev))
        for name in ("forward_metadata_ready", "forward_metadata_replan_equivalent"):
            if hasattr(fb, name):
                setattr(nb, name, False)
        for name in (
            "forward_metadata_planned_bs",
            "forward_metadata_planned_num_tokens",
        ):
            if hasattr(fb, name):
                setattr(nb, name, None)
        nb.mm_inputs = None
        nb.input_embeds = None
        nb.mrope_positions = None
        return nb

    def _boundary_graph(
        self, input_ids, positions, fb: ForwardBatch, hc_capture=None
    ) -> LogitsProcessorOutput:
        # NEXTN target graphs execute four-input VERIFY, not one-input DECODE.
        # The prompt boundary must run the true target one-token forward.
        runner = (
            self._boundary_runner
        )  # #287 (d): one-input DECODE graph with FULL hidden capture
        if os.environ.get("SGLANG_PREFILL_GRAPH_CAPTURE_ONLY", "0") == "1":
            runner = None  # diagnostic: captured, never replayed
        if runner is None:
            runner = self._decode_graph_runner() if hc_capture is None else None
        bk = get_attn_backend()
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        prefix = [int(x) for x in fb.extend_prefix_lens_cpu]
        ms = self._boundary_lens(fb)
        starts = [0] + list(itertools.accumulate(lens))
        logits = None
        for i in range(max(ms)):
            sel = [r for r in range(len(lens)) if ms[r] > i]
            tok = torch.tensor(
                [starts[r] + lens[r] - ms[r] + i for r in sel], dtype=torch.long
            )
            seq_new = [prefix[r] + lens[r] - ms[r] + i + 1 for r in sel]
            fbd = self._decode_batch(fb, input_ids, positions, tok, sel, seq_new)
            if runner is not None and runner.can_run_graph(fbd):
                out = runner.execute(fbd)
            else:
                self.n_graph_fallback += 1
                bk.init_forward_metadata(fbd)
                out = self.model.forward(fbd.input_ids, fbd.positions, fbd)
            if hc_capture is not None:
                hc_capture.put(tok.tolist(), out.hidden_states)
            nl = out.next_token_logits
            if logits is None:
                logits = (
                    nl.new_zeros(len(lens), nl.shape[-1])
                    if self.fullstack
                    else nl.new_empty(len(lens), nl.shape[-1])
                )
            last = [j for j, r in enumerate(sel) if ms[r] == i + 1]
            if last:
                logits[_dev([sel[j] for j in last], torch.long, nl.device)] = nl[
                    _dev(last, torch.long, nl.device)
                ]
        return LogitsProcessorOutput(next_token_logits=logits)

    # ------------------------------------------------------------------ weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        def body_weights():
            for name, w in weights:
                if name.startswith(_E_PREFIX):
                    l, pname = name[len(_E_PREFIX) :].split(".", 1)
                    if l in self.emitters:
                        self.emitters[l].load(pname, w)
                    continue
                if name.startswith(_B_PREFIX):
                    j, pname = name[len(_B_PREFIX) :].split(".", 1)
                    if int(j) < len(getattr(self, "bridges", [])):
                        self.bridges[int(j)].load(pname, w)
                    continue
                yield name, w

        loaded = self.model.load_weights(body_weights())
        if self.fullstack:
            from pathlib import Path

            from safetensors import safe_open

            from .release import COMPONENT_FILE

            path = Path(self.fullstack_contract["path"]) / COMPONENT_FILE
            with safe_open(path, framework="pt", device="cpu") as weights:
                for name in weights.keys():
                    if name.startswith("P.emitters."):
                        layer, pname = name[len("P.emitters.") :].split(".", 1)
                        self.emitters[layer].load(pname, weights.get_tensor(name))
                    elif (
                        name.startswith("P.latent.core.")
                        and self.latent_codec is not None
                    ):
                        self.latent_codec.load(
                            name[len("P.latent.core.") :], weights.get_tensor(name)
                        )
            if self.latent_codec is not None:
                self.latent_codec.finalize()
        # #873: k31-r4096-u does not export the indexer k_layernorm (a read-side module); the private emitter keeps the
        # target layer's own weight, which is what the reference applies to the emitted raw indexer key.
        base_filled = (
            ("self_attn.indexer.k_layernorm.weight",)
            if self.fullstack
            and (
                "duet_spec" in self.fullstack
                or self.fullstack.get("release_name") == "duet-fn-k31-r4096-u"
            )
            else ()
        )
        for e in self.emitters.values():
            e.finalize(strict=bool(self.fullstack), base_filled=base_filled)
        for b in getattr(self, "bridges", []):
            b.finalize()
        if os.environ.get("TWINSTAR_PIPELINE_PROBE"):
            diagnostic = optional("fullstack_pipeline_probe")
            if diagnostic is not None:
                diagnostic.attach(self)
        if os.environ.get("TWINSTAR_CUDA_TIMELINE"):
            diagnostic = optional("twinstar.bench.flashnext_cuda_timeline")
            if diagnostic is not None:
                diagnostic.install()
        n_priv = sum(1 for e in self.emitters.values() if e.private)
        if self.twinstar is not None:
            logger.info(
                "TwinStar Qwen4Exp: %d emitters (%d with trained private weights, %d aliasing the stock target layers)",
                len(self.emitters),
                n_priv,
                len(self.emitters) - n_priv,
            )
        return loaded


for _a in (
    "hf_to_sglang_mapper",
    "packed_modules_mapping",
    "supports_cuda_vmm_feature_transport",
):
    if hasattr(_stock.Qwen4ExpForConditionalGeneration, _a):
        setattr(
            Qwen4ExpForConditionalGeneration,
            _a,
            getattr(_stock.Qwen4ExpForConditionalGeneration, _a),
        )

# Instrumentation and PD diagnostics are intentionally optional, never bundled.
_pd = optional("pd_shallow_install")
if _pd is not None:
    from .pd_shallow_install import install as _install_pd

    _install_pd(Qwen4ExpForConditionalGeneration, _stock)

from sglang.srt.duet.adapters import release_value

EntryClass = [Qwen4ExpForConditionalGeneration] if release_value() else []
