"""TwinStar shallow-prefill model class for Kimi Linear (subtask kimi-linear-spd-20260918, docs/66).

The class is *named* `KimiLinearForCausalLM`: the common release registry selects this in-tree
adapter when --duet-release names a Kimi release, preserving the stock architecture name (ModelConfig classifies the MLA + KDA pools by that name).  Without a `twinstar` section in the
config it is a transparent wrapper around the stock model.

Shared form only (P = D's first k layers, the emitters write the caches of layers k..n-1).  One prefill batch runs as two
derived ForwardBatches so every kernel (KDA conv + chunked delta rule into the Mamba pool, MLA pool writes, TP collectives,
radix prefixes) is the stock one (same scheme as twinstar_glm5_next.py, minus mHC / DSA):

  chunk 1  every request's extend tokens except the last m (the P stream): stock layers 0..k-1 -> residual h (T_p, D);
           emitters: for each target layer l >= k, layer l's trained input_layernorm on h, then
             KDA target -> q/k/v/f/b projections, then a private RadixLinearAttention(layer_id=l) with the emitter's
                           conv / A_log / dt_bias (softplus gate, no lower bound) runs the stock KDA extend kernel, which
                           writes layer l's conv window + SSM state into the Mamba pool slots (this rank's head slice);
             MLA target -> latent c = kv_a_layernorm(kv_a[..., :512]), k_rot = kv_a[..., 512:] (NoPE: nothing rotated)
                           -> pool.set_mla_kv_buffer(layer l).  All MLA emitters run as one folded GEMM (TWINSTAR_EMIT_GROUP=1).
  chunk 2  every request's last m tokens through the stock 27-layer forward (P layers continue their own states, deep
           layers read the emitted caches) -> the boundary logits.  TWINSTAR_BOUNDARY=graph (default): m =
           TWINSTAR_BOUNDARY_M (3 = the chat template's `<|im_assistant|>assistant<|im_middle|>` anchors) DECODE steps through
           the stock decode CUDA graph runner; falls back to `extend` (one eager extend of the last token) when the runner
           cannot take the batch (logprobs, bs > capture, a request shorter than m+1 tokens).
Decode and any non-shallow extend (target-verify, mixed, bridge + prefix) go through the stock forward.
Env: TWINSTAR_SGL_DUMP=<dir> parity dumps; TWINSTAR_PROFILE=1 section
timers; TWINSTAR_BOUNDARY=extend|graph (+ TWINSTAR_BOUNDARY_M); TWINSTAR_EMIT_GROUP=0 per-layer MLA emitters;
TWINSTAR_EMIT_FUSED=state|0|checkstate (state-only fused KDA emitter group, docs/31 s4.3 port; default state) with
TWINSTAR_EMIT_BUDGET (emitter x token per fused call) and TWINSTAR_EMIT_FUSE_MAX_T (batch tokens above which the per-layer
path runs instead; default 16384).  The shallow
prefill never runs under a prefill CUDA graph (batch decomposition is CPU-shaped): --disable-prefill-cuda-graph.
"""
from __future__ import annotations

import copy
import itertools
import logging
import os
import re
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models import kimi_linear as _stock
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.duet import release, options, numerics
from sglang.srt.utils.common import BumpAllocator
from .controls import code_precision, component_upload

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_E_PREFIX = "model.emitters."
_B_PREFIX = "model.bridge."


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(x.dtype)


def _is_kda(cfg, l: int) -> bool:
    return cfg.is_kda_layer(l) if hasattr(cfg, "is_kda_layer") else ((l + 1) in cfg.linear_attn_config["kda_layers"])


# ----------------------------------------------------------------------------- emitters
class _KDAEmitter(nn.Module):
    """State emitter for a skipped KDA layer: layer l's trained input_layernorm + q/k/v/f/b projections (this rank's head
    slice), then the stock KDA extend kernel through a private RadixLinearAttention whose layer_id is the target layer ->
    the Mamba pool of layer l receives the conv window and SSM state."""

    EXPECTED = ("input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight", "self_attn.f_a_proj.weight",
                "self_attn.f_b_proj.weight", "self_attn.dt_bias", "self_attn.A_log", "self_attn.b_proj.weight")

    def __init__(self, cfg, layer_id: int, dtype):
        super().__init__()
        la = cfg.linear_attn_config
        self.layer_id = layer_id
        self.H, self.Dh, self.K = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
        self.tp = get_parallel().tp_size  # the stock KimiDeltaAttention shards on the global TP group
        self.tp_rank = get_parallel().tp_rank
        assert self.H % self.tp == 0
        self.Hl = self.H // self.tp
        self.h0 = self.tp_rank * self.Hl
        D, Dh, Hl, K = cfg.hidden_size, self.Dh, self.Hl, self.K
        self.norm_w = nn.Parameter(torch.empty(D, dtype=dtype), requires_grad=False)
        self.qkv_w = nn.Parameter(torch.empty(3 * Hl * Dh, D, dtype=dtype), requires_grad=False)
        self.conv_w = nn.Parameter(torch.empty(3 * Hl * Dh, K, dtype=torch.float32), requires_grad=False)
        self.f_a_w = nn.Parameter(torch.empty(Dh, D, dtype=dtype), requires_grad=False)
        self.f_b_w = nn.Parameter(torch.empty(Hl * Dh, Dh, dtype=dtype), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.empty(Hl * Dh, dtype=torch.float32), requires_grad=False)
        self.A_log = nn.Parameter(torch.empty(1, 1, Hl, 1, dtype=torch.float32), requires_grad=False)
        self.b_w = nn.Parameter(torch.empty(Hl, D, dtype=dtype), requires_grad=False)
        self.eps = cfg.rms_norm_eps
        self.attn = RadixLinearAttention(layer_id=layer_id, num_q_heads=Hl, num_k_heads=Hl, num_v_heads=Hl, head_q_dim=Dh,
                                         head_k_dim=Dh, head_v_dim=Dh, conv_weights=self.conv_w, bias=None, A_log=self.A_log,
                                         dt_bias=self.dt_bias, lower_bound=None)
        self._loaded = set()

    def _rows(self, w: torch.Tensor) -> torch.Tensor:  # head-major rows -> this rank's heads
        return w[self.h0 * self.Dh:(self.h0 + self.Hl) * self.Dh]

    @torch.no_grad()
    def load(self, pname: str, w: torch.Tensor):
        Hl, Dh = self.Hl, self.Dh
        seg = Hl * Dh
        if pname == "input_layernorm.weight":
            self.norm_w.copy_(w.to(self.norm_w.dtype))
        elif pname in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            i = "qkv".index(pname[len("self_attn.")])
            self.qkv_w[i * seg:(i + 1) * seg].copy_(self._rows(w).to(self.qkv_w.dtype))
        elif pname in ("self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"):
            i = "qkv".index(pname[len("self_attn.")])
            self.conv_w[i * seg:(i + 1) * seg].copy_(self._rows(w.reshape(w.shape[0], -1)).float())
        elif pname == "self_attn.f_a_proj.weight":
            self.f_a_w.copy_(w.to(self.f_a_w.dtype))
        elif pname == "self_attn.f_b_proj.weight":
            self.f_b_w.copy_(self._rows(w).to(self.f_b_w.dtype))
        elif pname == "self_attn.dt_bias":
            self.dt_bias.copy_(self._rows(w.reshape(-1)).float())
        elif pname == "self_attn.A_log":
            self.A_log.copy_(w.reshape(-1).float()[self.h0:self.h0 + Hl].view(1, 1, Hl, 1))
        elif pname == "self_attn.b_proj.weight":
            self.b_w.copy_(w[self.h0:self.h0 + Hl].to(self.b_w.dtype))
        else:
            raise KeyError(f"TwinStar KDA emitter {self.layer_id}: unexpected tensor {pname}")
        self._loaded.add(pname)

    def emit(self, h: torch.Tensor, fb: ForwardBatch) -> None:
        x = _rms(h, self.norm_w, self.eps)
        mixed = F.linear(x, self.qkv_w)
        f = F.linear(F.linear(x, self.f_a_w), self.f_b_w)  # raw forget gate (T, Hl*Dh): the extend kernel applies the gate
        beta = F.linear(x, self.b_w).float().sigmoid()  # the extend path wants pre-activated beta (stock KimiDeltaAttention)
        self.attn(fb, mixed_qkv=mixed, a=f.unflatten(-1, (-1, self.Dh)).unsqueeze(0), b=beta.unsqueeze(0))  # writes layer_id's slots


class _KDAEmitterGroup:
    """All KDA emitters of the export as one group (docs/31 s4.3, docs/66 s6): per-emitter input_layernorm on the shared
    residual, then the k/v (and q for the conv window) projections as batched GEMMs over the stacked inputs (n, T, D), one
    packed depthwise conv, and the *state-only* half of the image's vendored chunk_kda (gate + cumsum, WY/intra pass,
    inter-chunk state recurrence) over the concatenated n*Hl heads -- the attention-output pass, the q projection over the
    full sequence and its conv are skipped because a state emitter only needs the final SSM state and the conv windows.
    Fresh requests only (prefix 0, every request >= K tokens); otherwise the per-emitter RadixLinearAttention path runs."""

    def __init__(self, emitters: List[_KDAEmitter]):
        e0 = emitters[0]
        self.ems = emitters
        self.ids = [e.layer_id for e in emitters]
        self.n, self.Hl, self.Dh, self.K = len(emitters), e0.Hl, e0.Dh, e0.K
        self.eps = e0.eps
        seg = self.Hl * self.Dh
        self.seg = seg
        # emitter x token budget per fused call: the group materialises k/v (2*seg), the fp32 gate (seg) and the intra-pass
        # w/u/kg (3 x seg) per emitter per token; 8 emitters x 64K tokens OOMed a GB300 at mem-fraction 0.80 (docs/66 s6)
        self.budget = int(os.environ.get("TWINSTAR_EMIT_BUDGET", str(3 * 65536)))
        # Fuse only up to this many tokens per batch: at B=1 the fused group halves the emitter cost (tf16 15.7 -> 8.4 ms),
        # but on 20-65K-token mixed-length burst batches it is slower than the per-layer path and shows autotune-like
        # spikes (docs/66 s6, 787662: x8s 230 ms @21K, 174 ms @65K vs per-layer 99 ms @65K).  Above the cap the group
        # steps aside and the stock per-layer extend kernels run.
        self.fuse_max_t = int(os.environ.get("TWINSTAR_EMIT_FUSE_MAX_T", str(16384)))
        with torch.no_grad():
            self.qkv_w = torch.stack([e.qkv_w for e in emitters]).transpose(1, 2).contiguous()  # (n, D, 3*seg)
            self.f_a_w = torch.stack([e.f_a_w for e in emitters]).transpose(1, 2).contiguous()  # (n, D, Dh)
            self.f_b_w = torch.stack([e.f_b_w for e in emitters]).transpose(1, 2).contiguous()  # (n, Dh, seg)
            self.b_w = torch.stack([e.b_w for e in emitters]).transpose(1, 2).contiguous()  # (n, D, Hl)
            self.conv_w = torch.cat([e.conv_w for e in emitters]).contiguous()  # (n*3*seg, K) fp32
            self.A_log = torch.cat([e.A_log.reshape(-1) for e in emitters]).view(1, 1, self.n * self.Hl, 1).contiguous()
            self.dt_bias = torch.cat([e.dt_bias for e in emitters]).contiguous()  # (n*seg,)
            self.norm_w = torch.stack([e.norm_w for e in emitters]).float().contiguous()  # (n, D)

    def can_fuse(self, fb: ForwardBatch, kb) -> bool:
        if getattr(kb, "state_pruner", None) is not None:
            # The accuracy experiment needs each emitter's exact auxiliary sink
            # recurrence through the regular backend hook.
            return False
        fm = kb.forward_metadata
        if getattr(fm, "has_mamba_track_mask", False):  # radix-cache state tracking: keep the stock per-layer path
            return False
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        return (not bool((fb.extend_prefix_lens > 0).any())) and min(lens) >= self.K and sum(lens) <= self.fuse_max_t

    def emit(self, h: torch.Tensor, fb: ForwardBatch) -> None:
        T = h.shape[0]
        step = max(1, min(self.n, self.budget // max(T, 1)))
        for a in range(0, self.n, step):
            self._emit_sub(h, fb, a, min(self.n, a + step))

    def _emit_sub(self, h: torch.Tensor, fb: ForwardBatch, a: int, b: int) -> None:
        """Emitters [a, b) of the group as one fused call."""
        from sglang.kernels.ops.attention.fla import kda as _K
        from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
        from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

        kb = get_attn_backend().linear_attn_backend  # KDAAttnBackend, planned for fb by init_forward_metadata(fb)
        n, Hl, Dh, seg, Kc = b - a, self.Hl, self.Dh, self.seg, self.K
        T = h.shape[0]
        dev = h.device
        qkv_w, f_a_w, f_b_w, b_w = self.qkv_w[a:b], self.f_a_w[a:b], self.f_b_w[a:b], self.b_w[a:b]
        conv_w = self.conv_w.view(self.n, 3, seg, Kc)[a:b]  # (n, 3, seg, K)
        A_log = self.A_log.view(-1)[a * Hl:b * Hl].view(1, 1, n * Hl, 1).contiguous()
        dt_bias = self.dt_bias[a * seg:b * seg].contiguous()
        hf = h.float()
        X = ((self.norm_w[a:b]).unsqueeze(1) * (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + self.eps))).to(h.dtype)  # (n, T, D)
        kv = torch.bmm(X, qkv_w[:, :, seg:])  # (n, T, 2*seg)
        f = torch.bmm(torch.bmm(X, f_a_w), f_b_w)  # (n, T, seg) raw forget gate
        beta = torch.bmm(X, b_w)  # (n, T, Hl)
        qsl, cache_indices = kb.forward_metadata.query_start_loc, kb.forward_metadata.mamba_cache_indices
        B = cache_indices.shape[0]
        pool = kb.req_to_token_pool
        caches = [pool.mamba2_layer_cache(l) for l in self.ids[a:b]]
        loc = torch.arange(B, device=dev, dtype=cache_indices.dtype)
        # q conv window: the last K-1 raw q rows of every request (conv over a (K-1)-token "sequence" per request writes
        # exactly that window into the state slot; the outputs are discarded)
        ends = qsl[1:].to(torch.long)
        tail_idx = (ends.unsqueeze(1) - torch.arange(Kc - 1, 0, -1, device=dev)).reshape(-1)  # (B*(K-1),) time order
        q_tail = torch.bmm(X[:, tail_idx], qkv_w[:, :, :seg])  # (n, B*(K-1), seg)
        q_all = q_tail.permute(1, 0, 2).reshape(B * (Kc - 1), n * seg)
        conv_q = torch.zeros(B, n * seg, Kc - 1, dtype=caches[0].conv[0].dtype, device=dev)
        qsl_tail = torch.arange(0, B * (Kc - 1) + 1, Kc - 1, device=dev, dtype=qsl.dtype)
        causal_conv1d_fn(q_all.transpose(0, 1), conv_w[:, 0].reshape(n * seg, Kc), None, activation="silu",
                         conv_states=conv_q, has_initial_state=torch.zeros(B, dtype=torch.bool, device=dev), cache_indices=loc,
                         query_start_loc=qsl_tail, seq_lens_cpu=[Kc - 1] * B)
        # k/v conv over the full P stream
        kv_all = kv.permute(1, 0, 2).reshape(T, n * 2 * seg)
        conv_kv = torch.zeros(B, n * 2 * seg, Kc - 1, dtype=caches[0].conv[0].dtype, device=dev)
        kvc = causal_conv1d_fn(kv_all.transpose(0, 1), conv_w[:, 1:].reshape(n * 2 * seg, Kc), None,
                               activation="silu", conv_states=conv_kv, has_initial_state=torch.zeros(B, dtype=torch.bool, device=dev),
                               cache_indices=loc, query_start_loc=qsl, seq_lens_cpu=fb.extend_seq_lens_cpu).transpose(0, 1)
        kvc = kvc.view(T, n, 2, Hl, Dh)
        k = l2norm_fwd(kvc[:, :, 0].reshape(1, T, n * Hl, Dh).contiguous())
        v = kvc[:, :, 1].reshape(1, T, n * Hl, Dh).contiguous()
        g = f.permute(1, 0, 2).reshape(1, T, n * Hl, Dh).contiguous()
        b = beta.permute(1, 0, 2).reshape(1, T, n * Hl).float().sigmoid().contiguous()
        ssm_scr = torch.zeros(B, n * Hl, Dh, Dh, dtype=caches[0].temporal.dtype, device=dev)
        cu = qsl
        chunk_indices = prepare_chunk_indices(cu, 64)
        # Kimi gate: -exp(A_log) * softplus(f + dt_bias), fused with the chunk-local cumsum (lower_bound None; log2 domain)
        g = _K.kda_gate_chunk_cumsum(g, A_log=A_log, chunk_size=64, scale=_K.RCP_LN2, dt_bias=dt_bias, cu_seqlens=cu,
                                     chunk_indices=chunk_indices, lower_bound=None)
        small = B * chunk_indices.shape[0] * (n * Hl) <= 256
        w, u, _, kg, _Aqk, _ = chunk_kda_fwd_intra(q=k, k=k, v=v, gk=g, beta=b, scale=Dh ** -0.5, cu_seqlens=cu, chunk_size=64,
                                                    chunk_indices=chunk_indices, safe_gate=False, fuse_diagonal=small, fuse_recompute=small)
        chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=ssm_scr, initial_state_indices=loc, cu_seqlens=cu,
                                     chunk_indices=chunk_indices, use_exp2=True)
        idx = cache_indices.to(torch.long)
        for j, c in enumerate(caches):
            cs = c.conv[0]  # (slots, K-1, 3*seg)
            cs[idx, :, :seg] = conv_q[:, j * seg:(j + 1) * seg].transpose(-1, -2)
            cs[idx, :, seg:] = conv_kv[:, j * 2 * seg:(j + 1) * 2 * seg].transpose(-1, -2)
            c.temporal[idx] = ssm_scr[:, j * Hl:(j + 1) * Hl]

    def snapshot(self, fb: ForwardBatch):
        kb = get_attn_backend().linear_attn_backend
        idx = kb.forward_metadata.mamba_cache_indices.to(torch.long)
        return [(c.conv[0][idx].clone(), c.temporal[idx].clone()) for c in (kb.req_to_token_pool.mamba2_layer_cache(l) for l in self.ids)]

    def compare(self, fb: ForwardBatch, ref) -> str:
        cur = self.snapshot(fb)
        out = []
        for l, (c0, s0), (c1, s1) in zip(self.ids, ref, cur):
            out.append(f"L{l} conv {(c0.float() - c1.float()).abs().max().item():.2e}/{c0.float().abs().max().item():.2e} "
                       f"ssm {(s0.float() - s1.float()).abs().max().item():.2e}/{s0.float().abs().max().item():.2e}")
        return "; ".join(out)


class _MLAEmitter(nn.Module):
    """Latent emitter for a skipped MLA layer: layer l's trained input_layernorm + kv_a_proj_with_mqa + kv_a_layernorm
    -> (c 512 | k_rot 64) into the MLA pool of layer l (NoPE: no rotary)."""

    EXPECTED = ("input_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight")

    def __init__(self, cfg, layer_id: int, dtype):
        super().__init__()
        self.layer_id = layer_id
        D, self.lora, self.rope = cfg.hidden_size, cfg.kv_lora_rank, cfg.qk_rope_head_dim
        self.norm_w = nn.Parameter(torch.empty(D, dtype=dtype), requires_grad=False)
        self.kv_a_w = nn.Parameter(torch.empty(self.lora + self.rope, D, dtype=dtype), requires_grad=False)
        self.kv_norm_w = nn.Parameter(torch.empty(self.lora, dtype=dtype), requires_grad=False)
        self.eps = cfg.rms_norm_eps
        self._loaded = set()

    @torch.no_grad()
    def load(self, pname: str, w: torch.Tensor):
        m = {"input_layernorm.weight": self.norm_w, "self_attn.kv_a_proj_with_mqa.weight": self.kv_a_w,
             "self_attn.kv_a_layernorm.weight": self.kv_norm_w}
        if pname not in m:
            raise KeyError(f"TwinStar MLA emitter {self.layer_id}: unexpected tensor {pname}")
        m[pname].data.copy_(w.to(m[pname].dtype))
        self._loaded.add(pname)

    def latent(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _rms(h, self.norm_w, self.eps)
        lat = F.linear(x, self.kv_a_w)
        return _rms(lat[:, : self.lora], self.kv_norm_w, self.eps), lat[:, self.lora:]

    def emit(self, h: torch.Tensor, fb: ForwardBatch, target_layer) -> None:
        c, kr = self.latent(h)
        handle = getattr(target_layer.self_attn, "attn_mqa", None) or target_layer.self_attn.attn_mha
        get_token_to_kv_pool().set_mla_kv_buffer(handle, fb.out_cache_loc, c.unsqueeze(1).contiguous(), kr.unsqueeze(1).contiguous())


class _MLAEmitterGroup:
    """All MLA emitters read the same residual: input norms folded into the projections, one GEMM for the group, batched
    per-layer latent norm (weight as broadcast multiply), per-layer pool write (twinstar_glm_moe._EmitterGroup)."""

    def __init__(self, ems: List[_MLAEmitter]):
        self.ems = ems
        self.ids = [e.layer_id for e in ems]
        e0 = ems[0]
        self.lora, self.rope, self.eps = e0.lora, e0.rope, e0.eps
        with torch.no_grad():
            self.W = torch.cat([e.kv_a_w.float() * e.norm_w.float()[None, :] for e in ems], 0).to(e0.kv_a_w.dtype).contiguous()
            self.cw = torch.stack([e.kv_norm_w.float() for e in ems]).to(e0.kv_a_w.dtype).view(1, len(ems), self.lora)
        self.D = self.lora + self.rope

    def emit(self, h: torch.Tensor, fb: ForwardBatch, body) -> None:
        T, G = h.shape[0], len(self.ids)
        xf = h.float()
        x = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(h.dtype)  # unweighted norm; weights folded into W
        lat = F.linear(x, self.W).view(T, G, self.D)
        cf = lat[..., : self.lora].float()
        c = ((cf * torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + self.eps)).to(h.dtype) * self.cw)  # (T, G, lora)
        kr = lat[..., self.lora:]
        pool = get_token_to_kv_pool()
        for i, l in enumerate(self.ids):
            attn = body.layers[l].self_attn
            handle = getattr(attn, "attn_mqa", None) or attn.attn_mha
            pool.set_mla_kv_buffer(handle, fb.out_cache_loc, c[:, i].unsqueeze(1).contiguous(), kr[:, i].unsqueeze(1).contiguous())


# ----------------------------------------------------------------------------- bridge
class _KDABridge(nn.Module):
    """Trained mixer-only copy of layer k-1 (a KDA layer; spec.bridge_kind "mixer", docs/22 D1 / docs/25 br1, #214 arm b)
    run once on P's final residual before the emitters read it: input_layernorm + the full KimiDeltaAttention (q/k/v
    projections + depthwise convs, softplus forget gate, beta, chunk_kda over the whole sequence, gated o_norm, o_proj),
    residual-added.  Port of twinstar_glm5_next._KDABridge without the mHC streams and with Kimi's gate.  Replicated on
    every rank (40M params); transient KDA state (no pool slot) -> a batch with a cached prefix cannot be served shallow
    (see _is_twinstar_prefill).  Mirrors twinstar.kimi_linear.KimiMixerBridge (same parameter names)."""

    _NAMES = {"input_layernorm.weight": "norm_w", "self_attn.f_a_proj.weight": "f_a_w", "self_attn.f_b_proj.weight": "f_b_w",
              "self_attn.dt_bias": "dt_bias", "self_attn.A_log": "A_log", "self_attn.b_proj.weight": "b_w",
              "self_attn.g_a_proj.weight": "g_a_w", "self_attn.g_b_proj.weight": "g_b_w", "self_attn.o_norm.weight": "o_norm_w",
              "self_attn.o_proj.weight": "o_w"}
    _QKV = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight")

    def __init__(self, cfg, layer_id: int, dtype):
        super().__init__()
        la = cfg.linear_attn_config
        self.layer_id = layer_id
        self.H, self.Dh, self.K = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
        D, H, Dh, K = cfg.hidden_size, self.H, self.Dh, self.K
        qkv = H * Dh

        def P(*shape, dt=dtype):
            return nn.Parameter(torch.empty(*shape, dtype=dt), requires_grad=False)

        self.norm_w = P(D)
        self.qkv_w = P(3 * qkv, D)
        self.conv_w = P(3 * qkv, K, dt=torch.float32)
        self.f_a_w, self.f_b_w = P(Dh, D), P(qkv, Dh)
        self.dt_bias, self.A_log = P(qkv, dt=torch.float32), P(H, dt=torch.float32)
        self.b_w = P(H, D)
        self.g_a_w, self.g_b_w = P(Dh, D), P(qkv, Dh)
        self.o_norm_w = P(Dh)
        self.o_w = P(D, qkv)
        self.eps = cfg.rms_norm_eps
        self._loaded = set()

    @property
    def expected(self):
        return set(self._QKV) | set(self._NAMES)

    @torch.no_grad()
    def load(self, pname: str, w: torch.Tensor):
        seg = self.H * self.Dh
        if pname in self._QKV[:3]:
            i = "qkv".index(pname[len("self_attn.")])
            self.qkv_w[i * seg:(i + 1) * seg].copy_(w.to(self.qkv_w.dtype))
        elif pname in self._QKV[3:]:
            i = "qkv".index(pname[len("self_attn.")])
            self.conv_w[i * seg:(i + 1) * seg].copy_(w.reshape(w.shape[0], -1).float())
        elif pname in self._NAMES:
            p = getattr(self, self._NAMES[pname])
            p.data.copy_(w.reshape(p.shape).to(p.dtype))  # A_log arrives as (1, 1, H, 1)
        else:
            raise KeyError(f"TwinStar KDA bridge: unexpected tensor {pname}")
        self._loaded.add(pname)

    def _conv(self, mixed: torch.Tensor, cu: List[int]) -> torch.Tensor:
        out = torch.empty_like(mixed)
        w = self.conv_w.to(mixed.dtype).unsqueeze(1)  # (C, 1, K)
        for a, b in zip(cu[:-1], cu[1:]):
            seg = mixed[a:b].t().unsqueeze(0)  # (1, C, T)
            y = F.conv1d(seg, w, padding=self.K - 1, groups=w.shape[0])[:, :, : b - a]
            out[a:b] = F.silu(y[0]).t()
        return out

    def forward(self, h: torch.Tensor, cu_cpu: List[int]) -> torch.Tensor:
        from sglang.kernels.ops.attention.fla.kda import chunk_kda  # the image's vendored varlen kernel (the engine's own)

        H, Dh = self.H, self.Dh
        T = h.shape[0]
        x = _rms(h, self.norm_w, self.eps)
        mixed = self._conv(F.linear(x, self.qkv_w), cu_cpu)
        q, k, v = mixed.view(1, T, 3, H, Dh).unbind(2)
        f = (F.linear(F.linear(x, self.f_a_w), self.f_b_w).float() + self.dt_bias).view(1, T, H, Dh)
        g = -self.A_log.exp().view(1, 1, H, 1) * F.softplus(f)  # kimi_kda_gate: natural-log decay, no lower bound
        beta = torch.sigmoid(F.linear(x, self.b_w).float()).to(x.dtype).unsqueeze(0)
        cu = torch.tensor(cu_cpu, dtype=torch.int32, device=h.device)
        nb = len(cu_cpu) - 1
        s0 = torch.zeros(nb, H, Dh, Dh, dtype=torch.float32, device=h.device)  # the kernel dereferences the state slots
        core = chunk_kda(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), initial_state=s0,
                         initial_state_indices=torch.arange(nb, device=h.device, dtype=torch.int32),
                         use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        if isinstance(core, tuple):
            core = core[0]
        gate = F.linear(F.linear(x, self.g_a_w), self.g_b_w).view(T, H, Dh)
        cf = core.view(T, H, Dh).float()
        cf = cf * torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + self.eps) * self.o_norm_w.float()  # RMSNormGatedSigmoid
        o = (cf * torch.sigmoid(gate.float())).to(x.dtype).view(T, H * Dh)
        return h + F.linear(o, self.o_w)


# ----------------------------------------------------------------------------- the model class
class KimiLinearForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.config = config
        ts = getattr(config, "twinstar", None)
        from .checkpoint import KimiGeometry, base_model_name, validate_spec
        from .controls import configure_state_dtype, reject_legacy_overrides
        from .config import resolved_server_args
        args = resolved_server_args(get_server_args())
        self.duet_profile = numerics.require_profile(
            "kimi-linear", args, production_supported=True)
        self.duet_code_precision = options.resolve_code_precision(args)
        reject_legacy_overrides()
        self.duet_release = release.release_from_args(
            args, geometry=KimiGeometry(config.to_dict()),
            base_model=base_model_name(config.to_dict(), args.model_path,
                                       release.resolve_release_dir(args.duet_release)),
            model="kimi-linear",
            hf_root=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
        self.duet_report = None
        self.duet_options = None
        if self.duet_release:
            if ts is not None:
                raise ValueError("cannot combine a legacy TwinStar export with a HF DUET release")
            if quant_config is not None:
                raise ValueError("HF Kimi DUET functional mode requires the unchanged bf16 base")
            spec = self.duet_release.spec
            validate_spec(spec, config.to_dict())
            self.duet_report = dict(self.duet_release.audit, spec=spec,
                                    sha256=self.duet_release.sha256, verified=True)
            self.duet_options = options.DuetOptions.resolve(
                spec, args, state_dim=config.linear_attn_config["head_dim"])
            configure_state_dtype(args, self.duet_options, config=config)
            ts = {"p_layers": list(range(spec["prefill_depth"])),
                  "d_layers": list(range(config.num_hidden_layers)),
                  "emitters": list(range(spec["prefill_depth"], config.num_hidden_layers))}
        # All controls off keeps the registered DUET class but delegates the
        # unchanged base: no codec, emitters, state policy, or dtype override.
        self.duet_active = bool(self.duet_options and (
            self.duet_options.prefill_layer_trim or self.duet_options.decode_ssm_r))
        if self.duet_active and not args.disable_radix_cache:
            raise ValueError("Kimi DUET requires --disable-radix-cache until prefix restoration is implemented")
        if self.duet_active and not args.disable_prefill_cuda_graph:
            raise ValueError("Kimi DUET requires --disable-prefill-cuda-graph; production uses a private emitter graph")
        if self.duet_options is not None and not self.duet_options.prefill_layer_trim:
            ts = None
        self.model = _stock.KimiLinearForCausalLM(config, quant_config, prefix)
        self.twinstar = ts
        self.n_layers = config.num_hidden_layers
        self.emitters = nn.ModuleDict()
        self.bridges = nn.ModuleList()  # also for the stock mode (no twinstar section): load_weights iterates it
        self.bridge_n, self.bridge_kind = 0, "layer"
        self.strict = self.duet_active
        self.dump_dir = os.environ.get("TWINSTAR_SGL_DUMP") or None
        self.profile = os.environ.get("TWINSTAR_PROFILE", "0") == "1"
        self.boundary_mode = os.environ.get("TWINSTAR_BOUNDARY", "graph")
        assert self.boundary_mode in ("extend", "graph"), f"TWINSTAR_BOUNDARY={self.boundary_mode}"
        self.boundary_m = max(1, int(os.environ.get("TWINSTAR_BOUNDARY_M", "3")))  # the 3 chat-template anchor tokens
        self.emit_group = os.environ.get("TWINSTAR_EMIT_GROUP", "1") == "1"
        self.mla_group: Optional[_MLAEmitterGroup] = None
        # TWINSTAR_EMIT_FUSED: state (default) = state-only fused KDA emitter group | 0 = per-emitter RadixLinearAttention
        # extend | checkstate = per-emitter path first, then the fused group, log the pool-state difference
        self.emit_fused = os.environ.get("TWINSTAR_EMIT_FUSED", "state")
        assert self.emit_fused in ("state", "0", "checkstate"), f"TWINSTAR_EMIT_FUSED={self.emit_fused}"
        self.kda_group: Optional[_KDAEmitterGroup] = None
        self._model_runner = None  # set by the runner warm-up (prepare_before_cuda_graph_capture); owns the decode graph
        self._dump_n = 0
        self.n_fallback = self.n_twinstar = self.n_prefix = self.n_graph_fallback = 0
        self.tp_rank = get_parallel().tp_rank
        self.capture_aux_hidden_states = False
        if self.duet_active:
            from sglang.srt.duet.latent_codec import ResidualCode
            from .runtime import EmitterGraph, make_pruner
            if get_parallel().tp_size != 1:
                raise ValueError("HF Kimi DUET functional mode currently requires TP1")
            self.strict = True
            self.boundary_mode, self.boundary_m = "duet-decode", 1
            self.emit_group, self.emit_fused = False, "0"
            self.emitter_graph = EmitterGraph() if self.duet_profile == "production" else None
            if self.duet_options.prefill_layer_trim:
                self.latent = ResidualCode(
                    config.hidden_size, spec, tf32=self.duet_code_precision == "tf32")
            la = config.linear_attn_config
            self.register_buffer("duet_sink_dir", torch.empty(
                config.num_hidden_layers, la["num_heads"], la["head_dim"], dtype=torch.float32))
            if self.duet_options.decode_ssm_r > 0:
                self.kda_state_pruner_factory = partial(make_pruner, self)
        if ts is None:
            logger.info("TwinStar Kimi Linear class without a twinstar config section: stock behaviour")
            return
        assert get_pp_group().world_size == 1, "TwinStar: pipeline parallelism not supported"
        n = self.n_layers
        self.p_layer_ids = sorted(int(x) for x in ts["p_layers"])
        self.emitter_ids = sorted(int(x) for x in ts.get("emitters", []))
        assert sorted(int(x) for x in ts["d_layers"]) == list(range(n)), "D must be the full stock layer stack"
        k = len(self.p_layer_ids)
        assert self.p_layer_ids == list(range(k)) and self.emitter_ids == list(range(k, n)), \
            "shared form expected: P = layers 0..k-1, emitters for k..n-1"
        self.k = k
        dtype = torch.get_default_dtype()
        for l in self.emitter_ids:
            if self.duet_report is not None:
                from .runtime import DuetKDAEmitter, DuetMLAEmitter
                emitter_cls = DuetKDAEmitter if _is_kda(config, l) else DuetMLAEmitter
                self.emitters[str(l)] = emitter_cls(config, l, torch.float32)
                self.emitters[str(l)].prune_state = self.duet_options.decode_ssm_r > 0
                self.emitters[str(l)].require_fp32_state = self.duet_profile == "reference"
            else:
                self.emitters[str(l)] = (_KDAEmitter if _is_kda(config, l) else _MLAEmitter)(config, l, dtype)
        self.bridge_n = int(ts.get("bridge", 0) or 0)
        self.bridge_kind = ts.get("bridge_kind", "layer") or "layer"
        if self.bridge_n:
            assert self.bridge_kind == "mixer", f"bridge_kind {self.bridge_kind!r} is not served by the Kimi Linear class (mixer only)"
            assert _is_kda(config, k - 1), "the mixer bridge is implemented for a KDA layer k-1"
            for _ in range(self.bridge_n):
                self.bridges.append(_KDABridge(config, k - 1, dtype))
        kinds = ["KDA" if isinstance(self.emitters[str(l)], _KDAEmitter) else "MLA" for l in self.emitter_ids]
        logger.info("TwinStar Kimi Linear: P layers 0..%d, emitters %d (%d KDA + %d MLA), bridge %d x %s, tp %d, boundary %s m=%d, "
                    "KDA emit %s, MLA group %s", k - 1, len(self.emitter_ids), kinds.count("KDA"), kinds.count("MLA"), self.bridge_n,
                    self.bridge_kind, get_parallel().tp_size, self.boundary_mode, self.boundary_m, self.emit_fused, self.emit_group)

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

    def prepare_before_cuda_graph_capture(self, model_runner):
        self._model_runner = model_runner
        f = getattr(self.model, "prepare_before_cuda_graph_capture", None)
        if f is not None:
            f(model_runner)

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
        if self.duet_options is not None and not self.duet_options.prefill_layer_trim:
            return False
        if self.twinstar is None or get_is_capture_mode():
            return False
        fm = fb.forward_mode
        if not fm.is_extend() or fm.is_target_verify() or fm.is_draft_extend_v2() or fm.is_mixed():
            return False
        if fb.extend_prefix_lens_cpu is None or fb.extend_seq_lens_cpu is None or fb.spec_info is not None:
            return False
        if getattr(fb, "can_run_tbo", False) or fb.input_embeds is not None:
            return False
        if min(fb.extend_seq_lens_cpu) < 1:
            return False
        if self.duet_report is not None:
            if ((self.duet_profile == "reference" and fb.batch_size != 1)
                    or max(fb.extend_prefix_lens_cpu) > 0):
                raise ValueError("HF DUET functional prefill requires B1, radix off, and an unchunked prompt")
            # The reference's T=1 prefill runs the entire base once (no cut).
            if max(fb.extend_seq_lens_cpu) == 1:
                return False
        if self.bridge_n and max(fb.extend_prefix_lens_cpu) > 0:
            msg = (f"TwinStar: bridge layers have no cached state; extend batch with prefix {fb.extend_prefix_lens_cpu} "
                   f"-> stock forward (run the bridge export with the radix cache off and chunked_prefill_size >= the longest prompt)")
            if self.strict:
                raise RuntimeError(msg)
            self.n_fallback += 1
            if self.n_fallback <= 3 or self.n_fallback % 100 == 0:
                logger.warning("%s [count %d]", msg, self.n_fallback)
            return False
        return True

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,
                input_embeds: torch.Tensor = None, pp_proxy_tensors=None, **kwargs):
        if input_embeds is None and pp_proxy_tensors is None and self._is_twinstar_prefill(forward_batch):
            hidden = self._twinstar_prefill(input_ids, positions, forward_batch)
            if isinstance(hidden, LogitsProcessorOutput):  # graph boundary: the decode runner already applied lm_head
                out = hidden
            else:
                out = self.model.logits_processor(input_ids, hidden, self.model.lm_head, forward_batch)
            mode = "prefill"
        else:
            out = self.model.forward(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)
            mode = "fallback" if forward_batch.forward_mode.is_extend() else "decode"
        if (self.dump_dir is not None and getattr(out, "next_token_logits", None) is not None and not get_is_capture_mode()
                and self.tp_rank == 0):
            self._dump(mode, input_ids, positions, forward_batch, out.next_token_logits)
        return out

    def _dump(self, mode, input_ids, positions, fb, logits):
        os.makedirs(self.dump_dir, exist_ok=True)
        rec = {"mode": mode, "input_ids": input_ids.cpu(), "positions": positions.cpu(),
               "extend_seq_lens": list(fb.extend_seq_lens_cpu) if fb.extend_seq_lens_cpu is not None else None,
               "extend_prefix_lens": list(fb.extend_prefix_lens_cpu) if fb.extend_prefix_lens_cpu is not None else None,
               "seq_lens": fb.seq_lens.cpu(), "req_pool_indices": fb.req_pool_indices.cpu(),
               "logits": logits.float().cpu()}
        torch.save(rec, os.path.join(self.dump_dir, f"{self._dump_n:06d}.pt"))
        self._dump_n += 1

    # ------------------------------------------------------------------ batch decomposition
    def _boundary_form(self, fb: ForwardBatch) -> Tuple[str, List[int]]:
        """(form, boundary tokens per request): graph = m DECODE steps through the decode CUDA graph runner (m =
        TWINSTAR_BOUNDARY_M, capped by the request's extend length - 1 so every request keeps >= 1 P-stream token);
        extend = one eager extend of the last token."""
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        if self.duet_report is not None:
            return "duet-decode", [1] * len(lens)
        plain = fb.spec_info is None and not fb.return_logprob
        if self.boundary_mode == "graph" and plain and self._decode_graph_runner() is not None:
            ms = [min(self.boundary_m, l - 1) for l in lens]
            if all(m >= 1 for m in ms):
                return "graph", ms
        if self.boundary_mode == "graph" and plain and not getattr(self, "_warned_no_graph", False):
            self._warned_no_graph = True
            logger.warning("TwinStar boundary: TWINSTAR_BOUNDARY=graph requested but no decode CUDA graph runner is available "
                           "(graphs disabled / disaggregation prefill worker without --cuda-graph-backend-decode full): "
                           "falling back to the eager extend boundary")
        return "extend", [1] * len(lens)

    def _sub_batch(self, fb: ForwardBatch, input_ids, positions, ms: List[int], which: str):
        """Derived extend batch: which='p' = every request's extend tokens except its last m (requests whose whole extend
        is the boundary chunk are dropped); which='b' = every request's last m tokens with prefix = seq_len - m."""
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        prefix = [int(x) for x in fb.extend_prefix_lens_cpu]
        starts = [0] + list(itertools.accumulate(lens))
        dev = input_ids.device
        if which == "p":
            sel = [r for r, l in enumerate(lens) if l - ms[r] > 0]
            idx_cpu = torch.cat([torch.arange(starts[r], starts[r] + lens[r] - ms[r]) for r in sel]) if sel else torch.zeros(0, dtype=torch.long)
            new_lens = [lens[r] - ms[r] for r in sel]
            new_prefix = [prefix[r] for r in sel]
        else:
            sel = list(range(len(lens)))
            idx_cpu = torch.cat([torch.arange(starts[r] + lens[r] - ms[r], starts[r] + lens[r]) for r in sel])
            new_lens = [ms[r] for r in sel]
            new_prefix = [prefix[r] + lens[r] - ms[r] for r in sel]
        idx = idx_cpu.to(dev, non_blocking=True)
        sel_t = torch.tensor(sel, dtype=torch.long)
        nb = copy.copy(fb)
        nb.batch_size = len(sel)
        nb.input_ids = input_ids[idx]
        nb.positions = positions[idx]
        nb.req_pool_indices = fb.req_pool_indices[sel_t.to(dev)]
        seq_new = [p + l for p, l in zip(new_prefix, new_lens)]
        nb.seq_lens = torch.tensor(seq_new, dtype=fb.seq_lens.dtype, device=dev)
        nb.seq_lens_cpu = torch.tensor(seq_new, dtype=fb.seq_lens_cpu.dtype if fb.seq_lens_cpu is not None else torch.int64)
        nb.seq_lens_sum = int(sum(seq_new))
        if fb.orig_seq_lens is not None:
            nb.orig_seq_lens = fb.orig_seq_lens[sel_t.to(dev)]
        nb.out_cache_loc = fb.out_cache_loc[idx]
        if getattr(fb, "out_cache_loc_virtual", None) is not None:
            nb.out_cache_loc_virtual = fb.out_cache_loc_virtual[idx]
        nb.extend_num_tokens = int(idx_cpu.numel())
        nb.extend_seq_lens = torch.tensor(new_lens, dtype=fb.extend_seq_lens.dtype, device=dev)
        nb.extend_prefix_lens = torch.tensor(new_prefix, dtype=fb.extend_prefix_lens.dtype, device=dev)
        nb.extend_start_loc = torch.tensor([0] + list(itertools.accumulate(new_lens))[:-1], dtype=fb.extend_start_loc.dtype,
                                           device=dev) if fb.extend_start_loc is not None else None
        nb.extend_seq_lens_cpu = new_lens
        nb.extend_prefix_lens_cpu = new_prefix
        if fb.extend_logprob_start_lens_cpu is not None:
            nb.extend_logprob_start_lens_cpu = [0] * len(sel)
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, t[sel_t.to(t.device)])
        if getattr(fb, "global_num_token_non_padded_cpu", None) is not None:
            nb.global_num_token_non_padded_cpu = nb.extend_num_tokens
        if getattr(fb, "global_num_token_non_padded", None) is not None:
            nb.global_num_token_non_padded = torch.tensor(nb.extend_num_tokens, dtype=fb.global_num_token_non_padded.dtype, device=dev)
        if getattr(fb, "num_token_non_padded", None) is not None:
            nb.num_token_non_padded = torch.tensor(nb.extend_num_tokens, dtype=fb.num_token_non_padded.dtype, device=dev)
        nb.forward_metadata_ready = False
        nb.forward_metadata_planned_bs = None
        nb.forward_metadata_planned_num_tokens = None
        nb.mm_inputs = None
        nb.input_embeds = None
        return nb, idx

    def _twinstar_prefill(self, input_ids: torch.Tensor, positions: torch.Tensor, fb: ForwardBatch):
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        T = input_ids.shape[0]
        if sum(lens) != T:
            raise RuntimeError(f"TwinStar: padded extend batch (sum lens {sum(lens)} != {T} tokens); run the worker with "
                               f"--disable-prefill-cuda-graph")
        t0 = time.time() if self.profile else None
        bk = get_attn_backend()
        body = self.model.model  # KimiLinearModel
        rec = get_global_expert_distribution_recorder()
        form, ms = self._boundary_form(fb)
        fb1, p_idx = self._sub_batch(fb, input_ids, positions, ms, "p")
        prof = {}

        def tick(name, t_start):
            if self.profile:
                torch.cuda.synchronize()
                now = time.time()
                prof[name] = prof.get(name, 0.0) + (now - t_start) * 1e3
                return now
            return t_start

        with get_attn_tp_context().maybe_input_scattered(fb1):
            if fb1.batch_size > 0:
                ts = time.time() if self.profile else None
                bk.init_forward_metadata(fb1)
                ts = tick("meta1", ts)
                za = BumpAllocator(buffer_size=self.n_layers * 2, dtype=torch.float32, device=input_ids.device)
                hidden, residual = body.embed_tokens(fb1.input_ids), None
                # Stock fused residual norms can update the residual in-place.
                base_embeddings = hidden.clone() if self.duet_report is not None else None
                for l in self.p_layer_ids:
                    with rec.with_current_layer(l):
                        hidden, residual = body.layers[l](positions=fb1.positions, hidden_states=hidden, forward_batch=fb1,
                                                          residual=residual, zero_allocator=za)
                h = hidden if residual is None else hidden + residual
                if self.duet_report is not None:
                    with code_precision(self.duet_code_precision):
                        h = self.latent(h, base_embeddings)
                ts = tick("P%d" % len(self.p_layer_ids), ts)
                if self.bridges:  # trained mixer copies of layer k-1 on the final residual, once, before every emitter
                    cu = [0] + list(itertools.accumulate(fb1.extend_seq_lens_cpu))
                    for br in self.bridges:
                        h = br(h, cu)
                    ts = tick("bridge", ts)
                if self.emit_group and self.mla_group is not None:
                    self.mla_group.emit(h, fb1, body)
                    ts = tick("emitMLAx%d" % len(self.mla_group.ids), ts)
                fused_kda = self.kda_group is not None and self.emit_fused != "0" and self.kda_group.can_fuse(fb1, bk.linear_attn_backend)
                if fused_kda and self.emit_fused == "checkstate":  # per-emitter path first, then the fused group; compare the pool states
                    for e in self.kda_group.ems:
                        e.emit(h, fb1)
                    ref = self.kda_group.snapshot(fb1)
                    ts = tick("emitKDA(unfused)", ts)
                if fused_kda:
                    self.kda_group.emit(h, fb1)
                    ts = tick("emitKDAx%ds" % len(self.kda_group.ids), ts)
                    if self.emit_fused == "checkstate":
                        logger.warning("TwinStar fused-KDA check (max|d|/max|ref|): %s", self.kda_group.compare(fb1, ref))  # opt-in check: visible at warning
                graph_emit = self.duet_active and self.emitter_graph is not None
                if graph_emit:
                    def emit(hidden, batch):
                        for layer in self.emitter_ids:
                            emitter = self.emitters[str(layer)]
                            if isinstance(emitter, _MLAEmitter):
                                emitter.emit(hidden, batch, body.layers[layer])
                            else:
                                emitter.emit(hidden, batch)
                    self.emitter_graph.run(self, h, fb1, emit)
                for l in ([] if graph_emit else self.emitter_ids):
                    em = self.emitters[str(l)]
                    if isinstance(em, _MLAEmitter):
                        if self.emit_group and self.mla_group is not None:
                            continue
                        em.emit(h, fb1, body.layers[l])
                    else:
                        if fused_kda:
                            continue
                        em.emit(h, fb1)
                    if self.profile:
                        ts = tick("emitKDA" if isinstance(em, _KDAEmitter) else "emitMLA", ts)
        if self.profile:
            torch.cuda.synchronize()
            t1 = time.time()
        pruner = getattr(bk.linear_attn_backend, "state_pruner", None)
        boundary_context = (pruner.prefill_boundary(fb)
                            if pruner is not None and self.duet_report is None else nullcontext())
        with boundary_context:
            if form == "duet-decode":
                # Prefix cuts run before this first decode, and this boundary
                # counts as step 1 of W, exactly as Kimi.prefill in the reference.
                tok = torch.tensor([n - 1 for n in itertools.accumulate(lens)], dtype=torch.long)
                fbd = self._decode_batch(fb, input_ids, positions, tok, list(range(len(lens))), lens)
                bk.init_forward_metadata(fbd)
                if any(length == 1 for length in lens):
                    # Unlike EXTEND with prefix=0, DECODE reads an existing
                    # state. Fresh one-token requests must not read a reused slot.
                    kb = bk.linear_attn_backend
                    single = torch.tensor([i for i, length in enumerate(lens) if length == 1],
                                          device=fbd.req_pool_indices.device)
                    slots = kb.forward_metadata.mamba_cache_indices[single].long()
                    cache = kb.req_to_token_pool.mamba_pool.mamba_cache
                    cache.temporal[:, slots] = 0
                    for conv in cache.conv:
                        conv[:, slots] = 0
                out = self.model.forward(fbd.input_ids, fbd.positions, fbd)
                # T=1 requests in a mixed batch have just completed their full
                # prompt. Their first cut is pending and their decode clock is 0.
                if pruner is not None and any(length == 1 for length in lens):
                    single = torch.tensor([i for i, length in enumerate(lens) if length == 1],
                                          device=fbd.req_pool_indices.device)
                    slots = pruner.slots(fbd)[single]
                    pruner.pending_prefix[:, slots] = True
                    pruner.count[:, slots] = 0
            elif form == "graph":
                out = self._boundary_graph(input_ids, positions, fb, ms)
            else:
                fb2, b_idx = self._sub_batch(fb, input_ids, positions, ms, "b")
                bk.init_forward_metadata(fb2)
                with get_attn_tp_context().maybe_input_scattered(fb2):
                    hb = body(fb2.input_ids, fb2.positions, fb2)
                if isinstance(hb, tuple):
                    hb = hb[0]
                out = hb.new_zeros(T, hb.shape[-1])
                out[b_idx] = hb
        try:
            bk.init_forward_metadata(fb)  # leave the backend planned for the original batch
        except Exception:  # one crash seen (787879 eo19 GPQA, flashinfer MLAPlan qo_indptr < 0): dump the batch shape, then re-raise
            logger.error("TwinStar: re-plan of the original batch failed: bs=%d lens=%s prefix=%s seq_lens=%s ms=%s form=%s "
                         "seq_lens_cpu=%s", fb.batch_size, list(fb.extend_seq_lens_cpu), list(fb.extend_prefix_lens_cpu),
                         fb.seq_lens.tolist(), ms, form, None if fb.seq_lens_cpu is None else fb.seq_lens_cpu.tolist())
            raise
        self.n_twinstar += 1
        if any(p > 0 for p in fb.extend_prefix_lens_cpu):
            self.n_prefix += 1
        if self.profile:
            torch.cuda.synchronize()
            parts = " ".join(f"{k} {v:.1f}" for k, v in prof.items())
            logger.warning("TwinStar prefill: B=%d T=%d T_p=%d  P+emit %.1f ms [%s]  boundary[%s m=%s] %.1f ms", len(lens), T,  # opt-in profile: visible at the default warning level
                        int(p_idx.numel()), (t1 - t0) * 1e3, parts, form, ms, (time.time() - t1) * 1e3)
        if self.n_twinstar % 500 == 1:
            logger.info("TwinStar counters: shallow prefills %d (with prefix %d), stock fallbacks %d, graph-boundary fallbacks %d",
                        self.n_twinstar, self.n_prefix, self.n_fallback, self.n_graph_fallback)
        return out

    # ------------------------------------------------------------------ graph form of chunk 2
    def _decode_batch(self, fb: ForwardBatch, input_ids, positions, tok_idx_cpu: torch.Tensor, sel: List[int],
                      seq_new: List[int]) -> ForwardBatch:
        """A one-token-per-request DECODE batch for requests `sel` of the extend batch `fb`: token tok_idx[r] of the flat
        extend, sequence length seq_new[r] after it.  Mirrors ForwardBatch.init_new for decode (extend-only fields None)."""
        dev = input_ids.device
        idx = tok_idx_cpu.to(dev, non_blocking=True)
        sel_t = torch.tensor(sel, dtype=torch.long)
        nb = copy.copy(fb)
        nb.forward_mode = ForwardMode.DECODE
        nb.batch_size = len(sel)
        nb.input_ids = input_ids[idx]
        nb.positions = positions[idx]
        nb.req_pool_indices = fb.req_pool_indices[sel_t.to(dev)]
        nb.seq_lens = torch.tensor(seq_new, dtype=fb.seq_lens.dtype, device=dev)
        nb.seq_lens_cpu = torch.tensor(seq_new, dtype=fb.seq_lens_cpu.dtype if fb.seq_lens_cpu is not None else torch.int64)
        nb.seq_lens_sum = int(sum(seq_new))
        if fb.orig_seq_lens is not None:
            nb.orig_seq_lens = fb.orig_seq_lens[sel_t.to(dev)]
        nb.out_cache_loc = fb.out_cache_loc[idx]
        for name in ("out_cache_loc_virtual", "out_cache_loc_dsv4"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, t[idx])
        nb.extend_num_tokens = None
        nb.extend_seq_lens = nb.extend_prefix_lens = nb.extend_start_loc = None
        nb.extend_seq_lens_cpu = nb.extend_prefix_lens_cpu = nb.extend_logprob_start_lens_cpu = None
        nb.is_extend_in_batch = False
        nb.can_run_decode_cuda_graph = True
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, t[sel_t.to(t.device)])
        for name in ("global_num_token_non_padded_cpu",):
            if getattr(fb, name, None) is not None:
                setattr(nb, name, len(sel))
        for name in ("global_num_token_non_padded", "num_token_non_padded"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, torch.tensor(len(sel), dtype=t.dtype, device=dev))
        nb.forward_metadata_ready = False
        nb.forward_metadata_planned_bs = None
        nb.forward_metadata_planned_num_tokens = None
        nb.mm_inputs = None
        nb.input_embeds = None
        return nb

    def _boundary_graph(self, input_ids, positions, fb: ForwardBatch, ms: List[int]) -> LogitsProcessorOutput:
        """Chunk 2 as DECODE steps through the stock decode CUDA graph runner: step i feeds every request's (i+1)-th
        boundary token (requests with fewer boundary tokens drop out); each request's logits come from its last step."""
        runner = self._decode_graph_runner()
        bk = get_attn_backend()
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        prefix = [int(x) for x in fb.extend_prefix_lens_cpu]
        starts = [0] + list(itertools.accumulate(lens))
        logits = None
        for i in range(max(ms)):
            sel = [r for r in range(len(lens)) if ms[r] > i]
            tok = torch.tensor([starts[r] + lens[r] - ms[r] + i for r in sel], dtype=torch.long)
            seq_new = [prefix[r] + lens[r] - ms[r] + i + 1 for r in sel]
            fbd = self._decode_batch(fb, input_ids, positions, tok, sel, seq_new)
            if runner.can_run_graph(fbd):
                out = runner.execute(fbd)
            else:  # bs outside the captured range etc.: same DECODE step, eager
                self.n_graph_fallback += 1
                bk.init_forward_metadata(fbd)
                out = self.model.forward(fbd.input_ids, fbd.positions, fbd)
            nl = out.next_token_logits
            if logits is None:
                logits = nl.new_empty(len(lens), nl.shape[-1])
            last = [j for j, r in enumerate(sel) if ms[r] == i + 1]
            if last:
                logits[torch.tensor([sel[j] for j in last], device=nl.device)] = nl[torch.tensor(last, device=nl.device)]
        return LogitsProcessorOutput(next_token_logits=logits)

    # ------------------------------------------------------------------ weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        inherited = {}
        if self.duet_active and self.duet_options.prefill_layer_trim:
            from .checkpoint import frozen_q_contract
            inherited = frozen_q_contract(self.duet_report["spec"], self.config.to_dict())
        def body_weights():
            for name, w in weights:
                if name in inherited:
                    target, shape = inherited[name]
                    if tuple(w.shape) != shape:
                        raise ValueError(f"frozen base q shape mismatch: {name}")
                    l, pname = target[len(_E_PREFIX):].split(".", 1)
                    self.emitters[l].load(pname, w)
                if name.startswith(_E_PREFIX):
                    l, pname = name[len(_E_PREFIX):].split(".", 1)
                    if l in self.emitters:
                        self.emitters[l].load(pname, w)
                    continue
                if name.startswith(_B_PREFIX):
                    j, pname = name[len(_B_PREFIX):].split(".", 1)
                    if int(j) >= len(self.bridges):
                        raise KeyError(f"TwinStar: bridge tensor {name} in the checkpoint but the config declares bridge={len(self.bridges)}")
                    self.bridges[int(j)].load(pname, w)
                    continue
                yield name, w

        self.model.load_weights(body_weights())
        if self.duet_active:
            from safetensors import safe_open
            from .checkpoint import tensor_contract
            contract = tensor_contract(self.duet_report["spec"], self.config.to_dict())
            latent_weights = {}
            with component_upload(self.duet_profile, self.duet_sink_dir.device) as upload, safe_open(
                str(Path(self.duet_release.path) / "duet_components.safetensors"), framework="pt", device="cpu"
            ) as f:
                for source, (target, _) in contract.items():
                    tensor = upload(f.get_tensor(source))
                    if target.startswith(_E_PREFIX):
                        l, pname = target[len(_E_PREFIX):].split(".", 1)
                        if l in self.emitters:
                            self.emitters[l].load(pname, tensor)
                    elif source.startswith("latent."):
                        latent_weights[source[len("latent."):]] = tensor
                    elif source == "state.sink_dir":
                        self.duet_sink_dir.copy_(tensor)
                if self.duet_options.prefill_layer_trim:
                    self.latent.load_state_dict(latent_weights, strict=True)
            logger.info("HF DUET loaded: %s tensors=%d sha256=%s adapter=%s", self.duet_report["spec"]["name"],
                        self.duet_report["tensor_count"], self.duet_report["sha256"], __name__)
        for l, e in self.emitters.items():
            miss = [k for k in e.EXPECTED if k not in e._loaded]
            if miss:
                raise RuntimeError(f"TwinStar: emitter {l} tensors not in checkpoint: {miss}")
        for j, b in enumerate(self.bridges):
            miss = sorted(b.expected - b._loaded)
            if miss:
                raise RuntimeError(f"TwinStar: bridge {j} tensors not in checkpoint: {miss}")
        mla = [self.emitters[str(l)] for l in getattr(self, "emitter_ids", []) if isinstance(self.emitters[str(l)], _MLAEmitter)]
        if self.twinstar is not None and self.emit_group and mla:
            self.mla_group = _MLAEmitterGroup(mla)
            logger.info("TwinStar: folded MLA emitter group over layers %s", self.mla_group.ids)
        kda = [self.emitters[str(l)] for l in getattr(self, "emitter_ids", []) if isinstance(self.emitters[str(l)], _KDAEmitter)]
        if self.twinstar is not None and self.emit_fused != "0" and kda:
            self.kda_group = _KDAEmitterGroup(kda)
            logger.info("TwinStar: fused state-only KDA emitter group over layers %s (%d heads/rank, mode %s)", self.kda_group.ids,
                        self.kda_group.n * kda[0].Hl, self.emit_fused)


# class attributes the loader / quant config read off the model class: copied from the stock class when present
for _a in ("hf_to_sglang_mapper", "packed_modules_mapping"):
    if hasattr(_stock.KimiLinearForCausalLM, _a):
        setattr(KimiLinearForCausalLM, _a, getattr(_stock.KimiLinearForCausalLM, _a))

EntryClass = [KimiLinearForCausalLM]
