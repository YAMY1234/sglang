"""Flash-Next mixer bridge dependencies, preserving the original tensor arithmetic."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

FP32_PARAM_SUFFIXES = ("A_log", "dt_bias")
QWEN4_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")
GDNState = Tuple[torch.Tensor, torch.Tensor]
_GDN_DECODE = "torch"
_GDN_IMPL = "fla"

class RMSNormGated(nn.Module):
    """w * x_hat * silu(gate)  (norm before gate; HF Qwen3_5RMSNormGated)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * xf.to(dt)
        h = h * F.silu(gate.float())
        return h.to(dt)


def _to_dev(t, ref: torch.Tensor):
    """Move t to ref's device (multi-GPU layer placement via load_hybrid(devices=...)); no-op when already there."""
    return t if t is None or t.device == ref.device else t.to(ref.device)


def _fla():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def _torch_chunk_gdn(q, k, v, g, beta, initial_state, chunk_size: int = 64):
    """fp32 port of HF `torch_chunk_gated_delta_rule` (l2-normalised q/k, scale 1/sqrt(Dk), UT transform per chunk).
    q, k (B, T, H, Dk), v (B, T, H, Dv), g (B, T, H) log-decay, beta (B, T, H) -> out (B, T, H, Dv), S (B, H, Dk, Dv) fp32."""
    dt = v.dtype
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    q = q * (Dk ** -0.5)
    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    n_chunks = (T + pad) // chunk_size
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = [x.reshape(B, H, n_chunks, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta)]
    g = g.reshape(B, H, n_chunks, chunk_size)
    dev = q.device
    strict = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=dev).triu(1)
    cum = g.cumsum(dim=3)
    pair = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(strict, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pair
    intra = (q @ k.transpose(-1, -2)) * pair
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)
    new_v = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut, decayed_k_beta, upper=False, unitriangular=True)
    S = torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=dev) if initial_state is None else initial_state.float().to(dev)
    out = torch.zeros_like(new_v)
    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    for i in range(n_chunks):
        v_new = new_v[:, :, i] - k_cumdecay[:, :, i] @ S
        out[:, :, i] = q[:, :, i] @ S + intra[:, :, i] @ v_new
        S = S * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).contiguous().to(dt)
    return out, S


def _torch_recurrent_step(q, k, v, g, beta, s0):
    """One gated-delta-rule step in plain torch (HF torch_recurrent_gated_delta_rule, T=1).
    q,k (B,1,H,Dk), v (B,1,H,Dv), g (B,1,H) fp32, beta (B,1,H), s0 (B,H,Dk,Dv) fp32."""
    qf, kf, vf = q[:, 0].float(), k[:, 0].float(), v[:, 0].float()  # (B, H, D)
    qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
    kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    qf = qf * (qf.shape[-1] ** -0.5)
    gt = g[:, 0].exp()[:, :, None, None]  # (B, H, 1, 1)
    bt = beta[:, 0].float()[:, :, None]  # (B, H, 1)
    s = s0 * gt
    kv_mem = (s * kf[:, :, :, None]).sum(-2)  # (B, H, Dv)
    delta = (vf - kv_mem) * bt
    s = s + kf[:, :, :, None] * delta[:, :, None, :]
    out = (s * qf[:, :, :, None]).sum(-2)  # (B, H, Dv)
    return out.to(v.dtype)[:, None], s


def gdn_mix(m: nn.Module, cfg: HybridConfig, x: torch.Tensor, state: Optional[GDNState], want_output: bool):
    """Gated DeltaNet mixer on the *normed* input x (B, T, H).

    m provides in_proj_qkv / in_proj_b / in_proj_a / conv1d / dt_bias / A_log (and in_proj_z / norm /
    out_proj when want_output).  Returns (output or None, new_state)."""
    B, T, _ = x.shape
    K = cfg.linear_conv_kernel_dim
    Hk, Hv, Dk, Dv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    qkv = m.in_proj_qkv(x).transpose(1, 2)  # (B, C, T)
    conv_in = qkv if state is None else torch.cat([state[0].to(qkv.dtype), qkv], dim=-1)
    new_conv = conv_in[:, :, -(K - 1):]
    y = F.silu(m.conv1d(conv_in)[:, :, : conv_in.shape[-1]])  # causal conv (left padding k-1)
    y = y[:, :, -T:].transpose(1, 2)  # (B, T, C)
    q, k, v = torch.split(y, [cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1)
    q = q.reshape(B, T, Hk, Dk)
    k = k.reshape(B, T, Hk, Dk)
    v = v.reshape(B, T, Hv, Dv)
    beta = m.in_proj_b(x).sigmoid()
    g = -m.A_log.float().exp() * F.softplus(m.in_proj_a(x).float() + m.dt_bias.float())
    if Hv // Hk > 1:
        q = q.repeat_interleave(Hv // Hk, dim=2)
        k = k.repeat_interleave(Hv // Hk, dim=2)
    q, k, v, beta = q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous()
    s0 = None if state is None else state[1].contiguous()
    if s0 is not None and not torch.is_tensor(s0) and not (T == 1 and _GDN_DECODE == "factored"):
        s0 = s0.to_dense(getattr(m, "factored_vbar", None))  # factored decode state entering a dense (T > 1) path
    if T == 1 and s0 is not None and _GDN_DECODE == "torch":
        core, s1 = _torch_recurrent_step(q, k, v, g, beta, s0)
    elif _GDN_IMPL == "torch":
        core, s1 = _torch_chunk_gdn(q, k, v, g, beta, s0)
    else:
        chunk, recurrent = _fla()
        if T == 1 and s0 is not None and _GDN_DECODE == "recurrent":
            core, s1 = recurrent(q, k, v, g, beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True)
        else:
            core, s1 = chunk(q, k, v, g, beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    new_state = (new_conv, s1)
    if not want_output:
        return None, new_state
    z = m.in_proj_z(x).reshape(-1, Dv)
    out = m.norm(core.reshape(-1, Dv), z).reshape(B, T, Hv * Dv)
    return m.out_proj(out), new_state


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        h, Hv, C, K = cfg.hidden_size, cfg.linear_num_value_heads, cfg.conv_dim, cfg.linear_conv_kernel_dim
        self.in_proj_qkv = nn.Linear(h, C, bias=False)
        self.in_proj_z = nn.Linear(h, cfg.value_dim, bias=False)
        self.in_proj_b = nn.Linear(h, Hv, bias=False)
        self.in_proj_a = nn.Linear(h, Hv, bias=False)
        self.conv1d = nn.Conv1d(C, C, K, groups=C, padding=K - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(Hv))
        self.A_log = nn.Parameter(torch.zeros(Hv))
        self.norm = RMSNormGated(cfg.linear_value_head_dim, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(cfg.value_dim, h, bias=False)

    def forward(self, x, state: Optional[GDNState] = None):
        return gdn_mix(self, self.cfg, x, state, True)


def _cast_model(model: nn.Module, dtype):
    """Cast to dtype but keep the GDN decay parameters in fp32 (as HF/SGLang do)."""
    model = model.to(dtype)
    for name, p in model.named_parameters():
        if name.endswith(FP32_PARAM_SUFFIXES):
            p.data = p.data.float()
    return model


@dataclass
class Qwen4ExpConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    vocab_size: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    layer_types: List[str]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    hc_count: int = 4
    hc_lowrank: int = 320
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    ple_layer_ids: List[int] = field(default_factory=list)  # 1-indexed (HF: layer_idx + 1 in ple_layer_ids)
    ple_embed_dim: int = 0
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 128
    eos_token_id: int = 248044  # PLE segment delimiter (text_config eos)
    output_gate_type: str = "sigmoid"
    norm_topk_prob: bool = True
    bos_token_id: Optional[int] = None
    model_type: str = "qwen4_exp_text"
    intermediate_size: int = 0  # unused (every layer is MoE); kept for hybrid.MLP's signature
    mlp_only_layers: List[int] = field(default_factory=list)
    decoder_sparse_step: int = 1
    qk_norm: bool = True

    # ---- hybrid-compatible derived quantities
    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def layer_is_moe(self, layer: int) -> bool:
        return self.is_moe

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def hc_dim(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def dense_limit(self) -> int:
        """visible tokens up to which the sparse selection keeps every token (all complete blocks fit the budget + tail)."""
        return self.indexer_budget + self.indexer_compress_ratio - 1

    def is_attn(self, layer: int) -> bool:
        return self.layer_types[layer] != "linear_attention"

    def attn_layers(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t != "linear_attention"]

    def ple_index(self, layer: int) -> Optional[int]:
        """index of `layer` (0-based) within the PLE layers, or None (HF: `config.ple_layer_ids.index(layer_idx + 1)`)."""
        ids = sorted(set(self.ple_layer_ids))
        return ids.index(layer + 1) if (layer + 1) in ids else None

    @staticmethod
    def from_hf(path: str) -> "Qwen4ExpConfig":
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        t = c.get("text_config", c)
        mt = c.get("model_type", t.get("model_type"))
        assert mt in QWEN4_MODEL_TYPES, f"unsupported model_type {mt}"
        rope = t.get("rope_parameters") or {}
        n = t["num_hidden_layers"]
        lt = t.get("layer_types")
        if lt is None:
            k = t.get("full_attention_interval", 4)
            lt = ["linear_attention" if (i + 1) % k else "full_attention" for i in range(n)]
        eos = t.get("eos_token_id", c.get("eos_token_id"))
        if isinstance(eos, list):
            eos = eos[0]
        tie = c.get("tie_word_embeddings", t.get("tie_word_embeddings"))
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):  # the checkpoint is the ground truth for tying
            with open(idx) as f:
                tie = "lm_head.weight" not in json.load(f)["weight_map"]
        return Qwen4ExpConfig(
            hidden_size=t["hidden_size"], num_hidden_layers=n, num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"], head_dim=t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"],
            rms_norm_eps=t["rms_norm_eps"], rope_theta=rope.get("rope_theta", t.get("rope_theta", 1e7)),
            partial_rotary_factor=rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)),
            vocab_size=t["vocab_size"], tie_word_embeddings=bool(tie), max_position_embeddings=t.get("max_position_embeddings", 32768),
            layer_types=list(lt), linear_num_key_heads=t["linear_num_key_heads"], linear_num_value_heads=t["linear_num_value_heads"],
            linear_key_head_dim=t["linear_key_head_dim"], linear_value_head_dim=t["linear_value_head_dim"],
            linear_conv_kernel_dim=t["linear_conv_kernel_dim"], num_experts=t.get("num_experts", 0),
            num_experts_per_tok=t.get("num_experts_per_tok", 0), moe_intermediate_size=t.get("moe_intermediate_size", 0),
            shared_expert_intermediate_size=t.get("shared_expert_intermediate_size", 0), hc_count=t.get("hc_count", 4),
            hc_lowrank=t.get("hc_lowrank", 320), indexer_n_heads=t.get("indexer_n_heads", 4), indexer_kv_heads=t.get("indexer_kv_heads", 1),
            indexer_head_dim=t.get("indexer_head_dim", 128), indexer_budget=t.get("indexer_budget", 2048),
            indexer_compress_ratio=t.get("indexer_compress_ratio", 4), ple_layer_ids=list(t.get("ple_layer_ids") or []),
            ple_embed_dim=t.get("ple_embed_dim") or t["hidden_size"], ple_conv_kernel_size=t.get("ple_conv_kernel_size", 4),
            ngram_size=t.get("ngram_size", 3), heads_per_ngram=t.get("heads_per_ngram", 8),
            ngram_vocab_size_base=t.get("ngram_vocab_size_base", 20_000_000),
            make_ngram_vocab_size_divisible_by=t.get("make_ngram_vocab_size_divisible_by", 128), seed=t.get("seed", 1234),
            split_ngram_parts=t.get("split_ngram_parts", 128), eos_token_id=int(eos), output_gate_type=t.get("output_gate_type") or "silu",
            norm_topk_prob=bool(t.get("norm_topk_prob", True)), bos_token_id=t.get("bos_token_id", c.get("bos_token_id")),
            model_type=mt, intermediate_size=t.get("intermediate_size", 0) or 0,
        )


class GroupRMSNorm(nn.Module):
    """HF Qwen4ExpTextRMSNorm with group_size: RMS over each group (stream) of the last dim, x_hat * (1 + w) in fp32,
    w over the full flattened dim.  Accepts (..., groups, D) or (..., groups * D)."""

    def __init__(self, dim: int, groups: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.groups = groups
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.shape[-1] == self.weight.shape[0]
        xs = x.reshape(*x.shape[:-1], self.groups, -1) if flat else x
        xf = xs.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float().view(self.groups, -1))
        out = out.type_as(x)
        return out.flatten(-2) if flat else out


class RMSNormGatedAct(nn.Module):
    """HF Qwen4ExpTextRMSNormGated: w * x_hat (norm before gate) * act(gate), act = sigmoid (Qwen4-Exp) | silu."""

    def __init__(self, dim: int, eps: float, act: str = "sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.act = act

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * xf.to(dt)
        g = torch.sigmoid(gate.float()) if self.act == "sigmoid" else F.silu(gate.float())
        return (h * g).to(dt)


class GatedResidual(nn.Module):
    """HF Qwen4ExpTextGatedResidual on streams (B, T, hc, D).  `mix` -> (x, inject); `combine` writes a sublayer output
    back.  Calling the module (`forward`) returns x only, so it can stand in for an `input_layernorm` in shared code
    (`probes.common.kv_parts`); `.weight` exposes hc_norm's weight for device / dtype probing."""

    def __init__(self, cfg: Qwen4ExpConfig, use_combine: bool = True):
        super().__init__()
        self.hc = cfg.hc_count
        self.D = cfg.hidden_size
        self.hc_norm = GroupRMSNorm(cfg.hc_dim, cfg.hc_count, cfg.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(cfg.hc_dim, cfg.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(cfg.hc_lowrank, cfg.hc_dim, bias=False)
        self.block_inject_weight = nn.Linear(cfg.hc_dim, cfg.hc_count, bias=False) if use_combine else None

    @property
    def weight(self) -> torch.Tensor:
        return self.hc_norm.weight

    def mix(self, streams: torch.Tensor):
        """streams (B, T, hc, D) [or flat (B, T, hc*D)] -> (x (B, T, D), inject (B, T, hc) | None)."""
        w = self.hc_norm.weight
        streams = _to_dev(streams, w).to(w.dtype)
        if streams.shape[-1] == self.hc * self.D:
            streams = streams.reshape(*streams.shape[:-1], self.hc, self.D)
        normed = self.hc_norm(streams)  # (B, T, hc, D)
        flat = normed.flatten(-2)
        gate = torch.sigmoid(self.input_mix_weight_up(F.silu(self.input_mix_weight_down(flat) / self.hc)))
        x = (gate.view_as(normed) * normed).mean(dim=-2)
        inject = None
        if self.block_inject_weight is not None:
            inject = 2.0 * torch.sigmoid(self.block_inject_weight(flat) / self.hc)
        return x, inject

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        return self.mix(streams)[0]

    @staticmethod
    def combine(streams: torch.Tensor, out: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        """streams (B, T, hc, D) + out (B, T, D) (x) inject (B, T, hc)."""
        return streams + out.unsqueeze(-2) * inject.unsqueeze(-1).to(out.dtype)


class Qwen4GatedDeltaNet(GatedDeltaNet):
    def __init__(self, cfg: Qwen4ExpConfig):
        super().__init__(cfg)  # type: ignore[arg-type]  (duck-typed HybridConfig fields)
        self.norm = RMSNormGatedAct(cfg.linear_value_head_dim, cfg.rms_norm_eps, cfg.output_gate_type)

