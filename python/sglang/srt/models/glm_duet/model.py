"""GLM-5.3-Flash with DUET (spec.model "glm5-next"): the stock SGLang model (models/glm5_next.py) plus the DUET prompt path and the
KDA state cut.  Registered over Glm5NextForConditionalGeneration only when a release is given (--duet-release, adapters.py).

Prompt, per extend batch (chunked prefill allowed; a request is at its last chunk when prefix + extend == orig_seq_lens):
  P stream    every request's extend tokens except, at its last chunk, the last prompt token: stock layers 0..k-1 (their own
              KDA / DSA caches, TP collectives and kernels), then the coded 4 x 4096 residual (sglang.srt.duet.latent_codec
              ResidualCode, fp32, the token's input streams hc_expand(embed) as side information), then the emitters of every layer
              l >= k in the arithmetic they were trained with (emitter_math.py): the attention-site collapse + input norm on the
              coded residual, then
                KDA: k / v / forget / beta projections as bf16 GEMMs, fp32 conv taps over the slot's window, fla's chunk_kda (the
                     training kernel, varlen, the bounded gate) from the slot's state; layer l's conv window and recurrent state go
                     into its Mamba slot (this rank's 16 heads; continuation chunks start from the slot); the q path is the base's;
                DSA: c = kv_a_layernorm(kv_a(x)) (fp32 GEMM on the FP8 fake-quantised input) into the MLA pool, k = k_norm(wk x)
                     and s = G x into layer l's pooled FP8 index cache and open-pool tail through the layer's own
                     IndexerKPool._compress_write (its ape; the same pooling / compression path as the served indexer).
  boundary    at a request's last chunk: every KDA state of its slot is cut to the stored form (explicit sink + rank r, cold), then
              the last prompt token runs as decode step 1 through all 45 layers (the decode CUDA graph when it can take the batch);
              its logits are the request's first-token logits.  Requests in an intermediate chunk get placeholder logits (the
              scheduler drops their sampled token).
Decode: stock forward; the cut every W steps after the step (decode_cuda_graph_runner before_graph / after_graph on replayed steps,
the same hooks here on eager steps).  The cut is the reference's warm-started randomized subspace iteration (sglang.srt.duet.state_factor
project_state_batched, B=1 sketch per request drawn with all 64 heads and sliced to this rank's heads), scheduled on the host
(state_policy.GlmStatePolicy).
Env (diagnostics only, no numeric effect): SGLANG_GLM_DUET_SELFCHECK=1 compares, at the first prompt, the emitters with the stock
layers on the exact residual and the fake-quantised code with the packed NVFP4 / gap8 storage; SGLANG_GLM_DUET_PROFILE=1 section
timers; SGLANG_GLM_DUET_CUT_LOG=<dir> writes every cut (slot, prompt|decode, decode steps since the prompt end) of TP rank 0;
SGLANG_GLM_DUET_PLANCHECK=1 checks every extend batch's DSA kpool plan inputs before the prompt pass, after it and after the boundary.
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.layernorm.mhc import hc_expand
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models import glm5_next as _stock
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils.common import BumpAllocator

from ._common import load as _load
from . import emitter_math as em_math
from .geometry import MODEL, validate
from .state_policy import GlmStatePolicy

logger = logging.getLogger(__name__)

_options = _load("options")
_release = _load("release")
_codec = _load("latent_codec")

CODE_CHUNK = 8192
_SELFCHECK = os.environ.get("SGLANG_GLM_DUET_SELFCHECK", "0") not in ("", "0")
_PROFILE = os.environ.get("SGLANG_GLM_DUET_PROFILE", "0") not in ("", "0")
_CUT_LOG = os.environ.get("SGLANG_GLM_DUET_CUT_LOG") or None
_PLANCHECK = os.environ.get("SGLANG_GLM_DUET_PLANCHECK", "0") not in ("", "0")


# ----------------------------------------------------------------------------- emitters
COMMON = ("collapse.hc_attn_fn", "collapse.hc_attn_base", "collapse.hc_attn_scale", "collapse.input_layernorm.weight")


class _Collapse(nn.Module):
    """The emitter's attention-site collapse + input_layernorm (fp32 parameters; emitter_math.collapse)."""

    def __init__(self, cfg):
        super().__init__()
        n, D = cfg.hc_mult, cfg.hidden_size
        mix = (2 + n) * n
        self.hc_mult, self.eps, self.hc_eps = n, cfg.rms_norm_eps, cfg.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(mix, n * D, dtype=torch.float32), requires_grad=False)
        self.hc_base = nn.Parameter(torch.empty(mix, dtype=torch.float32), requires_grad=False)
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)
        self.norm_w = nn.Parameter(torch.empty(D, dtype=torch.float32), requires_grad=False)

    def load(self, rel, w):
        target = {"collapse.hc_attn_fn": self.hc_fn, "collapse.hc_attn_base": self.hc_base, "collapse.hc_attn_scale": self.hc_scale,
                  "collapse.input_layernorm.weight": self.norm_w}[rel]
        target.data.copy_(w.reshape(target.shape).float())

    def forward(self, streams):
        return em_math.collapse(streams, self.hc_fn, self.hc_scale, self.hc_base, self.norm_w, hc_mult=self.hc_mult, eps=self.eps,
                                hc_eps=self.hc_eps)


class _KDAEmitter(nn.Module):
    """Writes layer l's KDA prompt memory into its Mamba slot (this rank's heads) in the trained emitter's arithmetic (twinstar
    glm5_duet.GlmKDAEmitter: the layer's own -- bf16 GEMM outputs on the fp32 master weights cast to bf16, fp32 conv taps, gates and
    scan state): the conv windows of q / k / v (the last K-1 raw bf16 projections) and the recurrent state (fla's chunk_kda with the
    bounded gate, varlen over the batch; a continuation chunk starts from the slot)."""

    RELEASED = COMMON + ("mixer.k_proj.weight", "mixer.v_proj.weight", "mixer.k_conv1d.weight", "mixer.v_conv1d.weight",
                         "mixer.f_a_proj.weight", "mixer.f_b_proj.weight", "mixer.b_proj.weight", "mixer.A_log", "mixer.dt_bias")

    def __init__(self, cfg, layer_id):
        super().__init__()
        la = cfg.linear_attn_config
        self.layer_id = layer_id
        self.H, self.Dh, self.K = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
        self.lower_bound = float(la["gate_lower_bound"])
        tp, rank = get_parallel().attn_tp_size, get_parallel().attn_tp_rank
        assert self.H % tp == 0
        self.Hl = self.H // tp
        self.h0 = rank * self.Hl
        seg, D = self.Hl * self.Dh, cfg.hidden_size
        self.seg = seg
        self.collapse = _Collapse(cfg)
        P = lambda *shape, dt=torch.float32: nn.Parameter(torch.empty(*shape, dtype=dt), requires_grad=False)  # noqa: E731
        bf = torch.bfloat16                              # the projections run as bf16 GEMMs on the master weights cast to bf16
        self.qkv_w = P(3 * seg, D, dt=bf)                # this rank's [q | k | v] rows (q: the base layer's, never released)
        self.conv_w = P(3 * seg, self.K)
        self.f_a_w, self.f_b_w, self.b_w = P(self.Dh, D, dt=bf), P(seg, self.Dh, dt=bf), P(self.Hl, D, dt=bf)
        self.A_log = P(self.Hl)
        self.dt_bias = P(seg)
        self.loaded = set()

    def _rows(self, w):
        return w[self.h0 * self.Dh:(self.h0 + self.Hl) * self.Dh]

    @torch.no_grad()
    def load(self, rel, w):
        seg = self.seg
        if rel in COMMON:
            self.collapse.load(rel, w)
        elif rel in ("mixer.k_proj.weight", "mixer.v_proj.weight"):
            i = 1 if rel.startswith("mixer.k") else 2
            self.qkv_w[i * seg:(i + 1) * seg].copy_(self._rows(w).to(self.qkv_w.dtype))
        elif rel in ("mixer.k_conv1d.weight", "mixer.v_conv1d.weight"):
            i = 1 if rel.startswith("mixer.k") else 2
            self.conv_w[i * seg:(i + 1) * seg].copy_(self._rows(w.reshape(w.shape[0], -1)).float())
        elif rel == "mixer.f_a_proj.weight":
            self.f_a_w.copy_(w.to(self.f_a_w.dtype))
        elif rel == "mixer.f_b_proj.weight":
            self.f_b_w.copy_(self._rows(w).to(self.f_b_w.dtype))
        elif rel == "mixer.b_proj.weight":
            self.b_w.copy_(w[self.h0:self.h0 + self.Hl].to(self.b_w.dtype))
        elif rel == "mixer.A_log":
            self.A_log.copy_(w.reshape(-1)[self.h0:self.h0 + self.Hl].float())
        elif rel == "mixer.dt_bias":
            self.dt_bias.copy_(self._rows(w.reshape(-1)).float())
        else:
            raise KeyError(f"GLM DUET KDA emitter {self.layer_id}: unexpected tensor {rel}")
        self.loaded.add(rel)

    @torch.no_grad()
    def load_q_from(self, stock_attn):
        """The q path (it only fills the q conv window) is not released: this rank's q rows of the loaded base layer, never zeros."""
        seg = self.seg
        self.qkv_w[:seg].copy_(stock_attn.qkv_proj.weight[:seg].to(self.qkv_w.dtype))
        self.conv_w[:seg].copy_(stock_attn.qkv_conv1d.weight[:seg].reshape(seg, -1).float())
        self.loaded.add("q")

    def emit(self, h_hat, fb):
        from fla.ops.kda import chunk_kda       # the training kernel (twinstar GlmKDA.scan); ImportError = no fla in the server

        kb = get_attn_backend().linear_attn_backend
        cache = kb.req_to_token_pool.mamba2_layer_cache(self.layer_id)
        slots = kb.forward_metadata.mamba_cache_indices.long()
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        prefix = [int(x) for x in fb.extend_prefix_lens_cpu]
        B, K, Hl, Dh, seg = len(lens), self.K, self.Hl, self.Dh, self.seg
        x = self.collapse(h_hat)                                                        # bf16
        raw = F.linear(x, self.qkv_w).float()                                           # (T, 3 seg) bf16 values
        hist = cache.conv[0][slots].float()                                             # (B, K-1, 3 seg): the slot's window
        fresh = torch.tensor([p == 0 for p in prefix], device=hist.device)
        hist = torch.where(fresh[:, None, None], torch.zeros_like(hist), hist)
        rows, starts, at = [], [], 0
        offs = [0] + list(itertools.accumulate(lens))
        for r in range(B):
            starts.append(at)
            rows += [hist[r], raw[offs[r]:offs[r + 1]]]
            at += K - 1 + lens[r]
        full = torch.cat(rows, 0)                                                       # history + inputs per request
        conv = em_math.conv_taps(full[:, seg:], self.conv_w[seg:], starts, lens, K).to(torch.bfloat16)   # k | v (q's output unread)
        T = x.shape[0]
        k = em_math.l2norm(conv[:, :seg].float().reshape(1, T, Hl, Dh)).to(torch.bfloat16)
        v = conv[:, seg:].reshape(1, T, Hl, Dh)
        g, beta = em_math.kda_gates(x, self.f_a_w, self.f_b_w, self.b_w, self.A_log, self.dt_bias, heads=Hl, head_dim=Dh,
                                    lower_bound=self.lower_bound)
        cu = torch.tensor(offs, dtype=torch.long, device=x.device)
        factors = getattr(kb.req_to_token_pool, "factored_kda_pool", None)
        initial = factors.dense(self.layer_id, slots) if factors is not None else cache.temporal[slots]
        s0 = torch.where(fresh[:, None, None, None], torch.zeros_like(initial),
                         initial).transpose(-1, -2).float().contiguous()  # pool (H, V, K) -> (H, K, V)
        _, state = chunk_kda(q=k, k=k, v=v, g=g[None], beta=beta[None], initial_state=s0, output_final_state=True,
                             use_qk_l2norm_in_kernel=False, safe_gate=True, lower_bound=self.lower_bound, cu_seqlens=cu)
        window = torch.stack([full[s + lens[r]:s + lens[r] + K - 1] for r, s in enumerate(starts)])   # last K-1 raw bf16 inputs
        cache.conv[0][slots] = window.to(cache.conv[0].dtype)
        if factors is not None:
            factors.commit(self.layer_id, slots, state.float().transpose(-1, -2))
        else:
            cache.temporal[slots] = state.float().transpose(-1, -2).to(cache.temporal.dtype)


class _DSAEmitter(nn.Module):
    """Writes layer l's DSA prompt memory: the MLA latent c into the MLA pool and the indexer's (k, s) through the layer's own
    pooled-FP8 write path (IndexerKPool._compress_write: closed pools with the layer's ape, the open-pool tail)."""

    RELEASED = COMMON + ("mixer.kv_a_proj_with_mqa.weight", "mixer.kv_a_layernorm.weight", "mixer.indexer.wk.weight",
                         "mixer.indexer.k_norm.weight", "mixer.indexer.k_norm.bias", "mixer.indexer.index_kpool_compress_gate")

    def __init__(self, cfg, layer_id, act_quant):
        super().__init__()
        self.layer_id = layer_id
        self.eps, self.act_quant = cfg.rms_norm_eps, act_quant
        D, hd, r = cfg.hidden_size, cfg.index_head_dim, cfg.kv_lora_rank
        self.collapse = _Collapse(cfg)
        P = lambda *shape: nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)  # noqa: E731
        self.kv_a, self.kv_norm = P(r, D), P(r)
        self.wk, self.k_norm_w, self.k_norm_b, self.gate = P(hd, D), P(hd), P(hd), P(hd, D)
        self.names = {"mixer.kv_a_proj_with_mqa.weight": self.kv_a, "mixer.kv_a_layernorm.weight": self.kv_norm,
                      "mixer.indexer.wk.weight": self.wk, "mixer.indexer.k_norm.weight": self.k_norm_w,
                      "mixer.indexer.k_norm.bias": self.k_norm_b, "mixer.indexer.index_kpool_compress_gate": self.gate}
        self.loaded = set()

    @torch.no_grad()
    def load(self, rel, w):
        if rel in COMMON:
            self.collapse.load(rel, w)
        elif rel in self.names:
            self.names[rel].copy_(w.reshape(self.names[rel].shape).float())
        else:
            raise KeyError(f"GLM DUET DSA emitter {self.layer_id}: unexpected tensor {rel}")
        self.loaded.add(rel)

    def memory(self, x):
        return em_math.dsa_memory(x, self.kv_a, self.kv_norm, self.wk, self.k_norm_w, self.k_norm_b, self.gate, eps=self.eps,
                                  act_quant=self.act_quant)

    def emit(self, h_hat, fb, target_layer):
        x = self.collapse(h_hat)
        c, key, score = self.memory(x)
        attn = target_layer.self_attn
        get_token_to_kv_pool().set_mla_kv_buffer(attn.attn_mqa, fb.out_cache_loc, c.unsqueeze(1).contiguous(),
                                                 c.new_empty(c.shape[0], 1, 0))
        metadata = get_attn_backend().get_indexer_metadata(self.layer_id, fb)
        if metadata is None:
            raise RuntimeError(f"GLM DUET: no indexer metadata for layer {self.layer_id}")
        attn.indexer._compress_write(x=x, key=key, positions=fb.positions, forward_batch=fb, layer_id=self.layer_id,
                                     metadata=metadata, gate_score=score)


# ----------------------------------------------------------------------------- the model
class Glm5NextForConditionalGeneration(_stock.Glm5NextForConditionalGeneration):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        args = get_server_args()
        directory = _options.resolve_release(args)
        if not directory:
            raise ValueError("GLM DUET adapter registered without a release (--duet-release / SGLANG_DUET_DIR)")
        _load("numerics").require_profile(MODEL, args, production_supported=False)
        self._check_serving(args)
        super().__init__(config, quant_config, prefix)
        tc = config.text_config
        self.duet_geometry = validate(json.loads((Path(directory) / "spec.json").read_text()), tc)
        self.duet_identity = _release.verify_release(directory, geometry=self.duet_geometry, model=MODEL)
        self.duet_dir = Path(self.duet_identity.path)
        warning = _release.check_base(self.duet_dir, args.model_path)
        if warning:
            logger.warning(warning)
        spec = self.duet_spec = self.duet_identity.spec
        self.duet_options = _options.DuetOptions.resolve(spec, args, state_dim=self.duet_geometry.state_side_dim)
        self.k = spec["prefill_depth"]
        self.emitted = self.duet_geometry.emitted_layers(self.k)
        manifest = json.loads((self.duet_dir / "manifest.json").read_text())
        # the DSA emitter's kv_a input quantisation the components were trained with (twinstar glm5_duet act_quant; export_release
        # records it); "ue8m0" = the served DeepGEMM arithmetic of the base layer, the closed-form emitter's
        self.duet_act_quant = (manifest.get("provenance", {}).get("emitter_numerics") or {}).get("dsa_kv_a_act_quant", "ue8m0")
        self.duet_emitters = nn.ModuleDict({
            str(l): (_KDAEmitter(tc, l) if self.duet_geometry.is_kda(l) else _DSAEmitter(tc, l, self.duet_act_quant))
            for l in self.emitted})
        precision = _options.resolve_code_precision(args)
        self.duet_code = _codec.ResidualCode(self.duet_geometry.residual_dim, spec, tf32=precision == "tf32")
        self.register_buffer("duet_sink_dir", torch.empty(tc.num_hidden_layers, self.duet_geometry.state_heads,
                                                          self.duet_geometry.state_side_dim, dtype=torch.float32))
        self.duet_policy: Optional[GlmStatePolicy] = None
        self._model_runner = None
        self._selfchecked = not _SELFCHECK
        self._warned_logprob = False
        self.n_duet_prefill = self.n_boundary = self.n_graph_fallback = self.n_full_depth = self.n_chunks = 0
        # benchmark/duet_eval/launch_duet.py SERVED_CUT parses this line (one per TP rank)
        logger.info("DUET served cut: k=%d trim=%s r=%d W=%d sha256=%s", self.k, self.duet_options.prefill_layer_trim,
                    self.duet_options.decode_ssm_r, self.duet_options.decode_ssm_w, self.duet_identity.sha256)
        n_kda = sum(self.duet_geometry.is_kda(l) for l in self.emitted)
        logger.info("GLM DUET adapter: %s; emitters %d (%d KDA + %d DSA), code rank %d + %d (%s matmuls), attention TP %d rank %d",
                    self.duet_identity.describe(), len(self.emitted), n_kda, len(self.emitted) - n_kda, spec["latent_rank"],
                    spec["latent_spikes"], precision, get_parallel().attn_tp_size, get_parallel().attn_tp_rank)

    @staticmethod
    def _check_serving(args):
        if args.pp_size != 1 or get_parallel().attn_tp_size != get_parallel().tp_size:
            raise ValueError("GLM DUET serves TP only (no PP, no DP attention)")
        if not args.disable_radix_cache:
            raise ValueError("GLM DUET requires --disable-radix-cache (a cached prefix would skip the emitters)")
        if args.speculative_algorithm is not None or args.enable_mixed_chunk:
            raise ValueError("GLM DUET supports plain extend / decode batches (no speculative decoding, no mixed chunks)")
        if os.environ.get("SGLANG_MAMBA_SSM_DTYPE", "") != "float32":
            raise ValueError("GLM DUET requires the fp32 KDA state pool (SGLANG_MAMBA_SSM_DTYPE=float32)")

    # ------------------------------------------------------------------ weights
    def load_weights(self, weights, is_nextn=False):
        super().load_weights(weights, is_nextn=is_nextn)
        if is_nextn:
            return
        contract = _release.build_contract(self.duet_spec, self.duet_geometry)
        tensors = _release.load_components(self.duet_dir, contract, device="cpu")
        code = {}
        for name, t in tensors.items():
            if name.startswith("emitters."):
                l, rel = name[len("emitters."):].split(".", 1)
                self.duet_emitters[l].load(rel, t)
            elif name.startswith("latent.code."):
                code[name[len("latent.code."):]] = t
            elif name == "state.sink_dir":
                self.duet_sink_dir.copy_(t)
            else:
                raise KeyError(f"GLM DUET: unexpected release tensor {name}")
        if self.duet_code.code is not None:
            self.duet_code.code.E.data.copy_(code["E"]); self.duet_code.code.D.data.copy_(code["D"])
            self.duet_code.code.mu.copy_(code["mu"])
        for l in self.emitted:
            em = self.duet_emitters[str(l)]
            if isinstance(em, _KDAEmitter):
                em.load_q_from(self.model.layers[l].self_attn)
                missing = set(em.RELEASED + ("q",)) - em.loaded
            else:
                missing = set(em.RELEASED) - em.loaded
            if missing:
                raise RuntimeError(f"GLM DUET emitter {l}: tensors not loaded: {sorted(missing)}")
        logger.info("GLM DUET loaded: %s tensors=%d sha256=%s; emitters equal to the base layers' own weights: %s",
                    self.duet_identity.name, len(tensors), self.duet_identity.sha256, self._copy_report())

    @torch.no_grad()
    def _copy_report(self):
        """How many emitters are bit-identical to their base layer's served weights (a closed-form set: all; a trained set: none)."""
        same = 0
        for l in self.emitted:
            em, layer = self.duet_emitters[str(l)], self.model.layers[l]
            same32 = lambda a, b: torch.equal(a.float().reshape(-1), b.float().reshape(-1))   # noqa: E731
            eq = [same32(em.collapse.hc_fn, layer.hc_attn_fn), same32(em.collapse.hc_base, layer.hc_attn_base),
                  same32(em.collapse.hc_scale, layer.hc_attn_scale), same32(em.collapse.norm_w, layer.input_layernorm.weight)]
            a = layer.self_attn
            if isinstance(em, _KDAEmitter):
                eq += [same32(em.qkv_w, a.qkv_proj.weight), same32(em.conv_w, a.qkv_conv1d.weight), same32(em.f_a_w, a.f_a_proj.weight),
                       same32(em.f_b_w, a.f_b_proj.weight), same32(em.b_w, a.b_proj.weight), same32(em.A_log, a.A_log),
                       same32(em.dt_bias, a.dt_bias)]
            else:
                ix = a.indexer
                eq += [same32(em.kv_norm, a.kv_a_layernorm.weight), same32(em.wk, ix.wk.weight), same32(em.gate, ix.index_kpool_compress_gate),
                       same32(em.k_norm_w, ix.k_norm.weight), same32(em.k_norm_b, ix.k_norm.bias)]
            same += all(eq)
        return f"{same} of {len(self.emitted)}"

    # ------------------------------------------------------------------ runner hooks
    def prepare_before_cuda_graph_capture(self, model_runner):
        """Every runner warm-up calls this (base_runner.warmup), before graph capture and autotune: mark the runner's dummy batches
        (never counted as decode steps) and build the state cut on the KDA pool, which the decode graph runner's replay calls."""
        from sglang.srt.models.lightning_duet.boundary import mark_runner_dummy_batches
        mark_runner_dummy_batches(model_runner)
        self._model_runner = model_runner
        if self.duet_policy is not None or not self.duet_options.decode_ssm_r:
            return
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedHybridReqToTokenPool
        pool = model_runner.req_to_token_pool
        if isinstance(pool, UnifiedHybridReqToTokenPool):
            raise NotImplementedError("GLM DUET serves the plain hybrid pool (physical Mamba slot ids)")
        g = self.duet_geometry
        tp, rank = get_parallel().attn_tp_size, get_parallel().attn_tp_rank
        hl = g.state_heads // tp
        dirs = {l: self.duet_sink_dir[l, rank * hl:(rank + 1) * hl].contiguous() for l in g.kda_layers}
        spec = self.duet_options.effective_spec(self.duet_spec)
        if spec["state_sink"] != "explicit":
            raise NotImplementedError("GLM DUET serves the explicit sink")
        log = None
        if _CUT_LOG and get_parallel().tp_rank == 0:
            os.makedirs(_CUT_LOG, exist_ok=True)
            log = os.path.join(_CUT_LOG, f"cuts-{os.getpid()}.jsonl")
        factors = getattr(pool, "factored_kda_pool", None)
        device_projection = os.environ.get("SGLANG_GLM_W32_DEVICE", "0") == "1"
        if factors is not None and device_projection:
            raise ValueError("GLM device dense projection and factored pool are mutually exclusive")
        if factors is not None:
            if (factors.cfg.rank, factors.cfg.window) != (spec["state_rank"], spec["state_every"]):
                raise ValueError("KDA factor rank/window must match the DUET spec")
            self.duet_policy = factors.bind_directions(dirs)
        elif device_projection:
            from .graph_state_policy import GraphGlmStatePolicy
            self.duet_policy = GraphGlmStatePolicy(pool, dirs, spec, list(g.kda_layers), omega_heads=(rank * hl, g.state_heads))
            pool.glm_w32_policy = self.duet_policy
        else:
            self.duet_policy = GlmStatePolicy(pool, dirs, spec, list(g.kda_layers), omega_heads=(rank * hl, g.state_heads), cut_log=log)
        # The frozen 7a graph runner already executes this policy interface.
        model_runner.kda_state_pruner = self.duet_policy
        model_runner.duet_state_policy = self.duet_policy
        logger.info("GLM DUET state cut: explicit sink + rank %d every %d decode steps (heads %d..%d of %d)", spec["state_rank"],
                    spec["state_every"], rank * hl, (rank + 1) * hl - 1, g.state_heads)

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
    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None, pp_proxy_tensors=None, **kwargs):
        fb = forward_batch
        mode = fb.forward_mode
        if getattr(fb, "_lightning_dummy_batch", False) or get_is_capture_mode() or mode.is_idle():
            return super().forward(input_ids, positions, fb, input_embeds, pp_proxy_tensors)
        if self.duet_options.decode_ssm_r and self.duet_policy is None:
            raise RuntimeError("GLM DUET state cut was not attached (prepare_before_cuda_graph_capture did not run)")
        if mode.is_decode():
            return self._eager_step(input_ids, positions, fb)
        if not mode.is_extend() or mode.is_target_verify() or mode.is_draft_extend_v2() or mode.is_mixed() or fb.spec_info is not None:
            raise ValueError(f"GLM DUET: unsupported batch {mode.name}")
        if input_embeds is not None or fb.extend_seq_lens_cpu is None:
            raise ValueError("GLM DUET requires token ids and CPU extend lengths")
        factors = getattr(get_attn_backend().linear_attn_backend.req_to_token_pool, "factored_kda_pool", None)
        if factors is not None:
            slots = get_attn_backend().linear_attn_backend.forward_metadata.mamba_cache_indices.long()
            with factors.prefill(slots, self._requests(fb)[3]):
                if not self.duet_options.prefill_layer_trim:
                    return self._full_depth_extend(input_ids, positions, fb)
                return self._duet_extend(input_ids, positions, fb)
        if not self.duet_options.prefill_layer_trim:
            return self._full_depth_extend(input_ids, positions, fb)
        return self._duet_extend(input_ids, positions, fb)

    def _eager_step(self, input_ids, positions, fb):
        """A decode step the graph runner did not replay: the same cut hooks as the replay (prompt cuts before, count + due cuts after)."""
        if self.duet_policy is not None:
            self.duet_policy.before_graph(fb)
        out = super().forward(input_ids, positions, fb)
        if self.duet_policy is not None:
            self.duet_policy.after_graph(fb)
        return out

    def _requests(self, fb):
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        prefix = [int(x) for x in fb.extend_prefix_lens_cpu]
        full = fb.orig_seq_lens.tolist() if fb.orig_seq_lens is not None else [p + l for p, l in zip(prefix, lens)]
        last = [p + l >= int(o) for p, l, o in zip(prefix, lens, full)]
        return lens, prefix, [int(o) for o in full], last

    def _full_depth_extend(self, input_ids, positions, fb):
        """--no-prefill-layer-trim: the stock prefill; the prompt-final states are still cut once before the first decode step."""
        out = super().forward(input_ids, positions, fb)
        _, _, _, last = self._requests(fb)
        if self.duet_policy is not None and any(last):
            sel = torch.tensor([i for i, x in enumerate(last) if x], device=fb.req_pool_indices.device)
            rpi = fb.req_pool_indices[sel]
            slots = self.duet_policy.pool.get_mamba_indices(rpi).long()
            if not getattr(self.duet_policy, "device_scheduled", False):
                rpi, slots = rpi.tolist(), slots.tolist()
            self.duet_policy.mark_prompt_end(rpi, slots)
        self.n_full_depth += 1
        return out

    def _duet_extend(self, input_ids, positions, fb):
        lens, prefix, full, last = self._requests(fb)
        if sum(lens) != input_ids.shape[0]:
            raise RuntimeError(f"GLM DUET: padded extend batch ({sum(lens)} tokens of {input_ids.shape[0]}); prefill graphs must be off")
        plen = [l - (1 if lt else 0) for l, lt in zip(lens, last)]
        snap = self._plan_state(fb) if _PLANCHECK else None
        if _PLANCHECK:
            self._plan_check(fb, "start", snap)
        t0 = time.perf_counter() if _PROFILE else None
        prof = {}
        if sum(plen) > 0:
            fb1 = self._sub_batch(fb, input_ids, positions, plen)
            self._prompt_pass(fb1, prof)
        t1 = self._tick(prof, "prompt", t0)
        if _PLANCHECK:
            self._plan_check(fb, "after prompt pass", snap)
        out = self._boundary(input_ids, positions, fb, lens, full, last)
        if _PLANCHECK:
            self._plan_check(fb, "after boundary", snap)
        try:
            get_attn_backend().init_forward_metadata(fb)            # leave the backend planned for the original batch
        except Exception:
            logger.error("GLM DUET: re-plan of the original batch failed: bs=%d lens=%s prefix=%s full=%s", fb.batch_size, lens, prefix, full)
            raise
        self._tick(prof, "boundary", t1)
        self.n_duet_prefill += 1
        self.n_chunks += sum(1 for x in last if not x)
        if _PROFILE:
            logger.warning("GLM DUET prefill: B=%d tokens=%d P=%d last=%d [%s]", len(lens), sum(lens), sum(plen), sum(last),
                           " ".join(f"{k} {v:.1f} ms" for k, v in prof.items()))
        if self.n_duet_prefill % 500 == 1:
            logger.info("GLM DUET counters: prompt passes %d, boundary steps %d (graph fallbacks %d), intermediate chunks %d, "
                        "full-depth extends %d", self.n_duet_prefill, self.n_boundary, self.n_graph_fallback, self.n_chunks, self.n_full_depth)
        return out

    @staticmethod
    def _plan_state(fb):
        return dict(bs=fb.batch_size, ext=list(fb.extend_seq_lens_cpu), pre=list(fb.extend_prefix_lens_cpu), seq_cpu=fb.seq_lens_cpu.tolist(),
                    seq=fb.seq_lens.tolist(), rpi=fb.req_pool_indices.tolist(), mode=fb.forward_mode.name)

    def _plan_check(self, fb, tag, snap):
        """SGLANG_GLM_DUET_PLANCHECK=1 (diagnostic): the DSA kpool extend plan's page-table rows of this batch against its page table
        width, and every planning input of the batch against its value when the extend began."""
        from sglang.srt.layers.attention.dsa import kpool_plan as kp
        dsa = get_attn_backend().full_attn_backend
        spp, ps = dsa._kpool_slots_per_page(), dsa.dsa_index_kpool
        cpu = kp._kpool_cpu_plan(fb, ps, spp)
        ncols = -(-int(fb.seq_lens_cpu.max()) // dsa.real_page_size)
        rows = [(pid // spp) * ps for pid in cpu.pool_pool_id]
        bad = [(b, pid, r) for b, pid, r in zip(cpu.pool_batch_idx, cpu.pool_pool_id, rows) if r >= ncols]
        now = self._plan_state(fb)
        diff = {k: (snap[k], now[k]) for k in snap if snap[k] != now[k]}
        logger.warning("GLM DUET plancheck %s: bs %d spp %d pool %d page %d ncols %d max row %s; out of range %d %s; changed %s", tag, fb.batch_size,
                       spp, ps, dsa.real_page_size, ncols, max(rows) if rows else None, len(bad), bad[:4], diff)

    @staticmethod
    def _tick(prof, name, t):
        if not _PROFILE:
            return None
        torch.cuda.synchronize()
        now = time.perf_counter()
        prof[name] = prof.get(name, 0.0) + (now - t) * 1e3
        return now

    def _prompt_pass(self, fb1, prof):
        """Stock layers 0..k-1 on the P stream, the code, the emitters."""
        body = self.model
        bk = get_attn_backend()
        rec = get_global_expert_distribution_recorder()
        t = time.perf_counter() if _PROFILE else None
        bk.init_forward_metadata(fb1)
        with get_attn_tp_context().maybe_input_scattered(fb1):
            hidden, residual, topk = body.embed_tokens(fb1.input_ids), None, None
            za = BumpAllocator(buffer_size=len(body.layers) * 2, dtype=torch.float32, device=hidden.device)
            for l in range(self.k):
                with rec.with_current_layer(l):
                    hidden, residual, topk = body.layers[l](fb1.positions, hidden, fb1, residual, za, None, prev_topk_indices=topk)
            if residual is not None or hidden.shape[-1] != self.duet_geometry.residual_dim:
                raise RuntimeError(f"GLM DUET: expected the {self.duet_geometry.residual_dim}-wide mHC streams at the cut, got "
                                   f"{tuple(hidden.shape)} (residual {residual is not None})")
            t = self._tick(prof, f"P{self.k}", t)
            base = hc_expand(body.embed_tokens(fb1.input_ids), self.duet_geometry.hc_mult)   # a fresh lookup: the side information
            # the code is per token; 8,192-token pieces as the training code (twinstar glm5_duet.CODE_CHUNK) keep its GEMM shapes
            h_hat = torch.cat([self.duet_code(hidden[a:a + CODE_CHUNK], base[a:a + CODE_CHUNK])
                               for a in range(0, hidden.shape[0], CODE_CHUNK)])
            t = self._tick(prof, "code", t)
            if not self._selfchecked:
                self._selfcheck(hidden, base, h_hat, fb1)
            for l in self.emitted:
                em = self.duet_emitters[str(l)]
                if isinstance(em, _KDAEmitter):
                    em.emit(h_hat, fb1)
                else:
                    em.emit(h_hat, fb1, body.layers[l])
            self._tick(prof, f"emit{len(self.emitted)}", t)

    def _boundary(self, input_ids, positions, fb, lens, full, last):
        """Placeholder logits for every request (the stock logits processor on zeros: shapes, dtype and metadata of this batch),
        then the boundary step of the requests at their last chunk overwrites theirs."""
        T = input_ids.shape[0]
        zeros = torch.zeros(T, self.config.hidden_size, dtype=self.model.embed_tokens.weight.dtype, device=input_ids.device)
        if fb.return_logprob and not self._warned_logprob and fb.extend_logprob_start_lens_cpu is not None and any(
                int(s) < int(n) for s, n in zip(fb.extend_logprob_start_lens_cpu, lens)):
            self._warned_logprob = True       # the prompt runs k layers: no prompt-token logprobs exist (the output's are exact)
            logger.warning("GLM DUET: prompt-token logprobs were requested; DUET computes none (the values returned are placeholders)")
        out = self.logits_processor(input_ids, zeros, self.lm_head, fb)
        sel = [i for i, x in enumerate(last) if x]
        if not sel:
            return out
        starts = [0] + list(itertools.accumulate(lens))
        tok = torch.tensor([starts[i] + lens[i] - 1 for i in sel], dtype=torch.long)
        fbd = self._decode_batch(fb, input_ids, positions, tok, sel, [full[i] for i in sel])
        factors = getattr(get_attn_backend().linear_attn_backend.req_to_token_pool, "factored_kda_pool", None)
        if factors is not None:
            factors.finish_prefill(factors.slots(fbd))
        if self.duet_policy is not None:
            rpi, slots = fbd.req_pool_indices[:fbd.batch_size], self.duet_policy.slots(fbd)
            if not getattr(self.duet_policy, "device_scheduled", False):
                rpi, slots = rpi.tolist(), slots.tolist()
            self.duet_policy.mark_prompt_end(rpi, slots)
        runner = self._decode_graph_runner()
        saved = None
        if runner is not None and runner.can_run_graph(fbd):
            # The decode graph runner's static bs-axis inputs are the eager runner's (one shared pool,
            # cuda_graph_buffer_registry.build_eager_registry) and fb's fields are views of them: its load_batch writes fbd's rows and
            # the padding (seq_len 1, req_pool_index 0) over fb's; fb is restored after the step.
            saved = [(t, t.clone()) for t in vars(fb).values() if isinstance(t, torch.Tensor)]
            step = runner.execute(fbd)                         # before_graph: the prompt cuts; after_graph: step 1 counted
        else:
            self.n_graph_fallback += runner is not None
            get_attn_backend().init_forward_metadata(fbd)
            step = self._eager_step(fbd.input_ids, fbd.positions, fbd)
        nl = step.next_token_logits[:len(sel)]
        out.next_token_logits[torch.tensor(sel, device=nl.device)] = nl.to(out.next_token_logits.dtype)
        for t, c in saved or ():
            t.copy_(c)
        self.n_boundary += len(sel)
        return out

    # ------------------------------------------------------------------ derived batches (Kimi adapter's, per-request P lengths)
    def _sub_batch(self, fb, input_ids, positions, plen: List[int]):
        """The P stream: every request's first plen[r] extend tokens (same prefix); requests with plen 0 are dropped."""
        lens = [int(x) for x in fb.extend_seq_lens_cpu]
        prefix = [int(x) for x in fb.extend_prefix_lens_cpu]
        starts = [0] + list(itertools.accumulate(lens))
        dev = input_ids.device
        sel = [r for r in range(len(lens)) if plen[r] > 0]
        idx_cpu = torch.cat([torch.arange(starts[r], starts[r] + plen[r]) for r in sel])
        new_lens = [plen[r] for r in sel]
        new_prefix = [prefix[r] for r in sel]
        idx = idx_cpu.to(dev, non_blocking=True)
        sel_t = torch.tensor(sel, dtype=torch.long, device=dev)
        nb = copy.copy(fb)
        nb.batch_size = len(sel)
        nb.input_ids = input_ids[idx]
        nb.positions = positions[idx]
        nb.req_pool_indices = fb.req_pool_indices[sel_t]
        if fb.req_pool_indices_cpu is not None:
            nb.req_pool_indices_cpu = fb.req_pool_indices_cpu[torch.tensor(sel, dtype=torch.long)]
        seq_new = [p + l for p, l in zip(new_prefix, new_lens)]
        nb.seq_lens = torch.tensor(seq_new, dtype=fb.seq_lens.dtype, device=dev)
        nb.seq_lens_cpu = torch.tensor(seq_new, dtype=fb.seq_lens_cpu.dtype if fb.seq_lens_cpu is not None else torch.int64)
        nb.seq_lens_sum = int(sum(seq_new))
        if fb.orig_seq_lens is not None:
            nb.orig_seq_lens = fb.orig_seq_lens[sel_t]
        nb.out_cache_loc = fb.out_cache_loc[idx]
        if getattr(fb, "out_cache_loc_virtual", None) is not None:
            nb.out_cache_loc_virtual = fb.out_cache_loc_virtual[idx]
        nb.extend_num_tokens = int(idx_cpu.numel())
        nb.extend_seq_lens = torch.tensor(new_lens, dtype=fb.extend_seq_lens.dtype, device=dev)
        nb.extend_prefix_lens = torch.tensor(new_prefix, dtype=fb.extend_prefix_lens.dtype, device=dev)
        nb.extend_start_loc = (torch.tensor([0] + list(itertools.accumulate(new_lens))[:-1], dtype=fb.extend_start_loc.dtype,
                                            device=dev) if fb.extend_start_loc is not None else None)
        nb.extend_seq_lens_cpu = new_lens
        nb.extend_prefix_lens_cpu = new_prefix
        if fb.extend_logprob_start_lens_cpu is not None:
            nb.extend_logprob_start_lens_cpu = [0] * len(sel)
        nb.return_logprob = False
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, t[sel_t.to(t.device)])
        if getattr(fb, "global_num_token_non_padded_cpu", None) is not None:
            nb.global_num_token_non_padded_cpu = nb.extend_num_tokens
        for name in ("global_num_token_non_padded", "num_token_non_padded"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, torch.tensor(nb.extend_num_tokens, dtype=t.dtype, device=dev))
        nb.forward_metadata_ready = False
        nb.forward_metadata_planned_bs = None
        nb.forward_metadata_planned_num_tokens = None
        nb.mm_inputs = None
        nb.input_embeds = None
        return nb

    def _decode_batch(self, fb, input_ids, positions, tok_idx_cpu, sel: List[int], seq_new: List[int]):
        """One DECODE token per request `sel` of the extend batch fb: token tok_idx[r], sequence length seq_new[r] after it."""
        dev = input_ids.device
        idx = tok_idx_cpu.to(dev, non_blocking=True)
        sel_t = torch.tensor(sel, dtype=torch.long, device=dev)
        nb = copy.copy(fb)
        nb.forward_mode = ForwardMode.DECODE
        if getattr(nb, "global_forward_mode", None) is not None:
            nb.global_forward_mode = ForwardMode.DECODE
        nb.batch_size = len(sel)
        nb.input_ids = input_ids[idx]
        nb.positions = positions[idx]
        nb.req_pool_indices = fb.req_pool_indices[sel_t]
        if fb.req_pool_indices_cpu is not None:
            nb.req_pool_indices_cpu = fb.req_pool_indices_cpu[torch.tensor(sel, dtype=torch.long)]
        nb.seq_lens = torch.tensor(seq_new, dtype=fb.seq_lens.dtype, device=dev)
        nb.seq_lens_cpu = torch.tensor(seq_new, dtype=fb.seq_lens_cpu.dtype if fb.seq_lens_cpu is not None else torch.int64)
        nb.seq_lens_sum = int(sum(seq_new))
        if fb.orig_seq_lens is not None:
            nb.orig_seq_lens = fb.orig_seq_lens[sel_t]
        nb.out_cache_loc = fb.out_cache_loc[idx]
        for name in ("out_cache_loc_virtual", "out_cache_loc_dsv4"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, t[idx])
        nb.extend_num_tokens = None
        nb.extend_seq_lens = nb.extend_prefix_lens = nb.extend_start_loc = None
        nb.extend_seq_lens_cpu = nb.extend_prefix_lens_cpu = nb.extend_logprob_start_lens_cpu = None
        nb.return_logprob = False
        nb.top_logprobs_nums = nb.token_ids_logprobs = None
        nb.is_extend_in_batch = False
        nb.can_run_decode_cuda_graph = True
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            t = getattr(fb, name, None)
            if t is not None:
                setattr(nb, name, t[sel_t.to(t.device)])
        if getattr(fb, "global_num_token_non_padded_cpu", None) is not None:
            nb.global_num_token_non_padded_cpu = len(sel)
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

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def _selfcheck(self, streams, base, h_hat, fb1):
        """SGLANG_GLM_DUET_SELFCHECK=1, once: the emitters (the training arithmetic) against the stock layers (the served bf16 / FP8
        arithmetic) on the EXACT residual for the first DSA and KDA layer at or above the cut (a closed-form set: equal up to the
        collapse's fp32-vs-TF32 GEMM and the DSA key's fp32-vs-bf16 GEMM), the code's effect, the packed NVFP4 / gap8 storage against
        the fake-quantised code."""
        self._selfchecked = True
        rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))   # noqa: E731
        lines = [f"code: |h_hat - h| / |h - h0| = {float((h_hat.float() - streams.float()).norm() / (streams.float() - base.float()).norm()):.4f}"]
        if self.duet_code.code is not None and self.duet_spec["latent_z_format"] == "nvfp4" and self.duet_spec["latent_spikes"]:
            c = self.duet_code.code
            packed = _codec.PackedResidualCode(c.E, c.D, c.mu, self.duet_spec["latent_spikes"], self.duet_spec["latent_id_side"])
            n = min(64, streams.shape[0])
            rec = packed.encode(streams[:n], base[:n], fb1.input_ids[:n])
            dec = packed.decode(rec, base[:n])
            lines.append(f"packed storage vs fake-quant decode (first {n} tokens): max |d| {float((dec.float() - h_hat[:n].float()).abs().max()):.3e}, "
                         f"{rec.nbytes / n:.0f} B/token stored (incl. token id + offsets)")
        for want in ("attention", "state"):
            l = next((x for x in self.emitted if self.duet_geometry.memory_kind(x) == want), None)
            if l is None:
                continue
            em, layer = self.duet_emitters[str(l)], self.model.layers[l]
            x_em = em.collapse(streams)
            x_st, _, _, fused = layer.hc_attn_pre(streams, layer.input_layernorm.weight.data, layer.input_layernorm.variance_epsilon)
            if not fused:
                x_st = layer.input_layernorm(x_st)
            row = f"layer {l} ({want}): collapse rel {rel(x_em, x_st):.2e} (bit-identical {torch.equal(x_em, x_st)})"
            if want == "attention":
                c, key, s = em.memory(x_st)
                a = layer.self_attn
                lat = a.prepare_qkv_latent(x_st, fb1)
                c_st = a.kv_a_layernorm(lat[..., a.q_lora_rank:a.q_lora_rank + a.kv_lora_rank].contiguous())
                k_st = a.indexer._get_k_bf16(x_st, fb1.positions)
                s_st = F.linear(x_st, a.indexer.index_kpool_compress_gate)
                row += f"; c rel {rel(c, c_st):.2e}; k rel {rel(key, k_st):.2e}; s rel {rel(s, s_st):.2e}"
            else:
                raw = F.linear(x_st, em.qkv_w)
                q2, b2, _, _ = layer.self_attn.forward_qkvbfg(x_st, fb1)
                g, beta = em_math.kda_gates(x_st, em.f_a_w, em.f_b_w, em.b_w, em.A_log, em.dt_bias, heads=em.Hl, head_dim=em.Dh,
                                            lower_bound=em.lower_bound)
                row += (f"; qkv rel {rel(raw, q2):.2e} (bit-identical {torch.equal(raw, q2)}); beta rel "
                        f"{rel(beta, torch.sigmoid(b2.float())):.2e}")
            lines.append(row)
        logger.warning("GLM DUET selfcheck: %s", " | ".join(lines))


EntryClass = [Glm5NextForConditionalGeneration]
