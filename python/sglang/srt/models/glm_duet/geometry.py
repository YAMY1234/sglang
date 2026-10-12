"""GLM-5.3-Flash DUET release geometry and tensor contract (torch-free; twinstar docs/79).

The released tensor set is DERIVED from the spec and the base config (sglang.srt.duet.release.build_contract), never listed by name:

    latent.code.E (1, r, 16384), latent.code.D (1, 16384, r), latent.code.mu (1, 16384)     the code on the flattened 4 x 4096 mHC
                                                                                              residual (stream-major, hc_expand layout)
    state.sink_dir (45, 64, 128)                                                              mean value direction of every KDA head
    emitters.<l>.* for every layer l >= k (every GLM layer holds memory; names of twinstar/models/glm5_duet.py's emitters):
        collapse.hc_attn_fn (24, 16384), collapse.hc_attn_base (24), collapse.hc_attn_scale (3),
        collapse.input_layernorm.weight (4096)                                                the layer's attention-site collapse +
                                                                                              input_layernorm (only fn's 4 pre rows are read)
        KDA layer:  mixer.{k,v}_proj.weight (8192, 4096), mixer.{k,v}_conv1d.weight (8192, 1, 4), mixer.f_a_proj.weight (128, 4096),
                    mixer.f_b_proj.weight (8192, 128), mixer.b_proj.weight (64, 4096), mixer.A_log (1, 1, 64, 1), mixer.dt_bias (8192)
                    (the q path, q_proj / q_conv1d, only fills the decode step's q conv window: inherited from the base layer)
        DSA layer:  mixer.kv_a_proj_with_mqa.weight (512, 4096) (fp32; closed form: the base layer's served UE8M0-grid weight; its
                    input is FP8-quantised per token / 128 channels as manifest provenance.emitter_numerics.dsa_kv_a_act_quant says),
                    mixer.kv_a_layernorm.weight (512), mixer.indexer.wk.weight (128, 4096), mixer.indexer.k_norm.{weight,bias} (128),
                    mixer.indexer.index_kpool_compress_gate (128, 4096)
All fp32 (twinstar.duet.export_release writes every component tensor as fp32); the server computes the emitters in fp32 as they were
trained (emitter_math.py).
"""
from __future__ import annotations

MODEL = "glm5-next"                                   # spec.model (twinstar.models.glm5_next.Glm5Next.duet_model)
BASE_HF = "zai-org/GLM-5.3-Flash"


def _get(c, key, default=None):
    return c.get(key, default) if isinstance(c, dict) else getattr(c, key, default)


def text_config(config):
    """config.json dict (top level or text_config) or the SGLang Glm5NextConfig / Glm5NextTextConfig object -> the text config."""
    tc = _get(config, "text_config")
    return tc if tc is not None else config


class Geometry:
    """Structural facts of the base the release contract derives from (sglang.srt.duet.release, duck-typed geometry)."""

    def __init__(self, config):
        c = text_config(config)
        la = _get(c, "linear_attn_config")
        self.hidden = int(_get(c, "hidden_size"))
        self.hc_mult = int(_get(c, "hc_mult"))
        self.num_layers = int(_get(c, "num_hidden_layers"))
        self.residual_dim = self.hc_mult * self.hidden
        self.kda_layers = tuple(sorted(int(x) for x in la["kda_layers"]))   # 0-based in the GLM config
        types = _get(c, "layer_types")
        if types is not None and tuple(i for i, t in enumerate(types) if t == "linear_attention") != self.kda_layers:
            raise ValueError("GLM DUET: kda_layers differ from layer_types")
        self.state_heads = int(la["num_heads"])
        self.state_side_dim = int(la["head_dim"])
        self.conv = int(la["short_conv_kernel_size"])
        self.kv_lora_rank = int(_get(c, "kv_lora_rank"))
        self.index_head_dim = int(_get(c, "index_head_dim"))
        if not _get(c, "mhc", False) or int(_get(c, "qk_rope_head_dim", 0) or 0) != 0:
            raise ValueError("GLM DUET expects the mHC residual and NoPE MLA of GLM-5.3-Flash")
        if not (_get(c, "index_kpool_compress", False) and _get(c, "index_kpool_always_select_tail", False)):
            raise ValueError("GLM DUET expects the pooled (kpool compress, always-select-tail) indexer")

    def is_kda(self, layer):
        return layer in self.kda_layers

    def memory_kind(self, layer):
        return "state" if self.is_kda(layer) else "attention"

    def emitter_tensors(self, layer):
        D, n = self.hidden, self.hc_mult
        mix = (2 + n) * n
        out = {"collapse.hc_attn_fn": (mix, n * D), "collapse.hc_attn_base": (mix,), "collapse.hc_attn_scale": (3,),
               "collapse.input_layernorm.weight": (D,)}
        if self.is_kda(layer):
            H, Dh, K = self.state_heads, self.state_side_dim, self.conv
            inner = H * Dh
            out.update({"mixer.k_proj.weight": (inner, D), "mixer.v_proj.weight": (inner, D),
                        "mixer.k_conv1d.weight": (inner, 1, K), "mixer.v_conv1d.weight": (inner, 1, K),
                        "mixer.f_a_proj.weight": (Dh, D), "mixer.f_b_proj.weight": (inner, Dh), "mixer.b_proj.weight": (H, D),
                        "mixer.A_log": (1, 1, H, 1), "mixer.dt_bias": (inner,)})
        else:
            hd = self.index_head_dim
            out.update({"mixer.kv_a_proj_with_mqa.weight": (self.kv_lora_rank, D),
                        "mixer.kv_a_layernorm.weight": (self.kv_lora_rank,), "mixer.indexer.wk.weight": (hd, D),
                        "mixer.indexer.k_norm.weight": (hd,), "mixer.indexer.k_norm.bias": (hd,),
                        "mixer.indexer.index_kpool_compress_gate": (hd, D)})
        return out

    def inherited_tensors(self, layer):
        """The emitter halves the release does not carry but the emitter reads: the KDA q path (q_proj, q_conv1d), which only fills the
        q conv window of the first decode step; the server takes them from the loaded base layer."""
        if not self.is_kda(layer):
            return {}
        inner = self.state_heads * self.state_side_dim
        return {f"model.language_model.layers.{layer}.self_attn.q_proj.weight": (f"emitters.{layer}.mixer.q_proj.weight", (inner, self.hidden)),
                f"model.language_model.layers.{layer}.self_attn.q_conv1d.weight": (f"emitters.{layer}.mixer.q_conv1d.weight", (inner, 1, self.conv))}

    def emitted_layers(self, k):
        return tuple(range(k, self.num_layers))


def validate(spec, config):
    """The spec (sglang.srt.duet.spec.validate_spec, model glm5-next) against this base: the cut inside the stack, ranks in range."""
    from ._common import load as _load      # torch-free: the light sglang.srt.duet package on a bare CPU box
    _load("spec").validate_spec(spec, model=MODEL)
    g = Geometry(config)
    if not 1 <= spec["prefill_depth"] < g.num_layers:
        raise ValueError(f"GLM DUET cut {spec['prefill_depth']} must be inside the {g.num_layers}-layer stack")
    if spec["latent_rank"] > g.residual_dim or spec["latent_spikes"] > g.residual_dim:
        raise ValueError("GLM DUET code larger than the 16,384-wide residual")
    if spec["state_rank"] >= g.state_side_dim:
        raise ValueError("GLM DUET state rank must be below the head dimension (0 = exact state)")
    return g
