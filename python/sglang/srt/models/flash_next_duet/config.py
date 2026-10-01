"""In-process Flash-Next release view and numerics policy (G1/G2)."""
from __future__ import annotations

import fnmatch
import logging
import os
from sglang.srt.duet import numerics
from sglang.srt.duet.adapters import release_value, select
from sglang.srt.duet.options import DuetOptions
from sglang.srt.duet.spec import validate_spec
from .release import config_dict, sink_cache, validate_release

# Enable only after the qualification evidence has been approved; lead #1541 keeps the gate off.
PRODUCTION_SUPPORTED = False


def profile_controls(args):
    profile = numerics.profile_name(args)
    production = profile == "production"
    raw = getattr(args, "_raw_input", {}) or {}
    graphs = production and not raw.get("disable_cuda_graph") and not raw.get("disable_prefill_cuda_graph")
    return {"profile": profile, "latent_compute_precision": "tf32" if production else "fp32",
            "prefill_graph": graphs, "emitter_graph": graphs,
            "emitter_state_only": production, "async_h2d": production,
            "radix_cache": numerics.controls(profile)["radix_cache"]}


def runtime_controls():
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.arg_groups.overrides import resolved_view
    return profile_controls(resolved_view(get_server_args()))


def describe_numerics(args, spec):
    """Report effective controls, including explicit state/emitter overrides."""
    from sglang.srt.duet.options import resolve_emitter_precision, resolve_prefix_state
    options = DuetOptions.resolve(spec, args)
    result = {**numerics.describe(args, production_supported=PRODUCTION_SUPPORTED), **profile_controls(args)}
    result.update(duet_emitter_precision=resolve_emitter_precision(args),
                  duet_prefix_state=resolve_prefix_state(args),
                  duet_state_truncation=getattr(args, "duet_state_truncation", None)
                  or os.environ.get("SGLANG_DUET_STATE_TRUNCATION")
                  or numerics.defaults(result["profile"])["duet_state_truncation"],
                  decode_ssm_r=options.decode_ssm_r, decode_ssm_w=options.decode_ssm_w,
                  prefill_layer_trim=options.prefill_layer_trim,
                  prefill_saving_policy=options.prefill_saving_policy,
                  factored_state=getattr(args, "linear_attn_factored_state", None),
                  cuda_graph_backend_decode=getattr(args, "cuda_graph_backend_decode", None),
                  radix_cache=not getattr(args, "disable_radix_cache", False),
                  max_running_requests=getattr(args, "max_running_requests", None))
    result["emitter_state_only"] &= result["duet_emitter_precision"] == "bf16"
    # Conv/SSM pool dtype remains native on Flash-Next.
    graph = getattr(args, "cuda_graph_config", None)
    decode = getattr(getattr(graph, "decode", None), "backend", None)
    if decode is not None:
        result["cuda_graph_backend_decode"] = getattr(decode, "value", str(decode))
    result["controls"] = {
        **result["controls"], "mamba_state_dtype": None,
        "radix_cache": result["radix_cache"],
        "cuda_graph": bool(result["prefill_graph"] or result["cuda_graph_backend_decode"] not in (None, "disabled")),
        "batched_decode": result["max_running_requests"] is None or result["max_running_requests"] > 1,
    }
    return result


def prepare_base_config(hf_config):
    """Declare checkpoint PLE storage before the native pinned-host allocation.

    This also applies with no release: the stock loader cannot swap a pinned
    BF16 embedding to FP8 after allocation. It changes storage metadata only,
    leaving the native class and checkpoint arithmetic intact.
    """
    if getattr(hf_config, "architectures", []) != ["Qwen4ExpForConditionalGeneration"]:
        return
    base = config_dict(hf_config)
    quant = base.get("quantization_config") or base.get("text_config", {}).get("quantization_config") or {}
    ple = [cfg for name, cfg in quant.get("quantized_layers", {}).items()
           if ".ple.ple_embedding.ngram_embedding" in name]
    if ple and all(cfg.get("quant_algo") == "FP8" for cfg in ple):
        getattr(hf_config, "text_config", hf_config).ple_embedding_dtype = "float8_e4m3fn"
        logging.getLogger(__name__).info("Flash-Next checkpoint PLE storage: float8_e4m3fn")


def resolve_server_numerics(server_args):
    """Resolve before the framework parses graph configuration or sizes pools."""
    from sglang.srt.arg_groups.overrides import declare_resolution, model_config_of, resolving_view
    args = resolving_view(server_args)
    if not release_value(args):
        return
    model_config = model_config_of(server_args)
    if not getattr(model_config.hf_config, "_duet_identity", None):
        return
    profile = numerics.require_profile("flash-next", args, production_supported=PRODUCTION_SUPPORTED)
    controls = profile_controls(args)
    # The adapter owns its P/codec/emitter graphs; the framework's whole-model
    # prefill graph cannot capture the CPU-shaped batch decomposition.
    values = dict(disable_prefill_cuda_graph=True, cuda_graph_backend_prefill="disabled")
    if profile == "reference":
        values.update(cuda_graph_backend_decode="disabled", disable_radix_cache=True)
        if getattr(args, "max_running_requests", None) is None:
            values["max_running_requests"] = 16
    declare_resolution(server_args, "_flash_next_duet_numerics", **values)
    # Existing stock layer breaks read this switch; the profile owns its value.
    os.environ["SGLANG_QWEN4_PREFILL_GRAPH"] = str(int(controls["prefill_graph"]))

def validate_base(config):
    text = config.get("text_config", config)
    if (config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]
            or any(type(text.get(k)) is not int or text[k] <= 0 for k in
                   ("num_hidden_layers", "hidden_size", "hc_count", "num_experts", "num_experts_per_tok"))):
        raise ValueError("unexpected Flash-Next base architecture")
    kinds = text.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != text["num_hidden_layers"]:
        raise ValueError("base must declare one layer_type per layer")
    if any(kind not in ("linear_attention", "full_attention") for kind in kinds):
        raise ValueError("unsupported Flash-Next layer type")
    q = config.get("quantization_config") or text.get("quantization_config") or {}
    if not q:
        return q  # BF16 base: the same geometry, emitter and codec contract.
    if (q.get("quant_method") not in ("modelopt", "modelopt_fp4")
            or q.get("quant_algo") not in ("NVFP4", "MIXED_PRECISION")):
        raise ValueError("this release requires the modelopt NVFP4 base")
    if q["quant_algo"] == "MIXED_PRECISION":
        # The published NVIDIA base also stores PLE embeddings and MTP in FP8.
        # Keep their native loader; target-model expert blocks are NVFP4.
        layers = q.get("quantized_layers", {})
        expected = {f"model.language_model.layers.{i}.mlp.experts" for i in range(text["num_hidden_layers"])}
        found = {n for n, v in layers.items() if v.get("quant_algo") == "NVFP4"}
        if found != expected:
            raise ValueError("mixed base must quantize all target expert blocks to NVFP4")
        for name, value in layers.items():
            if name in expected or name.startswith("mtp."):
                continue
            if ".ple.ple_embedding.ngram_embedding" not in name or value.get("quant_algo") != "FP8":
                raise ValueError(f"unsupported mixed quantized module: {name}")
    # The RadixArk checkpoint excludes all emitter source projections. A future
    # checkpoint quantizing them needs an explicit dequantizing initializer.
    ignore = q.get("ignore", [])
    for layer in range(text["num_hidden_layers"]):
        suffixes = ["attn_hyper_connection.hc_norm", "attn_hyper_connection.input_mix_weight_down"]
        kinds = text.get("layer_types")
        if kinds is None:
            raise ValueError("base must declare layer_types")
        suffixes += (["self_attn.qkv_proj", "self_attn.indexer.index_qk_proj"] if kinds[layer] == "full_attention"
                     else ["linear_attn.in_proj_qkvz", "linear_attn.in_proj_ba"])
        for suffix in suffixes:
            name = f"model.language_model.layers.{layer}.{suffix}"
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in ignore):
                raise ValueError(f"base emitter source may be quantized: {name}")
    return q


def derive_fullstack(identity, hf_config, options, profile, *, sink_file=None):
    """Derive the former offline override dictionary in process (G1)."""
    base_config = config_dict(hf_config)
    release = vars(identity)
    sink_file = sink_file if sink_file is not None else sink_cache(identity)
    quant = validate_base(base_config)
    spec = release["spec"]
    validate_spec(spec, model="flash-next")
    if profile not in ("reference", "production"):
        raise ValueError("unknown Flash-Next numerics profile")
    if (spec["latent_z_format"], spec["latent_value_format"], spec["latent_index_format"], spec["state_sink"]) != ("nvfp4", "bf16", "gap8", "explicit"):
        raise NotImplementedError("Flash-Next requires NVFP4/bf16/gap8 code and explicit sink")
    text = base_config.get("text_config", base_config)
    layers = text["num_hidden_layers"]
    fs = dict(version=3, status="latent-serving-candidate", release_name=spec["name"],
              duet_spec=spec, prefill_layer_trim=options.prefill_layer_trim, prefill_saving_policy=options.prefill_saving_policy,
              latent_width=text["hidden_size"] * text["hc_count"],
              latent="on", latent_id_side=spec["latent_id_side"], latent_store=spec["latent_z_format"],
              latent_weight_precision="bf16-roundtrip-fp32", latent_compute_precision="fp32" if profile == "reference" else "tf32",
              latent_rank=spec["latent_rank"], latent_sparse=spec["latent_spikes"],
              latent_value_format=spec["latent_value_format"], latent_index_format=spec["latent_index_format"],
              latent_payload_bytes=spec["latent_rank"] // 2 + spec["latent_rank"] // 16 + 4 + 3 * spec["latent_spikes"],
              latent_rms=False, qsa_code="off", deep_gdn_prefix=True, qad=True,
              gdn_state=f"rank:{options.decode_ssm_r}" if options.decode_ssm_r else "dense",
              gdn_rank=options.decode_ssm_r, gdn_every=options.decode_ssm_w,
              gdn_prefill_truncation="k31-warm-subspace", state_sink=spec["state_sink"],
              state_sink_vbar=str(sink_file), deep_private_tokens=65536, materialization_chunk=8192,
              release=release["path"], sha256=release["sha256"])
    ts = dict(p_layers=list(range(spec["prefill_depth"])), d_layers=list(range(layers)), emit_mode="shared",
              emitters=list(range(spec["prefill_depth"], layers)), bridge=0, bridge_kind="layer", fullstack=fs)
    # Preserve all nested quantization, PLE, architecture and expert fields.
    result = dict(language_model_only=True, twinstar=ts)
    if "text_config" in base_config:
        result["text_config"] = {**base_config["text_config"], "twinstar": ts}
        if any(".ple.ple_embedding.ngram_embedding" in name and cfg.get("quant_algo") == "FP8"
               for name, cfg in quant.get("quantized_layers", {}).items()):
            # Pinned-host PLE storage cannot change dtype while loading shards.
            result["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
    return result


def install_config(hf_config, args=None):
    """Attach the derived view before stock model/cache configuration consumes it."""
    value = release_value(args)
    if not value:
        return None
    if getattr(hf_config, "architectures", []) != ["Qwen4ExpForConditionalGeneration"]:
        return None
    existing = getattr(hf_config, "_duet_identity", None)
    if existing is not None:
        return existing
    directory, spec, adapter = select(value)
    if adapter.model != "flash-next":
        raise ValueError("Flash-Next base requires a flash-next DUET release")
    base = config_dict(hf_config)
    validate_base(base)
    # Local aliases are not Hub identities. Geometry still gets a full audit.
    base_name = base.get("_name_or_path")
    if not base_name or os.path.exists(base_name) or base_name.startswith("/"):
        base_name = None
    identity = validate_release(directory, hf_config, base_model=base_name)
    text = base.get("text_config", base)
    options = DuetOptions.resolve(spec, args, state_dim=text["linear_key_head_dim"])
    view = derive_fullstack(identity, hf_config, options, numerics.profile_name(args))
    hf_config.twinstar = view["twinstar"]
    hf_config.language_model_only = True
    if hasattr(hf_config, "text_config"):
        hf_config.text_config.twinstar = view["twinstar"]
        if "ple_embedding_dtype" in view["text_config"]:
            hf_config.text_config.ple_embedding_dtype = view["text_config"]["ple_embedding_dtype"]
    hf_config._duet_identity = vars(identity)
    return identity
