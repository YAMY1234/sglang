"""Checkpoint-driven DUET controls shared by CLI, environment and model adapters."""
import os

POLICIES = ('latent-only', 'latent-and-kv', 'latent-and-ssm', 'kv-and-ssm')


def apply_duet_options(model_config, cli):
    from .fullstack_policy import _duet_options
    _o = _duet_options()
    resolve_emitter_precision, resolve_prefix_state = _o.resolve_emitter_precision, _o.resolve_prefix_state
    from .fullstack_policy import fullstack_config
    fs = fullstack_config(model_config)
    if not fs:
        return
    fs.update(duet_emitter_precision=resolve_emitter_precision(cli),
              duet_prefix_state=resolve_prefix_state(cli))
    if 'duet_spec' not in fs:
        return
    from dataclasses import asdict
    from sglang.srt.duet import numerics
    text = getattr(model_config.hf_config, "text_config", model_config.hf_config)
    opts = _o.DuetOptions.resolve(fs['duet_spec'], cli,
                                  state_dim=getattr(text, "linear_key_head_dim", None))
    fs.update(asdict(opts))
    fs.update(gdn_rank=opts.decode_ssm_r, gdn_every=opts.decode_ssm_w,
              gdn_state=f"rank:{opts.decode_ssm_r}" if opts.decode_ssm_r else "dense")
    profile = numerics.profile_name(cli)
    truncation = getattr(cli, "duet_state_truncation", None) or os.environ.get(
        "SGLANG_DUET_STATE_TRUNCATION") or numerics.defaults(profile)["duet_state_truncation"]
    if truncation not in ("reference-warm", "factored-iter"):
        raise ValueError("unsupported Flash-Next state truncation")
    fs.update(duet_numerics=profile, duet_state_truncation=truncation,
              latent_compute_precision="fp32" if profile == "reference" else "tf32",
              emitter_state_only=profile == "production" and fs["duet_emitter_precision"] == "bf16",
              async_h2d=profile == "production")


def validate_duet_config(fs):
    spec = fs['duet_spec']
    for target, source in (('latent_rank', 'latent_rank'), ('latent_sparse', 'latent_spikes'),
                           ('latent_store', 'latent_z_format'), ('state_sink', 'state_sink'),
                           ('latent_id_side', 'latent_id_side'), ('latent_value_format', 'latent_value_format'),
                           ('latent_index_format', 'latent_index_format')):
        if fs.get(target) != spec[source]:
            raise ValueError(f'DUET config {target} differs from checkpoint spec')
    if fs.get('state_sink') != 'explicit' or fs.get('latent_rms') is not False:
        raise NotImplementedError('this adapter requires explicit state sink and unnormalized residual code')
    expected_state = f"rank:{fs['gdn_rank']}" if fs['gdn_rank'] else 'dense'
    if fs.get('gdn_state') != expected_state:
        raise ValueError('inconsistent decoded state rank')
    if fs.get('prefill_saving_policy') != 'kv-and-ssm':
        raise NotImplementedError('saving policy requires unimplemented deep SSM replay')
    return fs
