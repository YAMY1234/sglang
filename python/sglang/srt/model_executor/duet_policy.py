"""Checkpoint-driven DUET controls shared by CLI, environment and model adapters."""
import os

POLICIES = ('latent-only', 'latent-and-kv', 'latent-and-ssm', 'kv-and-ssm')


def resolve_duet_options(spec, cli=None, environ=None):
    env = os.environ if environ is None else environ
    def value(name, default):
        explicit = getattr(cli, name, None) if cli is not None else None
        return explicit if explicit is not None else env.get('SGLANG_DUET_' + name.upper(), default)
    trim = value('prefill_layer_trim', True)
    if isinstance(trim, str):
        if trim.lower() not in ('0', '1', 'true', 'false'):
            raise ValueError('prefill-layer-trim must be true/false or 1/0')
        trim = trim.lower() in ('1', 'true')
    if type(trim) is not bool:
        raise ValueError('prefill-layer-trim must be boolean')
    policy = value('prefill_saving_policy', 'kv-and-ssm')
    if policy not in POLICIES:
        raise ValueError('unknown prefill-saving-policy')
    if policy != 'kv-and-ssm':
        raise NotImplementedError(f'{policy}: generic persistent latent/replay pool is not implemented in this adapter')
    if not trim and policy != 'kv-and-ssm':
        raise NotImplementedError('untrimmed prefill currently requires kv-and-ssm')
    values = dict(prefill_layer_trim=trim, prefill_saving_policy=policy)
    for name, key in (('decode_ssm_r', 'state_rank'), ('decode_ssm_w', 'state_every')):
        number = value(name, spec[key])
        if isinstance(number, str):
            number = int(number)
        if type(number) is not int or number <= 0:
            raise ValueError(f'{name} must be a positive integer')
        values[name] = number
    if values['decode_ssm_r'] + values['decode_ssm_w'] > 32:
        raise NotImplementedError('factored recurrence currently supports r + W <= 32')
    return values


def apply_duet_options(model_config, cli):
    from sglang.srt.duet.options import resolve_emitter_precision, resolve_prefix_state
    from .fullstack_policy import fullstack_config
    fs = fullstack_config(model_config)
    if not fs:
        return
    fs.update(duet_emitter_precision=resolve_emitter_precision(cli),
              duet_prefix_state=resolve_prefix_state(cli))
    if 'duet_spec' not in fs:
        return
    opts = resolve_duet_options(fs['duet_spec'], cli)
    fs.update(opts)
    fs.update(gdn_rank=opts['decode_ssm_r'], gdn_every=opts['decode_ssm_w'],
              gdn_state=f"rank:{opts['decode_ssm_r']}")


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
    if fs.get('gdn_state') != f"rank:{fs['gdn_rank']}":
        raise ValueError('inconsistent decoded state rank')
    if fs.get('prefill_saving_policy') != 'kv-and-ssm':
        raise NotImplementedError('saving policy requires unimplemented deep SSM replay')
    return fs
