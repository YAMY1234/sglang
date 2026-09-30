"""Model-independent DUET serving options and environment precedence."""
from dataclasses import dataclass
import argparse
import os

POLICIES = ('latent-only', 'latent-and-kv', 'latent-and-ssm', 'kv-and-ssm')


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('1', 'true', 'on', 'yes'):
        return True
    if str(value).lower() in ('0', 'false', 'off', 'no'):
        return False
    raise ValueError(f'invalid DUET boolean: {value!r}')


def add_arguments(parser):
    parser.add_argument('--prefill-layer-trim', type=boolean, nargs='?', const=True, default=None)
    parser.add_argument('--no-prefill-layer-trim', dest='prefill_layer_trim', action='store_false', default=None)
    parser.add_argument('--prefill-saving-policy', choices=POLICIES)
    parser.add_argument('--decode-ssm-r', type=int)
    parser.add_argument('--decode-ssm-w', type=int)


@dataclass(frozen=True)
class DuetOptions:
    prefill_layer_trim: bool
    prefill_saving_policy: str
    decode_ssm_r: int
    decode_ssm_w: int

    @classmethod
    def resolve(cls, spec, args=None, environ=None):
        env = os.environ if environ is None else environ
        def get(name, fallback, cast):
            value = getattr(args, name, None)
            if value is None:
                value = env.get('SGLANG_DUET_' + name.upper(), fallback)
            return cast(value)
        result = cls(get('prefill_layer_trim', True, boolean),
                     get('prefill_saving_policy', 'kv-and-ssm', str),
                     get('decode_ssm_r', spec['state_rank'], int),
                     get('decode_ssm_w', spec['state_every'], int))
        if result.prefill_saving_policy not in POLICIES:
            raise ValueError('unknown DUET saving policy')
        if result.prefill_saving_policy != 'kv-and-ssm':
            raise NotImplementedError(f'{result.prefill_saving_policy}: prefix reconstruction is not implemented')
        if result.decode_ssm_r < 0 or result.decode_ssm_w < 0:
            raise ValueError('DUET decode r/W must be nonnegative; zero follows release exact/no-prune semantics')
        return result
