"""Opt-in PD page geometry; AGG keeps the existing compressed-QSA policy."""


def qsa_page_size(config, backend):
    if not getattr(config, 'flashnext_pd_page256', False):
        return 64
    if config.disaggregation_mode not in ('prefill', 'decode'):
        raise ValueError('Flash-Next page256 is a PD-only switch')
    if backend != 'triton':
        raise ValueError('Flash-Next PD page256 requires the triton QSA backend')
    return 256


def validate_arena_page(page_size, config):
    expected = qsa_page_size(config, 'triton')
    if page_size not in (64, 256) or (page_size == 256 and expected != 256):
        raise ValueError('shared arena page256 requires the explicit PD switch')
