"""Model-independent DUET serving options -- moved to sglang.srt.duet.options (docs/162 §3.5, F5); this module keeps
the Lightning line's import path."""
from ._common import load as _load

_options = _load("options")
DuetOptions = _options.DuetOptions
ENV_PREFIX = _options.ENV_PREFIX
POLICIES = _options.POLICIES
add_arguments = _options.add_arguments
boolean = _options.boolean
