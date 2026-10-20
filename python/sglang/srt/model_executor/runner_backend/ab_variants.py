"""Same-service pre-captured A/B graph variants: a measurement harness, off unless Q35MF_AB_VARIANTS is set.

Q35MF_AB_VARIANTS  JSON object, variant name -> env overrides, e.g.
                   '{"A": {}, "B": {"SGLANG_Q35MF_FOO": "1"}, "A2": {}}'  (A2 = same as A: the in-situ null)
Q35MF_AB_PATTERN   comma list of variant names, one per block, cycled (default: the variant names in order)
Q35MF_AB_BLOCK     decode rounds per block (default 200)

FullCudaGraphBackend captures every decode-phase graph (target verify, EAGLE draft, draft extend) once per
variant, with that variant's env applied, so a feature must read its switch at capture time. Replays use the
variant of the current block; the block counter advances on the target decode runner's replays, so all TP
ranks switch on the same round. A block's last round replays its post-verify graphs with the next variant:
drop it in analysis. A later-captured copy of an identical graph measured up to ~0.03 ms/round slower, so
capture arms in mirrored order (A, B, B2, A2).
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_spec = os.environ.get("Q35MF_AB_VARIANTS")
VARIANTS = json.loads(_spec) if _spec else {}
PATTERN = [
    p.strip()
    for p in os.environ.get("Q35MF_AB_PATTERN", ",".join(VARIANTS)).split(",")
    if p.strip()
]
BLOCK = int(os.environ.get("Q35MF_AB_BLOCK", "200"))
assert all(p in VARIANTS for p in PATTERN), f"Q35MF_AB_PATTERN {PATTERN} not in {list(VARIANTS)}"
_round = 0
_base_env = {}


def enabled():
    return bool(VARIANTS)


def names():
    return list(VARIANTS)


def activate(name):
    """Apply one variant's env on top of the original env (keys of other variants restored)."""
    for overrides in VARIANTS.values():
        for k in overrides:
            _base_env.setdefault(k, os.environ.get(k))
    restore()
    for k, v in VARIANTS[name].items():
        os.environ[k] = str(v)


def restore():
    for k, v in _base_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def current():
    return PATTERN[(_round // BLOCK) % len(PATTERN)]


def env(key, default=None):
    """The current variant's override of an env switch, else the process env (for switches read at runtime)."""
    if VARIANTS and key in VARIANTS[current()]:
        return str(VARIANTS[current()][key])
    return os.environ.get(key, default)


def env_any(key, value):
    """True when the process env or any variant sets the switch to value."""
    return os.environ.get(key) == value or any(o.get(key) == value for o in VARIANTS.values())


def round_index():
    return _round


def advance():
    global _round
    _round += 1
    if _round % BLOCK == 0:
        logger.info("Q35MF A/B block %d: variant %s from round %d", _round // BLOCK, current(), _round)
