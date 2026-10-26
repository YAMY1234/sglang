"""Same-service pre-captured graph variants for timing experiments.

Q35MF_AB_VARIANTS  JSON object, variant name -> env overrides. Empty: off.
Q35MF_AB_PATTERN   comma list of variant names, one per block, cycled.
Q35MF_AB_BLOCK     target-verify rounds per block (default 200).
Q35MF_AB_PIN_FILE  optional file naming one pinned variant.
Q35MF_AB_BENCH_ALIAS_GRAPH  bench-only: identical all-off labels share one graph.

FullCudaGraphBackend captures every decode-phase graph once per variant while
that variant's environment is active. Features under test therefore must read
their switch at graph-capture time. The target decode runner alone advances
the block counter. The final round of a block can replay post-verify graphs
from the next variant and must be discarded by analysis.

The older SGLANG_TRTL17_PAD_AB harness remains available independently. Its
ACTIVE/pad/tick API is intentionally kept separate so generic variants are not
advanced twice by DecodeCudaGraphRunner.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_spec = os.environ.get("Q35MF_AB_VARIANTS")
VARIANTS = json.loads(_spec) if _spec else {}
PATTERN = [
    item.strip()
    for item in os.environ.get("Q35MF_AB_PATTERN", ",".join(VARIANTS)).split(",")
    if item.strip()
]
BLOCK = int(os.environ.get("Q35MF_AB_BLOCK", "200"))
assert all(item in VARIANTS for item in PATTERN), (
    f"Q35MF_AB_PATTERN {PATTERN} not in {list(VARIANTS)}"
)
_round = 0
_base_env: dict[str, str | None] = {}
_BENCH_ALIAS_GRAPH = os.environ.get("Q35MF_AB_BENCH_ALIAS_GRAPH", "0") == "1"

# Legacy graph-padding A/B. Keep ACTIVE limited to this mode: the generic
# backend owns its own replay counter and must not also take the old tick().
_LEGACY_LABELS = tuple(envs.SGLANG_TRTL17_PAD_AB.get())
ACTIVE = bool(_LEGACY_LABELS)


def enabled() -> bool:
    """Whether generic pre-captured graph variants are active."""
    return bool(VARIANTS)


def names() -> list[str]:
    return list(VARIANTS)


def bench_alias_source() -> str | None:
    """Null diagnostic only: one capture, identical declared all-off switches.

    This removes independent graph identity from a label/replay timing test.
    It is not a candidate optimization or an independent-capture control.
    """
    if not _BENCH_ALIAS_GRAPH:
        return None
    if not VARIANTS:
        raise ValueError("bench graph aliases require Q35MF_AB_VARIANTS")
    first = next(iter(VARIANTS))
    baseline = VARIANTS[first]
    if not baseline or any(str(value) != "0" for value in baseline.values()):
        raise ValueError("bench graph aliases require explicit all-off overrides")
    if any(overrides != baseline for overrides in VARIANTS.values()):
        raise ValueError("bench graph aliases require identical overrides")
    return first


def activate(name: str) -> None:
    """Apply one variant on top of the original process environment."""
    for overrides in VARIANTS.values():
        for key in overrides:
            _base_env.setdefault(key, os.environ.get(key))
    restore()
    for key, value in VARIANTS[name].items():
        os.environ[key] = str(value)


def restore() -> None:
    for key, value in _base_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


_PIN_FILE = os.environ.get("Q35MF_AB_PIN_FILE")
_pin: list[str | None | float] = [None, 0.0]


def _pinned() -> str | None:
    now = time.monotonic()
    if now - float(_pin[1]) >= 1.0:
        _pin[1] = now
        try:
            with open(_PIN_FILE) as handle:
                name = handle.read().strip()
        except OSError:
            name = ""
        name = name if name in VARIANTS else None
        if name != _pin[0]:
            logger.info("Q35MF A/B pin: %s", name or "none (pattern)")
        _pin[0] = name
    return _pin[0] if isinstance(_pin[0], str) else None


def current() -> str:
    if VARIANTS:
        if _PIN_FILE:
            name = _pinned()
            if name is not None:
                return name
        return PATTERN[(_round // BLOCK) % len(PATTERN)]
    return _LEGACY_LABELS[(_round // BLOCK) % len(_LEGACY_LABELS)]


def env(key: str, default: str | None = None) -> str | None:
    """Current variant override, or the process environment."""
    if VARIANTS and key in VARIANTS[current()]:
        return str(VARIANTS[current()][key])
    return os.environ.get(key, default)


def switch(cached: bool, key: str) -> bool:
    """Re-read a cached boolean switch while variants are being captured."""
    if not VARIANTS:
        return cached
    return os.environ.get(key, "0").lower() in ("1", "true")


def env_any(key: str, value: str) -> bool:
    """Whether the process or any declared variant sets key to value."""
    return os.environ.get(key) == value or any(
        overrides.get(key) == value for overrides in VARIANTS.values()
    )


def round_index() -> int:
    return _round


def advance() -> None:
    """Advance one target-model decode round for generic variants."""
    global _round
    _round += 1
    if _round % BLOCK == 0:
        logger.info(
            "Q35MF A/B block %d: variant %s from round %d",
            _round // BLOCK,
            current(),
            _round,
        )


def tick() -> None:
    """Advance one target-verify round for the legacy padding harness."""
    global _round
    _round += 1


def pad(raw_size: int, buckets: Sequence[int]) -> int | None:
    """Return the legacy ``pow2`` bucket, or None for the default."""
    if current() != "pow2" or buckets[0] != 1 or 2 not in buckets:
        return None
    return next(
        (
            bucket
            for bucket in buckets
            if bucket >= raw_size and bucket & (bucket - 1) == 0
        ),
        None,
    )
