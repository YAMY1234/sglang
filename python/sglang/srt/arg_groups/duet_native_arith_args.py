"""Opt-in DUET arithmetic CLI parsing; no model hooks or environment overrides."""

import argparse


DUET_NATIVE_ARITH_COMPONENTS = ("router", "rmsnorm", "state")


def parse_disabled_components(value: str) -> tuple[str, ...]:
    """Parse one comma-separated CLI value into a canonical immutable tuple.

    Supplying this option alone does not enable native arithmetic. Omit the
    option for an empty tuple; an explicit empty string also clears the list.
    Whitespace around names is ignored; unknown names and empty entries fail.
    """
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("expected a comma-separated component string")
    if not value.strip():
        return ()
    values = [item.strip() for item in value.split(",")]
    unknown = sorted(set(values) - set(DUET_NATIVE_ARITH_COMPONENTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            "invalid DUET arithmetic component(s): "
            + ", ".join(repr(item) for item in unknown)
            + "; choose router,rmsnorm,state"
        )
    return tuple(name for name in DUET_NATIVE_ARITH_COMPONENTS if name in values)
