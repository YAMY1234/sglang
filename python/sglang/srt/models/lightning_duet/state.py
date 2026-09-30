"""Compatibility names for the common DUET left-sink factor and slot modules."""

from ._common import load as _load

_factor = _load("state_factor")
factorize = _factor.factorize_left
orthonormalize = _factor.orthonormalize
LightningMambaStatePool = _load("state_pool").LeftSinkStatePool
