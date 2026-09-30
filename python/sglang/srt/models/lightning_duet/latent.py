"""Scheme C residual code with actual NVFP4/BF16/gap8 storage -- moved to sglang.srt.duet.latent_codec (docs/162 §3.5, F5).

Arithmetic/rounding contract: twinstar.duet.latent{,fmt} at dd9c7bdbd955.  This module keeps the Lightning line's
import path and names; `ResidualCode` here is the packed (real-storage) code.
"""
from ._common import load as _load

_codec = _load("latent_codec")
EDGES = _codec.EDGES
LEVELS = _codec.LEVELS
LatentRecord = _codec.LatentRecord
ResidualCode = _codec.PackedResidualCode
pack_gap8 = _codec.pack_gap8
pack_nvfp4 = _codec.pack_nvfp4
unpack_gap8 = _codec.unpack_gap8
unpack_nvfp4 = _codec.unpack_nvfp4
