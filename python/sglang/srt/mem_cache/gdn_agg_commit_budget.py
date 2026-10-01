"""Explicit M10 reservation, shared by pre/post-capture KV accounting."""
import os

FLAG = "SGLANG_GDN_AGG_COMMIT_GRAPH"
RESERVE = "SGLANG_GDN_AGG_COMMIT_RESERVE_MB"


def reservation_bytes(role):
    value = int(os.environ.get(RESERVE, "0"))
    if value < 0:
        raise ValueError("AGG commit reservation must be nonnegative")
    enabled = os.environ.get(FLAG, "0") == "1"
    if (value or enabled) and role != "null":
        raise ValueError("AGG commit reservation is restricted to AGG")
    if enabled and not value:
        raise ValueError("AGG commit graph requires an explicit byte reservation")
    return value << 20


def unused_reservation_bytes(role, charged_bytes):
    """Already-resident graph bytes are in measured free memory; charge once."""
    reserved = reservation_bytes(role)
    if not 0 <= charged_bytes <= reserved:
        raise RuntimeError("AGG commit graph exceeded its admitted reservation")
    return reserved - charged_bytes
