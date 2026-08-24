# Temporarily do this to avoid changing all imports in the repo
from sglang.srt.utils.common import *
from sglang.srt.utils.network import (
    get_local_ip_auto,
    get_zmq_socket,
    is_valid_ipv6_address,
)


def maybe_wrap_ipv6_address(address: str) -> str:
    """Keep the pre-split public network helper available to integrations."""
    if is_valid_ipv6_address(address):
        return f"[{address}]"
    return address
