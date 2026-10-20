"""Request broadcast carried on the HiCache ready-count exchange (default off).

With tensor parallelism across two hosts, every scheduler pass runs two cross-host CPU
syncs: the request broadcast (gloo broadcast_pyobj, also when nothing was received) and the
HiCache ready-count all-gather (HierTagExchange). With SGLANG_Q35MF_LL16_RECV_MERGE=1 the
payload size of the broadcast travels in the count all-gather, which then runs once during
request receipt; gloo only moves a non-empty payload. The count rows are kept for the cache's
own sync later in the same pass, so that sync does not exchange again.

SGLANG_Q35MF_LL16_RECV_AB_BLOCK=N (measurement only) alternates the stock path and the merged
path every N passes, starting with the stock path. Every rank counts the same passes, so all
ranks switch together. When Q35MF_RT_DIR is set, the switch times (perf_counter_ns) are appended
to $Q35MF_RT_DIR/ll16-recvab-r<rank>.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from typing import List, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.shm_tag_exchange import HierTagExchange

logger = logging.getLogger(__name__)


class _State:
    def __init__(self):
        self.enabled = envs.SGLANG_Q35MF_LL16_RECV_MERGE.get()
        self.block = envs.SGLANG_Q35MF_LL16_RECV_AB_BLOCK.get()
        self.cache = None
        self.passes = 0
        self.checked = False
        self.src_row: Optional[int] = None
        self.switches: List[List[int]] = []


_S = _State()


def register_cache(cache) -> None:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if _S.enabled and isinstance(cache, UnifiedRadixCache):
        _S.cache = cache


def _merged_this_pass() -> bool:
    p = _S.passes
    _S.passes += 1
    if _S.block <= 0:
        return True
    merged = (p // _S.block) % 2 == 1
    if p % _S.block == 0:
        _S.switches.append([time.perf_counter_ns(), int(merged)])
        if len(_S.switches) >= 8:
            _dump_switches()
    return merged


def _dump_switches() -> None:
    out = os.environ.get("Q35MF_RT_DIR")
    if not out:
        return
    # Append only the switches since the last dump; one JSON line per switch.
    new, _S.switches = _S.switches, []
    try:
        with open(os.path.join(out, f"ll16-recvab-r{dist.get_rank()}.jsonl"), "a") as f:
            f.writelines(json.dumps(dict(t_ns=t, merged=m, block=_S.block)) + "\n" for t, m in new)
    except OSError:
        pass


def _exchange(tp_group) -> Optional[HierTagExchange]:
    # Same answer on every rank: it depends on the group layout only.
    exchange = _S.cache._shm_count_exchange()
    if not isinstance(exchange, HierTagExchange):
        return None
    if not _S.checked:
        _S.checked = True
        ranks = dist.get_process_group_ranks(_S.cache._shm_count_group())
        if sorted(ranks) == sorted(tp_group.ranks):
            _S.src_row = ranks.index(tp_group.ranks[0])
            logger.info("Request broadcast merged into the HiCache count exchange")
    return exchange if _S.src_row is not None else None


def broadcast(recv_reqs: Optional[List], *, tp_group) -> Optional[List]:
    """Same result as broadcast_pyobj(recv_reqs, ..., src=tp_group.ranks[0]), or None when
    this pass takes the stock path."""
    if _S.cache is None:
        return None
    if not _merged_this_pass():
        _S.cache._ll16_count_rows = None
        return None
    exchange = _exchange(tp_group)
    if exchange is None:
        return None
    values = _S.cache._hicache_count_values()
    if 1 + len(values) > exchange.local.width:
        return None
    is_src = tp_group.rank == tp_group.ranks[0]
    payload = pickle.dumps(recv_reqs) if is_src and recv_reqs else b""
    rows = exchange.allgather([len(payload), *values])
    _S.cache._ll16_count_rows = [row[1:] for row in rows]
    size = rows[_S.src_row][0]
    if size == 0:
        return recv_reqs if is_src else []
    if is_src:
        data = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        dist.broadcast(data, src=tp_group.ranks[0], group=tp_group.cpu_group)
        return recv_reqs
    data = torch.empty(size, dtype=torch.uint8)
    dist.broadcast(data, src=tp_group.ranks[0], group=tp_group.cpu_group)
    return pickle.loads(data.numpy().tobytes())
