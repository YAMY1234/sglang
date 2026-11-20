"""Dump of the DSA indexer's prefill inputs for offline kernel work (off by default).

``SGLANG_DSA_INDEXER_DUMP_DIR`` turns it on for one DSA layer (``..._LAYER``; -1 takes the
first DSA layer this rank builds) on the prefill worker's first pipeline stage, attention-TP
rank 0. After ``..._SKIP`` prefill batches of that layer it saves the next ``..._BATCHES``
(one ``torch.save`` file each): FP8 queries, head weights, the gathered index keys and
scales, the per-row key windows, row-to-request map, page table, and the batch lengths.
Saving copies to the host (a sync), so only dumped batches pay anything; the env is read
once, when the layer is built.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg, get_parallel

logger = logging.getLogger(__name__)


class IndexerDump:
    _claimed = False  # one dumped layer per process

    def __init__(self, *, root: str, layer_id: int, skip: int, batches: int):
        self.dir = os.path.join(root, f"pp{get_parallel().pp_rank}_tp{get_parallel().tp_rank}_layer{layer_id}")
        self.layer_id = layer_id
        self.skip = skip
        self.batches = batches
        self.calls = 0
        os.makedirs(self.dir, exist_ok=True)
        logger.info("DSA indexer dump: layer %d, skip %d, %d batches -> %s", layer_id, skip, batches, self.dir)

    @classmethod
    def create(cls, layer_id: int) -> Optional["IndexerDump"]:
        root = envs.SGLANG_DSA_INDEXER_DUMP_DIR.get()
        if not root or cls._claimed:
            return None
        par = get_parallel()
        if get_disagg().disaggregation_mode == "decode" or par.pp_rank != 0 or par.attn_tp_rank != 0:
            return None
        want = envs.SGLANG_DSA_INDEXER_DUMP_LAYER.get()
        if want >= 0 and want != layer_id:
            return None
        cls._claimed = True
        return cls(
            root=root,
            layer_id=layer_id,
            skip=envs.SGLANG_DSA_INDEXER_DUMP_SKIP.get(),
            batches=envs.SGLANG_DSA_INDEXER_DUMP_BATCHES.get(),
        )

    def maybe_save(self, *, tensors: Dict[str, torch.Tensor], meta: Dict[str, object]) -> None:
        # Graph capture replays a dummy batch and cannot copy to the host: neither saved nor counted.
        if torch.cuda.is_current_stream_capturing():
            return
        n = self.calls - self.skip
        self.calls += 1
        if n < 0 or n >= self.batches:
            return
        payload = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
        payload["meta"] = dict(meta, layer_id=self.layer_id, call=self.calls - 1, time=time.time())
        path = os.path.join(self.dir, f"batch{n:03d}.pt")
        torch.save(payload, path)
        logger.info("DSA indexer dump: wrote %s (%d rows, %d keys)", path, meta.get("rows"), meta.get("keys"))
