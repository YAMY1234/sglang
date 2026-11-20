"""Dump of the DSA indexer's prefill inputs for offline kernel work (off by default).

``SGLANG_DSA_INDEXER_DUMP_DIR`` turns it on for one DSA layer (``..._LAYER``; -1 takes the
first DSA layer this rank builds) on the prefill worker's first pipeline stage, attention-TP
rank 0. After ``..._SKIP`` prefill batches of that layer it saves the next ``..._BATCHES``
(one ``torch.save`` file each): FP8 queries, head weights, the gathered index keys and
scales, the per-row key windows, row-to-request map, page table, and the batch lengths.
Saving copies to the host (a sync), so only dumped batches pay anything; the env is read
once, when the layer is built. It never raises into the forward: any error is logged once
and the dump turns itself off.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class IndexerDump:
    _claimed = False  # one dumped layer per process

    def __init__(self, *, directory: str, layer_id: int, skip: int, batches: int):
        self.dir = directory
        self.layer_id = layer_id
        self.skip = skip
        self.batches = batches
        self.calls = 0
        self.active = True

    @classmethod
    def create(cls, layer_id: int) -> Optional["IndexerDump"]:
        root = envs.SGLANG_DSA_INDEXER_DUMP_DIR.get()
        if not root or cls._claimed:
            return None
        try:
            from sglang.srt.runtime_context import get_disagg, get_parallel

            par = get_parallel()
            if get_disagg().disaggregation_mode == "decode" or par.pp_rank != 0 or par.attn_tp_rank != 0:
                return None
            want = envs.SGLANG_DSA_INDEXER_DUMP_LAYER.get()
            if want >= 0 and want != layer_id:
                return None
            directory = os.path.join(root, f"pp{par.pp_rank}_tp{par.tp_rank}_layer{layer_id}")
            os.makedirs(directory, exist_ok=True)
            cls._claimed = True
            dump = cls(
                directory=directory,
                layer_id=layer_id,
                skip=envs.SGLANG_DSA_INDEXER_DUMP_SKIP.get(),
                batches=envs.SGLANG_DSA_INDEXER_DUMP_BATCHES.get(),
            )
            logger.info(
                "DSA indexer dump: layer %d, skip %d, %d batches -> %s", layer_id, dump.skip, dump.batches, directory
            )
            return dump
        except Exception as e:  # noqa: BLE001 -- a debugging aid must not stop the server
            logger.warning("DSA indexer dump disabled at setup: %r", e)
            return None

    def capture(
        self,
        *,
        q_fp8,
        weights,
        k_fp8,
        k_scale,
        ks,
        ke,
        lengths,
        row_to_batch,
        metadata,
        forward_batch,
        rows,
        keys,
        page_size,
        index_topk,
    ) -> None:
        if not self.active:
            return
        try:
            # Graph capture replays a dummy batch and cannot copy to the host: neither saved nor counted.
            if torch.cuda.is_current_stream_capturing():
                return
            n = self.calls - self.skip
            self.calls += 1
            if n < 0:
                return
            if n >= self.batches:
                self.active = False
                return
            tensors = dict(
                q_fp8=q_fp8[:rows].view(torch.uint8),
                weights=weights[:rows],
                k_fp8=k_fp8.view(torch.uint8),
                k_scale=k_scale,
                ks=ks,
                ke=ke,
                lengths=lengths,
                row_to_batch=row_to_batch,
                page_table=metadata.get_page_table_64(),
            )
            payload = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
            payload["meta"] = dict(
                rows=rows,
                keys=keys,
                page_size=page_size,
                index_topk=index_topk,
                seq_lens=forward_batch.seq_lens_cpu.tolist(),
                extend_lens=list(forward_batch.extend_seq_lens_cpu),
                layer_id=self.layer_id,
                call=self.calls - 1,
                time=time.time(),
            )
            path = os.path.join(self.dir, f"batch{n:03d}.pt")
            torch.save(payload, path)
            logger.info("DSA indexer dump: wrote %s (%d rows, %d keys)", path, rows, keys)
        except Exception as e:  # noqa: BLE001 -- a debugging aid must not stop the server
            self.active = False
            logger.warning("DSA indexer dump disabled after an error: %r", e)
