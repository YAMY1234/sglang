"""Debug-only dump of real ratio-1 compressed-KV latents (SGLANG_H282_KV_DUMP="<dir>:<tokens>", default off), for an
offline study of the fp4 cache's scale format. Eager forwards only (prefill / extend), never under CUDA-graph capture.
Per sampled token: the post-norm pre-RoPE latent (bf16, 512), its position, the layer's RoPE row (fp32, 64) and the fp4
cache row written for it (288 B), about 1.6 KB; at most 65536 tokens per rank."""

import logging
import os

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_MAX_TOKENS = 65536
_PER_CALL = 512


class _KVDump:
    def __init__(self, spec: str):
        path, _, n = spec.partition(":")
        self.dir = path
        self.left = min(int(n or 4096), _MAX_TOKENS)
        self.seq = 0
        os.makedirs(self.dir, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.tag = f"r{rank}_p{os.getpid()}"
        logger.warning(
            "SGLANG_H282_KV_DUMP: dumping up to %d ratio-1 KV tokens to %s",
            self.left,
            self.dir,
        )

    def add(self, layer_id, latent, positions, freqs_cis, k_cache, out_loc, page_size):
        """latent [n, 512] bf16; positions [n]; freqs_cis [max_pos, 64] fp32; k_cache the pool's (pages, page_bytes)
        buffer (interleaved, 288 B rows); out_loc [n] the written slots (0: padded row)."""
        if self.left <= 0 or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            return
        keep = (out_loc > 0).nonzero().flatten()[: min(self.left, _PER_CALL)]
        if keep.numel() == 0:
            return
        loc = out_loc[keep].long()
        pos = positions[keep].long()
        rows = k_cache.view(torch.uint8)[:, : page_size * 288].reshape(-1, 288)[loc]
        torch.save(
            {
                "layer": layer_id,
                "latent": latent[keep].cpu(),
                "positions": pos.cpu(),
                "freqs": freqs_cis[pos].cpu(),
                "fp4_rows": rows.cpu(),
            },
            f"{self.dir}/kv_{self.tag}_l{layer_id}_{self.seq:04d}.pt",
        )
        self.seq += 1
        self.left -= keep.numel()
        if self.left <= 0:
            logger.warning("SGLANG_H282_KV_DUMP: done (%d files)", self.seq)


_DUMP = None


def kv_dump():
    """The process's dumper, or None when SGLANG_H282_KV_DUMP is unset."""
    global _DUMP
    spec = envs.SGLANG_H282_KV_DUMP.get()
    if not spec:
        return None
    if _DUMP is None:
        _DUMP = _KVDump(spec)
    return _DUMP
