"""Buffered, synchronization-free per-forward attention-DP observations."""

import atexit
import json
import os
import socket
import time
from collections import deque
from pathlib import Path

import torch

DADP_STEPLOG_DIR = os.environ.get("SGLANG_DADP_STEPLOG_DIR", "")


class DadpStepLogger:
    def __init__(self, directory, *, dp_rank, tp_rank, pp_rank):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.disabled_file = directory / "disabled"
        self.path = directory / (
            f"steps-{socket.gethostname()}-dp{dp_rank}-tp{tp_rank}-pp{pp_rank}-{os.getpid()}.jsonl"
        )
        self.file = self.path.open("x", buffering=1024 * 1024)
        self.ranks = dict(dp_rank=dp_rank, tp_rank=tp_rank, pp_rank=pp_rank)
        self.pending = deque()
        self.free_events = []
        self.step = 0
        self.closed = False
        atexit.register(self.close)

    def begin(self, batch, stream):
        self.drain()
        self.step += 1
        # A node-local switch permits a same-server observer overhead test.
        if self.disabled_file.exists():
            return None
        mode = batch.forward_mode
        decoding = (
            {id(r) for r in batch.decoding_reqs or []} if mode.is_mixed() else set()
        )
        if mode.is_decode():
            decode_lens = (
                batch.seq_lens_cpu.tolist()
                if batch.seq_lens_cpu is not None
                else [r.seqlen for r in batch.reqs]
            )
            extend = []
        elif mode.is_extend():
            extend = [
                [prefix, length]
                for req, prefix, length in zip(
                    batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True
                )
                if id(req) not in decoding
            ]
            decode_lens = [r.seqlen for r in batch.reqs if id(r) in decoding]
        else:
            extend, decode_lens = [], []
        record = dict(
            schema="sglang-dadp-step/1",
            step=self.step,
            forward_iter=batch.forward_iter,
            cpu_time_ns=time.time_ns(),
            cpu_monotonic_ns=time.monotonic_ns(),
            forward_mode=mode.name,
            request_rids=[r.rid for r in batch.reqs],
            extend=extend,
            decode_seq_lens=decode_lens,
            **self.ranks,
            **getattr(batch, "dadp_sync_metadata", {}),
        )
        events = (
            self.free_events.pop()
            if self.free_events
            else (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        )
        start, end = events
        start.record(stream)
        return record, start, end

    def end(self, observation, stream):
        if observation is None:
            return
        record, start, end = observation
        end.record(stream)
        record["cpu_enqueue_end_ns"] = time.monotonic_ns()
        self.pending.append((record, start, end))

    def drain(self, *, closing=False, flush=False):
        while self.pending:
            record, start, end = self.pending[0]
            if end.query():
                record["gpu_forward_ms"] = start.elapsed_time(end)
                record["gpu_timing_status"] = "complete"
                self.free_events.append((start, end))
            elif closing:
                record["gpu_forward_ms"] = None
                record["gpu_timing_status"] = "pending_at_exit"
            else:
                break
            self.pending.popleft()
            self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        if closing or flush:
            self.file.flush()

    def close(self):
        if not self.closed:
            self.drain(closing=True)
            self.file.close()
            self.closed = True
