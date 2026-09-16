# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GPU forward-time accounting for the scheduler (SGLANG_LOG_GPU_FORWARD_TIME).

Brackets every model forward with a pair of CUDA events on the current stream
and logs, every ``LOG_INTERVAL_S`` seconds, the fraction of wall time the GPU
spent inside forwards ("gpu_duty") and the GPU microseconds per new token. It
never synchronizes: completed event pairs are drained lazily with ``query()``,
so a forward still in flight is accounted in a later window. CUPTI-free, so it
works inside containers where the profiler cannot attach.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Tuple

import torch

logger = logging.getLogger(__name__)

LOG_INTERVAL_S = 30.0


class GpuForwardTimer:
    """Per-scheduler accumulator of GPU time spent in model forwards."""

    def __init__(self) -> None:
        now = time.monotonic()
        # (start_event, end_event, new_tokens) pairs not yet known complete.
        self._pending: Deque[Tuple[torch.cuda.Event, torch.cuda.Event, int]] = deque()
        self._window_fwd_ms = 0.0
        self._window_new_tok = 0
        self._window_calls = 0
        self._window_start = now
        self._total_fwd_ms = 0.0
        self._total_new_tok = 0
        self._total_start = now

    def begin(self) -> torch.cuda.Event:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def end(self, start_event: torch.cuda.Event, batch) -> None:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        new_tok = 0
        if batch is not None and batch.forward_mode.is_extend():
            new_tok = int(batch.extend_num_tokens or 0)
        self._pending.append((start_event, end_event, new_tok))
        self._drain_completed()
        self._maybe_log()

    def _drain_completed(self) -> None:
        # Only pop pairs whose end event has fired; never block on the GPU.
        while self._pending and self._pending[0][1].query():
            start_event, end_event, new_tok = self._pending.popleft()
            self._window_fwd_ms += start_event.elapsed_time(end_event)
            self._window_new_tok += new_tok
            self._window_calls += 1

    def _maybe_log(self) -> None:
        now = time.monotonic()
        window_s = now - self._window_start
        if window_s < LOG_INTERVAL_S:
            return
        self._total_fwd_ms += self._window_fwd_ms
        self._total_new_tok += self._window_new_tok
        total_s = now - self._total_start
        logger.info(
            "GPUTIME window=%.1fs fwd_gpu_ms=%.0f fwd_calls=%d new_tok=%d "
            "gpu_duty=%.1f%% us_per_new_tok=%.2f | cumulative fwd_gpu_s=%.1f "
            "wall_s=%.1f new_tok=%d gpu_duty=%.1f%% us_per_new_tok=%.2f",
            window_s,
            self._window_fwd_ms,
            self._window_calls,
            self._window_new_tok,
            _duty_pct(self._window_fwd_ms, window_s),
            _us_per_token(self._window_fwd_ms, self._window_new_tok),
            self._total_fwd_ms / 1000.0,
            total_s,
            self._total_new_tok,
            _duty_pct(self._total_fwd_ms, total_s),
            _us_per_token(self._total_fwd_ms, self._total_new_tok),
        )
        self._window_fwd_ms = 0.0
        self._window_new_tok = 0
        self._window_calls = 0
        self._window_start = now


def _duty_pct(fwd_ms: float, wall_s: float) -> float:
    return 100.0 * fwd_ms / 1000.0 / wall_s if wall_s > 0 else 0.0


def _us_per_token(fwd_ms: float, new_tok: int) -> float:
    return fwd_ms * 1000.0 / new_tok if new_tok else 0.0
