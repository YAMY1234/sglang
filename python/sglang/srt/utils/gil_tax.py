"""SGLANG_DEBUG_GIL_TAX: a background thread that holds the GIL for a fraction of
wall time, in bursts of SGLANG_DEBUG_GIL_TAX_BURST_MS. It measures what a
background thread's GIL duty (such as the external linker's I/O threads) costs
the scheduler. Each burst is one C-level call that does not yield the GIL, like
a numpy or pybind call. Both values are re-read every cycle (POST /linker_flags
switches them), and the held fraction is logged every 60 s."""

import logging
import threading
import time

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_started = False


def _calibrate(burst_s: float) -> int:
    n = 200_000
    t = time.perf_counter()
    sum(range(n))
    return max(1000, int(n * burst_s / max(time.perf_counter() - t, 1e-6)))


def tax_sleep_s(duty: float, burst_s: float) -> float:
    """Sleep after each burst so that bursts take ``duty`` of the cycle."""
    return burst_s * (1.0 - duty) / duty


def _run() -> None:
    n, n_burst = 0, None
    held = 0.0
    t0 = next_log = time.perf_counter()
    while True:
        duty = envs.SGLANG_DEBUG_GIL_TAX.get()
        burst_s = envs.SGLANG_DEBUG_GIL_TAX_BURST_MS.get() / 1e3
        now = time.perf_counter()
        if now >= next_log:
            if now > t0:
                logger.info(
                    "GIL tax: target duty %.3f burst %.2f ms, held %.4f of the last %.0f s",
                    duty, burst_s * 1e3, held / (now - t0), now - t0,
                )
            held, t0, next_log = 0.0, now, now + 60.0
        if duty <= 0:
            time.sleep(0.5)
            continue
        if n_burst != burst_s:
            n, n_burst = _calibrate(burst_s), burst_s
        t = time.perf_counter()
        sum(range(n))
        held += time.perf_counter() - t
        time.sleep(tax_sleep_s(min(duty, 0.9), burst_s))


def maybe_start_gil_tax() -> None:
    """Start the thread once per process; it idles while the duty is 0."""
    global _started
    if _started or envs.SGLANG_DEBUG_GIL_TAX.get() < 0:
        return
    _started = True
    threading.Thread(target=_run, name="gil-tax", daemon=True).start()
