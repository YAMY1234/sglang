"""In-graph shadow check for two decode switches (SGLANG_Q35MF_BUNDLE_PROOF).

fp8_pin:    a pinned FP8 bmm tactic (SGLANG_Q35MF_FP8_BMM_TACTICS) against the served (autotuned) kernel on
            the same activations;
norm_early: the early-load gated norm (SGLANG_Q35MF_GDN_NORM_EARLY) against the stock kernel on the same inputs.
Each check counts rows whose outputs differ in any byte. Counters live on the device and are mirrored into pinned
host memory by a copy that is part of the check, so a thread logs them every 30 s without CUDA calls (as the
K75P proof).
"""

import logging
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_PROOF = None


class BundleProof:
    PERIOD_S = 30.0
    KINDS = ("fp8_pin", "norm_early")

    def __init__(self):
        # [kind][calls, rows, mismatched rows]
        self.counters = torch.zeros(
            (2, 3), dtype=torch.int64, device=torch.cuda.current_device()
        )
        self.host_counters = torch.zeros(
            (2, 3), dtype=torch.int64, device="cpu", pin_memory=True
        )
        logger.info("Q35 bundle proof active")
        threading.Thread(target=self._report, daemon=True).start()

    def check(self, kind: int, out: torch.Tensor, ref: torch.Tensor) -> None:
        rows = out.shape[0]
        a = out.contiguous().view(torch.uint8).view(rows, -1)
        b = ref.contiguous().view(torch.uint8).view(rows, -1)
        counters = self.counters[kind]
        counters[0].add_(1)
        counters[1].add_(rows)
        counters[2].add_(torch.count_nonzero((a != b).any(dim=1)))
        self.host_counters.copy_(self.counters, non_blocking=True)

    def _report(self) -> None:
        while True:
            time.sleep(self.PERIOD_S)
            for kind, (calls, rows, mismatched) in zip(
                self.KINDS, self.host_counters.tolist()
            ):
                logger.info(
                    "Q35 bundle proof: kind=%s calls=%d rows=%d mismatched=%d",
                    kind,
                    calls,
                    rows,
                    mismatched,
                )


def maybe_init() -> None:
    """Create the counters at model init when the process env or any A/B variant turns the switch on."""
    global _PROOF
    if _PROOF is not None or not torch.cuda.is_available():
        return
    from sglang.srt.model_executor.runner_backend import ab_variants

    if envs.SGLANG_Q35MF_BUNDLE_PROOF.get() or ab_variants.env_any(
        "SGLANG_Q35MF_BUNDLE_PROOF", "1"
    ):
        _PROOF = BundleProof()


def active():
    """The proof object while the switch is on for the current call (variants apply their env at capture)."""
    if _PROOF is not None and envs.SGLANG_Q35MF_BUNDLE_PROOF.get():
        return _PROOF
    return None
