"""In-graph shadow check for the KRN-C L2 prefetch switches (SGLANG_Q35MF_KRNC_PROOF).

h0_pf:    the GDN verify with SGLANG_Q35MF_KRNC_H0_PF against the stock verify launch on the same inputs;
in_proj:  the GDN in_proj output after the SGLANG_Q35MF_KRNC_L2PF_LL prefetch against the same GEMM rerun;
out_proj: the GDN out_proj output after the SGLANG_Q35MF_KRNC_OPROJ_PF prefetch against the same GEMM rerun;
router:   the DEP router logits from the stock gate GEMM after a SGLANG_Q35MF_KRNC_ROUTER_PF_* prefetch against the
          same GEMM rerun.
Each check counts rows whose outputs differ in any byte; device counters are mirrored into pinned host memory by a
copy that is part of the check, and a thread logs them every 30 s without CUDA calls (as q35_bundle_proof).
"""

import logging
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_PROOF = None


class KrncProof:
    PERIOD_S = 30.0
    KINDS = ("h0_pf", "in_proj", "out_proj", "router")

    def __init__(self):
        # [kind][calls, rows, mismatched rows]
        self.counters = torch.zeros(
            (len(self.KINDS), 3), dtype=torch.int64, device=torch.cuda.current_device()
        )
        self.host_counters = torch.zeros(
            (len(self.KINDS), 3), dtype=torch.int64, device="cpu", pin_memory=True
        )
        logger.info("Q35 KRN-C proof active")
        threading.Thread(target=self._report, daemon=True).start()

    def check(self, kind: str, out: torch.Tensor, ref: torch.Tensor) -> None:
        rows = out.shape[0]
        a = out.contiguous().view(torch.uint8).view(rows, -1)
        b = ref.contiguous().view(torch.uint8).view(rows, -1)
        counters = self.counters[self.KINDS.index(kind)]
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
                    "Q35 KRN-C proof: kind=%s calls=%d rows=%d mismatched=%d",
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

    if envs.SGLANG_Q35MF_KRNC_PROOF.get() or ab_variants.env_any(
        "SGLANG_Q35MF_KRNC_PROOF", "1"
    ):
        _PROOF = KrncProof()


def active():
    """The proof object while the switch is on for the current call (variants apply their env at capture)."""
    if _PROOF is not None and envs.SGLANG_Q35MF_KRNC_PROOF.get():
        return _PROOF
    return None


def check_router(gate, hidden_states: torch.Tensor, logits: torch.Tensor) -> None:
    """Router kind; call only where the stock gate GEMM produced `logits` from `hidden_states`."""
    if _PROOF is None or not envs.SGLANG_Q35MF_KRNC_PROOF.get():
        return
    if not (
        envs.SGLANG_Q35MF_KRNC_ROUTER_PF_CORE.get()
        or envs.SGLANG_Q35MF_KRNC_ROUTER_PF_PRE.get()
    ):
        return
    ref, _ = gate(hidden_states)
    _PROOF.check("router", logits, ref)
