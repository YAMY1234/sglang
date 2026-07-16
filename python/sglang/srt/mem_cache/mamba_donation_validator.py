"""Non-blocking validation for Mamba slot donations."""

import logging

import torch

logger = logging.getLogger(__name__)

MAMBA_DONATION_CHECK_INTERVAL = 64
_MAX_FAILURES_TO_LOG = 4


class MambaDonationValidator:
    """Batch device scalar checks without waiting for their stream."""

    def __init__(self, check_interval: int = MAMBA_DONATION_CHECK_INTERVAL):
        if check_interval <= 0:
            raise ValueError("check_interval must be positive")

        self._check_interval = check_interval
        self._building_values: list[torch.Tensor] = []
        self._building_metadata: list[tuple[str, str, int, int]] = []

        self._copy_done = None
        self._host_values = None
        self._inflight_batch = None
        self._inflight_metadata: list[tuple[str, str, int, int]] = []
        self._copy_pending = False
        self._flush_requested = False
        self._observations_since_poll = 0
        self._dropped_observations = 0

    def observe(
        self,
        value: torch.Tensor,
        *,
        kind: str,
        rid: object,
        slot_idx: int,
        next_track_idx: int,
    ) -> None:
        """Queue a scalar snapshot for delayed validation."""
        if value.numel() != 1:
            raise ValueError(f"Expected one Mamba slot value, got shape {value.shape}")

        metadata = (kind, str(rid), slot_idx, next_track_idx)
        if value.device.type == "cpu":
            if value.reshape(-1).tolist()[0] == -1:
                self._log_failures([metadata])
            return
        if not value.is_cuda:
            return

        self._maybe_poll()
        if len(self._building_values) >= self._check_interval:
            self._dropped_observations += 1
            if self._dropped_observations == 1:
                logger.warning(
                    "Mamba slot validation is still pending; dropping additional "
                    "checks to keep the scheduler non-blocking"
                )
            return

        self._building_values.append(value.reshape(-1))
        self._building_metadata.append(metadata)
        self._flush_requested |= kind == "finished"
        if not self._copy_pending and (
            len(self._building_values) == self._check_interval or self._flush_requested
        ):
            self._start_copy()

    def flush(self) -> None:
        """Submit a partial batch without waiting for its device stream."""
        if self._building_values:
            self._flush_requested = True
        self.poll()

    def poll(self) -> None:
        """Poll a submitted copy once without waiting."""
        if self._copy_pending and self._copy_done.query():
            self._finish_copy()
        if not self._copy_pending and self._building_values and self._flush_requested:
            self._start_copy()

    def _maybe_poll(self) -> None:
        if not self._copy_pending:
            return

        self._observations_since_poll += 1
        if self._observations_since_poll < self._check_interval:
            return
        self._observations_since_poll = 0

        if self._copy_done.query():
            self._finish_copy()
        if (
            not self._copy_pending
            and self._building_values
            and (
                len(self._building_values) == self._check_interval
                or self._flush_requested
            )
        ):
            self._start_copy()

    def _start_copy(self) -> None:
        assert not self._copy_pending
        assert 0 < len(self._building_values) <= self._check_interval

        batch = torch.cat(self._building_values)
        if (
            self._host_values is None
            or self._host_values.dtype != batch.dtype
            or self._host_values.numel() != batch.numel()
        ):
            self._host_values = torch.empty(
                batch.numel(),
                dtype=batch.dtype,
                device="cpu",
                pin_memory=True,
            )
        if self._copy_done is None:
            self._copy_done = torch.cuda.Event()

        self._host_values.copy_(batch, non_blocking=True)
        self._copy_done.record(torch.cuda.current_stream(batch.device))

        self._inflight_batch = batch
        self._inflight_metadata = self._building_metadata
        self._building_values = []
        self._building_metadata = []
        self._copy_pending = True
        self._flush_requested = False
        self._observations_since_poll = 0

    def _finish_copy(self) -> None:
        host_values = self._host_values.tolist()
        failures = [
            self._inflight_metadata[i]
            for i, value in enumerate(host_values)
            if value == -1
        ]
        if failures:
            self._log_failures(failures)
        if self._dropped_observations:
            logger.warning(
                "Skipped %d Mamba donation checks while a prior async check was pending",
                self._dropped_observations,
            )
            self._dropped_observations = 0

        self._inflight_batch = None
        self._inflight_metadata = []
        self._copy_pending = False

    @staticmethod
    def _log_failures(failures: list[tuple[str, str, int, int]]) -> None:
        details = "; ".join(
            f"{kind}(rid={rid}, slot_idx={slot_idx}, next_track_idx={next_track_idx})"
            for kind, rid, slot_idx, next_track_idx in failures[:_MAX_FAILURES_TO_LOG]
        )
        if len(failures) > _MAX_FAILURES_TO_LOG:
            details += f"; ... {len(failures) - _MAX_FAILURES_TO_LOG} more"
        logger.error(
            "Detected %d invalid (-1) Mamba slot donation(s) asynchronously: %s. "
            "Set SGLANG_DEBUG_MAMBA_DONATE=1 for immediate fail-fast checks.",
            len(failures),
            details,
        )
