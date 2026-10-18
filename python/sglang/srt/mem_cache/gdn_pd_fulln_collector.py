"""Collect S_N for shallow layers and S_(N-1) for deep emitters once.

Checkpoint donation stays at the shared N-1 extent. A checkpoint can come
from intermediate h on shallow layers and from the final state on deep
layers. Normalize those two representations before the one batch graph;
never copy a live shallow S_N row into a checkpoint labelled N-1.
"""
import logging

import torch

from .gdn_prefill_batch_graph import BatchCollector

logger = logging.getLogger(__name__)


class ShallowFullNCollector(BatchCollector):
    def __init__(self, *args, same_extent=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.same_extent = same_extent

    def add(self, layer_id, dense, tracked, track_slots, final_src, final_dst):
        if self.same_extent:
            # No prompt-final row: both trunks end at N. Preserve the native
            # h/final-copy branches, rather than factor the same final state
            # twice as a joint B=2 input. That changes GEMM grouping and the
            # admitted singleton mixed-eigh eligibility (r11 N128 evidence).
            controls = (track_slots, final_src, final_dst)
            if self.controls is not None:
                for current, canonical in zip(controls, self.controls):
                    if (current is None) != (canonical is None):
                        raise RuntimeError("same-extent checkpoint source changed")
                    if current is not None:
                        if current.shape != canonical.shape:
                            raise RuntimeError("same-extent checkpoint shape changed")
                        torch._assert_async(current.eq(canonical).all(),
                                            "same-extent checkpoint destinations changed")
                controls = self.controls
            elif final_src is not None and final_src.numel() and not getattr(
                    self.pool, "_pd_fulln_shared_final_checkpoint_logged", False):
                logger.info("GDN_PD_FULLN_SHARED_FINAL_CHECKPOINT_ACTIVE layers=%d",
                            len(self.pool.layer_ids))
                self.pool._pd_fulln_shared_final_checkpoint_logged = True
            return super().add(layer_id, dense, tracked, *controls)
        destinations, states = [], []
        if track_slots is not None and track_slots.numel():
            destinations.append(track_slots)
            states.append(tracked)
        if final_src is not None and final_src.numel():
            matches = final_src[:, None] == self.plan.slots[None, :]
            torch._assert_async(matches.any(dim=1).all(), "full-N checkpoint source outside batch")
            states.append(dense[matches.to(torch.int64).argmax(dim=1)])
            destinations.append(final_dst)
        if destinations:
            slots, values = torch.cat(destinations), torch.cat(states)
            if self.controls is None:
                self.checkpoint_slots = slots
            elif not hasattr(self, "checkpoint_slots"):
                raise RuntimeError("shallow/deep checkpoint branch changed")
            canonical = self.checkpoint_slots
            matches = canonical[:, None] == slots[None, :]
            if canonical.numel() != slots.numel():
                raise RuntimeError("shallow/deep checkpoint row count changed")
            torch._assert_async(matches.sum(dim=1).eq(1).all(), "full-N checkpoint destinations changed")
            tracked = values[matches.to(torch.int64).argmax(dim=1)]
            track_slots = canonical
        else:
            tracked = track_slots = None
        return super().add(layer_id, dense, tracked, track_slots, None, None)
