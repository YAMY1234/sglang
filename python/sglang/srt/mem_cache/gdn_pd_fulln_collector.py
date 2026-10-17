"""Collect S_N for shallow layers and S_(N-1) for deep emitters once.

Checkpoint donation stays at the shared N-1 extent. A checkpoint can come
from intermediate h on shallow layers and from the final state on deep
layers. Normalize those two representations before the one batch graph;
never copy a live shallow S_N row into a checkpoint labelled N-1.
"""
import torch

from .gdn_prefill_batch_graph import BatchCollector


class ShallowFullNCollector(BatchCollector):
    def add(self, layer_id, dense, tracked, track_slots, final_src, final_dst):
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
