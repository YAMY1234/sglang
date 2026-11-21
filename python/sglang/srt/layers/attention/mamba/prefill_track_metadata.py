"""Host-side row selection for prefill state snapshots.

Slot IDs deliberately stay on the device: unified pools translate virtual IDs
before these row indices gather physical source and destination slots.
"""

from dataclasses import dataclass
from itertools import accumulate
from typing import Sequence


@dataclass
class PrefillTrackPlan:
    tracked_rows: list[int]
    final_rows: list[int]
    unaligned_rows: list[int]
    h_rows: list[int]
    h_src: list[int]
    recompute_rows: list[int]
    recompute_end_locs: list[int]
    chunk_indices: list[int]


def build_prefill_track_plan(
    mask: list[bool],
    track_lens: list[int],
    extend_lens: list[int],
    prefix_lens: list[int],
    chunk_size: int,
    *,
    mamba2: bool,
    fork_rows: Sequence[int] = (),
    fork_lens: Sequence[int] = (),
) -> PrefillTrackPlan:
    """Use the backend's actual chunk size, including Mamba2's flat grid.

    Fork ``j`` (an extra chunk-boundary state of batch row ``fork_rows[j]``) is
    item ``len(mask) + j``: ``h_rows`` and ``unaligned_rows`` index the batch's
    track destinations followed by the fork slots, and the snapshot rows."""
    assert len(mask) == len(track_lens) == len(extend_lens) == len(prefix_lens)
    starts = list(accumulate(extend_lens, initial=0))
    h_offsets = list(
        accumulate(((n + chunk_size - 1) // chunk_size for n in extend_lens), initial=0)
    )
    plan = PrefillTrackPlan([], [], [], [], [], [], [], [-1] * len(mask))
    for row, track in enumerate(mask):
        if not track:
            continue
        plan.tracked_rows.append(row)
        length = track_lens[row] - prefix_lens[row]
        if length % chunk_size == 0:
            plan.final_rows.append(row)
            continue
        chunk = length // chunk_size
        plan.unaligned_rows.append(row)
        plan.chunk_indices[row] = chunk
        end = starts[row] + chunk * chunk_size
        if mamba2 and end % chunk_size:
            plan.recompute_rows.append(row)
            plan.recompute_end_locs.append(end)
        else:
            plan.h_rows.append(row)
            plan.h_src.append(end // chunk_size if mamba2 else h_offsets[row] + chunk)
    assert len(fork_rows) == len(fork_lens) and not (fork_rows and mamba2)
    for j, (row, fork_len) in enumerate(zip(fork_rows, fork_lens)):
        chunk, rem = divmod(fork_len - prefix_lens[row], chunk_size)
        assert rem == 0 and 0 < chunk * chunk_size < extend_lens[row], (
            f"fork state off the chunk grid or outside the extend: {row=} "
            f"{fork_len=} prefix={prefix_lens[row]} extend={extend_lens[row]}"
        )
        plan.unaligned_rows.append(len(mask) + j)
        plan.h_rows.append(len(mask) + j)
        plan.h_src.append(h_offsets[row] + chunk)
    return plan
