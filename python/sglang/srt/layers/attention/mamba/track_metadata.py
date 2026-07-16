# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MambaTrackPlan:
    """Host-known row/offset plan; cache slot ids intentionally stay on device."""

    rows: tuple[int, ...]
    aligned_rows: tuple[int, ...]
    unaligned_rows: tuple[int, ...]
    h_src: tuple[int, ...]
    conv_indices: tuple[tuple[int, ...], ...]


def build_mamba_track_plan(
    *,
    track_mask: Sequence[bool],
    track_seqlens: Sequence[int],
    extend_seq_lens: Sequence[int],
    prefix_lens: Sequence[int],
    cache_chunk_size: int,
    conv_state_len: int,
    is_mamba2: bool,
) -> MambaTrackPlan:
    """Plan tracked rows on CPU without mirroring live virtual-to-physical ids."""

    batch_size = len(extend_seq_lens)
    if len(prefix_lens) != batch_size:
        raise ValueError(
            f"prefix_lens has {len(prefix_lens)} rows, expected {batch_size}"
        )
    if len(track_mask) < batch_size or len(track_seqlens) < batch_size:
        raise ValueError(
            "mamba CPU tracking metadata must cover every real extend row: "
            f"mask={len(track_mask)}, seqlens={len(track_seqlens)}, "
            f"extend={batch_size}"
        )
    if cache_chunk_size <= 0 or conv_state_len < 0:
        raise ValueError(
            f"invalid tracking sizes: {cache_chunk_size=}, {conv_state_len=}"
        )
    if any(int(length) <= 0 for length in extend_seq_lens):
        raise ValueError(f"extend sequence lengths must be positive: {extend_seq_lens}")

    total_tokens = sum(int(length) for length in extend_seq_lens)
    rows: list[int] = []
    aligned_rows: list[int] = []
    unaligned_rows: list[int] = []
    h_src: list[int] = []
    conv_indices: list[tuple[int, ...]] = []

    query_offset = 0
    h_offset = 0
    for row in range(batch_size):
        extend_len = int(extend_seq_lens[row])
        if track_mask[row]:
            lens_to_track = int(track_seqlens[row]) - int(prefix_lens[row])
            if lens_to_track < 0:
                raise ValueError(
                    f"tracked sequence length precedes prefix at row {row}: "
                    f"track={track_seqlens[row]}, prefix={prefix_lens[row]}"
                )

            rows.append(row)
            aligned_len = (lens_to_track // cache_chunk_size) * cache_chunk_size
            conv_start = query_offset + aligned_len - conv_state_len
            conv_indices.append(
                tuple(
                    min(max(conv_start + offset, 0), total_tokens - 1)
                    for offset in range(conv_state_len)
                )
            )

            if lens_to_track % cache_chunk_size == 0:
                aligned_rows.append(row)
            else:
                unaligned_rows.append(row)
                h_src.append(h_offset + lens_to_track // cache_chunk_size)

        num_h_states = (
            extend_len // cache_chunk_size
            if is_mamba2
            else (extend_len - 1) // cache_chunk_size + 1
        )
        query_offset += extend_len
        h_offset += num_h_states

    return MambaTrackPlan(
        rows=tuple(rows),
        aligned_rows=tuple(aligned_rows),
        unaligned_rows=tuple(unaligned_rows),
        h_src=tuple(h_src),
        conv_indices=tuple(conv_indices),
    )
