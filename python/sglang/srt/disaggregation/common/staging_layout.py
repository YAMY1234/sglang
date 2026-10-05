"""Versioned, CPU-only byte layouts for replicated MLA and sharded draft KV.

Tensor and stream ownership belongs to the managers. This module describes only
registered storage and computes a deterministic, independently verifiable plan.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

VERSION = 2
ALIGNMENT = 256
MAX_BYTES = (1 << 63) - 1


def checked_int(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= MAX_BYTES:
        raise ValueError(f"Invalid staging {name}: {value!r}")
    return value


def align_bytes(value: int) -> int:
    checked_int(value, "bytes")
    return checked_int(
        (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT, "aligned bytes"
    )


@dataclasses.dataclass(frozen=True)
class StagingEntry:
    component: str
    global_layer_id: int
    kind: str
    index: int
    dtype: str
    page_size: int
    row_stride_bytes: int
    copy_width_bytes: int
    total_heads: int = 0

    @property
    def key(self):
        return self.component, self.global_layer_id, self.kind

    def __post_init__(self):
        if self.component not in ("target", "draft"):
            raise ValueError("Invalid staging component")
        if self.kind not in ("mla_latent", "mha_k", "mha_v"):
            raise ValueError("Unsupported staging entry kind")
        if self.dtype not in ("float16", "bfloat16"):
            raise ValueError("Staging v2 requires BF16/FP16 storage")
        for field in ("global_layer_id", "index", "total_heads"):
            checked_int(getattr(self, field), field)
        for field in ("page_size", "row_stride_bytes", "copy_width_bytes"):
            checked_int(getattr(self, field), field, 1)
        if (
            self.copy_width_bytes > self.row_stride_bytes
            or self.row_stride_bytes % 2
            or self.copy_width_bytes % 2
        ):
            raise ValueError("Invalid staging row stride/width")
        if (self.kind == "mla_latent") != (self.total_heads == 0):
            raise ValueError("Only MHA entries carry a head count")


@dataclasses.dataclass(frozen=True)
class WriterLayout:
    session: str
    pp_rank: int
    tp_rank: int
    tp_size: int
    entries: tuple[StagingEntry, ...]
    cp_rank: int = 0
    version: int = VERSION

    @property
    def writer_id(self):
        return self.session, self.pp_rank, self.tp_rank, self.cp_rank

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("Staging layout version mismatch; upgrade both peers")
        if not isinstance(self.session, str) or not self.session:
            raise ValueError("Staging writer requires a session identity")
        checked_int(self.pp_rank, "PP rank")
        checked_int(self.tp_rank, "TP rank")
        checked_int(self.tp_size, "TP size", 1)
        checked_int(self.cp_rank, "CP rank")
        if self.tp_rank >= self.tp_size or self.cp_rank != 0:
            raise ValueError("Staging v2 requires valid TP rank and CP1")
        if len({entry.key for entry in self.entries}) != len(self.entries):
            raise ValueError("Duplicate staging entry key")
        if {entry.index for entry in self.entries} != set(range(len(self.entries))):
            raise ValueError("Staging entries must cover the registered buffers")
        for entry in self.entries:
            if entry.total_heads:
                heads = entry.total_heads
                if max(heads, self.tp_size) % min(heads, self.tp_size):
                    raise ValueError("Staging KV heads and TP size must divide evenly")
                if entry.copy_width_bytes % (2 * max(1, heads // self.tp_size)):
                    raise ValueError("Invalid per-head staging width")

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["entries"] = tuple(StagingEntry(**entry) for entry in data["entries"])
        return cls(**data)

    def to_bytes(self):
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()

    @classmethod
    def from_bytes(cls, data):
        return cls.from_dict(json.loads(data))


@dataclasses.dataclass(frozen=True)
class EntryCopy:
    source: StagingEntry
    destination: StagingEntry
    src_offset: int
    dst_offset: int
    width: int
    offset: int
    length: int


@dataclasses.dataclass(frozen=True)
class WriterRegion:
    writer: WriterLayout
    offset: int
    length: int
    entries: tuple[EntryCopy, ...]


@dataclasses.dataclass(frozen=True)
class ChunkLayout:
    regions: tuple[WriterRegion, ...]
    valid_tokens: int
    total_bytes: int
    manifest_id: str

    @property
    def expected_writers(self):
        return frozenset(region.writer.writer_id for region in self.regions)

    @property
    def payload_bytes(self):
        return sum(entry.length for region in self.regions for entry in region.entries)

    def region_for(self, writer_id):
        return next(
            (region for region in self.regions if region.writer.writer_id == writer_id),
            None,
        )


def _head_range(entry, layout):
    count = max(1, entry.total_heads // layout.tp_size)
    start = layout.tp_rank // max(1, layout.tp_size // entry.total_heads) * count
    return start, start + count


def plan_chunk(
    writers: tuple[WriterLayout, ...], destination: WriterLayout, valid_tokens: int
) -> ChunkLayout:
    """Tile every destination row exactly once, then pack each writer contiguously.

    Replica choice is per entry/head interval. A writer with no selected target
    may still own draft shards; state writers remain the full bootstrap set.
    """
    checked_int(valid_tokens, "valid token count")
    writers = tuple(
        sorted(writers, key=lambda w: (w.pp_rank, w.tp_rank, w.cp_rank, w.session))
    )
    if len({w.writer_id for w in writers}) != len(writers):
        raise ValueError("Duplicate staging writer")
    if not writers:
        raise ValueError("Missing staging writer manifests")
    selected = {w.writer_id: [] for w in writers}
    destination_keys = {entry.key for entry in destination.entries}
    if any(entry.key not in destination_keys for w in writers for entry in w.entries):
        raise ValueError("Source staging entry is missing from destination")
    for dst in destination.entries:
        candidates = [
            (writer, src)
            for writer in writers
            for src in writer.entries
            if src.key == dst.key
        ]
        if not candidates:
            raise ValueError(f"Missing staging entry {dst.key}")
        if len({writer.pp_rank for writer, _ in candidates}) != 1:
            raise ValueError("Overlapping staging PP layer ownership")
        covered = []
        for writer, src in candidates:
            if (
                src.dtype != dst.dtype
                or src.page_size != dst.page_size
                or src.total_heads != dst.total_heads
            ):
                raise ValueError(f"Staging entry geometry mismatch: {dst.key}")
            if src.kind == "mla_latent":
                if src.copy_width_bytes != dst.copy_width_bytes:
                    raise ValueError("MLA staging row widths differ")
                start, end, source_offset = 0, dst.copy_width_bytes, 0
            else:
                sh0, sh1 = _head_range(src, writer)
                dh0, dh1 = _head_range(dst, destination)
                src_head_bytes = src.copy_width_bytes // (sh1 - sh0)
                dst_head_bytes = dst.copy_width_bytes // (dh1 - dh0)
                if src_head_bytes != dst_head_bytes:
                    raise ValueError("Draft staging head widths differ")
                h0, h1 = max(sh0, dh0), min(sh1, dh1)
                if h0 >= h1:
                    continue
                start, end = (h0 - dh0) * dst_head_bytes, (h1 - dh0) * dst_head_bytes
                source_offset = (h0 - sh0) * src_head_bytes
            if (start, end) in covered:
                continue  # Exact replica; deterministic first writer owns it.
            if any(
                start < old_end and old_start < end for old_start, old_end in covered
            ):
                raise ValueError("Overlapping staging head intervals")
            covered.append((start, end))
            selected[writer.writer_id].append(
                (src, dst, source_offset, start, end - start)
            )
        cursor = 0
        for start, end in sorted(covered):
            if start != cursor:
                raise ValueError(f"Missing staging head shard: {dst.key}")
            cursor = end
        if cursor != dst.copy_width_bytes:
            raise ValueError(f"Incomplete staging entry: {dst.key}")
    regions, base = [], 0
    for writer in writers:
        entries, offset = [], 0
        for src, dst, src_offset, dst_offset, width in sorted(
            selected[writer.writer_id], key=lambda e: e[0].index
        ):
            length = checked_int(valid_tokens * width, "entry byte count")
            entries.append(
                EntryCopy(src, dst, src_offset, dst_offset, width, offset, length)
            )
            offset = align_bytes(offset + length)
        if entries:
            regions.append(WriterRegion(writer, base, offset, tuple(entries)))
            base = checked_int(base + offset, "allocation bytes")
    signature = json.dumps(
        [destination.to_dict(), [w.to_dict() for w in writers]],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return ChunkLayout(
        tuple(regions), valid_tokens, base, hashlib.sha256(signature).hexdigest()
    )


def negotiate_version(local: int, remote: int) -> None:
    """V1/disabled behavior is unchanged; V2 requires explicit bilateral opt-in."""
    if VERSION in (local, remote) and local != remote:
        raise ValueError("Staging v2 requires both peers to enable and confirm v2")
    if local not in (0, 1, VERSION) or remote not in (0, 1, VERSION):
        raise ValueError("Unsupported staging layout version")


def encode_manifest(writers, destination):
    plan = plan_chunk(writers, destination, 1)
    return json.dumps(
        {
            "version": VERSION,
            "writers": [writer.to_dict() for writer in writers],
            "destination": destination.to_dict(),
            "manifest_id": plan.manifest_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def decode_manifest(data):
    document = json.loads(data)
    negotiate_version(VERSION, document["version"])
    writers = tuple(WriterLayout.from_dict(w) for w in document["writers"])
    destination = WriterLayout.from_dict(document["destination"])
    if plan_chunk(writers, destination, 1).manifest_id != document["manifest_id"]:
        raise ValueError("Staging manifest confirmation mismatch")
    return writers, destination
