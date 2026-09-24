"""Logical Flash-Next payload layout. No device address or physical page on wire.

Pure Python so the exact wire validation and lease lifecycle run on CPU too.
The caller owns local token/page maps; a manifest only describes logical rows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import threading


DTYPE_BYTES = {"uint8": 1, "int8": 1, "float8_e4m3fn": 1, "bfloat16": 2,
               "float16": 2, "int16": 2, "int32": 4, "float32": 4,
               "int64": 8, "float64": 8}
ALIGNMENT = 256


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(frozen=True)
class Field:
    layer: int
    name: str
    dtype: str
    shape: tuple[int, ...]
    token_start: int
    token_end: int
    tokens_per_row: int
    compression: int = 1
    shard_axis: int = -1
    head_start: int = 0
    head_end: int = 0
    total_heads: int = 0
    # A latent payload is retained in D staging for transport validation only;
    # exact D decode still uses the supplied complete K/V.
    handoff_only: bool = False
    offset: int = 0

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]

    @property
    def key(self) -> tuple[int, str]:
        return self.layer, self.name

    def validate(self) -> None:
        if (not self.name or self.dtype not in DTYPE_BYTES or not self.shape
                or any(type(x) is not int or x <= 0 for x in self.shape)
                or self.token_start < 0 or self.token_end < self.token_start
                or self.tokens_per_row < 0 or self.compression <= 0
                or self.offset < 0 or self.offset % ALIGNMENT):
            raise ValueError("invalid staging field")
        if self.tokens_per_row and not (
            (self.shape[0]-1)*self.tokens_per_row < self.token_end-self.token_start
            <= self.shape[0]*self.tokens_per_row
        ):
            raise ValueError("logical interval does not match padded row count")
        if self.shard_axis == -1:
            if any((self.head_start, self.head_end, self.total_heads)):
                raise ValueError("replicated field must not advertise sharded heads")
        elif not (0 < self.shard_axis < len(self.shape)
                  and 0 <= self.head_start < self.head_end <= self.total_heads
                  and self.shape[self.shard_axis] == self.head_end-self.head_start):
            raise ValueError("invalid global head interval")


@dataclass(frozen=True)
class Manifest:
    room: int
    generation: int
    source_rank: int
    source_tp: int
    prompt_tokens: int
    chunk_index: int
    last_chunk: bool
    shallow_count: int
    deep_count: int
    fields: tuple[Field, ...]
    version: int = 1

    @classmethod
    def build(cls, *, fields, **kwargs):
        packed=[]
        offset=0
        for field in fields:
            field=replace(field, offset=offset)
            field.validate()
            packed.append(field)
            offset=align(offset+field.nbytes)
        result=cls(fields=tuple(packed), **kwargs)
        result.validate()
        return result

    @property
    def nbytes(self) -> int:
        return align(self.fields[-1].offset+self.fields[-1].nbytes) if self.fields else 0

    def validate(self) -> None:
        if (self.version != 1 or self.room < 0 or self.generation <= 0
                or not 0 <= self.source_rank < self.source_tp
                or self.prompt_tokens <= 0 or self.chunk_index < 0 or not self.fields
                or (self.shallow_count, self.deep_count) not in ((9, 8), (9, 9), (0, 0))):
            raise ValueError("invalid staging manifest identity or boundary phase")
        seen=set()
        end=0
        for field in self.fields:
            field.validate()
            if field.key in seen or field.offset != end or field.token_end > self.prompt_tokens:
                raise ValueError("duplicate field, noncanonical offset, or token interval")
            seen.add(field.key)
            end=align(field.offset+field.nbytes)

    def to_bytes(self) -> bytes:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, payload: bytes):
        if len(payload) > 1 << 20:
            raise ValueError("oversized staging manifest")
        obj=json.loads(payload)
        obj["fields"]=tuple(Field(**dict(f, shape=tuple(f["shape"]))) for f in obj["fields"])
        result=cls(**obj)
        result.validate()
        return result


@dataclass(frozen=True)
class Lease:
    slot: int
    generation: int
    room: int
    nbytes: int


class LeasePool:
    """Bounded slots with generation guards; outstanding DMA must drain on abort.

    There is no time-based eviction and no overcommit. A failed operation must
    still call finish only after its underlying CUDA/DMA event has drained.
    """
    def __init__(self, *, slots: int, slot_bytes: int):
        if slots <= 0 or slot_bytes <= 0:
            raise ValueError("positive staging capacity required")
        self.slot_bytes=slot_bytes
        self._generation=[0]*slots
        self._live={}
        self._lock=threading.Lock()
        self.peak_bytes=0
        self.peak_slots=0

    @property
    def reserved_bytes(self):
        return len(self._generation)*self.slot_bytes

    def acquire(self, *, room: int, nbytes: int) -> Lease | None:
        if room < 0 or not 0 < nbytes <= self.slot_bytes:
            raise ValueError("payload must be split to fit a staging slot")
        with self._lock:
            for slot in range(len(self._generation)):
                if slot not in self._live:
                    self._generation[slot]+=1
                    lease=Lease(slot, self._generation[slot], room, nbytes)
                    self._live[slot]=(lease, set(), False)
                    self.peak_slots=max(self.peak_slots, len(self._live))
                    self.peak_bytes=max(self.peak_bytes, sum(x[0].nbytes for x in self._live.values()))
                    return lease
        return None

    def _get(self, lease):
        state=self._live.get(lease.slot)
        if state is None or state[0] != lease:
            raise ValueError("stale staging lease")
        return state

    def begin(self, lease: Lease, *, operation: str):
        with self._lock:
            _, pending, aborted=self._get(lease)
            if aborted or not operation or operation in pending:
                raise ValueError("invalid operation on staging lease")
            pending.add(operation)

    def finish(self, lease: Lease, *, operation: str):
        with self._lock:
            _, pending, _=self._get(lease)
            if operation not in pending:
                raise ValueError("staging completion has no matching operation")
            pending.remove(operation)

    def abort(self, lease: Lease):
        with self._lock:
            _, pending, _=self._get(lease)
            self._live[lease.slot]=(lease,pending,True)

    def release(self, lease: Lease):
        with self._lock:
            _, pending, _=self._get(lease)
            if pending:
                raise RuntimeError("staging writer/scatter must drain before reuse")
            del self._live[lease.slot]
