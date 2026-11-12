"""CPU-only PD input preparation; admission and shared state stay on the scheduler.

A prepared generation owns its backing storage. It neither owns a live PP slot nor
reserves KV/rows. Only a disjoint, fresh text request is eligible. Prefix matching,
HiCache, sampling and allocation are committed by the ordinary scheduler later.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PPPreparedInput:
    slot: int
    generation: int
    req: Any
    rid: str
    tokens: bytes
    input_ids_cpu: Any

    @classmethod
    def prepare(cls, slot, generation, waiting, busy, build):
        for req in waiting:
            if id(req) in busy or req.inflight_steps or req.output_ids or req.finished():
                continue
            if (req.session is not None or req.multimodal_inputs is not None
                    or req.input_embeds is not None or req.positional_embed_overrides is not None
                    or req.beam_group is not None or req.is_dllm()):
                continue
            # Bound each generation to one context. The request has no producer
            # in flight; build copies its token storage, never aliases it.
            tokens = req.origin_input_ids.tobytes()
            return cls(slot, generation, req, req.rid, tokens, build(req.origin_input_ids))
        return None

    def for_batch(self, reqs):
        # Multi-request concatenation remains on the ordinary path. No guessed
        # prefix/chunk range or speculative row/cache ownership is published.
        if len(reqs) != 1 or reqs[0] is not self.req:
            return None
        req = reqs[0]
        if (req.rid != self.rid or req.output_ids or req.finished()
                or req.origin_input_ids.tobytes() != self.tokens):
            return None
        start, end = req.prefix_len, req.extend_end
        if not 0 <= start <= end <= len(req.origin_input_ids):
            return None
        # Check the post-admission fill, including abort/retraction replacement.
        if req.get_fill_ids()[start:end].tobytes() != self.tokens[start * 8:end * 8]:
            return None
        return self.input_ids_cpu.narrow(0, start, end - start)

    def can_commit(self, slot, generation, completed_generation):
        return (self.slot == slot and self.generation == generation + 1
                and completed_generation == generation)
