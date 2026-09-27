"""Whole-state and stream-order checks for zero-vbar prefill layer pipelining."""
import argparse
import copy
import importlib
import json
import os
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--device", choices=("cpu", "cuda"), required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
os.environ["REPLAY_TEST_DEVICE"] = a.device
import torch
import test_factored_prefill_graph as tx
pipeline_module = importlib.import_module("prefill_owners.gdn_prefill_pipeline")


class InterpretedGroup:
    def __init__(self):
        self.entries = {}

    def run(self, pool, plan, track_slots, *, factorize, policy, replay_stream=None):
        key = (len(plan.pending), track_slots is not None)
        if key not in self.entries:
            buffers = tx.graphs.CommitBuffers(pool, plan, track_slots)
            watched = {name: value.clone() for name, value in vars(pool).items() if isinstance(value, torch.Tensor)}
            buffers.evaluate(factorize)
            for name, value in watched.items():
                tx.same(value, getattr(pool, name), "inactive group capture " + name)
            self.entries[key] = buffers
        buffers = self.entries[key]
        buffers.bind(plan, track_slots)
        buffers.evaluate(factorize)
        return True


class CommitState:
    def __init__(self):
        self.side = torch.cuda.Stream()
        self.done = torch.cuda.Event()
        self.recorded = False
        self.generations = torch.zeros(10, dtype=torch.int64)

    def mark(self):
        self.done.record(self.side)
        self.recorded = True

    def join(self):
        if self.recorded:
            torch.cuda.current_stream().wait_event(self.done)

    def invalidate_slots(self, slots):
        self.generations[slots] += 1


def case(group, tracked, staged, external_vbar=False, batch=1):
    old = tx.pool(36, 24 if tx.GPU else 2, 128 if tx.GPU else 16)
    old.cfg.vbar_path = "nonzero-proof" if external_vbar else None
    if not external_vbar:
        old.vbar.zero_()
    new = copy.deepcopy(old)
    for pool in (old, new):
        pool.prefill_commit_graph = tx.graphs.PrefillCommitGraph() if tx.GPU else InterpretedGroup()
        if tx.GPU:
            pool.spec_state = CommitState()
            pool._prefill_commit_side_stream = lambda pool=pool: pool.spec_state.side
    pipe = new.prefill_commit_pipeline = pipeline_module.PrefillLayerCommitPipeline(group)
    if not tx.GPU:
        pipe.graph_type = InterpretedGroup
    for repeat in range(3):
        slots = torch.tensor(([2, 5] if repeat % 2 == 0 else [5, 2])[:batch])
        rings = torch.tensor(([0, 1] if repeat % 2 == 0 else [2, -1])[:batch])
        track_slots = slots.clone() if tracked == "alias" else (torch.tensor([6, 8][:batch]) if tracked else None)
        dense = torch.randn(36, batch, old.hv, old.v, old.k)
        extra = torch.randn_like(dense) if tracked else None
        for pool in (old, new):
            if tx.GPU:
                pool.spec_state.join()
            pool.dense_required[slots] = 99
            plan = tx.module.FactoredExtendPlan(
                slots=slots, use_ring=torch.zeros(batch, dtype=torch.bool),
                ring_src=torch.zeros(batch, dtype=torch.long), ring_dst=rings,
                ring_dst_rows=torch.where(rings >= 0)[0], last_layer=35,
                dense_required_after_commit=torch.full((batch,), repeat % 2, dtype=torch.int32),
                stage=dense if staged else None, track_stage=extra if staged else None)
            for li in range(36):
                pool.commit_extend_batched(li, plan, dense[li], extra[li] if tracked else None,
                                           track_slots, final_src=slots[:1], final_dst=torch.tensor([9]))
                if pool is new and li + 1 in (group, 2*group) and li < 35:
                    if tx.GPU:
                        torch.cuda.synchronize()
                    assert torch.equal(pool.prefix_valid[slots], torch.zeros(batch, dtype=torch.int32))
                    assert torch.equal(pool.dense_required[slots], torch.full((batch,), 99, dtype=torch.int32))
            assert plan.next_layer == 36 and not plan.pending
            if tx.GPU:
                # Future pool reads use the event, not a host synchronization.
                pool.spec_state.join()
        if tx.GPU:
            torch.cuda.synchronize()
        for name in tx.FIELDS:
            tx.same(getattr(old, name), getattr(new, name), "pipeline pool " + name)
        if tx.GPU:
            tx.same(old.spec_state.generations, new.spec_state.generations, "slot generations")
    eligible = not external_vbar and batch == 1
    assert bool(pipe.replayed) == eligible
    if eligible:
        assert pipe.replayed == 3 * (36 // group)
    return dict(group=group, tracked=tracked, staged=staged, batch=batch,
                external_vbar=external_vbar, bitwise=True, replayed=pipe.replayed,
                fallback=not eligible, delayed_publication=True)


os.environ["SGLANG_GDN_PREFILL_STORE_LAYERS"] = "1"
os.environ["SGLANG_GDN_PREFILL_FACTOR_PAIR"] = "0"
os.environ["SGLANG_GDN_PREFILL_COPY_METADATA_DEVICE"] = "1"
torch.manual_seed(298832)
start = time.monotonic()
rows = []
for group in (12, 18):
    for tracked, staged in ((False, False), (True, True), ("alias", True)):
        rows.append(case(group, tracked, staged))
        print(json.dumps(rows[-1]), flush=True)
    rows.append(case(group, True, True, external_vbar=True))
    rows.append(case(group, True, True, batch=2))
result = dict(complete=True, passed=True, device=a.device, rows=rows, seconds=time.monotonic()-start,
              scope="Primitive only; whole pool, live slots/rings, delayed publication, alias and fallback checks; no model admission")
a.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
