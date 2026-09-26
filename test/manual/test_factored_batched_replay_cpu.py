"""Real r8/W8 kernels under Triton CPU interpretation; no model/GPU claim.

TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' python test_factored_batched_replay_cpu.py
"""
import ast
import copy
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

VERIFY_FUSED = os.environ.get("REPLAY_TEST_VERIFY_FUSED") == "1"
GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
if not GPU and (os.environ.get('TRITON_INTERPRET') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
    raise RuntimeError('CPU interpretation with CUDA hidden is required')

import torch

if GPU:
    torch.set_default_device("cuda")

ROOT = Path(__file__).resolve().parents[2]


def package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def same(a, b, label):
    if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
        raise AssertionError(label)


def main():
    package('replay_kernels', ROOT/'python/sglang/srt/layers/attention/linear/kernels')
    package('replay_owners', ROOT/'python/sglang/srt/mem_cache')
    kernel = importlib.import_module('replay_kernels.gdn_factored')
    Old = importlib.import_module('replay_owners.gdn_factored_spec').FactoredGDNVerifyState
    New = importlib.import_module('replay_owners.gdn_factored_replay').FactoredGDNReplayState
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_factored'] = kernel
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_verify_io'] = importlib.import_module('replay_kernels.gdn_verify_io')
    path = ROOT/'python/sglang/srt/layers/attention/linear/gdn_backend.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'GDNAttnBackend')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_forward_verify_factored')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    verify = scope[method.name]
    torch.manual_seed(587)
    layers, capacity, batch, heads, key, value = 2, 3, 2, 2, 16, 16
    if GPU:
        heads, key, value = 4, 128, 128
    width = 2*key + heads*value
    original = SimpleNamespace(
        cfg=SimpleNamespace(r=8, m=8, rfull=16,
                            kernel_kwargs=lambda: dict(kernel='split', post_order=True)),
        a=torch.randn(layers, 8, heads, key)*.01,
        U=torch.randn(layers, 8, heads, 16, key, dtype=torch.float16)*.01,
        W=torch.randn(layers, 8, heads, 16, value, dtype=torch.float16)*.01,
        count=torch.full((layers, 8, heads), 8, dtype=torch.int32),
        stale=torch.zeros(8, dtype=torch.int32), dense_of=torch.arange(8, dtype=torch.int32),
        dense_required=torch.ones(8, dtype=torch.int32), prefix_valid=torch.ones(8, dtype=torch.int32),
        dense_ring=torch.randn(layers, 2, heads, value, key), ring_owner=[2, 5], ring_lru=[0, 1],
        vbar=torch.randn(layers, heads, value)*.01, layer_index=lambda layer: layer)
    descriptors = [SimpleNamespace(layer_id=i, num_q_heads=1, num_v_heads=heads,
        head_k_dim=key, head_v_dim=value, A_log=torch.randn(heads), dt_bias=torch.randn(heads))
        for i in range(layers)]
    slots = torch.tensor([2, 5])
    # Directly interpret the same CUDA snapshot kernel on CPU as well.
    # Include noncontiguous, reordered source IDs and untouched capacity rows.
    if os.environ.get('REPLAY_TEST_SNAPSHOT_KERNEL') == '1':
        io = sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_verify_io']
        for ids in (slots, slots.flip(0), torch.tensor([5, 0, 2, 0])[::2], slots[:1]):
            target = {name: torch.full_like(getattr(original, name)[:, :capacity], 7)
                      for name in Old.names}
            expected = {name: t.clone() for name, t in target.items()}
            for name in Old.names:
                expected[name][:, :ids.numel()].copy_(getattr(original, name).index_select(1, ids))
            io.snapshot_factors(original, target, ids)
            for name in Old.names:
                same(expected[name], target[name], 'snapshot kernel direct: '+name)
    cases = []
    for initial_count in range(8, 16):
        old, new, sequential = [copy.deepcopy(original) for _ in range(3)]
        for pool in (old, new, sequential):
            pool.count[:, slots] = initial_count
        old.spec_state = New(old, capacity, 4, qkv_width=width, batched_commit=False, verify_window_fused=False,
                             snapshot_kernel=False)
        new.spec_state = New(new, capacity, 4, qkv_width=width, batched_commit=True, verify_window_fused=VERIFY_FUSED,
                             snapshot_kernel=os.environ.get('REPLAY_TEST_SNAPSHOT_KERNEL') == '1')
        assert not new.spec_state.checkpoints
        pointers = [t.data_ptr() for t in new.spec_state.inputs.values()]
        # Seventeen consecutive zero-draft commits are needed once; the
        # remaining seeds independently cover each W8 position, not repeats.
        for iteration in range(21 if initial_count == 12 else 4):
            active = batch if iteration % 2 == 0 else 1
            active_slots = slots[:active]
            mixed = torch.randn(layers, capacity, 4, width, dtype=torch.bfloat16)
            ga = torch.randn(layers, capacity, 4, heads, dtype=torch.bfloat16)
            gb = torch.randn_like(ga)
            tickets = [p.spec_state.snapshot_commit(active_slots) for p in (old, new)]
            for name in Old.names:
                same(old.spec_state.working[name], new.spec_state.working[name], 'entry snapshot: '+name)
            same(tickets[0].slots, tickets[1].slots, 'ticket slots')
            same(tickets[0].generations, tickets[1].generations, 'ticket generations')
            assert tickets[0].epoch == tickets[1].epoch and not tickets[1].closed
            states = {n: getattr(new, n).clone() for n in Old.names}
            outputs = []
            for pool in (old, new):
                outputs.append([verify(SimpleNamespace(factored=pool, topk=1), desc,
                    mixed[li].flatten(0, 1), ga[li].flatten(0, 1), gb[li].flatten(0, 1))
                    for li, desc in enumerate(descriptors)])
            for li in range(layers):
                same(outputs[0][li], outputs[1][li], 'actual target GDN verify outputs')
            for name in Old.names:
                same(old.spec_state.working[name], new.spec_state.working[name], 'working state after verify: '+name)
            same(old.spec_state.written, new.spec_state.written, 'written candidate flags')
            for name in ('mixed', 'a', 'b'):
                same(old.spec_state.inputs[name], new.spec_state.inputs[name], 'owned replay inputs: '+name)
            for name, state in states.items():
                same(state, getattr(new, name), 'verify must not publish')
            accepted = torch.tensor([iteration%4, (iteration+2)%4] if iteration < 4 else [0, 0])[:active]
            track_slots = torch.tensor([-1, 7])[:active]
            track_steps = torch.tensor([-1, max(0, int(accepted[-1])-1)])[:active]
            # Independent sequential recurrence, masked after last consumed input.
            for li, desc in enumerate(descriptors):
                args = new.spec_state.layer_arguments[li]
                for step in range(4):
                    indices = torch.where(accepted >= step, active_slots, -1)
                    kernel.factored_packed_decode(mixed[li, :active, step], ga[li, :active, step],
                        gb[li, :active, step], fa=sequential.a[li], fu=sequential.U[li],
                        fw=sequential.W[li], fcount=sequential.count[li], stale=sequential.stale,
                        ssm_state_indices=indices, **args)
                    if active == 2 and step == int(track_steps[1]):
                        for name in Old.names:
                            getattr(sequential, name)[li, 7].copy_(getattr(sequential, name)[li, 5])
            # Destroy the caller's scratch; replay must use its owned window.
            mixed.fill_(99); ga.fill_(99); gb.fill_(99)
            for pool, ticket in zip((old, new), tickets):
                pool.spec_state.commit(ticket, accepted, track_slots=track_slots, track_steps=track_steps)
            for name in Old.names:
                same(getattr(old, name), getattr(new, name), 'per-layer/batched replay committed factors')
                same(getattr(sequential, name), getattr(new, name), 'sequential/replay committed factors')
            for name in ('stale', 'dense_of', 'dense_required', 'prefix_valid'):
                same(getattr(old, name), getattr(new, name), 'publication metadata')
            same(original.dense_ring, new.dense_ring, 'decode must preserve FP32 dense ring')
            assert new.ring_owner == original.ring_owner and new.ring_lru == original.ring_lru
            assert tickets[0].closed and tickets[1].closed
            same(old.spec_state.generations, new.spec_state.generations, 'slot generations')
            assert pointers == [t.data_ptr() for t in new.spec_state.inputs.values()]
            cases.append(dict(initial_count=initial_count, round=iteration,
                              active_batch=active, last_consumed_indices=accepted.tolist(), bitwise=True))
            print(f'case count={initial_count} round={iteration} passed', file=sys.stderr, flush=True)
        before = {n: getattr(new, n).clone() for n in (*Old.names, 'stale', 'dense_of', 'dense_required', 'prefix_valid')}
        ticket = new.spec_state.snapshot_commit(slots)
        # A request may stop before consuming any target input. Run verification
        # before aborting, so this checks an actually mutated working version.
        for li, desc in enumerate(descriptors):
            verify(SimpleNamespace(factored=new, topk=1), desc,
                   torch.randn(capacity*4, width, dtype=torch.bfloat16),
                   torch.randn(capacity*4, heads, dtype=torch.bfloat16),
                   torch.randn(capacity*4, heads, dtype=torch.bfloat16))
        new.spec_state.rollback(ticket)
        for name in before:
            same(before[name], getattr(new, name), 'abort state')
        same(original.dense_ring, new.dense_ring, 'aborted verify dense ring')
        # CUDA _assert_async deliberately poisons the context. Invalid-generation
        # rejection is covered on CPU; do not continue GPU comparison after an
        # intentional device assertion or pretend it raises synchronously.
        if not GPU:
            ticket = new.spec_state.snapshot_commit(slots)
            new.spec_state.invalidate_slots(slots[:1])
            try:
                new.spec_state.commit(ticket, torch.zeros(batch, dtype=torch.long))
            except RuntimeError:
                pass
            else:
                raise AssertionError('reused generation accepted')
            for name in Old.names:
                same(before[name], getattr(new, name), 'invalid commit state')
    print(json.dumps(dict(passed=True, device='CUDA' if GPU else 'CPU', triton_interpret=not GPU,
        verify_window_fused=VERIFY_FUSED, cases=cases, no_candidate_checkpoints=True, raw_inputs_owned=True,
        baseline_replay_bytes=old.spec_state.bytes(), replay_bytes=new.spec_state.bytes(),
        scope='real GDN verify, per-layer/batched replay/sequential states, tracking, W8 all four positions, 17 consecutive zero drafts; not full-model logits or GPU graph admission')))


if __name__ == '__main__':
    main()
