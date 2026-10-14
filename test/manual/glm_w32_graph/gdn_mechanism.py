"""No-weight GPU mechanism test of the GLM W32 graph policy (twinstar docs/210 §11).

Drives the real GraphGlmStatePolicy (fake pool, served TP4 shape: 34 KDA layers, 41 slots, 16 heads) through the
decode-graph pattern: per layer before_layer -> a stand-in decode op on the main stream -> after_layer (side-stream
groups, device IF per group, last-layer join and phase publication). For method in (qr2, gdn) x conditional in (0, 1)
x B in (1, 8, 32):
  * capture once, replay 64 steps; the same 64 steps eagerly from the same initial state; final states, bases and
    phases must be bitwise equal (graph == eager), and the cut must have happened at steps 32 and 64 only;
  * per-replay CUDA events: ordinary-step and due-step ms (policy nodes + stand-in decode);
  * prompt-end cold cut (eager) ms for B slots.
Not a service number: the stand-in decode is one elementwise op per layer.
"""

import argparse
import gc
import json
import os
import statistics
import types

import torch

LAYERS, SLOTS, HEADS, D = 34, 41, 16, 128


def fake_pool(states):
    cache = types.SimpleNamespace(temporal=states)
    mamba = types.SimpleNamespace(mamba_cache=cache, register_slot_state=lambda policy: None)
    return types.SimpleNamespace(mamba_pool=mamba, mamba2_layer_index=lambda layer: layer)


def make_policy(states, method, conditional, gen):
    os.environ["SGLANG_GLM_W32_METHOD"] = method
    os.environ["SGLANG_GLM_W32_CONDITIONAL"] = "1" if conditional else "0"
    from sglang.srt.models.glm_duet.graph_state_policy import GraphGlmStatePolicy

    dirs = {l: torch.randn(HEADS, D, generator=gen, device="cuda") for l in range(LAYERS)}
    spec = dict(state_rank=32, state_every=32, state_sink="explicit")
    return GraphGlmStatePolicy(fake_pool(states), dirs, spec, list(range(LAYERS)), omega_heads=(0, HEADS))


def step(policy, indices, decay):
    for layer in range(LAYERS):
        policy.before_layer(layer, indices)
        policy.states[layer].mul_(decay)  # stand-in decode: touches every slot of the layer on the main stream
        policy.after_layer(layer, indices)


def run_cell(method, conditional, batch, base, gen):
    states = base.clone()
    policy = make_policy(states, method, conditional, gen)
    slots = torch.arange(1, batch + 1, device="cuda")
    indices = slots.to(torch.int32)
    decay = torch.full((), 0.999, device="cuda")
    out = dict(method=method, conditional=conditional, batch=batch)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    policy.mark_prompt_end(None, slots)
    end.record()
    torch.cuda.synchronize()
    out["cold_ms"] = start.elapsed_time(end)
    initial = (states.clone(), policy.warm.clone(), policy.phase.clone())
    # eager reference: 64 steps
    for _ in range(64):
        step(policy, indices, decay)
    torch.cuda.synchronize()
    eager = (states.clone(), policy.warm.clone(), policy.phase.clone())
    # graph: restore, warm up on a side stream (restored again), capture, replay 64 steps
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(policy, indices, decay)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            step(policy, indices, decay)
    except Exception as exc:  # keep the failure, do not fall back
        out["capture_error"] = repr(exc)
        return out
    states.copy_(initial[0]), policy.warm.copy_(initial[1]), policy.phase.copy_(initial[2])
    torch.cuda.synchronize()
    events = []
    for i in range(64):
        b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        b.record()
        graph.replay()
        e.record()
        events.append((i + 1, b, e))
    torch.cuda.synchronize()
    times = {i: b.elapsed_time(e) for i, b, e in events}
    out["graph_equals_eager"] = dict(
        states=torch.equal(states, eager[0]), warm=torch.equal(policy.warm, eager[1]), phase=torch.equal(policy.phase, eager[2])
    )
    moved = (states[:, slots] - initial[0][:, slots] * (0.999 ** 64)).abs().amax().item()
    out["cut_changed_state"] = moved > 0
    out["phase_after_64"] = policy.phase[slots].tolist()[:4]
    ordinary = [t for i, t in times.items() if i % 32 and i > 2]
    due = [t for i, t in times.items() if i % 32 == 0]
    out["ordinary_ms_median"] = statistics.median(ordinary)
    out["due_ms"] = due
    # stand-in decode alone, same graph shape without the policy (tax baseline)
    del graph
    gc.collect()
    torch.cuda.empty_cache()
    return out


def baseline(batch, base):
    states = base.clone()
    decay = torch.full((), 0.999, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for layer in range(LAYERS):
            states[layer].mul_(decay)
    torch.cuda.synchronize()
    times = []
    for _ in range(64):
        b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        b.record()
        graph.replay()
        e.record()
        times.append((b, e))
    torch.cuda.synchronize()
    return statistics.median(b.elapsed_time(e) for b, e in times[2:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", default="1,8,32")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(210)
    decay = torch.exp(-torch.arange(D, device="cuda") / 10.0)
    base = torch.empty(LAYERS, SLOTS, HEADS, D, D, device="cuda")
    for layer in range(LAYERS):
        a = torch.randn(SLOTS, HEADS, D, D, device="cuda", generator=gen)
        b = torch.randn(SLOTS, HEADS, D, D, device="cuda", generator=gen)
        base[layer] = (a * decay) @ b / D
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cells=[])
    path = open(args.out, "w")
    path.close()
    for batch in [int(x) for x in args.batches.split(",")]:
        result.setdefault("stand_in_decode_ms", {})[batch] = baseline(batch, base)
        for method in ("gdn", "qr2"):
            for conditional in (True, False):
                try:
                    cell = run_cell(method, conditional, batch, base, gen)
                except Exception as exc:
                    import traceback

                    traceback.print_exc()
                    cell = dict(method=method, conditional=conditional, batch=batch, error=repr(exc))
                result["cells"].append(cell)
                print("GLMW32_MECH " + json.dumps(cell), flush=True)
                with open(args.out, "w") as f:
                    json.dump(result, f, indent=1)
                gc.collect()
                torch.cuda.empty_cache()
    result["complete"] = True
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
