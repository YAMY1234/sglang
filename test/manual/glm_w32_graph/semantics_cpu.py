"""Zero-GPU schedule oracle: actual host and graph-policy methods, same projector.

The graph policy's matrix launch is replaced with the unmodified CPU reference
projector; stream edges become ordered no-ops. This isolates policy semantics
from GPU rounding/stream execution. Nine layers exercise a full and partial group.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from lifecycle_gate import policy_module
from numeric_gate import ROOT, reference, relf

name = "sglang.srt.layers.attention.linear.kda_state_prune"
spec = importlib.util.spec_from_file_location(
    name, ROOT / "layers/attention/linear/kda_state_prune.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
old_module = importlib.import_module("glm_gate_models.glm_duet.state_policy")


def sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


class Stream:
    def wait_stream(self, other):
        pass


def run(out):
    torch.set_num_threads(4)
    torch.manual_seed(206)
    layers, batch, heads = 9, 32, 2
    state = torch.randn(layers, batch + 1, heads, 128, 128)
    directions = {l: torch.randn(heads, 128) for l in range(layers)}
    spec = {"state_rank": 32, "state_every": 32, "state_sink": "explicit"}

    def pool():
        return SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=state.clone()),
                register_slot_state=lambda s: None,
            ),
            mamba2_layer_index=lambda l: l,
            get_mamba_indices=lambda r: r + 1,
        )

    old = old_module.GlmStatePolicy(
        pool(), directions, spec, list(range(layers)), omega_heads=(0, heads)
    )
    new = policy_module.GraphGlmStatePolicy(
        pool(), directions, spec, list(range(layers)), omega_heads=(0, heads)
    )
    new.side = Stream()
    slots = list(range(1, batch + 1))
    indices = torch.tensor(slots, dtype=torch.int32)
    fb = SimpleNamespace(
        batch_size=batch,
        req_pool_indices=torch.arange(batch),
        req_pool_indices_cpu=torch.arange(batch),
    )
    events = {"host": [], "graph": []}
    current_step = 0
    sinks = {}

    def snapshot(policy, which, selected, layer, fresh):
        s = policy.states[layer, selected].transpose(-1, -2)
        vb = policy.vbar[layer]
        a = (
            torch.einsum("bhkv,hv->bhk", s, vb)
            / (vb.square().sum(-1).clamp_min(1e-12)[None, :, None])
        )
        key = f"{which}-step{current_step}-cold{int(fresh)}-layer{layer}"
        sinks[key] = a.clone()
        return {
            "step": current_step,
            "cold": fresh,
            "layer": layer,
            "slots": selected,
            "pre_sha": sha(s),
            "sink_sha": sha(a),
            "sink_values": key,
        }

    cut = old._cut

    def old_cut(selected, *, fresh):
        rows = [snapshot(old, "host", selected, l, fresh) for l in range(layers)]
        cut(selected, fresh=fresh)
        for r in rows:
            r["post_sha"] = sha(old.states[r["layer"], selected].transpose(-1, -2))
        events["host"].extend(rows)

    old._cut = old_cut

    def cpu_project(index, first, count, *, cold):
        selected = [
            int(s)
            for s in index
            if s >= 0
            and new.active[s]
            and (
                new.pending[s]
                if cold
                else not new.pending[s] and new.phase[s] + 1 == 32
            )
        ]
        if not selected:
            return
        pairs = [(l, s) for l in range(first, first + count) for s in selected]
        ll = torch.tensor([l for l, s in pairs])
        ss = torch.tensor([s for l, s in pairs])
        records = [
            snapshot(new, "graph", selected, l, cold)
            for l in range(first, first + count)
        ]
        prevs = [None if cold else new.warm[l, s : s + 1] for l, s in pairs]
        projected, bases = reference.project_state_batched(
            new.states[ll, ss].transpose(-1, -2),
            new.vbar[ll],
            32,
            prevs,
            omega_heads=(0, heads),
        )
        new.states[ll, ss] = projected.transpose(-1, -2)
        for (l, s), basis in zip(pairs, bases):
            new.warm[l, s] = basis[0]
        for r in records:
            r["post_sha"] = sha(new.states[r["layer"], selected].transpose(-1, -2))
        events["graph"].extend(records)

    new._project = cpu_project

    def phase(index, phase, pending, active, window, *, clear=False):
        selected = index[(index >= 0) & active[index.clamp_min(0)]]
        if clear:
            pending[selected] = False
        else:
            phase[selected] = (phase[selected] + 1) % window

    old.mark_prompt_end(list(range(batch)), slots)
    steps = []
    with (
        patch.object(torch.cuda, "current_stream", lambda *a: Stream()),
        patch.object(torch.cuda, "stream", lambda s: contextlib.nullcontext()),
        patch.object(policy_module, "phase_update", phase),
    ):
        new.mark_prompt_end(None, indices)
        for step in range(1, 65):
            current_step = step - 1
            old.before_graph(fb)
            # Cold is before step 1; warm events are after the numbered output.
            new.before_layer(0, indices)
            current_step = step
            qs = torch.nn.functional.normalize(
                torch.randn(layers, batch, heads, 128), dim=-1
            )
            ks = torch.nn.functional.normalize(torch.randn_like(qs), dim=-1)
            vs = torch.randn_like(qs)
            gs = torch.exp(-0.05 * torch.rand_like(qs))
            beta = torch.sigmoid(torch.randn(layers, batch, heads))
            outputs = {}
            for which, policy in (("host", old), ("graph", new)):
                hidden = torch.zeros(batch, heads, 128)
                for l in range(layers):
                    if which == "graph" and l:
                        new.before_layer(l, indices)
                    s = policy.states[l, 1:]
                    v = vs[l] + 0.05 * torch.tanh(hidden)
                    s.mul_(gs[l, ..., None, :])
                    residual = v - (s * ks[l, ..., None, :]).sum(-1)
                    s.add_(
                        beta[l, ..., None, None]
                        * residual[..., :, None]
                        * ks[l, ..., None, :]
                    )
                    hidden = (s * qs[l, ..., None, :]).sum(-1)
                    if which == "graph":
                        new.after_layer(l, indices)
                outputs[which] = hidden
            old.after_graph(fb)
            record = {
                "step": step,
                "host_sha": sha(old.states),
                "graph_sha": sha(new.states),
                "state_rel_f": relf(new.states, old.states),
                "output_rel_f": relf(outputs["graph"], outputs["host"]),
                "host_phase": [old.h_count[s] for s in slots],
                "graph_phase": new.phase[indices].tolist(),
            }
            steps.append(record)
            assert record["host_phase"] == record["graph_phase"], record
            if step in (1, 31, 32, 33, 63, 64):
                print("SEMANTIC_STEP " + json.dumps(record), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps.json").write_text(json.dumps(steps, indent=2))
    (out / "events.json").write_text(json.dumps(events, indent=2))
    torch.save(sinks, out / "sink-values.pt")
    keyed = {
        which: {(r["step"], r["cold"], r["layer"]): r for r in rows}
        for which, rows in events.items()
    }
    assert keyed["host"].keys() == keyed["graph"].keys()
    mismatches = []
    for key, a in keyed["host"].items():
        b = keyed["graph"][key]
        if any(a[k] != b[k] for k in ("slots", "pre_sha", "post_sha", "sink_sha")):
            mismatches.append(key)
    result = {
        "samples": batch,
        "steps": 64,
        "layers": layers,
        "heads": heads,
        "cuda_initialized": torch.cuda.is_initialized(),
        "events": len(events["host"]),
        "first_event_mismatch": mismatches[:1],
        "first_state_mismatch": [
            s["step"] for s in steps if s["host_sha"] != s["graph_sha"]
        ][:1],
        "max_state_rel_f": max(s["state_rel_f"] for s in steps),
        "max_output_rel_f": max(s["output_rel_f"] for s in steps),
        "passed": not mismatches
        and all(s["host_sha"] == s["graph_sha"] for s in steps),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print("GLMW32_SEMANTICS " + json.dumps(result), flush=True)
    assert result["passed"], result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
