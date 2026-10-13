"""GLM P1/P4 single-job qualification: TP4 micro N/S + paired GSM200 sanity.

Micro uses synthetic GLM-shaped states; service sanity uses the frozen GLM base,
release and existing docs/197 GSM evaluator. No C8/C32/C128 acceptance window.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
ATOL, RTOL = 2e-2, 1e-2


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def identity():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    assert (props.major, props.minor) == (10, 3), "qualification requires GB300/SM103"
    return {
        "sku": props.name,
        "capability": [props.major, props.minor],
        "memory_bytes": props.total_memory,
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "device": device,
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
        "nvcc": subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        ).stdout,
    }


def micro(out, repeats):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda", rank)
    proof = {
        "identity": identity(),
        "rank": rank,
        "source_sha": os.environ["GLM_ALIGN_SOURCE_SHA"],
        "state_fixture": "synthetic GLM local H16/K128/V128, full64 probe then TP4 slice",
        "timing_boundary": "34 dense recurrent layers, optional due cut after output, then 34-layer successor and completion",
        "cache_policy": "same live state/inputs, no explicit cache flush, identical reset/cold warm basis before timed pair",
        "clock_policy": "no clock manipulation; observed clocks saved",
        "samples": [],
    }
    from types import SimpleNamespace

    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    from sglang.srt.models.glm_duet.state_policy import GlmStatePolicy

    kernel = TritonKDAKernel()
    query_starts = {
        batch: torch.arange(batch + 1, device=dev, dtype=torch.int32)
        for batch in (1, 8, 32)
    }

    class Pool:
        def __init__(self, states, batch):
            self.mamba_pool = SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=states),
                register_slot_state=lambda _: None,
            )
            self.mapping = torch.arange(batch, device=dev)

        def mamba2_layer_index(self, layer):
            return layer

        def get_mamba_indices(self, indices):
            return self.mapping[indices]

    def update(qkv, a, b, A, dt, state, indices, packed):
        batch, heads, vdim, kdim = qkv.shape[0], *state.shape[1:]
        if packed:
            return kernel.packed_decode(
                mixed_qkv=qkv,
                a=a,
                b=b,
                A_log=A,
                dt_bias=dt,
                scale=kdim**-0.5,
                ssm_states=state,
                cache_indices=indices,
                num_v_heads=heads,
                head_v_dim=vdim,
                lower_bound=-5.0,
                glm_safe_packed=True,
            )
        q, k, v = qkv.split([heads * kdim, heads * kdim, heads * vdim], -1)
        return kernel.decode(
            q=q.view(1, batch, heads, kdim),
            k=k.view(1, batch, heads, kdim),
            v=v.view(1, batch, heads, vdim),
            a=a,
            b=b,
            A_log=A,
            dt_bias=dt,
            ssm_states=state,
            cache_indices=indices,
            query_start_loc=query_starts[batch],
            lower_bound=-5.0,
        )

    # Actual same-address CUDA capture/replay with changing gates, qkv and slots.
    numerical = []
    for batch in (1, 8, 32):
        for dtype in (torch.float32, torch.bfloat16):
            gen = torch.Generator(device=dev).manual_seed(31)
            rand = lambda *shape, generator=gen: torch.randn(
                *shape, generator=generator, device=dev
            )
            qkv = (rand(batch, 3 * 16 * 128) * 0.2).to(dtype)
            a = rand(batch, 16 * 128).to(dtype)
            b = rand(batch, 3 * 16).to(dtype)[:, :16].unsqueeze(0)
            A, dt = rand(16) * 0.2, rand(16 * 128) * 0.1
            envelope = rand(batch + 2, 3, 16, 128, 128) * 0.05
            state1, state2 = envelope[:, 1], envelope.clone()[:, 1]
            indices = torch.arange(batch, device=dev, dtype=torch.int32)
            # Warm both variants before capture, then restore identical initial state.
            initial = state1.clone()
            for _ in range(3):
                update(qkv, a, b, A, dt, state1, indices, False)
                update(qkv, a, b, A, dt, state2, indices, True)
            torch.cuda.synchronize()
            graphs, outputs = [], []
            for state, packed in ((state1, False), (state2, True)):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = update(qkv, a, b, A, dt, state, indices, packed)
                graphs.append(graph)
                outputs.append(output)
            state1.copy_(initial)
            state2.copy_(initial)
            for tick in range(4):
                qkv.copy_((rand(*qkv.shape) * 0.2).to(dtype))
                a.copy_(rand(*a.shape).to(dtype))
                b.copy_(rand(*b.shape).to(dtype))
                if tick == 1:
                    a[:, ::2] = -100
                    a[:, 1::2] = 100
                    beta_rows = b.reshape(batch, 16)
                    beta_rows[:, ::2] = -80
                    beta_rows[:, 1::2] = 80
                indices.copy_(
                    torch.arange(batch, device=dev, dtype=torch.int32).flip(0)
                )
                if tick % 2:
                    indices[-1] = -1
                before = state2.clone()
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    outputs[0].float(), outputs[1].float(), atol=ATOL, rtol=RTOL
                )
                torch.testing.assert_close(state1, state2, atol=ATOL, rtol=RTOL)
                assert torch.isfinite(outputs[1]).all() and torch.isfinite(state2).all()
                untouched = set(range(batch + 2)) - set(indices[indices >= 0].tolist())
                for slot in untouched:
                    assert torch.equal(before[slot], state2[slot]), (
                        "padding corrupted state"
                    )
                numerical.append(
                    {
                        "batch": batch,
                        "dtype": str(dtype),
                        "replay": tick,
                        "output_max_abs": float(
                            (outputs[0].float() - outputs[1].float()).abs().max()
                        ),
                        "state_max_abs": float((state1 - state2).abs().max()),
                    }
                )
    proof["packed_numerical_graph"] = numerical

    arms = {
        "baseline": (False, False),
        "p1": (True, False),
        "p4": (False, True),
        "both": (True, True),
    }
    for batch in (1, 8, 32):
        gen = torch.Generator(device=dev).manual_seed(83)
        rand = lambda *shape, generator=gen: torch.randn(
            *shape, generator=generator, device=dev
        )
        directions = rand(34, 16, 128)
        raw = rand(34, batch, 16, 128, 128) * 0.05
        qkv, a, b = (
            (rand(batch, 3 * 16 * 128) * 0.2).bfloat16(),
            rand(batch, 16 * 128).bfloat16(),
            rand(batch, 3 * 16).bfloat16()[:, :16].unsqueeze(0),
        )
        A, dt = rand(16) * 0.2, rand(16 * 128) * 0.1
        successor_a, successor_b = a * 0.7, b + 0.1
        indices = torch.arange(batch, device=dev, dtype=torch.int32)
        reqs = torch.arange(batch, device=dev)
        fb = SimpleNamespace(
            req_pool_indices=reqs,
            req_pool_indices_cpu=torch.arange(batch),
            batch_size=batch,
        )
        policies, cold, warm = {}, {}, {}
        for arm, (p1, _) in arms.items():
            policy = GlmStatePolicy(
                Pool(raw.clone(), batch),
                directions,
                {"state_rank": 32, "state_every": 32},
                list(range(34)),
                omega_heads=(rank * 16, 64),
                align_prepare=p1,
            )
            policy.mark_prompt_end(range(batch), range(batch))
            policy.before_graph(fb)
            policies[arm] = policy
            cold[arm] = policy.states.clone()
            warm[arm] = {
                key: value.clone()
                for key, value in (policy.warm_slots if p1 else policy.warm).items()
            }
        # N: same CUDA-generated full-head probes, shape and TP slice.
        assert torch.equal(cold["baseline"], cold["p1"])
        for candidate in ("p1", "p4", "both"):
            for phase in ("ordinary", "due"):
                for order in (("baseline", candidate), (candidate, "baseline")):
                    for repeat in range(repeats + 2):
                        results = {}
                        for arm in order:
                            p1, p4 = arms[arm]
                            policy = policies[arm]
                            policy.states.copy_(cold[arm])
                            owned = {
                                key: value.clone() for key, value in warm[arm].items()
                            }
                            if p1:
                                policy.warm_slots = owned
                            else:
                                policy.warm = owned
                            policy.h_count = {
                                s: 31 if phase == "due" else 0 for s in range(batch)
                            }
                            policy.steps = {s: 0 for s in range(batch)}
                            policy.h_pending.clear()
                            torch.cuda.synchronize()
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            host_start = time.perf_counter()
                            start.record()
                            first = []
                            for layer in range(34):
                                first.append(
                                    update(
                                        qkv,
                                        a,
                                        b,
                                        A,
                                        dt,
                                        policy.states[layer],
                                        indices,
                                        p4,
                                    )
                                )
                            policy.after_graph(fb)
                            successor = []
                            for layer in range(34):
                                successor.append(
                                    update(
                                        qkv,
                                        successor_a,
                                        successor_b,
                                        A,
                                        dt,
                                        policy.states[layer],
                                        indices,
                                        p4,
                                    )
                                )
                            policy.after_graph(fb)
                            end.record()
                            end.synchronize()
                            host_ms = (time.perf_counter() - host_start) * 1000
                            results[arm] = (
                                torch.stack(first + successor),
                                policy.states.clone(),
                            )
                            if repeat >= 2:
                                proof["samples"].append(
                                    {
                                        "arm": arm,
                                        "candidate": candidate,
                                        "order": list(order),
                                        "batch": batch,
                                        "due_D": batch if phase == "due" else 0,
                                        "phase": phase,
                                        "repeat": repeat - 2,
                                        "device_chain_ms": start.elapsed_time(end),
                                        "host_completion_ms": host_ms,
                                    }
                                )
                        if candidate == "p1":
                            assert all(
                                torch.equal(x, y)
                                for x, y in zip(results["baseline"], results[candidate])
                            )
                        else:
                            for baseline, changed in zip(
                                results["baseline"], results[candidate]
                            ):
                                torch.testing.assert_close(
                                    changed, baseline, atol=ATOL, rtol=RTOL
                                )
                        write(out / f"micro-rank{rank}.json", proof)
        del policies, cold, warm, raw
        torch.cuda.empty_cache()

    # Record actual compiler output and resource allocation, not a GPU-family estimate.
    from sglang.kernels.ops.attention.fla import (
        fused_recurrent,
        fused_sigmoid_gating_recurrent,
    )

    resources = []
    for name, jit in (
        ("packed", fused_recurrent.fused_recurrent_kda_packed_decode_kernel),
        (
            "generic",
            fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update_kernel,
        ),
    ):
        for cache in jit.device_caches.values():
            for index, compiled in enumerate(cache[0].values()):
                compiled._init_handles()
                prefix = out / f"rank{rank}-{name}-{index}"
                prefix.with_suffix(".ptx").write_text(compiled.asm["ptx"])
                prefix.with_suffix(".cubin").write_bytes(compiled.asm["cubin"])
                disasm = subprocess.run(
                    ["cuobjdump", "--dump-sass", str(prefix.with_suffix(".cubin"))],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                prefix.with_suffix(".sass").write_text(disasm.stdout)
                resources.append(
                    {
                        "route": name,
                        "n_regs": compiled.n_regs,
                        "n_spills": compiled.n_spills,
                        "shared": compiled.metadata.shared,
                        "ptx": prefix.name + ".ptx",
                        "sass_exit": disasm.returncode,
                    }
                )
    assert resources and all(row["sass_exit"] == 0 for row in resources), (
        "compiler resource evidence incomplete"
    )
    proof.update(
        passed=True,
        resources=resources,
        atol=ATOL,
        rtol=RTOL,
        performance_disposition="raw paired samples only; no acceptance-window claim",
    )
    write(out / f"micro-rank{rank}.json", proof)


def sanity(out, helpers, reference_root):
    sys.path[:0] = [str(helpers), str(reference_root), str(ROOT), str(ROOT / "python")]
    from gsm_endpoint import run as gsm

    model = "/work/models/GLM-5.3-Flash"
    release = "/work/hf-cache/duet-releases/Latent-Voyager--GLM-5.3-Flash-SEED/f0876024a34df39c76130eb0c5601c342e7d5425"
    records = {}
    for arm in ("stock", "duet"):
        root = out / arm
        root.mkdir(parents=True, exist_ok=True)
        enabled = arm == "duet"
        env = dict(
            os.environ,
            SGLANG_GLM_ALIGN_PREPARE=str(int(enabled)),
            SGLANG_GLM_SAFE_PACKED_DECODE=str(int(enabled)),
        )
        command = [
            sys.executable,
            "-m",
            "benchmark.duet_eval.launch_duet",
            "--model",
            model,
            "--duet-release",
            release,
            "--arm",
            arm,
            "--tp",
            "4",
            "--gpus",
            "0,1,2,3",
            "--port",
            "30610",
            "--out",
            str(root / "server"),
            "--context-length",
            "262144",
            "--max-running-requests",
            "32",
            "--sampling-backend",
            "pytorch",
            "--disable-overlap-schedule",
            "--detach",
        ]
        write(
            root / "configuration.json",
            {
                "argv": command,
                "source_sha": os.environ["GLM_ALIGN_SOURCE_SHA"],
                "switches": {
                    k: env[k]
                    for k in (
                        "SGLANG_GLM_ALIGN_PREPARE",
                        "SGLANG_GLM_SAFE_PACKED_DECODE",
                    )
                },
            },
        )
        with (root / "launcher.log").open("w") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        launch = json.loads((root / "server/launch.json").read_text())
        pid = launch["pid"]
        try:
            gsm(launch["url"], model, root / "gsm200.json")
            data = json.loads((root / "gsm200.json").read_text())
            assert data["complete"] and len(data["results"]) == 200
            records[arm] = data["results"]
            info = (
                __import__("urllib.request", fromlist=["urlopen"])
                .urlopen(launch["url"] + "/get_server_info", timeout=30)
                .read()
            )
            (root / "server-info.json").write_bytes(info)
            if enabled:
                assert (
                    "GLM P4 safe packed dense decode engaged"
                    in (root / "server/server.log").read_text()
                ), "safe packed route not observed"
        finally:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.5)
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    baseline, changed = records["stock"], records["duet"]
    for a, b in zip(baseline, changed):
        assert all(a[key] == b[key] for key in ("id", "gold", "prompt_hash"))
    b = sum(a["correct"] and not c["correct"] for a, c in zip(baseline, changed))
    c = sum(not a["correct"] and d["correct"] for a, d in zip(baseline, changed))
    passed = c - b >= -2 * math.sqrt(b + c)
    result = {
        "passed": passed,
        "stock_correct": sum(r["correct"] for r in baseline),
        "duet_correct": sum(r["correct"] for r in changed),
        "b": b,
        "c": c,
        "delta_count": c - b,
        "se_count": math.sqrt(b + c),
        "n": 200,
        "gate": "Delta=c-b >= -2*sqrt(b+c)",
        "protocol": "docs197 original GSM200 batch8/chat-boxed/enable_thinking=False/greedy2048",
        "no_formal_window": True,
        "source_sha": os.environ["GLM_ALIGN_SOURCE_SHA"],
    }
    write(out / "paired-sanity.json", result)
    assert passed, "paired GSM200 sanity FAIL"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("micro", "sanity"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--helpers", type=Path)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "micro":
        micro(args.out, args.repeats)
    else:
        sanity(args.out, args.helpers, args.reference_root)
