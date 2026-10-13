"""One-job Lightning bridge CUDA stages + three-arm prompt micro + GSM200.

Source/model/service arguments are frozen by a plan generated before submission.
CUDA timing is unprofiled; launch diagnostics are a separate prompt replay.
No AgentX window or graph qualification is performed by this harness.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def http(url, endpoint, data=None):
    request = urllib.request.Request(
        url + endpoint,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def difference(a, b):
    import torch

    delta = a.float() - b.float()
    return {
        "bitwise": torch.equal(a, b),
        "max_abs": delta.abs().max().item(),
        "relative_frobenius": (delta.norm() / b.float().norm().clamp_min(1e-30)).item(),
        "unequal": int((a != b).sum()),
        "elements": a.numel(),
    }


def cuda_stages(plan, root):
    import torch

    sys.path.insert(0, str(Path(plan["bridge_source"]) / "python/sglang/srt/models"))
    from lightning_duet._common import load
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import selective_state_update

    Bridge = load("dense_bridge").DenseBridgeStatePool
    Old = load("state_pool").LeftSinkStatePool
    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for batch in (1, 2, 8, 31, 32):
        h, p, n, groups, channels = 64, 64, 128, 8, 6144
        slots = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
        if batch > 1:
            slots[-1] = -1
        host_slots = slots.cpu().tolist()
        conv = torch.randn(batch + 1, channels, 3, device="cuda")
        expected_conv = conv.clone()
        xbc = torch.randn(batch, channels, device="cuda").bfloat16().float()
        weights = torch.randn(channels, 4, device="cuda").bfloat16().float()
        expected_out = torch.zeros_like(xbc)
        for row, slot in enumerate(host_slots):
            if slot == -1:
                continue
            full = torch.cat((conv[slot], xbc[row, :, None]), -1)
            expected_out[row] = (full * weights).sum(-1)
            expected_conv[slot] = full[:, 1:]
        out = causal_conv1d_update(
            xbc,
            conv,
            weights,
            activation=None,
            conv_state_indices=slots,
            enable_fp_fusion=False,
        )
        real_rows = torch.tensor(
            [i for i, s in enumerate(host_slots) if s != -1], device="cuda"
        )
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        conv_delta = difference(out[real_rows], expected_out[real_rows])
        # Isolate state/readout from convolution reduction: both receive x/B/C
        # from the SAME reference convolution output.
        x, b, c = torch.nn.functional.silu(expected_out).split([4096, 1024, 1024], -1)
        x, b, c = (
            x.reshape(batch, h, p),
            b.reshape(batch, groups, n),
            c.reshape(batch, groups, n),
        )
        dt = torch.nn.functional.softplus(torch.randn(batch, h, device="cuda"))
        a = -torch.rand(h, device="cuda")
        decay = (dt * a).exp()
        d = torch.randn(h, device="cuda")
        state = torch.randn(batch + 1, h, p, n, device="cuda")
        expected_state = state.clone()
        expected_y = torch.zeros_like(x)
        for row, slot in enumerate(host_slots):
            if slot == -1:
                continue
            bb, cc = b[row].repeat_interleave(8, 0), c[row].repeat_interleave(8, 0)
            current = (
                state[slot] * decay[row, :, None, None]
                + (dt[row, :, None] * x[row])[:, :, None] * bb[:, None]
            )
            expected_state[slot] = current
            expected_y[row] = (
                torch.einsum("hpn,hn->hp", current, cc) + x[row] * d[:, None]
            )
        y = torch.zeros_like(x)
        selective_state_update(
            state,
            x,
            dt[..., None].expand_as(x),
            a[:, None, None].expand(h, p, n),
            b,
            c,
            d[:, None].expand(h, p),
            state_batch_indices=slots,
            out=y,
            reference_decay=decay,
        )
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(
            y[-1:], expected_y[-1:], rtol=0, atol=0
        ) if batch > 1 else None
        rows.append(
            {
                "B": batch,
                "conv": conv_delta,
                "ssm_state": difference(state, expected_state),
                "readout": difference(y[real_rows], expected_y[real_rows]),
                "readout_bf16": difference(
                    y[real_rows].bfloat16(), expected_y[real_rows].bfloat16()
                ),
                "history_bitwise": True,
                "pad_does_not_write": True,
            }
        )
    # Cross-layer warm projection and successor: unchanged seed/math, but CUDA
    # batch solver/GEMM choices may differ from reference batch=1.
    directions = torch.randn(2, 64, 64, device="cuda")
    kwargs = {"n": 128, "rank": 16, "window": 16, "conv_dim": 4, "conv_width": 3}
    new, old = (
        Bridge(3, (0, 1), directions, **kwargs),
        Old(3, (0, 1), directions, **kwargs),
    )
    for layer in (0, 1):
        for slot in (1, 2):
            state = torch.randn(64, 64, 128, device="cuda")
            conv = torch.randn(4, 3, device="cuda")
            new.initialize(layer, slot, state, conv)
            old.initialize(layer, slot, state, conv)
    layers = torch.tensor([0, 0, 1, 1], device="cuda")
    slots = torch.tensor([1, 2, 1, 2], device="cuda")
    states = (
        new.dense[layers, slots] * 0.8
        + torch.randn(4, 64, 64, 128, device="cuda") * 0.1
    )
    new._prune(layers, slots, states, warm=True)
    for row, (layer, slot) in enumerate(
        zip(layers.cpu().tolist(), slots.cpu().tolist())
    ):
        old.prune(layer, slot, states[row], warm=True)
    projected = torch.stack(
        [old.materialize(l, s) for l, s in ((0, 1), (0, 2), (1, 1), (1, 2))]
    )
    projection_delta = difference(new.dense[layers, slots], projected)
    report = {
        "status": "PASS"
        if all(r["conv"]["bitwise"] and r["readout"]["bitwise"] for r in rows)
        and projection_delta["bitwise"]
        else "DIFFERENCES_RECORDED",
        "source_sha": plan["bridge_sha"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "rows": rows,
        "warm_projection": projection_delta,
        "rounding_sources": [
            "native conv sequential accumulation vs torch last-axis reduction",
            "native SSU tl.sum tree vs reference einsum/cuBLAS readout",
            "batched cuBLAS/Cholesky/eigh vs single-slot calls",
        ],
        "tolerance_changed": False,
        "state_gate": "FP32 recurrence and conv history bitwise",
        "downstream_gate": "paired GSM200; stage differences remain recorded",
    }
    # Record compiled-kernel admission/resources; artifact availability is runtime dependent.
    resources = []
    from sglang.kernels.ops.mamba import causal_conv1d_triton as conv_module
    from sglang.kernels.ops.mamba.triton_ops import mamba_ssm as ssu_module

    for name, jit in [
        ("conv", conv_module._causal_conv1d_update_kernel),
        ("ssu", ssu_module._selective_scan_update_kernel),
    ]:
        while not hasattr(jit, "device_caches") and hasattr(jit, "fn"):
            jit = jit.fn
        caches = getattr(jit, "device_caches", {})
        for entry in caches.values():
            cache = entry[0] if isinstance(entry, tuple) else entry
            if not isinstance(cache, dict):
                continue
            for compiled in cache.values():
                metadata = getattr(compiled, "metadata", None)
                resources.append(
                    {
                        "kernel": name,
                        "registers": getattr(compiled, "n_regs", None),
                        "spills": getattr(compiled, "n_spills", None),
                        "shared_bytes": getattr(metadata, "shared", None),
                        "num_warps": getattr(metadata, "num_warps", None),
                    }
                )
    dump = root / "kernel-resources"
    dump.mkdir(exist_ok=True)
    cubins = list(
        Path(os.environ.get("TRITON_CACHE_DIR", str(root / "triton"))).rglob("*.cubin")
    )
    inspector = shutil.which("cuobjdump")
    if inspector:
        with (dump / "cubin-resource-usage.txt").open("w") as out:
            for cubin in cubins:
                result = subprocess.run(
                    [inspector, "--dump-resource-usage", str(cubin)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                out.write(str(cubin) + "\n" + result.stdout + result.stderr + "\n")
    report["cubin_count"] = len(cubins)
    report["cuobjdump_available"] = inspector is not None
    report["resource_inventory"] = resources
    atomic(root / "cuda-stages.json", report)
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,clocks.current.sm,clocks.current.memory,uuid",
            "--format=csv",
        ],
        stdout=(root / "hardware.csv").open("w"),
        check=True,
    )
    return report


def worker(plan, arm, gpu, root, *, micro=True, gsm=True):
    root.mkdir(parents=True, exist_ok=True)
    source = plan["bridge_source"] if arm == "bridge" else plan["published_source"]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(
            ("SGLANG_DUET_", "TWINSTAR_LIGHTNING_", "SGLANG_EXTERNAL_MODEL_PACKAGE")
        ):
            env.pop(key)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        NVIDIA_TF32_OVERRIDE="0",
        SGLANG_MAMBA_CONV_DTYPE="float32",
        SGLANG_MAMBA_SSM_DTYPE="float32",
        LTNBRIDGE_PROBE_DIR=str(root / "probe") if micro else "",
        PYTHONPATH=":".join(
            [plan["probe_source"], source + "/python", source, *plan["helper_paths"]]
        ),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        HF_HOME="/work/hf-cache",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_DATASETS_CACHE=plan["datasets_cache"],
        TOKENIZERS_PARALLELISM="false",
    )
    port = 30700 + gpu
    argv = list(plan["service_argv"])
    argv[argv.index("--port") + 1] = str(port)
    if "--disable-cuda-graph" not in argv:
        argv.append("--disable-cuda-graph")
    if arm != "original-off":
        argv += ["--duet-release", plan["release"], "--duet-numerics", "reference"]
        env.update(
            SGLANG_DUET_DIR=plan["release"],
            SGLANG_EXTERNAL_MODEL_PACKAGE="sglang.srt.models.lightning_duet",
        )
    command = [sys.executable, "-m", "sglang.launch_server", *argv]
    for key in (
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "TORCH_EXTENSIONS_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "SGLANG_CACHE_DIR",
        "HF_MODULES_CACHE",
    ):
        env[key] = str(root / key.lower())
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    atomic(
        root / "execution.json",
        {
            "arm": arm,
            "gpu": gpu,
            "source_sha": plan["bridge_sha"]
            if arm == "bridge"
            else plan["published_sha"],
            "source": source,
            "argv": command,
            "env": {
                k: v
                for k, v in env.items()
                if k.startswith(
                    ("CUDA_", "SGLANG_", "LTNBRIDGE_", "PYTHON", "NVIDIA_TF32", "HF_")
                )
            },
            "job": os.environ.get("SLURM_JOB_ID"),
        },
    )
    url = f"http://127.0.0.1:{port}"
    log = (root / "server.log").open("w")
    proc = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    started = time.monotonic()
    try:
        while time.monotonic() - started < 1200:
            if proc.poll() is not None:
                raise RuntimeError(f"{arm} server exited {proc.returncode}")
            try:
                urllib.request.urlopen(url + "/health", timeout=5).close()
                break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError(arm + " health")
        atomic(
            root / "load.json",
            {
                "seconds": time.monotonic() - started,
                "info": http(url, "/get_server_info"),
            },
        )
        for _ in range(2):
            http(
                url,
                "/generate",
                {
                    "input_ids": [1000] * 128,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                        "ignore_eos": True,
                    },
                },
            )
        if micro:
            probe = root / "probe"
            probe.mkdir(exist_ok=True)
            micro = []
            for batch in (1, 8, 32):
                # Warm each actual shape (including a W16 cut) before application
                # timing. Kernel compilation/profiling stay outside scored samples.
                http(
                    url,
                    "/generate",
                    {
                        "input_ids": [[1000] * 128 for _ in range(batch)],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 65,
                            "ignore_eos": True,
                        },
                    },
                )
                tag = f"B{batch}"
                atomic(probe / "trigger.json", {"tag": tag, "mode": "events"})
                response = http(
                    url,
                    "/generate",
                    {
                        "input_ids": [[1000] * 128 for _ in range(batch)],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 65,
                            "ignore_eos": True,
                        },
                    },
                )
                (probe / "trigger.json").unlink()
                atomic(root / f"{tag}-responses.json", response)
                time.sleep(0.1)
                atomic(
                    probe / "trigger.json",
                    {"tag": tag + "-trace", "mode": "trace", "steps": 17},
                )
                http(
                    url,
                    "/generate",
                    {
                        "input_ids": [[1000] * 128 for _ in range(batch)],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 18,
                            "ignore_eos": True,
                        },
                    },
                )
                (probe / "trigger.json").unlink()
                micro.append(
                    {
                        "tag": tag,
                        "requested_batch": batch,
                        "input_ids_sha256": hashlib.sha256(
                            json.dumps([1000] * 128).encode()
                        ).hexdigest(),
                    }
                )
            atomic(
                root / "micro-complete.json",
                {
                    "complete": True,
                    "rows": micro,
                    "formal_window": False,
                    "event_timing_unprofiled": True,
                    "launch_diagnostics_separate": True,
                },
            )
        if gsm:
            # The pinned endpoint evaluator preserves questions, gold, prompt,
            # greedy2048/chat-boxed scoring. Stock and bridge share this file.
            sys.path[:0] = plan["helper_paths"]
            from gsm_endpoint import run

            run(url, plan["model"], root / "gsm200.json")
        atomic(
            root / "complete.json",
            {"complete": True, "arm": arm, "seconds": time.monotonic() - started},
        )
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        log.close()


def reduce(plan, root):
    arms = {}
    for arm in ("bridge", "published-off", "original-off"):
        arms[arm] = {}
        for order in ("AB", "BA"):
            measured = root / "micro" / order / arm
            rows = [
                json.loads(line)
                for line in (measured / "probe/steps.jsonl").read_text().splitlines()
            ]
            launches = [
                json.loads(line)
                for line in (measured / "probe/launches.jsonl").read_text().splitlines()
            ]
            by_tag = {}
            for tag in sorted({r["tag"] for r in rows}):
                selected = [r for r in rows if r["tag"] == tag]
                traces = [r for r in launches if r["tag"] == tag + "-trace"]
                by_tag[tag] = {
                    "steps": len(selected),
                    "actual_B": sorted({r["batch_size"] for r in selected}),
                    "mean_device_ms": statistics.mean(
                        r["device_span_ms"] for r in selected
                    ),
                    "mean_host_ms": statistics.mean(
                        r["host_submit_ms"] for r in selected
                    ),
                    "phase_device_ms": {
                        str(phase): statistics.mean(
                            r["device_span_ms"]
                            for r in selected
                            if r["step"] % 16 == phase
                        )
                        for phase in range(16)
                        if any(r["step"] % 16 == phase for r in selected)
                    },
                    "kernel_counts": [r["kernel_events"] for r in traces],
                    "cpu_item": [r["cpu_item"] for r in traces],
                    "cpu_local_scalar": [r["cpu_local_scalar"] for r in traces],
                }
            arms[arm][order] = by_tag
    a, b = [
        json.loads((root / "gsm" / arm / "gsm200.json").read_text())
        for arm in ("original-off", "bridge")
    ]
    assert (
        a["complete"]
        and b["complete"]
        and len(a["results"]) == len(b["results"]) == 200
    )
    for x, y in zip(a["results"], b["results"]):
        assert (x["id"], x["gold"], x["prompt_hash"]) == (
            y["id"],
            y["gold"],
            y["prompt_hash"],
        )
    bad = sum(
        x["correct"] and not y["correct"] for x, y in zip(a["results"], b["results"])
    )
    good = sum(
        not x["correct"] and y["correct"] for x, y in zip(a["results"], b["results"])
    )
    se = math.sqrt(bad + good)
    paired = {
        "n": 200,
        "original_correct": sum(x["correct"] for x in a["results"]),
        "bridge_correct": sum(x["correct"] for x in b["results"]),
        "b": bad,
        "c": good,
        "delta": good - bad,
        "se": se,
        "gate": "PASS" if good - bad >= -2 * se else "FAIL",
        "same_id_gold_prompt": True,
    }
    atomic(
        root / "qualification.json",
        {
            "source_sha": plan["bridge_sha"],
            "original_sha": plan["published_sha"],
            "micro": arms,
            "gsm200": paired,
            "formal_window": False,
            "graph": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stages-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    # The GSM evaluator runs in this coordinator, not in the server children.
    # Bind it to the same frozen offline dataset/tokenizer cache as the servers.
    client_modules = args.out / "client-hf-modules"
    client_modules.mkdir(exist_ok=True)
    os.environ.update(
        HF_HOME="/work/hf-cache",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_DATASETS_CACHE=plan["datasets_cache"],
        HF_MODULES_CACHE=str(client_modules),
    )
    if args.stages_only:
        cuda_stages(plan, args.out)
        return
    # GPU3: stages; GPU0: same-device micro controls in both arm orders;
    # GPU1/2: paired GSM without profiling.
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="3",
        PYTHONPATH=plan["bridge_source"] + "/python:" + ":".join(plan["helper_paths"]),
    )
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--plan",
            str(args.plan),
            "--out",
            str(args.out),
            "--stages-only",
        ],
        env=env,
        check=True,
    )
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {
            arm: pool.submit(
                worker, plan, arm, gpu, args.out / "gsm" / arm, micro=False, gsm=True
            )
            for gpu, arm in ((1, "bridge"), (2, "original-off"))
        }
        for order, arms in (
            ("AB", ("bridge", "published-off", "original-off")),
            ("BA", ("original-off", "published-off", "bridge")),
        ):
            for arm in arms:
                try:
                    worker(
                        plan,
                        arm,
                        0,
                        args.out / "micro" / order / arm,
                        micro=True,
                        gsm=False,
                    )
                except Exception as error:
                    failures[f"micro/{order}/{arm}"] = repr(error)
                    atomic(args.out / "failures.json", failures)
        for arm, future in tasks.items():
            try:
                future.result()
            except Exception as error:
                failures["gsm/" + arm] = repr(error)
                atomic(args.out / "failures.json", failures)
    if failures:
        raise RuntimeError(failures)
    reduce(plan, args.out)


if __name__ == "__main__":
    main()
