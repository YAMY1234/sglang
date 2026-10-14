"""Unprofiled CUDA event timing and separate per-forward launch diagnostics."""

import collections
import functools
import gzip
import json
import os
import threading
import time
from pathlib import Path

import torch


def install(cls):
    original = cls.forward
    root = Path(os.environ["LTNBRIDGE_PROBE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    pending = []
    counts = collections.Counter()
    started = False

    def flush():
        with lock:
            ready = [x for x in pending if x[1].query()]
            for item in ready:
                start, end, row = item
                row["device_span_ms"] = start.elapsed_time(end)
                with (root / "steps.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
                pending.remove(item)

    def writer():
        while True:
            flush()
            time.sleep(0.02)

    @functools.wraps(original)
    def forward(self, fb, *args, **kwargs):
        nonlocal started
        trigger = root / "trigger.json"
        if not fb.forward_mode.is_decode() or not trigger.exists():
            return original(self, fb, *args, **kwargs)
        config = json.loads(trigger.read_text())
        if not started:
            threading.Thread(target=writer, daemon=True).start()
            started = True
        key = config["tag"]
        step = counts[key]
        counts[key] += 1
        row = {
            "tag": key,
            "step": step,
            "batch_size": fb.batch_size,
            "mode": config["mode"],
            "epoch_ns": time.time_ns(),
        }
        runtime = getattr(getattr(self, "model", None), "lightning_runtime", None)
        pool = getattr(runtime, "pool", None)
        if pool is not None and getattr(runtime, "factored_decode", False):
            # Device counters remain device-only; synthetic micro starts each
            # request at phase zero. Do not label the stale bridge host mirror.
            row["phase_source"] = "device_fcount; infer synthetic phase from step"
        elif pool is not None and hasattr(pool, "_host_count"):
            request_ids = fb.req_pool_indices_cpu.tolist()
            host_slots = [runtime.host_slots_by_request[r] for r in request_ids]
            phases = pool._host_count[:, host_slots]
            row["slot_phases_before"] = sorted(set(phases.reshape(-1).tolist()))
            row["due_layer_slots"] = (
                int(((phases + 1) == pool.window).sum()) if not pool.exact_mode else 0
            )
        if config["mode"] == "trace":
            if step >= config.get("steps", 17):
                return original(self, fb, *args, **kwargs)
            with (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    with_stack=False,
                    profile_memory=False,
                ) as prof,
                torch.profiler.record_function("LTNBRIDGE_FORWARD"),
            ):
                result = original(self, fb, *args, **kwargs)
            events = prof.events()
            cuda_activities = [
                e for e in events if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            row.update(
                cuda_activity_events=len(cuda_activities),
                cuda_activity_device_sum_ms=sum(
                    e.device_time_total for e in cuda_activities
                )
                / 1000,
                cpu_item=sum(e.name == "aten::item" for e in events),
                cpu_local_scalar=sum(
                    e.name == "aten::_local_scalar_dense" for e in events
                ),
            )
            path = root / f"{key}-step{step:03d}.json"
            prof.export_chrome_trace(str(path))
            # CUDA FunctionEvents include memcpy and GPU annotations. Chrome
            # activity categories distinguish actual kernel launches.
            trace = json.loads(path.read_text())
            kernels = [
                event for event in trace["traceEvents"] if event.get("cat") == "kernel"
            ]
            row.update(
                kernel_events=len(kernels),
                kernel_device_sum_ms=sum(event.get("dur", 0) for event in kernels)
                / 1000,
            )
            with (
                path.open("rb") as src,
                gzip.open(str(path) + ".gz", "wb", compresslevel=2) as dst,
            ):
                dst.write(src.read())
            path.unlink()
            with (root / "launches.jsonl").open("a") as out:
                out.write(json.dumps(row) + "\n")
            return result
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        wall = time.perf_counter()
        result = original(self, fb, *args, **kwargs)
        end.record()
        row["host_submit_ms"] = 1000 * (time.perf_counter() - wall)
        with lock:
            pending.append((start, end, row))
        return result

    cls.forward = forward
