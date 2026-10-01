"""CPU-only execution of the actual stage-transition method, without CUDA imports."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS


def check(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SchedulerProfilerManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_profile_batch_predicate")
    method.args.args[1].annotation = None
    code = ast.Module(body=[method], type_ignores=[])
    calls = []
    limit = [64]
    scope = dict(
        envs=NS(SGLANG_PROFILE_V2=NS(get=lambda: False),
                SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS=NS(get=lambda: limit[0])),
        torch=NS(distributed=NS(barrier=lambda group: calls.append(("barrier", group)))),
        ForwardMode=NS(EXTEND="extend", DECODE="decode"),
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    run = scope[method.name]
    manager = NS(profile_by_stage=True, profiler_decode_ct=0,
                 profile_in_progress=True, profiler_target_decode_ct=192,
                 dp_tp_cpu_group="tp-cpu")

    def stop(stage):
        calls.append(("stop-export-gc", stage))
        manager.profile_in_progress = False

    def start(mode):
        calls.append(("start-profiler", "decode"))
        manager.profile_in_progress = True

    manager._stop_profile = stop
    manager._start_profile = start
    mode = NS(is_prefill=lambda: False, is_decode=lambda: True, is_idle=lambda: False)
    batch = NS(forward_mode=mode, batch_size=lambda: 32)
    run(manager, batch)
    assert calls == [("stop-export-gc", "extend")]
    assert manager.profiler_decode_ct == 0  # Admission wait must not start profiling.
    calls.clear()
    batch.batch_size = lambda: 64
    run(manager, batch)
    assert calls == [("barrier", "tp-cpu"), ("start-profiler", "decode"), ("barrier", "tp-cpu")]
    assert manager.profiler_decode_ct == 1
    calls.clear()
    run(manager, batch)
    assert calls == [] and manager.profiler_decode_ct == 2  # No per-step barrier.
    manager.profile_by_stage = False
    manager.profiler_target_forward_ct = None
    manager.profiler_start_forward_ct = None
    run(manager, batch)
    assert calls == []  # Normal, unprofiled static timing is unchanged.
    return dict(passed=True, cases=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3] /
                    "python/sglang/srt/managers/scheduler_components/profiler_manager.py")
    args = ap.parse_args()
    print(check(args.source))
