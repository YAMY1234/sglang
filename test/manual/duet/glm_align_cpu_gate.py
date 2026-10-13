"""Required GLM P1/P4 CPU N/S gate, with explicit counts and CUDA-init proof."""

import argparse
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

ROOT = Path(__file__).resolve().parents[3]


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, str(ROOT / "test/registered/unit/duet"))
    names = [
        "test_glm_duet",
        "test_glm_align_prepare",
        "test_glm_safe_packed_dispatch",
        "test_kda_state_policy",
        "test_duet_dense_state_policy",
        "test_duet_batched_projection",
    ]
    start = time.monotonic()
    with (out / "units.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.TestLoader().loadTestsFromNames(names)
        )
    units = {
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skips": len(result.skipped),
        "passed": result.wasSuccessful(),
        "seconds": time.monotonic() - start,
    }
    (out / "units.json").write_text(json.dumps(units, indent=2) + "\n")
    if not result.wasSuccessful():
        raise RuntimeError("GLM CPU unit gate failed; see units.log")
    with (out / "packed-cpu.log").open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "test/manual/duet/glm_safe_packed_cpu_gate.py"),
                "--output",
                str(out / "packed-cpu.json"),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    packed = json.loads((out / "packed-cpu.json").read_text())
    assert (
        packed["passed"]
        and not packed["cuda_initialized"]
        and not torch.cuda.is_initialized()
    )
    proof = {
        "passed": True,
        "units": units,
        "packed_cases": packed["cases"],
        "cpu_cases_passed": units["tests"] + packed["cases"],
        "cuda_initialized": False,
        "torch": torch.__version__,
        "seconds": time.monotonic() - start,
        "scope": "GLM P1/P4 CPU N/S and unchanged GLM/KDA/dense/batched regression; actual GPU graph is separate",
    }
    (out / "complete.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
