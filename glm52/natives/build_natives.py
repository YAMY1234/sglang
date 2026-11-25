# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build the GLM-5.2 prefill natives on CPU in the serving image (no CUDA context).

usage: python3 build_natives.py <sglang repo root> <out dir> [<dir of reference binaries>]

- topk_overflow_prefix_v2_545e00d5_sm107.so: the prefill DSA top-k overflow-prefix op
  (SGLANG_DSA_TOPK_OVERFLOW_PREFIX). Same name, sources, flags and build directory as the
  reference build, so the two can be byte-compared; the loader
  (sglang/srt/layers/attention/dsa/topk_activation.py) checks the sha.
- hcb_prefill_trt_port_107.so: the prefill DSA top-k selector port (SGLANG_HCB_PREFILL_TRT_PORT).
Writes <out>/BUILD.json. With a reference directory, compares each result with the binary of the
same name there (sha, and SASS through cuobjdump). Exits non-zero when REQUIRE_FROZEN_TOPK_SHA=1
and the overflow op's sha differs from the reference sha below.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REFERENCE_SHA = {
    "topk_overflow_prefix_v2_545e00d5_sm107.so": "f2b7bc2e09c4331fa5aa7f3624e06ee2d89f62d8854d651f7d970289b7497b36",
    "hcb_prefill_trt_port_107.so": "a9e170741b75255bdf35cfcd84043ecf77872eff56e7442d1bedaeb7d9a16b64",
}
# The overflow op's sources as the reference build compiled them (native_binding.cc only
# gained a license header; its TORCH_LIBRARY_FRAGMENT line, which the binary records, is unchanged).
TOPK_SOURCES = {
    "topk_overflow_prefix.cu": "29b0561a1388b92e49c842c2d5cedb45e9ffeae7fa97ab415c2798e5509a670b",
    "native_binding.cc": "44bf40f8ae3241b2b4bf174a6da56addbbfd4fb46b104e79f74185695588afa3",
    "include/sgl_kernel/dsa/legacy_radix_topk_overflow_prefix_v2.cuh": "545e00d515e5d8d77f249921a5d8e4fbb85e2129a407edf890d3805a165252b0",
    "include/sgl_kernel/dsa/legacy_radix_topk_overflow_prefix_serving_v2.cuh": "b1b6797e1ffa7edc95f1f6f42bba06cf6c5047b258fbf1586131f9eec81a5db2",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sass(path):
    out = subprocess.run(["cuobjdump", "-sass", str(path)], capture_output=True, text=True, check=True).stdout
    # Keep the instructions; drop the per-file headers (names, sizes, identifiers).
    skip = ("Fatbin", "=====", "arch", "code version", "host", "compile_size", "identifier")
    return "\n".join(l for l in out.splitlines() if not l.strip().startswith(skip))


def build_topk_overflow(repo: Path, load) -> Path:
    # The reference build compiled from /work/src into /work/build.
    src, build = Path("/work/src"), Path("/work/build")
    (src / "include/sgl_kernel/dsa").mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    aot = repo / "python/sglang/kernels/aot"
    shutil.copy(aot / "csrc/elementwise/topk_overflow_prefix.cu", src)
    for name in ("legacy_radix_topk_overflow_prefix_v2.cuh", "legacy_radix_topk_overflow_prefix_serving_v2.cuh"):
        shutil.copy(aot / "include/sgl_kernel/dsa" / name, src / "include/sgl_kernel/dsa")
    shutil.copy(repo / "glm52/natives/topk-overflow-prefix/native_binding.cc", src)
    for rel, want in TOPK_SOURCES.items():
        assert sha(src / rel) == want, f"{rel} is not the reviewed source"
    flags = ["-O3", "-std=c++20", "-DNDEBUG", "-gencode=arch=compute_107,code=sm_107"]
    return Path(load(
        name="topk_overflow_prefix_v2_545e00d5_sm107",
        sources=[str(src / "topk_overflow_prefix.cu"), str(src / "native_binding.cc")],
        extra_include_paths=[str(src / "include")],
        extra_cflags=["-O3", "-std=c++20", "-DNDEBUG"],
        extra_cuda_cflags=flags, build_directory=str(build),
        with_cuda=True, is_python_module=False, verbose=True,
    ))


def build_hcb(repo: Path, load) -> Path:
    # The reference layout: the .cu and both headers in one src/ directory (also the include path).
    src, build = Path("/work/hcb/src"), Path("/work/build-sm107")
    src.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    for f in ("prefill_port.cu", "trt_selector.cuh"):
        shutil.copy(repo / "glm52/natives/hcb-prefill-port" / f, src)
    shutil.copy(repo / "python/sglang/kernels/aot/include/sgl_kernel/dsa/legacy_radix_topk.cuh", src)
    return Path(load(
        name="hcb_prefill_trt_port_107",
        sources=[str(src / "prefill_port.cu")], extra_include_paths=[str(src)],
        build_directory=str(build),
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_107,code=sm_107"],
        with_cuda=True, is_python_module=False, verbose=True,
    ))


def main():
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    reference = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.utils.cpp_extension import load

    assert torch.__version__ == "2.15.0+cu134" and torch.version.cuda == "13.4", torch.__version__
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "started": time.time(), "natives": {}}
    for name, build in (
        ("topk_overflow_prefix_v2_545e00d5_sm107.so", build_topk_overflow),
        ("hcb_prefill_trt_port_107.so", build_hcb),
    ):
        shutil.copy(build(repo, load), out / name)
        entry = {"sha256": sha(out / name), "bytes": (out / name).stat().st_size,
                 "reference_sha256": REFERENCE_SHA[name]}
        entry["same_as_reference"] = entry["sha256"] == REFERENCE_SHA[name]
        if reference is not None and (reference / name).exists():
            assert sha(reference / name) == REFERENCE_SHA[name], f"{reference / name} is not the reference binary"
            entry["same_sass_as_reference"] = sass(out / name) == sass(reference / name)
        report["natives"][name] = entry
    assert not torch.cuda.is_initialized()
    report["ended"] = time.time()
    (out / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    topk = report["natives"]["topk_overflow_prefix_v2_545e00d5_sm107.so"]
    if os.environ.get("REQUIRE_FROZEN_TOPK_SHA", "1") == "1" and not topk["same_as_reference"]:
        sys.exit("the overflow op differs from the reference sha; update "
                 "python/sglang/srt/layers/attention/dsa/topk_overflow_native.sha256 if it is accepted")


if __name__ == "__main__":
    main()
