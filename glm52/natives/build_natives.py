# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build the GLM-5.2 prefill natives on CPU in the serving image (no CUDA context).

usage: python3 build_natives.py <sglang repo root> <out dir> [<reference dir> [<pinned dir>]]
       python3 build_natives.py --selftest   (the SASS comparison; also run first by a build)

- topk_overflow_prefix_v2_545e00d5_sm107.so: the prefill DSA top-k overflow-prefix op
  (SGLANG_DSA_TOPK_OVERFLOW_PREFIX). Same name, sources, flags and build directory as the
  reference build; the loader (sglang/srt/layers/attention/dsa/topk_activation.py) checks the sha
  listed in TOPK_MANIFEST.
- hcb_prefill_trt_port_107.so: the prefill DSA top-k selector port (SGLANG_HCB_PREFILL_TRT_PORT).
nvcc output is not byte-reproducible (internal symbol names carry per-build hashes), so when a
rebuild's sha is not an expected one, a binary of the same name from the pinned directory (an
earlier image build) or the reference directory is installed instead if its sha is expected and its
SASS (cuobjdump, without nvcc's per-build names) equals the rebuild's. Writes <out>/BUILD.json.
Exits non-zero when REQUIRE_FROZEN_TOPK_SHA=1 (the default) and the installed overflow op is not
listed in TOPK_MANIFEST, since the loader would refuse it.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REFERENCE_SHA = {
    "topk_overflow_prefix_v2_545e00d5_sm107.so": "f2b7bc2e09c4331fa5aa7f3624e06ee2d89f62d8854d651f7d970289b7497b36",
    "hcb_prefill_trt_port_107.so": "a9e170741b75255bdf35cfcd84043ecf77872eff56e7442d1bedaeb7d9a16b64",
}
# Expected shas: the overflow op's accepted shas are TOPK_MANIFEST's (the first image build and the
# reference, same SASS); the selector port's is the first image build's.
TOPK_MANIFEST = "python/sglang/srt/layers/attention/dsa/topk_overflow_native.sha256"
HCB_EXPECTED_SHA = "40f43eeb177ba788c37e7a34c53b652acc6d042524f66c972240265470a5cf94"
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


# nvcc names an anonymous namespace (and an internal-linkage symbol) after a hash of the build path and a
# per-run id: <len>_GLOBAL__N__<hex>_<len>_<file>_cu_<hex>_<id> in mangled names (or _INTERNAL_...), and its
# temporaries tmpxft_<id>. Builds of one source differ in these alone, so SASS is compared without them.
_ANON = re.compile(r"(\d+)(?=_GLOBAL__N__|_INTERNAL_)")
_ANON_HEAD = re.compile(r"(_GLOBAL__N__|_INTERNAL_)[0-9a-f]+_(\d+)_")
_ANON_TAIL = re.compile(r"_[0-9a-f]+(_\d+)?")


def _anon_name(ident):
    """`_GLOBAL__N__<file>` for one anonymous-namespace / internal name, or None if ident is not one."""
    m = _ANON_HEAD.match(ident)
    if not m:
        return None
    n = int(m.group(2))
    file, tail = ident[m.end() : m.end() + n], ident[m.end() + n :]
    return f"{m.group(1)}<{file}>" if len(file) == n and _ANON_TAIL.fullmatch(tail) else None


def without_build_names(text):
    out, pos = [], 0
    while m := _ANON.search(text, pos):
        digits, start, name = m.group(1), m.end(), None
        for k in range(len(digits)):  # the mangled length prefix: the longest digit suffix that fits
            n = int(digits[k:])
            if name := _anon_name(text[start : start + n]):
                out += [text[pos : m.start() + k], name]
                pos = start + n
                break
        if name is None:
            out.append(text[pos:start])
            pos = start
    out.append(text[pos:])
    # The same names without a length prefix (not mangled): the hashes and the id go.
    text = re.sub(r"(_GLOBAL__N__|_INTERNAL_)[0-9a-f]+_", r"\1<h>_", "".join(out))
    text = re.sub(r"_cu_[0-9a-f]{8}(_\d+)?\b", "_cu_<h>", text)
    return re.sub(r"tmpxft_[0-9a-f]+_[0-9a-f]+(-\d+)?", "tmpxft_<id>", text)


def sass(path):
    out = subprocess.run(["cuobjdump", "-sass", str(path)], capture_output=True, text=True, check=True).stdout
    # Keep the instructions; drop the per-file headers (names, sizes, identifiers) and nvcc's per-build names.
    skip = ("Fatbin", "=====", "arch", "code version", "host", "compile_size", "identifier")
    return without_build_names("\n".join(l for l in out.splitlines() if not l.strip().startswith(skip)))


def _mangled(kind, path_hash, file, file_hash, run_id, rest):
    ident = f"{kind}{path_hash}_{len(file)}_{file}_{file_hash}" + (f"_{run_id}" if run_id else "")
    return f"_ZN{len(ident)}{ident}{rest}"


SELFTEST_CU = r"""
namespace {
__device__ __noinline__ float scaled(float x) { return x * FACTOR; }
__global__ void scale_kernel(float* x, int n) { int i = threadIdx.x; if (i < n) x[i] = scaled(x[i]); }
}
__global__ void entry_kernel(float* x) { x[threadIdx.x] = scaled(x[threadIdx.x]); }
void launch(float* x, int n) { scale_kernel<<<1, 256>>>(x, n); entry_kernel<<<1, 256>>>(x); }
"""


def selftest(require_binary=False):
    """The SASS comparison on builds that differ only in nvcc's per-build names, and on real differences:
    text samples in cuobjdump's format (the two hcb builds' names), then, with nvcc and cuobjdump, one
    source built from two directories (equal) and with another constant (different)."""
    def dump(name_a, name_b, instr="IMAD R1, R2, 0x2, RZ"):
        return (f"\t\tFunction : {name_a}\n\t/*0000*/  {instr} ;\n"
                f"\t/*0010*/  CALL.REL.NOINC `({name_b}) ;\n\t// tmpxft_0000a1b2_00000000-6_prefill_port\n")
    k = "16topk_page_kernelILi512ELb1ELb1EEEvPKfPKiS4_PiS4_S4_lli"
    old = lambda rest, file="prefill_port_cu": _mangled("_GLOBAL__N__", "4da679a6", file, "61aee233", "81", rest)
    new = lambda rest, file="prefill_port_cu": _mangled("_GLOBAL__N__", "229f5afd", file, "f2e4ace0", "3625996", rest)
    internal = _mangled("_INTERNAL_", "d7e87ae8", "topk_overflow_prefix_cu", "095d8ec8", "", "4funcEv")
    a = dump(old(k), internal) + "tmpxft_0000a1b2_00000000-6"
    b = dump(new(k), internal.replace("d7e87ae8", "0badc0de").replace("095d8ec8", "1234abcd")) + "tmpxft_00ff0001_00000000-6"
    cases = {  # name: (text a, text b, equal after normalizing)
        "names only": (a, b, True),
        "one instruction": (a, b.replace("0x2", "0x3"), False),
        "another kernel": (a, b.replace("16topk_page_kernel", "16topk_pagf_kernel"), False),
        "another source file": (a, dump(new(k, "prefill_pore_cu"), internal) + "tmpxft_0000a1b2_00000000-6", False),
    }
    failed = [c for c, (x, y, eq) in cases.items() if (without_build_names(x) == without_build_names(y)) != eq]
    failed += ["raw texts equal"] if a == b else []
    result = {"text_cases": len(cases), "failed": failed}
    cuda_bin = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin"
    nvcc, cuobjdump = shutil.which("nvcc") or cuda_bin / "nvcc", shutil.which("cuobjdump") or cuda_bin / "cuobjdump"
    if Path(nvcc).exists() and Path(cuobjdump).exists():
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dumps = []
            for sub, factor in (("a", "2.0f"), ("build-dir-b", "2.0f"), ("c", "3.0f")):
                d = Path(tmp) / sub
                d.mkdir()
                (d / "selftest_kernel.cu").write_text(SELFTEST_CU)
                subprocess.run([str(nvcc), "-cubin", "-O3", "-gencode=arch=compute_107,code=sm_107", f"-DFACTOR={factor}",
                                "-o", "k.cubin", "selftest_kernel.cu"], cwd=d, check=True)
                raw = subprocess.run([str(cuobjdump), "-sass", str(d / "k.cubin")], capture_output=True, text=True, check=True).stdout
                dumps.append((raw, sass(d / "k.cubin")))
        (raw_a, sass_a), (raw_b, sass_b), (_, sass_c) = dumps
        result["binary"] = {"names_differ": raw_a != raw_b, "equal_after_normalizing": sass_a == sass_b,
                            "other_constant_differs": sass_a != sass_c, "anonymous_names": "_GLOBAL__N__" in raw_a}
        # names_differ / anonymous_names only say how much the build exercised the normalizing
        failed += [f"binary: {c}" for c in ("equal_after_normalizing", "other_constant_differs") if not result["binary"][c]]
    elif require_binary:
        failed.append("binary: no nvcc / cuobjdump")
    else:
        result["binary"] = "skipped: no nvcc / cuobjdump"
    return result


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
    if sys.argv[1:] == ["--selftest"]:
        result = selftest()
        print(json.dumps(result, indent=2))
        sys.exit(1 if result["failed"] else 0)
    check = selftest(require_binary=True)
    if check["failed"]:
        sys.exit(f"the SASS comparison's selftest failed: {check}")
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    reference = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    pinned = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.utils.cpp_extension import load

    assert torch.__version__ == "2.15.0+cu134" and torch.version.cuda == "13.4", torch.__version__
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "started": time.time(), "sass_selftest": check,
              "natives": {}}
    lines = [l.split() for l in (repo / TOPK_MANIFEST).read_text().splitlines() if l.strip()]
    expected = {"topk_overflow_prefix_v2_545e00d5_sm107.so": {s for s, n in lines if n == "topk_overflow_prefix_v2_545e00d5_sm107.so"},
                "hcb_prefill_trt_port_107.so": {HCB_EXPECTED_SHA}}
    for name, build in (
        ("topk_overflow_prefix_v2_545e00d5_sm107.so", build_topk_overflow),
        ("hcb_prefill_trt_port_107.so", build_hcb),
    ):
        rebuilt = build(repo, load)
        if reference is not None and (reference / name).exists():
            assert sha(reference / name) == REFERENCE_SHA[name], f"{reference / name} is not the reference binary"
        installed, source = rebuilt, "rebuild"
        if sha(rebuilt) not in expected[name]:
            for kind, d in (("pinned", pinned), ("reference", reference)):
                cand = d / name if d is not None else None
                if cand is not None and cand.exists() and sha(cand) in expected[name] and sass(cand) == sass(rebuilt):
                    installed, source = cand, kind
                    break
        shutil.copy(installed, out / name)
        entry = {"sha256": sha(out / name), "bytes": (out / name).stat().st_size, "installed_from": source,
                 "rebuilt_sha256": sha(rebuilt), "rebuilt_bytes": rebuilt.stat().st_size,
                 "expected_sha256": sorted(expected[name]), "reference_sha256": REFERENCE_SHA[name]}
        entry["same_as_expected"] = entry["sha256"] in expected[name]
        entry["same_as_reference"] = entry["sha256"] == REFERENCE_SHA[name]
        if reference is not None and (reference / name).exists():
            entry["same_sass_as_reference"] = sass(rebuilt) == sass(reference / name)
        report["natives"][name] = entry
    assert not torch.cuda.is_initialized()
    report["ended"] = time.time()
    (out / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    topk = report["natives"]["topk_overflow_prefix_v2_545e00d5_sm107.so"]
    if os.environ.get("REQUIRE_FROZEN_TOPK_SHA", "1") == "1" and not topk["same_as_expected"]:
        sys.exit(f"the overflow op is not listed in {TOPK_MANIFEST} and no listed binary has its SASS; "
                 "the loader would refuse it")


if __name__ == "__main__":
    main()
