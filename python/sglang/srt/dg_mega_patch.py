"""Serve DeepGEMM's MegaMoE from patched headers without a new DeepGEMM wheel.

DeepGEMM JIT-compiles its kernels from ``<package>/include`` (``_C.init`` takes the
package directory at import time). With ``SGLANG_K31_DG_MEGA_PATCH=<set>`` this module
builds a node-local mirror of the installed ``deep_gemm`` package: symlinks to every
installed file except the headers in ``dg_mega_patch_data/<set>/include``, which are
real copies, and puts the mirror first on ``sys.path`` before anything imports
``deep_gemm``. The installed files the set replaces must match ``_BASES`` (the base
release the set was made against); otherwise nothing changes. Compiled kernels go to
their own JIT cache subdirectory, so no kernel built from the stock headers is reused.
"""

import fcntl
import hashlib
import importlib.util
import os
import sys

# set -> {header path under include/ -> sha256 of the installed file it replaces}
_BASES = {
    # DeepGEMM PRs #157 (scheduler: L1 tasks first) + #156 (barrier-free dispatch) on sgl-deep-gemm 0.2.8
    "p157_156": {
        "deep_gemm/scheduler/mega_moe.cuh": "ff588434e7371cae4ad1fec0a3750af3103aca22d8d24658d7db3f2ddde54c6a",
        "deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh": "6e98221db612fbcb88d4747d0ac6238fe17af25490e68e68c7047fc553b54a70",
        "deep_gemm/impls/sm100_bf16_mega_moe.cuh": "8d9d86c29ff2f018ebffd38bb1ecdf67156027c95d08df479d45fbada5c5320b",
    },
}


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _say(msg: str) -> None:
    print(f"[sglang dg_mega_patch pid {os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _build_mirror(installed: str, patch_inc: str, files, mirror_pkg: str) -> None:
    os.makedirs(mirror_pkg, exist_ok=True)
    for name in os.listdir(installed):
        if name in ("include", "__pycache__"):
            continue
        dst = os.path.join(mirror_pkg, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(installed, name), dst)
    src_inc = os.path.join(installed, "include")
    for root, _, names in os.walk(src_inc):
        rel = os.path.relpath(root, src_inc)
        os.makedirs(os.path.join(mirror_pkg, "include", rel), exist_ok=True)
        for n in names:
            rel_file = os.path.normpath(os.path.join(rel, n))
            dst = os.path.join(mirror_pkg, "include", rel_file)
            if os.path.lexists(dst):
                continue
            if rel_file in files:
                with open(os.path.join(patch_inc, rel_file), "rb") as fi, open(dst, "wb") as fo:
                    fo.write(fi.read())
            else:
                os.symlink(os.path.join(root, n), dst)


def apply() -> None:
    from sglang.srt.environ import envs

    tag = envs.SGLANG_K31_DG_MEGA_PATCH.get()
    if not tag:
        return
    if "deep_gemm" in sys.modules:
        _say(f"set {tag} NOT engaged: deep_gemm was imported before sglang")
        return
    if tag not in _BASES:
        _say(f"set {tag} NOT engaged: unknown set (known: {sorted(_BASES)})")
        return
    spec = importlib.util.find_spec("deep_gemm")
    if spec is None or not spec.origin:
        _say(f"set {tag} NOT engaged: deep_gemm is not installed")
        return
    installed = os.path.dirname(os.path.realpath(spec.origin))
    files = _BASES[tag]
    patch_inc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dg_mega_patch_data", tag, "include")
    bad = [f for f, want in files.items() if _sha256(os.path.join(installed, "include", f)) != want]
    if bad:
        _say(f"set {tag} NOT engaged: installed headers differ from the set's base: {bad}")
        return
    key = hashlib.sha256(
        (installed + "".join(_sha256(os.path.join(patch_inc, f)) for f in sorted(files))).encode()
    ).hexdigest()[:12]
    root = os.path.join(envs.SGLANG_DG_MEGA_PATCH_ROOT.get(), f"{tag}-{key}")
    mirror_pkg = os.path.join(root, "deep_gemm")
    done = os.path.join(root, "DONE")
    if not os.path.exists(done):
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, ".lock"), "w") as lk:  # one builder per node; the others wait here
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not os.path.exists(done):
                _build_mirror(installed, patch_inc, files, mirror_pkg)
                open(done, "w").close()
    if root in sys.path:
        return
    sys.path.insert(0, root)
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != root]
    os.environ["PYTHONPATH"] = os.pathsep.join([root] + paths)
    # kernels compiled from these headers get their own cache (DG_JIT_CACHE_DIR follows SGLANG_DG_CACHE_DIR)
    cache = envs.SGLANG_DG_CACHE_DIR.get().rstrip("/")
    if not cache.endswith(f"dgpatch-{tag}-{key}"):
        os.environ["SGLANG_DG_CACHE_DIR"] = os.path.join(cache, f"dgpatch-{tag}-{key}")
    _say(f"set {tag} engaged: deep_gemm from {mirror_pkg} (base {installed}), patched "
         + ", ".join(f"{f}={_sha256(os.path.join(mirror_pkg, 'include', f))[:12]}" for f in sorted(files)))
