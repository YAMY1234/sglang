"""One launcher for the DUET accuracy exams: detect the base model family, start the matching SGLang server for
either arm with IDENTICAL ServerArgs except the DUET enablement, record the command + environment, wait for /health, then
check the arm in server.log (check_arm: the DUET arm loaded the component file of this release (sha256) and serves the
expected prefill depth k / trim / state rank / cadence; the stock arm loaded no DUET) and stop the server if it fails.
Before launching, the base is checked against the release manifest's training base (manifest.base.config_sha256).

    python -m benchmark.duet_eval.launch_duet --model <base dir> --duet-release <dir|hf-repo[@rev]> --arm duet|stock \
        --port P [--context-length N] [--gpus 0,1] [--extra "..."] [--out DIR] [--detach] [--dry-run]
    python -m benchmark.duet_eval.launch_duet --model <GLM base dir> --arm stock --port P --gpus 0,1,2,3   # glm base arm: no release needed
    python -m benchmark.duet_eval.launch_duet --model <owner>/<base repo>[@rev] --duet-release <owner>/<release repo>[@rev] \
        [--duet-release-sha256 <hex>] --arm duet|stock --port P ...    # both from the Hub (README.md "Evaluate a released DUET set
                                                                         # from the Hub")

A Hub release is downloaded into <hf-cache>/duet-releases/<owner>--<repo>/<commit> and must match the Hub's LFS sha256 of its
component file and its manifest; a Hub base is fetched into the standard cache <hf-cache>/hub only after its repository, revision
and config.json sha256 match the release manifest's `base` (release.fetch_base).

--extra takes only the DUET switches --no-prefill-layer-trim, --decode-ssm-r N, --decode-ssm-w N and applies them to
the duet arm (ablations); any other server flag is refused, so the two arms differ only in the DUET enablement.

The server environment is the caller's minus every SGLANG_DUET_* / TWINSTAR_* / SGLANG_FLASHNEXT_* / SGLANG_QWEN4_*
variable and SGLANG_EXTERNAL_MODEL_PACKAGE (the launcher sets the ones a family needs itself; the dropped names are
printed and recorded); SGLANG_DUET_HOOK_PROFILE (state-cut hook timing, no numeric effect) passes through.

Family detection routes Kimi and Flash-Next through sglang.launch_server and the
canonical in-tree kimi_linear_duet / flash_next_duet adapters. There is no
external-package or legacy-launcher alternative. Lightning and GLM retain their
native server recipes (reference numerics). --dry-run records the canonical
adapter and full argv without starting a server.

Performance numbers come from the canonical native entry points. The benchmark
launcher redirect is qualified on CPU and has not been measured separately on GPU.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FORK_ROOT = Path(__file__).resolve().parents[2]
FORK_PYTHON = FORK_ROOT / "python"

FAMILIES = {
    # family: (architectures, model_types, default context length)
    "kimi": (("KimiLinearForCausalLM",), ("kimi_linear",), 36864),
    "lightning": (("NemotronHForCausalLM",), ("nemotron_h",), 40960),
    "flash_next": (("Qwen4ExpForConditionalGeneration",), ("qwen4_exp",), 40960),
    "glm": (("Glm5NextForConditionalGeneration",), ("glm5_next",), 262144),
}
# families whose stock arm reads no release (the launcher serves the base without --duet-release)
RELEASE_OPTIONAL_STOCK = ("glm",)
HEALTH_TIMEOUT = 1500.0


# ----------------------------------------------------------------------------- detection
def detect_family(model_dir):
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"base model has no config.json: {model_dir}")
    config = json.loads(config_path.read_text())
    archs = tuple(config.get("architectures") or ())
    model_type = config.get("model_type")
    for family, (fam_archs, fam_types, _) in FAMILIES.items():
        if any(a in fam_archs for a in archs) or model_type in fam_types:
            return family, {"architectures": list(archs), "model_type": model_type}
    raise SystemExit(f"unknown base family: architectures={list(archs)} model_type={model_type!r} ({config_path})")


# ----------------------------------------------------------------------------- release resolution
def _load_fork_release_module():
    """`sglang.srt.duet.release` from this fork, by path, without initialising `sglang` (its `_sibling` loader
    registers the light `sglang.srt.duet` package the same way)."""
    key = "sglang.srt.duet.release"
    if key in sys.modules:
        return sys.modules[key]
    package = "sglang.srt.duet"
    if package not in sys.modules:
        path = FORK_PYTHON / "sglang" / "srt" / "duet"
        spec = importlib.util.spec_from_file_location(package, path / "__init__.py", submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return importlib.import_module(key)


def resolve_release(value, hf_cache, *, verify=True):
    """<dir> -> itself; <owner>/<repo>[@rev] -> downloaded into <hf_cache>/duet-releases/<repo>/<commit> (pinned,
    regular files).  Returns (directory, identity dict); without `verify` the identity is spec.json and the manifest's
    declared sha256, read but not checked."""
    release = _load_fork_release_module()
    repo_or_path, revision = release.parse_release_arg(value)
    try:
        directory = Path(release.fetch_release(repo_or_path, hf_cache, revision)).resolve()
    except ValueError as exc:                       # the message names the repository and the cause
        raise SystemExit(str(exc)) from exc
    try:
        ident = release.verify_release(directory) if verify else None   # no geometry here: the server's adapter checks it
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"DUET release {directory}: {exc}") from exc
    if verify:
        return directory, {"name": ident.name, "sha256": ident.sha256, "base_model": ident.base_model, "spec": ident.spec,
                           "trained_spec": ident.trained_spec, "decode_differs_from_trained": ident.decode_differs_from_trained,
                           "verified": True}
    spec = json.loads((directory / "spec.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    return directory, {"name": manifest.get("name") or spec.get("name", ""), "sha256": manifest.get("sha256", ""),
                       "base_model": manifest.get("base_model", ""), "spec": spec, "verified": False}


def resolve_base(value, hf_cache, release_dir, *, config_only=False):
    """A Hub id `<owner>/<repo>[@rev]` for --model -> the snapshot directory in <hf_cache>/hub, checked against the release's
    manifest `base` before the download (release.fetch_base); config_only for a dry run."""
    if release_dir is not None and "@" not in str(value):
        base = json.loads((Path(release_dir) / "manifest.json").read_text()).get("base") or {}
        if isinstance(base, dict) and not base.get("revision"):
            print(f"NOTE the release manifest records no base revision: {value} is fetched at its default branch; the "
                  "config.json sha256 check still applies (pin the training revision with <repo>@<commit>)",
                  file=sys.stderr, flush=True)
    try:
        return Path(_load_fork_release_module().fetch_base(value, hf_cache, release_dir=release_dir, config_only=config_only)).resolve()
    except ValueError as exc:
        raise SystemExit(f"base from the Hub: {exc}") from exc


def check_base_identity(release_dir, model_dir):
    """Refuse a base other than the release's training base (manifest.base.config_sha256); warn when the manifest
    predates that field."""
    try:
        warning = _load_fork_release_module().check_base(release_dir, model_dir)
    except ValueError as exc:
        raise SystemExit(f"base identity: {exc}") from exc
    if warning:
        print(f"WARNING base identity: {warning}", file=sys.stderr, flush=True)
    return warning


# ----------------------------------------------------------------------------- environment
DROPPED_ENV_PREFIXES = ("SGLANG_DUET_", "TWINSTAR_", "SGLANG_FLASHNEXT_", "SGLANG_QWEN4_")
DROPPED_ENV_NAMES = ("SGLANG_EXTERNAL_MODEL_PACKAGE",)
HONOURED_ENV = ("SGLANG_DUET_HOOK_PROFILE",)   # wall time of the state-cut hook; host-side timing only


def server_environment(environ):
    """(the caller's environment minus the DUET-family variables the launcher does not set itself, the dropped names)."""
    dropped = sorted(k for k in environ
                     if (k.startswith(DROPPED_ENV_PREFIXES) or k in DROPPED_ENV_NAMES) and k not in HONOURED_ENV)
    return {k: v for k, v in environ.items() if k not in dropped}, dropped


# ----------------------------------------------------------------------------- --extra
DUET_EXTRA_FLAGS = ("--no-prefill-layer-trim", "--decode-ssm-r", "--decode-ssm-w")


def parse_extra(extra):
    """--extra -> {prefill_layer_trim: False, decode_ssm_r: R, decode_ssm_w: W} (the keys given); anything but the three
    DUET switches is refused."""
    toks, out, i = shlex.split(extra), {}, 0
    while i < len(toks):
        token = toks[i]
        flag, eq, inline = token.partition("=")
        if token == "--no-prefill-layer-trim":
            out["prefill_layer_trim"] = False
            i += 1
            continue
        if flag in ("--decode-ssm-r", "--decode-ssm-w"):
            value, step = (inline, 1) if eq else ((toks[i + 1] if i + 1 < len(toks) else ""), 2)
            if value.isdigit():
                out[flag[2:].replace("-", "_")] = int(value)
                i += step
                continue
        raise SystemExit(f"--extra accepts only {', '.join(DUET_EXTRA_FLAGS)} (the last two with an integer >= 0); "
                         f"refusing {token!r} in {extra!r}")
    return out


def extra_cli(overrides):
    """The server CLI flags of parsed --extra overrides (Kimi launcher and sglang.launch_server spell them alike)."""
    cmd = ["--no-prefill-layer-trim"] if overrides.get("prefill_layer_trim") is False else []
    for key in ("decode_ssm_r", "decode_ssm_w"):
        if key in overrides:
            cmd += ["--" + key.replace("_", "-"), str(overrides[key])]
    return cmd


# ----------------------------------------------------------------------------- per-family commands
CANONICAL_PACKAGES = {
    "kimi": "sglang.srt.models.kimi_linear_duet",
    "flash_next": "sglang.srt.models.flash_next_duet",
}


def canonical_command(args, release, env, *, family):
    """Both arms use the native server; --duet-release selects the in-tree adapter.

    This launcher redirect has CPU qualification only. Performance measurements
    use the independently frozen native launch contracts, not these defaults.
    """
    mrr = args.max_running_requests
    tp = 1 if family == "kimi" else (args.tp or 2)
    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model-path", str(args.model), "--trust-remote-code",
           "--host", args.host, "--port", str(args.port), "--tp-size", str(tp),
           "--context-length", str(args.context_length),
           "--max-running-requests", str(mrr),
           "--max-mamba-cache-size", str(max(mrr + 8, 24)),
           "--cuda-graph-max-bs-decode", str(mrr),
           "--disable-radix-cache", "--disable-prefill-cuda-graph",
           "--disable-overlap-schedule", "--skip-server-warmup",
           "--enable-custom-logit-processor", "--random-seed", "62",
           "--linear-attn-backend", "triton", "--duet-numerics", "production",
           "--duet-code-precision", "fp32"]
    if family == "kimi":
        cmd += ["--dtype", "bfloat16", "--attention-backend", "triton",
                "--chunked-prefill-size", "-1", "--max-prefill-tokens", "32768",
                "--mem-fraction-static", str(args.mem_fraction if args.mem_fraction is not None else .7)]
    else:
        cmd += ["--page-size", "64", "--chunked-prefill-size", "8192",
                "--max-prefill-tokens", "32768", "--mem-fraction-static", ".8",
                "--moe-runner-backend", "flashinfer_cutlass"]
        for key in ("NVFP4_OVERLAY_MAP", "DUET_BASE_SRT_ROOT"):
            env.pop(key, None)
        gpus = [g for g in (env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if g]
        if len(gpus) < tp:
            raise SystemExit(f"flash_next with --tp {tp}: pass --gpus with {tp} devices")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:tp])
    if not args.decode_cuda_graph:
        cmd += ["--cuda-graph-backend-decode", "disabled"]
    if args.sampling_backend:
        cmd += ["--sampling-backend", args.sampling_backend]
    if args.watchdog_timeout is not None:
        cmd += ["--watchdog-timeout", str(args.watchdog_timeout)]
    env.update(SGLANG_MAMBA_CONV_DTYPE="float32", SGLANG_MAMBA_SSM_DTYPE="float32")
    if args.arm == "duet":
        spec = json.loads((Path(release) / "spec.json").read_text())
        controls = dict(decode_ssm_r=spec["state_rank"], decode_ssm_w=spec["state_every"])
        controls.update(args.duet_overrides)
        cmd += ["--duet-release", str(release), "--prefill-saving-policy", "kv-and-ssm"]
        cmd += ["--prefill-layer-trim"] if controls.get("prefill_layer_trim", True) else []
        cmd += extra_cli(controls)
    return cmd


def kimi_command(args, release, env):
    return canonical_command(args, release, env, family="kimi")


def lightning_command(args, release, env):
    """Both arms on SGLang's native NemotronH path with the same flags: decode through the CUDA graph, the whole prompt in one
    extend (no chunking, no radix reuse, no overlap scheduling -- the DUET cut runs between steps), the fp32 Mamba state pool,
    one Mamba slot per running request.  The DUET arm adds the release and the reference numerics profile (the production
    profile is not validated on this adapter, models/lightning_duet/engine.py)."""
    mrr = args.max_running_requests
    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model-path", str(args.model), "--served-model-name", "lightning", "--random-seed", "20260929",
           "--host", args.host, "--port", str(args.port), "--tp-size", "1", "--watchdog-timeout", "1800",
           "--dtype", "bfloat16", "--trust-remote-code", "--context-length", str(args.context_length),
           "--max-running-requests", str(mrr), "--cuda-graph-max-bs-decode", str(mrr), "--max-mamba-cache-size", str(max(mrr + 8, 24)),
           "--mem-fraction-static", "0.7", "--disable-overlap-schedule", "--disable-radix-cache",
           "--chunked-prefill-size", "-1", "--skip-server-warmup",
           "--enable-custom-logit-processor"]          # the forced-NLL checks (forced_logits.ForcedRawLogits); no effect on sampling
    if args.sampling_backend:
        cmd += ["--sampling-backend", args.sampling_backend]
    env.update(SGLANG_MAMBA_CONV_DTYPE="float32", SGLANG_MAMBA_SSM_DTYPE="float32")
    if args.arm == "duet":
        env.update(SGLANG_DUET_DIR=str(release), SGLANG_EXTERNAL_MODEL_PACKAGE="sglang.srt.models.lightning_duet")
        cmd += ["--duet-release", str(release), "--duet-numerics", "reference"] + extra_cli(args.duet_overrides)
    return cmd


# GLM-5.3-Flash on GB300 (twinstar docs/74 section 74.2: the flags the bring-up probes settled on), both arms:
# * the bf16 MLA cache (SGLang's Blackwell default, an FP8 cache with unit scales, would quantize every cache entry);
# * DSA prefill and decode on the trtllm kernels (the vendor's Blackwell recipe);
# * SGLang's default FP8 arithmetic on SM10x, the one the twinstar reference mirrors (docs/73 section 73.2): routed experts on the
#   FlashInfer TRT-LLM kernels with the checkpoint's fp32 block scales, dense FP8 layers on DeepGEMM (`--fp8-gemm-backend` auto),
#   whose weights SGLang requantizes to power-of-two (UE8M0) block scales at load.  `--fp8-gemm-backend flashinfer_trtllm` would
#   keep the dense layers' fp32 scales at the same speed (probe C2): a change of the served model, to be made together with the
#   reference.  SGLANG_JIT_DEEPGEMM_PRECOMPILE=0: the DeepGEMM kernels compile on first use instead of a start-up warm-up over every
#   M (measured: ~20 min per weight shape); no numeric effect;
# * the KDA conv and recurrent states in fp32 (as for Kimi-Linear), radix cache off.
GLM_SERVER_FLAGS = ("--kv-cache-dtype", "bfloat16", "--dsa-prefill-backend", "trtllm", "--dsa-decode-backend", "trtllm",
                    "--moe-runner-backend", "flashinfer_trtllm")
GLM_ENV = dict(SGLANG_MAMBA_CONV_DTYPE="float32", SGLANG_MAMBA_SSM_DTYPE="float32", SGLANG_JIT_DEEPGEMM_PRECOMPILE="0")
GLM_MEM_FRACTION = 0.85
GLM_TP = 4


def glm_duet_cli(release, overrides):
    """The DUET arm's server flags and environment for GLM (twinstar docs/79): the release (the adapter package
    sglang.srt.models.glm_duet replaces Glm5NextForConditionalGeneration through sglang.srt.duet.adapters) and the reference numerics
    profile (the only one the adapter implements), plus the --extra ablation switches."""
    if release is None:
        raise SystemExit("glm --arm duet needs --duet-release")
    flags = ["--duet-release", str(release), "--duet-numerics", "reference"] + extra_cli(overrides)
    return flags, {"SGLANG_DUET_DIR": str(release)}


def glm_command(args, release, env):
    """Both arms on SGLang's native GLM-5.3 path (models/glm5_next.py) with the same flags: TP 4 (the 310 GB FP8 release is 78 GB
    per GPU), decode through the CUDA graph, chunked prefill at SGLang's default, no radix reuse.  The DUET arm adds
    glm_duet_cli's flags (phase F)."""
    mrr = args.max_running_requests
    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model-path", str(args.model), "--served-model-name", "glm", "--random-seed", "20261007",
           "--host", args.host, "--port", str(args.port), "--tp-size", str(args.tp or GLM_TP), "--watchdog-timeout",
           str(int(args.watchdog_timeout or 1800)), "--trust-remote-code", "--context-length", str(args.context_length),
           "--max-running-requests", str(mrr), "--cuda-graph-max-bs-decode", str(mrr), "--max-mamba-cache-size", str(max(mrr + 8, 24)),
           "--mem-fraction-static", str(args.mem_fraction or GLM_MEM_FRACTION), "--disable-radix-cache",
           *GLM_SERVER_FLAGS, *(("--disable-overlap-schedule",) if args.disable_overlap_schedule else ()),
           "--enable-custom-logit-processor"]          # the forced-NLL checks (forced_logits.ForcedRawLogits); no effect on sampling
    if args.sampling_backend:
        cmd += ["--sampling-backend", args.sampling_backend]
    env.update(GLM_ENV)
    if args.arm == "duet":
        flags, extra_env = glm_duet_cli(release, args.duet_overrides)
        cmd += flags
        env.update(extra_env)
    return cmd


def flash_next_command(args, release, env, out):
    """Canonical in-tree Flash-Next; no external package or config-view pre-step."""
    return canonical_command(args, release, env, family="flash_next"), []


# ----------------------------------------------------------------------------- arm check
# Lines each family's DUET model writes to server.log while it loads the release (kimi_duet/twinstar_kimi_linear.py,
# flash_next/twinstar_sgl/twinstar_qwen4_exp.py, lightning_duet/engine.py).  Without them the DUET arm serves the base model.
DUET_MARKERS = {
    "kimi": ("TwinStar Kimi Linear: P layers", "HF DUET loaded: "),
    "lightning": ("Lightning DUET loaded: ",),
    "flash_next": ("TwinStar Qwen4Exp: P layers", "DUET controls: "),
    "glm": ("GLM DUET loaded: ",),          # models/glm_duet/model.py load_weights
}
# Every family's load path logs the cut it serves and the sha256 of the component file it verified.
SERVED_CUT_MARKER = "DUET served cut: "
SERVED_CUT = r"DUET served cut: k=(\d+) trim=(True|False) r=(\d+) W=(\d+) sha256=([0-9a-f]{64})"
# The original twinstar_sgl tree (--twinstar-root) predates that line: k from its layer line, trim / r / W from its controls
# line, no sha256.
LEGACY_FN_DEPTH = r"TwinStar Qwen4Exp: P layers 0\.\.(\d+),"
LEGACY_FN_CUT = r"DUET controls: trim=(True|False) \S+ r=(\d+) W=(\d+)"


def expected_cut(spec, overrides):
    """(prefill depth k, prefill trim, state rank, cadence) the DUET arm must serve: the release spec, unless --extra
    overrides trim / r / W."""
    return (spec["prefill_depth"], overrides.get("prefill_layer_trim", True),
            overrides.get("decode_ssm_r", spec["state_rank"]), overrides.get("decode_ssm_w", spec["state_every"]))


def served_cuts(text, *, legacy_fn):
    """server.log -> ({(k, trim, r, W) served}, {component sha256 served} or None when the tree does not log it)."""
    import re
    if legacy_fn:
        depths = {int(m) + 1 for m in re.findall(LEGACY_FN_DEPTH, text)}
        cuts = {(t == "True", int(r), int(w)) for t, r, w in re.findall(LEGACY_FN_CUT, text)}
        return {(k,) + c for k in depths for c in cuts}, None
    rows = re.findall(SERVED_CUT, text)
    return {(int(k), t == "True", int(r), int(w)) for k, t, r, w, _ in rows}, {sha for *_, sha in rows}


def check_arm(family, arm, text, identity, overrides, *, legacy_fn=False):
    """server.log of this launch -> (ok, message).  The DUET arm must show every load marker of its family and serve
    the expected k / trim / r / W from the release's component file (sha256); the stock arm must show no marker."""
    if arm == "stock":
        stray = [m for ms in DUET_MARKERS.values() for m in ms + (SERVED_CUT_MARKER,) if m in text]
        return (False, f"the stock arm loaded DUET ({sorted(set(stray))})") if stray else (True, "stock arm, no DUET marker")
    missing = [m for m in DUET_MARKERS[family] if m not in text]
    if missing:
        return False, f"the DUET arm did not load the release (no {missing} in server.log): the base model would answer"
    served, shas = served_cuts(text, legacy_fn=legacy_fn)
    want = expected_cut(identity["spec"], overrides)
    if served != {want}:
        return False, f"the DUET arm serves k / trim / r / W {sorted(served)}, expected {want}"
    if shas is not None and shas != {identity["sha256"]}:
        return False, f"the DUET arm loaded components {sorted(shas)}, not the release's {identity['sha256']}"
    unchecked = " (sha256 not logged by the original tree)" if shas is None else ""
    return True, (f"DUET arm loaded {identity['name']} [{identity['sha256'][:12]}]{unchecked}, "
                  f"k {want[0]} trim {want[1]} r {want[2]} W {want[3]}")


def check_canonical_arm(family, arm, text, identity, overrides, info, release):
    """Check native runtime args/adapter and verified release path, plus load evidence.

    /get_server_info describes the registered adapter; it is combined with model
    load markers and /health, not treated alone as proof of loaded weights.
    Canonical Flash does not log a component hash: both verified loaders check
    the same release path, recorded separately from a server-emitted hash.
    """
    report = info.get("duet") or {}
    if arm == "stock":
        ok, why = check_arm(family, arm, text, identity, overrides)
        return (ok and not report.get("enabled"), why if ok else why)
    want = expected_cut(identity["spec"], overrides)
    if (report.get("adapter") != CANONICAL_PACKAGES[family]
            or not report.get("enabled") or not report.get("adapter_in_tree")
            or report.get("error") or Path(report.get("release", "")).resolve() != Path(release).resolve()):
        return False, f"canonical adapter/release mismatch: {report}"
    actual = (info.get("prefill_layer_trim"), info.get("decode_ssm_r"), info.get("decode_ssm_w"))
    if actual != want[1:]:
        return False, f"native trim/r/W {actual}, expected {want[1:]}"
    import re
    if family == "kimi":
        shas = set(re.findall(r"HF DUET loaded: .*sha256=([0-9a-f]{64})", text))
        if shas != {identity["sha256"]}:
            return False, f"Kimi loaded hashes {shas}, expected {identity['sha256']}"
        depth_pattern = r"TwinStar Kimi Linear: P layers 0\.\.(\d+),"
    else:
        if "DUET controls: " not in text:
            return False, "canonical Flash-Next load/control marker absent"
        depth_pattern = LEGACY_FN_DEPTH
    if want[1]:
        depths = {int(k) + 1 for k in re.findall(depth_pattern, text)}
        if depths != {want[0]}:
            return False, f"native prefill depth {depths}, expected {want[0]}"
    return True, f"canonical {report['adapter']}; verified release {identity['sha256']}; k/trim/r/W {want}"


# ----------------------------------------------------------------------------- process control
def wait_health(url, proc, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server exited early with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True, "ok"
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(2)
    return False, f"/health not ready within {timeout:.0f} s"


def run_steps(steps, env, log):
    for cmd in steps:
        print("STEP", shlex.join(cmd), flush=True)
        rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(f"preparation step failed (rc={rc}); see {log.name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="unchanged base checkpoint directory (config.json read for the family), "
                    "or a Hub id <owner>/<repo>[@revision] fetched into <hf-cache>/hub after the release manifest's base check")
    ap.add_argument("--duet-release", default=None, help="DUET release: directory, or Hub id with optional @revision (required "
                    "except for --arm stock of a family in RELEASE_OPTIONAL_STOCK: glm)")
    ap.add_argument("--duet-release-sha256", default=None, help="optional pin: the component file's sha256 (or a prefix of at "
                    "least 12 hex digits) the release must have")
    ap.add_argument("--arm", required=True, choices=("duet", "stock"))
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--context-length", type=int, default=None, help="per family: kimi 36864, lightning 40960, flash_next 40960, "
                    "glm 262144")
    ap.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES for the server, e.g. 0,1 (default: inherited)")
    ap.add_argument("--extra", default="", help="duet arm only: --no-prefill-layer-trim, --decode-ssm-r N, --decode-ssm-w N "
                    "(ablations); any other flag is refused")
    ap.add_argument("--out", type=Path, default=None, help="output directory (launch.json, server.log, pid); default ./duet_eval_out/<family>-<arm>-<port>")
    ap.add_argument("--hf-cache", type=Path, default=None, help="HF cache root for Hub release downloads and the Kimi launcher "
                    "(default $HF_HOME, else ~/.cache/huggingface)")
    ap.add_argument("--max-running-requests", type=int, default=64, help="concurrent requests (flash_next: the colleague's guard used 16)")
    ap.add_argument("--decode-cuda-graph", type=lambda v: str(v).lower() not in ("0", "false", "no", "off"), default=True,
                    help="flash_next: replay decode through the CUDA graph (default on; 0 = the colleague's eager guard setting)")
    ap.add_argument("--tp", type=int, default=None, help="flash_next tensor-parallel size (default 2; the NVFP4 base fits TP 1 on a 277 GiB GPU, "
                    "the BF16 base needs 2); glm (default 4); kimi and lightning serve TP 1 only")
    ap.add_argument("--sampling-backend", choices=("flashinfer", "pytorch"), default=None,
                    help="server sampler for both arms; pytorch supports per-request sampling seeds together with top-k/top-p")
    ap.add_argument("--mem-fraction", type=float, default=None, help="kimi: static memory share (default the launcher's 0.7); the rest "
                    "is headroom for the unchunked prefill, which a 512K-1M-token prompt needs more of; glm: default GLM_MEM_FRACTION")
    ap.add_argument("--watchdog-timeout", type=float, default=None, help="kimi: scheduler watchdog seconds (default SGLang's 300); "
                    "glm: default 1800")
    ap.add_argument("--disable-overlap-schedule", action="store_true", help="glm, both arms: no overlap scheduling (the "
                    "teacher-forced parity checks: forced_logits reads the request history; numerics-neutral, slower)")
    ap.add_argument("--no-verify", action="store_true", help="skip spec/manifest/sha256 verification of the release in this process "
                    "(the server's loader still verifies; the arm check then compares with the unverified spec / manifest)")
    ap.add_argument("--health-timeout", type=float, default=HEALTH_TIMEOUT)
    ap.add_argument("--detach", action="store_true", help="print PID after READY and exit, leaving the server running")
    ap.add_argument("--dry-run", action="store_true", help="resolve, write launch.json, print the command; start nothing")
    args = ap.parse_args(argv)

    hub_base = None                         # --model <owner>/<repo>[@rev]: resolve the release first, then the base it names
    early_release = None
    if _load_fork_release_module().is_hub_id(args.model):
        if args.hf_cache is None:
            args.hf_cache = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
        if args.duet_release is not None:
            early_release = resolve_release(args.duet_release, args.hf_cache, verify=not args.no_verify)
        hub_base = str(args.model)
        args.model = resolve_base(hub_base, args.hf_cache, early_release[0] if early_release else None, config_only=args.dry_run)
        print(f"BASE {hub_base} -> {args.model}" + (" (config.json only: dry run)" if args.dry_run else ""), flush=True)
    family, detected = detect_family(args.model)
    if family in ("kimi", "lightning") and args.tp not in (None, 1):
        raise SystemExit(f"--tp {args.tp}: {family} serves at TP 1 (its server command sets the tensor-parallel size)")
    if family not in ("kimi", "glm") and (args.mem_fraction is not None or args.watchdog_timeout is not None):
        raise SystemExit("--mem-fraction / --watchdog-timeout are kimi and glm launcher options")
    if family != "glm" and args.disable_overlap_schedule:
        raise SystemExit("--disable-overlap-schedule is a glm launcher option (the other families' launchers set it themselves)")
    if args.duet_release is None and not (args.arm == "stock" and family in RELEASE_OPTIONAL_STOCK):
        raise SystemExit(f"--duet-release is required for {family} --arm {args.arm} (only the stock arm of "
                         f"{', '.join(RELEASE_OPTIONAL_STOCK)} runs without a release)")
    if args.context_length is None:
        args.context_length = FAMILIES[family][2]
    if args.hf_cache is None:
        args.hf_cache = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    args.duet_overrides = parse_extra(args.extra)
    if args.extra and args.arm == "stock":
        print(f"NOTE --extra {args.extra!r} applies to the duet arm only; the stock arm ignores it", flush=True)
    out = args.out or Path("duet_eval_out") / f"{family}-{args.arm}-{args.port}"
    out.mkdir(parents=True, exist_ok=True)

    if args.no_verify:
        print("WARNING --no-verify: this launcher does NOT verify the release (spec validation, manifest identity, component "
              "sha256); the arm check compares the served k / trim / r / W and sha256 with the unverified spec.json and "
              "manifest sha256 (the server's own loader still hashes the component file)", file=sys.stderr, flush=True)
    if args.duet_release is not None:
        release_dir, identity = early_release or resolve_release(args.duet_release, args.hf_cache, verify=not args.no_verify)
        if args.duet_release_sha256 is not None:
            try:
                _load_fork_release_module().check_pin(identity["sha256"], args.duet_release_sha256)
            except ValueError as exc:
                raise SystemExit(f"release pin: {exc}") from exc
        base_warning = check_base_identity(release_dir, args.model)
    else:                                   # a release-free stock arm (RELEASE_OPTIONAL_STOCK)
        release_dir, identity, base_warning = None, None, None

    env, dropped_env = server_environment(os.environ)
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTHONPATH"] = os.pathsep.join([str(FORK_PYTHON), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    pre = []
    if family == "kimi":
        cmd = kimi_command(args, release_dir, env)
    elif family == "lightning":
        cmd = lightning_command(args, release_dir, env)
    elif family == "glm":
        cmd = glm_command(args, release_dir, env)
    else:
        cmd, pre = flash_next_command(args, release_dir, env, out)

    url = f"http://{args.host}:{args.port}"
    tracked = sorted(k for k in env if k.startswith(("SGLANG_", "TWINSTAR_", "NVIDIA_", "CUDA_VISIBLE", "PYTHONPATH")))
    record = {"family": family, "detected": detected, "arm": args.arm, "model": str(args.model),
              "canonical_adapter": CANONICAL_PACKAGES.get(family),
              "release": str(release_dir) if release_dir is not None else None,
              "release_identity": identity, "base_identity_warning": base_warning, "context_length": args.context_length,
              "url": url, "duet_overrides": args.duet_overrides if args.arm == "duet" else {},
              "preparation": [shlex.join(c) for c in pre], "command": shlex.join(cmd), "argv": cmd,
              "env": {k: env[k] for k in tracked}, "env_dropped": dropped_env, "cwd": os.getcwd(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "dry_run": args.dry_run, "pid": None}
    if hub_base is not None:
        record["model_hub"] = hub_base
    if args.duet_release_sha256 is not None:
        record["release_sha256_pin"] = args.duet_release_sha256
    (out / "launch.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"FAMILY {family} {detected}", flush=True)
    print("ENV", " ".join(f"{k}={shlex.quote(env[k])}" for k in tracked if not k.startswith("PYTHONPATH")), flush=True)
    if dropped_env:
        print("ENV DROPPED (inherited, not set by this launcher):", " ".join(dropped_env), flush=True)
    print("CMD", shlex.join(cmd), flush=True)
    if args.dry_run:
        for c in pre:
            print("STEP", shlex.join(c))
        print(f"DRY-RUN wrote {out / 'launch.json'}")
        return 0

    log = (out / "server.log").open("ab")
    log_start = (out / "server.log").stat().st_size      # a reused --out keeps earlier launches' lines; check only this one's
    run_steps(pre, env, log)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=args.detach)
    record["pid"] = proc.pid
    (out / "launch.json").write_text(json.dumps(record, indent=1) + "\n")
    (out / "server.pid").write_text(f"{proc.pid}\n")
    print(f"PID {proc.pid} log {out / 'server.log'}", flush=True)

    ok, why = wait_health(url, proc, args.health_timeout)
    if not ok:
        print(f"FAILED {why}; see {out / 'server.log'}", file=sys.stderr, flush=True)
        if proc.poll() is None:
            proc.terminate()
        return 1
    log.flush()
    with (out / "server.log").open("rb") as fh:
        fh.seek(log_start)
        text = fh.read().decode("utf-8", "replace")
    if family in CANONICAL_PACKAGES:
        try:
            with urllib.request.urlopen(url + "/get_server_info", timeout=30) as response:
                info = json.load(response)
            (out / "server-info.json").write_text(json.dumps(info, indent=1) + "\n")
            ok, why = check_canonical_arm(family, args.arm, text, identity, args.duet_overrides, info, release_dir)
        except (OSError, ValueError) as exc:
            ok, why = False, f"canonical runtime identity unavailable: {exc}"
    else:
        ok, why = check_arm(family, args.arm, text, identity, args.duet_overrides)
    if not ok:
        print(f"FAILED arm check: {why}; see {out / 'server.log'}", file=sys.stderr, flush=True)
        proc.terminate()
        return 1
    print(f"ARM CHECK {why}", flush=True)
    print(f"READY {url}", flush=True)
    if args.detach:
        print(f"DETACHED pid={proc.pid}")
        return 0

    def forward(signum, _frame):
        if proc.poll() is None:
            proc.send_signal(signum)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, forward)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
