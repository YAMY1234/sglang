"""Fetch a released DUET set and its base checkpoint from the Hugging Face Hub, verified, before serving or an exam job.

    python -m benchmark.duet_eval.fetch_duet --release Latent-Voyager/Kimi-Linear-48B-A3B-Instruct-SEED[@<commit>] --base \
        [--release-sha256 <hex>] [--hf-cache DIR] [--config-only]

1. The release (`--release <owner>/<repo>[@rev]`, or a local directory) is downloaded into
   <hf-cache>/duet-releases/<owner>--<repo>/<commit> (spec.json, manifest.json, duet_components.safetensors; regular files,
   pinned to the resolved commit) and verified: the component file's sha256 equals the Hub's LFS sha256 and the manifest's,
   spec.json equals the manifest's spec, and an optional `--release-sha256` pin (sglang.srt.duet.release fetch_release,
   verify_release, check_pin).
2. With `--base` (no value: the base the manifest names, at its training revision; or `--base <owner>/<repo>[@rev]`), the base
   checkpoint is fetched into the standard cache <hf-cache>/hub (models--<owner>--<repo>/snapshots/<commit>) only after its
   repository, revision and config.json sha256 match the manifest's `base` (release.fetch_base), then checked again on the
   snapshot (release.check_base).  `--config-only` stops after config.json (to check the identity without the download).
3. The last lines are shell assignments for the exam job (`exam_lmeval_job.sh`: DUET_REL and the family's base variable
   KIMI_BASE / FN_BASE / LIGHTNING_BASE / GLM_BASE) and for launch_duet.py.

Gated or private repositories need a login (`hf auth login`, or HF_TOKEN in the environment; huggingface_hub reads it, this
script never handles a token).  Needs huggingface_hub; no torch, no GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_VARIABLE = {"kimi": "KIMI_BASE", "flash_next": "FN_BASE", "lightning": "LIGHTNING_BASE", "glm": "GLM_BASE"}


def _launch_duet():
    key = "benchmark.duet_eval.launch_duet"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location("duet_launch_duet", HERE / "launch_duet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch(release_arg, *, base_arg=None, want_base=False, hf_cache=None, release_sha256=None, config_only=False):
    """-> dict(release=dir, identity=..., base=dir or None, base_hub=str or None, family=str or None)."""
    ld = _launch_duet()
    rel_mod = ld._load_fork_release_module()
    hf_cache = Path(hf_cache or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    release_dir, identity = ld.resolve_release(release_arg, hf_cache, verify=True)
    try:
        rel_mod.check_pin(identity["sha256"], release_sha256)
    except ValueError as exc:
        raise SystemExit(f"release pin: {exc}") from exc
    out = dict(release=release_dir, identity=identity, base=None, base_hub=None, family=None)
    if not (want_base or base_arg):
        return out
    if base_arg is None:
        manifest = json.loads((release_dir / "manifest.json").read_text())
        base = manifest.get("base") or {}
        if not base.get("hf_id"):
            raise SystemExit("the release manifest names no base (manifest.base.hf_id): pass --base <owner>/<repo>@<rev>")
        base_arg = base["hf_id"] + (f"@{base['revision']}" if base.get("revision") else "")   # no revision: resolve_base notes it
    if Path(base_arg).is_dir():
        base_dir = Path(base_arg).resolve()
    else:
        base_dir = ld.resolve_base(base_arg, hf_cache, release_dir, config_only=config_only)
        out["base_hub"] = base_arg
    ld.check_base_identity(release_dir, base_dir)
    out["base"] = base_dir
    out["family"] = ld.detect_family(base_dir)[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="<owner>/<repo>[@revision] of a DUET release, or a release directory")
    ap.add_argument("--release-sha256", default=None, help="optional pin of the component file's sha256 (12-64 hex digits)")
    ap.add_argument("--base", nargs="?", const="", default=None, help="also fetch the base: no value = the base the manifest names "
                    "at its training revision; or <owner>/<repo>[@revision] or a local directory")
    ap.add_argument("--hf-cache", type=Path, default=None, help="cache root (default $HF_HOME, else ~/.cache/huggingface)")
    ap.add_argument("--config-only", action="store_true", help="fetch only the base's config.json (identity check without the download)")
    args = ap.parse_args(argv)
    r = fetch(args.release, base_arg=args.base or None, want_base=args.base is not None, hf_cache=args.hf_cache,
              release_sha256=args.release_sha256, config_only=args.config_only)
    ident = r["identity"]
    print(f"RELEASE {ident['name']} [{ident['sha256'][:12]}] base {ident['base_model']} -> {r['release']}")
    exports = [f"DUET_REL={shlex.quote(str(r['release']))}"]
    if r["base"] is not None:
        print(f"BASE {r['base_hub'] or r['base']} -> {r['base']} (family {r['family']})")
        exports.append(f"{BASE_VARIABLE[r['family']]}={shlex.quote(str(r['base']))}")
    print("export " + " ".join(exports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
