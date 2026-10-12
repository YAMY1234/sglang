"""spec.model -> model adapter for `--duet-release <dir | owner/repo[@rev]>`.

The release is fetched (pinned, no symlinks) and its `spec.json` names the model family; this table maps the family
to the package that serves it and to the base architecture whose registry entry that package replaces.  Nothing here
loads weights or verifies the component file -- the adapter's own loader does -- so the selection is cheap enough to
run in every process that imports the model registry.

A requested release is never answered by the base model: a family whose package is not importable, a family that
needs its own launcher, or a release that fails to resolve stops the server at model registration.
"""
import importlib
import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .release import fetch_release, is_hub_id, parse_release_arg
from .spec import MODELS, validate_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Adapter:
    model: str              # spec.model
    package: str            # package providing EntryClass for the architecture
    architecture: str       # base architecture name the adapter replaces in the registry
    production_profile: bool  # the production numerics profile has been validated on this adapter
    production_notes: str = ""  # what the adapter implements of the production profile (reported in /server_info)
    launcher: str = ""      # non-empty: --duet-release alone cannot serve this family; the command that can


ADAPTERS = {
    "kimi-linear": Adapter("kimi-linear", "sglang.srt.models.kimi_linear_duet", "KimiLinearForCausalLM", True,
                           "P2 (INFORK-K): tf32 code / graphs / batching per docs/167 brief; emitter bf16 pending emitter_runner"),
    "lightning": Adapter("lightning", "sglang.srt.models.lightning_duet", "NemotronHForCausalLM", False,
                         "tf32 code + async H2D implemented; emitter state-only already; CUDA graphs / batched decode "
                         "unsupported-by-adapter (per-slot python decode, docs/167 §4.1 P4b); eager required"),
    "glm5-next": Adapter("glm5-next", "sglang.srt.models.glm_duet", "Glm5NextForConditionalGeneration", False,
                         "reference numerics only (fp32 code, layer-arithmetic emitters); TP 4; decode is CUDA-graph capable"),
    # Retain the qualified in-tree Flash-Next performance adapter and direct Hub route.
    "flash-next": Adapter("flash-next", "sglang.srt.models.flash_next_duet", "Qwen4ExpForConditionalGeneration", True,
                          "docs/167 §8.F: NLL 4.189x / no bias, needle 6/6, four-arm smoke; "
                          "C1/C16 production/reference 25.97x/19.57x, production/stock 0.981x/1.002x"),
}
assert set(ADAPTERS) == set(MODELS) | {"glm5-next"}, (set(ADAPTERS), set(MODELS))

ENV_RELEASE = "SGLANG_DUET_DIR"


def release_value(args=None, environ=None):
    """CLI `--duet-release` beats the canonical environment variable; None = DUET off."""
    env = os.environ if environ is None else environ
    cli = getattr(args, "duet_release", None) if args is not None else None
    return cli or env.get(ENV_RELEASE) or None


def default_hf_root(environ=None):
    """HF_HOME, else the Hub client's standard cache root (~/.cache/huggingface): a Hub id must work with no extra
    setup, and registration runs before any adapter could supply a root."""
    env = os.environ if environ is None else environ
    if env.get("HF_HOME"):
        return env["HF_HOME"]
    if env.get("HF_HUB_CACHE"):
        return str(Path(env["HF_HUB_CACHE"]).parent)
    return str(Path(env.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "huggingface")


def resolve_directory(value, *, hf_root=None, environ=None, **fetch_kw):
    """`<dir>` or `<owner>/<repo>[@rev]` -> local release directory (downloads a Hub id into `hf_root`)."""
    env = os.environ if environ is None else environ
    repo_or_path, revision = parse_release_arg(value)
    hf_root = hf_root or default_hf_root(env)
    return Path(fetch_release(repo_or_path, hf_root, revision, **fetch_kw))


def localize_release(args, environ=None, **fetch_kw):
    """A Hub id given as `--duet-release` (or SGLANG_DUET_DIR) is fetched once, in the launching process, and
    `args.duet_release` is set to the local directory before it is exported to the workers: the model classes read the
    release as a directory.  Anything else (a directory, a path, no release) is left exactly as given.  Returns the directory
    or None."""
    value = release_value(args, environ)
    if not value or not is_hub_id(value):
        return None
    directory = resolve_directory(value, environ=environ, **fetch_kw)
    args.duet_release = str(directory)
    logger.info("DUET release %s fetched to %s", value, directory)
    return directory


def read_spec(directory):
    """The validated spec of a release directory (unknown keys raise; nothing else is checked here)."""
    spec = json.loads((Path(directory) / "spec.json").read_text())
    return validate_spec(spec)


def select(value, *, hf_root=None, environ=None, **fetch_kw):
    """-> (directory, spec, Adapter) for a release argument."""
    directory = resolve_directory(value, hf_root=hf_root, environ=environ, **fetch_kw)
    spec = read_spec(directory)
    return directory, spec, ADAPTERS[spec["model"]]


def package_available(adapter):
    """True when the adapter package can be found.  On a bare CPU box the light `sglang.srt.duet` package may be
    registered without its parents (test loader convention); find_spec then raises KeyError -- that is "not
    available here", never an error."""
    try:
        return importlib.util.find_spec(adapter.package) is not None
    except Exception:  # noqa: BLE001
        return False


class UnservedRelease(RuntimeError):
    """A DUET release was requested but nothing in this process can serve it; the model registry re-raises it."""


def register(registry, value, *, hf_root=None, environ=None, **fetch_kw):
    """Register the package named by the release's spec over its base architecture; raise UnservedRelease when
    this process cannot serve the release (family launcher required, or package not importable)."""
    directory, spec, adapter = select(value, hf_root=hf_root, environ=environ, **fetch_kw)
    if adapter.launcher:
        raise UnservedRelease(f"DUET release {directory} is for {spec['model']!r}, which --duet-release alone cannot "
                              f"serve; start it with: {adapter.launcher}")
    if not package_available(adapter):
        raise UnservedRelease(f"DUET release {directory} is for {spec['model']!r}; its adapter package "
                              f"{adapter.package} is not importable in this process")
    registry.register(adapter.package, overwrite=True)
    logger.info("DUET adapter %s registered for %s from release %s", adapter.package, adapter.architecture, directory)
    return adapter


def register_from_environment(registry, environ=None):
    """Called by the model registry at import: the launcher has already exported --duet-release to the
    environment (server_args.export_cli_environment), so the value is read from there in every process."""
    env = os.environ if environ is None else environ
    value = env.get(ENV_RELEASE)
    if not value:
        return None
    try:
        return register(registry, value, environ=env)
    except UnservedRelease:
        raise
    except Exception as exc:  # noqa: BLE001 -- a broken release must not leave the base model serving in its place
        raise UnservedRelease(f"DUET release {value} could not be registered ({type(exc).__name__}: {exc}); "
                              "refusing to serve the base model") from exc


def adapter_extension(adapter, name):
    """Optional per-adapter extension point `<package>.config.<name>` (e.g. resolve_server_numerics,
    describe_numerics).  None when the package or the function is absent; generic code never imports an adapter
    module by its literal name."""
    if not package_available(adapter):
        return None
    try:
        module = importlib.import_module(adapter.package + ".config")
    except Exception:  # noqa: BLE001 -- absent or half-importable adapter package: no extension
        return None
    return getattr(module, name, None)


def _base_architecture(model_path):
    """architectures[0] of a local base checkpoint's config.json; None for a Hub id or an unreadable config (not checked)."""
    if not model_path:
        return None
    path = os.path.join(str(model_path), "config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            archs = json.load(fh).get("architectures") or []
    except (OSError, ValueError):
        return None
    return archs[0] if archs else None


def run_resolution_hook(server_args, environ=None):
    """Called from the ServerArgs resolution pipeline (arg_groups/pipeline.py) with the live record: when a
    release is set, the matching adapter may declare resolution-time defaults through
    `<package>.config.resolve_server_numerics(server_args, spec)`. Kimi and Flash-Next retain their
    performance-stack resolution hooks; adapters without a hook use the launcher's explicit flags.
    Release errors propagate so a wrong family fails startup."""
    localize_release(server_args, environ)
    value = release_value(server_args, environ)
    if not value:
        return None
    directory, spec, adapter = select(value, environ=environ)
    base_arch = _base_architecture(getattr(server_args, "model_path", None))
    if base_arch is not None and base_arch != adapter.architecture:
        # a release of another family would register its adapter over an architecture this base does not use, and the base
        # model would answer every request (audit 2026-10-06)
        raise UnservedRelease(f"DUET release {directory} is for {spec['model']!r} ({adapter.architecture}); the base model "
                              f"{server_args.model_path} is {base_arch}")
    hook = adapter_extension(adapter, "resolve_server_numerics")
    if hook is None:
        return None
    hook(server_args, spec)
    return adapter


def describe(args=None, environ=None):
    """Cheap description for /server_info: resolved directory, spec summary, adapter and whether it is in-tree."""
    value = release_value(args, environ)
    if not value:
        return {"release": None, "enabled": False}
    try:
        directory, spec, adapter = select(value, environ=environ)
    except Exception as exc:  # noqa: BLE001 -- the endpoint reports, it does not fail the server
        return {"release": value, "enabled": True, "error": f"{type(exc).__name__}: {exc}"}
    from .spec import describe as describe_spec
    info = {"release": str(directory), "enabled": True, "model": spec["model"], "spec": describe_spec(spec),
            "name": spec.get("name", ""), "adapter": adapter.package, "architecture": adapter.architecture,
            "adapter_in_tree": package_available(adapter), "production_validated": adapter.production_profile,
            "production_notes": adapter.production_notes}
    extra = adapter_extension(adapter, "describe_numerics")
    if extra is not None:
        try:
            info["adapter_numerics"] = extra(args, spec)
        except Exception as exc:  # noqa: BLE001 -- reporting only
            info["adapter_numerics"] = {"error": f"{type(exc).__name__}: {exc}"}
    return info
