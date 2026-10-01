"""spec.model -> model adapter (docs/167 §3 B, DUET-INFORK P1).

One `--duet-release <dir | owner/repo[@rev]>` drives all three models: the release is fetched (pinned, no
symlinks) and its `spec.json` names the model family; this table maps the family to the adapter package inside
the fork and to the base architecture whose registry entry the adapter replaces.  Nothing here loads weights
or verifies the component file -- that stays in the adapter's own loader (`release.open_release`) -- so the
selection is cheap enough to run in every process that imports the model registry.

Transition rule (P1 -> P3): an adapter package that is not yet in this tree is reported, not invented; the
legacy `SGLANG_EXTERNAL_MODEL_PACKAGE=twinstar_sgl` registration keeps working until P2 / P3 move the Kimi and
Flash-Next adapters in (docs/167 §2.2).
"""
import importlib
import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .release import fetch_release, parse_release_arg
from .spec import MODELS, validate_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Adapter:
    model: str              # spec.model
    package: str            # fork package providing EntryClass for the architecture
    architecture: str       # base architecture name the adapter replaces in the registry
    production_profile: bool  # docs/167 §4: production numerics validated on this adapter


ADAPTERS = {
    "kimi-linear": Adapter("kimi-linear", "sglang.srt.models.kimi_linear_duet", "KimiLinearForCausalLM", False),
    "lightning": Adapter("lightning", "sglang.srt.models.lightning_duet", "NemotronHForCausalLM", False),
    "flash-next": Adapter("flash-next", "sglang.srt.models.flash_next_duet", "Qwen4ExpForConditionalGeneration", True),
}
assert set(ADAPTERS) == set(MODELS), (set(ADAPTERS), set(MODELS))

ENV_RELEASE = "SGLANG_DUET_DIR"


def release_value(args=None, environ=None):
    """CLI `--duet-release` beats the canonical environment variable; None = DUET off."""
    env = os.environ if environ is None else environ
    cli = getattr(args, "duet_release", None) if args is not None else None
    return cli or env.get(ENV_RELEASE) or None


def resolve_directory(value, *, hf_root=None, environ=None, **fetch_kw):
    """`<dir>` or `<owner>/<repo>[@rev]` -> local release directory (downloads a Hub id into `hf_root`)."""
    env = os.environ if environ is None else environ
    repo_or_path, revision = parse_release_arg(value)
    hf_root = hf_root or env.get("HF_HOME")
    return Path(fetch_release(repo_or_path, hf_root, revision, **fetch_kw))


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
    try:
        return importlib.util.find_spec(adapter.package) is not None
    except (ImportError, ValueError):
        return False


def register(registry, value, *, hf_root=None, environ=None, strict=False, **fetch_kw):
    """Register the adapter named by the release's spec over its base architecture.

    Returns the Adapter when the fork package was registered, None when the package is not in this tree
    (the legacy external package path then applies; a warning says so).  `strict=True` raises instead.
    """
    directory, spec, adapter = select(value, hf_root=hf_root, environ=environ, **fetch_kw)
    if not package_available(adapter):
        message = (f"DUET release {directory} is for {spec['model']!r}; adapter package {adapter.package} is not in "
                   f"this tree yet (docs/167 P2/P3) -- serving needs SGLANG_EXTERNAL_MODEL_PACKAGE=twinstar_sgl")
        if strict:
            raise ImportError(message)
        logger.warning(message)
        return None
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
    except Exception as exc:  # noqa: BLE001 -- registration must never take the registry down silently
        logger.warning("DUET adapter registration skipped for %s: %s", value, exc)
        return None


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
    return {"release": str(directory), "enabled": True, "model": spec["model"], "spec": describe_spec(spec),
            "name": spec.get("name", ""), "adapter": adapter.package, "architecture": adapter.architecture,
            "adapter_in_tree": package_available(adapter), "production_validated": adapter.production_profile}
