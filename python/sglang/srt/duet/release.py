"""A DUET component release: fetch, identity, spec, tensor contract, fp32 load (docs/162 §3.1) -- model independent.

There is no release-name or SHA whitelist anywhere: the identity (name, sha256, base model, trained spec) is
recorded, never matched against constants.  Everything the components do is read from spec.json; the tensor set and
shapes are DERIVED from the spec plus the base model's geometry (supplied by the model adapter) and then checked
against manifest.json and the safetensors header, tensor by tensor.  Moved / generalised from the Kimi line
(twinstar_sgl/kimi_duet_checkpoint.py: read_header, audit_metadata, verify_release, resolve_release) and the NVFP4
line (twinstar/fullstack_hf_release.py: header offset checks); reference: release/hf_duet.py L60-77, ckpt.py L98-111.

Adapter geometry contract (duck typed; see `build_contract`):
    residual_dim: int                      # hidden size coded by the latent (Flash-Next: hc * hidden)
    num_layers: int
    state_heads: int; state_side_dim: int  # sink_dir is (num_layers, state_heads, state_side_dim)
    latent_key: str = "latent.code"        # Flash-Next: "P.latent.core"
    state_key: str = "state.sink_dir"      # Flash-Next: "P.state.sink_dir"
    emitter_prefix: str = "emitters."      # Flash-Next: "P.emitters."
    memory_kind(layer) -> "state" | "attention" | None
    emitter_tensors(layer) -> {relative_name: shape}     # the layer's exported writing half
    inherited_tensors(layer) -> {base_name: (target_name, shape)}   # unexported halves the emitter still READS
                                                                     # (Kimi q path); filled from the base, never zeros
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

COMPONENT_FILE = "duet_components.safetensors"
RELEASE_FILES = ("spec.json", "manifest.json", COMPONENT_FILE)
LEGACY_DIR_ALIASES = ("TWINSTAR_KIMI_DUET_DIR", "TWINSTAR_LIGHTNING_DUET_DIR")


def _sibling(name):
    """A sibling module of this package, whether this file was imported as sglang.srt.duet.release inside the
    runtime or loaded stand-alone on a CPU box (docs/162 F5 loader rule, one convention with
    models/lightning_duet/_common.load and twinstar_sgl/_duet_common.load): when `sglang.srt` is not initialised,
    register the light `sglang.srt.duet` package by path so relative imports inside it work, then import normally."""
    import importlib
    import importlib.util
    import sys
    key = f"sglang.srt.duet.{name}"
    if key in sys.modules:
        return sys.modules[key]
    if "sglang.srt" not in sys.modules:
        package = "sglang.srt.duet"
        if package not in sys.modules:
            path = Path(__file__).resolve().parent
            spec = importlib.util.spec_from_file_location(package, path / "__init__.py", submodule_search_locations=[str(path)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            spec.loader.exec_module(module)
    return importlib.import_module(key)


# ----------------------------------------------------------------------------- the master switch
def add_release_argument(parser):
    """`--duet-release <dir | hf-repo[@revision]>` for a stand-alone launcher (the fork's ServerArgs declares
    `duet_release` in arg_groups/fields/exec_.py)."""
    parser.add_argument("--duet-release", default=None,
                        help="DUET component release: a verified directory or a Hub id with optional @revision; "
                             "absent = DUET off, the stock model class is served (SGLANG_DUET_DIR)")


def resolve_release_dir(cli=None, *, environ=None):
    """Thin wrapper over options.resolve_release (the canonical CLI > SGLANG_DUET_DIR > legacy-alias order) that
    also walks the one-version legacy aliases with a deprecation warning.  None = DUET off."""
    from types import SimpleNamespace
    options = _sibling("options")
    env = os.environ if environ is None else environ
    value = options.resolve_release(SimpleNamespace(duet_release=cli), env)
    if value:
        return value
    for alias in LEGACY_DIR_ALIASES:
        value = options.resolve_release(None, env, legacy_directory=alias)
        if value:
            logging.getLogger(__name__).warning(
                "%s is deprecated and retained for one version; use SGLANG_DUET_DIR (CLI and SGLANG_DUET_DIR take precedence).",
                alias)
            return value
    return None


def open_release(value, *, hf_root=None, geometry=None, base_model=None, model=None, model_info=None, snapshot_download=None):
    """The one-call entry (docs/162 §3.1): `<dir>` or `<repo>[@rev]` -> fetch (pinned, no symlinks) -> verify
    (spec, manifest identity, derived tensor contract when the adapter's geometry is given) -> ReleaseIdentity."""
    repo_or_path, revision = parse_release_arg(value)
    if repo_or_path is None:
        raise ValueError("open_release needs a release directory or Hub id")
    directory = fetch_release(repo_or_path, hf_root, revision, model_info=model_info, snapshot_download=snapshot_download)
    return verify_release(directory, geometry=geometry, base_model=base_model, model=model)


def release_from_args(args=None, *, environ=None, legacy_directory=None, **open_kw):
    """`ServerArgs` / launcher args -> ReleaseIdentity, or None when DUET is off (no --duet-release, no
    SGLANG_DUET_DIR, no legacy alias).  Adapters call this once at model construction."""
    options = _sibling("options")
    value = options.resolve_release(args, environ, legacy_directory=legacy_directory)
    if not value:
        return None
    return open_release(value, **open_kw)


def parse_release_arg(value):
    """'<dir>' -> (dir, None); '<owner>/<repo>[@rev]' -> (repo, rev or None)."""
    if value is None:
        return None, None
    if Path(value).is_dir():
        return value, None
    repo, _, rev = value.partition("@")
    return repo, (rev or None)


def fetch_release(repo_or_path, hf_root=None, revision=None, *, model_info=None, snapshot_download=None):
    """A local directory is used as is.  A Hub id is downloaded into `hf_root/duet-releases/<repo>/<commit>` with
    regular files (no symlinks), pinned to the resolved commit for every file.  The two Hub calls are injectable
    for tests."""
    if Path(repo_or_path).is_dir():
        return Path(repo_or_path)
    if not hf_root:
        raise ValueError("an explicit HF cache root is required to download a DUET release")
    if model_info is None or snapshot_download is None:
        from huggingface_hub import HfApi, snapshot_download as _snapshot_download
        model_info = model_info or (lambda repo, rev: HfApi().model_info(repo, revision=rev))
        snapshot_download = snapshot_download or _snapshot_download
    sha = model_info(repo_or_path, revision).sha
    directory = Path(hf_root) / "duet-releases" / repo_or_path.replace("/", "--") / sha
    snapshot_download(repo_id=repo_or_path, revision=sha, local_dir=str(directory),
                      cache_dir=str(Path(hf_root) / "hub"), allow_patterns=list(RELEASE_FILES))
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("DUET release download contains a symlink")
    return directory


# ----------------------------------------------------------------------------- safetensors header
def read_header(path):
    """(header dict, payload offset) of a safetensors file without loading tensors."""
    with Path(path).open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= 16 * 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError("truncated safetensors metadata")
        return json.loads(raw), size + 8


def sha256_of(path, chunk=16 << 20):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            sha.update(block)
    return sha.hexdigest()


# ----------------------------------------------------------------------------- tensor contract
@dataclass
class TensorContract:
    """source name -> (target name inside the serving model, exact unsharded shape); plus the tensors an emitter
    still reads that the release does not carry (filled from the base model)."""
    expected: dict = field(default_factory=dict)      # name -> (target, shape)
    inherited: dict = field(default_factory=dict)     # base name -> (target, shape)

    @property
    def parameter_count(self):
        return sum(math.prod(shape) for _, shape in self.expected.values())


def build_contract(spec, geometry):
    """Derive the released tensor set from the spec and the adapter's geometry (docs/162 §3.1 step 5)."""
    validate_spec = _sibling("spec").validate_spec
    validate_spec(spec)
    k, n = spec["prefill_depth"], geometry.num_layers
    if not 1 <= k <= n:
        raise ValueError("DUET cut must be inside the base layer stack")
    latent_key = getattr(geometry, "latent_key", "latent.code")
    state_key = getattr(geometry, "state_key", "state.sink_dir")
    emitter_prefix = getattr(geometry, "emitter_prefix", "emitters.")
    contract = TensorContract()
    contract.expected[state_key] = (state_key, (n, geometry.state_heads, geometry.state_side_dim))
    r, d = spec["latent_rank"], geometry.residual_dim
    if r:
        contract.expected[f"{latent_key}.E"] = (f"{latent_key}.E", (1, r, d))
        contract.expected[f"{latent_key}.D"] = (f"{latent_key}.D", (1, d, r))
        contract.expected[f"{latent_key}.mu"] = (f"{latent_key}.mu", (1, d))
    if spec["latent_spikes"] > d:
        raise ValueError("exact coordinate count exceeds the residual dimension")
    for layer in range(k, n):
        if geometry.memory_kind(layer) is None:
            continue
        for rel, shape in geometry.emitter_tensors(layer).items():
            name = f"{emitter_prefix}{layer}.{rel}"
            contract.expected[name] = (name, tuple(shape))
        inherit = getattr(geometry, "inherited_tensors", None)
        if inherit is not None:
            for base_name, (target, shape) in inherit(layer).items():
                contract.inherited[base_name] = (target, tuple(shape))
    return contract


def audit_metadata(contract, manifest, header):
    """The header and the manifest must carry exactly the contract's tensors, all F32, with the declared shapes,
    contiguous byte ranges and the manifest's parameter total (Kimi line audit_metadata, generalised)."""
    expected = contract.expected
    header = {k: v for k, v in header.items() if k != "__metadata__"}
    if set(header) != set(expected) or set(manifest["tensors"]) != set(expected):
        raise ValueError(f"DUET tensor names: missing={sorted(set(expected) - set(header))}, "
                         f"extra={sorted(set(header) - set(expected))}, "
                         f"manifest_extra={sorted(set(manifest['tensors']) - set(expected))}")
    rows, ranges = [], []
    for name, (target, shape) in expected.items():
        item, declared = header[name], manifest["tensors"][name]
        if tuple(item["shape"]) != shape or tuple(declared["shape"]) != shape:
            raise ValueError(f"shape mismatch: {name}: expected {shape}, header {item['shape']}, manifest {declared['shape']}")
        if item["dtype"] != "F32":
            raise ValueError(f"release tensor must retain fp32: {name}")
        start, end = item["data_offsets"]
        size = 4 * math.prod(shape)
        if start < 0 or end - start != size or declared["bytes"] != size:
            raise ValueError(f"invalid byte count: {name}")
        ranges.append((start, end))
        rows.append({"source": name, "target": target, "shape": list(shape), "dtype": "F32", "bytes": size})
    last = 0
    for start, end in sorted(ranges):
        if start != last:
            raise ValueError("non-contiguous or overlapping tensor ranges")
        last = end
    if contract.parameter_count != manifest["total_params"]:
        raise ValueError("manifest parameter count mismatch")
    return {"tensor_count": len(rows), "parameter_count": contract.parameter_count, "payload_bytes": last,
            "mapping": rows,
            "inherited_base_tensors": [{"source": k, "target": v[0], "shape": list(v[1])} for k, v in contract.inherited.items()]}


# ----------------------------------------------------------------------------- identity
@dataclass
class ReleaseIdentity:
    path: str
    name: str
    sha256: str
    base_model: str
    spec: dict
    trained_spec: dict
    decode_differs_from_trained: bool
    file_bytes: int
    audit: dict

    def describe(self):
        describe = _sibling("spec").describe
        d = self.trained_spec
        trained = f"trained r/W {d.get('state_rank')}/{d.get('state_every')} -> " if self.decode_differs_from_trained else ""
        return f"{self.name} [{self.sha256[:12]}] base {self.base_model}: {trained}{describe(self.spec)}"


def verify_release(directory, *, geometry=None, base_model=None, model=None):
    """spec.json is the algorithm (validated; unknown keys raise); manifest.json is the identity; the component file
    must match the manifest's sha256 / size and -- when the adapter's geometry is given -- the derived tensor
    contract.  Returns a ReleaseIdentity; nothing is matched against a release-name whitelist."""
    validate_spec = _sibling("spec").validate_spec
    root = Path(directory)
    for f in RELEASE_FILES:
        if not (root / f).is_file():
            raise FileNotFoundError(f"DUET release lacks {f}: {root}")
    spec = json.loads((root / "spec.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    validate_spec(spec, model=model)
    provenance = manifest.get("provenance", {})
    if spec != provenance.get("spec"):
        raise ValueError("spec.json differs from manifest.provenance.spec (edited directory)")
    if base_model is not None and manifest.get("base_model") != base_model:
        raise ValueError(f"components were trained for {manifest.get('base_model')!r}, this base is {base_model!r}")
    weights = root / COMPONENT_FILE
    header, offset = read_header(weights)
    audit = {}
    if geometry is not None:
        audit = audit_metadata(build_contract(spec, geometry), manifest, header)
        if weights.stat().st_size != offset + audit["payload_bytes"]:
            raise ValueError("component file size differs from header + payload")
    size = weights.stat().st_size
    if size != manifest.get("file_bytes"):
        raise ValueError("component file size differs from manifest")
    digest = sha256_of(weights)
    if digest != manifest.get("sha256"):
        raise ValueError("component SHA256 differs from manifest")
    trained = provenance.get("trained_spec") or spec
    return ReleaseIdentity(str(root), manifest.get("name") or spec.get("name", ""), digest, manifest.get("base_model", ""),
                           spec, trained, bool(provenance.get("decode_spec_differs_from_trained",
                                                                 trained != spec)), size, audit)


# ----------------------------------------------------------------------------- fp32 load + base fill
def load_components(directory, contract, *, device="cpu"):
    """Every released tensor as fp32 on `device`, keyed by its TARGET name (the serving model's parameter path)."""
    import torch
    from safetensors import safe_open
    out = {}
    with safe_open(str(Path(directory) / COMPONENT_FILE), framework="pt", device=str(device)) as f:
        keys = set(f.keys())
        missing = set(contract.expected) - keys
        if missing:
            raise ValueError(f"release file lacks used tensors: {sorted(missing)[:5]}")
        for name, (target, shape) in contract.expected.items():
            t = f.get_tensor(name)
            if t.dtype != torch.float32 or tuple(t.shape) != shape:
                raise ValueError(f"{name}: expected fp32 {shape}, got {t.dtype} {tuple(t.shape)}")
            out[target] = t
    return out


def inherit_from_base(contract, lookup, *, dtype=None):
    """The emitter halves the release does not export but the emitter still reads (e.g. Kimi's q path) come from
    the base model's own layer -- never zeros.  `lookup(base_name) -> tensor`."""
    import torch
    out = {}
    for base_name, (target, shape) in contract.inherited.items():
        t = lookup(base_name)
        if t is None:
            raise ValueError(f"base model lacks {base_name}, required by an emitter")
        if tuple(t.shape) != shape:
            raise ValueError(f"{base_name}: expected {shape}, got {tuple(t.shape)}")
        out[target] = t.to(dtype or torch.float32)
    return out
