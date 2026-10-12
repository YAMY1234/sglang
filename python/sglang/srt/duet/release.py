"""A DUET component release: fetch, identity, spec, tensor contract, base identity, fp32 load -- model independent.

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
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

COMPONENT_FILE = "duet_components.safetensors"
RELEASE_FILES = ("spec.json", "manifest.json", COMPONENT_FILE)
LEGACY_DIR_ALIASES = ("TWINSTAR_KIMI_DUET_DIR", "TWINSTAR_LIGHTNING_DUET_DIR")


def _sibling(name):
    """A sibling module of this package, whether this file was imported as sglang.srt.duet.release inside the
    runtime or loaded stand-alone on a CPU box (one convention with
    models/lightning_duet/_common.load and models/glm_duet/_common.load): when `sglang.srt` is not initialised,
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
    """Delegate CLI > canonical DIR > legacy aliases and their warning to options."""
    from types import SimpleNamespace
    options = _sibling("options")
    return options.resolve_release(
        SimpleNamespace(duet_release=cli), environ, legacy_directory=LEGACY_DIR_ALIASES
    )


def open_release(value, *, hf_root=None, geometry=None, base_model=None, model=None, model_info=None, snapshot_download=None):
    """The one-call entry: `<dir>` or `<repo>[@rev]` -> fetch (pinned, no symlinks) -> verify
    (spec, manifest identity, derived tensor contract when the adapter's geometry is given) -> ReleaseIdentity."""
    repo_or_path, revision = parse_release_arg(value)
    if repo_or_path is None:
        raise ValueError("open_release needs a release directory or Hub id")
    directory = fetch_release(repo_or_path, hf_root, revision, model_info=model_info, snapshot_download=snapshot_download)
    return verify_release(directory, geometry=geometry, base_model=base_model, model=model)


def release_from_args(args=None, *, environ=None, legacy_directory=None, **open_kw):
    """`ServerArgs` / launcher args -> ReleaseIdentity, or None when DUET is off (no --duet-release, no
    SGLANG_DUET_DIR, no legacy alias).  A one-call entry for launchers and tests; the in-tree adapters resolve the value with
    options.resolve_release and verify the directory with verify_release themselves."""
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
    regular files (no symlinks), pinned to the resolved commit for every file.  When the Hub reports the LFS sha256 of the
    component file (model_info with file metadata), the downloaded file must have it (verify_release then compares it with
    manifest.json).  The two Hub calls are injectable for tests."""
    if Path(repo_or_path).is_dir():
        return Path(repo_or_path)
    if not hf_root:
        raise ValueError("an explicit HF cache root is required to download a DUET release")
    if model_info is None or snapshot_download is None:
        from huggingface_hub import HfApi, snapshot_download as _snapshot_download
        model_info = model_info or (lambda repo, rev: HfApi().model_info(repo, revision=rev, files_metadata=True))
        snapshot_download = snapshot_download or _snapshot_download
    info = _hub_call(model_info, repo_or_path, revision, what="DUET release")
    sha = info.sha
    directory = Path(hf_root) / "duet-releases" / repo_or_path.replace("/", "--") / sha
    snapshot_download(repo_id=repo_or_path, revision=sha, local_dir=str(directory),
                      cache_dir=str(Path(hf_root) / "hub"), allow_patterns=list(RELEASE_FILES))
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("DUET release download contains a symlink")
    hub_sha = hub_lfs_sha256(info, COMPONENT_FILE)
    if hub_sha is None:
        print(f"WARNING the Hub reports no LFS sha256 for {COMPONENT_FILE} of {repo_or_path}@{sha}: only the manifest's "
              "sha256 is checked", file=sys.stderr, flush=True)
    else:
        local = sha256_of(directory / COMPONENT_FILE)
        if local != hub_sha:
            raise ValueError(f"downloaded {COMPONENT_FILE} sha256 {local} differs from the Hub's LFS sha256 {hub_sha} "
                             f"({repo_or_path}@{sha})")
    return directory


_HUB_ACCESS_ERRORS = ("GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError")


def _hub_call(model_info, repo, revision, *, what):
    """model_info(repo, revision); an access error is re-raised as ValueError naming the likely cause (huggingface_hub
    reports a private repository without a valid token as RepositoryNotFoundError)."""
    try:
        return model_info(repo, revision)
    except Exception as exc:                                    # noqa: BLE001 -- re-raised with the likely cause named
        if type(exc).__name__ in _HUB_ACCESS_ERRORS:
            raise ValueError(f"{what} {repo}@{revision or 'main'}: {type(exc).__name__} -- a private or gated repository "
                             "needs access on the Hub and a login (hf auth login, or HF_TOKEN in the environment)") from exc
        raise


def hub_lfs_sha256(info, filename):
    """The sha256 the Hub reports for an LFS file of a model_info result (files_metadata=True: siblings[].lfs.sha256;
    a dict or an object, by huggingface_hub version), or None when the result carries no such record."""
    for sibling in getattr(info, "siblings", None) or ():
        if getattr(sibling, "rfilename", None) != filename:
            continue
        lfs = getattr(sibling, "lfs", None)
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        return sha if sha and re.fullmatch(r"[0-9a-f]{64}", str(sha)) else None
    return None


def check_pin(identity_sha256, pin):
    """An optional user pin of the component file's sha256 (64 hex, or a prefix of at least 12) against the release's."""
    if pin is None:
        return
    pin = str(pin).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{12,64}", pin):
        raise ValueError(f"sha256 pin must be 12-64 hex digits, got {pin!r}")
    if not str(identity_sha256).startswith(pin):
        raise ValueError(f"release component sha256 {identity_sha256} does not match the pinned {pin}")


# ----------------------------------------------------------------------------- base checkpoint from the Hub
def is_hub_id(value):
    """True for `<owner>/<repo>[@revision]` that is not an existing local path."""
    text = str(value)
    if Path(text).exists():
        return False
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+(@[A-Za-z0-9._/-]+)?", text) is not None


def fetch_base(value, hf_root=None, *, release_dir=None, config_only=False, model_info=None, hf_hub_download=None,
               snapshot_download=None):
    """`<owner>/<repo>[@revision]` of an unchanged base checkpoint -> its snapshot directory in the Hugging Face cache
    (`hf_root/hub`, the standard layout `models--<owner>--<repo>/snapshots/<commit>`; the default cache when hf_root is None).

    With a release directory, its manifest's `base` decides before anything large is downloaded: the repository must be
    `base.hf_id`; without `@revision` the training revision `base.revision` is fetched, and a given revision must resolve to
    it; `config.json` is fetched first and must hash to `base.config_sha256`.  check_base runs on the snapshot afterwards.
    config_only fetches config.json alone (a dry run).  Gated repositories need a login (`hf auth login` or HF_TOKEN in the
    environment, which huggingface_hub reads itself).  The three Hub calls are injectable for tests."""
    repo, _, rev = str(value).partition("@")
    rev = rev or None
    base = None
    if release_dir is not None:
        manifest = json.loads((Path(release_dir) / "manifest.json").read_text())
        base = manifest.get("base") if isinstance(manifest.get("base"), dict) else None
        if base and base.get("hf_id") and repo != base["hf_id"]:
            raise ValueError(f"the release was trained on {base['hf_id']}"
                             + (f"@{base['revision']}" if base.get("revision") else "") + f"; --model names {repo}")
    if model_info is None or hf_hub_download is None or snapshot_download is None:
        from huggingface_hub import HfApi, hf_hub_download as _hf_hub_download, snapshot_download as _snapshot_download
        model_info = model_info or (lambda r, v: HfApi().model_info(r, revision=v))
        hf_hub_download = hf_hub_download or _hf_hub_download
        snapshot_download = snapshot_download or _snapshot_download
    want = rev or (base or {}).get("revision")
    commit = _hub_call(model_info, repo, want, what="base").sha
    trained = (base or {}).get("revision")
    if trained and re.fullmatch(r"[0-9a-f]{40}", trained) and commit != trained:
        raise ValueError(f"base {repo}@{rev} resolves to {commit}; the release was trained on {repo}@{trained}")
    cache_dir = str(Path(hf_root) / "hub") if hf_root else None
    config = Path(hf_hub_download(repo_id=repo, filename="config.json", revision=commit, cache_dir=cache_dir))
    if base and base.get("config_sha256"):
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        if digest != base["config_sha256"]:
            raise ValueError(f"base {repo}@{commit}: config.json sha256 {digest} differs from the training base's "
                             f"{base['config_sha256']}; the components serve only on the base they were trained on")
    snapshot = snapshot_download(repo_id=repo, revision=commit, cache_dir=cache_dir,
                                 allow_patterns=["config.json"] if config_only else None)
    return Path(snapshot)


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
    """Derive the released tensor set from the spec and the adapter's geometry."""
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


# ----------------------------------------------------------------------------- base identity
def check_base(release_dir, base_dir):
    """The served base checkpoint against the training base the release's manifest.json names:

        manifest["base"] = {"hf_id": str, "revision": str | None, "config_sha256": str}

    config_sha256 is the sha256 of the training base's config.json bytes.  Raises ValueError on a mismatch.  Returns
    a warning text when nothing can be checked (a manifest exported before the field, or a base that is not a local
    directory), else None.  A 40-hex revision is also compared when the base is a Hub cache snapshot directory."""
    manifest = json.loads((Path(release_dir) / "manifest.json").read_text())
    base = manifest.get("base")
    if base is None:
        return (f"DUET release manifest has no base.config_sha256 (exported before the field existed): the base "
                f"checkpoint is NOT identity-checked against the training base {manifest.get('base_model')!r}")
    if not isinstance(base, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(base.get("config_sha256", ""))):
        raise ValueError(f"manifest.base must be {{hf_id, revision, config_sha256 (64 hex)}}, got {base!r}")
    hf_id = base.get("hf_id")
    if hf_id and manifest.get("base_model") and hf_id != manifest["base_model"]:
        raise ValueError(f"manifest.base.hf_id {hf_id!r} differs from manifest.base_model {manifest['base_model']!r}")
    config = Path(base_dir) / "config.json"
    if not config.is_file():
        return f"base {base_dir} is not a local checkpoint directory: it is NOT identity-checked against {hf_id!r}"
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    if digest != base["config_sha256"]:
        raise ValueError(f"base {base_dir}: config.json sha256 {digest} differs from the training base {hf_id!r} "
                         f"(revision {base.get('revision')}, config sha256 {base['config_sha256']}); the components "
                         "serve only on the base they were trained on")
    revision = base.get("revision")
    snapshot = Path(base_dir).resolve()
    if (revision and re.fullmatch(r"[0-9a-f]{40}", revision) and snapshot.parent.name == "snapshots"
            and re.fullmatch(r"[0-9a-f]{40}", snapshot.name) and snapshot.name != revision):
        raise ValueError(f"base {base_dir} is Hub snapshot {snapshot.name}; the components were trained on "
                         f"{hf_id}@{revision}")
    return None


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
