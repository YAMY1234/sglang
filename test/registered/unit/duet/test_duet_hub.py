"""CPU checks of evaluating a released DUET set from the Hugging Face Hub (no network, no torch, no GPU): a fake
`huggingface_hub` module serves a release repository (snapshot into a local directory, model_info with the LFS sha256 of the
component file) and a base repository (the standard cache layout models--<owner>--<repo>/snapshots/<commit>); the real code
paths of sglang.srt.duet.release (fetch_release, fetch_base, check_pin, check_base), sglang.srt.duet.adapters.localize_release (a Hub
id given to sglang.launch_server), benchmark/duet_eval/launch_duet.py and benchmark/duet_eval/fetch_duet.py run against it.

Run from this directory:  python -m unittest test_duet_hub
"""
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import shutil
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ld = _load("hub_test_launch_duet", ROOT / "benchmark/duet_eval/launch_duet.py")
fd = _load("hub_test_fetch_duet", ROOT / "benchmark/duet_eval/fetch_duet.py")
release = ld._load_fork_release_module()

SPEC = dict(model="lightning", prefill_depth=2, latent_rank=16, latent_spikes=2, latent_id_side=True,
            latent_z_format="nvfp4", latent_value_format="bf16", latent_index_format="gap8",
            state_rank=3, state_every=4, state_sink="explicit", latent_init="", state_init="", name="toy-k2")
BASE_CONFIG = json.dumps({"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"}).encode()
BASE_COMMIT = "a" * 40
RELEASE_COMMIT = "c" * 40
PRIVATE_REPO = "Latent-Voyager/Toy-Model-SEED"     # a private repository named unlike its manifest's name ("toy-release")


def write_release(root, *, base_config=BASE_CONFIG, base_revision=BASE_COMMIT, base_id="toy/base"):
    """A minimal valid release directory without torch: a one-tensor safetensors file written by hand."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"__metadata__": {"format": "pt"}, "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    payload = struct.pack("<2f", 1.0, 2.0)
    path = root / release.COMPONENT_FILE
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"name": "toy-release", "base_model": base_id, "sha256": sha, "file_bytes": path.stat().st_size,
                "base": {"hf_id": base_id, "revision": base_revision, "config_sha256": hashlib.sha256(base_config).hexdigest()},
                "provenance": {"spec": SPEC, "trained_spec": SPEC}, "total_params": 2, "tensors": {"x": {"shape": [2], "bytes": 8}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "spec.json").write_text(json.dumps(SPEC))
    return sha


class FakeHub:
    """A stand-in for the huggingface_hub module: repos = {repo: dict(commit=..., release_dir=... | config=bytes, lfs=sha)}."""

    def __init__(self, repos, *, logged_in=True):
        self.repos, self.calls, self.logged_in = repos, [], logged_in
        self.module = types.ModuleType("huggingface_hub")
        hub = self

        class RepositoryNotFoundError(Exception):
            pass

        class HfApi:
            def model_info(self, repo, revision=None, files_metadata=False):
                hub.calls.append(("model_info", repo, revision, files_metadata))
                r = hub.repos[repo]
                if r.get("private") and not hub.logged_in:
                    raise RepositoryNotFoundError(f"401 Client Error. Repository Not Found for url: .../api/models/{repo}")
                if revision not in (None, "main", r["commit"]) and revision not in r.get("aliases", ()):
                    return SimpleNamespace(sha="f" * 40, siblings=None)      # another commit of the repository
                siblings = None
                if files_metadata and "lfs" in r:
                    siblings = [SimpleNamespace(rfilename="spec.json", lfs=None),
                                SimpleNamespace(rfilename=release.COMPONENT_FILE, lfs={"sha256": r["lfs"], "size": 0})]
                return SimpleNamespace(sha=r["commit"], siblings=siblings)

        def snapshot_download(repo_id, revision=None, local_dir=None, cache_dir=None, allow_patterns=None, **_):
            hub.calls.append(("snapshot_download", repo_id, revision, allow_patterns))
            r = hub.repos[repo_id]
            if local_dir is not None:                      # a release: the files into local_dir
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                for f in allow_patterns or release.RELEASE_FILES:
                    shutil.copy(Path(r["release_dir"]) / f, Path(local_dir) / f)
                return local_dir
            snap = hub.snapshot_dir(repo_id, revision, cache_dir)
            (snap / "config.json").write_bytes(r["config"])
            if allow_patterns != ["config.json"]:
                (snap / "model.safetensors").write_bytes(b"weights")
            return str(snap)

        def hf_hub_download(repo_id, filename, revision=None, cache_dir=None, **_):
            hub.calls.append(("hf_hub_download", repo_id, filename, revision))
            snap = hub.snapshot_dir(repo_id, revision, cache_dir)
            (snap / filename).write_bytes(hub.repos[repo_id]["config"])
            return str(snap / filename)

        self.module.HfApi, self.module.snapshot_download, self.module.hf_hub_download = HfApi, snapshot_download, hf_hub_download

    @staticmethod
    def snapshot_dir(repo_id, revision, cache_dir):
        snap = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
        snap.mkdir(parents=True, exist_ok=True)
        return snap

    def __enter__(self):
        self.saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = self.module
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = self.saved


class HubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.src = self.d / "src-release"
        self.sha = write_release(self.src)
        self.cache = self.d / "hf"
        self.repos = {"o/rel": dict(commit=RELEASE_COMMIT, release_dir=self.src, lfs=self.sha, aliases=("v3.5",)),
                      PRIVATE_REPO: dict(commit=RELEASE_COMMIT, release_dir=self.src, lfs=self.sha, private=True),
                      "toy/base": dict(commit=BASE_COMMIT, config=BASE_CONFIG)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_release_from_the_hub_matches_the_lfs_sha256_and_the_manifest(self):
        with FakeHub(self.repos) as hub:
            path = release.fetch_release("o/rel", self.cache, "v3.5")
        self.assertEqual(path, self.cache / "duet-releases" / "o--rel" / RELEASE_COMMIT)
        self.assertIn(("model_info", "o/rel", "v3.5", True), hub.calls)          # file metadata requested (LFS sha256)
        self.assertEqual(release.verify_release(path).sha256, self.sha)
        self.assertEqual(release.hub_lfs_sha256(SimpleNamespace(siblings=None), release.COMPONENT_FILE), None)

    def test_release_whose_file_differs_from_the_hub_record_is_refused(self):
        self.repos["o/rel"]["lfs"] = "0" * 64
        with FakeHub(self.repos), self.assertRaisesRegex(ValueError, "LFS sha256"):
            release.fetch_release("o/rel", self.cache, None)

    def test_tampered_download_fails_the_manifest_check(self):
        with FakeHub(self.repos):
            path = release.fetch_release("o/rel", self.cache, None)
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["sha256"] = "1" * 64
        (path / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "SHA256 differs from manifest"):
            release.verify_release(path)
        (path / release.COMPONENT_FILE).unlink()
        with self.assertRaises(FileNotFoundError):
            release.verify_release(path)

    def test_sha256_pin(self):
        release.check_pin(self.sha, None)
        release.check_pin(self.sha, self.sha)
        release.check_pin(self.sha, self.sha[:12].upper())
        with self.assertRaisesRegex(ValueError, "does not match"):
            release.check_pin(self.sha, "0" * 12)
        with self.assertRaisesRegex(ValueError, "12-64 hex"):
            release.check_pin(self.sha, "abc")

    def test_base_from_the_hub_at_the_training_revision(self):
        with FakeHub(self.repos) as hub:
            snap = release.fetch_base("toy/base", self.cache, release_dir=self.src)
        self.assertEqual(snap, self.cache / "hub" / "models--toy--base" / "snapshots" / BASE_COMMIT)
        self.assertIn(("model_info", "toy/base", BASE_COMMIT, False), hub.calls)  # no @rev: the manifest's training revision
        self.assertTrue((snap / "model.safetensors").is_file())
        self.assertIsNone(release.check_base(self.src, snap))
        self.assertTrue(release.is_hub_id("toy/base@main"))
        self.assertFalse(release.is_hub_id(str(self.src)))
        self.assertFalse(release.is_hub_id("/abs/path"))

    def test_base_mismatches_are_refused_before_the_download(self):
        with FakeHub(self.repos) as hub:
            with self.assertRaisesRegex(ValueError, "trained on toy/base"):
                release.fetch_base("other/base", self.cache, release_dir=self.src)
            with self.assertRaisesRegex(ValueError, "resolves to f{40}"):
                release.fetch_base("toy/base@someotherbranch", self.cache, release_dir=self.src)
            self.repos["toy/base"]["config"] = b'{"architectures": ["NemotronHForCausalLM"], "changed": 1}'
            with self.assertRaisesRegex(ValueError, "config.json sha256"):
                release.fetch_base("toy/base", self.cache, release_dir=self.src)
            self.assertFalse(any(c[0] == "snapshot_download" for c in hub.calls))   # nothing large was fetched

    def test_base_without_a_recorded_revision_is_checked_by_its_config(self):
        rel = self.d / "rel-norev"
        write_release(rel, base_revision=None)
        with FakeHub(self.repos) as hub:
            snap = release.fetch_base("toy/base", self.cache, release_dir=rel, config_only=True)
        self.assertIn(("model_info", "toy/base", None, False), hub.calls)
        self.assertIn(("snapshot_download", "toy/base", BASE_COMMIT, ["config.json"]), hub.calls)
        self.assertFalse((snap / "model.safetensors").exists())

    def test_launcher_dry_run_with_release_and_base_from_the_hub(self):
        out = self.d / "out"
        argv = ["--model", "toy/base", "--duet-release", "o/rel@v3.5", "--arm", "duet", "--port", "30600",
                "--hf-cache", str(self.cache), "--duet-release-sha256", self.sha[:16], "--dry-run", "--out", str(out)]
        with FakeHub(self.repos), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ld.main(argv), 0)
        rec = json.loads((out / "launch.json").read_text())
        snap = (self.cache / "hub" / "models--toy--base" / "snapshots" / BASE_COMMIT).resolve()
        self.assertEqual((rec["model_hub"], rec["model"], rec["family"]), ("toy/base", str(snap), "lightning"))
        self.assertEqual(rec["release"], str((self.cache / "duet-releases" / "o--rel" / RELEASE_COMMIT).resolve()))
        self.assertEqual(rec["release_identity"]["sha256"], self.sha)
        self.assertEqual(rec["release_sha256_pin"], self.sha[:16])
        self.assertIn("--duet-release", rec["argv"])
        bad = argv[:argv.index("--duet-release-sha256") + 1] + ["0" * 16] + argv[argv.index("--duet-release-sha256") + 2:]
        with FakeHub(self.repos), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "release pin"):
            ld.main(bad)

    def test_local_directories_record_no_hub_fields(self):
        base = self.d / "base"
        base.mkdir()
        (base / "config.json").write_bytes(BASE_CONFIG)
        out = self.d / "out-local"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ld.main(["--model", str(base), "--duet-release", str(self.src), "--arm", "stock", "--port", "30600",
                                      "--hf-cache", str(self.cache), "--dry-run", "--out", str(out)]), 0)
        rec = json.loads((out / "launch.json").read_text())
        self.assertNotIn("model_hub", rec)
        self.assertNotIn("release_sha256_pin", rec)
        self.assertEqual(rec["model"], str(base))

    def test_fetch_cli_prints_the_exam_job_variables(self):
        buf = io.StringIO()
        with FakeHub(self.repos), contextlib.redirect_stdout(buf):
            self.assertEqual(fd.main(["--release", "o/rel", "--base", "--hf-cache", str(self.cache)]), 0)
        last = buf.getvalue().strip().splitlines()[-1]
        snap = (self.cache / "hub" / "models--toy--base" / "snapshots" / BASE_COMMIT).resolve()
        self.assertEqual(last, f"export DUET_REL={self.cache.resolve() / 'duet-releases' / 'o--rel' / RELEASE_COMMIT} "
                               f"LIGHTNING_BASE={snap}")

    def test_private_repository_named_unlike_its_manifest(self):
        out = self.d / "out-private"
        argv = ["--model", "toy/base", "--duet-release", PRIVATE_REPO, "--arm", "duet", "--port", "30600",
                "--hf-cache", str(self.cache), "--dry-run", "--out", str(out)]
        with FakeHub(self.repos), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ld.main(argv), 0)
        rec = json.loads((out / "launch.json").read_text())
        self.assertEqual(rec["release_identity"]["name"], "toy-release")         # the manifest's name; nothing compares it
        self.assertEqual(rec["release"], str((self.cache / "duet-releases" / "Latent-Voyager--Toy-Model-SEED" / RELEASE_COMMIT).resolve()))

    def test_private_repository_without_a_login_names_the_cause(self):
        with FakeHub(self.repos, logged_in=False):
            with self.assertRaisesRegex(ValueError, "needs access on the Hub and a login"):
                release.fetch_release(PRIVATE_REPO, self.cache, None)
            argv = ["--model", "toy/base", "--duet-release", PRIVATE_REPO, "--arm", "duet", "--port", "30600",
                    "--hf-cache", str(self.cache), "--dry-run", "--out", str(self.d / "out-nologin")]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "needs access on the Hub"):
                ld.main(argv)

    def test_server_args_hub_id_is_localized_once(self):
        adapters = importlib.import_module("sglang.srt.duet.adapters")
        env = {"HF_HOME": str(self.cache)}
        with FakeHub(self.repos):
            args = SimpleNamespace(duet_release=PRIVATE_REPO + "@" + RELEASE_COMMIT)
            directory = adapters.localize_release(args, environ=env)
            self.assertEqual(directory, self.cache / "duet-releases" / "Latent-Voyager--Toy-Model-SEED" / RELEASE_COMMIT)
            self.assertEqual(args.duet_release, str(directory))
            self.assertIsNone(adapters.localize_release(args, environ=env))     # a directory stays exactly as given
            from_env = SimpleNamespace(duet_release=None)
            self.assertEqual(adapters.localize_release(from_env, environ=dict(env, SGLANG_DUET_DIR=PRIVATE_REPO)), directory)
            self.assertEqual(from_env.duet_release, str(directory))
        for value in (None, "/no/such/release"):                            # no release, or a path: never a Hub lookup
            other = SimpleNamespace(duet_release=value)
            self.assertIsNone(adapters.localize_release(other, environ={}))
            self.assertEqual(other.duet_release, value)

    def test_base_without_a_recorded_revision_is_noted_once(self):
        rel = self.d / "rel-norev"
        write_release(rel, base_revision=None)
        for value, noted in (("toy/base", True), ("toy/base@" + BASE_COMMIT, False)):
            err = io.StringIO()
            with FakeHub(self.repos), contextlib.redirect_stderr(err):
                ld.resolve_base(value, self.cache, rel, config_only=True)
            self.assertEqual(err.getvalue().count("records no base revision"), int(noted))


if __name__ == "__main__":
    unittest.main()
