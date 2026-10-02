"""Apply the SHA-pinned patch only to a private container's FlashInfer package."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    assets = Path(__file__).resolve().parent
    package = args.package.resolve()
    assert package.name == "flashinfer" and (package / "__init__.py").is_file()
    manifest = json.loads((assets / "source-sha256.json").read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {name: sha(package / name) for name in manifest}
    if all(before[n] == v["before"] for n, v in manifest.items()):
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-d", str(package)],
            input=(assets / "flashinfer.patch").read_bytes(),
            check=True,
        )
    else:
        assert all(before[n] == v["after"] for n, v in manifest.items()), before
    target = package / "fused_moe/cute_dsl/compact_init.py"
    shutil.copyfile(assets / "compact_init.py", target)
    after = {name: sha(package / name) for name in manifest}
    assert all(after[n] == v["after"] for n, v in manifest.items())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(
            {
                "package": str(package),
                "before": before,
                "after": after,
                "helper_sha256": sha(target),
                "default_enabled": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
