"""docs/222 §15 PD gate: seed the arm's compile caches exactly as the 1P:1D window does (trio.sh -> the package's
support/prelude/cache_seed_node.py, storage only: verified bytes copied to the receipt's cache paths, existing entries kept).
Host python3, no container, no GPU.  The receipt goes to $OUT/prelude-<arm>.json instead of the campaign tree.
    python3 pdgate_seed.py $OUT <arm>
"""
import importlib.util
import json
import sys
from pathlib import Path


def main(out, arm):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    a = plan["arms"][arm]
    binding = json.loads(Path(a["prelude"]).read_text())
    # the window seeds the cache paths its servers use; the gate keeps those paths, so both must agree
    for role in ("P", "D"):
        for k, v in binding["cache_environment"].items():
            assert a["cache_env"][role].get(k) == v, (arm, role, k, a["cache_env"][role].get(k), v)
    spec = importlib.util.spec_from_file_location("cache_seed_node", Path(a["prelude"]).parent / "cache_seed_node.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.populate(binding["seed"], binding["seed_MANIFEST_SHA256"], binding["cache_environment"],
                       out / f"prelude-{arm}.json")
    print(json.dumps({k: v for k, v in res["mappings"].items()}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
