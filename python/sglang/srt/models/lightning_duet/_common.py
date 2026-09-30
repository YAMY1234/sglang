"""Import the model-independent DUET layer (sglang.srt.duet.<name>).

Inside the image `import sglang` works and the regular import is used.  On a bare CPU box without the runtime's
dependencies (triton, orjson, ...) `sglang/__init__.py` cannot be executed, so the module file is loaded directly --
the common modules depend on torch only.  Both paths hand out one module object per name (sys.modules cache), so
`is` identity between the line shims and the common layer holds either way.
"""
import importlib
import importlib.util
import sys
from pathlib import Path


def _duet_dir():
    spec = importlib.util.find_spec("sglang")           # does not execute sglang/__init__.py
    if spec is not None and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0]) / "srt" / "duet"
    return Path(__file__).resolve().parents[2] / "duet"


# Decide once, without touching the sglang package: a half-executed `sglang/__init__.py` (missing triton / orjson)
# leaves importlib in a state where later imports fail with KeyError('sglang'), so the regular import is only
# attempted when the runtime's dependencies are present.
_RUNTIME_AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("triton", "orjson", "psutil"))


def load(name):
    key = f"sglang.srt.duet.{name}"
    if key in sys.modules:
        return sys.modules[key]
    if _RUNTIME_AVAILABLE:
        return importlib.import_module(key)
    spec = importlib.util.spec_from_file_location(key, _duet_dir() / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module
