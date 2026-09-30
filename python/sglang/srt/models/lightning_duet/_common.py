"""Import the model-independent DUET layer (sglang.srt.duet.<name>).

When the runtime is initialized, the regular import is used. Stand-alone CPU tests load the lightweight common
package without executing `sglang/__init__.py` (dependency availability alone does not establish a runtime) --
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


def load(name):
    key = f"sglang.srt.duet.{name}"
    if key in sys.modules:
        return sys.modules[key]
    if "sglang.srt" in sys.modules:
        return importlib.import_module(key)
    # Register the light common package so its modules can use relative imports
    # without bootstrapping SGLang's GPU-facing top-level public API.
    package = "sglang.srt.duet"
    if package not in sys.modules:
        path = _duet_dir()
        spec = importlib.util.spec_from_file_location(
            package, path / "__init__.py", submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return importlib.import_module(key)
