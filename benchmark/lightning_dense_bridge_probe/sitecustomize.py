"""Opt-in qualification instrumentation; does not modify model math/arguments."""

import os

if os.environ.get("LTNBRIDGE_PROBE_DIR"):
    import importlib.abc
    import importlib.machinery
    import sys

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return (
                self.original.create_module(spec)
                if hasattr(self.original, "create_module")
                else None
            )

        def exec_module(self, module):
            self.original.exec_module(module)
            from probe import install

            install(module.ModelRunner)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sglang.srt.model_executor.model_runner":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())
