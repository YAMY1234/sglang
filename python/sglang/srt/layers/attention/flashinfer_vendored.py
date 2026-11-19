"""FlashInfer files vendored by the SM107 overlay, loaded as their flashinfer.* names.

A vendored file keeps FlashInfer's relative imports, so it runs as the module it
replaces; each is checked against its upstream git blob before it can load.
"""

import hashlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path


def git_blob_sha1(data: bytes) -> str:
    header = b"blob %d\0" % len(data)
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


class _FilesFinder(importlib.abc.MetaPathFinder):
    """Loads each named module from its file, under that name."""

    def __init__(self, files: dict[str, Path]):
        self.files = files

    def find_spec(self, fullname, path=None, target=None):
        file = self.files.get(fullname)
        return importlib.util.spec_from_file_location(fullname, file) if file else None


def load_vendored(files: dict[str, tuple[Path, str]]) -> None:
    """Serve each module {qualified name: (vendored file, its git blob)} from its file
    when it is first imported; refuses a module already imported or a changed file."""
    for name, (file, blob) in files.items():
        if name in sys.modules:
            raise RuntimeError(f"{name} was imported before its vendored copy loaded")
        if git_blob_sha1(file.read_bytes()) != blob:
            raise RuntimeError(f"{file} is not the vendored file (git blob {blob})")
    finder = _FilesFinder({name: file for name, (file, _) in files.items()})
    sys.meta_path.insert(0, finder)
