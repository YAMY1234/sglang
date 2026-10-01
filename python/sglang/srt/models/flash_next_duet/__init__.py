"""In-tree Flash-Next DUET adapter for NVFP4 and BF16 bases.

Keep host configuration imports lightweight; registry scans model.py for EntryClass.
"""


def __getattr__(name):
    if name == "EntryClass":
        from .model import EntryClass

        return EntryClass
    raise AttributeError(name)
