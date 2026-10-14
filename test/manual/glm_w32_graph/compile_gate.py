"""Offline SM103 compilation, including both due branches, without CUDA init."""

import json

from numeric_gate import kernel
from triton import compile as compile_triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

POINTERS = {
    "B": "i32",
    "FIRST": "i32",
    "SLOTS": "i32",
    "Indices": "*i32",
    "Phase": "*i32",
    "Pending": "*i1",
    "Active": "*i1",
    "G": "*fp64",
    "Z": "*fp64",
    "G_ptr": "*fp64",
    "Z_ptr": "*fp64",
    "D_ptr": "*fp64",
}
BASE = {
    "B": 3,
    "H": 16,
    "FIRST": 0,
    "COLD": True,
    "W": 32,
    "D": 128,
    "R": 32,
    "M": 40,
    "PITCH": 40,
    "T": 64,
    "SLOTS": 41,
    "N": 40,
    "TILE": 64,
    "MAX_SWEEPS": 12,
    "REL_TOL": 1e-14,
    "TRANSPOSE": True,
    "CLEAR": False,
    "BLOCK": 128,
}


def run():
    records = []
    for name in (
        "_prepare",
        "_qr2",
        "_range_product",
        "_gram",
        "_eigh_kernel",
        "_reconstruct",
        "_publish_warm",
        "_phase",
    ):
        fn = getattr(kernel, name)
        variants = {
            "_qr2": ({"M": 40, "T": 64}, {"M": 32, "T": 32}),
            "_range_product": ({"TRANSPOSE": True}, {"TRANSPOSE": False}),
            "_phase": ({"CLEAR": True}, {"CLEAR": False}),
        }.get(name, ({},))
        variants = [v | {"H": h} for v in variants for h in (1, 16)]
        for variant in variants:
            for cold in (True, False):
                constants = {
                    k: v
                    for k, v in (BASE | variant | {"COLD": cold}).items()
                    if k in fn.arg_names and k not in ("B", "FIRST", "SLOTS")
                }
                signature = {
                    k: ("constexpr" if k in constants else POINTERS.get(k, "*fp32"))
                    for k in fn.arg_names
                }
                compiled = compile_triton(
                    ASTSource(fn, signature, constants),
                    target=GPUTarget("cuda", 103, 32),
                    options={
                        "num_warps": 16
                        if name == "_qr2"
                        else (4 if name in ("_publish_warm", "_phase") else 8),
                        "enable_fp_fusion": name
                        in ("_range_product", "_publish_warm", "_phase"),
                    },
                )
                record = {
                    "kernel": name,
                    "cold": cold,
                    "variant": variant,
                    "metadata": compiled.metadata._asdict(),
                }
                records.append(record)
                print("GLMW32_COMPILE " + json.dumps(record, default=str), flush=True)
    return records


if __name__ == "__main__":
    run()
