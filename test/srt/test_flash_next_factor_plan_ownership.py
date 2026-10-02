"""Replay pinned integer-only pool planning on CPU with a tensor indexing shim.

No model inference, floating-point equivalence, CUDA timing or GPU submission.
"""

import ast
import argparse
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

H = Path(__file__).resolve().parent
REPO = H.parents[1]
SHA = "working-tree"
PATH = "python/sglang/srt/mem_cache/gdn_factored_pool.py"


class V:
    def __init__(self, values, dtype=None, device=None):
        self.values = copy.deepcopy(list(values))
        self.shape = (len(self.values),)

    def to(self, *a, **k):
        return V(self.values)

    def clamp(self, min):
        return V([max(x, min) for x in self.values])

    def tolist(self):
        return copy.deepcopy(self.values)

    def __getitem__(self, k):
        return (
            V([self.values[i] for i in k.values])
            if isinstance(k, V)
            else self.values[k]
        )

    def __setitem__(self, k, value):
        if isinstance(k, V):
            values = value.values if isinstance(value, V) else [value] * len(k.values)
            for i, v in zip(k.values, values):
                self.values[i] = v
        else:
            self.values[k] = value


def normalize(plan):
    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in vars(plan).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-torch", action="store_true")
    args = parser.parse_args()
    tensor = V
    raw = (REPO / PATH).read_bytes()
    tree = ast.parse(raw)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "plan_extend"
    )
    fn = copy.deepcopy(fn)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name="Pool", bases=[], keywords=[], body=[fn], decorator_list=[]
            ),
        ],
        type_ignores=[],
    )
    scope = dict(
        torch=NS(
            long="int64",
            bool="bool",
            int32="int32",
            stack=lambda vs: V([v.values for v in vs]),
            tensor=V,
        ),
        FactoredExtendPlan=NS,
        guard_abort_enabled=lambda: False,
    )
    if args.real_torch:
        import torch

        torch.set_num_threads(1)
        scope["torch"] = torch
        tensor = lambda values, dtype=None: torch.tensor(
            list(values), dtype=getattr(torch, dtype or "int64")
        )
    exec(
        compile(ast.fix_missing_locations(tree), "<pinned-plan-extend>", "exec"), scope
    )
    Pool = scope["Pool"]

    def pool(states, ring):
        p = Pool()
        p.pside_join = lambda: None
        p.cfg = NS(strict_chunk=True, factored_prefix=True)
        p.layer_ids = list(range(36))
        p.device = "cpu"
        p.stats = defaultdict(int)
        n = 64
        p.stale = tensor([1] * n, dtype="int32")
        p.dense_of = tensor([-1] * n, dtype="int32")
        p.dense_required = tensor([0] * n, dtype="int32")
        p.prefix_valid = tensor([1] * n, dtype="int32")
        p.ring_owner = ring[:]
        p.ring_lru = list(range(len(ring)))
        for s, stale, required in states:
            p.stale[s] = stale
            p.dense_required[s] = required
            if s in ring:
                p.dense_of[s] = ring.index(s)
        return p

    def run(p, lens, prefix, final, twice):
        active = [i for i, (l, f) in enumerate(zip(lens, final)) if l - int(f) > 0]
        plans = []
        try:
            if twice:
                plans.append(
                    normalize(
                        p.plan_extend(
                            tensor(range(len(lens))),
                            lens,
                            prefix_lens=prefix,
                            prompt_final=final,
                        )
                    )
                )
            plans.append(
                normalize(
                    p.plan_extend(
                        tensor(active),
                        [lens[i] - int(final[i]) for i in active],
                        prefix_lens=[prefix[i] for i in active],
                        prompt_final=[final[i] for i in active],
                    )
                )
            )
            return dict(
                ok=True,
                plans=plans,
                owners=p.ring_owner,
                dense_of=p.dense_of.tolist(),
                stats=dict(p.stats),
            )
        except RuntimeError as e:
            return dict(
                ok=False,
                error=str(e),
                plans=plans,
                owners=p.ring_owner,
                dense_of=p.dense_of.tolist(),
            )

    rows = []
    # Legal integer metadata examples, independent of the observed service workload.
    # Inactive owners are either disposable completed states or protected unfinished prompts.
    for B in [1, 2, 4, 8]:
        for kind in ["fresh", "prefix", "continuation", "mixed"]:
            for final_kind in ["intermediate", "final", "mixed"]:
                for filled in [False, True]:
                    lens = [
                        32768 if final_kind == "intermediate" else 4096
                        for _ in range(B)
                    ]
                    final = [
                        final_kind == "final" or (final_kind == "mixed" and i % 2 == 0)
                        for i in range(B)
                    ]
                    prefix = [
                        0 if kind == "fresh" or (kind == "mixed" and i % 2) else 32768
                        for i in range(B)
                    ]
                    states = []
                    owners = [-1] * 16
                    for i in range(B):
                        cont = kind == "continuation" or (
                            kind == "mixed" and i % 2 == 0
                        )
                        if cont:
                            owners[i] = i
                            states.append((i, 0, 1))
                        else:
                            states.append((i, 1, 0))
                    if filled:
                        for r in range(16):
                            if owners[r] < 0:
                                owners[r] = 16 + r
                                states.append((16 + r, 1, 0))
                    p = pool(states, owners)
                    a = run(copy.deepcopy(p), lens, prefix, final, True)
                    b = run(copy.deepcopy(p), lens, prefix, final, False)
                    # Physical destinations and source modes, excluding plan counters.
                    equal = (
                        a["ok"]
                        and b["ok"]
                        and a["plans"][-1] == b["plans"][-1]
                        and a["owners"] == b["owners"]
                    )
                    rows.append(
                        dict(
                            B=B,
                            kind=kind,
                            final_kind=final_kind,
                            filled=filled,
                            equal=equal,
                            twice=a,
                            single=b,
                        )
                    )
    # All other ring entries hold protected continuations. A completing source
    # may be reused by a mandatory destination because reads precede writes.
    # An abandoned first plan changes ownership before those reads can happen.
    p = pool(
        [(0, 0, 1), (1, 1, 0)] + [(i, 0, 1) for i in range(2, 17)],
        [0] + list(range(2, 17)),
    )
    twice = run(copy.deepcopy(p), [4096, 32768], [32768, 0], [True, False], True)
    single = run(copy.deepcopy(p), [4096, 32768], [32768, 0], [True, False], False)
    assert not twice["ok"] and single["ok"], (twice, single)
    assert (
        twice["error"] == "unfinished x256 prompt lost its exact GDN continuation state"
    )
    counterexample = dict(
        slots=[0, 1],
        lengths=[4096, 32768],
        prefixes=[32768, 0],
        final=[True, False],
        initial_owners=p.ring_owner,
        initial_stale=p.stale.tolist(),
        initial_required=p.dense_required.tolist(),
        twice=twice,
        single=single,
    )
    result = dict(
        sha=SHA,
        path=PATH,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        real_torch=args.real_torch,
        cases=len(rows),
        equal=sum(r["equal"] for r in rows),
        rows=rows,
        counterexample=counterexample,
        scope="Exact integer planning function, vector indexing shim. Synthetic legal slot states; not observed service failure. No model tensors or numerical parity assertion.",
    )
    # Summary only by default; detailed fixtures remain in the research report.
    if __import__("os").environ.get("PFACTOR4_TEST_OUTPUT"):
        Path(__import__("os").environ["PFACTOR4_TEST_OUTPUT"]).write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps({k: result[k] for k in ["cases", "equal", "real_torch", "scope"]}))
    print(
        json.dumps(
            dict(counterexample_twice=twice.get("error"), single_ok=single["ok"])
        )
    )


if __name__ == "__main__":
    main()
