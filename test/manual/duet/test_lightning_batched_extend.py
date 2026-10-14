"""GPU admission for standard emitter scan + batched factor publication.

No model weights: only release components. Reports numeric changes to the
frozen serial emitter and factorizer; lifecycle assertions are hard failures.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.models.lightning_duet.components import Components
from sglang.srt.models.lightning_duet._common import load
from sglang.srt.layers.attention.linear.mamba2_emitter_extend import emitter_extend

Pool = load("mamba2_factored_pool").Mamba2FactoredStatePool


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(20261014)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = dict(complete=False, pool=[], emitters=[])

    def save():
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    for batch in (1, 8):
        directions = torch.randn(3, 4, 64, device="cuda")
        kw = dict(n=128, rank=16, window=16, conv_dim=4, conv_width=3)
        new, old = (Pool(batch + 2, (0, 2), directions, **kw) for _ in range(2))
        slots = torch.arange(batch, 0, -1, device="cuda")
        for repeat in range(2):
            plan = new.begin_extend(slots)
            try:
                new.reset_slots(slots)
                raise AssertionError("lifecycle was admitted during extend")
            except RuntimeError:
                pass
            for lid in (0, 2):
                state = torch.randn(batch, 4, 64, 128, device="cuda")
                if repeat == 0:
                    new.commit_extend(lid, plan, state)
                else:
                    # Request-local shallow writes and packed emitter writes
                    # share the collector; row order need not be slot order.
                    for row in reversed(range(batch)):
                        new.commit_extend(lid, plan, state[row:row+1], rows=(row,))
                    try:
                        new.commit_extend(lid, plan, state[:1], rows=(0,))
                        raise AssertionError("duplicate layer-row accepted")
                    except RuntimeError:
                        pass
                for row, slot in enumerate(slots.cpu().tolist()):
                    old.initialize(lid, slot, state[row], torch.zeros(4, 3, device="cuda"))
            new.finish_extend(plan)
            delta = max(rel(new.materialize(lid, slot), old.materialize(lid, slot))
                        for lid in (0, 2) for slot in slots.cpu().tolist())
            assert new.dense.numel() == 0 and new.resident_bytes() == old.resident_bytes()
            result["pool"].append(dict(B=batch, repeat=repeat, old_new_materialized_rel_f=delta,
                                       commit_groups=plan.commits, empty_dense=True,
                                       rowwise=bool(repeat)))
            save()
        plan = new.begin_extend(slots)
        try:
            new.finish_extend(plan)
            raise AssertionError("missing layers accepted")
        except RuntimeError:
            new.abort_extend(plan)
        assert not bool(new.valid[:, slots].any())
        new.reset_slots(slots)

    from sglang.srt.configs.nemotron_h import NemotronHConfig

    # Use the serving config's n_groups/chunk_size compatibility aliases.
    config = NemotronHConfig(**json.loads((args.model / "config.json").read_text()))
    comp = Components(args.release, config, "cuda")
    g = comp.geometry
    for lengths in ([1], [127], [1, 3, 63, 127, 128, 129, 191, 257]):
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        residual = torch.randn(cu[-1], g.hidden, device="cuda").bfloat16()
        metadata = SimpleNamespace(query_start_loc=torch.tensor(cu, device="cuda", dtype=torch.int32),
            mixed_metadata=SimpleNamespace(extend_seq_lens_cpu=lengths, chunk_size=comp.chunk,
                has_initial_states=torch.zeros(len(lengths), device="cuda", dtype=torch.bool),
                seq_idx=torch.repeat_interleave(torch.arange(len(lengths), device="cuda", dtype=torch.int32),
                    torch.tensor(lengths, device="cuda"), output_size=sum(lengths))[None],
                chunk_indices=None, chunk_offsets=None))
        states = {}
        slots = torch.arange(len(lengths), 0, -1, device="cuda")
        pool = SimpleNamespace(layer_map={lid: i for i, lid in enumerate(comp.mamba_emitters)},
            conv=torch.zeros(len(comp.mamba_emitters), len(lengths)+1, g.conv_dim, g.conv_width, device="cuda"),
            commit_extend=lambda lid, plan, state: states.__setitem__(lid, state))
        for lid in (*comp.mamba_emitters, *comp.attention_emitters):
            emitted = emitter_extend(comp, lid, residual, metadata, pool, SimpleNamespace(slots=slots))
            deltas = []
            for row, length in enumerate(lengths):
                old = comp.emitter(lid, residual[cu[row]:cu[row+1]])
                if emitted is None:
                    delta = dict(state=rel(states[lid][row], old["state"]),
                                 conv=rel(pool.conv[pool.layer_map[lid], slots[row]], old["conv"]))
                else:
                    delta = {key: rel(emitted[key][cu[row]:cu[row+1]], old[key]) for key in ("k", "v")}
                assert all(value == 0.0 for value in delta.values()), delta
                deltas.append(delta)
            result["emitters"].append(dict(layer=lid, lengths=lengths, relative_f=deltas))
            save()
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
