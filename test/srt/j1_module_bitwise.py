"""docs/223 §16 J1 module-level bitwise gate: the deferred publication split into a normal and a tracked half.

Served-shape FactoredGDNPool + whole-prefix k31 graph (built by k42_module_bitwise.build_pool, prewarm with
SGLANG_GDN_J1_SPLIT_PUBLICATION=1, i.e. the split variants of every bucket captured next to the one-graph ones).
Per rep, from identical pool bytes:
  F   today's one-graph publication on the forward stream, then the readers (reference).
  J   J1, label B: PDBatchPublication with J1Split("1") on the side stream after the "forward"; the stand-in logits;
      publication.join_decode(); the DECODE reader (factored_packed_decode on the request slots); the decode-visible
      rows (request slots + aligned final copies) snapshot; pool.pside_join() (the whole publication); the CHECKPOINT
      reader (track slots + final_dst); every pool tensor.
  JR  J with a ~20 ms spin on the publication stream between the halves (the tracked half certainly still runs while
      the decode reader runs).
  COW JR, then pool.copy_slots(track slot -> a fresh slot) right after join_decode (a radix COW: the pool method must
      join the whole publication first), then the fresh slot's rows.
Gate (all must hold, bitwise): G1 J / JR decode reader == F; G2 decode-visible rows after join_decode == F; G3 every
pool tensor after the whole join == F and the checkpoint reader == F; G4 COW rows == F's copy.  Tracked branch on/off.
Negative controls (must be DETECTED): n1 = JR's checkpoint reader after join_decode only (no whole join);
n2 = the decode reader with no join at all, spin before the normal half.

  PYTHONPATH=python python3 test/srt/j1_module_bitwise.py --heads 24 --batch 1,2 --reps 3 --out j1.json
"""
import argparse
import json
import os
import sys
import time

os.environ["SGLANG_GDN_J1_SPLIT_PUBLICATION"] = "1"
os.environ.pop("SGLANG_GDN_PREFILL_JOIN_BRANCHES", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k42_module_bitwise as k42  # noqa: E402  (sets the served C+ env defaults on import)

import torch  # noqa: E402

from sglang.srt.mem_cache import gdn_factored_pool as gp  # noqa: E402
from sglang.srt.mem_cache.gdn_pd_publication import CudaPublicationRuntime, J1Split, PDBatchPublication  # noqa: E402

DEV = k42.DEV
SPIN_CYCLES = int(os.environ.get("J1_SPIN_CYCLES", str(40_000_000)))  # ~20 ms at ~2 GHz


def reader_on(rd, slots):
    """The k42 reader inputs restricted to some of its rows (same activations per row)."""
    idx = [i for i, s in enumerate(rd["slots"].tolist()) if s in set(slots.tolist())]
    sel = torch.tensor(idx, device=DEV)
    out = dict(rd)
    out["slots"] = rd["slots"][sel]
    for k in ("mixed", "a", "b"):
        out[k] = rd[k][sel]
    return out


def rows(pool, slots):
    s = slots.long()
    out = dict(a=pool.a[:, s], U=pool.U[:, s], W=pool.W[:, s], count=pool.count[:, s], stale=pool.stale[s],
               dense_of=pool.dense_of[s], prefix_valid=pool.prefix_valid[s])
    if pool.dense_required is not None:
        out["dense_required"] = pool.dense_required[s]
    return {k: v.clone() for k, v in out.items()}


def eq_dict(x, y):
    return {k: k42.compare(x[k], y[k]) for k in x}


def all_equal(d):
    return all(v["equal"] for v in d.values())


def run_variant(kind, pool, graph, src, plan, controls, reader, live, standin, runtime):
    eager = gp.factorize_layers
    policy = (gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE, gp.factorize_dense)
    track_slots, final_src, final_dst = controls
    for t in live.values():
        if t is not None:
            t.zero_()
    torch.cuda.synchronize()
    standin()
    for k, t in live.items():
        if t is not None:
            t.copy_(src[k])
    tracked = live["tracked"]
    states = [(live["normal"][l], None if tracked is None else tracked[l]) for l in range(len(k42.GDN_LAYERS))]
    decode_rd = reader_on(reader, plan.slots)
    ckpt_slots = final_dst if track_slots is None else torch.cat([track_slots, final_dst])
    res = {}
    if kind == "F":
        pool._pd_batch_publication = None
        graph.run(pool, plan, states, *controls, eager=eager, policy=policy)
        res["decode_out"] = k42.run_reader(pool, decode_rd)
        res["decode_rows"] = rows(pool, torch.cat([plan.slots, final_dst]))
        res["ckpt_rows"] = rows(pool, ckpt_slots)
        torch.cuda.synchronize()
        return res
    publication = PDBatchPublication(pool, runtime, agg=True)
    publication.j1 = J1Split("1")
    pool._pd_batch_publication = publication
    original_run = graph.run
    spin = kind in ("JR", "COW", "n1")
    if spin or kind == "n2":
        def run(*a, between=None, **k):
            if kind == "n2":
                torch.cuda._sleep(SPIN_CYCLES)  # delay the whole publication

                return original_run(*a, between=between, **k)

            def between_then_spin():
                between()
                torch.cuda._sleep(SPIN_CYCLES)  # the tracked half starts ~20 ms after the decode join point
            return original_run(*a, between=between_then_spin, **k)
        graph.run = run
    try:
        publication.submit(graph, plan, states, controls, eager=eager, policy=policy)
        publication.start_after_forward()
        standin()
        if kind != "n2":
            publication.join_decode()
            if publication.pending is None:
                raise RuntimeError("label B join_decode joined the whole publication")
        res["decode_out"] = k42.run_reader(pool, decode_rd)
        res["decode_rows"] = rows(pool, torch.cat([plan.slots, final_dst]))
        if kind == "n1":
            res["ckpt_rows"] = rows(pool, ckpt_slots)  # early checkpoint reader: no whole join
        if kind == "COW" and track_slots is not None:
            fresh = controls_fresh(plan, controls)
            pool.copy_slots(track_slots, fresh)  # pool method: must join the whole publication first
            res["cow_rows"] = rows(pool, fresh)
        if kind not in ("n1", "n2"):
            pool.pside_join()
            if publication.pending is not None:
                raise RuntimeError("pside_join left the J1 publication pending")
            res["ckpt_rows"] = rows(pool, ckpt_slots)
        torch.cuda.synchronize()
    finally:
        graph.run = original_run
        if publication.pending is not None:
            torch.cuda.synchronize()
            publication.join()
        pool._pd_batch_publication = None
    res["stats"] = dict(publication.stats)
    return res


def controls_fresh(plan, controls):
    """Fresh destination slots for the COW case: the highest slots not used by this rep."""
    used = set(plan.slots.tolist()) | set(controls[0].tolist()) | set(controls[2].tolist())
    free = [s for s in range(k42.SIZE, 0, -1) if s not in used][:controls[0].numel()]
    return torch.tensor(free, dtype=torch.long, device=DEV)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", type=int, choices=(24, 48), default=24)
    ap.add_argument("--batch", default="1,2")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=316)
    ap.add_argument("--standin-n", type=int, default=8192)
    ap.add_argument("--standin-iters", type=int, default=6)
    args = ap.parse_args()
    batches = [int(x) for x in args.batch.split(",") if x]
    t_start = time.time()
    import tempfile

    pool, graph = k42.build_pool(args.heads, tempfile.mkdtemp(prefix="j1_bitwise_"))
    split_keys = [k for k in graph.entries if k[-1] == "j1-split"]
    if not split_keys:
        raise SystemExit("prewarm captured no J1 split variant (SGLANG_GDN_J1_SPLIT_PUBLICATION not seen)")
    standin = k42.StandIn(args.standin_n, args.standin_iters)
    runtime = CudaPublicationRuntime(pool.a.device)
    result = dict(meta=dict(argv=vars(args), device=torch.cuda.get_device_name(), pool_file=gp.__file__,
                            split_entries=sorted(str(k[:2]) for k in split_keys), spin_cycles=SPIN_CYCLES,
                            env={k: v for k, v in os.environ.items() if k.startswith("SGLANG_GDN")}), reps=[])
    ok = True
    print(f"pool + graphs ready in {time.time() - t_start:.0f}s; split variants {result['meta']['split_entries']}",
          flush=True)
    for tracked in (True, False):
        for b in batches:
            for rep in range(args.reps):
                seed = args.seed * 100003 + b * 1009 + rep + (7 if tracked else 0)
                src, plan, controls, reader = k42.make_rep(pool, b, tracked, seed)
                live = {k: None if v is None else torch.empty_like(v) for k, v in src.items()}
                init = k42.snapshot(pool)
                out = {}
                for kind in ("F", "J", "JR", "COW", "n1", "n2"):
                    if kind == "COW" and not tracked:
                        continue
                    k42.restore(pool, init)
                    out[kind] = run_variant(kind, pool, graph, src, plan, controls, reader, live, standin, runtime)
                    if kind == "F" and tracked:
                        f_pool = k42.snapshot(pool)
                        fresh = controls_fresh(plan, controls)
                        pool.copy_slots(controls[0], fresh)  # F's COW reference, after F's pool snapshot
                        out["F"]["cow_rows"] = rows(pool, fresh)
                        k42.restore(pool, f_pool)
                        del f_pool
                    if kind in ("F", "J", "JR"):
                        out[kind]["pool"] = k42.snapshot(pool) if kind == "F" else {
                            k: k42.compare(out["F"]["pool"][k], v) for k, v in k42.pool_tensors(pool).items()}
                f = out["F"]
                row = dict(tracked=tracked, batch=b, rep=rep, seed=seed)
                for kind in ("J", "JR"):
                    r = out[kind]
                    row[kind] = dict(
                        G1_decode_out=k42.compare(f["decode_out"], r["decode_out"]),
                        G2_decode_rows=all_equal(eq_dict(f["decode_rows"], r["decode_rows"])),
                        G3_pool=all_equal(r["pool"]),
                        G3_ckpt_rows=all_equal(eq_dict(f["ckpt_rows"], r["ckpt_rows"])),
                        normal_joins=r["stats"].get("normal_joins", 0), whole_joins=r["stats"]["joins"])
                    row[kind]["pass"] = (row[kind]["G1_decode_out"]["equal"] and row[kind]["G2_decode_rows"]
                                         and row[kind]["G3_pool"] and row[kind]["G3_ckpt_rows"]
                                         and row[kind]["normal_joins"] == 1)
                if tracked:
                    row["COW"] = dict(G4_cow_rows=all_equal(eq_dict(f["cow_rows"], out["COW"]["cow_rows"])),
                                      G1_decode_out=k42.compare(f["decode_out"], out["COW"]["decode_out"])["equal"])
                    row["COW"]["pass"] = row["COW"]["G4_cow_rows"] and row["COW"]["G1_decode_out"]
                row["n1_detected"] = (not all_equal(eq_dict(f["ckpt_rows"], out["n1"]["ckpt_rows"]))
                                      if tracked else None)
                row["n2_detected"] = not k42.compare(f["decode_out"], out["n2"]["decode_out"])["equal"]
                row["publication_wrote"] = not torch.equal(k42.bits(init["U"][:, plan.slots]),
                                                           k42.bits(f["pool"]["U"][:, plan.slots]))
                row["pass"] = (row["J"]["pass"] and row["JR"]["pass"] and (row["COW"]["pass"] if tracked else True)
                               and row["n2_detected"] and (row["n1_detected"] if tracked else True)
                               and row["publication_wrote"])
                ok &= row["pass"]
                print(f"tracked={tracked} b{b} rep{rep}: J {row['J']['pass']} JR {row['JR']['pass']}"
                      + (f" COW {row['COW']['pass']}" if tracked else "")
                      + f" | n1 detected {row['n1_detected']} n2 detected {row['n2_detected']}"
                      + f" -> {'PASS' if row['pass'] else 'FAIL'}", flush=True)
                result["reps"].append(row)
                k42.restore(pool, init)
                del init, out
                torch.cuda.empty_cache()
                result["pass"] = ok
                with open(args.out, "w") as fh:
                    json.dump(result, fh, indent=1, default=str)
    result["pass"] = ok
    result["meta"]["wall_s"] = time.time() - t_start
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(f"{'PASS' if ok else 'FAIL'}: {len(result['reps'])} reps, result {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
