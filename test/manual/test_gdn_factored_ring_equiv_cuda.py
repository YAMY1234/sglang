"""K5-ring module-level bitwise gate (docs/224 §3.3): ring 16 vs ring 2 on the real pool and kernels.

Served dumps are not run-to-run reproducible on the gate recipe (docs/224 §3.2, CMH 4527109), so the ring-size
equivalence is checked where it can be deterministic: the same cloned inputs drive two FactoredGDNPool instances
(BF16 TP2 rank shapes: 36 layers, HV 24, K = V = 128, R16/W16 k31 fp16, strict_chunk, factored_prefix) through the
host planner (plan_extend), exact continuation reads (initial_dense), k31 commits with ring writes (commit_extend) and
64 factored decode steps with expiry truncation (post order).  Everything except the ring position must be byte-equal:
initial states, committed states, a/U/W/count/stale/prefix_valid/dense_required, which slots own an exact copy, and the
decode outputs.  Scenarios: serial chunking (one unfinished prompt per forward, as the scheduler serves), and bursts of
8 and 17 simultaneously unfinished prompts (ring 2 must grow; ring 16 grows only for 17).
"""
import hashlib
import json
import os
import unittest
from types import SimpleNamespace

import torch

L, HV, H, D = 36, 24, 8, 128
FIELDS = ("a", "U", "W", "count", "stale", "dense_required", "prefix_valid")


# k31 needs the release's explicit sink directions (heads x V per layer id); the served BF16 v3.5-b-r2 file by default.
VBAR = os.environ.get("K5RING_VBAR", "/work/results/r16-agg-20261014/releases/duet-flash-next-v3.5-b-r2/.cache/state_sink_vbar.pt")


def make_pool(ring, device="cuda"):
    from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool

    cfg = FactoredGDNConfig.parse(
        f"r=16,m=16,dtype=fp16,ring={ring},async=1,strict_chunk=1,init_method=k31,decode_method=iter,"
        f"vbar={VBAR},factored_prefix=1")
    return FactoredGDNPool(size=64, max_running_requests=96,
                           cache_params=SimpleNamespace(shape=SimpleNamespace(temporal=(HV, D, D))),
                           mamba_layer_ids=list(range(L)), device=device, cfg=cfg)


def digest(t):
    t = t.to(torch.uint8) if t.dtype == torch.bool else t
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def scenarios():
    """Lists of forwards; a row = (slot, prompt_final, prefix_len)."""
    serial, slot = [], 1
    for p in range(6):  # one chunked prompt at a time + two one-chunk prompts completing alongside
        long_slot = slot; slot += 1
        for chunk in range(3):
            rows = [(long_slot, chunk == 2, chunk * 32768)]
            rows += [(slot + j, True, 0) for j in range(2)]; slot += 2
            serial.append(rows)
    bursts = {}
    for width in (8, 17):
        base = 1
        bursts[f"burst{width}"] = [[(base + i, chunk == 2, chunk * 32768) for i in range(width)] for chunk in range(3)]
    return {"serial": serial, **bursts}


def run(ring, forwards, seed):
    from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_decode

    torch.manual_seed(seed)
    pool = make_pool(ring)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    trace = []
    for rows in forwards:
        slots = torch.tensor([r[0] for r in rows], device="cuda")
        fresh = [r[0] for r in rows if r[2] == 0]
        if fresh:  # the allocator resets every newly admitted slot
            pool.reset_slots(torch.tensor(fresh, device="cuda"))
        plan = pool.plan_extend(slots, [64] * len(rows), prefix_lens=[r[2] for r in rows],
                                prompt_final=[r[1] for r in rows])
        for li in range(L):
            initial = pool.initial_dense(li, plan)
            final = initial * 0.97 + 0.05 * torch.randn(initial.shape, device="cuda", generator=gen)
            pool.commit_extend(li, plan, final)
            if li in (0, L - 1):
                trace.append((digest(initial), digest(final)))
        trace.append(("use_ring", plan.use_ring.tolist(), plan.n_ring_src))
    torch.cuda.synchronize()
    finished = sorted({r[0] for rows in forwards for r in rows if r[1]})
    slots = torch.tensor(finished, device="cuda")
    B = slots.numel()
    A_log = torch.randn(HV, device="cuda", generator=gen) * 0.1
    dt_bias = torch.randn(HV, device="cuda", generator=gen) * 0.1
    outputs = []
    for step in range(64):
        for li in range(L):
            mixed = torch.randn(B, 2 * H * D + HV * D, device="cuda", generator=gen).bfloat16()
            a = torch.randn(B, HV, device="cuda", generator=gen).bfloat16()
            b = torch.randn(B, HV, device="cuda", generator=gen).bfloat16()
            out = factored_packed_decode(mixed, a, b, A_log=A_log, dt_bias=dt_bias, scale=D ** -0.5,
                                         vbar=pool.vbar[li], fa=pool.a[li], fu=pool.U[li], fw=pool.W[li],
                                         fcount=pool.count[li], stale=pool.stale, ssm_state_indices=slots,
                                         num_q_heads=H, num_v_heads=HV, head_k_dim=D, head_v_dim=D,
                                         r=pool.cfg.r, rfull=pool.cfg.rfull, **pool.cfg.kernel_kwargs())
            if li in (0, L - 1) or step == 63:
                outputs.append(digest(out))
    torch.cuda.synchronize()
    state = {name: digest(getattr(pool, name)) for name in FIELDS if getattr(pool, name) is not None}
    owners = (pool.dense_of >= 0).int()
    return dict(trace=trace, outputs=outputs, state=state, owners=digest(owners), ring=pool.dense_ring.shape[1],
                grown=pool.stats.get("ring_grown", 0), ring_src=pool.stats["ring_src"], ring_miss=pool.stats["ring_miss"])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class RingEquivalenceCUDA(unittest.TestCase):
    def test_ring_two_is_bitwise_ring_sixteen(self):
        report = {}
        wanted = os.environ.get("K5RING_SCENARIOS")  # comma list; one scenario per GPU in the job
        expected_grown = {"serial": (0, 0), "burst8": (0, 1), "burst17": (1, 1)}
        expected_ring = {"serial": (16, 2), "burst8": (16, 8), "burst17": (17, 17)}
        for name, forwards in scenarios().items():
            if wanted and name not in wanted.split(","):
                continue
            ref = run(16, forwards, 224)
            rep = run(16, forwards, 224)  # determinism control: same arm twice
            cand = run(2, forwards, 224)
            report[name] = dict(control_equal=ref == rep, ring16=ref["ring"], ring2=cand["ring"],
                                grown16=ref["grown"], grown2=cand["grown"],
                                ring_src=(ref["ring_src"], cand["ring_src"]), ring_miss=(ref["ring_miss"], cand["ring_miss"]),
                                trace_equal=ref["trace"] == cand["trace"], outputs_equal=ref["outputs"] == cand["outputs"],
                                state_equal=ref["state"] == cand["state"],
                                # informational: which completed rows got OPTIONAL retention differs by design
                                optional_retention_equal=ref["owners"] == cand["owners"])
            print("K5RING_EQUIV " + json.dumps({name: report[name]}), flush=True)
            self.assertEqual(ref, rep, f"{name}: same-arm repeat is not deterministic; the gate is void")
            for key in ("trace", "outputs", "state", "ring_src"):
                self.assertEqual(ref[key], cand[key], f"{name}: ring 2 differs from ring 16 in {key}")
            # one growth call per forward deficit: burst8 ring 2 -> 8; burst17 ring 2 -> 17 and ring 16 -> 17
            self.assertEqual((report[name]["grown16"], report[name]["grown2"]), expected_grown[name])
            self.assertEqual((report[name]["ring16"], report[name]["ring2"]), expected_ring[name])
        self.assertTrue(report, "no scenario selected")


if __name__ == "__main__":
    unittest.main()
