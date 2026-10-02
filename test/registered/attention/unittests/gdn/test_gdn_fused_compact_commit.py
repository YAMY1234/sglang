"""Compare deferred BF16 compact folds against the eager production path."""

import argparse
import json
import time
from pathlib import Path

import torch
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
    commit_gdn_replayssm_spec,
    gdn_replayssm_spec_decode,
)


def ulp_stats(a, b):
    finite = torch.isfinite(a) & torch.isfinite(b)
    bits_a = a.view(torch.int16).to(torch.int32)
    bits_b = b.view(torch.int16).to(torch.int32)
    ordered_a = torch.where(bits_a < 0, -32768 - bits_a, bits_a)
    ordered_b = torch.where(bits_b < 0, -32768 - bits_b, bits_b)
    ulp = (ordered_a - ordered_b).abs()[finite]
    return {
        "max_ulp": int(ulp.max().item()) if ulp.numel() else 0,
        "unequal": int(((a != b) & finite).sum().item()),
        "nonfinite_mismatch": int(
            (
                (torch.isnan(a) != torch.isnan(b))
                | (torch.isposinf(a) != torch.isposinf(b))
                | (torch.isneginf(a) != torch.isneginf(b))
            )
            .sum()
            .item()
        ),
        "elements": a.numel(),
    }


class Case:
    def __init__(self, batch, draft, seed, layers=2):
        self.B, self.T, self.L = batch, draft, 32
        self.H, self.HV, self.K, self.V = 16, 64, 128, 128
        self.layers, self.slots = layers, 2 * batch + 3
        torch.manual_seed(seed)
        device, dtype = "cuda", torch.bfloat16

        def rand(*shape, scale=1):
            return torch.randn(*shape, device=device, dtype=dtype) * scale

        self.inputs = {
            name: rand(batch * draft, heads, dim)
            for name, heads, dim in (
                ("q", self.H, self.K),
                ("k", self.H, self.K),
                ("v", self.HV, self.V),
            )
        }
        self.inputs.update(
            a=rand(batch * draft, self.HV), b=rand(batch * draft, self.HV)
        )
        self.inputs.update(
            A_log=torch.linspace(-2, 1, self.HV, device=device),
            dt_bias=torch.zeros(self.HV, device=device),
        )
        self.initial = rand(layers, self.slots, self.HV, self.V, self.K, scale=0.1)
        self.sid = torch.arange(1, batch + 1, device=device, dtype=torch.int64)
        self.rid = torch.arange(batch, device=device, dtype=torch.int64)
        self.track = torch.arange(
            batch + 1, 2 * batch + 1, device=device, dtype=torch.int64
        )
        self.steps = torch.zeros(batch, device=device, dtype=torch.int64)
        self.accept = torch.ones(batch, device=device, dtype=torch.int32)
        self.qsl = torch.arange(
            0, batch * draft + 1, draft, device=device, dtype=torch.int32
        )
        self.paths = []
        for fused in (False, True):

            def zeros(*shape, dt=dtype):
                return torch.zeros(*shape, device=device, dtype=dt)

            p = dict(
                state=self.initial.clone(),
                d=zeros(layers, self.slots, self.HV, self.L, self.V),
                k=zeros(layers, self.slots, self.H, self.L, self.K),
                g=zeros(layers, self.slots, self.HV, self.L, dt=torch.float32),
                dr=zeros(layers, self.slots, self.HV, self.L, self.V),
                kr=zeros(layers, self.slots, self.H, self.L, self.K),
                wp=zeros(self.slots, dt=torch.int32),
                base=zeros(self.slots, dt=torch.int32),
                flush=zeros(self.slots, dt=torch.int8),
                out=zeros(layers, batch * draft, self.HV, self.V),
                fused=fused,
            )
            self.paths.append(p)

    def commit(self, p, track=True, track_only=None):
        commit_gdn_replayssm_circular(
            checkpoint_state=p["state"],
            d_cache=p["d"],
            k_cache=p["k"],
            g_cache=p["g"],
            d_residual_cache=p["dr"],
            k_residual_cache=p["kr"],
            state_batch_indices=self.sid,
            replay_indices=self.rid,
            write_pos=p["wp"],
            cache_base=p["base"],
            is_flush=p["flush"],
            accept_lens=self.accept,
            mamba_track_indices=self.track if track else None,
            mamba_steps_to_track=self.steps if track else None,
            track_only=p["fused"] if track_only is None else track_only,
        )

    def step(self, p):
        for layer in range(self.layers):
            gdn_replayssm_spec_decode(
                **self.inputs,
                checkpoint_state=p["state"][layer],
                d_cache=p["d"][layer],
                k_cache=p["k"][layer],
                g_cache=p["g"][layer],
                rawv_cache=p["dr"][layer],
                rawk_cache=p["kr"][layer],
                beta_cache=None,
                out=p["out"][layer],
                query_start_loc=self.qsl,
                ssm_state_indices=self.sid,
                replay_indices=self.rid,
                write_pos=p["wp"],
                cache_base=p["base"],
                is_flush=p["flush"],
                max_cache_len=self.L,
                max_spec_len=self.T,
                null_block_id=-1,
                launch_mode="verify",
                fuse_commit=p["fused"],
            )
        commit_gdn_replayssm_spec(
            p["wp"],
            p["base"],
            p["flush"],
            self.accept,
            self.rid,
            max_cache_len=self.L,
            max_spec_len=self.T,
            fold_every_commit=True,
            null_block_id=-1,
        )
        self.commit(p)

    def reset(self):
        for p in self.paths:
            p["state"].copy_(self.initial)
            for name in ("d", "k", "g", "dr", "kr", "wp", "base", "flush", "out"):
                p[name].zero_()

    def logical_state(self, p):
        if not p["fused"]:
            return p["state"]
        shadow = dict(p)
        for name in ("state", "wp", "base", "flush"):
            shadow[name] = p[name].clone()
        self.commit(shadow, track=False, track_only=False)
        return shadow["state"]

    def run(self, rounds):
        # Warm both launch specializations, then capture identical buffer lifetimes.
        for p in self.paths:
            self.step(p)
        torch.cuda.synchronize()
        self.reset()
        graphs = []
        for p in self.paths:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.step(p)
            graphs.append(graph)
        self.reset()
        stats = {
            key: {"max_ulp": 0, "unequal": 0, "elements": 0, "nonfinite_mismatch": 0}
            for key in ("output", "state", "track")
        }
        first_fail = None
        start = time.monotonic()
        for step in range(rounds):
            # Every prefix, wraparound, batch permutations, and padded request slots.
            accepts = (torch.arange(self.B, device="cuda") + step) % self.T + 1
            if step % 64 == 31:
                accepts.zero_()
            self.accept.copy_(accepts)
            self.steps.copy_(
                torch.where(
                    (step + torch.arange(self.B, device="cuda")) % 7 == 0,
                    accepts - 1,
                    -1,
                )
            )
            ids = torch.arange(1, self.B + 1, device="cuda").roll(step % self.B)
            self.sid.copy_(ids)
            self.rid.copy_(ids - 1)
            if step % 17 == 0:
                self.sid[-1] = -1
                self.rid[-1] = -1
            for name in ("q", "k", "v", "a", "b"):
                self.inputs[name].normal_()
            for g in graphs:
                g.replay()
            ref, cand = self.paths
            logical = self.logical_state(cand)
            pairs = {
                "output": (ref["out"], cand["out"]),
                "state": (ref["state"][:, 1 : self.B + 1], logical[:, 1 : self.B + 1]),
                "track": (
                    ref["state"][:, self.B + 1 : 2 * self.B + 1],
                    cand["state"][:, self.B + 1 : 2 * self.B + 1],
                ),
            }
            # A padded row can keep a previous pending state; materialize it separately before comparison.
            valid = self.sid[self.sid >= 0]
            pairs["state"] = (ref["state"][:, valid], logical[:, valid])
            for key, (a, b) in pairs.items():
                result = ulp_stats(a, b)
                stats[key]["max_ulp"] = max(stats[key]["max_ulp"], result["max_ulp"])
                for field in ("unequal", "elements", "nonfinite_mismatch"):
                    stats[key][field] += result[field]
                if (
                    result["max_ulp"] > 1 or result["nonfinite_mismatch"]
                ) and first_fail is None:
                    first_fail = dict(step=step, key=key, **result)
            if step % 64 == 63:
                # Restore a previously published snapshot into a new active slot (cache hit/eviction).
                for p in self.paths:
                    p["state"][:, 1].copy_(p["state"][:, self.B + 1])
                    p["wp"][0] = 0
                    p["base"][0] = 0
                    p["flush"][0] = 0
            if step % 128 == 127:
                print(
                    json.dumps(
                        {
                            "progress": step + 1,
                            "batch": self.B,
                            "draft": self.T,
                            "first_fail": first_fail,
                        }
                    ),
                    flush=True,
                )
        return {
            "batch": self.B,
            "draft": self.T,
            "rounds": rounds,
            "layers": self.layers,
            "stats": stats,
            "first_fail": first_fail,
            "pass": first_fail is None,
            "seconds": time.monotonic() - start,
            "cuda_graph": True,
        }


def test_fused_compact_commit():
    result = Case(3, 7, 234).run(64)
    assert result["pass"], result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1024)
    args = parser.parse_args()
    results = []
    for batch, draft, seed in ((1, 7, 234), (8, 7, 235), (16, 8, 236)):
        result = Case(batch, draft, seed).run(args.rounds)
        results.append(result)
        args.output.write_text(
            json.dumps(
                {"cases": results, "pass": all(x["pass"] for x in results)}, indent=2
            )
            + "\n"
        )
        print(json.dumps(result), flush=True)
    raise SystemExit(0 if all(x["pass"] for x in results) else 1)
