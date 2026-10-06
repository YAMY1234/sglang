"""PDL must preserve each committed prefix, checkpoint and ring under graph reuse.

Guards moving a dependent load above the wait, missing an early-return trigger,
changing a recurrent reduction, or advancing/recycling a cursor before its
checkpoint consumer finishes. Compare the same implementation with PDL off/on;
the BF16 bound is one ULP, integer bookkeeping and FP32 rings are exact.
"""

import json
import time
import unittest

import torch
import triton
import triton.language as tl
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
    commit_gdn_replayssm_spec,
    gdn_replayssm_spec_decode,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@triton.jit
def _producer(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    # Consumer can launch while these stores are still pending. Its wait, not
    # this launch signal, must establish data visibility.
    tl.extra.cuda.gdc_launch_dependents()
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + x, tl.load(src + x, x < N, other=0), x < N)


def _difference(a, b):
    unequal = int((a != b).sum().item())
    if a.dtype != torch.bfloat16:
        return {
            "unequal": unequal,
            "max_ulp": 0,
            "nonfinite_mismatch": 0,
            "passed": unequal == 0,
        }
    finite = torch.isfinite(a) & torch.isfinite(b)
    ai, bi = a.view(torch.int16).int(), b.view(torch.int16).int()
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    distance = torch.where(finite, (ai - bi).abs(), 0)
    ulp = int(distance.max().item())
    mismatch = int(
        (
            (torch.isnan(a) != torch.isnan(b))
            | (torch.isposinf(a) != torch.isposinf(b))
            | (torch.isneginf(a) != torch.isneginf(b))
        )
        .sum()
        .item()
    )
    return {
        "unequal": unequal,
        "max_ulp": ulp,
        "nonfinite_mismatch": mismatch,
        "passed": ulp <= 1 and mismatch == 0,
    }


class ReplayCase:
    def __init__(self, batch, draft, exact=False, fp32=False):
        self.B, self.T, self.L = batch, draft, 32
        self.H, self.HV, self.K, self.V = 2, 8, 128, 128
        self.layers, self.slots = 2, 2 * batch + 3
        self.exact = exact
        self.residual = not fp32 or exact
        torch.manual_seed(42 + batch)
        device, dtype = "cuda", torch.bfloat16
        self.source = {
            name: torch.randn(batch * draft, heads, dim, device=device, dtype=dtype)
            for name, heads, dim in (
                ("q", self.H, self.K),
                ("k", self.H, self.K),
                ("v", self.HV, self.V),
            )
        }
        self.source.update(
            {
                n: torch.randn(batch * draft, self.HV, device=device, dtype=dtype)
                for n in ("a", "b")
            }
        )
        self.gating = {
            "A_log": torch.linspace(-2, 1, self.HV, device=device),
            "dt_bias": torch.zeros(self.HV, device=device),
        }
        self.initial = (
            torch.randn(
                self.layers,
                self.slots,
                self.HV,
                self.V,
                self.K,
                device=device,
                dtype=torch.float32 if fp32 else dtype,
            )
            * 0.1
        )
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
        for enabled in (False, True):

            def zeros(*shape, dt=dtype):
                return torch.zeros(*shape, device=device, dtype=dt)

            p = {
                "state": self.initial.clone(),
                "d": zeros(self.layers, self.slots, self.HV, self.L, self.V),
                "k": zeros(self.layers, self.slots, self.H, self.L, self.K),
                "g": zeros(self.layers, self.slots, self.HV, self.L, dt=torch.float32),
                "dr": zeros(self.layers, self.slots, self.HV, self.L, self.V),
                "kr": zeros(self.layers, self.slots, self.H, self.L, self.K),
                "beta": zeros(
                    self.layers, self.slots, self.HV, self.L, dt=torch.float32
                ),
                "wp": zeros(self.slots, dt=torch.int32),
                "base": zeros(self.slots, dt=torch.int32),
                "flush": zeros(self.slots, dt=torch.int8),
                "out": zeros(self.layers, batch * draft, self.HV, self.V),
                "inputs": {n: t.clone() for n, t in self.source.items()},
                "enabled": enabled,
            }
            self.paths.append(p)

    def step(self, p):
        for name, source in self.source.items():
            _producer[(triton.cdiv(source.numel(), 256),)](
                source, p["inputs"][name], source.numel(), 256
            )
        for layer in range(self.layers):
            gdn_replayssm_spec_decode(
                **p["inputs"],
                **self.gating,
                checkpoint_state=p["state"][layer],
                d_cache=p["d"][layer],
                k_cache=p["k"][layer],
                g_cache=p["g"][layer],
                rawv_cache=p["dr"][layer] if self.residual else None,
                rawk_cache=p["kr"][layer] if self.residual else None,
                beta_cache=p["beta"][layer] if self.exact else None,
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
                launch_mode="both" if self.exact else "verify",
            )
        commit_gdn_replayssm_spec(
            p["wp"],
            p["base"],
            p["flush"],
            self.accept,
            self.rid,
            max_cache_len=self.L,
            max_spec_len=self.T,
            fold_every_commit=p["state"].dtype != torch.float32,
            null_block_id=-1,
        )
        if not self.exact:
            commit_gdn_replayssm_circular(
                checkpoint_state=p["state"],
                d_cache=p["d"],
                k_cache=p["k"],
                g_cache=p["g"],
                d_residual_cache=p["dr"] if self.residual else None,
                k_residual_cache=p["kr"] if self.residual else None,
                state_batch_indices=self.sid,
                replay_indices=self.rid,
                write_pos=p["wp"],
                cache_base=p["base"],
                is_flush=p["flush"],
                accept_lens=self.accept,
                mamba_track_indices=self.track,
                mamba_steps_to_track=self.steps,
                null_block_id=-1,
            )

    def reset(self):
        for p in self.paths:
            p["state"].copy_(self.initial)
            for name in (
                "d",
                "k",
                "g",
                "dr",
                "kr",
                "beta",
                "wp",
                "base",
                "flush",
                "out",
            ):
                p[name].zero_()

    def run(self, rounds):
        graphs = []
        for p in self.paths:
            with envs.SGLANG_ENABLE_GDN_REPLAYSSM_PDL.override(p["enabled"]):
                self.step(p)
        torch.cuda.synchronize()
        self.reset()
        for p in self.paths:
            graph = torch.cuda.CUDAGraph()
            with (
                envs.SGLANG_ENABLE_GDN_REPLAYSSM_PDL.override(p["enabled"]),
                torch.cuda.graph(graph),
            ):
                self.step(p)
            graphs.append(graph)
        self.reset()
        stats = {}
        start = time.monotonic()
        for step in range(rounds):
            ids = torch.arange(1, self.B + 1, device="cuda").roll(step % self.B)
            self.sid.copy_(ids)
            self.rid.copy_(ids - 1)
            accepts = (torch.arange(self.B, device="cuda") + step) % (self.T + 1)
            self.accept.copy_(accepts)
            self.steps.copy_(
                torch.where(
                    (step + torch.arange(self.B, device="cuda")) % 7 == 0,
                    accepts - 1,
                    -1,
                )
            )
            if step % 17 == 0:
                self.sid[-1] = self.rid[-1] = -1
            for tensor in self.source.values():
                tensor.normal_()
            for graph in graphs:
                graph.replay()
            a, b = self.paths
            for key in (
                "out",
                "state",
                "d",
                "k",
                "g",
                "dr",
                "kr",
                "beta",
                "wp",
                "base",
                "flush",
            ):
                result = _difference(a[key], b[key])
                record = stats.setdefault(
                    key, {"max_ulp": 0, "unequal": 0, "nonfinite_mismatch": 0}
                )
                record["max_ulp"] = max(record["max_ulp"], result["max_ulp"])
                record["unequal"] += result["unequal"]
                record["nonfinite_mismatch"] += result["nonfinite_mismatch"]
                if not result["passed"]:
                    return {
                        "passed": False,
                        "step": step,
                        "key": key,
                        "stats": stats,
                        "difference": result,
                    }
            if step % 31 == 30:
                # Simulate tracked-state restoration and request-slot recycling:
                # state slots and replay slots have different identities.
                for p in self.paths:
                    p["state"][:, 1].copy_(p["state"][:, self.B + 1])
                    p["wp"][0] = p["base"][0] = p["flush"][0] = 0
        return {
            "passed": True,
            "batch": self.B,
            "draft": self.T,
            "exact": self.exact,
            "dtype": str(self.initial.dtype),
            "rounds": rounds,
            "layers": self.layers,
            "stats": stats,
            "seconds": time.monotonic() - start,
            "cuda_graph": True,
        }


class TestGdnReplayssmPdl(CustomTestCase):
    def test_graph_prefix_state_and_recycling(self):
        if not is_arch_support_pdl():
            self.skipTest("Programmatic dependent launch requires SM90+")
        for batch, draft, exact, fp32 in (
            (1, 7, False, False),
            (8, 7, False, False),
            (16, 8, False, False),
            (3, 7, False, True),
            (3, 7, True, True),
        ):
            with self.subTest(batch=batch, draft=draft, exact=exact, fp32=fp32):
                result = ReplayCase(batch, draft, exact, fp32).run(256)
                print("GDN_PDL_NUMERICS " + json.dumps(result), flush=True)
                self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
