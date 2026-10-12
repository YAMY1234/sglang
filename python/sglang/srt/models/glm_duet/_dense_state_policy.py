"""The DUET recurrent-state cut on SGLang's dense fp32 Mamba pool, for every hooked family (Kimi-Linear KDA, Lightning Mamba-2;
GLM-5.3-Flash's KDA cut subclasses it, models/glm_duet/state_policy.py).

The pool keeps the exact state between cuts; at the prompt end (cold) and every W decode steps (warm) each due (layer, slot)
state is replaced by its explicit sink plus rank-r content (state_factor.project_state_batched).  Between cuts this is the
factored form of the reference exactly, because a step only scales the state and adds one rank-1 write.  The cut runs outside
CUDA-graph replay, after the step whose output it follows, so it affects the next token as in the reference.
"""
import os
import time

import torch

from sglang.srt.layers.attention.linear.kda_state_prune import KDAStatePolicy

from ._state_factor import project_state_batched

# SGLANG_DUET_HOOK_PROFILE=1: wall time of the after-step hook (host syncs + cuts) per decode step, printed every 500 steps
_HOOK_PROFILE = os.environ.get("SGLANG_DUET_HOOK_PROFILE", "0") not in ("", "0")


class DenseStatePolicy(KDAStatePolicy):
    """Slot lifecycle and cut scheduling of KDAStatePolicy, the reference current-state sink and a warm basis per (layer, slot).

    side: the sink's side in the cut layout (B, H, P, N), "right" (KDA / GDN, direction (H, N)) or "left" (Mamba-2,
    direction (H, P)).  pool_transposed: the pool stores the cut layout transposed (KDA's (H, V, K)).  Functional acceptance
    fixes B=1, so the random subspace initialisation has the shape and seed of the B=1 reference."""

    def __init__(self, pool, directions, spec, layer_ids, *, side, pool_transposed, tag):
        self.warm = {}
        self.omega_heads = None     # (offset, total): a TP rank's head slice draws the unsharded reference's sketch
        super().__init__(pool, {l: directions[l] for l in layer_ids}, spec["state_rank"], max(1, spec["state_every"]),
                         prefix_only=spec["state_every"] == 0)
        self.explicit_sink = spec.get("state_sink", "explicit") == "explicit"
        self.side, self.pool_transposed, self.tag = side, pool_transposed, tag
        self.hook_s, self.hook_steps = 0.0, 0

    def _cut_many(self, plan):
        # One batched projection for every due (layer, slot) pair of the step, each row with its layer's direction; Omega
        # is the B=1 draw shared by all rows, so each row is the per-slot reference up to fp32 rounding (batched vs single
        # GEMM / eigh), see state_factor.project_state_batched.  Two host syncs per step (pairs, pending flags).
        layers = torch.cat([torch.full_like(slots, layer_index) for layer_index, slots in plan])
        slots = torch.cat([slots for _, slots in plan])
        pairs = list(zip(layers.tolist(), slots.tolist()))
        pending = self.pending_prefix[layers, slots].tolist()
        states = self.states[layers, slots]
        if self.pool_transposed:
            states = states.transpose(-1, -2)
        prevs = [None if fresh else self.warm.get(key) for key, fresh in zip(pairs, pending)]
        projected, bases = project_state_batched(states, self.vbar[layers], self.rank, prevs, explicit=self.explicit_sink,
                                                 side=self.side, omega_heads=self.omega_heads)
        self.states[layers, slots] = projected.transpose(-1, -2) if self.pool_transposed else projected
        for key, basis in zip(pairs, bases):
            if basis is None:
                self.warm.pop(key, None)
            else:
                self.warm[key] = basis

    def note_prompt_end(self, slots):
        """The prompt-final states of `slots` are in the pool (prefill or emitters): cut them once, cold, before any decode step."""
        self.pending_prefix[:, slots] = True
        self.count[:, slots] = 0
        self.flush(slots, prefix=True)

    def reset_slots(self, indices):
        super().reset_slots(indices)
        selected = set(torch.as_tensor(indices).reshape(-1).tolist())
        self.warm = {k: v for k, v in self.warm.items() if k[1] not in selected}

    def copy_slots(self, src, dst):
        super().copy_slots(src, dst)
        sources = torch.as_tensor(src).reshape(-1).tolist()
        targets = torch.as_tensor(dst).reshape(-1).tolist()
        saved = {(layer, target): self.warm.get((layer, source))
                 for source, target in zip(sources, targets) for layer in self.layer_map.values()}
        for key, value in saved.items():
            if value is None:
                self.warm.pop(key, None)
            else:
                self.warm[key] = value.clone()

    def get_cpu_slots(self, indices):
        selected = torch.as_tensor(indices).reshape(-1).tolist()
        bases = {(layer, i): self.warm[(layer, slot)].cpu()
                 for i, slot in enumerate(selected) for layer in self.layer_map.values()
                 if (layer, slot) in self.warm}
        return super().get_cpu_slots(indices), bases

    def load_cpu_slots(self, data, indices):
        self.reset_slots(indices)
        ordinary, bases = data
        super().load_cpu_slots(ordinary, indices)
        selected = torch.as_tensor(indices).reshape(-1).tolist()
        for (layer, i), value in bases.items():
            self.warm[(layer, selected[i])] = value.to(self.states.device)

    def after_graph(self, batch):
        """A decode step of `batch` ran (replayed by the runner, or eager): every layer of every request advanced by one
        token, so count it here and then apply the due cuts -- count, then flush(prefix=False) after the step's output."""
        profile = _HOOK_PROFILE
        if profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        slots = self.slots(batch)
        valid = slots[slots >= 0]
        self.count[:, valid] += 1
        self.flush(valid, prefix=False)
        if profile:
            torch.cuda.synchronize()
            self.hook_s += time.perf_counter() - t0
            self.hook_steps += 1
            if self.hook_steps % 500 == 0:
                print(f"DUET-HOOK {self.tag} steps={self.hook_steps} bs={batch.batch_size} hook_s={self.hook_s:.1f} "
                      f"ms/step={1000 * self.hook_s / self.hook_steps:.2f}", flush=True)
