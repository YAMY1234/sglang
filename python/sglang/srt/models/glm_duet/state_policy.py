"""The GLM-5.3-Flash DUET KDA state cut on SGLang's dense fp32 Mamba pool (torch only; twinstar docs/79).

GlmStatePolicy is sglang.srt.duet.dense_state_policy.DenseStatePolicy on one TP rank's head slice of GLM's KDA pool (right sink v-bar,
the pool stores (H, V, K), the random sketch drawn with all heads and sliced) with the cut schedule kept on the host.  The parent keeps
the schedule on the GPU and reads it back every decode step (3-4 host syncs per step); here a request's slot is known from its prompt
end, every slot of a decode batch advances one step and a slot is cut when its count reaches W -- the same cuts, after the same steps,
on the same rows in the same order (every resident layer, slots ascending) as the parent's flush, so the numbers are the parent's bit
for bit.  The host waits for the GPU only at a slot reset and inside a cut.
"""
import json

import torch

from ._common import load as _load

from ._dense_state_policy import DenseStatePolicy
from . import _state_factor


def _device_index(values, device):
    """A long index tensor on `device` without a host wait (pinned staging) when the device is a GPU."""
    t = torch.tensor(values, dtype=torch.long)
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


class GlmStatePolicy(DenseStatePolicy):
    def __init__(self, pool, directions, spec, layer_ids, *, omega_heads, cut_log=None):
        super().__init__(pool, directions, spec, layer_ids, side="right", pool_transposed=True, tag="glm")
        self.omega_heads = omega_heads
        self.cut_log = cut_log
        self.slot_of = {}           # request pool index -> Mamba slot, set at the request's prompt end
        self.h_count = {}           # slot -> decode steps since its last cut
        self.h_pending = set()      # slots whose prompt-final states await the cold cut
        self.steps = {}             # slot -> decode steps since its prompt end (the cut log's step)

    def mark_prompt_end(self, req_pool_indices, slots):
        """The prompt-final states of these requests (host lists) are in the pool: cut them once, cold, before their boundary step."""
        for r, s in zip(req_pool_indices, slots):
            self.slot_of[int(r)] = int(s)
            self.h_pending.add(int(s))
            self.h_count[int(s)] = 0
            self.steps[int(s)] = 0

    def _batch_slots(self, batch):
        rpi = batch.req_pool_indices_cpu
        if rpi is not None and len(rpi) >= batch.batch_size:
            out = [self.slot_of.get(int(r)) for r in rpi[:batch.batch_size].tolist()]
            if None not in out:
                return out
        return [s for s in self.slots(batch).tolist() if s >= 0]        # a request without a prompt end here: one GPU read

    def before_graph(self, batch):
        due = sorted({s for s in self._batch_slots(batch) if s in self.h_pending})
        if due:
            self._cut(due, fresh=True)
            self.h_pending.difference_update(due)
            self.prefix_cuts += len(due) * self.states.shape[0]

    def after_graph(self, batch):
        due = []
        for s in self._batch_slots(batch):
            self.steps[s] = self.steps.get(s, 0) + 1
            if s in self.h_pending:
                continue
            c = self.h_count.get(s, 0) + 1
            self.h_count[s] = c
            if c >= self.every:
                due.append(s)
        if due and not self.prefix_only:
            due = sorted(set(due))
            self._cut(due, fresh=False)
            for s in due:
                self.h_count[s] = 0
            self.decode_cuts += len(due) * self.states.shape[0]

    def _cut(self, slots, *, fresh):
        """Every resident KDA layer of `slots`, layer-major and slots ascending (the rows of DenseStatePolicy._cut_many)."""
        n_layers, dev = self.states.shape[0], self.states.device
        pairs = [(i, s) for i in range(n_layers) for s in slots]
        layers = _device_index([i for i, _ in pairs], dev)
        rows = _device_index([s for _, s in pairs], dev)
        states = self.states[layers, rows].transpose(-1, -2)
        prevs = [None if fresh else self.warm.get(key) for key in pairs]
        projected, bases = _state_factor.project_state_batched(states, self.vbar[layers], self.rank, prevs, explicit=self.explicit_sink,
                                                               side=self.side, omega_heads=self.omega_heads)
        self.states[layers, rows] = projected.transpose(-1, -2)
        for key, basis in zip(pairs, bases):
            if basis is None:
                self.warm.pop(key, None)
            else:
                self.warm[key] = basis
        if self.cut_log is not None:
            with open(self.cut_log, "a") as f:
                for s in slots:
                    f.write(json.dumps({"slot": s, "kind": "prompt" if fresh else "decode", "step": self.steps.get(s, -1),
                                        "layers": n_layers}) + "\n")

    def reset_slots(self, indices):
        super().reset_slots(indices)
        cleared = set(torch.as_tensor(indices).reshape(-1).tolist())
        for s in cleared:
            self.h_count.pop(s, None)
            self.h_pending.discard(s)
        self.slot_of = {r: s for r, s in self.slot_of.items() if s not in cleared}

    def copy_slots(self, src, dst):
        super().copy_slots(src, dst)
        for s, d in zip(torch.as_tensor(src).reshape(-1).tolist(), torch.as_tensor(dst).reshape(-1).tolist()):
            if s in self.h_count:
                self.h_count[d] = self.h_count[s]
            if s in self.h_pending:
                self.h_pending.add(d)
