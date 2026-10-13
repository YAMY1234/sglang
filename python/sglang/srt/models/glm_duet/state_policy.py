"""The GLM-5.3-Flash DUET KDA state cut on SGLang's dense fp32 Mamba pool (torch only; twinstar docs/79).

GlmStatePolicy is sglang.srt.duet.dense_state_policy.DenseStatePolicy on one TP rank's head slice of GLM's KDA pool (right sink v-bar,
the pool stores (H, V, K), the random sketch drawn with all heads and sliced) with the cut schedule kept on the host.  The parent keeps
the schedule on the GPU and reads it back every decode step (3-4 host syncs per step); here a request's slot is known from its prompt
end, every slot of a decode batch advances one step and a slot is cut when its count reaches W -- the same cuts, after the same steps,
on the same rows in the same order (every resident layer, slots ascending) as the parent's flush, so the numbers are the parent's bit
for bit.  The host waits for the GPU only at a slot reset and inside a cut.
"""
import json
import os

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
    def __init__(self, pool, directions, spec, layer_ids, *, omega_heads, cut_log=None, align_prepare=None):
        super().__init__(pool, directions, spec, layer_ids, side="right", pool_transposed=True, tag="glm")
        self.omega_heads = omega_heads
        self.cut_log = cut_log
        self.slot_of = {}           # request pool index -> Mamba slot, set at the request's prompt end
        self.h_count = {}           # slot -> decode steps since its last cut
        self.h_pending = set()      # slots whose prompt-final states await the cold cut
        self.steps = {}             # slot -> decode steps since its prompt end (the cut log's step)
        self.align_prepare = (os.environ.get("SGLANG_GLM_ALIGN_PREPARE", "0") == "1"
                              if align_prepare is None else align_prepare)
        if self.align_prepare:
            self.workspace = _state_factor.ProjectionWorkspace()
            self.warm_slots = {}    # slot owns a contiguous (layers, heads, N, rank) tensor
            self.generation = {}    # physical slot's ownership epoch
            self.req_generation = {}
            self.request_epoch = {}
            self.owner_clock = 0

    def mark_prompt_end(self, req_pool_indices, slots):
        """The prompt-final states of these requests (host lists) are in the pool: cut them once, cold, before their boundary step."""
        for r, s in zip(req_pool_indices, slots):
            if self.align_prepare:
                r, s = int(r), int(s)
                if r < 0 or not self._valid_slot(s):
                    continue
                self.owner_clock += 1
                self.request_epoch[r] = self.owner_clock
                self.req_generation[r] = self.generation.get(s, 0)
            self.slot_of[int(r)] = int(s)
            self.h_pending.add(int(s))
            self.h_count[int(s)] = 0
            self.steps[int(s)] = 0

    def _batch_slots(self, batch):
        if self.align_prepare:
            return self._aligned_batch_slots(batch)
        rpi = batch.req_pool_indices_cpu
        if rpi is not None and len(rpi) >= batch.batch_size:
            out = [self.slot_of.get(int(r)) for r in rpi[:batch.batch_size].tolist()]
            if None not in out:
                return out
        return [s for s in self.slots(batch).tolist() if s >= 0]        # a request without a prompt end here: one GPU read

    def _valid_slot(self, slot):
        return 0 <= slot < self.states.shape[1]

    def _aligned_batch_slots(self, batch):
        rpi = getattr(batch, "req_pool_indices_cpu", None)
        if rpi is not None and len(rpi) >= batch.batch_size:
            requests = rpi[:batch.batch_size].tolist()
            slots = [self.slot_of.get(int(r)) for r in requests]
            if all(s is not None and self._valid_slot(s) and
                   self.req_generation.get(int(r)) == self.generation.get(s, 0)
                   for r, s in zip(requests, slots)):
                return list(dict.fromkeys(slots))
        # Preserve the existing contract for missing host metadata: resolve the
        # current device mapping, once per hook, instead of guessing/failing.
        return list(dict.fromkeys(s for s in self.slots(batch).tolist() if self._valid_slot(s)))

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
        if self.align_prepare:
            slots = sorted(set(s for s in slots if self._valid_slot(s)))
            if not slots:
                return
        pairs = [(i, s) for i in range(n_layers) for s in slots]
        make_index = self.workspace.index if self.align_prepare else _device_index
        layers = make_index([i for i, _ in pairs], dev)
        rows = make_index([s for _, s in pairs], dev)
        states = self.states[layers, rows].transpose(-1, -2)
        if self.align_prepare:
            prevs = [None if fresh or s not in self.warm_slots else self.warm_slots[s][i:i + 1]
                     for i, s in pairs]
        else:
            prevs = [None if fresh else self.warm.get(key) for key in pairs]
        projected, bases = _state_factor.project_state_batched(states, self.vbar[layers], self.rank, prevs, explicit=self.explicit_sink,
                                                               side=self.side, omega_heads=self.omega_heads,
                                                               workspace=self.workspace if self.align_prepare else None)
        self.states[layers, rows] = projected.transpose(-1, -2)
        if self.align_prepare:
            for j, s in enumerate(slots):
                owned = [bases[i * len(slots) + j] for i in range(n_layers)]
                if owned[0] is None:
                    self.warm_slots.pop(s, None)
                else:
                    self.warm_slots[s] = torch.cat(owned, dim=0).contiguous()
        else:
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
        if self.align_prepare:
            return self._aligned_reset(indices)
        super().reset_slots(indices)
        cleared = set(torch.as_tensor(indices).reshape(-1).tolist())
        for s in cleared:
            self.h_count.pop(s, None)
            self.h_pending.discard(s)
        self.slot_of = {r: s for r, s in self.slot_of.items() if s not in cleared}

    def copy_slots(self, src, dst):
        if self.align_prepare:
            return self._aligned_copy(src, dst)
        super().copy_slots(src, dst)
        for s, d in zip(torch.as_tensor(src).reshape(-1).tolist(), torch.as_tensor(dst).reshape(-1).tolist()):
            if s in self.h_count:
                self.h_count[d] = self.h_count[s]
            if s in self.h_pending:
                self.h_pending.add(d)

    def get_cpu_slots(self, indices):
        if not self.align_prepare:
            return super().get_cpu_slots(indices)
        selected = self._host_slots(indices)
        self._validate_transfer_slots(selected)
        index = self.workspace.index(selected, self.states.device)
        return {
            "glm_align_version": 1,
            "ordinary": tuple(x[:, index].cpu().clone() for x in (self.count, self.pending_prefix)),
            "slots": [self._snapshot_slot(s, cpu=True) for s in selected],
        }

    def load_cpu_slots(self, data, indices):
        if not self.align_prepare:
            return super().load_cpu_slots(data, indices)
        selected = self._host_slots(indices)
        self._validate_transfer_slots(selected)
        # A legacy checkpoint lacks host cadence/owner metadata. Recover its
        # device counts/pending flags; do not invent request ownership.
        if not isinstance(data, dict):
            ordinary, bases = data
            snapshots = []
            for i in range(len(selected)):
                owned = [bases.get((l, i)) for l in range(self.states.shape[0])]
                snapshots.append({"count": int(ordinary[0][0, i]),
                                  "pending": bool(ordinary[1][:, i].any()), "steps": 0,
                                  "generation": 0, "owners": [],
                                  "warm": torch.cat(owned, 0) if all(v is not None for v in owned) else None})
        else:
            if data.get("glm_align_version") != 1 or len(data["slots"]) != len(selected):
                raise ValueError("invalid GLM alignment checkpoint")
            ordinary, snapshots = data["ordinary"], data["slots"]
        # Validate the complete sibling payload before changing slot authority.
        expected = (self.states.shape[0], len(selected))
        if len(ordinary) != 2 or any(tuple(v.shape) != expected for v in ordinary):
            raise ValueError("invalid GLM checkpoint count/pending shape")
        if ordinary[0].dtype != self.count.dtype or ordinary[1].dtype != self.pending_prefix.dtype:
            raise ValueError("invalid GLM checkpoint count/pending dtype")
        warm_shape = (self.states.shape[0], self.states.shape[2], self.states.shape[3], self.rank)
        for snapshot in snapshots:
            value = snapshot["warm"]
            if value is not None and (tuple(value.shape) != warm_shape or value.dtype != torch.float32):
                raise ValueError("invalid GLM checkpoint warm basis")
            for key in ("count", "steps", "generation"):
                scalar = snapshot[key]
                if scalar is not None and (type(scalar) is not int or scalar < 0):
                    raise ValueError("invalid GLM checkpoint host metadata")
            if type(snapshot["pending"]) is not bool:
                raise ValueError("invalid GLM checkpoint pending flag")
        self._aligned_reset(selected)
        index = self.workspace.index(selected, self.states.device)
        for target, saved in zip((self.count, self.pending_prefix), ordinary):
            target[:, index] = saved.to(target.device)
        for s, snapshot in zip(selected, snapshots):
            self._restore_slot(s, snapshot, owners=True)

    def _host_slots(self, indices):
        return [int(s) for s in torch.as_tensor(indices).reshape(-1).tolist()]

    def _validate_transfer_slots(self, slots):
        if any(not self._valid_slot(s) for s in slots) or len(set(slots)) != len(slots):
            raise ValueError("invalid or duplicate GLM checkpoint slots")

    def _aligned_reset(self, indices):
        selected = sorted(set(s for s in self._host_slots(indices) if self._valid_slot(s)))
        index = self.workspace.index(selected, self.states.device)
        self.count[:, index] = 0
        self.pending_prefix[:, index] = False
        for s in selected:
            self.generation[s] = self.generation.get(s, 0) + 1
            self.h_count.pop(s, None)
            self.h_pending.discard(s)
            self.steps.pop(s, None)
            self.warm_slots.pop(s, None)
        for r, s in list(self.slot_of.items()):
            if s in selected:
                del self.slot_of[r]
                self.req_generation.pop(r, None)

    def _snapshot_slot(self, s, *, cpu=False):
        value = self.warm_slots.get(s)
        return {"count": self.h_count.get(s), "pending": s in self.h_pending,
                "steps": self.steps.get(s), "generation": self.generation.get(s, 0),
                "owners": [(r, self.request_epoch[r]) for r, slot in self.slot_of.items()
                           if slot == s and self.req_generation.get(r) == self.generation.get(s, 0)],
                "warm": None if value is None else (value.cpu().clone() if cpu else value.clone())}

    def _restore_slot(self, s, snapshot, *, owners):
        self.generation[s] = max(self.generation.get(s, 0), snapshot["generation"] + 1)
        if snapshot["count"] is not None:
            self.h_count[s] = snapshot["count"]
        if snapshot["steps"] is not None:
            self.steps[s] = snapshot["steps"]
        if snapshot["pending"]:
            self.h_pending.add(s)
        if snapshot["warm"] is not None:
            self.warm_slots[s] = snapshot["warm"].to(self.states.device).clone().contiguous()
        if owners:
            for r, epoch in snapshot["owners"]:
                # Do not steal a request re-admitted since the export.
                if r not in self.slot_of and self.request_epoch.get(r, epoch) == epoch:
                    self.slot_of[r] = s
                    self.req_generation[r] = self.generation[s]
                    self.request_epoch[r] = epoch
                    self.owner_clock = max(self.owner_clock, epoch)

    def _aligned_copy(self, src, dst):
        sources_host, targets_host = self._host_slots(src), self._host_slots(dst)
        if len(sources_host) != len(targets_host):
            raise ValueError("mismatched GLM copy slots")
        pairs = [(s, d) for s, d in zip(sources_host, targets_host)
                 if self._valid_slot(s) and self._valid_slot(d) and s != d]
        if len({d for _, d in pairs}) != len(pairs):
            raise ValueError("duplicate GLM copy destination")
        # Snapshot all sources before clearing any targets (also permits swaps).
        snapshots = [self._snapshot_slot(s) for s, _ in pairs]
        sources = self.workspace.index([s for s, _ in pairs], self.states.device)
        saved = tuple(x[:, sources].clone() for x in (self.count, self.pending_prefix))
        self._aligned_reset([d for _, d in pairs])
        targets = self.workspace.index([d for _, d in pairs], self.states.device)
        for target, value in zip((self.count, self.pending_prefix), saved):
            target[:, targets] = value
        for (_, d), snapshot in zip(pairs, snapshots):
            # Pool copies fork physical checkpoints; the source request remains
            # its owner. A destination's request mapping is resolved by fallback
            # until mark_prompt_end binds it, never copied onto a second owner.
            self._restore_slot(d, snapshot, owners=False)
