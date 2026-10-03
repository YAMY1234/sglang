"""P31-only factor publication: 2b tracked replay, or dense shallow tails + A.

A publishes F(S_N) for shallow layers and F(S_N-1) for deep emitters. D still
executes the deep boundary exactly once. Wire valid=2 tags this changed state
contract; the receiver must explicitly opt in before normalizing it to 1.
"""

import copy
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)
WIRE_FINAL_DEFERRED = 2


def graph_shape(normal, tracked):
    from .gdn_prefill_commit_graph import BATCH_BUCKETS

    bucket = next(b for b in BATCH_BUCKETS if max(normal, tracked) <= b)
    return (1 if normal == 1 else bucket, 1 if tracked == 1 else bucket)


def launch_side(side, plan, states, controls, *, final, eager, policy):
    """One T graph; A additionally runs F on the same stream, after T."""
    side.launch_pending(reason="next_bind")
    normal, tracked = graph_shape(plan.slots.numel(), controls[0].numel())
    key = side.whole_graph.key(normal, tracked, eager, policy, False)
    k = side.parity
    buffers, final_graph, tracked_graph = (side.alt_entries if k else side.entries)[key]
    current = torch.cuda.current_stream(side.pool.a.device)
    if side.set_recorded[k]:
        current.wait_event(side.set_done[k])
    buffers.bind(plan, states, *controls)
    if side.after_boundary:
        graphs = (tracked_graph, final_graph) if final else (tracked_graph,)
        side.defer_until_boundary(graphs, k, current)
        return
    # Normal per-layer commits (2b), or all dense shallow tails (A), precede this.
    side.final_done.record(current)
    side.stream.wait_event(side.final_done)
    side.done = side.set_done[k]
    with torch.cuda.stream(side.stream):
        tracked_graph.replay()
        if final:
            final_graph.replay()
        side.done.record(side.stream)
    side.set_recorded[k] = True
    side.parity = 1 - k
    side.recorded = side._prefill_side_pending = True
    side.waited_streams.clear()


class PDDeferredController:
    def __init__(self, pool, request_pool, state, *, final):
        self.pool, self.request_pool, self.state = pool, request_pool, state
        self.final = final
        self.stats = dict(published=0, fallback=0)

    def join(self):
        side = self.pool._tracked_factor_side
        if side is not None:
            side.join()

    def join_for_batch(self, batch):
        # P serial dispatch; conservatively protect final factors and ring reuse.
        self.join()

    def fallback(self, reason):
        self.stats["fallback"] += 1
        if self.stats["fallback"] == 1 or self.stats["fallback"] % 500 == 0:
            logger.info("pd_factor_deferred fallback=%s counts=%s", reason, self.stats)
        return None


class PDDeferredTransaction:
    def __init__(self, controller, batch, plan, metadata):
        self.controller, self.pool = controller, controller.pool
        self.batch, self.plan = batch, plan
        self.final = controller.final
        self.controls = (
            metadata.track_ssm_h_dst,
            metadata.track_ssm_final_src,
            metadata.track_ssm_final_dst,
        )
        self.states, self.tail_layers = [], set()
        self.published = False
        self.ids = torch.as_tensor(batch.req_pool_indices_cpu, dtype=torch.long).clone()
        rp = controller.request_pool
        if self.ids.device.type != "cpu" or rp.req_generation.device.type != "cpu":
            raise RuntimeError(
                "PD deferred publication requires host request generations"
            )
        self.generations = rp.req_generation[self.ids].clone()
        self.boundary_slots = None

    def __enter__(self):
        if getattr(self.pool, "_exact_tail_transaction", None) is not None:
            raise RuntimeError(
                "PD deferred publication conflicts with another transaction"
            )
        self.pool.pside_join()
        self.pool.invalidate_prefix_dense(self.controls[0])
        if self.final:
            self.pool.invalidate_prefix_dense(self.plan.slots)
        self.pool._exact_tail_transaction = self
        return self

    def prepare_split(self, prefix_ids, boundary_batch, boundary_metadata):
        if prefix_ids != list(map(int, boundary_batch.req_pool_indices_cpu)):
            raise RuntimeError("PD deferred prefix/boundary rows differ")
        if not torch.equal(
            self.plan.slots, boundary_metadata.mamba_cache_indices.long()
        ):
            raise RuntimeError("PD deferred prefix/boundary state slots differ")
        self.boundary_slots = boundary_metadata.mamba_cache_indices

    @contextmanager
    def ordinary(self):
        # Use the unchanged native per-layer commit/recurrent path for 2b.
        self.pool._exact_tail_transaction = None
        try:
            yield
        finally:
            self.pool._exact_tail_transaction = self

    def add(self, layer_id, plan, dense, tracked, tracks, sources, destinations):
        index = self.pool.layer_map[layer_id]
        if plan is not self.plan or index != len(self.states) or tracked is None:
            raise RuntimeError(
                "PD deferred prefix changed layer plan or checkpoint shape"
            )
        if any(
            a is not b for a, b in zip((tracks, sources, destinations), self.controls)
        ):
            raise RuntimeError("PD deferred checkpoint controls changed")
        # Preserve each layer before the next chunk kernel reuses its scratch.
        self.states.append(
            (
                dense.clone(memory_format=torch.contiguous_format),
                tracked.clone(memory_format=torch.contiguous_format),
            )
        )
        if not self.final:
            single = copy.copy(plan)
            single.next_layer = single.last_layer = index
            single.pending = []
            with self.ordinary():
                self.pool.commit_extend_batched(
                    layer_id, single, dense, None, None, None, None
                )
            if sources is not None and sources.numel():
                self.pool.copy_slots_layer(layer_id, sources, destinations)
            if single.pending or single.next_layer != index + 1:
                raise RuntimeError(
                    "PD live factors not committed before recurrent tail"
                )
        plan.next_layer += 1
        if len(self.states) == len(self.pool.layer_ids):
            self.publish()  # Last GDN: overlap remaining QSA/publication work.

    def decode(self, backend, layer, batch, mixed, a, b, conv, temporal, slots):
        index = self.pool.layer_map[layer.layer_id]
        if (
            layer.layer_id >= 31
            or index >= len(self.states)
            or layer.layer_id in self.tail_layers
        ):
            raise RuntimeError("PD shallow tail must run once after its own prefix")
        self.tail_layers.add(layer.layer_id)
        if not self.final:
            with self.ordinary():
                return backend._forward_decode_factored(
                    layer, batch, mixed, a, b, conv, temporal, slots
                )
        state = self.states[index][0]
        if self.boundary_slots is not slots:
            raise RuntimeError("PD dense tail changed its admitted state slots")
        # Same dispatcher/parameters as P's native dense forward. Exact local
        # state remains fp32; factor approximation is delayed until S_N exists.
        rows = torch.arange(state.shape[0], dtype=slots.dtype, device=state.device)
        output = backend.kernel_dispatcher.packed_decode(
            mixed_qkv=mixed,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=layer.head_k_dim**-0.5,
            ssm_states=state,
            cache_indices=rows,
            num_v_heads=layer.num_v_heads,
            head_v_dim=layer.head_v_dim,
            replayssm_d=None,
            replayssm_k=None,
            replayssm_g=None,
            replayssm_write_pos=None,
            replayssm_force_flush=None,
        )
        backend._track_mamba_state_decode(batch, conv, temporal, slots, layer.layer_id)
        return output

    def publish(self):
        if self.published:
            return
        if len(self.states) != len(self.pool.layer_ids) or self.tail_layers != {
            lid for lid in self.pool.layer_ids if lid < 31
        }:
            raise RuntimeError(
                "PD publication before all prefix states and shallow tails"
            )
        rp = self.controller.request_pool
        if not torch.equal(self.generations, rp.req_generation[self.ids]):
            raise RuntimeError("PD request generation changed before publication")
        from .gdn_factored_pool import (
            factorize_layers,
            factorize_dense,
            ORTH_METHOD,
            ORTH_WARPS_OVERRIDE,
        )

        launch_side(
            self.pool._tracked_factor_side,
            self.plan,
            self.states,
            self.controls,
            final=self.final,
            eager=factorize_layers,
            policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense),
        )
        self.published = True
        self.controller.stats["published"] += 1
        if (
            self.controller.stats["published"] == 1
            or self.controller.stats["published"] % 500 == 0
        ):
            logger.info(
                "PD_FACTOR_DEFERRED_ACTIVE final=%d layers=%d batch=%d wire=%s counts=%s",
                self.final,
                len(self.states),
                self.plan.slots.numel(),
                "2:r/r" if self.final else "1:r+1/r",
                self.controller.stats,
            )

    def finish_return(self):
        if not self.published:
            raise RuntimeError("PD forward returned without factor publication")
        self.pool.launch_pending_tracked()
        self.pool.pside_join()  # Enqueue before any return/transfer reader.
        if self.final:
            self.controller.state.valid.index_fill_(
                0, self.plan.slots.long(), WIRE_FINAL_DEFERRED
            )

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.publish()
            elif self.published:
                self.pool.pside_join()
        finally:
            self.pool._exact_tail_transaction = None
        return False


class PDDeferredHandoff:
    """Same wire tensors, explicit version in the existing boundary-valid word."""

    def __init__(self, original, pool, state, *, final, role):
        self.original, self.pool, self.state = original, pool, state
        self.final, self.role = final, role

    def before_send(self, req):
        self.pool.pside_join()
        slot = req.kv.mamba_pool_idx
        if slot is None:
            raise RuntimeError("PD deferred send has no state slot")
        tag = self.state.valid[slot]
        if bool((tag == WIRE_FINAL_DEFERRED).all().item()):
            if not self.final:
                raise RuntimeError("PD final-deferred wire requires paired opt-in")
            if not bool((self.pool.count[:, slot] == self.pool.cfg.r).all().item()):
                raise RuntimeError(
                    "PD final-deferred send before r/r factor publication"
                )
            # .item() above synchronizes after e_F: RDMA is not a CUDA reader.
            return
        self.original.before_send(req)

    def prepare_receive(self, req):
        return self.original.prepare_receive(req)

    def commit_receive(self, req):
        self.original.commit_receive(req)
        slot = req.kv.mamba_pool_idx
        tag = self.state.valid[slot]
        if bool((tag == WIRE_FINAL_DEFERRED).all().item()):
            if not self.final or self.role != "decode":
                raise RuntimeError("PD final-deferred wire requires D opt-in")
            if not bool((self.pool.count[:, slot] == self.pool.cfg.r).all().item()):
                raise RuntimeError("PD final-deferred receive requires r/r factors")
            # Only shallow layers already contain token N. The existing D
            # boundary runs deep layers once, then ordinary full-depth decode.
            self.state.valid.index_fill_(0, slot.reshape(-1).long(), 1)
            req.pd_final_factor_deferred_received = True
        elif not bool((tag == 1).all().item()):
            raise RuntimeError("unknown PD shallow-boundary wire version")
