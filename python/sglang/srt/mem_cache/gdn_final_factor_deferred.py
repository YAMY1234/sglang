"""AGG candidate A: dense prompt boundary, then F(S_N) on the tracked stream.

This changes the truncation point (r+1 -> r), not merely stream scheduling.
The switch is off by default. The first version admits a complete B1 prompt,
with disjoint tracked checkpoints, on the existing k31 / 2b graph recipe.
"""
import logging

import torch

logger = logging.getLogger(__name__)


def request_rejection(owner, batch):
    if torch.cuda.is_current_stream_capturing():
        return "capture"
    if batch.spec_info is not None:
        return "spec"
    if batch.return_logprob:
        return "logprob"
    if (batch.batch_size != 1 or batch.extend_seq_lens_cpu is None
            or len(batch.extend_seq_lens_cpu) != 1
            or int(batch.extend_seq_lens_cpu[0]) <= 1
            or batch.twinstar_prompt_final != [True]
            or owner._boundary_lens(batch) != [1]):
        return "shape"
    if (batch.req_pool_indices_cpu is None
            or batch.req_pool_indices_cpu.device.type != "cpu"):
        return "host_identity"
    if (owner.state_audit_dir or owner.fullstack_v3_latent
            or batch.capture_hidden_mode.is_full()):
        return "observer_or_latent"
    return None


def dense_boundary_decode(backend, layer, batch, mixed, a, b, conv, temporal, slots):
    """The P arm's packed decode dispatcher, with the exact ring as its state.

    Conv has already run with the ordinary request indices. Only the recurrent
    state addresses differ; dtype, QKV/gates/weights, scale and kernel match P.
    Capture padding maps to -1, never ring[0]. No factor or tracked reader here.
    """
    pool = backend.factored
    ring_slots = torch.where(slots >= 0, pool.dense_of[slots.clamp_min(0)].to(slots.dtype), -1)
    output = backend.kernel_dispatcher.packed_decode(
        mixed_qkv=mixed, a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias,
        scale=layer.head_k_dim ** -0.5,
        ssm_states=pool.dense_ring[pool.layer_map[layer.layer_id]],
        cache_indices=ring_slots, num_v_heads=layer.num_v_heads,
        head_v_dim=layer.head_v_dim, replayssm_d=None, replayssm_k=None,
        replayssm_g=None, replayssm_write_pos=None, replayssm_force_flush=None,
    )
    backend._track_mamba_state_decode(batch, conv, temporal, slots, layer.layer_id)
    return output


def copy_exact_to_ring(buffers):
    from .gdn_prefill_batch_graph import scatter_rows

    for state, ring in zip(buffers.normal, buffers.pool.dense_ring):
        scatter_rows(state[None], ring[None], buffers.ring_dst)
    # plan_extend has already published ring ownership. Factors and their
    # validity remain unpublished until F(S_N) finishes.
    p = buffers.pool
    zeros = torch.zeros_like(buffers.slots, dtype=p.stale.dtype)[None, :, None]
    scatter_rows(zeros, p.stale[None, :, None], buffers.slots)


def factorize_ring(buffers, eager):
    # This body is captured once; all normal states are overwritten with S_N
    # before calling the exact same #23 factorize/store/publish graph body.
    rows = buffers.ring_dst.clamp_min(0)
    for state, ring in zip(buffers.normal, buffers.pool.dense_ring):
        state.copy_(ring.index_select(0, rows))
    buffers.evaluate(eager, branch="normal")


class FinalFactorDeferred:
    def __init__(self, pool, side):
        self.pool, self.side = pool, side
        self.entries = {}
        self.ring_generation = pool.ring_generation
        self.boundary = None
        self.boundary_ready = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.capture_stream = torch.cuda.Stream(device=pool.a.device)
        self.arena = torch.cuda.graph_pool_handle()
        self.recorded = False
        self._prefill_side_pending = False
        self.waited_streams = set()
        self.request_ids = set()
        self.active_plan = None
        self.staged = None
        self.controls = None
        self.stats = {"deferred": 0, "joins": 0}

    def fallback(self, reason):
        name = "fallback_" + reason
        self.stats[name] = self.stats.get(name, 0) + 1
        self.log_stats()
        return False

    def log_stats(self):
        total = sum(v for k, v in self.stats.items() if k != "joins")
        if total == 1 or total % 500 == 0:
            logger.info("final factor deferred counts: %s", self.stats)

    def prewarm(self, eager):
        from sglang.srt.utils.graph_capture import graph_capture_lock

        current = torch.cuda.current_stream(self.pool.a.device)
        before = torch.cuda.memory_allocated(self.pool.a.device)
        for k, entries in enumerate((self.side.entries, self.side.alt_entries)):
            for key, (buffers, _, tracked) in entries.items():
                if key[:2] != (1, 1):
                    continue
                graphs = []
                for evaluate in (lambda: copy_exact_to_ring(buffers),
                                 lambda: factorize_ring(buffers, eager)):
                    self.capture_stream.wait_stream(current)
                    with torch.cuda.stream(self.capture_stream):
                        evaluate()
                    current.wait_stream(self.capture_stream)
                    graph = torch.cuda.CUDAGraph()
                    with graph_capture_lock, torch.cuda.graph(
                        graph, stream=self.capture_stream, pool=self.arena,
                        capture_error_mode="thread_local"
                    ):
                        evaluate()
                    graphs.append(graph)
                self.entries[(k, key)] = (buffers, graphs[0], tracked, graphs[1])
        torch.cuda.synchronize(self.pool.a.device)
        logger.info("final factor deferred prewarm: entries=%d extra_retained_bytes=%d "
                    "boundary=B1 source=S_N count=r join_branches=0",
                    len(self.entries), torch.cuda.memory_allocated(self.pool.a.device)-before)

    def request(self, owner, batch):
        reason = request_rejection(owner, batch)
        if reason is not None:
            return self.fallback(reason)
        if self.boundary is None:
            return self.fallback("boundary")
        if self.pool.ring_generation != self.ring_generation:
            return self.fallback("ring_generation")
        return True

    def prepare(self, batch, plan, metadata, *, eager, policy):
        """Admit before any P kernel. Fallbacks never partially run candidate A."""
        from .gdn_tracked_factor_side import disjoint_destinations

        if self.active_plan is not None or self.staged is not None:
            raise RuntimeError("a deferred final transaction is already active")
        if torch.cuda.is_current_stream_capturing():
            return self.fallback("capture")
        if (self.pool.ring_generation != self.ring_generation
                or any(x < 0 for x in plan.final_ring_cpu)):
            return self.fallback("ring")
        if (len(plan.final_slots_cpu) != 1 or plan.final_slots_cpu[0] < 0
                or plan.next_layer != 0 or plan.last_layer != len(self.pool.layer_ids)-1
                or plan.pending or plan.checkpoint_group is not None):
            return self.fallback("plan")
        if not metadata.has_mamba_track_mask:
            return self.fallback("no_tracked")
        tracks = metadata.track_ssm_h_dst
        sources, destinations = metadata.track_ssm_final_src, metadata.track_ssm_final_dst
        # A grid-aligned checkpoint refers to S_N-1; copying F(S_N) there would
        # silently change prefix depth. Keep the original path for that case.
        if sources.numel() or destinations.numel():
            return self.fallback("prefix_final_copy")
        if tracks.numel() != 1:
            return self.fallback("tracked_shape")
        reason = disjoint_destinations(plan.final_slots_cpu, tracks.tolist(), [], [])
        if reason is not None:
            return self.fallback(reason)
        key = self.side.whole_graph.key(1, 1, eager, policy, False)
        k = self.side.parity
        if (k, key) not in self.entries:
            return self.fallback("signature")
        self.join()  # A's shared normal buffers cannot outlive the previous F.
        self.pool.invalidate_prefix_dense(plan.slots)
        self.pool.invalidate_prefix_dense(tracks)
        self.active_plan = plan
        self.controls = (tracks, sources, destinations)
        self.staged_key = (k, key)
        self.request_ids = set(map(int, batch.req_pool_indices_cpu.tolist()))
        plan.defer_final_factor = True
        return True

    def add(self, layer_id, plan, dense, tracked, tracks, sources, destinations):
        if plan is not self.active_plan or self.pool.layer_map[layer_id] != plan.next_layer:
            raise RuntimeError("deferred final layers arrived out of order")
        if tracked is None or any(a is not b for a, b in zip(
                (tracks, sources, destinations), self.controls)):
            raise RuntimeError("deferred final checkpoint controls changed")
        plan.pending.append((dense, tracked))
        plan.next_layer += 1
        if plan.next_layer == len(self.pool.layer_ids):
            self.stage()

    def stage(self):
        plan, side = self.active_plan, self.side
        k, _ = self.staged_key
        buffers, ring_graph, tracked_graph, final_graph = self.entries[self.staged_key]
        current = torch.cuda.current_stream(self.pool.a.device)
        if side.set_recorded[k]:
            current.wait_event(side.set_done[k])  # 2b's tracked input ownership.
        buffers.bind(plan, plan.pending, *self.controls)
        ring_graph.replay()  # G_A: exact S_N-1 only, no final factorization.
        side.final_done.record(current)
        side.stream.wait_event(side.final_done)
        side.done = side.set_done[k]
        with torch.cuda.stream(side.stream):
            tracked_graph.replay()
            side.done.record(side.stream)
        side.recorded = side._prefill_side_pending = True
        side.waited_streams.clear()
        side.set_recorded[k] = True
        side.parity = 1-k
        plan.pending.clear()
        self.staged = final_graph

    def after_boundary(self):
        if self.staged is None:
            raise RuntimeError("dense boundary has no staged final batch")
        current = torch.cuda.current_stream(self.pool.a.device)
        self.boundary_ready.record(current)
        self.side.stream.wait_event(self.boundary_ready)
        with torch.cuda.stream(self.side.stream):
            self.staged.replay()  # T -> wait(e_B) -> F(S_N), one side stream.
            self.done.record(self.side.stream)
        self.recorded = self._prefill_side_pending = True
        self.waited_streams.clear()
        self.active_plan.defer_final_factor = False
        self.active_plan = self.staged = self.controls = None
        self.stats["deferred"] += 1
        if self.stats["deferred"] == 1:
            logger.info("FINAL_FACTOR_DEFERRED_ACTIVE boundary=dense source=S_N "
                        "count=%d layers=%d batch=1", self.pool.cfg.r, len(self.pool.layer_ids))
        self.log_stats()

    def join_for_batch(self, batch):
        # Mirrors are optional in the native replay view. Unknown identity is
        # a conservative wait, never a new D2H synchronization on decode.
        ids = getattr(batch, "req_pool_indices_cpu", None)
        if ids is not None and ids.device.type == "cpu":
            if not self.request_ids.intersection(map(int, ids.tolist())):
                return
        self.join()

    def join(self):
        if self.staged is not None:
            raise RuntimeError("factor reader arrived before the dense boundary")
        if not self.recorded:
            return
        current = torch.cuda.current_stream(self.pool.a.device)
        if current == self.side.stream:
            return
        if torch.cuda.is_current_stream_capturing():
            if self.done.query():
                self.recorded = self._prefill_side_pending = False
                return
            raise RuntimeError("deferred final work must finish before graph capture")
        if current.cuda_stream not in self.waited_streams:
            current.wait_event(self.done)
            self.waited_streams.add(current.cuda_stream)
            self.stats["joins"] += 1
        self._prefill_side_pending = False

    def abort(self):
        # An exception aborts the request; do not publish uncomputed S_N. Join
        # tracked work before the scheduler clears slots and ring ownership.
        if self.active_plan is not None:
            self.side.join()
            self.active_plan.defer_final_factor = False
            self.active_plan.pending.clear()
        self.active_plan = self.staged = self.controls = None
