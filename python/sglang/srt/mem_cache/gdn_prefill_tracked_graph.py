"""Defer independent checkpoints until a full-N model forward has finished."""
import logging
import os
from functools import wraps

import torch
import triton

from sglang.srt.utils.graph_capture import graph_capture_lock
from .gdn_prefill_commit_graph import BATCH_BUCKETS, _publish_valid
from .gdn_prefill_checkpoint_graph import independent_slots

logger = logging.getLogger(__name__)
FLAG = "SGLANG_GDN_PREFILL_TRACKED_GRAPH"


class TrackedBuffers:
    def __init__(self, pool, batch, storage):
        self.pool, self.batch = pool, batch
        self.states = [row[:batch] for row in storage]
        self.slots = torch.full((batch,), -1, dtype=torch.long, device=pool.a.device)
        self.omega = pool.init_omega(batch)

    def bind(self, states, slots):
        if len(states) != len(self.states) or slots.numel() > self.batch:
            raise RuntimeError("tracked graph changed its layer or slot count")
        for dst, src in zip(self.states, states):
            if src.shape[0] != slots.numel() or src.shape[1:] != dst.shape[1:]:
                raise RuntimeError("tracked graph state shape differs from its checkpoint slots")
            dst[:src.shape[0]].copy_(src)
            dst[src.shape[0]:].zero_()
        self.slots[:slots.numel()].copy_(slots)
        self.slots[slots.numel():].fill_(-1)

    def evaluate(self, eager):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored

        p = self.pool
        factors = eager(self.states, p.vbar, p.cfg, omega=self.omega)
        for i, values in enumerate(factors):
            store_factored(*values, p.a[i], p.U[i], p.W[i], p.count[i],
                           p.stale, p.dense_of, self.slots, p.cfg.r, stale_value=1)
        _publish_valid[(1,)](p.prefix_valid, self.slots, self.batch,
                            triton.next_power_of_2(self.batch))


class TrackedGraph:
    def __init__(self):
        self.entries, self.storage = {}, None
        self.stream = self.memory_pool = None
        self.warmed = False
        self.stats = dict(captured=0, replayed=0)

    @staticmethod
    def key(batch, eager, policy):
        return (batch, eager, policy, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

    def run(self, pool, states, slots, *, batch, eager, policy):
        if batch not in BATCH_BUCKETS or len(states) != len(pool.layer_ids):
            raise RuntimeError("tracked graph requires every layer and a prewarmed bucket")
        key = self.key(batch, eager, policy)
        entry = self.entries.get(key)
        if entry is None:
            if self.warmed:
                raise RuntimeError("tracked graph missing after complete prewarm")
            if self.storage is None:
                self.storage = [torch.zeros(BATCH_BUCKETS[-1], pool.hv, pool.v, pool.k,
                                           dtype=torch.float32, device=pool.a.device)
                                for _ in pool.layer_ids]
                self.stream = torch.cuda.Stream(device=pool.a.device)
                self.memory_pool = torch.cuda.graph_pool_handle()
            buffers = TrackedBuffers(pool, batch, self.storage)
            buffers.bind(states, slots)
            current = torch.cuda.current_stream(pool.a.device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                buffers.evaluate(eager)
            current.wait_stream(self.stream)
            graph = torch.cuda.CUDAGraph()
            with graph_capture_lock, torch.cuda.graph(graph, stream=self.stream,
                    pool=self.memory_pool, capture_error_mode="thread_local"):
                buffers.evaluate(eager)
            self.entries[key] = entry = (buffers, graph)
            self.stats["captured"] += 1
            logger.info("GDN prefill tracked graph captured: layers=%d batch=%d dtype=fp32",
                        len(pool.layer_ids), batch)
        else:
            entry[0].bind(states, slots)
        entry[1].replay()
        self.stats["replayed"] += 1

    def prewarm(self, pool, *, eager, policy):
        if self.warmed:
            return
        before = torch.cuda.memory_allocated(pool.a.device)
        for batch in BATCH_BUCKETS:
            state = torch.zeros(batch, pool.hv, pool.v, pool.k,
                                dtype=torch.float32, device=pool.a.device)
            slots = torch.full((batch,), -1, dtype=torch.long, device=pool.a.device)
            self.run(pool, [state] * len(pool.layer_ids), slots,
                     batch=batch, eager=eager, policy=policy)
        expected = {self.key(b, eager, policy) for b in BATCH_BUCKETS}
        if set(self.entries) != expected:
            raise RuntimeError("tracked graph prewarm did not cover every bucket")
        torch.cuda.synchronize(pool.a.device)
        self.warmed = True
        logger.info("GDN prefill tracked prewarm complete: expected=%d captured=%d retained_bytes=%d",
                    len(expected), len(self.entries), torch.cuda.memory_allocated(pool.a.device) - before)


def tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None  # Inference tensors have no host version counter.


class TrackedTransaction:
    def __init__(self, pool, request_pool, batch):
        self.pool, self.request_pool, self.batch = pool, request_pool, batch
        self.states, self.plan, self.controls = [], None, None
        self.fallback = None
        indices = getattr(batch, "req_pool_indices_cpu", None)
        if indices is None:
            raise RuntimeError("tracked graph needs CPU request identities")
        self.request_indices = torch.as_tensor(indices, dtype=torch.long).clone()
        if (self.request_indices.device.type != "cpu"
                or request_pool.req_generation.device.type != "cpu"):
            raise RuntimeError("tracked checkpoint generations must remain on the host")
        self.generations = request_pool.req_generation[self.request_indices].clone()

    def __enter__(self):
        if getattr(self.pool, "_tracked_transaction", None) is not None:
            raise RuntimeError("tracked graph cannot overlap model forwards")
        self.pool.pside_join()
        self.pool._tracked_transaction = self
        return self

    def prepare(self, plan, metadata):
        if plan is None or not metadata.has_mamba_track_mask:
            return
        tracked = metadata.track_ssm_h_dst
        if tracked is None or tracked.numel() == 0:
            return
        if self.plan is not None:
            raise RuntimeError("tracked graph received a second prefix plan")
        self.plan = plan
        size = max(plan.slots.numel(), tracked.numel())
        finals = (metadata.track_ssm_final_src, metadata.track_ssm_final_dst)
        final_cpu = [int(s) for t in finals if t is not None for s in t.tolist()]
        if (plan.next_layer != 0 or plan.last_layer != len(self.pool.layer_ids) - 1
                or size > BATCH_BUCKETS[-1]
                or not independent_slots(plan.slots.tolist(), tracked.tolist(), final_cpu)):
            self.fallback = "alias-or-plan-or-bucket"
            logger.warning("GDN prefill tracked fallback: reason=%s", self.fallback)
            return
        normal_bucket = next(b for b in BATCH_BUCKETS if size <= b)
        self.bucket = 1 if tracked.numel() == 1 else normal_bucket
        plan.prefill_normal_bucket = normal_bucket
        self.source_slots, self.slots = tracked, tracked.clone()
        self.controls = [(t, tensor_version(t), t.data_ptr()) for t in (plan.slots, tracked, *finals)
                         if t is not None]
        plan.tracked_transaction = self
        self.pool.invalidate_prefix_dense(tracked)

    def add(self, layer_id, state, slots):
        index = self.pool.layer_map[layer_id]
        if (index != len(self.states) or state is None or slots is None
                or slots.data_ptr() != self.source_slots.data_ptr()):
            raise RuntimeError("tracked checkpoints changed layer order or identity")
        # Advanced-indexed checkpoint states own their storage; retain every layer.
        self.states.append(state)

    def __exit__(self, exc_type, exc, traceback):
        from .gdn_factored_pool import factorize_layers, factorize_dense, ORTH_METHOD, ORTH_WARPS_OVERRIDE

        try:
            if exc_type is not None or self.plan is None or self.fallback is not None:
                return False
            if not self.states:
                raise RuntimeError("tracked checkpoint plan was never consumed")
            if len(self.states) != len(self.pool.layer_ids):
                raise RuntimeError("tracked checkpoint forward ended with missing layers")
            if not torch.equal(self.generations, self.request_pool.req_generation[self.request_indices]):
                raise RuntimeError("tracked checkpoint request slot generation changed")
            if any(tensor_version(t) != version or t.data_ptr() != ptr for t, version, ptr in self.controls):
                raise RuntimeError("tracked checkpoint controls changed before publication")
            graph = self.pool._prefill_tracked_graph
            if not graph.warmed:
                raise RuntimeError("tracked graph must be prewarmed before model execution")
            graph.run(self.pool, self.states, self.slots, batch=self.bucket,
                      eager=factorize_layers, policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense))
        finally:
            self.pool._tracked_transaction = None
            self.states.clear()
        return False


def prepare(pool, metadata):
    transaction = getattr(pool, "_tracked_transaction", None)
    if transaction is not None:
        transaction.prepare(metadata.factored_extend, metadata)


def install(runner):
    if os.environ.get(FLAG) != "1":
        return
    from sglang.srt.runtime_context import get_schedule

    model, pool = runner.model, runner.req_to_token_pool.factored_gdn_pool
    if getattr(model, "_prefill_tracked_installed", False):
        return
    conflicts = ("SGLANG_GDN_PREFILL_BATCH_GRAPH", "SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH",
                 "SGLANG_GDN_PSIDE_GRAPH", "SGLANG_GDN_PSIDE_COMPOSITE", "TWINSTAR_PD_EMITTER_GRAPH")
    if any(os.environ.get(flag) == "1" for flag in conflicts):
        raise ValueError("tracked graph requires the original full-N P model path")
    shallow = getattr(model, "pd_shallow_role", None) == "prefill"
    factor = os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL") == "1"
    if (runner.server_args.disaggregation_mode != "prefill" or not (shallow or factor)
            or not get_schedule().disable_overlap_schedule or not pool.batch_prefill
            or os.environ.get("SGLANG_GDN_PREFILL_COMMIT_GRAPH") != "1"
            or len(pool.layer_ids) != 36
            or pool.prefix_layer_count() != 36 or pool.prefix_dense is not None
            or not pool.cfg.strict_chunk or not pool.cfg.factored_prefix
            or pool.cfg.init_method != "k31"):
        raise ValueError("tracked graph requires the strict k31 P31/P48 nonoverlap recipe")
    original = model.forward

    @wraps(original)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        mode = forward_batch.forward_mode
        if not mode.is_extend():
            return original(input_ids, positions, forward_batch, *args, **kwargs)
        if mode.is_mixed() or getattr(forward_batch, "can_run_tbo", False):
            raise ValueError("tracked graph cannot defer across mixed or overlapping forwards")
        with TrackedTransaction(pool, runner.req_to_token_pool, forward_batch):
            return original(input_ids, positions, forward_batch, *args, **kwargs)

    model.forward = forward
    model._prefill_tracked_installed = True
    logger.info("GDN prefill tracked full-N adapter installed: arm=%s layers=36", "1+2" if shallow else "1")
