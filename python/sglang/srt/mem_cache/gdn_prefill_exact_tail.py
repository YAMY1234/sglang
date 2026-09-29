"""Full-N exact recurrent tails with one deferred factor/wire publication."""
import logging
import os
from contextlib import contextmanager
from functools import wraps

import torch


FLAG = "SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH"
logger = logging.getLogger(__name__)


def tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


class ExactTailTransaction:
    def __init__(self, pool, request_pool, batch):
        self.pool, self.request_pool, self.batch = pool, request_pool, batch
        self.states, self.tails, self.plan, self.controls = [], {}, None, None
        ids = getattr(batch, "req_pool_indices_cpu", None)
        if ids is None or request_pool.req_generation.device.type != "cpu":
            raise RuntimeError("exact tail needs host request generations")
        self.request_indices = torch.as_tensor(ids, dtype=torch.long).clone()
        if self.request_indices.device.type != "cpu":
            raise RuntimeError("exact tail request identities must stay on the host")
        self.generations = request_pool.req_generation[self.request_indices].clone()
        self.prefix_rows = None
        self.published = False
        self.tail_expected = False

    def __enter__(self):
        if getattr(self.pool, "_exact_tail_transaction", None) is not None:
            raise RuntimeError("exact-tail model forwards cannot overlap")
        self.pool.pside_join()
        self.pool._exact_tail_transaction = self
        return self

    def add(self, layer_id, plan, dense, tracked, track_slots, final_src, final_dst):
        p = self.pool
        if p.layer_map[layer_id] != len(self.states):
            raise RuntimeError("exact-tail prefix layers arrived out of order")
        controls = (plan.slots, track_slots, final_src, final_dst)
        if self.plan is None:
            normal = plan.slots.tolist()
            checkpoint = [] if track_slots is None else track_slots.tolist()
            destinations = [] if final_dst is None else final_dst.tolist()
            if (not 0 < len(normal) <= 16 or len(checkpoint) > 16
                    or len(set(normal + checkpoint + destinations)) != len(normal + checkpoint + destinations)):
                raise RuntimeError("exact-tail checkpoint destinations alias or exceed buckets")
            self.plan = plan
            self.controls = [(t, None if t is None else t.clone(),
                              None if t is None else tensor_version(t)) for t in controls]
            p.invalidate_prefix_dense(plan.slots)
            if track_slots is not None:
                p.invalidate_prefix_dense(track_slots)
        elif any((a is None) != (b[0] is None) or
                 (a is not None and a.data_ptr() != b[0].data_ptr())
                 for a, b in zip(controls, self.controls)):
            raise RuntimeError("exact-tail checkpoint destinations changed between layers")
        if self.states and (tracked is None) != (self.states[0][1] is None):
            raise RuntimeError("exact-tail tracked branch changed between layers")
        # Own S_N-1 before the dense recurrent kernel can update its scratch.
        self.states.append((dense.clone(memory_format=torch.contiguous_format),
                            None if tracked is None else tracked.clone(memory_format=torch.contiguous_format)))
        plan.next_layer += 1

    def decode(self, backend, layer, batch, mixed, a, b, conv, temporal, slots):
        index = self.pool.layer_map[layer.layer_id]
        if index >= len(self.states) or layer.layer_id in self.tails:
            raise RuntimeError("exact tail requires exactly one completed prefix per layer")
        if self.prefix_rows is None:
            raise RuntimeError("exact tail is missing its prefix row mapping")
        state = self.states[index][0].index_select(0, self.prefix_rows).contiguous()
        if state.shape[0] != mixed.shape[0]:
            raise RuntimeError("exact tail row mapping differs from its activations")
        self.tails[layer.layer_id] = (mixed.clone(), a.clone(), b.clone(), slots.clone())
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

        rows = torch.arange(state.shape[0], dtype=torch.int32, device=state.device)
        output = TritonGDNKernel().packed_decode(mixed, a, b, A_log=layer.A_log,
            dt_bias=layer.dt_bias, scale=layer.head_k_dim ** -0.5,
            ssm_states=state, cache_indices=rows, num_v_heads=layer.num_v_heads,
            head_v_dim=layer.head_v_dim)
        backend._track_mamba_state_decode(batch, conv, temporal, slots, layer.layer_id)
        return output

    def publish(self):
        from .gdn_factored_pool import factorize_layers, factorize_dense, ORTH_METHOD, ORTH_WARPS_OVERRIDE

        if self.published:
            return
        if self.plan is None or len(self.states) != len(self.pool.layer_ids):
            raise RuntimeError("exact-tail forward ended before every GDN layer")
        if not torch.equal(self.generations, self.request_pool.req_generation[self.request_indices]):
            raise RuntimeError("exact-tail request generation changed before publication")
        for tensor, snapshot, version in self.controls:
            if tensor is not None and (tensor_version(tensor) != version or not torch.equal(tensor, snapshot)):
                raise RuntimeError("exact-tail controls changed before publication")
        graph = self.pool._prefill_batch_graph
        if not graph.warmed:
            raise RuntimeError("exact-tail publication graph must be prewarmed")
        expected = {layer.layer_id for layer in self.pool._exact_tail_layers}
        if self.tail_expected and set(self.tails) != expected:
            raise RuntimeError("exact-tail forward is missing recurrent layer activations")
        self.plan.exact_tail_inputs = self.tails
        graph.run(self.pool, self.plan, self.states, *[x[0] for x in self.controls[1:]],
                  eager=factorize_layers, policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense))
        self.published = True

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.publish()
        finally:
            self.pool._exact_tail_transaction = None
            if self.plan is not None:
                self.plan.exact_tail_inputs = None
            self.states.clear()
            self.tails.clear()
        return False


def _observe(name, function, args, kwargs, batch, layer):
    if not os.environ.get("TWINSTAR_CUDA_TIMELINE"):
        return function(*args, **kwargs)
    from twinstar_sgl.pd_boundary_observe import call

    return call(name, function, args, kwargs, batch=batch, layer=layer)


@contextmanager
def split_boundary(backend, prefix_batch, prefix_indices, boundary_batch, boundary_indices,
                   prefix_metadata, boundary_metadata, *, split_layer_limit=31,
                   publication_observer=None):
    transaction = getattr(backend.factored, "_exact_tail_transaction", None)
    if transaction is None or prefix_metadata is None or not prefix_batch.batch_size:
        raise RuntimeError("exact-tail split needs a nonempty prefix transaction")
    if publication_observer is not None:
        raise ValueError("exact-tail publication observer requires the deferred publication boundary")
    if (boundary_batch.mamba_track_mask is not None
            and bool(boundary_batch.mamba_track_mask.any().item())):
        raise RuntimeError("exact-tail boundary cannot publish decode checkpoints")
    prefix_ids = list(map(int, prefix_batch.req_pool_indices_cpu))
    tail_ids = list(map(int, boundary_batch.req_pool_indices_cpu))
    if len(set(prefix_ids)) != len(prefix_ids) or any(i not in prefix_ids for i in tail_ids):
        raise RuntimeError("exact-tail prefix and boundary request identities differ")
    transaction.tail_expected = True
    transaction.prefix_rows = torch.tensor([prefix_ids.index(i) for i in tail_ids],
        dtype=torch.long, device=prefix_metadata.mamba_cache_indices.device)
    original, saved = backend.forward_extend, backend.forward_metadata

    def prefix(layer, batch, mixed, a, b, **kwargs):
        backend.forward_metadata = prefix_metadata
        return _observe("P_gdn_prefix_with_W8", original,
                        (layer, batch, mixed, a, b), kwargs, batch, layer.layer_id)

    def forward(layer, forward_batch, mixed_qkv, a, b, **kwargs):
        batch, mixed = forward_batch, mixed_qkv
        if layer.layer_id >= split_layer_limit:
            return prefix(layer, batch, mixed, a, b, **kwargs)
        output = kwargs.get("linear_attn_output")
        if output is None:
            output = mixed.new_empty((1, mixed.shape[0], layer.num_v_heads, layer.head_v_dim))
        inner = {k: v for k, v in kwargs.items() if k != "linear_attn_output"}
        output[:, prefix_indices] = prefix(layer, prefix_batch, mixed[prefix_indices],
                                           a[prefix_indices], b[prefix_indices], **inner)
        backend.forward_metadata = boundary_metadata
        output[:, boundary_indices] = _observe("P_shallow_recurrent_tail", backend.forward_decode,
            (layer, boundary_batch, mixed[boundary_indices].contiguous(),
             a[boundary_indices].contiguous(), b[boundary_indices].contiguous()),
            inner, boundary_batch, layer.layer_id)
        backend.forward_metadata = prefix_metadata
        return output

    backend.forward_extend = forward
    try:
        yield
        transaction.publish()
    finally:
        backend.forward_extend, backend.forward_metadata = original, saved


def install(runner):
    if os.environ.get(FLAG) != "1":
        return
    from sglang.srt.runtime_context import get_schedule
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    from twinstar_sgl import pd_shallow_gdn

    owner = runner.model
    pool = runner.req_to_token_pool.factored_gdn_pool
    conflicts = ("SGLANG_GDN_PREFILL_BATCH_GRAPH", "SGLANG_GDN_PREFILL_TRACKED_GRAPH",
                 "SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH", "SGLANG_GDN_PREFILL_TAIL_GRAPH",
                 "SGLANG_GDN_PREFILL_MODEL_TAIL_GRAPH", "SGLANG_GDN_PSIDE_GRAPH",
                 "SGLANG_GDN_PSIDE_COMPOSITE", "TWINSTAR_PD_EMITTER_GRAPH")
    shallow = getattr(owner, "pd_shallow_role", None) == "prefill"
    if (any(os.environ.get(k) == "1" for k in conflicts)
            or not get_schedule().disable_overlap_schedule
            or os.environ.get("SGLANG_FLASHNEXT_ARRIVAL_OVERLAP") == "1"
            or runner.server_args.disaggregation_mode != "prefill"
            or (not shallow and os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL") != "1")
            or not pool.cfg.strict_chunk or not pool.cfg.factored_prefix
            or pool.cfg.init_method != "k31" or pool.prefix_dense is not None
            or len(pool.layer_ids) != 36 or pool.prefix_layer_count() != 36
            or not pool.batch_prefill or os.environ.get("SGLANG_GDN_PREFILL_COMMIT_GRAPH") != "1"):
        raise ValueError("exact-tail batch requires the isolated strict k31 full-N P recipe")
    limit = 31 if shallow else 48
    layers = [m for m in owner.model.model.modules() if isinstance(m, RadixLinearAttention)
              and m.layer_id < limit]
    layers.sort(key=lambda m: m.layer_id)
    if [m.layer_id for m in layers] != [i for i in pool.layer_ids if i < limit]:
        raise ValueError("exact-tail model is missing native recurrent layer parameters")
    pool._exact_tail_layers = layers
    original = owner.forward
    if getattr(owner, "_exact_tail_installed", False):
        return

    @wraps(original)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        batch = forward_batch
        if not batch.forward_mode.is_extend():
            return original(input_ids, positions, batch, *args, **kwargs)
        if (batch.forward_mode.is_mixed() or not 1 <= batch.batch_size <= 16
                or getattr(batch, "can_run_tbo", False)):
            raise ValueError("exact-tail batch does not support mixed, overlapping or oversized extends")
        with ExactTailTransaction(pool, runner.req_to_token_pool, batch):
            return original(input_ids, positions, batch, *args, **kwargs)

    pd_shallow_gdn.split_boundary = split_boundary
    owner.forward = forward
    owner._exact_tail_installed = True
    logger.info("GDN exact-tail batch installed: model_depth=%d gdn_layers=%d recurrent_layers=%d",
                limit, len(pool.layer_ids), len(layers))
