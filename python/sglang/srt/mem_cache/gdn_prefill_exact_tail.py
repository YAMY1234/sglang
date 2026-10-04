"""Full-N exact recurrent tails with one deferred factor/wire publication."""
import logging
import os
from contextlib import contextmanager
from functools import wraps

import torch

from .gdn_prefill_recipe_guard import warn_install_rejection


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
        self.empty_states, self.empty_slots = [], None
        self.normal_tail_rows = self.empty_tail_rows = None
        self.boundary_control = None

    def __enter__(self):
        if getattr(self.pool, "_exact_tail_transaction", None) is not None:
            raise RuntimeError("exact-tail model forwards cannot overlap")
        self.pool.pside_join()
        self.pool._exact_tail_transaction = self
        return self

    def prepare_split(self, prefix_ids, boundary_batch, boundary_metadata):
        from .gdn_factored_pool import FactoredExtendPlan

        tail_ids = list(map(int, boundary_batch.req_pool_indices_cpu))
        own_ids = self.request_indices.tolist()
        if (len(set(prefix_ids)) != len(prefix_ids) or len(set(tail_ids)) != len(tail_ids)
                or any(i not in own_ids for i in prefix_ids + tail_ids)):
            raise RuntimeError("exact-tail prefix and boundary request identities differ")
        slots = boundary_metadata.mamba_cache_indices
        self.boundary_control = (slots, slots.clone(), tensor_version(slots))
        normal = [i for i, rid in enumerate(tail_ids) if rid in prefix_ids]
        empty = [i for i, rid in enumerate(tail_ids) if rid not in prefix_ids]
        dev = slots.device
        self.normal_tail_rows = torch.tensor(normal, dtype=torch.long, device=dev)
        self.empty_tail_rows = torch.tensor(empty, dtype=torch.long, device=dev)
        self.prefix_rows = torch.tensor([prefix_ids.index(tail_ids[i]) for i in normal],
                                       dtype=torch.long, device=dev)
        self.tail_expected = True
        if not empty:
            return
        rows = [own_ids.index(tail_ids[i]) for i in empty]
        if any(int(self.batch.extend_seq_lens_cpu[i]) != 1
               or not self.batch.twinstar_prompt_final[i] for i in rows):
            raise RuntimeError("empty exact prefix requires one final prompt token")
        prefix_lens = [int(self.batch.extend_prefix_lens_cpu[i]) for i in rows]
        self.empty_slots = slots.index_select(0, self.empty_tail_rows).long()
        p = self.pool
        safe = self.empty_slots.clamp_min(0)
        slot_ids, stale, dense = torch.stack((self.empty_slots, p.stale[safe], p.dense_of[safe])).tolist()
        if any(i < 0 for i in slot_ids) or len(set(slot_ids)) != len(slot_ids):
            raise RuntimeError("empty exact prefix has invalid or duplicate state slots")
        use_ring = [d >= 0 and d < len(p.ring_owner) and p.ring_owner[d] == slot
                    and stale[i] == 0 for i, (slot, d) in enumerate(zip(slot_ids, dense))]
        required = p.dense_required[safe].tolist() if p.dense_required is not None else [0] * len(rows)
        valid = p.prefix_valid[safe].tolist()
        if any(required[i] and not use_ring[i] for i in range(len(rows))):
            raise RuntimeError("unfinished exact tail lost its dense continuation state")
        if any(prefix_lens[i] > 0 and not use_ring[i] and not valid[i] for i in range(len(rows))):
            raise RuntimeError("cached empty exact prefix has no GDN checkpoint")
        # Read native initial states without reserving or overwriting any ring owner.
        initial = FactoredExtendPlan(slots=self.empty_slots,
            use_ring=torch.tensor(use_ring, device=dev),
            ring_src=torch.tensor([d if use_ring[i] else 0 for i, d in enumerate(dense)], device=dev),
            ring_dst=torch.full_like(self.empty_slots, -1), ring_dst_rows=self.empty_slots[:0],
            n_ring_src=sum(use_ring), all_fresh=not any(prefix_lens),
            last_layer=len(p.layer_ids)-1)
        self.empty_states = [p.initial_dense(lid, initial).clone() for lid in p.layer_ids]

    def add(self, layer_id, plan, dense, tracked, track_slots, final_src, final_dst):
        p = self.pool
        if p.layer_map[layer_id] != len(self.states):
            raise RuntimeError("exact-tail prefix layers arrived out of order")
        controls = (plan.slots, track_slots, final_src, final_dst)
        if self.plan is None:
            normal = plan.slots.tolist()
            checkpoint = [] if track_slots is None else track_slots.tolist()
            destinations = [] if final_dst is None else final_dst.tolist()
            empty = [] if self.empty_slots is None else self.empty_slots.tolist()
            if set(empty).intersection(normal + checkpoint + destinations):
                raise RuntimeError("empty exact prefix aliases a publication destination")
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
        if ((self.normal_tail_rows is None or self.normal_tail_rows.numel())
                and index >= len(self.states)) or layer.layer_id in self.tails:
            raise RuntimeError("exact tail requires exactly one completed prefix per layer")
        if self.prefix_rows is None:
            raise RuntimeError("exact tail is missing its prefix row mapping")
        if self.empty_slots is None:
            state = self.states[index][0].index_select(0, self.prefix_rows).contiguous()
        else:
            state = self.empty_states[index].new_empty(mixed.shape[0], self.pool.hv, self.pool.v, self.pool.k)
            state.index_copy_(0, self.empty_tail_rows, self.empty_states[index])
            if self.normal_tail_rows.numel():
                state.index_copy_(0, self.normal_tail_rows,
                                  self.states[index][0].index_select(0, self.prefix_rows))
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
        if ((self.plan is not None and len(self.states) != len(self.pool.layer_ids))
                or (self.plan is None and (self.empty_slots is None or self.normal_tail_rows.numel()))):
            raise RuntimeError("exact-tail forward ended before every GDN layer")
        if not torch.equal(self.generations, self.request_pool.req_generation[self.request_indices]):
            raise RuntimeError("exact-tail request generation changed before publication")
        controls = list(self.controls or [])
        if self.boundary_control is not None:
            controls.append(self.boundary_control)
        for tensor, snapshot, version in controls:
            if tensor is not None and (tensor_version(tensor) != version or not torch.equal(tensor, snapshot)):
                raise RuntimeError("exact-tail controls changed before publication")
        expected = {layer.layer_id for layer in self.pool._exact_tail_layers}
        if self.tail_expected and set(self.tails) != expected:
            raise RuntimeError("exact-tail forward is missing recurrent layer activations")
        if self.plan is not None:
            graph = self.pool._prefill_batch_graph
            if not graph.warmed:
                raise RuntimeError("exact-tail publication graph must be prewarmed")
            tails = self.tails
            if self.empty_slots is not None:
                tails = {lid: tuple(t.index_select(0, self.normal_tail_rows) for t in values)
                         for lid, values in tails.items()}
            self.plan.exact_tail_inputs = tails
            graph.run(self.pool, self.plan, self.states, *[x[0] for x in self.controls[1:]],
                      eager=factorize_layers, policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense))
        if self.empty_slots is not None:
            self.publish_empty()
        self.published = True

    def publish_empty(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_decode

        p = self.pool
        p.invalidate_prefix_dense(self.empty_slots)
        # No new prefix exists: retain its factors, append only the saved tail to wire.
        for layer in p._exact_tail_layers:
            mixed, a, b, slots = (t.index_select(0, self.empty_tail_rows)
                                  for t in self.tails[layer.layer_id])
            i = p.layer_map[layer.layer_id]
            factored_packed_decode(mixed, a, b, A_log=layer.A_log, dt_bias=layer.dt_bias,
                scale=layer.head_k_dim ** -0.5, vbar=p.vbar[i], fa=p.a[i], fu=p.U[i],
                fw=p.W[i], fcount=p.count[i], stale=p.stale, ssm_state_indices=slots,
                num_q_heads=layer.num_q_heads, num_v_heads=layer.num_v_heads,
                head_k_dim=layer.head_k_dim, head_v_dim=layer.head_v_dim,
                r=p.cfg.r, rfull=p.cfg.rfull, truncate=False, **p.cfg.kernel_kwargs())
        if p.dense_required is not None:
            p.dense_required.index_fill_(0, self.empty_slots, 0)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.publish()
        finally:
            self.pool._exact_tail_transaction = None
            if self.plan is not None:
                self.plan.exact_tail_inputs = None
            self.states.clear()
            self.empty_states.clear()
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
    if transaction is None:
        raise RuntimeError("exact-tail split needs a transaction")
    if prefix_batch.batch_size and prefix_metadata is None:
        raise RuntimeError("exact-tail nonempty prefix is missing metadata")
    if publication_observer is not None:
        raise ValueError("exact-tail publication observer requires the deferred publication boundary")
    if (boundary_batch.mamba_track_mask is not None
            and bool(boundary_batch.mamba_track_mask.any().item())):
        raise RuntimeError("exact-tail boundary cannot publish decode checkpoints")
    prefix_ids = list(map(int, prefix_batch.req_pool_indices_cpu))
    transaction.prepare_split(prefix_ids, boundary_batch, boundary_metadata)
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
        if prefix_batch.batch_size:
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


@warn_install_rejection("exact-tail")
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

    layerwise_split = pd_shallow_gdn.split_boundary

    @wraps(layerwise_split)
    def dispatch_split(backend, *args, **kwargs):
        # A fallback must retain the old prefix-commit-before-tail adapter too.
        if getattr(backend.factored, "_exact_tail_transaction", None) is None:
            return layerwise_split(backend, *args, **kwargs)
        return split_boundary(backend, *args, **kwargs)

    @wraps(original)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        batch = forward_batch
        if not batch.forward_mode.is_extend():
            return original(input_ids, positions, batch, *args, **kwargs)
        mixed = batch.forward_mode.is_mixed() or getattr(batch, "_pfactor_legacy_mixed", False)
        if (mixed or not 1 <= batch.batch_size <= 16
                or getattr(batch, "can_run_tbo", False)
                or getattr(batch, "tbo_split_seq_index", None) is not None):
            owner._exact_tail_fallbacks = getattr(owner, "_exact_tail_fallbacks", 0) + 1
            logger.warning("GDN exact-tail fallback: count=%d batch_size=%d mixed=%s "
                           "can_run_tbo=%s route=layerwise",
                           owner._exact_tail_fallbacks, batch.batch_size,
                           mixed, getattr(batch, "can_run_tbo", False))
            return original(input_ids, positions, batch, *args, **kwargs)
        with ExactTailTransaction(pool, runner.req_to_token_pool, batch):
            return original(input_ids, positions, batch, *args, **kwargs)

    pd_shallow_gdn.split_boundary = dispatch_split
    owner.forward = forward
    owner._exact_tail_installed = True
    logger.info("GDN exact-tail batch installed: model_depth=%d gdn_layers=%d recurrent_layers=%d",
                limit, len(pool.layer_ids), len(layers))
