"""Isolated, startup-captured recurrent tails for k31 model splits."""

import copy
import json
import logging
import os
import time

import torch

from sglang.srt.model_executor import pd_tail_cuda_graph_runner as native

FLAG = "SGLANG_GDN_PREFILL_MODEL_TAIL_GRAPH"
LEGACY_FLAG = "SGLANG_GDN_PREFILL_TAIL_GRAPH"
BUCKETS = (1, 2, 4, 8, 16)
logger = logging.getLogger(__name__)


class TailBody:
    def __init__(self, owner):
        self.owner = owner
        self.body = owner.model.model
        self.shallow = getattr(owner, "pd_shallow_role", None) == "prefill"
        self.original = self.body.forward
        self.layers = 31 if self.shallow else 48

    def forward(self, input_ids, positions, batch):
        if self.shallow:
            from sglang.srt.model_executor.gdn_prefill_model_split import _shallow_hidden

            return _shallow_hidden(self.owner, batch)[0]
        return self.original(input_ids, positions, batch)

    @property
    def last_hc_hidden_states(self):
        return None if self.shallow else self.body.last_hc_hidden_states

    @last_hc_hidden_states.setter
    def last_hc_hidden_states(self, value):
        if not self.shallow:
            self.body.last_hc_hidden_states = value


class CaptureState:
    def __init__(self, runner):
        rp, kv = runner.req_to_token_pool, runner.token_to_kv_pool
        self.rows = []
        zero = torch.zeros(1, dtype=torch.int64, device=runner.device)
        slots = rp.translate_mamba_indices(rp.get_mamba_indices(zero)).long()
        slots = torch.unique(torch.cat((zero, slots.clamp_min(0))))
        for _, tensor, _, _ in rp.mamba_pool._iter_transfer_state_entries():
            self.add(tensor, slots)
        pool = rp.factored_gdn_pool
        for name in ("stale", "dense_of", "dense_required", "prefix_valid"):
            value = getattr(pool, name, None)
            if value is not None:
                self.add(value, slots)
        self.add(pool.dense_ring)
        self.storage = [(pool, name, getattr(pool, name)) for name in
                        ("dense_ring", "a", "U", "W", "count") if hasattr(pool, name)]
        self.host = [(pool, name, copy.deepcopy(getattr(pool, name)))
                     for name in ("ring_owner", "ring_lru", "ring_generation", "stats")]
        self.ple_cache = getattr(rp, "ple_window_cache", None)
        self.rp = rp
        pending = torch.arange(4, device=runner.device)
        for layer in range(3, 48, 4):
            local = kv._transfer_full_attention_id(layer)
            self.add(kv.get_key_buffer(layer), zero)
            self.add(kv.get_value_buffer(layer), zero)
            self.add(kv.get_qsa_compressed_k_buffer(layer), zero)
            self.add(kv.qsa_key_state_buffer_pool[local], pending)
        self.add(kv.qsa_rope_position_buffer, pending)
        self.ple = [layer.ple for layer in runner.model.model.model.layers
                    if getattr(layer, "ple", None) is not None]
        if any(getattr(layer, "_prefetch_state", None) is not None for layer in self.ple):
            raise RuntimeError("tail capture requires no pending PLE prefetch")

    def add(self, tensor, indices=None):
        value = tensor if indices is None else tensor[indices]
        self.rows.append((tensor, indices, value.detach().cpu().clone()))

    def restore(self):
        for tensor, indices, value in self.rows:
            if indices is None:
                tensor.copy_(value)
            else:
                tensor[indices] = value.to(tensor.device)
        for owner, name, value in self.host:
            setattr(owner, name, copy.deepcopy(value))
        self.rp.ple_window_cache = self.ple_cache
        for layer in self.ple:
            if getattr(layer, "_prefetch_state", None) is not None:
                raise RuntimeError("tail capture left an unconsumed PLE prefetch")

    def check(self):
        for owner, name, tensor in self.storage:
            if getattr(owner, name) is not tensor:
                raise RuntimeError("tail capture replaced graph-owned state: " + name)
        for tensor, indices, value in self.rows:
            actual = tensor if indices is None else tensor[indices]
            if not torch.equal(value.contiguous().view(torch.uint8),
                               actual.detach().cpu().contiguous().view(torch.uint8)):
                raise RuntimeError("tail capture failed to restore reserved state")
        for owner, name, value in self.host:
            if getattr(owner, name) != value:
                raise RuntimeError("tail capture failed to restore " + name)


def memory_plan(runner, *, free_bytes, total_bytes):
    config = runner.model_config.hf_text_config
    tokens = max(runner.server_args.max_prefill_tokens or 0,
                 runner.server_args.chunked_prefill_size or 0)
    live_io = 4 * tokens * config.hidden_size * (1 + config.hc_count) + tokens * 64
    reserved = (live_io + native.GRAPH_LIMIT_BYTES + native.METADATA_LIMIT_BYTES
                + native.FORWARD_RESERVE_BYTES)
    result = dict(device_total_bytes=total_bytes, free_bytes=free_bytes,
        existing_factored_pool_bytes=runner.req_to_token_pool.factored_gdn_pool.mem_usage_bytes(),
        split_input_output_bound_bytes=live_io, graph_limit_bytes=native.GRAPH_LIMIT_BYTES,
        metadata_limit_bytes=native.METADATA_LIMIT_BYTES,
        forward_peak_reserve_bytes=native.FORWARD_RESERVE_BYTES,
        projected_free_bytes=free_bytes-reserved, headroom_bytes=native.HEADROOM_BYTES,
        commit_graphs_already_allocated=True, extra_dense_checkpoint_bytes=0)
    if tokens <= 0 or result["projected_free_bytes"] < native.HEADROOM_BYTES:
        raise RuntimeError("k31 tail graph memory budget: " + str(result))
    return result


def install(runner):
    if os.environ.get(FLAG) != "1":
        return False
    if os.environ.get(LEGACY_FLAG) == "1":
        raise ValueError("k31 model tail graph requires the per-layer tail graph disabled")
    if (runner.server_args.disaggregation_mode != "prefill"
            or not getattr(runner.model, "_gdn_prefill_model_split_installed", False)):
        raise ValueError("k31 tail graph requires the installed P model split")
    if getattr(runner.model, "_gdn_prefill_tail_graph", None) is not None:
        return True
    from sglang.srt.utils.graph_capture import graph_capture_lock

    body = TailBody(runner.model)
    started = time.monotonic()
    state = CaptureState(runner)
    try:
        with graph_capture_lock:
            graph = native.make_runner(runner, body=body, buckets=BUCKETS,
                                       plan_factory=memory_plan, layers=body.layers)
    finally:
        state.restore()
    state.check()
    expected = {graph._make_graph_key(size) for size in BUCKETS}
    if list(graph.capture_bs) != list(BUCKETS) or set(graph.backend._graphs) != expected:
        raise RuntimeError("k31 tail prewarm is missing required buckets")
    graph.memory_receipt.update(capture_state_restored=True,
        capture_seconds=time.monotonic()-started, formal_capture_allowed=False,
        captured_buckets=sorted(key.size for key in graph.backend._graphs))
    runner.model._gdn_prefill_tail_graph = graph
    logger.info("GDN model tail graph prewarmed: %s", json.dumps(graph.memory_receipt, sort_keys=True))
    return True


def execute(owner, batch):
    if os.environ.get(FLAG) != "1":
        return None
    graph = getattr(owner, "_gdn_prefill_tail_graph", None)
    if graph is None:
        raise RuntimeError("enabled k31 tail graph was not captured at startup")
    graph.communication.check()
    if not native.eligible(batch, max(BUCKETS)):
        raise ValueError("k31 tail graph requires native text decode metadata")
    return graph.execute_tail(batch)
