"""Real cache install/eviction/refill on CPU; only CUDA driver/I/O leaves emulated.

No AST extraction. These tests do not claim CUDA visibility, model Ready or
needle/NLL quality; those are mandatory in the service handoff.
"""

import contextlib
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import torch
from sglang.srt.environ import envs
from sglang.srt.managers import cache_controller as cc
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hc
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.pool_host import base as hb
from sglang.srt.mem_cache.pool_host import mha as mh
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
    EvictLayer,
)
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

RECEIPT = {
    "cpu_only": True,
    "GPU_model_ready": "NOT_RUN",
    "needle_NLL": "GPU_HANDOFF",
    "arms": [],
    "events": [],
}
PTR = {}


class Event:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        RECEIPT["events"].append("record")

    def query(self):
        return True

    def synchronize(self):
        pass

    def wait(self, *args, **kwargs):
        pass

    def elapsed_time(self, other):
        return 0.0


class Stream:
    cuda_stream = 1

    def __init__(self, *args, **kwargs):
        pass

    def wait_event(self, *args, **kwargs):
        RECEIPT["events"].append("wait_event")

    def wait_stream(self, *args, **kwargs):
        pass

    def synchronize(self):
        pass


class Device:
    Event, Stream = Event, Stream

    @staticmethod
    def current_stream(*args, **kwargs):
        return Stream()

    @staticmethod
    def stream(*args, **kwargs):
        return contextlib.nullcontext()

    @staticmethod
    def synchronize(*args, **kwargs):
        pass


def lf_pf(
    src_k_layers,
    dst_k,
    src_v_layers,
    dst_v,
    src_indices,
    dst_indices,
    item_size,
    dst_layout_dim,
    num_layers,
):
    for srcs, dst in ((src_k_layers, dst_k), (src_v_layers, dst_v)):
        out = dst.view(torch.uint8).reshape(-1, dst_layout_dim)
        for li, ptr in enumerate(srcs.tolist()):
            inp = PTR[ptr].view(torch.uint8).reshape(-1, item_size)
            out[dst_indices.long(), li * item_size : (li + 1) * item_size] = inp[
                src_indices.long()
            ]


def pf_lf(
    src_k,
    dst_k,
    src_v,
    dst_v,
    src_indices,
    dst_indices,
    layer_id,
    item_size,
    src_layout_dim,
):
    for src, dst in ((src_k, dst_k), (src_v, dst_v)):
        inp = src.view(torch.uint8).reshape(-1, src_layout_dim)
        out = dst.view(torch.uint8).reshape(-1, item_size)
        out[dst_indices.long()] = inp[
            src_indices.long(), layer_id * item_size : (layer_id + 1) * item_size
        ]


def mlfpf(
    src_layers, dst, src_indices, dst_indices, item_size, dst_layout_dim, num_layers
):
    lf_pf(
        src_layers,
        dst,
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
    )


def mpflf(src, dst, src_indices, dst_indices, layer_id, item_size, src_layout_dim):
    pf_lf(
        src,
        dst,
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
    )


def digest(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def snapshot(rp):
    pool = rp.mamba_pool
    ts = list(pool.mamba_cache.conv) + [pool.mamba_cache.temporal]
    factor = rp.factored_gdn_pool
    if factor is not None:
        ts += [factor.a, factor.U, factor.W, factor.count, factor.prefix_valid]
    ts += [rp.short_conv_pool.conv_state, rp.ngram_pool.context]
    return [digest(t) for t in ts if t is not None]


@contextlib.contextmanager
def cpu_driver():
    # Driver and native memory-copy leaves only. All production cache, host
    # allocation/layout, pool, assembler, controller and tree methods execute.
    with contextlib.ExitStack() as stack:
        stack.enter_context(get_parallel().override(attn_tp_rank=0, attn_tp_size=1))
        for module in (cc, hc):
            if hasattr(module, "device_module"):
                stack.enter_context(mock.patch.object(module, "device_module", Device))
        from sglang.srt.mem_cache import l2_transfer

        if hasattr(l2_transfer, "device_module"):
            stack.enter_context(mock.patch.object(l2_transfer, "device_module", Device))
        for name in ("empty", "zeros", "ones", "full", "arange"):
            original = getattr(torch, name)

            def make(*args, _fn=original, **kwargs):
                if kwargs.get("pin_memory"):
                    kwargs["pin_memory"] = False
                return _fn(*args, **kwargs)

            stack.enter_context(mock.patch.object(torch, name, make))
        stack.enter_context(
            mock.patch.object(torch.Tensor, "pin_memory", lambda t, *a, **k: t)
        )
        stack.enter_context(
            mock.patch.object(torch.Tensor, "record_stream", lambda *a, **k: None)
        )
        stack.enter_context(
            mock.patch.object(mh, "can_use_hicache_jit_kernel", lambda **kw: False)
        )
        stack.enter_context(
            mock.patch.object(mh, "transfer_kv_all_layer_lf_pf", lf_pf, create=True)
        )
        stack.enter_context(
            mock.patch.object(mh, "transfer_kv_per_layer_pf_lf", pf_lf, create=True)
        )
        import sgl_kernel.kvcacheio as native_io

        stack.enter_context(
            mock.patch.object(
                native_io, "transfer_kv_all_layer_mla_lf_pf", mlfpf, create=True
            )
        )
        stack.enter_context(
            mock.patch.object(
                native_io, "transfer_kv_per_layer_mla_pf_lf", mpflf, create=True
            )
        )
        stack.enter_context(
            mock.patch.object(hb, "host_memory_budget_bytes", lambda: 1 << 40)
        )
        # CPU tensors cannot cudaHostRegister; this does not replace allocation.
        from sglang.srt.mem_cache.pool_host import common as host_common

        stack.enter_context(
            mock.patch.object(host_common, "_cuda_host_register", lambda *a, **k: None)
        )
        stack.enter_context(
            mock.patch.object(
                host_common, "_cuda_host_unregister", lambda *a, **k: None
            )
        )
        yield


def fixture(arm, *, key=True, hc_enabled=True):
    torch.manual_seed(245)
    os.environ["SGLANG_GDN_PROMPT_ONLY_STATE_CACHE"] = "all" if arm == "S" else "1"
    args = ServerArgs(
        model_path="dummy",
        page_size=64,
        enable_hierarchical_cache=hc_enabled,
        hicache_size=0,
        hicache_ratio=3,
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
        hicache_write_policy="write_back",
    )
    args._mamba_cache_chunk_size = 64
    set_global_server_args_for_scheduler(args)
    # Real constructors, 48 layer identities / 36 GDN layers. Small head width
    # bounds CPU fixture memory; production r16/W16 is retained.
    layers = [i for i in range(48) if i % 4 != 3]
    params = NS(
        shape=NS(
            conv=[(64, 3)],
            temporal=(1, 32, 32),
            disable_conv_window_dedup=False,
            conv_kernel=4,
        ),
        dtype=NS(conv=torch.bfloat16, temporal=torch.bfloat16),
        is_kda=False,
        layers=layers,
    )
    rp = HybridReqToTokenPool(
        size=4,
        mamba_size=8,
        mamba_spec_state_size=4,
        max_context_len=512,
        device="cpu",
        enable_memory_saver=False,
        cache_params=params,
        mamba_layer_ids=layers,
        enable_mamba_extra_buffer=True,
        short_conv_layer_ids=[1],
        short_conv_state_shape=(3, 32),
        ngram_context_len=4,
        linear_attn_factored_state=(
            None
            if arm == "S"
            else "r=16,m=16,dtype=fp16,ring=4,strict_chunk=1,factored_prefix=1"
        ),
    )
    kv = QSATokenToKVPool(
        size=256,
        dtype=torch.bfloat16,
        page_size=64,
        head_num=1,
        head_dim=16,
        full_attention_layer_ids=[i for i in range(48) if i % 4 == 3],
        device="cpu",
        mamba_pool=rp.mamba_pool,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=16,
        qsa_compress_ratio=4,
        qsa_token_topk=64,
        num_request_slots=5,
    )
    alloc = PagedTokenToKVPoolAllocator(
        size=256,
        page_size=64,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv,
        need_sort=False,
    )
    cp = CacheInitParams(
        req_to_token_pool=rp,
        token_to_kv_pool_allocator=alloc,
        page_size=64,
        disable=False,
        tree_components=(CT.FULL, CT.MAMBA),
        enable_mamba_extra_buffer=True,
        eviction_policy="lru",
    )
    registered = []
    worker = NS(register_hicache_layer_transfer_counter=registered.append)
    cache = create_tree_cache(
        TreeCacheBuildContext(
            server_args=args,
            params=cp,
            is_hybrid_swa=False,
            is_hybrid_ssm=True,
            enable_hierarchical_cache=hc_enabled,
            disable_radix_cache=False,
            effective_chunked_prefill_size=32768,
            tp_worker=worker,
            model_config=NS(),
            tp_size=1,
            tp_rank=0,
            tp_group=None,
        )
    )
    assert len(registered) == int(hc_enabled)
    for t in (
        kv.full_kv_pool.k_buffer
        + kv.full_kv_pool.v_buffer
        + kv.qsa_compressed_k_buffer_pool
    ):
        t.normal_()
        PTR[t.data_ptr()] = t
    for t in rp.mamba_pool.mamba_cache.conv:
        t.normal_()
    if rp.mamba_pool.mamba_cache.temporal.numel():
        rp.mamba_pool.mamba_cache.temporal.normal_()
    if rp.factored_gdn_pool is not None:
        f = rp.factored_gdn_pool
        f.a.normal_()
        f.U.normal_()
        f.W.normal_()
        f.prefix_valid.fill_(1)
        f.count.fill_(17)
        if arm == "PC+":
            f.count[24:] = 16
    rp.short_conv_pool.conv_state.normal_()
    rp.ngram_pool.context.fill_(7)
    return cache, alloc, rp, kv, args


class Tests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(envs.SGLANG_FLASHNEXT_HICACHE_KV_ONLY.override(True))
        self.stack.enter_context(
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python")
        )
        self.stack.enter_context(cpu_driver())

    def make(self, arm="C+", hc_enabled=True):
        parts = fixture(arm, hc_enabled=hc_enabled)
        self.addCleanup(parts[0].release_host_resources)
        return parts

    def node(self, c, a, rp, tokens=None):
        tokens = list(range(64)) if tokens is None else tokens
        values = a.alloc(len(tokens))
        states = rp.mamba_allocator.alloc(1)
        c.insert(InsertParams(key=RadixKey(tokens), value=values, mamba_value=states))
        result = c.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens), cow_mamba=False)
        )
        return c.tree_core.node_by_id(result.best_match_node), values, states

    def backup_demote(self, c, node):
        # Real controller and cache commit path. CPU I/O leaf emulates DMA.
        n = c._execute_and_commit_kv_backup(
            c.tree_core._build_backup_kv_action(node), True
        )
        self.assertGreater(n, 0)
        c.writing_check(write_back=True)
        self.assertIsNotNone(node.component_data[CT.FULL].host_value)
        result = c.tree_core.demote(node.id)
        c._free_values(result.device_frees, result.host_frees)
        return result

    def test_three_arm_ready_and_host_refill(self):
        for arm in ("S", "C+", "PC+"):
            with self.subTest(arm=arm):
                c, a, rp, kv, _ = self.make(arm)
                self.assertTrue(c.components[CT.MAMBA].retain_on_kv_host)
                self.assertEqual(len(c.host_pool_group.entries), 1)
                self.assertEqual(c.cache_controller.layer_num, 48)
                self.assertIsNone(rp.layer_transfer_counter)
                node, values, states = self.node(c, a, rp)
                before = snapshot(rp)
                original = [
                    t[values].clone()
                    for t in kv.full_kv_pool.k_buffer + kv.full_kv_pool.v_buffer
                ]
                qsa_before = [
                    t[values[:: kv.qsa_compress_ratio] // kv.qsa_compress_ratio].clone()
                    for t in kv.qsa_compressed_k_buffer_pool
                ]
                self.backup_demote(c, node)
                self.assertIsNone(node.component_data[CT.FULL].value)
                self.assertTrue(
                    torch.equal(node.component_data[CT.MAMBA].value, states)
                )
                self.assertEqual(snapshot(rp), before)
                match = c.match_prefix(
                    MatchPrefixParams(key=RadixKey(list(range(64))), cow_mamba=False)
                )
                self.assertEqual(match.host_hit_length, 64)
                self.assertEqual(match.mamba_host_hit_length, 0)
                c.sanity_check()
                loaded = c.load_back(node.id)
                self.assertIsNotNone(loaded)
                c.ready_to_load_host_cache()
                c.loading_check()
                dst = node.component_data[CT.FULL].value
                self.assertIsNotNone(dst)
                for t, old in zip(
                    kv.full_kv_pool.k_buffer + kv.full_kv_pool.v_buffer,
                    original,
                    strict=True,
                ):
                    self.assertTrue(
                        torch.equal(t[dst].view(torch.uint8), old.view(torch.uint8))
                    )
                self.assertEqual(snapshot(rp), before)
                for t, old in zip(
                    kv.qsa_compressed_k_buffer_pool, qsa_before, strict=True
                ):
                    self.assertTrue(
                        torch.equal(
                            t[
                                dst[:: kv.qsa_compress_ratio] // kv.qsa_compress_ratio
                            ].view(torch.uint8),
                            old.view(torch.uint8),
                        )
                    )
                c.sanity_check()
                RECEIPT["arms"].append(
                    dict(
                        arm=arm,
                        cache_ready=True,
                        host_hit=64,
                        state_host_hit=0,
                        refill_bytes_equal=True,
                        state_unchanged=True,
                        scope="CPU production cache + native I/O shim",
                    )
                )

    def test_state_eviction_turns_host_kv_into_miss(self):
        c, a, rp, _, _ = self.make()
        node, _, _ = self.node(c, a, rp)
        self.backup_demote(c, node)
        c.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertIsNone(node.component_data[CT.MAMBA].value)
        result = c.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(64))), cow_mamba=False)
        )
        self.assertEqual(len(result.device_indices) + result.host_hit_length, 0)

    def test_real_hot_hit_cow_and_factor_restore_bytes(self):
        from array import array

        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.gdn_factored_pool import FactoredExtendPlan
        from sglang.srt.sampling.sampling_params import SamplingParams

        for arm in ("S", "C+", "PC+"):
            c, a, rp, _, _ = self.make(arm)
            node, _, src = self.node(c, a, rp)
            self.backup_demote(c, node)
            req = Req(
                rid=f"hicache-hot-{arm}",
                origin_input_text="cache-prefix",
                origin_input_ids=array("q", range(65)),
                sampling_params=SamplingParams(max_new_tokens=1),
            )
            hit = c.match_prefix(
                MatchPrefixParams(key=RadixKey(list(range(64))), req=req)
            )
            self.assertEqual(hit.host_hit_length, 64)
            dst = req.kv.mamba_pool_idx.view(-1)
            rp.mamba_pool.copy_from(
                rp.translate_mamba_indices(req.kv.mamba_cow_src_index),
                rp.translate_mamba_indices(dst),
            )
            for tensor in (
                *rp.mamba_pool.mamba_cache.conv,
                rp.mamba_pool.mamba_cache.temporal,
                rp.short_conv_pool.conv_state,
            ):
                if tensor.numel():
                    self.assertTrue(torch.equal(tensor[:, src], tensor[:, dst]))
            self.assertTrue(
                torch.equal(rp.ngram_pool.context[src], rp.ngram_pool.context[dst])
            )
            f = rp.factored_gdn_pool
            if f is not None:
                for tensor in (f.a, f.U, f.W, f.count):
                    self.assertTrue(torch.equal(tensor[:, src], tensor[:, dst]))
                self.assertEqual(f.dense_of[dst].item(), -1)
                plan = FactoredExtendPlan(
                    slots=dst,
                    use_ring=torch.tensor([False]),
                    ring_src=torch.tensor([0]),
                    ring_dst=torch.tensor([0]),
                    ring_dst_rows=torch.tensor([0]),
                    n_ring_src=0,
                    all_fresh=False,
                    last_layer=35,
                )
                for lid in f.layer_ids:
                    self.assertTrue(
                        torch.equal(
                            f.initial_dense(lid, plan), f.dense_of_slots(lid, src)
                        )
                    )
            rp.mamba_allocator.free(dst)
            c.sanity_check()

    def test_host_kv_uses_last_resident_ancestor(self):
        c, a, rp, _, _ = self.make()
        parent, _, parent_state = self.node(c, a, rp)
        long_key = list(range(128))
        child, _, _child_state = self.node(c, a, rp, long_key)
        self.backup_demote(c, child)
        # Explicit eviction of only the deeper checkpoint, through the
        # production component path; its host KV remains a legal tombstone.
        from collections import defaultdict

        frees = defaultdict(list)
        hosts = defaultdict(list)
        tracker = defaultdict(int)
        c.tree_core._evict_component_and_detach_lru(
            child,
            c.components[CT.MAMBA],
            target=EvictLayer.DEVICE,
            tracker=tracker,
            device_frees=frees,
            host_frees=hosts,
        )
        c._free_values(frees, hosts)
        match = c.match_prefix(
            MatchPrefixParams(key=RadixKey(long_key), cow_mamba=False)
        )
        self.assertEqual(match.best_match_node, parent.id)
        self.assertEqual(len(match.device_indices) + match.host_hit_length, 64)
        self.assertTrue(
            torch.equal(parent.component_data[CT.MAMBA].value, parent_state)
        )
        c.sanity_check()

    def test_no_state_host_transfers(self):
        c, a, rp, _, _ = self.make()
        node, _, _ = self.node(c, a, rp)
        m = c.components[CT.MAMBA]
        self.assertFalse(m.needs_incremental_backup(node))
        for phase in CacheTransferPhase:
            self.assertIsNone(m.build_hicache_transfers(node, phase))

    def test_host_eviction_releases_resident_state(self):
        c, a, rp, _, _ = self.make()
        node, _, states = self.node(c, a, rp)
        self.backup_demote(c, node)
        n = rp.mamba_allocator.available_size()
        c.evict_host(64, CT.FULL)
        self.assertEqual(rp.mamba_allocator.available_size(), n + len(states))

    def test_no_hicache_even_key_on_is_byte_identical(self):
        from sglang.srt.mem_cache.flashnext_hicache_policy import kv_only_enabled

        for arm in ("S", "C+", "PC+"):
            _, _, rp1, kv1, _ = self.make(arm, hc_enabled=False)
            s1 = snapshot(rp1)
            self.assertFalse(kv_only_enabled())
            self.assertFalse(hasattr(kv1.full_kv_pool, "_hicache_qsa_owner"))
            with envs.SGLANG_FLASHNEXT_HICACHE_KV_ONLY.override(False):
                _, _, rp0, kv0, _ = self.make(arm, hc_enabled=False)
                self.assertEqual(s1, snapshot(rp0))
                self.assertFalse(hasattr(kv0.full_kv_pool, "_hicache_qsa_owner"))

    def test_default_http_warmup_ready_uses_real_cache(self):
        # Import and execute the original warmup functions, including their
        # default callback. Only HTTP transport/GPU compute are fixtures.
        from sglang.srt.entrypoints import http_server as http

        for arm in ("S", "C+", "PC+"):
            c, a, rp, _kv, args = self.make(arm)
            called = []
            ready = []
            tokenizer = NS(server_status=None)
            response = NS(
                status_code=200,
                text="ok",
                json=lambda: {"is_generation": True},
                raise_for_status=lambda: None,
            )

            def post(url, c=c, a=a, rp=rp, called=called, response=response, **kwargs):
                if url.endswith("/generate"):
                    self.assertEqual(
                        kwargs["json"]["sampling_params"]["max_new_tokens"], 8
                    )
                    node, _, _ = self.node(c, a, rp)
                    self.backup_demote(c, node)
                    self.assertTrue(c.load_back(node.id))
                    c.ready_to_load_host_cache()
                    c.loading_check()
                    called.append("real-install-backup-refill")
                return response

            with (
                mock.patch.object(http.requests, "get", return_value=response),
                mock.patch.object(http.requests, "post", side_effect=post),
                mock.patch.object(http.time, "sleep", lambda _: None),
                mock.patch.object(
                    http, "_global_state", NS(tokenizer_manager=tokenizer)
                ),
                mock.patch.object(
                    http,
                    "kill_process_tree",
                    side_effect=AssertionError("warmup killed server"),
                ),
            ):
                http._wait_and_warmup(args, lambda ready=ready: ready.append(True))
            self.assertEqual(called, ["real-install-backup-refill"])
            self.assertEqual(ready, [True])
            self.assertEqual(tokenizer.server_status, http.ServerStatus.Up)
            RECEIPT.setdefault("default_warmup", []).append(arm)

    def test_refill_pins_checkpoint_until_ack(self):
        c, a, rp, _, _ = self.make()
        node, _, states = self.node(c, a, rp)
        self.backup_demote(c, node)
        self.assertTrue(c.load_back(node.id))
        self.assertGreater(node.component_data[CT.MAMBA].lock_ref, 0)
        c.evict(EvictParams(num_tokens=0, mamba_num=8))
        self.assertTrue(torch.equal(node.component_data[CT.MAMBA].value, states))
        c.ready_to_load_host_cache()
        c.loading_check()
        self.assertEqual(node.component_data[CT.MAMBA].lock_ref, 0)
        c.evict(EvictParams(num_tokens=0, mamba_num=8))
        self.assertIsNone(node.component_data[CT.MAMBA].value)

    def test_failed_refill_releases_locks(self):
        c, a, rp, _, _ = self.make()
        node, _, _ = self.node(c, a, rp)
        self.backup_demote(c, node)
        self.assertFalse(c.load_back(node.id, mem_quota=0))
        self.assertEqual(node.component_data[CT.MAMBA].lock_ref, 0)
        self.assertEqual(node.component_data[CT.FULL].host_lock_ref, 0)
        self.assertTrue(c.load_back(node.id))
        c.ready_to_load_host_cache()
        c.loading_check()

    def test_fail_closed_layout_backend(self):
        from sglang.srt.mem_cache.flashnext_hicache_policy import validate_kv_only_stack

        c, _, _, kv, _ = self.make()
        for name, value in [
            ("hicache_io_backend", "direct"),
            ("hicache_mem_layout", "layer_first"),
            ("hicache_write_policy", "write_through"),
            ("hicache_host_memory_mode", "buffer_only"),
        ]:
            with get_memory().override(**{name: value}):
                with self.assertRaises(ValueError):
                    validate_kv_only_stack(c, kv.full_kv_pool, get_memory(), None)
        with self.assertRaises(ValueError):
            validate_kv_only_stack(c, kv.full_kv_pool, get_memory(), "file")
        c._tree_core_backend = "rust"
        with self.assertRaises(ValueError):
            validate_kv_only_stack(c, kv.full_kv_pool, get_memory(), None)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    RECEIPT.update(
        passed=result.wasSuccessful(),
        tests=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
    )
    out = os.environ.get("PFACTOR245_RECEIPT")
    if out:
        Path(out).write_text(json.dumps(RECEIPT, indent=2) + "\n")
    sys.exit(not result.wasSuccessful())
