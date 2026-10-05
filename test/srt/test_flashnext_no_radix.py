"""Real imports/install/capture/warmup and factor publication on CPU.

Only CUDA driver/native store kernels and model weights are emulated. No AST
extraction and no whole capture_cuda_graphs replacement. This is not CUDA model
Ready, neural needle/NLL or GPU visibility evidence.
"""

import contextlib
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.mem_cache import gdn_factored_pool as fp
from sglang.srt.mem_cache import gdn_prefill_batch_graph as bg
from sglang.srt.mem_cache import gdn_prefill_commit_graph as cg
from sglang.srt.layers.attention.linear.kernels import gdn_factored_io as io
from sglang.srt.model_executor import fullstack_policy as policy
from sglang.srt.model_executor import model_runner as mr
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as setup
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.models.flash_next_duet.model import (
    Qwen4ExpForConditionalGeneration as Model,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.disaggregation.state_handoff import FactorStateHandoff

KEY = "SGLANG_FLASHNEXT_NO_RADIX_FACTORS"
IDS = [i for i in range(48) if i % 4 != 3]
EVENTS = []
ROOT = Path(__file__).resolve().parents[2]


def load_base(name, relative):
    path = Path(os.environ["PFACTOR247_BASE"]) / relative
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old_policy = load_base(
    "sglang.srt.model_executor._no_radix_old_policy",
    "python/sglang/srt/model_executor/fullstack_policy.py",
)
old_pool = load_base(
    "sglang.srt.mem_cache._no_radix_old_pool",
    "python/sglang/srt/mem_cache/gdn_factored_pool.py",
)
old_batch = load_base(
    "sglang.srt.mem_cache._no_radix_old_batch",
    "python/sglang/srt/mem_cache/gdn_prefill_batch_graph.py",
)


def sha(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def config(arm="C+", *, reference=False, prefix="factored"):
    fs = dict(
        release="/release",
        version=3,
        latent_rank=8192,
        latent_sparse=256,
        latent_store="nvfp4",
        state_sink="explicit",
        latent_id_side=True,
        latent_value_format="bf16",
        latent_index_format="gap8",
        latent_rms=False,
        gdn_state="dense" if arm == "S" else "rank:16",
        gdn_rank=0 if arm == "S" else 16,
        gdn_every=16,
        prefill_layer_trim=arm == "PC",
        prefill_saving_policy="kv-and-ssm",
        qsa_code="off",
        state_sink_vbar=__file__,
        duet_prefix_state=prefix,
        duet_state_truncation="reference-warm" if reference else "factored-iter",
        duet_spec=dict(
            latent_rank=8192,
            latent_spikes=256,
            latent_z_format="nvfp4",
            state_sink="explicit",
            latent_id_side=True,
            latent_value_format="bf16",
            latent_index_format="gap8",
        ),
    )
    return NS(hf_config=NS(twinstar={"fullstack": fs}))


def pool_from(raw, module=fp, *, actual=False):
    cfg = module.FactoredGDNConfig.parse(raw)
    if cfg is None:
        return None
    # Small head/width fixture runs the production solver; rank/metadata/36
    # layer coverage is unchanged. Actual TP1 allocation has a separate test.
    cfg = copy.copy(cfg)
    cfg.ring = 1
    shape = NS(temporal=(48, 128, 128) if actual else (1, 32, 32))
    h, v, _ = shape.temporal
    sink = Path(os.environ["LOWC_OUT"]) / f"sink-{h}-{v}.pt"
    if not sink.exists():
        gen = torch.Generator().manual_seed(247)
        torch.save(
            {"vbar": {lid: torch.randn(h, v, generator=gen) for lid in IDS}}, sink
        )
    cfg.vbar_path = str(sink)
    p = module.FactoredGDNPool(
        size=2, cache_params=NS(shape=shape), mamba_layer_ids=IDS, device="cpu", cfg=cfg
    )
    return p


def plan(p):
    return fp.FactoredExtendPlan(
        slots=torch.tensor([1]),
        use_ring=torch.tensor([False]),
        ring_src=torch.tensor([0]),
        ring_dst=torch.tensor([0]),
        ring_dst_rows=torch.tensor([0]),
        last_layer=35,
        all_fresh=True,
        dense_required_after_commit=torch.tensor([0], dtype=torch.int32),
    )


class Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


def publish_valid(valid, slots, *args):
    assert valid is not None, "NO_PREFIX_WRITE"
    EVENTS.append("prefix-valid")
    valid[slots[slots >= 0]] = 1


def scatter(src, dst, slots):
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            dst[:, slot] = src[:, row]


def store(
    a, u, w, da, du, dw, count, stale, dense_of, slots, r, *, stale_value=1, **kw
):
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        da[slot] = a[row]
        du[slot] = u[row]
        dw[slot] = w[row]
        count[slot] = r
        stale[slot] = stale_value
        # CPU native-kernel stand-in: tests compare factor bytes/count and
        # publication metadata; device ring pointer dereference is separate.
        if kw.get("ring_dst") is not None:
            dense_of[slot] = kw["ring_dst"][row]


class Driver:
    def __init__(self):
        self.active = None
        self.graphs = []

    def stream(self, *a, **kw):
        return NS(wait_stream=lambda *a: None, cuda_stream=7)

    @contextlib.contextmanager
    def graph(self, g, **kw):
        self.active = g
        try:
            yield
        finally:
            self.active = None

    def new_graph(self):
        g = NS(actions=[], pool=lambda: (0, 0))
        g.replay = lambda: [fn(*args) for fn, args in g.actions]
        self.graphs.append(g)
        return g

    @contextlib.contextmanager
    def use(self):
        with contextlib.ExitStack() as st:
            st.enter_context(
                patch.object(torch.Tensor, "is_cuda", property(lambda t: True))
            )
            for name, fn in dict(
                is_current_stream_capturing=lambda: False,
                current_stream=self.stream,
                Stream=self.stream,
                stream=lambda *a, **kw: contextlib.nullcontext(),
                CUDAGraph=self.new_graph,
                graph=self.graph,
                synchronize=lambda *a: None,
                memory_allocated=lambda *a: 0,
                memory_reserved=lambda *a: 0,
                graph_pool_handle=lambda: object(),
            ).items():
                st.enter_context(patch.object(torch.cuda, name, fn))
            st.enter_context(patch.object(fp, "k31_graph_safe", lambda *a: True))
            st.enter_context(patch.object(cg, "k31_graph_safe", lambda *a: True))
            st.enter_context(patch.object(io, "store_factored", store))
            st.enter_context(patch.object(cg, "_publish_valid", Kernel(publish_valid)))
            st.enter_context(patch.object(bg, "_publish_valid", Kernel(publish_valid)))
            st.enter_context(
                patch.object(old_batch, "_publish_valid", Kernel(publish_valid))
            )
            st.enter_context(patch.object(bg, "scatter_rows", scatter))
            st.enter_context(patch.object(old_batch, "scatter_rows", scatter))
            # Representative bucket capture invokes evaluate, not a fake
            # prewarm. A separate assertion verifies all production buckets.
            st.enter_context(patch.object(cg, "BATCH_BUCKETS", (1, 2)))
            st.enter_context(patch.object(bg, "BATCH_BUCKETS", (1, 2)))
            for cls in (bg.BatchBuffers, cg.CommitBuffers):
                original = cls.evaluate

                def evaluate(buffers, eager, _fn=original):
                    result = _fn(buffers, eager)
                    if self.active is not None:
                        self.active.actions.append((_fn, (buffers, eager)))
                    return result

                st.enter_context(patch.object(cls, "evaluate", evaluate))
            yield self


def owner_for(arm):
    fs = config(arm).hf_config.twinstar["fullstack"]
    o = NS(
        fullstack=fs,
        twinstar={"fullstack": fs},
        fullstack_v3_latent=False,
        fullstack_final=True,
        pd_shallow_role=None,
        n_layers=48,
        p_layer_ids=list(range(48)),
        emitters={str(i): object() for i in range(31, 48)},
        emitter_ids=list(range(31, 48)),
        config=NS(
            layers_block_type=["attention" if i % 4 == 3 else "gdn" for i in range(48)]
        ),
        model=NS(model=NS(layers=[None] * 48), forward=lambda *a, **k: None),
        forward=lambda *a, **k: None,
        prepare_forward_batch=lambda fb: None,
    )
    o._emit_ids = MethodType(Model._emit_ids, o)
    o.prepare_before_cuda_graph_capture = MethodType(
        Model.prepare_before_cuda_graph_capture, o
    )
    return o


@contextlib.contextmanager
def capture_leaves(runner, *, overlap=False):
    # Real ModelRunner.init_cuda_graphs -> capture_cuda_graphs -> warmup ->
    # imported native prepare hook. Only static model buffers/device work are
    # leaves. Crucially the whole capture function is NOT replaced.
    def eager(mr):
        e = NS(
            model_runner=mr,
            _pre_initialize_flashinfer_allreduce_workspace=lambda: None,
            _pre_initialize_fi_a2a_workspace=lambda: None,
        )
        BaseRunner.warmup(e)
        EVENTS.append("warmup")
        return e

    with contextlib.ExitStack() as st:
        st.enter_context(
            patch.object(
                setup.GraphSharedOutput, "create_for_model_runner", return_value=NS()
            )
        )
        st.enter_context(patch.object(setup, "EagerRunner", side_effect=eager))
        st.enter_context(patch.object(setup, "refresh_deep_gemm_layout_memory_budget"))
        st.enter_context(
            patch.object(
                setup,
                "capture_prefill_graph",
                return_value=setup.GraphCapture(
                    runner=None,
                    memory_phase="prefill",
                    memory_usage_gb=0,
                    capture_time=0,
                ),
            )
        )
        st.enter_context(
            patch.object(
                setup,
                "capture_decode_graph",
                return_value=setup.GraphCapture(
                    runner=None,
                    memory_phase="decode",
                    memory_usage_gb=0,
                    capture_time=0,
                ),
            )
        )
        st.enter_context(patch.object(setup, "prealloc_symmetric_memory_pool"))
        st.enter_context(
            patch.object(
                setup, "get_observability", return_value=NS(forward_hooks=None)
            )
        )
        st.enter_context(
            patch.object(
                setup, "get_exec", return_value=NS(comm=NS(enable_symm_mem=False))
            )
        )
        st.enter_context(
            patch(
                "sglang.srt.model_executor.runner.base_runner.should_run_flashinfer_autotune",
                return_value=False,
            )
        )
        st.enter_context(envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.override(False))
        # Capture hooks are restored between real installer invocations.
        for cls, names in (
            (ForwardBatch, ["init_new", "_pfactor_agg_contract_installed"]),
            (ScheduleBatch, ["_mamba_radix_cache_v2_req_prepare_for_extend"]),
            (GDNAttnBackend, ["init_forward_metadata"]),
            (FactorStateHandoff, ["before_send"]),
        ):
            for name in names:
                st.enter_context(
                    patch.object(cls, name, cls.__dict__.get(name, False), create=True)
                )
        st.enter_context(
            patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=NS(disable_overlap_schedule=not overlap),
            )
        )
        yield


def http_warmup(pool, arm, radix):
    from sglang.srt.entrypoints import http_server as http
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(model_path="/model", device="cpu", disable_radix_cache=not radix)
    called, ready = [], []
    tokenizer = NS(server_status=None)
    response = NS(
        status_code=200,
        text="ok",
        json=lambda: {"is_generation": True},
        raise_for_status=lambda: None,
    )

    def post(url, **kwargs):
        if url.endswith("/generate"):
            assert kwargs["json"]["sampling_params"]["max_new_tokens"] == 8
            if pool is not None:
                pool.reset_slots(torch.tensor([1]))
                pl = pool.plan_extend(
                    torch.tensor([1]), [64], prefix_lens=[0], prompt_final=[True]
                )
                collector = (
                    bg.BatchCollector(pool, pl, graph=pool._agg_prefill_graph)
                    if arm == "C+"
                    else contextlib.nullcontext()
                )
                with collector:
                    for lid in IDS:
                        dense = pool.initial_dense(lid, pl)
                        pool.commit_extend_batched(lid, pl, dense)
                assert bool(torch.all(pool.count[:, 1] == pool.cfg.r))
                if not radix:
                    assert pool.prefix_valid is None
            called.append("real-plan-commit")
        return response

    with (
        # HTTP runtime bags are process-local configuration leaves. Keep the
        # actual default warmup functions and their Ready decision intact.
        patch.object(
            http,
            "get_model",
            return_value=NS(
                checkpoint_engine_wait_weights_before_ready=False,
                delete_ckpt_after_loading=False,
                model_path="/model",
            ),
        ),
        patch.object(
            http, "get_exec", return_value=NS(moe=NS(is_ep_scale_joiner=False))
        ),
        patch.object(
            http,
            "get_serving",
            return_value=NS(
                api_key=None,
                admin_api_key=None,
                skip_server_warmup=False,
                skip_tokenizer_init=False,
            ),
        ),
        patch.object(
            http,
            "get_disagg",
            return_value=NS(
                disaggregation_mode="null",
                language_only=False,
                language_model_only=False,
            ),
        ),
        patch.object(http, "get_parallel", return_value=NS(dp_size=1)),
        patch.object(
            http,
            "get_observability",
            return_value=NS(debug_tensor_dump_input_file=None),
        ),
        patch.object(http.requests, "get", return_value=response),
        patch.object(http.requests, "post", side_effect=post),
        patch.object(http.time, "sleep", lambda _: None),
        patch.object(http, "_global_state", NS(tokenizer_manager=tokenizer)),
        patch.object(
            http, "kill_process_tree", side_effect=AssertionError("warmup failed")
        ),
    ):
        http._wait_and_warmup(args, lambda: ready.append(True))
    assert called == ["real-plan-commit"] and ready == [True]
    assert tokenizer.server_status == http.ServerStatus.Up
    EVENTS.append(dict(arm=arm, radix=radix, default_http_warmup=True))


class NoRadixTest(unittest.TestCase):
    def test_policy_three_arms_two_radix_two_keys_and_role_scope(self):
        for arm in ("S", "C+", "PC"):
            for radix in (False, True):
                for enabled in ("0", "1"):
                    with (
                        self.subTest(arm=arm, radix=radix, enabled=enabled),
                        patch.dict(os.environ, {KEY: enabled}),
                    ):
                        new = policy.fullstack_state_config(config(arm), radix=radix)
                        old = old_policy.fullstack_state_config(
                            config(arm), radix=radix
                        )
                        if arm == "S" or radix or enabled == "0":
                            self.assertEqual(new, old)
                        else:
                            cfg = fp.FactoredGDNConfig.parse(new)
                            self.assertEqual(cfg.dtype, torch.float16)
                            self.assertEqual(
                                (cfg.no_radix, cfg.factored_prefix, cfg.exact_prefix),
                                (1, 0, 0),
                            )
        with patch.dict(os.environ, {KEY: "1"}):
            for role in ("prefill", "decode"):
                for radix in (False, True):
                    kw = dict(radix=radix, disaggregation_mode=role)
                    self.assertEqual(
                        policy.fullstack_state_config(config(), **kw),
                        old_policy.fullstack_state_config(config(), **kw),
                    )
            for reference, prefix in ((True, "factored"), (False, "exact")):
                c = config(reference=reference, prefix=prefix)
                self.assertEqual(
                    policy.fullstack_state_config(c),
                    old_policy.fullstack_state_config(c),
                )

    def test_actual_tp1_fp16_allocation_and_no_restore(self):
        with patch.dict(os.environ, {KEY: "1"}):
            raw = policy.fullstack_state_config(config(), radix=False)
        p = pool_from(raw, actual=True)
        self.assertEqual(p.U.shape, (36, 3, 48, 32, 128))
        self.assertEqual(p.W.dtype, torch.float16)
        state = sum(x.numel() * x.element_size() for x in (p.a, p.U, p.W, p.count)) / 3
        conv = 36 * 10240 * 3 * 2
        self.assertAlmostEqual((state + conv) / 2**20, 29.959716796875)
        self.assertIsNone(p.prefix_valid)
        p.prewarm_restore_graph()
        self.assertIsNone(p.prefill_restore_graph)

    def test_real_install_prewarm_and_default_runner_warmup(self):
        for arm, radix, overlap in itertools.product(
            ("S", "C+", "PC"), (False, True), (False, True)
        ):
            with (
                self.subTest(arm=arm, radix=radix, overlap=overlap),
                patch.dict(
                    os.environ,
                    {
                        KEY: "1",
                        "SGLANG_GDN_AGG_FULLN_PREFILL": str(int(arm == "C+")),
                        "SGLANG_GDN_AGG_FULLN_OVERLAP_OK": "1",
                        "SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS": "0",
                        "SGLANG_GDN_PREFILL_COMMIT_GRAPH": "1",
                        "SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31": "1",
                        "SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31_MAX_BATCH": "1",
                        "SGLANG_GDN_PREFILL_RESTORE_GRAPH": "0",
                        "SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH": "0",
                        "TWINSTAR_PD_FACTOR_ONLY_TAIL": "0",
                    },
                ),
            ):
                raw = policy.fullstack_state_config(config(arm), radix=radix)
                p = pool_from(raw)
                o = owner_for(arm)
                runner = NS(
                    model=o,
                    req_to_token_pool=NS(factored_gdn_pool=p),
                    device="cuda",
                    is_draft_worker=False,
                    canary_manager=None,
                    forward_stream=None,
                    server_args=NS(
                        disaggregation_mode="null",
                        disable_radix_cache=not radix,
                        dp_size=1,
                        pp_size=1,
                        is_embedding=False,
                        speculative_algorithm=None,
                    ),
                )
                with capture_leaves(runner, overlap=overlap), Driver().use():
                    mr.ModelRunner.init_cuda_graphs(runner)
                    http_warmup(p, arm, radix)
                self.assertIs(o._model_runner, runner)
                self.assertTrue(runner._kernel_warmed_up)
                if arm == "C+":
                    self.assertTrue(o._pfactor_agg_installed)
                    self.assertEqual(hasattr(p, "_agg_fulln_overlap"), overlap)
                    self.assertTrue(p._agg_prefill_graph.warmed)
                    self.assertGreater(p._agg_prefill_graph.stats["captured"], 0)
                if p is not None:
                    self.assertIsNotNone(p._k31_batch_graph)
                    self.assertTrue(p._k31_batch_graph.warmed)
                    if not radix:
                        self.assertTrue(
                            all(k[1] is None for k in p._k31_batch_graph.entries)
                        )
                EVENTS.append(
                    dict(arm=arm, radix=radix, overlap=overlap, CPU_READY=True)
                )

    def test_batch_publish_bytes_equal_radix_and_frozen_source(self):
        with patch.dict(os.environ, {KEY: "1"}):
            on = policy.fullstack_state_config(config(), radix=True)
            off = policy.fullstack_state_config(config(), radix=False)
        p, q, old = pool_from(on), pool_from(off), pool_from(on, module=old_pool)
        gen = torch.Generator().manual_seed(247)
        states = [(torch.randn(1, 1, 32, 32, generator=gen), None) for _ in IDS]
        for pool, module in ((p, bg), (q, bg), (old, old_batch)):
            b = module.BatchBuffers(pool, 1, None, include_tail=False)
            b.bind(plan(pool), states, None, None, None)
            with Driver().use():
                b.evaluate(
                    fp.factorize_layers if module is bg else old_pool.factorize_layers
                )
        for key in ("a", "U", "W", "count", "stale", "dense_of", "dense_required"):
            self.assertEqual(sha(getattr(p, key)), sha(getattr(q, key)), key)
            self.assertEqual(sha(getattr(p, key)), sha(getattr(old, key)), key)
        self.assertTrue(torch.all(q.count[:, 1] == 16))
        self.assertIsNone(q.prefix_valid)
        with Driver().use():
            with self.assertRaisesRegex(ValueError, "tracked branch"):
                bg.BatchBuffers(q, 1, 1)
            b = bg.BatchBuffers(q, 1, None)
            with self.assertRaisesRegex(ValueError, "tracking metadata"):
                b.bind(plan(q), states, torch.tensor([2]), None, None)

    def test_no_tree_graph_signatures_and_default_preserved(self):
        self.assertEqual(
            list(cg.prewarm_shapes(include_tracked=False)),
            [(b, None) for b in (1, 2, 4, 8, 16)],
        )
        self.assertEqual(len(list(cg.prewarm_shapes())), 18)
        for radix in (False, True):
            with patch.dict(os.environ, {KEY: "0"}):
                raw = policy.fullstack_state_config(config(), radix=radix)
            new, old = pool_from(raw), pool_from(raw, module=old_pool)
            for name in ("a", "U", "W", "count", "dense_ring", "stale", "dense_of"):
                self.assertEqual(sha(getattr(new, name)), sha(getattr(old, name)), name)
            self.assertEqual(new.cfg.dtype, old.cfg.dtype)

    def test_request_local_chunk_ring_reset_and_reuse(self):
        with patch.dict(os.environ, {KEY: "1"}):
            raw = policy.fullstack_state_config(config(), radix=False)
        p = pool_from(raw)
        p.reset_slots(torch.tensor([1]))
        first = p.plan_extend(
            torch.tensor([1]), [64], prefix_lens=[0], prompt_final=[False]
        )
        self.assertTrue(first.all_fresh)
        self.assertIsNone(first.use_prefix)
        p.stale[1] = 0
        p.dense_required[1] = 1
        p.dense_ring[:, first.ring_dst[0]].fill_(0.25)
        second = p.plan_extend(
            torch.tensor([1]), [64], prefix_lens=[64], prompt_final=[True]
        )
        self.assertEqual(second.n_ring_src, 1)
        self.assertTrue(
            torch.equal(p.initial_dense(IDS[0], second), p.dense_ring[0, :1])
        )
        p.reset_slots(torch.tensor([1]))
        third = p.plan_extend(
            torch.tensor([1]), [64], prefix_lens=[0], prompt_final=[True]
        )
        self.assertTrue(third.all_fresh)
        self.assertTrue(torch.count_nonzero(p.initial_dense(IDS[0], third)) == 0)
        self.assertIsNone(p.prefix_valid)

    def test_unexpected_track_is_rejected_before_any_live_write(self):
        with patch.dict(os.environ, {KEY: "1"}):
            p = pool_from(policy.fullstack_state_config(config(), radix=False))
        before = sha(p.U)
        d = torch.zeros(1, 1, 32, 32)
        with self.assertRaisesRegex(ValueError, "tracking metadata"):
            p.commit_extend_batched(IDS[0], plan(p), d, d, torch.tensor([2]))
        self.assertEqual(sha(p.U), before)

    def test_no_tree_factory_and_no_track_allocation(self):
        from sglang.srt.mem_cache import registry
        from sglang.srt.mem_cache.chunk_cache import ChunkCache
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.arg_groups.model_override_base import mamba_extra_buffer_of

        ctx = NS(
            server_args=NS(),
            params=NS(
                req_to_token_pool=None, token_to_kv_pool_allocator=None, page_size=64
            ),
            disable_radix_cache=True,
            effective_chunked_prefill_size=8192,
            is_hybrid_swa=False,
        )
        with patch.object(
            registry,
            "get_disagg",
            return_value=NS(disaggregation_decode_retraction_backup=None),
        ):
            cache = registry.default_radix_cache_factory(ctx)
        self.assertIsInstance(cache, ChunkCache)
        self.assertTrue(cache.disable)
        self.assertEqual(
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey(list(range(128))))
            ).device_indices.numel(),
            0,
        )
        for strategy in ("extra_buffer", "extra_buffer_lazy"):
            self.assertFalse(
                mamba_extra_buffer_of(
                    NS(disable_radix_cache=True, mamba_radix_cache_strategy=strategy)
                )
            )

    def test_count_and_nminus1_contract_unchanged(self):
        # Public policy still chooses full-N only via the contract marker;
        # the shallow path retains N-1 independently of tree selection.
        for arm in ("C+", "PC"):
            for radix in (False, True):
                for enabled in ("0", "1"):
                    with patch.dict(os.environ, {KEY: enabled}):
                        req = NS(
                            extend_range=NS(length=8192, end=8192),
                            origin_input_ids=[0] * 8192,
                            _pfactor_agg_contract=(arm == "C+"),
                        )
                        self.assertEqual(
                            policy.prompt_p_extent(req), 8192 if arm == "C+" else 8191
                        )
                        self.assertEqual(
                            policy.prompt_p_extent(req), old_policy.prompt_p_extent(req)
                        )
        with self.assertRaisesRegex(ValueError, "prefix-cache metadata"):
            fp.FactoredGDNConfig.parse("r=16,m=16,no_radix=1,factored_prefix=1")


if __name__ == "__main__":
    reverse = os.environ.get("PFACTOR247_REVERSE")
    if reverse:
        name = (
            "test_policy_three_arms_two_radix_two_keys_and_role_scope"
            if reverse == "policy"
            else "test_batch_publish_bytes_equal_radix_and_frozen_source"
        )
        target, attr, old = (
            (policy, "fullstack_state_config", old_policy.fullstack_state_config)
            if reverse == "policy"
            else (bg.BatchBuffers, "evaluate", old_batch.BatchBuffers.evaluate)
        )
        with patch.object(target, attr, old):
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.TestSuite([NoRadixTest(name)])
            )
        negative_pass = (
            not result.wasSuccessful() and bool(result.failures) and not result.errors
        )
    else:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(NoRadixTest)
        )
        negative_pass = False
    out = os.environ.get("PFACTOR247_RECEIPT")
    if out:
        Path(out).write_text(
            json.dumps(
                dict(
                    passed=result.wasSuccessful(),
                    reverse=reverse,
                    negative_pass=negative_pass,
                    tests=result.testsRun,
                    failures=len(result.failures),
                    errors=len(result.errors),
                    skipped=len(result.skipped),
                    events=EVENTS,
                    cuda_quality="NOT_RUN",
                    gpu=0,
                ),
                indent=2,
            )
            + "\n"
        )
    sys.exit(not (negative_pass if reverse else result.wasSuccessful()))
