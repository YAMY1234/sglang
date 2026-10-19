"""CPU contracts of the EAGLE binding and of SGLang's side of the EAGLE entries: a
covered step's view is the running forward's own metadata and pools, sealed at their
first read and checked after capture; the structs carry data only and the tolerated
lists name real members; offered entries run only at the batch sizes every rank
serves, an unservable setup is refused, and without an offer SGLang's path runs; with
--enable-nvidia-megakernel the forced acceptance lengths are one rank-identical
stream, and without it they come from the global CPU RNG as before."""

import typing
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.megakernels import eagle_binding as eb
from sglang.srt.megakernels import runtime
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative import eagle_megakernel, eagle_worker_v2, spec_utils
from sglang.srt.speculative.eagle_megakernel import EagleMegakernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def _backend(cls, **fields):
    """A backend of cls holding only the given fields."""
    backend = cls.__new__(cls)
    backend.__dict__.update(fields)
    return backend


class KvPool:
    start_layer, layer_num = 3, 2

    def __init__(self):
        self.k = {i: torch.zeros(4, 2) for i in (3, 4)}

    def get_key_buffer(self, layer_id):
        return self.k[layer_id]

    def get_value_buffer(self, layer_id):
        return self.k[layer_id]


def _req_pool():
    tensor = torch.zeros(2)
    state = SimpleNamespace(
        conv=[tensor],
        temporal=tensor,
        intermediate_conv_window=[tensor],
        intermediate_ssm=tensor,
        **{f"replayssm_{n}": None for n in ("d", "k", "g", "rawv", "rawk", "beta")},
    )
    return SimpleNamespace(
        mamba_map={0: 0, 2: 1},
        get_speculative_mamba2_params_all_layers=lambda: state,
        mamba_pool=SimpleNamespace(
            replayssm_spec_write_pos=None,
            replayssm_cache_base=None,
            replayssm_is_flush=None,
        ),
    )


def _trtllm(pool):
    metadata = SimpleNamespace(
        page_table=torch.zeros(2, 3, dtype=torch.int32),
        cache_seqlens_int32=torch.zeros(2, dtype=torch.int32),
        cu_seqlens_q=torch.zeros(3, dtype=torch.int32),
        cu_seqlens_k=torch.zeros(3, dtype=torch.int32),
        max_seq_len_q=7,
    )
    return _backend(
        TRTLLMHAAttnBackend,
        forward_metadata=metadata,
        token_to_kv_pool=pool,
        data_type=torch.bfloat16,
        use_sliding_window_kv_pool=False,
    )


def _hybrid():
    pool = KvPool()
    gdn = _backend(
        GDNAttnBackend,
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=torch.zeros(2, dtype=torch.int32),
            query_start_loc=torch.zeros(3, dtype=torch.int32),
            mamba_track_indices=None,
        ),
        verify_intermediate_state_indices=torch.arange(2, dtype=torch.int32),
        pad_slot_id=-1,
    )
    return _backend(
        HybridLinearAttnBackend,
        full_attn_backend=_trtllm(pool),
        linear_attn_backend=gdn,
        token_to_kv_pool=pool,
        req_to_token_pool=_req_pool(),
    )


def _view(binding, backend):
    with patch.object(eb, "get_attn_backend", return_value=backend):
        return binding.forward_view()


def _model(**entries):
    """A model whose built megakernel offers these entries."""
    model = torch.nn.Module()
    model.__dict__["_megakernel"] = SimpleNamespace(widths=(), **entries)
    return model


def _megakernel(target_entries=None, **entries):
    """SGLang's side of a target and a draft offering these entries."""
    return EagleMegakernel(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=_model(**(target_entries or {})),
                watermark_state=None,
                req_to_token_pool=_req_pool(),
            ),
            random_seed=0,
            device="cpu",
        ),
        draft_worker=SimpleNamespace(
            draft_runner=SimpleNamespace(
                model=_model(**entries),
                model_config=SimpleNamespace(hidden_size=4, dtype=torch.float32),
            )
        ),
        simulated_acc_len_rng=eagle_megakernel.forced_length_generator(0),
    )


def _worker(
    *,
    target=None,
    draft=None,
    topk=1,
    rejection_sampling=False,
    target_backend=None,
    draft_backend=None,
    extends=None,
    flag=False,
):
    """An EAGLE worker after init_attention_backends, with --enable-nvidia-megakernel
    if flag, whose models' megakernels offer these entries; by default its target
    runs GDN and trtllm_mha under a hybrid backend and its draft trtllm_mha. extends
    collects what reached the target backend's enable_prefill_graph_extend."""
    if target_backend is None:
        target_backend = _hybrid()
    target_backend.enable_prefill_graph_extend = (
        [] if extends is None else extends
    ).append
    if draft_backend is None:
        draft_backend = _trtllm(KvPool())
    worker = eagle_worker_v2.EAGLEWorkerV2.__new__(eagle_worker_v2.EAGLEWorkerV2)
    worker.__dict__.update(
        topk=topk,
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=_model(**(target or {})),
                attn_backend=target_backend,
            ),
            random_seed=11,
            device="cpu",
        ),
        _draft_worker=SimpleNamespace(
            init_attention_backends=lambda: None,
            draft_runner=SimpleNamespace(
                model=_model(**(draft or {})),
                model_config=None,
                attn_backend=draft_backend,
            ),
            draft_attn_backend=SimpleNamespace(attn_backends=(draft_backend,)),
            draft_extend_attn_backend=None,
            megakernel=None,
        ),
        # the rest of what verify hands run_eagle_verify
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        plan_stream=None,
        plan_stream_ctx=None,
        speculative_num_draft_tokens=7,
        device="cpu",
    )
    spec = SimpleNamespace(speculative_use_rejection_sampling=rejection_sampling)
    with (
        patch.object(eagle_megakernel, "get_spec", return_value=spec),
        patch.object(eagle_worker_v2, "megakernel_enabled", return_value=flag),
    ):
        worker.init_attention_backends()
    return worker


def _verify_rng(worker):
    """The generator the worker's verify hands SGLang's eager sampling."""
    with patch.object(eagle_worker_v2, "run_eagle_verify") as run_eagle_verify:
        worker.verify(batch=None)
    return run_eagle_verify.call_args.kwargs["simulated_acc_len_rng"]


def _eager_length(generator):
    """The forced acceptance length SGLang's eager verify draws for one round."""
    correct = torch.zeros(1, dtype=torch.int32)
    spec_utils.generate_simulated_accept_index(
        accept_index=torch.zeros((1, 7), dtype=torch.int32),
        predict=torch.zeros(7, dtype=torch.int32),
        num_correct_drafts=correct,
        candidates=None,
        target_predict=None,
        bs=1,
        spec_steps=6,
        simulate_acc_len=4.8,
        simulate_acc_method="match-expected",
        simulate_acc_token_mode="fixed",
        generator=generator,
    )
    return int(correct) + 1


def _ranks(other=None):
    """megakernel_agree's gather as rank 0 of two ranks; other maps this rank's value
    to rank 1's, by default the same."""
    return patch.object(
        runtime,
        "_gather_tp",
        side_effect=lambda value, group=None: [
            value,
            value if other is None else other(value),
        ],
    )


def _batch(*rids):
    reqs = [SimpleNamespace(rid=rid) for rid in rids]
    return SimpleNamespace(reqs=reqs, seq_lens=torch.zeros(len(rids)))


def _leaves(hint):
    args = typing.get_args(hint)
    if not args:
        return [hint]
    return [leaf for arg in args if arg is not Ellipsis for leaf in _leaves(arg)]


class TestEagleBinding(CustomTestCase):
    def test_a_qwen3_5_view_is_the_running_forward_and_tolerated_members_exist(self):
        """A covered step is captured once and replays SGLang's refills of the
        backends' metadata, so the view must hold those very tensors."""
        for arch in ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForCausalLMMTP"):
            self.assertIs(runtime._binding_class([arch]), eb.EagleBinding)
        backend = _hybrid()
        view = _view(eb.EagleBinding(model=torch.nn.Module()), backend)
        attn = backend.full_attn_backend.forward_metadata
        gdn = backend.linear_attn_backend
        self.assertIs(view.attn.cache_seqlens, attn.cache_seqlens_int32)
        self.assertIs(view.attn.cu_seqlens_k, attn.cu_seqlens_k)
        self.assertIs(view.gdn.state_indices, gdn.forward_metadata.mamba_cache_indices)
        self.assertIs(
            view.gdn.intermediate_indices, gdn.verify_intermediate_state_indices
        )
        self.assertEqual(view.kv.layer_ids, (3, 4))
        self.assertIs(view.kv.k[1], backend.token_to_kv_pool.k[4])
        self.assertEqual(view.kv.k_scale, (1.0, 1.0))
        self.assertEqual(view.mamba.layer_ids, (0, 2))
        # a model without GDN layers, such as the MTP draft
        draft = _view(eb.EagleBinding(model=torch.nn.Module()), _trtllm(KvPool()))
        self.assertIsNone(draft.gdn)
        self.assertIsNone(draft.mamba)
        # a plugin pins these names: a rename here must fail first
        members = {
            "EagleBinding": set(dir(eb.EagleBinding)),
            "ForwardBatch": set(ForwardBatch.__dataclass_fields__),
        }
        for name in eb.TOLERATED_CALLABLES + eb.TOLERATED_READS:
            owner, _, attr = name.partition(".")
            if owner in members:
                self.assertIn(attr, members[owner], name)

    def test_a_reallocated_pool_fails_assert_live_by_name(self):
        backend = _hybrid()
        binding = eb.EagleBinding(model=torch.nn.Module())
        first = _view(binding, backend)
        self.assertIs(_view(binding, backend).kv, first.kv)  # sealed at the first read
        with patch.object(eb, "get_attn_backend", return_value=backend):
            binding.assert_live()
            backend.token_to_kv_pool.k[3] = torch.zeros(4, 2)
            with self.assertRaisesRegex(RuntimeError, "kv was reallocated"):
                binding.assert_live()

    def test_entry_structs_carry_data_only(self):
        structs = {
            cls
            for cls in vars(eb).values()
            if isinstance(cls, type) and issubclass(cls, msgspec.Struct)
        }
        allowed = {torch.Tensor, int, float, bool, type(None)} | structs
        for cls in structs:
            for field, hint in typing.get_type_hints(cls).items():
                for leaf in _leaves(hint):
                    self.assertIn(leaf, allowed, f"{cls.__name__}.{field}")


class TestEagleEntries(CustomTestCase):
    def test_without_an_offered_entry_sglang_keeps_its_own_path(self):
        """Without an offer SGLang's side of the entries does not exist and prefill
        keeps breaking at every GDN layer; an offer captures SGLang's own GDN extend,
        unless the target offers gdn_extend."""
        extends = []
        self.assertIsNone(_worker(extends=extends)._megakernel)
        self.assertEqual(extends, [])
        for offer in (
            dict(draft={"draft_loop": print}),
            dict(target={"verify_sample": print}),
        ):
            extends = []
            worker = _worker(extends=extends, **offer)
            self.assertIsNotNone(worker._megakernel)
            self.assertIs(worker._draft_worker.megakernel, worker._megakernel)
            self.assertEqual(extends, [None])
        extends = []
        worker = _worker(target={"gdn_extend": print}, extends=extends)
        self.assertEqual(extends, [worker._megakernel.gdn_extend_entry()])
        runner = SimpleNamespace(capture_bs=[1, 2])
        self.assertEqual(_megakernel().bind_draft_graphs(runner), frozenset())

    def test_every_unservable_setup_is_refused_before_any_capture(self):
        """The entries hand over a topk-1 chain's trtllm_mha and GDN metadata; any
        other setup is refused at startup, with all its reasons."""
        offer = {"draft_loop": print}
        sliding = _hybrid()
        sliding.full_attn_backend.use_sliding_window_kv_pool = True
        with self.assertRaises(ValueError) as refused:
            _worker(
                draft=offer,
                topk=2,
                rejection_sampling=True,
                target_backend=sliding,
                draft_backend=SimpleNamespace(),
            )
        for reason in (
            "chain draft (topk 1)",
            "rejection sampling",
            "sliding-window pool",
            "the draft must run trtllm_mha, not SimpleNamespace",
        ):
            self.assertIn(reason, str(refused.exception))
        with self.assertRaisesRegex(ValueError, "hybrid .* not TRTLLMHAAttnBackend"):
            _worker(draft=offer, target_backend=_trtllm(KvPool()))

    def test_each_factory_runs_once_and_serves_only_what_every_rank_serves(self):
        """A rank on SGLang's code at a batch size would wait in collectives the
        entry's step skips, so a size not every rank serves must be refused."""
        calls = []

        def factory(**buffers):
            calls.append(buffers)
            return lambda bs: print if bs <= 2 else None

        mk = _megakernel(
            target_entries={"verify_sample": factory},
            draft_loop=factory,
            draft_extend=factory,
        )
        draft_runner = SimpleNamespace(
            max_bs=4, speculative_num_steps=3, device="cpu", capture_bs=[1, 2, 4]
        )
        extend_runner = SimpleNamespace(
            max_bs=4, captured_req_width=7, device="cpu", capture_bs=[1, 2, 4]
        )
        with _ranks():
            self.assertEqual(mk.bind_draft_graphs(draft_runner), {1, 2})
            self.assertEqual(mk.bind_extend_graphs(extend_runner), {1, 2})
            mk.bind_verify_sample(max_bs=4, tokens_per_request=7)
        self.assertEqual(sorted(mk._verify_steps), [1, 2])
        loop, extend, verify = calls
        self.assertEqual(loop["draft_tokens"].shape, (4, 3))
        self.assertEqual((loop["num_steps"], loop["max_bs"]), (3, 4))
        self.assertEqual((extend["tokens_per_request"], extend["max_bs"]), (7, 4))
        self.assertEqual(extend["hidden"].shape, (4, 4))
        # every round's verify_sample draws on this key and advances this counter
        self.assertEqual((verify["max_bs"], verify["noise_key"]), (4, mk._noise_key))
        self.assertIs(verify["noise_offset"], mk._noise_offset)
        with (
            _ranks(lambda sizes: sizes[:1]),
            self.assertRaisesRegex(RuntimeError, "draft_loop batch sizes differ"),
        ):
            mk.bind_draft_graphs(draft_runner)

    def test_a_draft_round_reuses_a_served_extend_only_for_the_input_it_made(self):
        """The megakernel may reuse what its draft extend left behind only when that
        extend produced this round's input: not after another extend, not for other
        requests (a PD decode instance's first round), and only once. The extend's
        tokens are SGLang's own copy, which the next replay leaves alone."""
        rounds, outs = [], {}

        def draft_extend(**buffers):
            outs.update(buffers)
            return lambda bs: print if bs == 1 else None

        mk = _megakernel(
            draft_loop=lambda **_: lambda bs: print if bs == 1 else None,
            draft_extend=draft_extend,
            begin_draft_round=rounds.append,
        )
        draft_runner = SimpleNamespace(
            max_bs=2, speculative_num_steps=3, device="cpu", capture_bs=[1, 2]
        )
        extend_runner = SimpleNamespace(
            max_bs=2, captured_req_width=4, device="cpu", capture_bs=[1, 2]
        )
        with _ranks():
            mk.bind_draft_graphs(draft_runner)
            mk.bind_extend_graphs(extend_runner)
        draft_input = SimpleNamespace(
            topk_index=torch.zeros((1, 1), dtype=torch.int64),
            hidden_states=torch.zeros((1, 4)),
        )

        def served_extend(*rids):
            mk.note_draft_extend()
            extend_runner.bs = extend_runner.raw_bs = len(rids)
            return mk.extend_tokens(runner=extend_runner, batch=_batch(*rids))

        def begin(*rids):
            mk.begin_draft_round(
                runner=draft_runner, batch=_batch(*rids), draft_input=draft_input
            )
            return rounds[-1].from_served_extend

        outs["topk_index"][0] = 5  # what the extend's replay wrote
        _, topk_index = served_extend("a")
        outs["topk_index"].zero_()  # the next replay rewrites it
        self.assertEqual(topk_index.tolist(), [[5]])
        self.assertTrue(begin("a"))
        self.assertFalse(begin("a"))
        served_extend("a")
        mk.note_draft_extend()  # an extend the megakernel did not serve
        self.assertFalse(begin("a"))
        served_extend("a")
        self.assertFalse(begin("b"))
        calls = len(rounds)
        begin("a", "b")
        self.assertEqual(len(rounds), calls)  # batch size 2 is not served
        self.assertIsNone(served_extend("a", "b"))  # SGLang's top-1 there

    def test_a_captured_bucket_hands_the_megakernels_gdn_extend_its_own_tables(self):
        """A captured prefill bucket replays SGLang's refills of its static tables, so
        the target's gdn_extend must get those very tensors, the layer's inputs, the
        mamba pool and SGLang's zeroed output, whose rows past the batch stay zero."""
        calls = []

        def gdn_extend(io):
            calls.append(io)
            io.out[:, :3] = 1.0

        mk = _megakernel(target_entries={"gdn_extend": gdn_extend})
        scalars = ("num_tokens", "max_seqs", "max_chunks", "max_conv_blocks", "track")
        meta = SimpleNamespace(
            num_tokens=6,
            max_seqs=2,
            max_chunks=3,
            max_conv_blocks=4,
            track=False,
            **{
                name: torch.zeros(1)
                for name in eb.GdnExtendTables.__struct_fields__
                if name not in scalars
            },
        )
        backend = _backend(
            GDNAttnBackend,
            prefill_graph_metadata=meta,
            megakernel_extend=mk.gdn_extend_entry(),
        )
        inputs = dict(
            mixed_qkv=torch.zeros((6, 16)), a=torch.zeros((6, 2)), b=torch.zeros((6, 2))
        )
        layer = SimpleNamespace(layer_id=5, num_v_heads=2, head_v_dim=4)
        out = backend._forward_extend_in_graph(layer=layer, **inputs)
        (io,) = calls
        self.assertEqual((io.layer_id, io.mamba.layer_ids), (5, (0, 2)))
        for name in eb.GdnExtendTables.__struct_fields__:
            self.assertIs(getattr(io.tables, name), getattr(meta, name), name)
        for name, tensor in inputs.items():
            self.assertIs(getattr(io, name), tensor, name)
        self.assertIs(io.out, out)
        self.assertEqual(tuple(out.shape), (1, 6, 2, 4))
        self.assertTrue(bool((out[:, :3] == 1).all() and (out[:, 3:] == 0).all()))

    def test_only_batches_a_graph_can_replay_take_the_post_verify_graphs(self):
        """A batch whose sampling a captured graph cannot reproduce (grammar, penalties,
        seeds, ...) or whose verify ran eagerly must take SGLang's eager path."""
        mk = _megakernel()
        logits_output = SimpleNamespace(next_token_logits=torch.zeros((7, 8)))

        def key(can_run_cuda_graph=True, topk=1, **changes):
            info = dict(
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                logit_bias=None,
                sampling_seed=None,
                need_min_p_sampling=False,
                has_watermark_candidates=False,
                is_all_greedy=False,
                need_top_k_sampling=True,
                need_top_p_sampling=True,
            )
            fields = dict(has_grammar=False, hisparse_coordinator=None)
            for name, value in changes.items():
                (info if name in info else fields)[name] = value
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                seq_lens=torch.zeros(1),
                reqs=[SimpleNamespace(sampling_params=SimpleNamespace(top_k=1))],
                sampling_info=SimpleNamespace(**info),
                mamba_track_indices=None,
                **fields,
            )
            verify_input = SimpleNamespace(
                topk=topk, draft_probs=None, draft_token_num=7, max_tree_depth=7
            )
            return mk._post_verify_key(
                batch, verify_input, logits_output, can_run_cuda_graph
            )

        self.assertIsNotNone(key())
        for change in (
            dict(can_run_cuda_graph=False),
            dict(topk=2),
            dict(has_grammar=True),
            dict(hisparse_coordinator=object()),
            dict(acc_additive_penalties=torch.zeros(1)),
            dict(acc_scaling_penalties=torch.zeros(1)),
            dict(logit_bias=torch.zeros(1)),
            dict(sampling_seed=torch.zeros(1)),
            dict(need_min_p_sampling=True),
        ):
            with self.subTest(change=sorted(change)):
                self.assertIsNone(key(**change))

    def test_the_compact_branch_takes_only_batches_of_finite_top_k(self):
        """A batch samples on the top-k support only when every request has a finite
        top-k within the compact limit; its graphs are keyed apart from the dense
        branch's, whose sampling differs."""

        def batch(*top_ks):
            reqs = [
                SimpleNamespace(sampling_params=SimpleNamespace(top_k=k))
                for k in top_ks
            ]
            return SimpleNamespace(reqs=reqs)

        limit = SimpleNamespace(_SPARSE_TOP_K_LIMIT=128)
        with patch.dict("sys.modules", {"sglang.srt.speculative.uno_utils": limit}):
            self.assertEqual(eagle_megakernel._compact_max_top_k(batch(20, 64)), 64)
            for dense in ((20, 1 << 30), (20, 1), (129,), ()):
                with self.subTest(top_ks=dense):
                    self.assertIsNone(
                        eagle_megakernel._compact_max_top_k(batch(*dense))
                    )
            mk = _megakernel()
            info = SimpleNamespace(
                is_all_greedy=False,
                need_top_k_sampling=True,
                need_top_p_sampling=False,
                need_min_p_sampling=False,
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                logit_bias=None,
                sampling_seed=None,
                has_watermark_candidates=False,
            )
            verify_input = SimpleNamespace(
                topk=1, draft_probs=None, draft_token_num=7, max_tree_depth=7
            )
            logits = SimpleNamespace(next_token_logits=torch.zeros((7, 8)))
            keys = set()
            for top_k in (20, 1 << 30):
                verify_batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    seq_lens=torch.zeros(1),
                    reqs=batch(top_k).reqs,
                    sampling_info=info,
                    mamba_track_indices=None,
                    has_grammar=False,
                    hisparse_coordinator=None,
                )
                keys.add(mk._post_verify_key(verify_batch, verify_input, logits, True))
            self.assertEqual(len(keys), 2)

    def test_a_packed_verify_result_is_one_buffer_holding_the_stock_values(self):
        """The post-verify graphs broadcast a sampled result once: predict,
        accept_index and num_correct_drafts must be views of one buffer, in order,
        starting from the values the unpacked tensors start from."""
        from sglang.srt.speculative.eagle_utils import _verify_outputs

        kw = dict(rows=14, bs=2, depth=7, device="cpu")
        stock = _verify_outputs(packed=False, **kw)
        *views, buffer = _verify_outputs(packed=True, **kw)
        self.assertIsNone(stock[3])
        for got, want in zip(views[:2], stock[:2]):
            self.assertTrue(torch.equal(got, want))
        offset = 0
        for view in views:
            self.assertEqual(view.data_ptr(), buffer.data_ptr() + 4 * offset)
            offset += view.numel()
        self.assertEqual(offset, buffer.numel())

    def test_forced_lengths_are_one_stream_with_the_flag_and_unchanged_without(self):
        """With --enable-nvidia-megakernel the post-verify graphs and the eager verify
        draw each round's forced length from the worker's one generator, seeded from
        the server seed: every rank forces the same lengths, whichever path each round
        took and whatever else used its global RNG. Without the flag the eager verify
        draws from the global CPU RNG exactly as before."""

        def lengths(paths, global_seed):
            worker = _worker(target={"verify_sample": print}, flag=True)
            rng = worker._simulated_acc_len_rng
            self.assertIs(_verify_rng(worker), rng)
            graphs = worker._megakernel._post_verify
            # the draw a graph round makes, without its pinned ring on a CPU host
            graphs._sim_host = [torch.zeros(1, dtype=torch.int64) for _ in range(4)]
            graphs._sim_done = [None] * 4
            torch.manual_seed(global_seed)  # this rank's global RNG
            drawn = []
            for path in paths:
                torch.rand(3)  # other uses of the global RNG
                if path == "graph":
                    drawn.append(int(graphs._draw_sim_len(7)[0][0]))
                else:
                    drawn.append(_eager_length(rng))
            return drawn

        with (
            patch.object(spec_utils, "SIMULATE_ACC_LEN", 4.8),
            patch.object(spec_utils, "SIMULATE_ACC_METHOD", "match-expected"),
        ):
            all_graph = lengths(["graph"] * 48, global_seed=1)
            all_eager = lengths(["eager"] * 48, global_seed=2)
            mixed = lengths(["graph", "eager", "eager"] * 16, global_seed=3)
        self.assertEqual(set(all_graph), {4, 5})
        self.assertEqual(all_eager, all_graph)
        self.assertEqual(mixed, all_graph)
        server_seeded = eagle_megakernel.forced_length_generator(11)
        self.assertEqual(all_graph, [_eager_length(server_seeded) for _ in range(48)])

        self.assertIsNone(_verify_rng(_worker()))
        torch.manual_seed(3)
        drawn = [_eager_length(None) for _ in range(32)]
        state = torch.get_rng_state()
        torch.manual_seed(3)
        weight = 4.8 - 4
        probs = torch.tensor([1.0 - weight, weight])
        before = [4 + int(torch.multinomial(probs, num_samples=1)) for _ in range(32)]
        self.assertEqual(drawn, before)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))


if __name__ == "__main__":
    unittest.main()
