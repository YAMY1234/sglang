"""CPU contracts for the generic speculative-runtime hook binding."""

import dataclasses
import typing
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.kernels.plugin_kernels.nvidia import spec
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.kernel_plugins.nvidia import speculative_runtime as runtime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


def _tree() -> spec.VerifyTreeView:
    return spec.VerifyTreeView(
        draft_tokens=torch.tensor([4, 0, 1, 4, 2, 3]),
        draft_probabilities=None,
        retrieve_index=torch.arange(6).view(2, 3),
        next_child=torch.tensor([[1, 2, -1], [1, 2, -1]]),
        next_sibling=torch.full((2, 3), -1),
        custom_mask=None,
        positions=None,
        tokens_per_request=3,
        max_depth=3,
        topk=1,
    )


def _sampling() -> spec.VerifySamplingView:
    return spec.VerifySamplingView(
        sequence_lengths=torch.tensor([7, 11]),
        temperatures=torch.ones((2, 1)),
        top_ks=torch.ones(2, dtype=torch.int32),
        top_ps=torch.ones(2),
        min_ps=torch.zeros(2),
        additive_penalties=None,
        scaling_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        is_all_greedy=True,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=None,
        has_watermark_candidates=False,
        max_top_k=1,
        device=torch.device("cpu"),
    )


def _post_io(logits: torch.Tensor | None = None) -> spec.PostVerifyIO:
    if logits is None:
        logits = torch.tensor(
            [
                [9.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 8.0, 1.0, -1.0, -2.0],
                [0.0, 1.0, 7.0, -1.0, -2.0],
                [0.0, 1.0, 6.0, -1.0, -2.0],
                [0.0, 1.0, 2.0, 5.0, -2.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
    return spec.PostVerifyIO(
        logits=SimpleNamespace(
            next_token_logits=logits,
            next_token_logits_vocab_shard=None,
        ),
        verify_tree=_tree(),
        sampling=_sampling(),
        grammar_mask=None,
        watermark=None,
        watermark_full_mask=False,
        request_pool_indices=torch.tensor([3, 7]),
        return_logprob=False,
        is_idle=False,
        simulated_accept_length=None,
        simulated_accept_length_rng=None,
        sparse_target_max_top_k=None,
        sparse_bonus_argmax=False,
    )


def _defaults() -> spec.SpeculativeRuntimeHooks:
    return runtime.default_speculative_runtime_hooks()


class TestSpeculativeRuntimeHooks(CustomTestCase):
    def test_frozen_contract_field_lists(self):
        expected = {
            spec.VocabShardView: ("start", "valid", "vocab_size"),
            spec.ShardedLogitsView: (
                "values",
                "shard",
                "temperatures",
                "seeds",
                "splits",
            ),
            spec.VerifyTreeView: (
                "draft_tokens",
                "draft_probabilities",
                "retrieve_index",
                "next_child",
                "next_sibling",
                "custom_mask",
                "positions",
                "tokens_per_request",
                "max_depth",
                "topk",
            ),
            spec.VerifySamplingView: (
                "sequence_lengths",
                "temperatures",
                "top_ks",
                "top_ps",
                "min_ps",
                "additive_penalties",
                "scaling_penalties",
                "logit_bias",
                "sampling_seed",
                "is_all_greedy",
                "need_top_k_sampling",
                "need_top_p_sampling",
                "need_min_p_sampling",
                "watermark_keys",
                "watermark_context_windows",
                "watermark_enabled",
                "has_watermark_candidates",
                "max_top_k",
                "device",
            ),
            spec.PostVerifyIO: (
                "logits",
                "verify_tree",
                "sampling",
                "grammar_mask",
                "watermark",
                "watermark_full_mask",
                "request_pool_indices",
                "return_logprob",
                "is_idle",
                "simulated_accept_length",
                "simulated_accept_length_rng",
                "sparse_target_max_top_k",
                "sparse_bonus_argmax",
            ),
            spec.PostVerifyOutput: (
                "token_ids",
                "accepted_lengths",
                "accepted_indices",
            ),
            spec.PostVerifyStrategy: ("run", "local_keys"),
            spec.SpeculativeRuntimeHooks: ("post_verify",),
        }
        for contract, fields in expected.items():
            with self.subTest(contract=contract.__name__):
                self.assertEqual(
                    tuple(field.name for field in dataclasses.fields(contract)), fields
                )
                self.assertTrue(contract.__dataclass_params__.frozen)

    def test_frozen_contract_field_types_resolve_exactly_without_any(self):
        tensor = torch.Tensor
        optional = typing.Optional
        expected = {
            spec.VocabShardView: {"start": int, "valid": int, "vocab_size": int},
            spec.ShardedLogitsView: {
                "values": tensor,
                "shard": spec.VocabShardView,
                "temperatures": optional[tensor],
                "seeds": optional[tensor],
                "splits": int,
            },
            spec.VerifyTreeView: {
                "draft_tokens": tensor,
                "draft_probabilities": optional[tensor],
                "retrieve_index": tensor,
                "next_child": tensor,
                "next_sibling": tensor,
                "custom_mask": optional[tensor],
                "positions": optional[tensor],
                "tokens_per_request": int,
                "max_depth": int,
                "topk": int,
            },
            spec.VerifySamplingView: {
                "sequence_lengths": tensor,
                "temperatures": tensor,
                "top_ks": tensor,
                "top_ps": tensor,
                "min_ps": tensor,
                "additive_penalties": optional[tensor],
                "scaling_penalties": optional[tensor],
                "logit_bias": optional[tensor],
                "sampling_seed": optional[tensor],
                "is_all_greedy": bool,
                "need_top_k_sampling": bool,
                "need_top_p_sampling": bool,
                "need_min_p_sampling": bool,
                "watermark_keys": optional[tensor],
                "watermark_context_windows": optional[tensor],
                "watermark_enabled": optional[tensor],
                "has_watermark_candidates": bool,
                "max_top_k": optional[int],
                "device": torch.device,
            },
            spec.PostVerifyIO: {
                "logits": spec.VerifyLogitsView,
                "verify_tree": spec.VerifyTreeView,
                "sampling": spec.VerifySamplingView,
                "grammar_mask": optional[spec.GrammarMaskView],
                "watermark": optional[spec.WatermarkStateView],
                "watermark_full_mask": bool,
                "request_pool_indices": tensor,
                "return_logprob": bool,
                "is_idle": bool,
                "simulated_accept_length": optional[tensor],
                "simulated_accept_length_rng": optional[torch.Generator],
                "sparse_target_max_top_k": optional[int],
                "sparse_bonus_argmax": bool,
            },
            spec.PostVerifyOutput: {
                "token_ids": tensor,
                "accepted_lengths": tensor,
                "accepted_indices": tensor,
            },
            spec.PostVerifyStrategy: {
                "run": spec.VerifyRunner,
                "local_keys": spec.LocalKeysFn,
            },
            spec.SpeculativeRuntimeHooks: {
                "post_verify": spec.PostVerifyStrategy,
            },
        }
        for contract, types in expected.items():
            with self.subTest(contract=contract.__name__):
                resolved = typing.get_type_hints(contract)
                self.assertEqual(resolved, types)
                self.assertNotIn("typing.Any", repr(resolved))

        self.assertEqual(
            typing.get_type_hints(spec.VerifyLogitsView),
            {
                "next_token_logits": tensor,
                "next_token_logits_vocab_shard": optional[spec.VocabShardView],
            },
        )
        self.assertEqual(
            spec.HooksFactory,
            typing.Callable[
                [spec.SpeculativeRuntimeHooks], spec.SpeculativeRuntimeHooks
            ],
        )
        self.assertEqual(
            spec.LocalKeysFn,
            typing.Callable[[spec.ShardedLogitsView], typing.Optional[torch.Tensor]],
        )

    def test_local_key_output_contract_and_logits_writeback_are_documented(self):
        from sglang.srt.speculative.sharded_verify import VocabShard

        self.assertIs(VocabShard, spec.VocabShardView)
        key_contract = " ".join(spec.ShardedLogitsView.__doc__.split())
        for phrase in (
            "floating ``[rows, local_cols]``",
            "unit last stride",
            "contiguous float32 ``[rows]``",
            "contiguous int64 ``[batch]``",
            "positive runtime-selected",
            "contiguous int64",
            "high 32 bits",
            "0xffffffff - global_token_id",
            "one split returns ``[rows]``",
            "``[rows, splits]``",
            "``splits`` must be used unchanged",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, key_contract)
        logits_contract = spec.VerifyLogitsView.__doc__
        self.assertIn("writes the full tensor back", logits_contract)
        self.assertIn("clears ``next_token_logits_vocab_shard``", logits_contract)

    def test_unsupported_local_keys_fall_back_before_any_rng_draw(self):
        from sglang.srt.speculative import eagle_utils

        events = []
        group = SimpleNamespace(world_size=1)

        def local_keys(view):
            events.append(("local", view))
            self.assertIsNone(view.seeds)
            self.assertEqual(tuple(view.values.shape), (2, 4))
            self.assertEqual(view.values.stride(-1), 1)
            self.assertEqual(tuple(view.temperatures.shape), (2,))
            self.assertEqual(view.temperatures.dtype, torch.float32)
            self.assertTrue(view.temperatures.is_contiguous())
            self.assertGreater(view.splits, 0)

        rng_before = torch.random.get_rng_state().clone()
        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch.object(
                torch,
                "randint",
                side_effect=AssertionError("runtime drew RNG before fallback"),
            ),
        ):
            got = eagle_utils._sharded_tokens(
                torch.ones((2, 4)),
                spec.VocabShardView(start=0, valid=4, vocab_size=4),
                local_keys=local_keys,
                temperatures=torch.ones(2),
            )
        self.assertIsNone(got)
        self.assertEqual([event[0] for event in events], ["local"])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_existing_sampling_seeds_are_passed_unchanged_then_declined(self):
        from sglang.srt.speculative import eagle_utils

        group = SimpleNamespace(world_size=1)
        seeds = torch.tensor([17, 29], dtype=torch.int64)
        seen = []

        def local_keys(view):
            seen.append(view.seeds)
            # A per-request sampling seed is not a per-row local-key seed.  The
            # provider must decline so stock seeded sampling remains authoritative.

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch.object(
                torch,
                "randint",
                side_effect=AssertionError("provider fallback must not draw RNG"),
            ),
        ):
            got = eagle_utils._sharded_tokens(
                torch.ones((6, 4)),
                spec.VocabShardView(start=0, valid=4, vocab_size=4),
                local_keys=local_keys,
                temperatures=torch.ones(6),
                seeds=seeds,
            )

        self.assertIsNone(got)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], seeds)
        self.assertEqual(tuple(seen[0].shape), (2,))
        self.assertNotEqual(seen[0].shape[0], 6)

    def test_provider_splits_use_rank_uniform_padded_width(self):
        from sglang.srt.speculative import eagle_utils

        logits = torch.empty((2, 4104))
        shard = spec.VocabShardView(start=7 * 4104, valid=4041, vocab_size=32769)
        group = SimpleNamespace(world_size=8)
        tokens = torch.empty(2, dtype=torch.int64)
        seen = []

        def local_keys(view):
            seen.append(view.splits)
            return torch.empty((2, view.splits), dtype=torch.int64)

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch.object(
                eagle_utils,
                "reduce_sharded_token_keys",
                return_value=tokens,
            ) as reduce,
        ):
            got = eagle_utils._sharded_tokens(
                logits,
                shard,
                local_keys=local_keys,
            )

        self.assertIs(got, tokens)
        self.assertEqual(seen, [4])
        reduce.assert_called_once()
        keys = reduce.call_args.args[0]
        self.assertEqual(tuple(keys.shape), (2, 4))
        self.assertEqual(reduce.call_args.kwargs, {"splits": 4})

    def test_default_local_keys_serves_greedy_and_declines_all_temperature(self):
        from sglang.srt.speculative import sharded_verify

        values = torch.ones((2, 4))
        shard = spec.VocabShardView(start=0, valid=4, vocab_size=4)
        sentinel = torch.ones(2, dtype=torch.int64)
        with patch.object(
            sharded_verify, "shard_keys", return_value=sentinel
        ) as shard_keys:
            greedy = spec.ShardedLogitsView(
                values=values,
                shard=shard,
                temperatures=None,
                seeds=None,
                splits=1,
            )
            self.assertIs(runtime.default_local_keys(greedy), sentinel)
            shard_keys.assert_called_once_with(
                values,
                shard,
                temperatures=None,
                seeds=None,
                splits=1,
            )
            shard_keys.reset_mock()
            temperature = dataclasses.replace(
                greedy, temperatures=torch.ones(2), seeds=None
            )
            self.assertIsNone(runtime.default_local_keys(temperature))
            seeded_temperature = dataclasses.replace(
                temperature, seeds=torch.arange(2, dtype=torch.int64)
            )
            self.assertIsNone(runtime.default_local_keys(seeded_temperature))
            shard_keys.assert_not_called()

    def test_default_greedy_with_sampling_seed_matches_stock_bitwise(self):
        from sglang.srt.speculative import eagle_utils, sharded_verify

        logits = torch.tensor([[1.0, 8.0, 3.0], [7.0, 2.0, 4.0]])
        seeds = torch.tensor([123], dtype=torch.int64)
        stock = torch.argmax(logits, dim=-1)
        encoded = torch.zeros((2, 2), dtype=torch.int64)
        group = SimpleNamespace(world_size=1)

        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch.object(
                sharded_verify, "shard_keys", return_value=encoded
            ) as shard_keys,
            patch.object(
                eagle_utils,
                "reduce_sharded_token_keys",
                return_value=stock.clone(),
            ) as reduce,
        ):
            got = eagle_utils._sharded_tokens(
                logits,
                spec.VocabShardView(start=0, valid=3, vocab_size=3),
                local_keys=runtime.default_local_keys,
                temperatures=None,
                seeds=seeds,
            )

        self.assertTrue(torch.equal(got, stock))
        shard_keys.assert_called_once_with(
            logits,
            spec.VocabShardView(start=0, valid=3, vocab_size=3),
            temperatures=None,
            seeds=seeds,
            splits=2,
        )
        reduce.assert_called_once_with(encoded, group, splits=2)

    def test_temperature_decline_gathers_then_enters_stock_sampling(self):
        from sglang.srt.speculative import eagle_utils

        sampling = dataclasses.replace(
            _sampling(),
            is_all_greedy=False,
            max_top_k=None,
        )
        logits = SimpleNamespace(
            next_token_logits=_post_io().logits.next_token_logits.clone(),
            next_token_logits_vocab_shard=spec.VocabShardView(
                start=0, valid=5, vocab_size=5
            ),
        )
        io = dataclasses.replace(_post_io(), logits=logits, sampling=sampling)
        group = SimpleNamespace(world_size=1)
        group.all_gather_into_tensor = lambda output, value: output.copy_(value)
        parallel = SimpleNamespace(tp_group=group, attn_tp_group=group)
        runtime_spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_use_block_verification=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        events = []
        state_before = torch.random.get_rng_state().clone()

        def local_keys(view):
            events.append("local")
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))
            result = runtime.default_local_keys(view)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))
            return result

        original_gather = eagle_utils.gather_full_logits

        def gather(*args, **kwargs):
            events.append("gather")
            return original_gather(*args, **kwargs)

        def stock_sample(**kwargs):
            events.append("stock")
            self.assertFalse(torch.equal(torch.random.get_rng_state(), state_before))
            kwargs["predicts"].zero_()
            kwargs["accept_index"].fill_(-1)
            kwargs["accept_token_num"].zero_()

        with (
            patch.object(eagle_utils, "_is_cuda", True),
            patch.object(eagle_utils, "get_spec", return_value=runtime_spec),
            patch.object(eagle_utils, "gather_full_logits", side_effect=gather),
            patch.object(
                eagle_utils,
                "tree_speculative_sampling_target_only",
                side_effect=stock_sample,
            ),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            eagle_utils.eagle_sample(
                io.verify_tree,
                io,
                io.logits,
                local_keys=local_keys,
            )

        self.assertEqual(events, ["local", "gather", "stock"])
        self.assertIsNone(logits.next_token_logits_vocab_shard)
        self.assertEqual(tuple(logits.next_token_logits.shape), (6, 5))

    def test_sampled_local_keys_recompute_greedy_real_draft_bonus(self):
        from sglang.srt.speculative import eagle_utils, spec_utils

        sampling = dataclasses.replace(
            _sampling(),
            is_all_greedy=False,
            max_top_k=None,
        )
        logits = SimpleNamespace(
            next_token_logits=_post_io().logits.next_token_logits.clone(),
            next_token_logits_vocab_shard=spec.VocabShardView(
                start=0, valid=5, vocab_size=5
            ),
        )
        io = dataclasses.replace(
            _post_io(),
            logits=logits,
            sampling=sampling,
            simulated_accept_length=torch.tensor([1]),
        )
        runtime_spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_use_block_verification=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        sampled = torch.full((6,), 4, dtype=torch.int64)
        greedy_bonus = torch.arange(6, dtype=torch.int64)
        simulated_targets = []

        def greedy_check(**kwargs):
            self.assertTrue(
                torch.equal(kwargs["target_predict"], sampled.reshape(2, 3))
            )
            return (
                kwargs["predicts"],
                kwargs["accept_index"],
                kwargs["accept_token_num"],
            )

        def simulate(**kwargs):
            simulated_targets.append(kwargs["target_predict"].clone())
            return kwargs["accept_index"]

        with (
            patch.object(eagle_utils, "_is_cuda", True),
            patch.object(eagle_utils, "get_spec", return_value=runtime_spec),
            patch.object(
                eagle_utils,
                "_sharded_tokens",
                side_effect=(sampled, greedy_bonus),
            ) as sharded_tokens,
            patch.object(
                eagle_utils, "verify_tree_greedy_func", side_effect=greedy_check
            ),
            patch.object(
                spec_utils,
                "generate_simulated_accept_index_device",
                side_effect=simulate,
            ),
            patch.object(spec_utils, "SIMULATE_ACC_LEN", 1.0),
            patch.object(spec_utils, "SIMULATE_ACC_TOKEN_MODE", "real-draft-token"),
        ):
            eagle_utils.eagle_sample(
                io.verify_tree,
                io,
                io.logits,
                simulated_acc_len=io.simulated_accept_length,
                local_keys=lambda view: None,
            )

        self.assertEqual(sharded_tokens.call_count, 2)
        self.assertIsNotNone(sharded_tokens.call_args_list[0].kwargs["temperatures"])
        self.assertIsNone(sharded_tokens.call_args_list[1].kwargs.get("temperatures"))
        self.assertEqual(len(simulated_targets), 1)
        self.assertTrue(torch.equal(simulated_targets[0], greedy_bonus.reshape(2, 3)))

    def test_local_key_reducer_rejects_noncanonical_tensors(self):
        from sglang.srt.speculative.sharded_verify import reduce_sharded_token_keys

        group = SimpleNamespace(world_size=1)
        with self.assertRaisesRegex(TypeError, "dtype torch.int64"):
            reduce_sharded_token_keys(torch.ones(2), group, splits=1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            reduce_sharded_token_keys(
                torch.ones((2, 4), dtype=torch.int64)[:, ::2], group, splits=2
            )
        with self.assertRaisesRegex(ValueError, "incompatible with splits=2"):
            reduce_sharded_token_keys(torch.ones(2, dtype=torch.int64), group, splits=2)
        with self.assertRaisesRegex(ValueError, "incompatible with splits=1"):
            reduce_sharded_token_keys(
                torch.ones((2, 1), dtype=torch.int64), group, splits=1
            )

    def test_local_key_output_rows_must_match_logits_before_collective(self):
        from sglang.srt.speculative import eagle_utils

        group = SimpleNamespace(world_size=1)
        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch.object(
                eagle_utils,
                "reduce_sharded_token_keys",
                side_effect=AssertionError("invalid shape reached collective"),
            ) as reduce,
            self.assertRaisesRegex(ValueError, r"shape \(3, 2\), expected \(2, 2\)"),
        ):
            eagle_utils._sharded_tokens(
                torch.ones((2, 4)),
                spec.VocabShardView(start=0, valid=4, vocab_size=4),
                local_keys=lambda view: torch.zeros((3, 2), dtype=torch.int64),
            )
        reduce.assert_not_called()

    def test_post_verify_strategy_passes_the_local_policy_to_runtime(self):
        io = _post_io()
        output = spec.PostVerifyOutput(
            token_ids=torch.tensor([1]),
            accepted_lengths=torch.tensor([1]),
            accepted_indices=torch.tensor([0]),
        )
        local_keys = lambda view: view.values.argmax(-1)
        calls = []

        def run(received_io, received_local_keys):
            calls.append((received_io, received_local_keys))
            return output

        strategy = spec.PostVerifyStrategy(run=run, local_keys=local_keys)
        self.assertIs(strategy(io), output)
        self.assertEqual(calls, [(io, local_keys)])

    def test_ngram_uses_the_same_unconditional_post_verify_strategy(self):
        from sglang.srt.speculative import eagle_worker_common

        source = (
            Path(eagle_worker_common.__file__).with_name("ngram_worker.py").read_text()
        )
        self.assertNotIn("eagle_sample(", source)
        self.assertIn("self._speculative_runtime_hooks.post_verify(", source)

        # NGRAM has no proposal probabilities. The shared constructor must form
        # the same model-independent view without requiring an EAGLE-only field.
        tree = _tree()
        verify_input = SimpleNamespace(
            draft_token=tree.draft_tokens,
            retrieve_index=tree.retrieve_index,
            retrieve_next_token=tree.next_child,
            retrieve_next_sibling=tree.next_sibling,
            custom_mask=tree.custom_mask,
            positions=tree.positions,
            draft_token_num=tree.tokens_per_request,
            max_tree_depth=tree.max_depth,
            tree_topk=-1,
        )
        io = eagle_worker_common.make_post_verify_io(
            verify_input=verify_input,
            sampling_info=_sampling(),
            sequence_lengths=torch.tensor([7, 11]),
            request_pool_indices=torch.tensor([3, 7]),
            device="cpu",
            logits_output=_post_io().logits,
            grammar_mask=None,
            watermark=None,
            watermark_full_mask=True,
            return_logprob=False,
            is_idle=False,
        )
        self.assertIsNone(io.verify_tree.draft_probabilities)

    def test_graph_sample_uses_the_bound_local_key_strategy(self):
        from sglang.srt.speculative.eagle_post_verify_graphs import _sample

        custom_keys = lambda view: None
        output = spec.PostVerifyOutput(
            token_ids=torch.tensor([5]),
            accepted_lengths=torch.tensor([1]),
            accepted_indices=torch.tensor([0]),
        )
        calls = []

        def run(io, local_keys):
            calls.append((io, local_keys))
            return output

        hooks = spec.SpeculativeRuntimeHooks(
            post_verify=spec.PostVerifyStrategy(run=run, local_keys=custom_keys)
        )
        sampling = _sampling()
        slot = SimpleNamespace(
            staged={
                "seq_lens": sampling.sequence_lengths,
                "req_pool_indices": torch.tensor([3, 7]),
                "draft_tokens": _tree().draft_tokens,
                "retrieve_index": _tree().retrieve_index,
                "retrieve_next_token": _tree().next_child,
                "retrieve_next_sibling": _tree().next_sibling,
            },
            sampling_info=sampling,
            is_all_greedy=True,
            need_top_k=False,
            need_top_p=False,
            unused_sampling_tensor=torch.empty(0),
            tokens_per_request=3,
            max_tree_depth=3,
            device="cpu",
            logits=torch.ones((6, 5)),
            vocab_shard=None,
            return_logprob=False,
            max_top_k=None,
        )

        got = _sample(slot, hooks)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1], custom_keys)
        self.assertIs(calls[0][0].logits.next_token_logits, slot.logits)
        expected_outputs = (
            output.token_ids,
            output.accepted_lengths,
            output.accepted_indices,
        )
        for actual, expected in zip(got, expected_outputs):
            self.assertIs(actual, expected)

    def test_master_switch_reads_only_its_exact_env_bool(self):
        for value in (False, True):
            with (
                self.subTest(value=value),
                envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(value),
                patch.object(
                    binding,
                    "megakernel_enabled",
                    side_effect=AssertionError("must be orthogonal to the CLI"),
                ) as cli_enabled,
            ):
                self.assertIs(binding.nv_mega_runtime_optimization_enabled(), value)
                cli_enabled.assert_not_called()

    def test_switch_off_returns_stock_identity_without_provider_work(self):
        defaults = _defaults()
        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=lambda _: (_ for _ in ()).throw(
                AssertionError("factory must not run")
            )
        )
        with (
            envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False),
            patch.object(binding, "megakernel_agree") as agree,
            patch.object(
                binding,
                "built_megakernel",
                side_effect=AssertionError("off must not inspect a provider"),
            ) as built,
        ):
            selected = binding.bind_speculative_runtime_hooks(model, defaults)
        self.assertIs(selected, defaults)
        agree.assert_not_called()
        built.assert_not_called()

    def test_missing_provider_keeps_exact_defaults(self):
        defaults = _defaults()
        models = (torch.nn.Module(), torch.nn.Module())
        models[1].__dict__["_megakernel"] = SimpleNamespace(widths=())

        for model in models:
            with (
                self.subTest(has_built=binding.built_megakernel(model) is not None),
                envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                patch.object(
                    binding, "_gather_tp", side_effect=lambda value, group=None: [value]
                ),
            ):
                selected = binding.bind_speculative_runtime_hooks(model, defaults)
            self.assertIs(selected, defaults)
            self.assertIs(selected.post_verify.run, runtime.default_post_verify)
            self.assertIs(selected.post_verify.local_keys, runtime.default_local_keys)

    def test_present_factory_can_wrap_defaults_and_change_only_local_keys(self):
        defaults = _defaults()
        custom_keys = lambda view: torch.zeros(
            view.values.shape[0], dtype=torch.int64, device=view.values.device
        )
        received = []

        def factory(stock):
            received.append(stock)
            return spec.SpeculativeRuntimeHooks(
                post_verify=spec.PostVerifyStrategy(
                    run=stock.post_verify.run,
                    local_keys=custom_keys,
                )
            )

        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=factory
        )
        with (
            envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(binding, "megakernel_agree") as agree,
        ):
            selected = binding.bind_speculative_runtime_hooks(model, defaults)

        self.assertEqual(received, [defaults])
        self.assertIs(selected.post_verify.run, defaults.post_verify.run)
        self.assertIs(selected.post_verify.local_keys, custom_keys)
        agree.assert_called_once_with(
            True, what="speculative runtime hooks factory presence"
        )

    def test_factory_cannot_replace_the_runtime_verify_runner(self):
        defaults = _defaults()

        def factory(stock):
            return dataclasses.replace(
                stock,
                post_verify=dataclasses.replace(
                    stock.post_verify,
                    run=lambda io, local_keys: stock.post_verify(io),
                ),
            )

        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=factory
        )
        with (
            envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(binding, "_gather_tp", return_value=[True]),
            self.assertRaisesRegex(ValueError, "cannot replace.*runner"),
        ):
            binding.bind_speculative_runtime_hooks(model, defaults)

    def test_factory_must_return_a_callable_local_key_policy(self):
        defaults = _defaults()

        def factory(stock):
            return dataclasses.replace(
                stock,
                post_verify=dataclasses.replace(stock.post_verify, local_keys=None),
            )

        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=factory
        )
        with (
            envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(binding, "_gather_tp", return_value=[True]),
            self.assertRaisesRegex(TypeError, "local_keys must be callable"),
        ):
            binding.bind_speculative_runtime_hooks(model, defaults)

    def test_provider_factory_presence_must_agree_across_ranks(self):
        defaults = _defaults()
        model = torch.nn.Module()
        model.__dict__["_megakernel"] = SimpleNamespace(
            speculative_runtime_hooks=lambda stock: stock
        )
        with (
            envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(
                binding,
                "_gather_tp",
                side_effect=lambda value, group=None: [value, not value],
            ),
            self.assertRaisesRegex(
                RuntimeError, "speculative runtime hooks factory presence differ"
            ),
        ):
            binding.bind_speculative_runtime_hooks(model, defaults)

    def test_default_post_verify_matches_direct_stock_greedy_bitwise(self):
        from sglang.srt.speculative import eagle_utils

        io = _post_io()
        old_tree = SimpleNamespace(
            draft_token=io.verify_tree.draft_tokens,
            draft_probs=io.verify_tree.draft_probabilities,
            retrieve_index=io.verify_tree.retrieve_index,
            retrieve_next_token=io.verify_tree.next_child,
            retrieve_next_sibling=io.verify_tree.next_sibling,
            custom_mask=io.verify_tree.custom_mask,
            positions=io.verify_tree.positions,
            draft_token_num=io.verify_tree.tokens_per_request,
            max_tree_depth=io.verify_tree.max_depth,
            tree_topk=io.verify_tree.topk,
        )
        old_sampling = SimpleNamespace(
            temperatures=io.sampling.temperatures,
            top_ks=io.sampling.top_ks,
            top_ps=io.sampling.top_ps,
            min_ps=io.sampling.min_ps,
            acc_additive_penalties=io.sampling.additive_penalties,
            acc_scaling_penalties=io.sampling.scaling_penalties,
            logit_bias=io.sampling.logit_bias,
            sampling_seed=io.sampling.sampling_seed,
            is_all_greedy=io.sampling.is_all_greedy,
            need_top_k_sampling=io.sampling.need_top_k_sampling,
            need_top_p_sampling=io.sampling.need_top_p_sampling,
            need_min_p_sampling=io.sampling.need_min_p_sampling,
            watermark_keys=io.sampling.watermark_keys,
            watermark_context_windows=io.sampling.watermark_context_windows,
            watermark_enabled=io.sampling.watermark_enabled,
            has_watermark_candidates=io.sampling.has_watermark_candidates,
            max_top_k=io.sampling.max_top_k,
        )
        old_batch = SimpleNamespace(
            device=io.sampling.device,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=io.sampling.sequence_lengths,
            sampling_info=old_sampling,
            req_pool_indices=io.request_pool_indices,
            return_logprob=io.return_logprob,
        )
        runtime_spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )

        def cpu_greedy(
            *,
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
            topk,
        ):
            del topk
            target_flat = target_predict.reshape(-1)
            for row in range(candidates.shape[0]):
                last = int(retrieve_index[row, 0])
                accept_index[row, 0] = last
                accepted = 0
                child = 0
                for _ in range(1, accept_index.shape[1]):
                    child = int(retrieve_next_token[row, child])
                    while child != -1:
                        draft_index = int(retrieve_index[row, child])
                        if int(candidates[row, child]) == int(target_flat[last]):
                            predicts[last] = target_flat[last]
                            accepted += 1
                            accept_index[row, accepted] = draft_index
                            last = draft_index
                            break
                        child = int(retrieve_next_sibling[row, child])
                    if child == -1:
                        break
                accept_token_num[row] = accepted
                predicts[last] = target_flat[last]
            return predicts, accept_index, accept_token_num

        with (
            envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False),
            patch.object(eagle_utils, "get_spec", return_value=runtime_spec),
            patch.object(
                eagle_utils, "verify_tree_greedy_func", side_effect=cpu_greedy
            ),
        ):
            direct = eagle_utils.eagle_sample(
                old_tree,
                old_batch,
                SimpleNamespace(
                    next_token_logits=io.logits.next_token_logits.clone(),
                    next_token_logits_vocab_shard=None,
                ),
            )
            defaults = _defaults()
            actual = defaults.post_verify(_post_io(io.logits.next_token_logits.clone()))

        for stock, selected in zip(direct, dataclasses.astuple(actual)):
            self.assertTrue(torch.equal(stock, selected))


if __name__ == "__main__":
    unittest.main()
