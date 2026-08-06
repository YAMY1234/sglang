import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferCombineInput,
    FlashinferDispatcher,
    _scattered_source_token_counts,
    _workspace_size_for_namespace,
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
    TorchDistributedCommBackend,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    is_speculative_moe_a2a_context,
    speculative_moe_a2a_backend_context,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _FakeMoeAlltoAll:
    def __init__(self, workspace_payload):
        self.workspace_payload = workspace_payload
        self.payload_in_workspace = None
        self.runtime_max_tokens_per_rank = None

    def dispatch(
        self,
        _token_selected_experts,
        input_payloads,
        runtime_max_tokens_per_rank,
        **_kwargs,
    ):
        self.runtime_max_tokens_per_rank = runtime_max_tokens_per_rank
        hidden_states, topk_ids, topk_weights = input_payloads
        num_recv_tokens = 2 * runtime_max_tokens_per_rank
        return (
            torch.empty(
                num_recv_tokens,
                hidden_states.shape[-1],
                dtype=hidden_states.dtype,
            ),
            torch.empty(
                num_recv_tokens,
                topk_ids.shape[-1],
                dtype=topk_ids.dtype,
            ),
            torch.empty(
                num_recv_tokens,
                topk_weights.shape[-1],
                dtype=topk_weights.dtype,
            ),
        )

    def get_combine_payload_tensor_in_workspace(
        self, runtime_max_tokens_per_rank, hidden_size, dtype
    ):
        expected_shape = (2, runtime_max_tokens_per_rank, hidden_size)
        assert self.workspace_payload.shape == expected_shape
        assert self.workspace_payload.dtype == dtype
        return self.workspace_payload

    def combine(
        self,
        hidden_states,
        runtime_max_tokens_per_rank,
        *,
        payload_in_workspace,
    ):
        self.payload_in_workspace = payload_in_workspace
        return hidden_states[0]


class TestFlashinferA2AWorkspaceOwnership(unittest.TestCase):
    def test_target_and_draft_decode_use_distinct_workspace_keys(self):
        self.assertEqual(_workspace_size_for_namespace(4096, speculative=False), 4096)
        self.assertEqual(_workspace_size_for_namespace(4096, speculative=True), 4224)

    def test_prefill_counts_expand_to_post_scatter_sources(self):
        self.assertEqual(
            _scattered_source_token_counts([7, 3], 4),
            [2, 2, 2, 1, 1, 1, 1, 0],
        )
        self.assertEqual(_scattered_source_token_counts([4] * 4, 1), [4] * 4)

    def test_speculative_context_is_nested_and_restored(self):
        self.assertFalse(is_speculative_moe_a2a_context())
        with patch(
            "sglang.srt.layers.moe.utils.get_speculative_moe_a2a_backend",
            return_value=object(),
        ):
            with speculative_moe_a2a_backend_context():
                self.assertTrue(is_speculative_moe_a2a_context())
                with speculative_moe_a2a_backend_context():
                    self.assertTrue(is_speculative_moe_a2a_context())
                self.assertTrue(is_speculative_moe_a2a_context())
        self.assertFalse(is_speculative_moe_a2a_context())

    def test_workspace_cache_identity_follows_process_group(self):
        group_a = object()
        group_b = object()

        self.assertEqual(
            TorchDistributedCommBackend(group_a).workspace_cache_key(),
            TorchDistributedCommBackend(group_a).workspace_cache_key(),
        )
        self.assertNotEqual(
            TorchDistributedCommBackend(group_a).workspace_cache_key(),
            TorchDistributedCommBackend(group_b).workspace_cache_key(),
        )

    def test_dispatch_capacity_is_per_rank(self):
        original_empty = torch.empty
        original_full = torch.full
        original_zeros = torch.zeros
        constructor_args = {}
        workspace_args = {}

        def cpu_empty(*args, **kwargs):
            kwargs = dict(kwargs)
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs)

        def cpu_full(*args, **kwargs):
            kwargs = dict(kwargs)
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return original_full(*args, **kwargs)

        def cpu_zeros(*args, **kwargs):
            kwargs = dict(kwargs)
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return original_zeros(*args, **kwargs)

        def fake_workspace_size(**kwargs):
            workspace_args.update(kwargs)
            return 123

        def fake_moe_alltoall(**kwargs):
            constructor_args.update(kwargs)
            return object()

        group = SimpleNamespace(size=lambda: 4, rank=lambda: 0)
        runner_backend = SimpleNamespace(is_flashinfer_cutlass=lambda: False)
        speculative_algo = SimpleNamespace(is_eagle=lambda: False)
        server_args = SimpleNamespace(speculative_algorithm=None)

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_moe_runner_backend",
                return_value=runner_backend,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_int_env_var",
                return_value=4096,
            ) as token_cap,
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_global_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.SpeculativeAlgorithm.from_string",
                return_value=speculative_algo,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.moe_a2a_get_workspace_size_per_rank",
                side_effect=fake_workspace_size,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.TorchDistributedCommBackend",
                return_value=object(),
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.MnnvlConfig",
                return_value=object(),
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.Mapping",
                return_value=object(),
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.MoeAlltoAll",
                side_effect=fake_moe_alltoall,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.torch.cuda.device_count",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.torch.empty",
                side_effect=cpu_empty,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.torch.full",
                side_effect=cpu_full,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.torch.zeros",
                side_effect=cpu_zeros,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.MOE_NVFP4_DISPATCH",
                False,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.StandardDispatcher",
                return_value=object(),
            ),
        ):
            dispatcher = FlashinferDispatcher(
                group=group,
                router_topk=1,
                num_experts=8,
                num_local_experts=2,
                hidden_size=4,
            )

        token_cap.assert_called_once_with(
            "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
        )
        self.assertEqual(dispatcher.max_num_tokens, 4096)
        self.assertEqual(workspace_args["max_num_tokens"], 4096)
        self.assertEqual(constructor_args["max_num_tokens"], 4096)
        self.assertEqual(constructor_args["workspace_size_per_rank"], 123)

    @staticmethod
    def _make_dispatcher(workspace_payload):
        dispatcher = FlashinferDispatcher.__new__(FlashinferDispatcher)
        dispatcher.ep_size = 2
        dispatcher.ep_rank = 0
        dispatcher.router_topk = 1
        dispatcher.hidden_size = 4
        dispatcher.num_local_experts = 1
        dispatcher.quant_config = {}
        dispatcher.allocate_combine_payload_in_workspace = True
        dispatcher._workspace_combine_payload = None
        dispatcher.moe_a2a = _FakeMoeAlltoAll(workspace_payload)
        dispatcher.dummy_x = torch.empty(1, 4)
        dispatcher.dummy_topk_ids = torch.full((1, 1), -1, dtype=torch.int32)
        dispatcher.dummy_topk_ids_current_rank = torch.zeros(1, 1, dtype=torch.int32)
        dispatcher.dummy_topk_weights = torch.zeros(1, 1)
        return dispatcher

    @staticmethod
    def _dispatch(
        dispatcher, num_tokens=3, global_num_tokens=None, dtype=torch.float32
    ):
        hidden_states = torch.empty(num_tokens, 4, dtype=dtype)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(num_tokens, 1),
            topk_ids=torch.zeros(num_tokens, 1, dtype=torch.int32),
            router_logits=None,
        )
        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_dp_global_num_tokens",
                return_value=global_num_tokens,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_is_extend_in_batch",
                return_value=False,
            ),
        ):
            return dispatcher.dispatch(hidden_states, topk_output)

    def test_prefill_uses_allgatherv_and_reduce_scatterv(self):
        dispatcher = FlashinferDispatcher.__new__(FlashinferDispatcher)
        dispatcher.ep_size = 2
        dispatcher.ep_rank = 0

        class FakePrefillDispatcher:
            def dispatch(_self, hidden_states, topk_output):
                self.assertEqual(hidden_states.shape, (3, 4))
                self.assertEqual(topk_output.topk_ids.shape, (3, 1))
                return StandardDispatchOutput(hidden_states, None, topk_output)

        class FakeTpGroup:
            def all_gatherv(_self, tensors, *, sizes):
                self.assertEqual(sizes, [2, 1])
                return tuple(
                    torch.cat([tensor, tensor[:1]], dim=0) for tensor in tensors
                )

            def reduce_scatterv(_self, hidden_states, *, sizes):
                self.assertEqual(sizes, [2, 1])
                self.assertEqual(hidden_states.shape, (3, 4))
                return hidden_states[:2]

        dispatcher.prefill_dispatcher = FakePrefillDispatcher()
        hidden_states = torch.empty(2, 4, dtype=torch.bfloat16)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            router_logits=None,
        )

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_is_extend_in_batch",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_dp_global_num_tokens",
                return_value=[3],
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_attention_tp_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_tp_group",
                return_value=FakeTpGroup(),
            ),
        ):
            dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
            combined = dispatcher.combine(
                StandardCombineInput(dispatch_output.hidden_states)
            )

        self.assertEqual(dispatch_output.format.name, "STANDARD")
        self.assertEqual(combined.shape, (2, 4))
        self.assertFalse(hasattr(dispatcher, "prefill_source_sizes"))

    def test_sequence_parallel_uses_post_scatter_token_count(self):
        workspace_payload = torch.empty(2, 4, 4)
        dispatcher = self._make_dispatcher(workspace_payload)

        dispatch_output = self._dispatch(
            dispatcher,
            num_tokens=4,
            global_num_tokens=[16],
        )

        self.assertEqual(dispatcher.runtime_max_tokens_per_rank, 4)
        self.assertEqual(dispatcher.moe_a2a.runtime_max_tokens_per_rank, 4)
        self.assertEqual(dispatch_output.hidden_states.shape, (8, 4))
        self.assertEqual(dispatch_output.moe_output.shape, (8, 4))

    def test_dp_attention_uses_largest_rank_token_count(self):
        workspace_payload = torch.empty(2, 5, 4)
        dispatcher = self._make_dispatcher(workspace_payload)

        dispatch_output = self._dispatch(
            dispatcher,
            num_tokens=3,
            global_num_tokens=[3, 5],
        )

        self.assertEqual(dispatcher.runtime_max_tokens_per_rank, 5)
        self.assertEqual(dispatcher.moe_a2a.runtime_max_tokens_per_rank, 5)
        self.assertEqual(dispatch_output.hidden_states.shape, (10, 4))
        self.assertEqual(dispatch_output.moe_output.shape, (10, 4))

    def test_exact_workspace_payload_uses_in_place_combine(self):
        workspace_payload = torch.empty(2, 3, 4)
        dispatcher = self._make_dispatcher(workspace_payload)
        dispatch_output = self._dispatch(dispatcher)

        self.assertEqual(
            dispatch_output.moe_output.data_ptr(), workspace_payload.data_ptr()
        )
        dispatcher.combine(FlashinferCombineInput(dispatch_output.moe_output))

        self.assertTrue(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))

    def test_unquantized_mtp_runner_writes_to_workspace_payload(self):
        workspace_payload = torch.empty(2, 3, 4)
        dispatcher = self._make_dispatcher(workspace_payload)
        dispatch_output = self._dispatch(dispatcher)

        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        method.use_flashinfer_cutlass = True
        method.moe_runner_config = SimpleNamespace(activation="silu")
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
        layer = SimpleNamespace(
            w13_weight=torch.empty(1, 8, 4),
            w2_weight=torch.empty(1, 4, 4),
            moe_ep_size=2,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        )

        def fake_cutlass_fused_moe(**kwargs):
            self.assertIs(kwargs["output"], dispatch_output.moe_output)
            self.assertTrue(kwargs["enable_alltoall"])
            return [kwargs["output"]]

        flashinfer_a2a_backend = SimpleNamespace(is_flashinfer=lambda: True)
        with (
            patch(
                "sglang.srt.layers.quantization.unquant.flashinfer_cutlass_fused_moe",
                side_effect=fake_cutlass_fused_moe,
            ),
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_a2a_backend",
                return_value=flashinfer_a2a_backend,
            ),
        ):
            combine_input = method.forward_cuda(layer, dispatch_output)

        self.assertEqual(
            combine_input.hidden_states.data_ptr(), workspace_payload.data_ptr()
        )
        self.assertEqual(combine_input.format.name, "FLASHINFER")
        dispatcher.combine(combine_input)

        self.assertTrue(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))

    def test_unquantized_mtp_external_result_uses_copy_combine(self):
        workspace_payload = torch.empty(2, 3, 4)
        external_payload = torch.empty(6, 4)
        dispatcher = self._make_dispatcher(workspace_payload)
        dispatch_output = self._dispatch(dispatcher)

        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        method.use_flashinfer_cutlass = True
        method.moe_runner_config = SimpleNamespace(activation="silu")
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
        layer = SimpleNamespace(
            w13_weight=torch.empty(1, 8, 4),
            w2_weight=torch.empty(1, 4, 4),
            moe_ep_size=2,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        )

        def fake_cutlass_fused_moe(**kwargs):
            self.assertIs(kwargs["output"], dispatch_output.moe_output)
            self.assertTrue(kwargs["enable_alltoall"])
            return [external_payload]

        flashinfer_a2a_backend = SimpleNamespace(is_flashinfer=lambda: True)
        with (
            patch(
                "sglang.srt.layers.quantization.unquant.flashinfer_cutlass_fused_moe",
                side_effect=fake_cutlass_fused_moe,
            ),
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_a2a_backend",
                return_value=flashinfer_a2a_backend,
            ),
        ):
            combine_input = method.forward_cuda(layer, dispatch_output)

        self.assertEqual(
            combine_input.hidden_states.data_ptr(), external_payload.data_ptr()
        )
        self.assertEqual(combine_input.format.name, "FLASHINFER")
        dispatcher.combine(combine_input)

        self.assertFalse(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))

    def test_unquantized_standard_prefill_disables_cutlass_internal_alltoall(self):
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        method.use_flashinfer_cutlass = True
        method.moe_runner_config = SimpleNamespace(activation="silu")
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
        layer = SimpleNamespace(
            w13_weight=torch.empty(1, 8, 4),
            w2_weight=torch.empty(1, 4, 4),
            moe_ep_size=2,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.empty(6, 4),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.ones(6, 1),
                topk_ids=torch.zeros(6, 1, dtype=torch.int32),
                router_logits=None,
            ),
        )

        def fake_cutlass_fused_moe(**kwargs):
            self.assertIsNone(kwargs["output"])
            self.assertFalse(kwargs["enable_alltoall"])
            return [torch.empty_like(dispatch_output.hidden_states)]

        with patch(
            "sglang.srt.layers.quantization.unquant.flashinfer_cutlass_fused_moe",
            side_effect=fake_cutlass_fused_moe,
        ):
            combine_input = method.forward_cuda(layer, dispatch_output)

        self.assertEqual(combine_input.format.name, "STANDARD")
        self.assertEqual(combine_input.hidden_states.shape, (6, 4))

    def test_modelopt_fp4_decode_returns_flashinfer_combine_input(self):
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )

        workspace_payload = torch.empty(2, 3, 4, dtype=torch.bfloat16)
        dispatcher = self._make_dispatcher(workspace_payload)
        dispatch_output = self._dispatch(dispatcher, dtype=torch.bfloat16)

        method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        method.enable_flashinfer_trtllm_moe = False
        method.moe_runner_config = SimpleNamespace(
            activation="silu",
            apply_router_weight_on_input=False,
        )
        layer = SimpleNamespace(
            w13_weight=torch.empty(1, 8, 8, dtype=torch.uint8),
            w2_weight=torch.empty(1, 4, 8, dtype=torch.uint8),
            w13_input_scale_quant=torch.ones(1),
            w13_blockscale_swizzled=torch.ones(1, dtype=torch.int32),
            g1_alphas=torch.ones(1),
            w2_input_scale_quant=torch.ones(1),
            w2_blockscale_swizzled=torch.ones(1, dtype=torch.int32),
            g2_alphas=torch.ones(1),
            moe_ep_size=2,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        )

        def fake_cutlass_fused_moe(**kwargs):
            self.assertIs(kwargs["output"], dispatch_output.moe_output)
            self.assertTrue(kwargs["enable_alltoall"])
            return [kwargs["output"]]

        with (
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=SimpleNamespace(
                    is_flashinfer_cutedsl=lambda: False,
                    is_flashinfer_cutlass=lambda: True,
                ),
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.flashinfer_cutlass_fused_moe",
                side_effect=fake_cutlass_fused_moe,
            ),
        ):
            combine_input = method.apply(layer, dispatch_output)

        self.assertEqual(combine_input.format.name, "FLASHINFER")
        self.assertEqual(
            combine_input.hidden_states.data_ptr(), workspace_payload.data_ptr()
        )
        dispatcher.combine(combine_input)
        self.assertTrue(dispatcher.moe_a2a.payload_in_workspace)

    def test_modelopt_fp4_standard_prefill_disables_cutlass_internal_alltoall(self):
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )

        method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        method.enable_flashinfer_trtllm_moe = False
        method.moe_runner_config = SimpleNamespace(
            activation="silu",
            apply_router_weight_on_input=False,
        )
        layer = SimpleNamespace(
            w13_weight=torch.empty(1, 8, 8, dtype=torch.uint8),
            w2_weight=torch.empty(1, 4, 8, dtype=torch.uint8),
            w13_input_scale_quant=torch.ones(1),
            w13_blockscale_swizzled=torch.ones(1, dtype=torch.int32),
            g1_alphas=torch.ones(1),
            w2_input_scale_quant=torch.ones(1),
            w2_blockscale_swizzled=torch.ones(1, dtype=torch.int32),
            g2_alphas=torch.ones(1),
            moe_ep_size=2,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.empty(6, 4, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.ones(6, 1),
                topk_ids=torch.zeros(6, 1, dtype=torch.int32),
                router_logits=None,
            ),
        )

        def fake_cutlass_fused_moe(**kwargs):
            self.assertFalse(kwargs["enable_alltoall"])
            return [torch.empty_like(dispatch_output.hidden_states)]

        with (
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=SimpleNamespace(
                    is_flashinfer_cutedsl=lambda: False,
                    is_flashinfer_cutlass=lambda: True,
                ),
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.flashinfer_cutlass_fused_moe",
                side_effect=fake_cutlass_fused_moe,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_tp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
        ):
            combine_input = method.apply(layer, dispatch_output)

        self.assertEqual(combine_input.format.name, "STANDARD")
        self.assertEqual(combine_input.hidden_states.shape, (6, 4))

    def test_external_payload_uses_copy_combine(self):
        workspace_payload = torch.empty(2, 3, 4)
        external_payload = torch.empty(6, 4)
        dispatcher = self._make_dispatcher(workspace_payload)
        self._dispatch(dispatcher)

        dispatcher.combine(FlashinferCombineInput(external_payload))

        self.assertFalse(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))

    def test_same_storage_with_different_offset_uses_copy_combine(self):
        backing = torch.empty(7, 4)
        workspace_payload = backing[:6].view(2, 3, 4)
        offset_alias = backing[1:]
        self.assertEqual(
            workspace_payload.untyped_storage().data_ptr(),
            offset_alias.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(workspace_payload.data_ptr(), offset_alias.data_ptr())
        dispatcher = self._make_dispatcher(workspace_payload)
        self._dispatch(dispatcher)

        dispatcher.combine(FlashinferCombineInput(offset_alias))

        self.assertFalse(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))

    def test_all_idle_dp_ranks_still_dispatch_one_dummy_token(self):
        workspace_payload = torch.empty(2, 1, 4)
        dispatcher = self._make_dispatcher(workspace_payload)
        topk_output = StandardTopKOutput(
            topk_weights=torch.empty(0, 1),
            topk_ids=torch.empty(0, 1, dtype=torch.int32),
            router_logits=None,
        )

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_dp_global_num_tokens",
                return_value=[0, 0],
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_is_extend_in_batch",
                return_value=False,
            ),
        ):
            dispatch_output = dispatcher.dispatch(torch.empty(0, 4), topk_output)

        self.assertEqual(dispatcher.runtime_max_tokens_per_rank, 1)
        self.assertEqual(dispatcher.moe_a2a.runtime_max_tokens_per_rank, 1)
        hidden_states = dispatcher.combine(
            FlashinferCombineInput(dispatch_output.moe_output)
        )
        self.assertEqual(hidden_states.shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
