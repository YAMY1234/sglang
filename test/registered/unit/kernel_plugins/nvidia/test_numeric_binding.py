"""CPU tests of local numeric contracts and exact-capability fallback."""

import dataclasses
import unittest
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

from sglang.kernels.plugin_kernels.nvidia import numeric as api
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.kernel_plugins.nvidia.numeric_binding import bind_numeric_operator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class TestNumericBinding(unittest.TestCase):
    def test_collector_accepts_every_declared_signature_type(self):
        classes = get_args(api.OperatorSignature)
        self.assertEqual(len(classes), 9)
        signatures = tuple(
            cls(
                **{
                    field.name: 1
                    for field in dataclasses.fields(cls)
                    if field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                }
            )
            for cls in classes
        )
        producer = SimpleNamespace(numeric_signatures=lambda *_: signatures)
        self.assertEqual(
            numeric_binding.collect_numeric_signatures((producer,), (4,), 4),
            signatures,
        )

    def test_reserved_operations_have_no_removed_surface(self):
        for reserved in (1, 13, 15, 16, 20, 21):
            self.assertNotIn(
                reserved, [int(operation) for operation in api.NumericOperation]
            )
        for name in (
            "QkFrontIO",
            "NUMERIC_API_VERSION",
            "GdnCoreVariant",
            "GdnNativeStorage",
            "PleSignature",
            "PleHashIO",
            "PleGatherIO",
            "HcExtendIO",
            "HcSharedApplyIO",
            "HcSharedApplyNormIO",
        ):
            self.assertFalse(hasattr(api, name))

    def setUp(self):
        self.signature = api.NumericSignature(
            api.NumericOperation.SPARSE_SELECT, 4, 6, 1, 256, 512, 4, 1
        )
        # Buffer stand-ins test ownership/identity only; no tensor math is claimed.
        self.output = []
        self.io = api.SparseSelectIO([], [], 512, self.output, [], [])
        self.stock_calls = []
        self.provider_calls = []

        def stock(io):
            self.stock_calls.append(io)

        def provider(io):
            self.provider_calls.append(io)
            io.selected_indices_out.append(7)

        self.stock = stock
        self.provider = provider

    def bind(self, **kwargs):
        args = dict(
            enabled=True,
            signature=self.signature,
            prepared=(self.signature,),
            offers=(api.NumericOffer(self.signature, self.provider),),
            stock=self.stock,
        )
        args.update(kwargs)
        return bind_numeric_operator(**args)

    def test_disabled_is_stock_by_identity(self):
        self.assertIs(self.bind(enabled=False), self.stock)

    def test_absent_provider_is_stock_by_identity(self):
        self.assertIs(self.bind(offers=()), self.stock)

    def test_unprepared_backend_cannot_be_enabled_by_provider(self):
        self.assertIs(self.bind(prepared=()), self.stock)

    def test_exact_shape_required(self):
        other = dataclasses.replace(self.signature, rows=8)
        self.assertIs(self.bind(signature=other), self.stock)

    def test_exact_numeric_policy_required(self):
        other = dataclasses.replace(self.signature, scalar_type=2)
        self.assertIs(self.bind(signature=other, prepared=(other,)), self.stock)

    def test_provider_writes_caller_output_without_stock(self):
        self.bind()(self.io)
        self.assertEqual(self.output, [7])
        self.assertIs(self.provider_calls[0], self.io)
        self.assertEqual(self.stock_calls, [])

    def test_duplicate_offers_refused(self):
        offer = api.NumericOffer(self.signature, self.provider)
        with self.assertRaises(ValueError):
            self.bind(offers=(offer, offer))

    def test_wrong_view_refused_before_provider(self):
        with self.assertRaises(TypeError):
            self.bind()(object())
        self.assertEqual(self.provider_calls, [])

    def test_returned_replacement_buffer_refused_without_replaying_stock(self):
        offer = api.NumericOffer(self.signature, lambda io: [])
        with self.assertRaises(TypeError):
            self.bind(offers=(offer,))(self.io)
        self.assertEqual(self.stock_calls, [])

    def test_frozen_view_retains_borrowed_buffer(self):
        self.assertIs(self.io.selected_indices_out, self.output)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.io.selected_count = 1

    def test_reserved_packed_operation_cannot_bind_even_with_a_provider_offer(self):
        signature = dataclasses.replace(
            self.signature, operation=api.NumericOperation.PACKED_ATTENTION
        )
        kernel = lambda io: self.fail("unsupported provider must never run")
        self.assertIs(
            bind_numeric_operator(
                enabled=True,
                signature=signature,
                prepared=(signature,),
                offers=(api.NumericOffer(signature, kernel),),
                stock=self.stock,
            ),
            self.stock,
        )

    def test_removed_packed_views_have_no_surface_and_direct_pool_is_explicit(self):
        self.assertFalse(hasattr(api, "SparseSelection"))
        self.assertFalse(hasattr(api, "PackedAttentionIO"))
        self.assertNotIn(
            api.NumericOperation.PACKED_ATTENTION, numeric_binding._IO_TYPES
        )
        direct = {field.name for field in dataclasses.fields(api.DirectPoolAttentionIO)}
        self.assertIn("kv", direct)
        self.assertIn("out_cache_loc", direct)
        self.assertNotIn("packed_k", direct)


class TestLocalMix(unittest.TestCase):
    def test_only_explicit_combined_and_normalized_outputs_written(self):
        residual, combined, normalized = [4], [], []
        io = api.HcApplyNormIO([2], residual, [1], [1], 1e-6, 2, combined, normalized)
        signature = api.HcSignature(api.NumericOperation.HC_APPLY_NORM, 1, 4, 2560, 1)

        def kernel(view):
            view.combined_out.append(view.block[0] + view.residual[0])
            view.normalized_out.append(1)

        run = bind_numeric_operator(
            enabled=True,
            signature=signature,
            prepared=(signature,),
            offers=(api.NumericOffer(signature, kernel),),
            stock=lambda io: None,
        )
        run(io)
        self.assertEqual(residual, [4])
        self.assertEqual(combined, [6])
        self.assertEqual(normalized, [1])


class TestRecurrentAndExperts(unittest.TestCase):
    def test_core_cannot_receive_checkpoint_commit_callback(self):
        fields = {field.name for field in dataclasses.fields(api.GdnCoreIO)}
        self.assertIn("checkpoint", fields)
        self.assertIn("ring", fields)
        self.assertNotIn("commit", fields)
        self.assertNotIn("backend", fields)
        self.assertNotIn("runner", fields)

    def test_wide_shape_is_not_implied_by_small_offer(self):
        small = api.GdnSignature(
            api.NumericOperation.GDN_CORE, 4, 4, 12, 128, 128, 4, 1
        )
        wide = dataclasses.replace(small, rows=32)
        stock = lambda io: None
        self.assertIs(
            bind_numeric_operator(
                enabled=True,
                signature=wide,
                prepared=(wide,),
                offers=(api.NumericOffer(small, lambda io: None),),
                stock=stock,
            ),
            stock,
        )

    def test_local_finalize_has_no_communication_state(self):
        fields = {field.name for field in dataclasses.fields(api.FinalizeSharedIO)}
        self.assertEqual(
            fields,
            {
                "hidden",
                "gate_weight",
                "shared",
                "expert_rows",
                "route_weights",
                "cols",
                "use_pdl",
                "output",
            },
        )
        output = []
        io = api.FinalizeSharedIO([], [], [], [], [], 256, False, output)
        signature = api.ExpertSignature(
            api.NumericOperation.FINALIZE_SHARED, 4, 2560, 192, 512, 10, 1
        )

        def kernel(view):
            view.output.append(1)

        run = bind_numeric_operator(
            enabled=True,
            signature=signature,
            prepared=(signature,),
            offers=(api.NumericOffer(signature, kernel),),
            stock=lambda io: None,
        )
        run(io)
        self.assertEqual(output, [1])

    def test_expert_geometry_is_exact(self):
        base = api.ExpertSignature(
            api.NumericOperation.LOCAL_EXPERTS, 4, 2560, 192, 512, 10, 1
        )
        other = dataclasses.replace(base, intermediate_size=640)
        stock = lambda io: None
        self.assertIs(
            bind_numeric_operator(
                enabled=True,
                signature=other,
                prepared=(other,),
                offers=(api.NumericOffer(base, lambda io: None),),
                stock=stock,
            ),
            stock,
        )


class TestNumericRegistration(unittest.TestCase):
    def setUp(self):
        self.registry = api.NumericRegistry()
        self.signature = api.NumericSignature(
            api.NumericOperation.SPARSE_SELECT,
            4,
            8,
            1,
            128,
            512,
            4,
            api.ScalarType.FP32,
        )

    def test_same_offer_is_idempotent(self):
        offer = api.NumericOffer(self.signature, lambda io: None)
        self.registry.register(offer)
        self.registry.register(offer)
        self.assertIs(self.registry.get(self.signature), offer)

    def test_conflicting_offer_is_refused(self):
        self.registry.register(api.NumericOffer(self.signature, lambda io: None))
        with self.assertRaises(ValueError):
            self.registry.register(api.NumericOffer(self.signature, lambda io: None))

    def test_missing_offer_does_not_create_a_capability(self):
        self.assertIsNone(self.registry.get(self.signature))

    def test_scalar_ids_are_shared_by_all_families(self):
        self.assertEqual(api.ScalarType.BF16, 1)
        self.assertEqual(api.ScalarType.FP16, 2)
        self.assertEqual(api.ScalarType.FP32, 3)


class TestFrameworkNumericGate(unittest.TestCase):
    def setUp(self):
        self.signature = api.NumericSignature(
            api.NumericOperation.SPARSE_SELECT,
            1,
            8,
            1,
            128,
            512,
            4,
            api.ScalarType.FP32,
        )
        self.registry = api.NumericRegistry()
        self.calls = []
        self.registry.register(
            api.NumericOffer(self.signature, lambda io: self.calls.append(io))
        )

    def test_no_offer_does_not_import_or_activate_a_provider(self):
        with patch.object(numeric_binding, "numeric_registry", api.NumericRegistry()):
            self.assertIsNone(
                numeric_binding.numeric_operator(self.signature, prepared=True)
            )

    def test_master_off_preserves_stock_with_plugin_enabled(self):
        from sglang.srt.environ import envs

        with (
            patch.object(numeric_binding, "numeric_registry", self.registry),
            patch(
                "sglang.srt.kernel_plugins.nvidia.binding.megakernel_enabled",
                return_value=True,
            ),
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False),
        ):
            self.assertIsNone(
                numeric_binding.numeric_operator(self.signature, prepared=True)
            )
        self.assertEqual(self.calls, [])

    def test_plugin_off_preserves_stock_with_master_enabled(self):
        from sglang.srt.environ import envs

        with (
            patch.object(numeric_binding, "numeric_registry", self.registry),
            patch(
                "sglang.srt.kernel_plugins.nvidia.binding.megakernel_enabled",
                return_value=False,
            ),
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
        ):
            self.assertIsNone(
                numeric_binding.numeric_operator(self.signature, prepared=True)
            )
        self.assertEqual(self.calls, [])

    def test_both_gates_require_complete_framework_metadata(self):
        from sglang.srt.environ import envs

        with (
            patch.object(numeric_binding, "numeric_registry", self.registry),
            patch(
                "sglang.srt.kernel_plugins.nvidia.binding.megakernel_enabled",
                return_value=True,
            ),
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
        ):
            self.assertIsNone(
                numeric_binding.numeric_operator(self.signature, prepared=False)
            )
            run = numeric_binding.numeric_operator(self.signature, prepared=True)
            io = api.SparseSelectIO([], [], 512, [], None, [])
            run(io)
        self.assertEqual(self.calls, [io])


class TestNumericStartupMetadata(unittest.TestCase):
    def setUp(self):
        self.signature = api.NumericSignature(
            api.NumericOperation.SPARSE_SELECT,
            4,
            8,
            1,
            128,
            512,
            4,
            api.ScalarType.FP32,
        )

    def test_only_finite_signatures_cross_the_startup_boundary(self):
        calls = []
        signature = self.signature

        class Producer:
            def numeric_signatures(self, widths, tokens):
                calls.append((widths, tokens))
                return (signature, signature)

        out = numeric_binding.collect_numeric_signatures(
            (object(), Producer()), (1, 4, 8), 4
        )
        self.assertEqual(out, (signature,))
        self.assertEqual(calls, [((1, 4, 8), 4)])

    def test_mutable_declaration_refused(self):
        class Producer:
            def numeric_signatures(self, widths, tokens):
                return []

        with self.assertRaises(TypeError):
            numeric_binding.collect_numeric_signatures((Producer(),), (4,), 4)

    def test_framework_object_cannot_be_smuggled_into_signature(self):
        signature = dataclasses.replace(self.signature, rows=object())

        class Producer:
            def numeric_signatures(self, widths, tokens):
                return (signature,)

        with self.assertRaises(TypeError):
            numeric_binding.collect_numeric_signatures((Producer(),), (4,), 4)

    def test_old_context_and_spec_keep_numeric_provider_disabled(self):
        from sglang.kernels.plugin_kernels.nvidia import spec

        context = spec.BuildContext(widths=(4,), tokens_per_request=4, deadline_s=1.0)
        self.assertEqual(context.numeric_signatures, ())
        descriptor = spec.MegakernelSpec(
            name="numeric-default",
            requires=spec.Requirements(isas=frozenset(), tp_sizes=frozenset()),
            features=frozenset(),
            numerics=spec.Numerics(spec.NumericsKind.BITWISE),
            max_rows=0,
            build=lambda *args: None,
            api_version=spec.MEGAKERNEL_API_VERSION,
        )
        self.assertEqual(descriptor.numeric_max_rows, 0)


class TestNumericOfferLog(unittest.TestCase):
    LOGGER = "sglang.srt.kernel_plugins.nvidia.numeric_binding"

    def setUp(self):
        def select(rows):
            return api.NumericSignature(
                api.NumericOperation.SPARSE_SELECT,
                rows,
                8,
                1,
                128,
                512,
                4,
                api.ScalarType.FP32,
            )

        self.offered, self.missing = select(4), select(8)
        self.registry = api.NumericRegistry()
        self.registry.register(api.NumericOffer(self.offered, lambda io: None))
        for patcher in (
            patch.object(numeric_binding, "numeric_registry", self.registry),
            patch.object(numeric_binding, "_offer_log_keys", set()),
            patch.object(numeric_binding, "_OFFER_LOG", True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def messages(self, logs):
        return [record.getMessage() for record in logs.records]

    def test_off_by_default_and_silent_on_dispatch(self):
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_NVIDIA_MEGAKERNEL_OFFER_LOG.get())
        with (
            patch.object(numeric_binding, "_OFFER_LOG", False),
            self.assertNoLogs(self.LOGGER, level="INFO"),
        ):
            numeric_binding.numeric_operator(self.missing, prepared=False)
            numeric_binding.numeric_operator(self.missing, prepared=True)

    def test_dispatch_logs_each_stock_reason_and_binding_once(self):
        from sglang.srt.environ import envs

        with self.assertLogs(self.LOGGER, level="INFO") as logs:
            for _ in range(2):
                numeric_binding.numeric_operator(self.missing, prepared=False)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("view not prepared", logs.records[0].getMessage())
        for plugin, master, why in (
            (True, False, "plugin or runtime master off"),
            (True, True, "provider bound"),
        ):
            numeric_binding._offer_log_keys.clear()
            with (
                patch(
                    "sglang.srt.kernel_plugins.nvidia.binding.megakernel_enabled",
                    return_value=plugin,
                ),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master),
                self.assertLogs(self.LOGGER, level="INFO") as logs,
            ):
                numeric_binding.numeric_operator(self.offered, prepared=True)
                numeric_binding.numeric_operator(self.missing, prepared=True)
            self.assertEqual(len(logs.records), 2)
            self.assertIn(why, logs.records[0].getMessage())
            self.assertIn("no provider offer", logs.records[1].getMessage())

    def test_capability_outcome_logged_once_per_query(self):
        from sglang.srt.environ import envs

        spec = SimpleNamespace(
            numeric_capability=lambda query: query.attributes[0][1] == 4
        )
        with (
            patch(
                "sglang.srt.kernel_plugins.nvidia.binding.megakernel_enabled",
                return_value=True,
            ),
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            self.assertLogs(self.LOGGER, level="INFO") as logs,
        ):
            for signature in (self.offered, self.missing, self.missing):
                numeric_binding.signature_capability_supported(
                    signature, spec=spec, tp_size=4
                )
        messages = self.messages(logs)
        self.assertEqual(len(messages), 2)
        self.assertTrue(messages[0].endswith("accepted"))
        self.assertIn("'tp_size': 4", messages[1])
        self.assertTrue(messages[1].endswith("rejected by the plugin"))

    def test_startup_ledger_names_role_layer_op_and_silent_producers(self):
        import torch

        offered, missing = self.offered, self.missing

        class Producer(torch.nn.Module):
            def __init__(self, declared):
                super().__init__()
                self.declared = declared

            def numeric_signatures(self, widths, tokens):
                return self.declared

        model = torch.nn.Module()
        model.indexer = Producer((offered, missing))
        model.silent = Producer(())
        declarations = {}
        numeric_binding.collect_numeric_signatures(
            tuple(model.modules()), (4, 8), 4, declarations=declarations
        )
        self.assertEqual(
            declarations, {model.indexer: (offered, missing), model.silent: ()}
        )
        with self.assertLogs(self.LOGGER, level="INFO") as logs:
            for _ in range(2):
                numeric_binding.log_numeric_offers(model, declarations, role="target")
        self.assertEqual(
            self.messages(logs),
            [
                "NVIDIA megakernel offer: target indexer SPARSE_SELECT: "
                "offered rows [4]; no offer for rows [8]",
                "NVIDIA megakernel offer: target silent: declared nothing",
            ],
        )


class TestLocalProducerViews(unittest.TestCase):
    def test_mix_passes_framework_buffers_and_explicit_tp_policy(self):
        import torch

        from sglang.kernels.ops.gemm import hc_mix_splitk

        class CudaMetadataTensor(torch.Tensor):
            # CPU storage tests only host publication, never a GPU launch.
            @property
            def is_cuda(self):
                return True

        x = torch.zeros((4, 10240), dtype=torch.bfloat16).as_subclass(
            CudaMetadataTensor
        )
        down = torch.zeros((16, 10240), dtype=torch.bfloat16)
        up = torch.zeros((10240, 128), dtype=torch.bfloat16)
        scratch = torch.zeros((32, 128), dtype=torch.bfloat16)
        seen = []

        def bind(signature, *, prepared):
            self.assertTrue(prepared)
            self.assertEqual(signature.variant, 1)
            self.assertEqual(signature.lowrank, 16)
            return lambda io: seen.append(io)

        with (
            patch.object(hc_mix_splitk, "_get_scratch", return_value=scratch),
            patch.object(numeric_binding, "numeric_operator", side_effect=bind),
        ):
            out = hc_mix_splitk.hc_mix(x, down, up, 4, 2560, numeric_variant=1)
        self.assertIs(seen[0].normed, x)
        self.assertIs(seen[0].output, out)
        self.assertEqual(seen[0].scratch_out.shape, (4, 128))
        self.assertEqual(seen[0].scratch_out.data_ptr(), scratch.data_ptr())


class TestLocalHcBoundary(unittest.TestCase):
    def test_no_next_gate_keeps_two_result_stock_api(self):
        from types import SimpleNamespace

        import torch

        from sglang.srt.layers.hyperconnection import GatedResidual

        combined, normalized = object(), object()
        calls = []
        layer = SimpleNamespace(
            combine_and_normalize=lambda *args: (
                calls.append(args) or (combined, normalized)
            ),
        )
        block = torch.empty((4, 2560), dtype=torch.bfloat16)
        residual = torch.empty((4, 10240), dtype=torch.bfloat16)
        returned = GatedResidual.combine_normalize_and_gate(
            layer,
            block,
            (residual, residual, None),
            None,
            None,
        )
        self.assertEqual(returned, (combined, normalized, None))
        self.assertEqual(len(calls), 1)

    def test_complete_boundary_writes_only_three_caller_outputs(self):
        from types import SimpleNamespace

        import torch

        from sglang.srt.layers.hyperconnection import GatedResidual

        class CudaMetadataTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        block = torch.zeros((4, 2560), dtype=torch.bfloat16).as_subclass(
            CudaMetadataTensor
        )
        residual = torch.zeros((4, 10240), dtype=torch.bfloat16)
        partials = torch.zeros((4, 8, 4), dtype=torch.float32)
        weight = torch.zeros((10240,), dtype=torch.bfloat16)
        next_weight = torch.zeros((4, 10240), dtype=torch.bfloat16)
        norm = SimpleNamespace(group_size=2560, weight=weight, variance_epsilon=1e-6)
        calls = []
        layer = SimpleNamespace(
            hc_count=4,
            hidden_size=2560,
            _numeric_hc_macro=lambda rows: lambda io: calls.append(io),
            combine_and_normalize=lambda *args: self.fail(
                "stock must not replay after provider writes"
            ),
        )
        returned = GatedResidual.combine_normalize_and_gate(
            layer,
            block,
            (residual, residual, partials),
            norm,
            next_weight,
        )
        self.assertEqual(len(calls), 1)
        io = calls[0]
        self.assertIs(io.residual, residual)
        self.assertIs(io.gate_partials, partials)
        self.assertIs(io.next_inject_weight, next_weight)
        self.assertTrue(
            all(
                a is b
                for a, b in zip(
                    returned,
                    (io.combined_out, io.normalized_out, io.next_partials_out),
                )
            )
        )
        self.assertEqual(io.next_partials_out.shape, (4, 8, 4))
        self.assertEqual(io.parts, 2)

    def test_macro_policy_off_never_selects_offer(self):
        from types import SimpleNamespace

        from sglang.srt.environ import envs
        from sglang.srt.layers.hyperconnection import GatedResidual

        layer = SimpleNamespace(
            hc_count=4,
            hidden_size=2560,
            config=SimpleNamespace(hc_per_branch_norm=True),
        )
        with envs.SGLANG_HC_MACRO.override(False):
            self.assertIsNone(GatedResidual._numeric_hc_macro(layer, 4))


class TestDirectAttentionProducer(unittest.TestCase):
    def test_full_lengths_and_explicit_pool_writes_reach_raw_view(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace

        import torch

        path = (
            Path(api.__file__).resolve().parents[3]
            / "srt/layers/attention/qwen_sparse_attn_backend.py"
        )
        module = ast.parse(path.read_text())
        cls = next(
            n
            for n in module.body
            if isinstance(n, ast.ClassDef) and n.name == "QwenSparseAttnBackend"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_try_numeric_direct_attention"
        )
        scope = {"torch": torch, "get_bool_env_var": lambda *args: True}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            scope,
        )

        class CudaMetadataTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        q = torch.zeros((4, 6, 256), dtype=torch.bfloat16).as_subclass(
            CudaMetadataTensor
        )
        gate = torch.zeros_like(q)
        kv = torch.zeros((8, 1, 256), dtype=torch.bfloat16)
        new_k, new_v = kv[:4], kv[4:]
        ids = torch.zeros((4, 512), dtype=torch.int32)
        loc = torch.arange(4, dtype=torch.int64)
        lengths = torch.tensor([2052, 2053, 2054, 2055], dtype=torch.int32)
        pos = torch.arange(2052, 2056, dtype=torch.int64)
        rows = torch.zeros(4, dtype=torch.int64)
        slot_map = torch.zeros((1, 8192), dtype=torch.int64)
        metadata = SimpleNamespace(
            indexer_metadata=SimpleNamespace(decode_logical_positions=pos),
            row_req_pool_indices=rows,
            sequence_lengths=lengths,
            is_cuda_graph=True,
        )
        backend = SimpleNamespace(
            qsa_profile=SimpleNamespace(block_topk=512),
            compress_ratio=4,
            req_to_token_pool=SimpleNamespace(req_to_token=slot_map),
            _uses_block_indices=lambda tensor: True,
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: True),
            spec_info=SimpleNamespace(topk=1),
            req_pool_indices=rows,
        )
        layer = SimpleNamespace(tp_q_head_num=6, head_dim=256, scaling=0.0625)
        seen = []

        def bind(signature, *, prepared):
            self.assertTrue(prepared)
            self.assertEqual(
                signature.operation, api.NumericOperation.DIRECT_POOL_ATTENTION
            )
            return lambda io: seen.append(io)

        with patch.object(numeric_binding, "numeric_operator", side_effect=bind):
            out = scope["_try_numeric_direct_attention"](
                backend,
                q,
                new_k,
                new_v,
                layer,
                batch,
                ids,
                gate,
                kv,
                kv.clone(),
                loc,
                4,
                metadata,
            )
        self.assertEqual(out.shape, (4, 1536))
        self.assertEqual(len(seen), 1)
        view = seen[0]
        self.assertIs(view.selection.sequence_lengths, lengths)
        self.assertIs(view.selection.full_slot_map, slot_map)
        self.assertIs(view.selection.query_positions, pos)
        self.assertIs(view.out_cache_loc, loc)
        self.assertTrue(view.store_new_kv)
        self.assertEqual(view.scratch_out.numel(), 4 * 33 * 16 * (256 + 2))
        self.assertEqual(view.output.data_ptr(), out.data_ptr())
        self.assertNotIn("compressed_kv", {f.name for f in dataclasses.fields(view)})
        self.assertNotIn(
            "compressed_lengths", {f.name for f in dataclasses.fields(view.selection)}
        )


class TestLocalExpertProducer(unittest.TestCase):
    def test_recipe_policy_never_infers_batch34_from_rows(self):
        from unittest.mock import patch

        from sglang.srt.kernel_plugins.nvidia.local_experts import (
            local_expert_source_context,
        )

        flags = {
            "SGLANG_Q38_MK_MOE_SF_PACK": "1",
            "SGLANG_Q38_MKM_WIDE": "1",
            "SGLANG_Q38_MKM_WIDE_BATCH34": "1",
            "SGLANG_Q38_MKM_WIDE_M8": "0",
        }
        with patch.dict("os.environ", flags):
            for rows in (1, 4, 8, 12, 16):
                facts = local_expert_source_context(
                    rows, 4, is_target_verify=False, batch_size=0
                )
                self.assertEqual(facts["batch_size"], 0)
                self.assertEqual(facts["is_target_verify"], 0)
                self.assertEqual(facts["wide_batch34_enabled"], 1)
                self.assertEqual(facts["mkm_wide_m8_enabled"], 0)

    def test_deferred_result_uses_framework_identity_and_no_final_output(self):
        import os
        from enum import IntEnum
        from types import SimpleNamespace

        import torch

        from sglang.srt.kernel_plugins.nvidia import local_experts

        class Activation(IntEnum):
            Swiglu = 0

        class CudaMetadataTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        def tensor(shape, dtype):
            return torch.empty(shape, dtype=dtype, device="meta")

        x = tensor((4, 1280), torch.uint8).as_subclass(CudaMetadataTensor)
        kwargs = dict(
            hidden_states=x,
            hidden_states_scale=tensor((4, 160), torch.float8_e4m3fn),
            routing_logits=tensor((4, 512), torch.bfloat16),
            gemm1_weights=tensor((512, 384, 1280), torch.uint8),
            gemm2_weights=tensor((512, 2560, 96), torch.uint8),
            gemm1_weights_scale=tensor((1,), torch.float8_e4m3fn),
            gemm2_weights_scale=tensor((1,), torch.float8_e4m3fn),
            output1_scale_gate_scalar=tensor((512,), torch.float32),
            output1_scale_scalar=tensor((512,), torch.float32),
            output2_scale_scalar=tensor((512,), torch.float32),
            num_experts=512,
            intermediate_size=192,
            local_num_experts=512,
            local_expert_offset=0,
            top_k=10,
            routing_method_type=4,
            activation_type=Activation.Swiglu,
            routing_bias=None,
            per_token_scale=None,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            do_finalize=False,
        )
        seen = []

        def bind(signature, *, prepared):
            self.assertTrue(prepared)
            if signature.variant != 1:
                return None
            return lambda io: seen.append(io)

        with (
            patch.dict(
                os.environ,
                {"SGLANG_Q38_MK_MOE": "1", "SGLANG_Q38_MK_MOE_SF_PACK": "1"},
            ),
            patch.dict(
                "sys.modules",
                {"flashinfer.tllm_enums": SimpleNamespace(ActivationType=Activation)},
            ),
            patch(
                "sglang.srt.model_executor.forward_context.has_forward_context",
                return_value=False,
            ),
            patch.object(local_experts, "numeric_operator", side_effect=bind),
        ):
            rows, weights, permutation = local_experts.try_numeric_local_experts(kwargs)
            sink = []
            kwargs["do_finalize"] = True
            with local_experts.capture_local_finalize(sink):
                (staged_output,) = local_experts.try_numeric_local_experts(kwargs)
        self.assertEqual(len(sink), 1)
        self.assertIs(sink[0][2], staged_output)
        self.assertIs(sink[0][0], seen[1].expert_rows_out)
        self.assertIsNone(seen[1].final_out)
        self.assertEqual(rows.shape, (40, 2560))
        self.assertEqual(weights.shape, (4, 10))
        self.assertEqual(permutation.shape, (40,))
        self.assertEqual(permutation.dtype, torch.int32)
        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[0].final_out)
        self.assertIs(seen[0].packed_x, x)
        self.assertEqual(seen[0].expert_rows_out.shape, (4, 10, 2560))


class TestLocalSharedFinalize(unittest.TestCase):
    def test_sink_scope_restores_after_nested_exception(self):
        from sglang.srt.kernel_plugins.nvidia import local_experts

        outer, inner = [], []
        with local_experts.capture_local_finalize(outer):
            self.assertIs(local_experts._finalize_sink.get()[0], outer)
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with local_experts.capture_local_finalize(inner):
                    self.assertIs(local_experts._finalize_sink.get()[0], inner)
                    raise RuntimeError("stop")
            self.assertIs(local_experts._finalize_sink.get()[0], outer)
        self.assertIsNone(local_experts._finalize_sink.get())

    def test_native_producer_does_not_stage_or_change_output(self):
        from sglang.srt.kernel_plugins.nvidia.local_experts import (
            finish_local_shared_finalize,
        )

        # A fallback producer never registers a staged record.
        self.assertFalse(finish_local_shared_finalize([], None, None, None, None))

    def test_finish_uses_real_shared_gate_and_caller_output(self):
        import os

        import torch

        from sglang.srt.kernel_plugins.nvidia import local_experts

        class CudaMetadataTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        hidden = torch.zeros((4, 2560), dtype=torch.bfloat16).as_subclass(
            CudaMetadataTensor
        )
        gate = torch.zeros((2560,), dtype=torch.bfloat16)
        shared = torch.zeros((4, 2560), dtype=torch.bfloat16)
        expert_rows = torch.zeros((4, 10, 2560), dtype=torch.bfloat16)
        weights = torch.zeros((4, 10), dtype=torch.bfloat16)
        output = torch.empty((4, 2560), dtype=torch.bfloat16)
        seen = []
        sink = [(expert_rows, weights, output, lambda io: seen.append(io))]

        def bind(signature, *, prepared):
            self.assertTrue(prepared)
            self.assertEqual(signature.operation, api.NumericOperation.FINALIZE_SHARED)
            return lambda io: seen.append(io)

        flags = {
            "SGLANG_Q38_FINALIZE_SHARED": "1",
            "SGLANG_Q38_FINALIZE_SHARED_COLS": "256",
            "SGLANG_Q38_FINALIZE_SHARED_PDL": "1",
        }
        with (
            patch.dict(os.environ, flags),
            patch.object(local_experts, "numeric_operator", side_effect=bind),
        ):
            self.assertTrue(
                local_experts.finish_local_shared_finalize(
                    sink,
                    hidden,
                    gate,
                    shared,
                    output,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "prepared consumer"):
                local_experts.finish_local_shared_finalize(
                    sink,
                    hidden,
                    gate,
                    shared,
                    torch.empty_like(output),
                )
        self.assertEqual(len(seen), 1)
        view = seen[0]
        self.assertIs(view.hidden, hidden)
        self.assertIs(view.gate_weight, gate)
        self.assertIs(view.shared, shared)
        self.assertIs(view.expert_rows, expert_rows)
        self.assertIs(view.route_weights, weights)
        self.assertIs(view.output, output)
        self.assertEqual(view.cols, 256)
        self.assertTrue(view.use_pdl)


if __name__ == "__main__":
    unittest.main()
