import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import torch

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.srt.layers.attention.linear.kernels.kda_helion import HelionKDAKernel
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHelionKDADispatcher(unittest.TestCase):
    def _make_dispatcher(
        self,
        decode_backend,
        prefill_backend,
        verify_backend=LinearAttnKernelBackend.TRITON,
    ):
        helion_kernel = MagicMock(
            spec=HelionKDAKernel,
            supports_packed_decode=True,
            supports_packed_decode_safe_gate=HelionKDAKernel.supports_packed_decode_safe_gate,
            supports_safe_gate_verify=HelionKDAKernel.supports_safe_gate_verify,
        )
        with (
            patch(
                "sglang.srt.layers.attention.linear.kda_backend.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.linear.kernels.kda_helion.HelionKDAKernel",
                return_value=helion_kernel,
            ) as constructor,
        ):
            dispatcher = KDAKernelDispatcher(
                decode_backend=decode_backend,
                prefill_backend=prefill_backend,
                verify_backend=verify_backend,
            )
        return dispatcher, helion_kernel, constructor

    def test_combined_backend_reuses_adapter_and_keeps_triton_verify(self):
        dispatcher, helion_kernel, constructor = self._make_dispatcher(
            LinearAttnKernelBackend.HELION,
            LinearAttnKernelBackend.HELION,
        )

        constructor.assert_called_once_with(
            triton_fallback=ANY,
            enable_decode=True,
            enable_prefill=True,
        )
        self.assertIs(dispatcher.decode_kernel, helion_kernel)
        self.assertIs(dispatcher.extend_kernel, helion_kernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonKDAKernel)
        self.assertTrue(dispatcher.supports_packed_decode)

    def test_decode_only_keeps_triton_prefill_and_verify(self):
        dispatcher, helion_kernel, constructor = self._make_dispatcher(
            LinearAttnKernelBackend.HELION,
            LinearAttnKernelBackend.TRITON,
        )

        constructor.assert_called_once_with(
            triton_fallback=ANY,
            enable_decode=True,
            enable_prefill=False,
        )
        self.assertIs(dispatcher.decode_kernel, helion_kernel)
        self.assertIsInstance(dispatcher.extend_kernel, TritonKDAKernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonKDAKernel)

    def test_prefill_only_keeps_triton_decode_and_verify(self):
        dispatcher, helion_kernel, constructor = self._make_dispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.HELION,
        )

        constructor.assert_called_once_with(
            triton_fallback=ANY,
            enable_decode=False,
            enable_prefill=True,
        )
        self.assertIsInstance(dispatcher.decode_kernel, TritonKDAKernel)
        self.assertIs(dispatcher.extend_kernel, helion_kernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonKDAKernel)

    def test_explicit_helion_verify_reuses_adapter(self):
        for decode in (LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.HELION):
            with self.subTest(decode=decode):
                dispatcher, helion_kernel, _ = self._make_dispatcher(
                    decode,
                    LinearAttnKernelBackend.TRITON,
                    LinearAttnKernelBackend.HELION,
                )
                self.assertIs(dispatcher.verify_kernel, helion_kernel)
                self.assertIsInstance(dispatcher.extend_kernel, TritonKDAKernel)
                if decode.is_helion():
                    self.assertIs(dispatcher.decode_kernel, helion_kernel)
                else:
                    self.assertIsInstance(dispatcher.decode_kernel, TritonKDAKernel)

    def test_replayssm_decode_uses_native_helion_kernel(self):
        kernel = HelionKDAKernel.__new__(HelionKDAKernel)
        kernel._packed_decode = MagicMock()
        kernel._replayssm_decode = MagicMock()
        kernel._triton = MagicMock()
        mixed_qkv = torch.empty(2, 20)
        a = torch.empty(2, 8)
        b = torch.empty(2, 1)
        state = torch.empty(2, 1, 4, 8)
        indices = torch.arange(2, dtype=torch.int32)
        force_flush = torch.zeros(2, dtype=torch.int32)
        replay_args = {
            "replayssm_d": torch.empty(2, 1, 4, 4),
            "replayssm_k": torch.empty(2, 1, 4, 8),
            "replayssm_g": torch.empty(2, 1, 4, 8),
            "replayssm_write_pos": torch.zeros(2, dtype=torch.int32),
            "replayssm_force_flush": force_flush,
        }

        result = kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=torch.empty(1),
            dt_bias=torch.empty(8),
            scale=0.5,
            ssm_states=state,
            cache_indices=indices,
            num_v_heads=1,
            head_v_dim=4,
            lower_bound=-5.0,
            **replay_args,
        )

        self.assertEqual(result.shape, (1, 2, 1, 4))
        kernel._packed_decode.assert_not_called()
        kernel._replayssm_decode.assert_called_once()
        self.assertIs(
            kernel._replayssm_decode.call_args.kwargs["force_flush"], force_flush
        )
        self.assertEqual(kernel._replayssm_decode.call_args.kwargs["lower_bound"], -5.0)
        kernel._triton.packed_decode.assert_not_called()

    def test_replayssm_accepts_helion_and_rejects_other_backends(self):
        with (
            override_platform(is_sm100=False),
            override_platform(is_cuda=False),
        ):
            helion_args = ServerArgs(
                model_path="dummy",
                linear_attn_decode_backend="helion",
                enable_linear_replayssm=True,
            )
            handle_linear_attn_backend(helion_args)

            flashinfer_args = ServerArgs(
                model_path="dummy",
                linear_attn_decode_backend="flashinfer",
                enable_linear_replayssm=True,
            )
            with self.assertRaisesRegex(ValueError, "Triton, or Helion"):
                handle_linear_attn_backend(flashinfer_args)

    def test_explicit_base_backend_is_not_replaced_by_flashinfer(self):
        args = ServerArgs(
            model_path="dummy",
            linear_attn_backend="helion",
            mamba_ssm_dtype="bfloat16",
        )
        with (
            override_platform(is_sm100=True),
            override_platform(is_cuda=False),
        ):
            handle_linear_attn_backend(args)

        self.assertIsNone(args.linear_attn_decode_backend)
        self.assertEqual(args.linear_attn_backend, "helion")

    def test_pp_spec_flashinfer_verify_fallback_is_kda_only(self):
        cases = (
            ("KimiK3LinearForCausalLM", {"kda_layers": [0]}, None, "triton"),
            ("KimiK3LinearForCausalLM", {"kda_layers": [0]}, "flashinfer", "triton"),
            ("Qwen3NextForCausalLM", {"linear_attention_layers": [0]}, None, None),
        )
        for architecture, linear_config, requested, expected in cases:
            with self.subTest(architecture=architecture, requested=requested):
                args = ServerArgs(
                    model_path="dummy",
                    linear_attn_decode_backend="flashinfer",
                    linear_attn_verify_backend=requested,
                )
                config = SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=[architecture], linear_attn_config=linear_config
                    )
                )
                with (
                    patch(
                        "sglang.srt.arg_groups.attention_hook.model_config_of",
                        return_value=config,
                    ),
                    patch(
                        "sglang.srt.arg_groups.attention_hook."
                        "pp_spec_stable_rows_enabled",
                        return_value=True,
                    ),
                    override_platform(is_sm100=False),
                    override_platform(is_cuda=False),
                ):
                    handle_linear_attn_backend(args)

                self.assertEqual(
                    resolution_result(args, "linear_attn_verify_backend"), expected
                )


class TestKDATritonExtendPrecompile(unittest.TestCase):
    """The KDA backend joins the precompile that overlapped weight loading."""

    def _backend(self, pool_state):
        backend = kda_backend.KDAAttnBackend.__new__(kda_backend.KDAAttnBackend)
        dispatcher = MagicMock()
        dispatcher.effective_extend_kernel.return_value = dispatcher.triton_kernel
        backend.kernel_dispatcher = dispatcher
        pool = MagicMock(mamba_map={4: 0})
        pool.mamba2_layer_cache.return_value = SimpleNamespace(temporal=pool_state)
        backend.req_to_token_pool = pool
        return backend

    def _precompile(self, backend, handoff):
        runner = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
        config = SimpleNamespace(linear_attn_config={"gate_lower_bound": -5.0})
        with patch.dict(kda_backend._triton_extend_precompile, handoff, clear=True), patch(
            "sglang.srt.configs.hybrid_arch.hybrid_kda_config", return_value=config
        ), patch.object(
            kda_backend, "_run_triton_extend_precompile", return_value=1.0
        ) as inline, patch.object(kda_backend, "rank0_log"), patch.object(
            kda_backend, "_retune_idle", return_value=({}, {})
        ) as retune:
            backend._precompile_triton_extend(runner)
            left = dict(kda_backend._triton_extend_precompile)
        self.retune = retune
        return inline, left

    def test_side_thread_result_is_used_when_the_pool_layout_matches(self):
        state = torch.zeros(3, 2, 4, 4)
        thread = MagicMock()
        inline, left = self._precompile(
            self._backend(state),
            {"thread": thread, "seconds": 2.0, "layout": kda_backend._state_layout(state)},
        )
        thread.join.assert_called_once()
        inline.assert_not_called()
        self.retune.assert_called_once()  # configs picked again with the GPU idle
        self.assertNotIn("thread", left)

    def test_a_different_pool_layout_precompiles_again_on_the_pool(self):
        state = torch.zeros(3, 2, 4, 4)
        other = kda_backend._state_layout(torch.zeros(3, 2, 4, 8))
        backend = self._backend(state)
        inline, _ = self._precompile(
            backend, {"thread": MagicMock(), "seconds": 2.0, "layout": other}
        )
        inline.assert_called_once_with(
            backend.kernel_dispatcher.triton_kernel, state, -5.0, torch.bfloat16
        )
        self.retune.assert_not_called()  # the inline precompile already tuned idle

    def test_a_side_thread_failure_is_raised_at_backend_init(self):
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            self._precompile(
                self._backend(torch.zeros(3, 2, 4, 4)),
                {"thread": MagicMock(), "error": RuntimeError("compile failed")},
            )

    def test_without_a_side_thread_it_precompiles_on_the_pool(self):
        state = torch.zeros(3, 2, 4, 4)
        backend = self._backend(state)
        inline, _ = self._precompile(backend, {})
        inline.assert_called_once()
        self.assertEqual(
            [c.args[1:] for c in backend.kernel_dispatcher.effective_extend_kernel.call_args_list],
            [(n, n) for n, _ in kda_backend._TRITON_EXTEND_PRECOMPILE_CALLS],
        )


class _Tuner:
    def __init__(self, cache, cache_results=True):
        self.cache, self.cache_results = dict(cache), cache_results


class TestKDAIdleRetune(unittest.TestCase):
    def test_retune_drops_choices_and_disk_results_then_restores_disk_results(self):
        tuners = {"kda.o": _Tuner({"k1": "A"}), "kda.w": _Tuner({}, False)}
        seen = []

        def tune(*args):
            seen.append({n: (dict(t.cache), t.cache_results) for n, t in tuners.items()})
            tuners["kda.o"].cache["k1"] = "B"

        with patch.object(
            kda_backend, "_kda_autotuners", return_value=tuners
        ), patch.object(kda_backend, "_run_triton_extend_precompile", side_effect=tune):
            got, before = kda_backend._retune_idle(None, None, None, None)
        self.assertEqual(seen, [{"kda.o": ({}, False), "kda.w": ({}, False)}])
        self.assertEqual(before, {"kda.o": {"k1": "A"}})
        self.assertEqual(
            (tuners["kda.o"].cache_results, tuners["kda.w"].cache_results), (True, False)
        )
        self.assertEqual(kda_backend._same_choices(got, before, "overlapped"), (0, 1))
        self.assertEqual(
            kda_backend._same_choices(got, {"kda.o": {"k1": "B"}}, "idle"), (1, 1)
        )


if __name__ == "__main__":
    unittest.main()
