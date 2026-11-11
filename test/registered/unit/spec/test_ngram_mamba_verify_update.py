import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.srt
from sglang.kernels.ops.attention.linear import gdn_replayssm_commit_cuda as cuda
from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as replay
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=34, suite="base-a-test-cpu")


class TestNgramMambaVerifyUpdate(CustomTestCase):
    def _make_mock_target_worker(self):
        target_worker = MagicMock()
        target_worker.model_runner.model = MagicMock()
        target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify = (
            MagicMock()
        )
        mamba_pool = MagicMock()
        # The runner's KV pool router binds the state pool.
        router = target_worker.model_runner.token_to_kv_pool.layer_router
        router.part_view.return_value = mamba_pool
        mamba_pool.replayssm_spec_fold = False
        mamba_pool.replayssm_cache_base = None
        return target_worker

    def test_mamba_verify_update_called_with_correct_indices(self):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker()
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = None
        batch.seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        accept_lens = torch.tensor([3, 1, 5], dtype=torch.int32)
        accept_index = torch.tensor(
            [
                [0, 1, 2, -1, -1],
                [5, -1, -1, -1, -1],
                [10, 11, 12, 13, 14],
            ],
            dtype=torch.int32,
        )

        with patch(
            "sglang.srt.speculative.spec_utils.mambaish_config",
            return_value={"some": "config"},
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=5,
            )

        update_call = (
            target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify
        )
        update_call.assert_called_once()
        call_kwargs = update_call.call_args[1]
        self.assertTrue(
            torch.equal(
                call_kwargs["last_correct_step_indices"],
                torch.tensor([2, 0, 4], dtype=torch.int32),
            )
        )
        self.assertIsNone(call_kwargs["mamba_track_indices"])
        self.assertIsNone(call_kwargs["mamba_steps_to_track"])

    def test_mamba_verify_update_not_called_for_non_mamba_model(self):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker()
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = None
        accept_lens = torch.tensor([1], dtype=torch.int32)
        accept_index = torch.tensor([[0, -1, -1, -1, -1]], dtype=torch.int32)

        with patch(
            "sglang.srt.speculative.spec_utils.mambaish_config",
            return_value=None,
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=5,
            )

        update_call = (
            target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify
        )
        update_call.assert_not_called()

    def test_mamba_verify_update_with_track_indices(self):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker()
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = torch.tensor([100, 200], dtype=torch.int64)
        # Only the first request crosses the 256-token tracking boundary.
        batch.seq_lens = torch.tensor([253, 128], dtype=torch.int32)
        accept_lens = torch.tensor([4, 3], dtype=torch.int32)
        accept_index = torch.tensor(
            [
                [0, 1, 2, 3, -1],
                [5, 6, 7, -1, -1],
            ],
            dtype=torch.int32,
        )

        with (
            patch(
                "sglang.srt.speculative.spec_utils.mambaish_config",
                return_value={"some": "config"},
            ),
            patch(
                "sglang.srt.speculative.spec_utils.mamba_track_grid",
                return_value=256,
            ),
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=5,
            )

        update_call = (
            target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify
        )
        update_call.assert_called_once()
        call_kwargs = update_call.call_args[1]
        self.assertTrue(
            torch.equal(
                call_kwargs["last_correct_step_indices"],
                torch.tensor([3, 2], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                call_kwargs["mamba_steps_to_track"],
                torch.tensor([2, -1], dtype=torch.int32),
            )
        )


class TestPPReplaySSMVerifySourceRows(CustomTestCase):
    @staticmethod
    def _spec_state():
        state = torch.empty((1, 8, 4, 2), dtype=torch.float32)
        spec_state = MagicMock()
        for name in "temporal replayssm_d replayssm_k replayssm_rawv replayssm_g replayssm_beta".split():  # noqa: SIM905
            setattr(spec_state, name, state)
        spec_state.replayssm_rawk = state.unsqueeze(2)
        spec_state.conv = spec_state.intermediate_conv_window = [state]
        return spec_state

    def test_fold_helpers_read_pp_request_rows(self):
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_after_verify,
        )
        from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
            commit_kda_replayssm_after_verify,
        )

        cases = (
            (
                commit_gdn_replayssm_fold_after_verify,
                "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold",
                "commit_gdn_replayssm_fold_all_layers",
            ),
            (
                commit_kda_replayssm_after_verify,
                "sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode",
                "commit_kda_replayssm_spec_all_layers",
            ),
        )
        for commit, module, fold_name in cases:
            with (
                self.subTest(module=module),
                patch(f"{module}.{fold_name}"),
                patch(
                    "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                    "fused_conv_window_scatter_with_mask"
                ) as scatter,
            ):
                source_rows = torch.tensor([17, 23])
                commit(
                    spec_state=self._spec_state(),
                    state_batch_indices=torch.tensor([5, 7]),
                    accept_lens=torch.tensor([3, 1]),
                    last_correct_step_indices=torch.tensor([2, 0]),
                    src_indices_raw=source_rows,
                )

                scatter.assert_called_once()
                torch.testing.assert_close(scatter.call_args.args[4], source_rows)

    def test_circular_commit_reads_pp_request_rows(self):
        from sglang.srt.layers.attention.mamba.mamba2_metadata import (
            ForwardStateSlots,
        )
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = MagicMock()
        target_worker.model_runner.attn_backend.forward_state_slots.return_value = (
            ForwardStateSlots(torch.tensor([5, 7], dtype=torch.int32))
        )
        mamba_pool = MagicMock()
        router = target_worker.model_runner.token_to_kv_pool.layer_router
        router.part_view.return_value = mamba_pool
        mamba_pool.replayssm_spec_fold = False
        mamba_pool.replayssm_is_kda = False
        mamba_pool.replayssm_cache_base = torch.empty(1)
        mamba_pool.get_speculative_mamba2_params_all_layers.return_value = (
            self._spec_state()
        )
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.req_pool_indices = torch.tensor([17, 23], dtype=torch.int64)
        batch.mamba_track_indices = None
        batch.seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        accept_lens = torch.tensor([2, 1], dtype=torch.int32)
        accept_index = torch.tensor([[0, 1, -1], [3, -1, -1]], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.speculative.spec_utils.mambaish_config",
                return_value={"some": "config"},
            ),
            patch(
                "sglang.srt.speculative.spec_utils.pp_spec_stable_rows_enabled",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_spec"
            ),
            patch(
                "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_circular"
            ),
            patch(
                "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                "fused_conv_window_scatter_with_mask"
            ) as scatter,
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend."
                "commit_ple_states_after_verify"
            ),
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=3,
            )

        scatter.assert_called_once()
        torch.testing.assert_close(scatter.call_args.args[4], batch.req_pool_indices)


class TestDelayedMambaCommitBatchPairing(CustomTestCase):
    def test_flashinfer_gdn_positional_scratch_moves_to_request_rows(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            copy_verify_intermediate_rows,
        )

        destination = torch.zeros((8, 2, 3), dtype=torch.float32)
        positional = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        rows = torch.tensor([5, 2, 7], dtype=torch.int32)

        copy_verify_intermediate_rows(destination, positional, rows)

        torch.testing.assert_close(destination[rows.long()], positional)

    def test_ple_commit_reads_stable_request_rows(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        dst = torch.zeros((1, 8, 2), dtype=torch.float32)
        src = torch.zeros((1, 16, 4, 2), dtype=torch.float32)
        src[0, 11, 2] = torch.tensor([3.0, 7.0])

        HybridLinearAttnBackend._scatter_speculative_state_with_mask(
            dst,
            src,
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([11], dtype=torch.int64),
        )

        torch.testing.assert_close(dst[0, 5], torch.tensor([3.0, 7.0]))

    def test_pp_verify_scratch_uses_stable_request_rows_for_all_backends(self):
        from sglang.srt.layers.attention.linear.utils import (
            select_verify_intermediate_state_indices,
        )

        default = torch.arange(8, dtype=torch.int32)
        req_rows = torch.tensor([17, 23, 31], dtype=torch.int64)
        cache_indices = torch.tensor([4, -1, 9], dtype=torch.int32)

        with patch(
            "sglang.srt.layers.attention.linear.utils.pp_spec_stable_rows_enabled",
            return_value=True,
        ):
            result = select_verify_intermediate_state_indices(
                default, req_rows, cache_indices >= 0, pool_size=64
            )

        torch.testing.assert_close(
            result, torch.tensor([17, 64, 31], dtype=torch.int32)
        )


class TestMtpVerifyHookSignature(CustomTestCase):
    """Every ``update_mamba_state_after_mtp_verify`` override must accept the full
    keyword call the spec workers make, or it raises TypeError at verify time on
    whatever hardware it serves.

    Parses sources rather than importing: the accelerator backends defining
    overrides are exactly the ones whose deps are absent on most hosts, so an
    import-based check would skip the cases that matter.
    """

    CALL_KWARGS = {
        "last_correct_step_indices",
        "mamba_track_indices",
        "mamba_steps_to_track",
        "model",
        "req_pool_indices",
        "state_slots",
    }
    HOOK = "update_mamba_state_after_mtp_verify"

    def test_all_overrides_accept_the_call_kwargs(self):
        import ast
        import pathlib

        srt = pathlib.Path(next(iter(sglang.srt.__path__)))
        found = []
        for path in srt.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name != self.HOOK:
                    continue
                args = node.args
                if args.kwarg is not None:
                    continue  # **kwargs passthrough accepts everything
                names = {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}
                found.append((path.relative_to(srt), node.lineno, names))

        self.assertTrue(found, f"no {self.HOOK} definitions found under sglang.srt")
        for rel, lineno, names in found:
            missing = self.CALL_KWARGS - names
            self.assertFalse(
                missing,
                f"{rel}:{lineno} {self.HOOK} is missing {sorted(missing)}; "
                "the spec workers call this hook by keyword.",
            )


def _cuda_commit_inputs():
    return dict(
        checkpoint_state=torch.zeros(1, 1, 1, 128, 128, dtype=torch.bfloat16),
        d_cache=torch.zeros(1, 1, 1, 16, 128, dtype=torch.bfloat16),
        k_cache=torch.zeros(1, 1, 1, 16, 128, dtype=torch.bfloat16),
        g_cache=torch.zeros(1, 1, 1, 16),
        d_residual_cache=None,
        k_residual_cache=None,
        state_batch_indices=torch.zeros(1, dtype=torch.int64),
        replay_indices=torch.zeros(1, dtype=torch.int64),
        write_pos=torch.zeros(1, dtype=torch.int32),
        cache_base=torch.zeros(1, dtype=torch.int32),
        is_flush=torch.zeros(1, dtype=torch.int8),
        accept_lens=torch.ones(1, dtype=torch.int32),
    )


class TestGdnCudaCommitDispatch(CustomTestCase):
    """SGLANG_OPT_GDN_REPLAYSSM_CUDA_COMMIT: CPU dispatch and ABI checks; no CUDA
    compilation or numerical qualification."""

    def test_dtype_ring_and_stride_support(self):
        args = _cuda_commit_inputs()
        state, d, k = (
            args[name] for name in ("checkpoint_state", "d_cache", "k_cache")
        )
        self.assertTrue(cuda.supported(state, d, k))
        self.assertFalse(cuda.supported(state.float(), d, k))
        self.assertFalse(cuda.supported(state.transpose(-1, -2), d, k))
        self.assertFalse(cuda.supported(state, d[..., :8, :], k))

    def test_capture_before_compilation_falls_back_without_compiling(self):
        with (
            patch.object(cuda, "_KERNELS", {}),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch.object(cuda, "_kernel") as compile_kernel,
            patch.object(cuda, "commit") as launch,
        ):
            self.assertFalse(cuda.try_commit(**_cuda_commit_inputs()))
            compile_kernel.assert_not_called()
            launch.assert_not_called()

    def test_cached_variants_keep_original_arguments(self):
        args = _cuda_commit_inputs()
        kernels = {(None, 1, 1, 16, track): object() for track in (False, True)}
        with (
            patch.object(cuda, "_KERNELS", kernels),
            patch.object(cuda, "commit") as launch,
        ):
            self.assertTrue(cuda.try_commit(**args))
            launch.assert_called_once_with(**args)

    def test_cuda_success_and_native_fallback_finish_once(self):
        for enabled, success in ((False, False), (True, False), (True, True)):
            with (
                self.subTest(enabled=enabled, success=success),
                patch.object(replay, "_commit_cuda", return_value=success) as dispatch,
                patch.object(replay, "_finish_circular_fold") as finish,
                patch.object(
                    replay, "gdn_replayssm_compact_commit_kernel", MagicMock()
                ) as native,
            ):
                replay.commit_gdn_replayssm_circular(
                    **_cuda_commit_inputs(), use_cuda=enabled
                )
                self.assertEqual(dispatch.call_count, int(enabled))
                self.assertEqual(
                    native.__getitem__.call_count, 0 if enabled and success else 1
                )
                finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
