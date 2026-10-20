"""The DSpark draft runner's metadata prep through the glue graph. Only a backend that declares its prep host-free accepts it
(the DSV4 backend, then taking the draft block's extend lengths from device
constants), and the runner hands the graph the registry's out_cache_loc slot instead
of the step's own tensor."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.srt.model_executor.runner import metadata_glue_graph as glue_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend(**overrides):
    backend = dsv4.DeepseekV4AttnBackend.__new__(dsv4.DeepseekV4AttnBackend)
    fields = dict(
        is_dspark_draft=True,
        trtllm_attn=False,
        online_c128_mtp=SimpleNamespace(enabled=lambda: False),
        token_to_kv_pool=SimpleNamespace(request_window=None),
        draft_metadata_glued=False,
        _draft_extend_lens={},
    )
    fields.update(overrides)
    for name, value in fields.items():
        setattr(backend, name, value)
    return backend


def _parallel(cp=1):
    return mock.patch.object(
        dsv4, "get_parallel", return_value=SimpleNamespace(attn_cp_size=cp)
    )


class TestHostFreeDraftPrep(CustomTestCase):
    def test_the_audited_setup_accepts_and_glues(self):
        backend = _backend()
        with _parallel():
            self.assertEqual(backend.glue_draft_metadata(), "")
        self.assertTrue(backend.draft_metadata_glued)

    def test_a_backend_that_declares_nothing_refuses(self):
        why = AttentionBackend.glue_draft_metadata(SimpleNamespace())
        self.assertIn("declares no host-free draft prep", why)

    def test_every_other_setup_says_why(self):
        cases = {
            "not the DSpark draft runner": dict(is_dspark_draft=False),
            "trtllm-gen attention": dict(trtllm_attn=True),
            "online c128 MTP": dict(
                online_c128_mtp=SimpleNamespace(enabled=lambda: True)
            ),
            "a request window": dict(
                token_to_kv_pool=SimpleNamespace(request_window=object())
            ),
        }
        for why, overrides in cases.items():
            backend = _backend(**overrides)
            with _parallel(), self.subTest(why=why):
                self.assertEqual(backend.glue_draft_metadata(), why)
                self.assertFalse(backend.draft_metadata_glued)
        with _parallel(cp=2):
            self.assertEqual(_backend().glue_draft_metadata(), "context parallelism")


class TestDraftExtendLengths(CustomTestCase):
    def _moving(self, backend):
        moved = []

        def move(values):
            moved.append(list(values))
            return torch.tensor(values, dtype=torch.int32)

        backend._move_to_device = move
        return moved

    def test_unglued_moves_every_call(self):
        backend = _backend()
        moved = self._moving(backend)
        backend._draft_block_extend_lens([6, 6])
        backend._draft_block_extend_lens([6, 6])
        self.assertEqual(moved, [[6, 6], [6, 6]])

    def test_glued_keeps_one_constant_per_batch_size(self):
        backend = _backend(draft_metadata_glued=True)
        moved = self._moving(backend)
        first = backend._draft_block_extend_lens([6, 6])
        self.assertIs(backend._draft_block_extend_lens([6, 6]), first)
        backend._draft_block_extend_lens([6])
        self.assertEqual(moved, [[6, 6], [6]])
        self.assertEqual(first.tolist(), [6, 6])

    def test_glued_refuses_a_new_constant_inside_a_capture(self):
        backend = _backend(draft_metadata_glued=True)
        self._moving(backend)
        backend._draft_block_extend_lens([6])
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            # a warm-up made this one: the capture reads it by address
            backend._draft_block_extend_lens([6])
            with self.assertRaisesRegex(RuntimeError, "first seen inside"):
                backend._draft_block_extend_lens([6, 6])


def _runner(*, draft=True, backend=None):
    runner = mod.DecodeCudaGraphRunner.__new__(mod.DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(is_draft_worker=draft)
    runner.attn_backend = backend if backend is not None else _backend()
    return runner


class TestRunnerGluesTheDraft(CustomTestCase):
    def test_on_for_the_draft_with_a_host_free_backend(self):
        runner = _runner()
        with _parallel():
            self.assertTrue(runner._glues_draft_metadata())
        self.assertTrue(runner.attn_backend.draft_metadata_glued)

    def test_off_for_the_target_or_when_host_fed(self):
        with _parallel():
            self.assertFalse(_runner(draft=False)._glues_draft_metadata())
            host_fed = _runner(backend=_backend(trtllm_attn=True))
            self.assertFalse(host_fed._glues_draft_metadata())
            self.assertFalse(host_fed.attn_backend.draft_metadata_glued)
            with envs.SGLANG_RAGGED_VERIFY_MODE.override("on"):
                ragged = _runner()
                self.assertFalse(ragged._glues_draft_metadata())
                self.assertFalse(ragged.attn_backend.draft_metadata_glued)

    def test_the_glue_reads_the_registry_slot(self):
        runner = _runner()
        runner.buffers = SimpleNamespace(out_cache_loc=torch.arange(16))
        step_own = torch.arange(100, 106)
        view = SimpleNamespace(out_cache_loc=step_own)
        runner._draft_metadata_glue = False
        runner._glue_draft_inputs(view)
        self.assertIs(view.out_cache_loc, step_own)
        runner._draft_metadata_glue = True
        runner._glue_draft_inputs(view)
        slot = view.out_cache_loc
        self.assertEqual(slot.data_ptr(), runner.buffers.out_cache_loc.data_ptr())
        self.assertEqual(slot.shape, step_own.shape)
        idle = SimpleNamespace(out_cache_loc=None)
        runner._glue_draft_inputs(idle)
        self.assertIsNone(idle.out_cache_loc)


class _Prep:
    """An attention backend whose out-graph prep only counts its eager runs."""

    attn_backend_list = None
    forward_metadata = None

    def __init__(self):
        self.eager = 0

    def init_forward_metadata_out_graph(self, fb_view):
        self.eager += 1


class TestGlueTally(CustomTestCase):
    """N counts only steps a captured graph served; M every eager step after the
    first warm-up, including run()'s own fallback when the capture fails."""

    def test_a_replay_counts_n(self):
        glue, prep = glue_mod.MetadataGlueGraph("cpu"), _Prep()
        replayed = []
        glue._states["k"] = {
            "warmups": 2,
            "graph": SimpleNamespace(replay=lambda: replayed.append(1)),
            "meta": [],
        }
        glue.run(prep, None, "k")
        self.assertEqual(
            (glue.replays, glue.eager, prep.eager, replayed), (1, 0, 0, [1])
        )

    def test_a_failed_capture_counts_m_not_n(self):
        glue, prep = glue_mod.MetadataGlueGraph("cpu"), _Prep()
        with (
            mock.patch.object(glue_mod.torch.cuda, "Stream", return_value=object()),
            mock.patch.object(glue_mod.torch.cuda, "CUDAGraph", return_value=object()),
            mock.patch.object(
                glue_mod.torch.cuda, "graph", side_effect=RuntimeError("host read")
            ),
            self.assertLogs(glue_mod.logger, "WARNING"),
        ):
            for _ in range(glue.NUM_WARMUP + 1):
                glue.run(prep, None, "k")
        self.assertTrue(glue.disabled)
        # the warm-ups are not counted; the fallback that served the step is
        self.assertEqual((glue.replays, glue.eager, prep.eager), (0, 1, 3))

    def test_steps_that_never_reach_run_count_m_after_the_first_warm_up(self):
        glue, prep = glue_mod.MetadataGlueGraph("cpu"), _Prep()
        glue.note_not_routed()
        self.assertEqual(glue.eager, 0)
        glue.run(prep, None, "k")
        glue.note_not_routed()
        self.assertEqual(glue.eager, 1)

    def test_the_runner_logs_the_tally_every_k_steps(self):
        runner = _runner()
        runner._metadata_glue = glue_mod.MetadataGlueGraph("cpu")
        runner._draft_glue_logged = 0
        with mock.patch.object(mod, "DRAFT_GLUE_LOG_EVERY", 3):
            runner._metadata_glue.replays, runner._metadata_glue.eager = 2, 0
            with self.assertNoLogs(mod.logger, "DEBUG"):
                runner._maybe_log_draft_glue()
            runner._metadata_glue.eager = 1
            with self.assertLogs(mod.logger, "DEBUG") as logs:
                runner._maybe_log_draft_glue()
        self.assertIn(
            "draft glue: 2 graph replays, 1 eager (since warm-up)", logs.output[0]
        )


if __name__ == "__main__":
    unittest.main()
