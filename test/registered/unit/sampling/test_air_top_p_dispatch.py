"""Which kernel the deterministic top-p renorm dispatches to, under the formal DEP16 config.

SGLANG_OPT_AIR_TOP_P_PARALLEL routes the deterministic branch to the parallel AIR kernel
(bitwise equal to flashinfer's deterministic AIR) when the inputs fit it; off, or on inputs it
does not take, the flashinfer call is unchanged. CUDA calls are mocked; the dispatch, the
policy, the env registry and the input checks are the real ones.

TestAirTopPWorkerRecord runs the scheduler's internal-state readback (the per-worker observation
hook) with the real imports; only the CUDA pieces (JIT load, device tensors) are stand-ins.
"""

import asyncio
import hashlib
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.jit.utils.arch import CompileArch
from sglang.kernels.ops.sampling import air_top_p_parallel as air
from sglang.srt.environ import envs
from sglang.srt.layers import sampling_renorm
from sglang.srt.managers.io_struct import (
    GetInternalStateReq,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler_components import internal_state
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.runtime_context import (
    SpawnRanks,
    get_context,
    get_parallel,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_FORMAL = dict(tp_size=16, attn_dp_size=16, enable_dp_lm_head=True)


class _CudaProbs:
    """Stands in for a CUDA fp32 [rows, vocab] tensor (this test runs without a GPU)."""

    is_cuda = True
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self, rows=112, vocab=154880, ptr=1 << 20, contiguous=True):
        self.shape = (rows, vocab)
        self._ptr, self._contiguous = ptr, contiguous

    def dim(self):
        return 2

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._ptr


class TestAirTopPDispatch(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.fi = mock.patch.object(
            sampling_renorm._flashinfer_sampling, "top_p_renorm_probs", return_value="fi"
        ).start()
        self.par = mock.patch.object(
            sampling_renorm, "air_top_p_renorm_parallel", return_value="parallel"
        ).start()
        self.fast = mock.patch.object(
            sampling_renorm._platform, "top_p_renorm_prob", return_value="fast"
        ).start()
        self.addCleanup(mock.patch.stopall)
        self.enterContext(mock.patch.dict(air._bypass_totals, clear=True))
        assert sampling_renorm._FI_TOP_P_DETERMINISTIC

    def _call(self, probs, top_p, flag, parallel=True):
        with envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.override(flag):
            with get_context().override_server_args(**_FORMAL):
                self.assertEqual(get_parallel().tp_size, 16)
                self.assertEqual(get_parallel().attn_tp_size, 1)
                self.assertTrue(sampling_renorm.renorm_deterministic())
                # top_p_renorm_prob is this function wherever the platform has a fast kernel
                # (CUDA); on a CPU-only host it is None, so call the dispatch directly.
                return sampling_renorm._top_p_renorm_prob(probs, top_p, parallel=parallel)

    @unittest.skipUnless(torch.cuda.is_available(), "public alias is None without CUDA")
    def test_public_alias_is_the_dispatch(self):
        self.assertIs(sampling_renorm.top_p_renorm_prob, sampling_renorm._top_p_renorm_prob)

    def test_flag_off_keeps_flashinfer(self):
        probs, top_p = _CudaProbs(), torch.full((112,), 0.95)
        self.assertEqual(self._call(probs, top_p, False), "fi")
        self.fi.assert_called_once_with(probs, top_p, is_deterministic=True)
        self.par.assert_not_called()
        self.fast.assert_not_called()

    def test_flag_on_takes_parallel(self):
        for top_p in (torch.full((112,), 0.95), 0.95):
            self.par.reset_mock()
            probs = _CudaProbs()
            self.assertEqual(self._call(probs, top_p, True), "parallel")
            self.par.assert_called_once_with(probs, top_p)
        self.fi.assert_not_called()
        self.fast.assert_not_called()

    def test_flag_on_without_opt_in_keeps_flashinfer(self):
        # The plain sampler and DFLASH call the dispatch without `parallel`.
        probs, top_p = _CudaProbs(), torch.full((112,), 0.95)
        self.assertEqual(self._call(probs, top_p, True, parallel=False), "fi")
        self.fi.assert_called_once_with(probs, top_p, is_deterministic=True)
        self.par.assert_not_called()

    def test_verify_wrapper_opts_in(self):
        probs, top_p = _CudaProbs(), torch.full((112,), 0.95)
        with mock.patch.object(eagle_utils, "top_p_renorm_prob", sampling_renorm._top_p_renorm_prob):
            with envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.override(True):
                with get_context().override_server_args(**_FORMAL):
                    self.assertEqual(eagle_utils._verify_top_p_renorm_prob(probs, top_p), "parallel")
        self.par.assert_called_once_with(probs, top_p)

    def test_flag_on_unsupported_inputs_keep_flashinfer(self):
        cases = [
            (_CudaProbs(vocab=1000), 0.95),  # flashinfer's small-vocab kernel
            (_CudaProbs(vocab=154882), 0.95),  # not a whole number of float4
            (_CudaProbs(ptr=(1 << 20) + 4), 0.95),  # misaligned
            (_CudaProbs(contiguous=False), 0.95),
            (_CudaProbs(rows=112), torch.full((8,), 0.95)),  # top_p rows mismatch
        ]
        for probs, top_p in cases:
            self.fi.reset_mock()
            self.assertEqual(self._call(probs, top_p, True), "fi")
            self.fi.assert_called_once()
        self.par.assert_not_called()
        # Counted as bypassed (stock kernel used), never as parallel calls.
        self.assertEqual(
            air._bypass_totals,
            {"vocab": [2, 224], "layout": [2, 224], "top_p_tensor": [1, 112]},
        )

    def test_flag_off_or_no_opt_in_counts_no_bypass(self):
        self._call(_CudaProbs(vocab=1000), 0.95, False)
        self._call(_CudaProbs(vocab=1000), 0.95, True, parallel=False)
        self.assertEqual(air._bypass_totals, {})


class _CudaProbsOn1(_CudaProbs):
    device = torch.device("cuda", 1)


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


class TestAirTopPWorkerRecord(CustomTestCase):
    """The record a D worker returns in its /server_info internal state, read inside the worker."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        # DEP16 worker at world rank 5 on local GPU 1.
        publish(
            ServerArgs(model_path="dummy", **_FORMAL),
            role="scheduler",
            ranks=SpawnRanks(world_rank=5, gpu_id=1),
        )
        for state in (air._fallback_totals, air._generations, air._host_totals):
            self.enterContext(mock.patch.dict(state, clear=True))
        self.enterContext(mock.patch.dict(air._bypass_totals, clear=True))
        self.enterContext(mock.patch.dict(air._loaded, clear=True))
        # The query must never load, build or allocate.
        self.enterContext(
            mock.patch.object(air, "load_jit", side_effect=AssertionError("query loaded the JIT module"))
        )

    def _readback(self, flag=True):
        # The other contributors read a live scheduler; the air_top_p one is real.
        others = {
            name: mock.MagicMock()
            for name in vars(SchedulerInternalStateController)
            if name.startswith("_add_") and name != "_add_air_top_p"
        }
        controller = SchedulerInternalStateController(scheduler=SimpleNamespace(), record_step_time=False)
        with mock.patch.multiple(SchedulerInternalStateController, **others):
            with envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.override(flag):
                output = controller.get_internal_state(GetInternalStateReq())
        # As it crosses the scheduler -> tokenizer IPC.
        return msgpack_decode(msgpack_encode(output)).internal_state["air_top_p"]

    def _run_parallel(self, rows):
        real_zeros = torch.zeros
        with mock.patch.object(air, "reserve_air_top_p_workspace", return_value=(None, 128)), mock.patch.object(
            air, "_jit_module"
        ), mock.patch.object(torch, "zeros", side_effect=lambda *a, device=None, **k: real_zeros(*a, **k)), mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            air.air_top_p_renorm_parallel(_CudaProbsOn1(rows=rows), 0.95, out=object())

    def test_required_target_checked_before_build(self):
        build = air._jit_module.__wrapped__
        with mock.patch.object(air, "load_jit") as load, mock.patch.object(air, "_module_identity", return_value={}):
            with mock.patch.object(air, "get_compile_arch", return_value=CompileArch(10, 7, "a")):
                with envs.SGLANG_AIR_TOP_P_REQUIRED_TARGET.override("10.0f"):
                    with self.assertRaisesRegex(RuntimeError, "'10.7a' != required '10.0f'"):
                        build(air.CUDA_FILE)
                load.assert_not_called()
            with mock.patch.object(air, "get_compile_arch", return_value=CompileArch(10, 0, "f")):
                with envs.SGLANG_AIR_TOP_P_REQUIRED_TARGET.override("10.0f"):
                    build(air.CUDA_FILE)
            load.assert_called_once()

    def test_not_loaded(self):
        record = self._readback(flag=False)
        self.assertEqual(record["status"], "OK")
        self.assertFalse(record["flag"])
        self.assertEqual(
            {k: record[k] for k in ("pid", "device", "tp_rank", "attn_dp_rank", "attn_tp_rank", "pp_rank")},
            {"pid": os.getpid(), "device": 1, "tp_rank": 5, "attn_dp_rank": 5, "attn_tp_rank": 0, "pp_rank": 0},
        )
        self.assertEqual(record["py_sha256"], _sha256(air.__file__))
        self.assertEqual(record["cuh_sha256"], _sha256(air._CUH_PATH))
        self.assertTrue(record["cuh_path"].endswith("csrc/sampling/air_top_p_parallel.cuh"))
        self.assertEqual((record["module_status"], record["module"]), ("NOT_LOADED", None))
        self.assertIsNone(record["counter_generation"])
        for key in ("calls", "total_rows", "kernel_fallback_rows", "engaged_rows"):
            self.assertEqual(record[key], 0)
        self.assertEqual(record["bypass"], {})

    def test_loaded_identity_and_counters(self):
        tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        leaf = tmp / "cuda_10.0f" / "sgl_kernel_jit_sampling_air_top_p_parallel" / "build-abc" / "deps-def"
        leaf.mkdir(parents=True)
        so = leaf / "sgl_kernel_jit_sampling_air_top_p_parallel.so"
        so.write_bytes(b"fake library")
        nvcc = tmp / "nvcc"
        nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n")
        nvcc.chmod(0o755)
        # The identity half of `_jit_module`, with the build stood in.
        with mock.patch.object(air, "load_jit", return_value="module"), mock.patch.object(
            air, "_mapped_library", return_value=str(so)
        ), mock.patch.object(air.toolchain, "device_compiler_path", return_value=str(nvcc)), mock.patch.object(
            air, "get_compile_arch", return_value=CompileArch(10, 0, "f")
        ):
            self.assertEqual(air._jit_module.__wrapped__(air.CUDA_FILE), "module")
        self._run_parallel(112)
        self._run_parallel(96)
        air._fallback_totals[1].fill_(3)  # the kernel's device counter
        air.air_top_p_note_bypass("vocab", 8)

        first, second = self._readback(), self._readback()
        self.assertTrue(first["flag"])
        self.assertEqual(first["module_status"], "LOADED")
        module = first["module"]
        self.assertEqual(module["compile_target"], "10.0f")
        self.assertIn("-DSGL_CUDA_ARCH=1000", module["compile_cuda_flags"])
        self.assertIn("-use_fast_math", module["compile_cuda_flags"])
        self.assertEqual((module["build_key_dir"], module["deps_key_dir"]), ("build-abc", "deps-def"))
        self.assertEqual((module["so_path"], module["so_sha256"]), (str(so), _sha256(so)))
        self.assertEqual(module["cuh_sha256_at_load"], first["cuh_sha256"])
        self.assertEqual((module["nvcc_path"], module["nvcc_sha256"]), (str(nvcc), _sha256(nvcc)))
        self.assertEqual(module["nvcc_version"], "Cuda compilation tools, release 13.0, V13.0.88")
        self.assertEqual(
            {k: first[k] for k in ("calls", "total_rows", "kernel_fallback_rows", "engaged_rows")},
            {"calls": 2, "total_rows": 208, "kernel_fallback_rows": 3, "engaged_rows": 205},
        )
        self.assertEqual(first["bypass"], {"vocab": {"calls": 1, "rows": 8}})
        self.assertRegex(first["counter_generation"], "^[0-9a-f]{32}$")
        self.assertEqual(first["counter_generation"], second["counter_generation"])
        self.assertEqual(first["bypass_generation"], second["bypass_generation"])
        self.assertLessEqual(first["wall_time"], second["wall_time"])

    def test_error_is_reported_not_zeroed(self):
        with mock.patch.object(internal_state, "air_top_p_worker_record", side_effect=RuntimeError("boom")):
            record = self._readback()
        self.assertEqual((record["status"], record["error"]), ("ERROR", "RuntimeError: boom"))
        self.assertEqual(record["pid"], os.getpid())
        self.assertNotIn("calls", record)

    def test_tokenizer_keeps_one_record_per_worker(self):
        outputs = []
        for world_rank in (0, 1):
            publish(
                ServerArgs(model_path="dummy", **_FORMAL),
                role="scheduler",
                ranks=SpawnRanks(world_rank=world_rank, gpu_id=world_rank),
            )
            others = {
                name: mock.MagicMock()
                for name in vars(SchedulerInternalStateController)
                if name.startswith("_add_") and name != "_add_air_top_p"
            }
            with mock.patch.multiple(SchedulerInternalStateController, **others):
                outputs.append(
                    SchedulerInternalStateController(SimpleNamespace(), False).get_internal_state(
                        GetInternalStateReq()
                    )
                )

        async def communicator(req):
            return outputs

        tm = SimpleNamespace(auto_create_handle_loop=lambda: None, get_internal_state_communicator=communicator)
        states = asyncio.run(TokenizerControlMixin.get_internal_state(tm))
        self.assertEqual([s["air_top_p"]["attn_dp_rank"] for s in states], [0, 1])
        self.assertEqual([s["air_top_p"]["device"] for s in states], [0, 1])

    def test_mapped_library_reads_this_process(self):
        path = air._mapped_library("/libc10.so")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
