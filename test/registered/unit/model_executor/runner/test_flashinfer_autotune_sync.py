"""FlashInfer autotune must reach the same tactics on every TP rank.

Without a cross-rank reduction each rank's ``argmin`` follows local timing noise
(measured: 20/20 tuned MoE shapes diverged across 4 ranks on gpt-oss-120b). The
reduction holds only if ranks also enter tuning with the same cache, so these
cover that gate and the digest it decides on.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
register_cuda_ci(est_time=22, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import json
import multiprocessing
import os
import sys
import tempfile
import traceback
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    _autotune_cache_digest,
    _autotune_tactic_sync_group,
    _drop_diverged_autotune_cache,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase, find_available_port

ENV = {"flashinfer_version": "0.6.17", "gpu": "NVIDIA GB300"}


def _gate_worker(rank, world_size, master_port, cache_paths, writer):
    """Run the entry gate on one rank per cache; report which caches survived."""
    try:
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            MASTER_ADDR="localhost",
            MASTER_PORT=str(master_port),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        kept = []
        for cache_path in cache_paths:
            _drop_diverged_autotune_cache(Path(cache_path), dist.group.WORLD, ENV)
            kept.append(Path(cache_path).is_file())
        writer.send(("ok", kept))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        writer.send(("error", f"{e}"))
    finally:
        writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


class TestAutotuneTacticSyncGroup(CustomTestCase):
    def test_single_rank_has_nobody_to_agree_with(self):
        # A 1-rank group would add a collective per tactic for no agreement.
        tp_group = SimpleNamespace(world_size=1, cpu_group=object())
        self.assertIsNone(_autotune_tactic_sync_group(tp_group))


class TestMegaMoEAutotuneStartup(CustomTestCase):
    def test_context_bridges_profile_limits(self):
        from sglang.srt.layers.moe import flashinfer_megamoe_autotune as mega_autotune

        runner = SimpleNamespace(
            device="cuda",
            forward_stream=Mock(),
            max_decode_logits_rows=Mock(),
            decode_num_tokens_per_req=Mock(),
            max_running_requests=256,
            is_draft_worker=False,
        )
        general = Mock(side_effect=lambda *_args, **_kwargs: nullcontext())
        modules = {
            "flashinfer": SimpleNamespace(autotune_v2=lambda **_: nullcontext()),
            "flashinfer.autotuner": SimpleNamespace(
                AutoTuner=SimpleNamespace(
                    get=lambda: SimpleNamespace(
                        get_namespaced_records=lambda _: {},
                        load_configs=lambda _: None,
                        save_configs=lambda _: None,
                        publish_namespaced_records=lambda *_: None,
                    )
                ),
                _collect_metadata=lambda: ENV,
                autotune=general,
                get_autotune_process_group=lambda: None,
                set_autotune_process_group=lambda _: None,
            ),
            "sglang.srt.layers.logits_processor": SimpleNamespace(
                autotune_dummy_run_mode=lambda **_: nullcontext()
            ),
        }
        # The resolved chunk size is already per DP rank. Disabled chunking
        # uses the scheduler's prefill ceiling; skip-op suppresses the bridge.
        # (chunk, skip, draft, graph rows, maximum MTP width, decode, prefill)
        cases = [
            (4096, False, False, 128 * 6, 6, 256 * 6, 4096),
            (-1, False, False, 8192, 6, 8192, 16384),
            (4096, False, False, 4096, 32, 8192, 4096),
            (4096, True, False, 128 * 6, 6, None, None),
            (4096, False, True, 128 * 6, 6, None, None),
        ]
        previous = mega_autotune._active_context
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "rank.json"
            for chunk_size, skip, draft, graph_rows, width, decode, prefill in cases:
                runner.is_draft_worker = draft
                runner.max_decode_logits_rows.return_value = graph_rows
                runner.decode_num_tokens_per_req.return_value = width
                skip_ops = {"flashinfer_megamoe"} if skip else set()
                with (
                    self.subTest(chunk_size=chunk_size, skip=skip, draft=draft),
                    get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
                    patch.dict(sys.modules, modules),
                    patch.dict(
                        os.environ,
                        SGLANG_FLASHINFER_AUTOTUNE_CACHE="1",
                    ),
                    patch.multiple(
                        autotune,
                        flashinfer_autotune_cache_path=lambda _: cache_path,
                        get_flashinfer_autotune_skip_ops=lambda _: skip_ops,
                        get_exec=lambda: SimpleNamespace(
                            moe=SimpleNamespace(moe_runner_backend="flashinfer_megamoe")
                        ),
                        get_eager_max_batch_size=lambda bs: bs,
                        max_speculative_num_draft_tokens=lambda: width,
                        max_prefill_buffer_tokens=lambda: max(chunk_size, 0),
                        get_schedule=lambda: SimpleNamespace(max_prefill_tokens=16384),
                    ),
                    patch.object(
                        autotune.torch.cuda,
                        "current_stream",
                        return_value=Mock(),
                    ),
                    patch.object(
                        autotune.torch,
                        "get_device_module",
                        return_value=SimpleNamespace(stream=lambda _: nullcontext()),
                    ),
                ):
                    general.reset_mock()
                    with autotune.flashinfer_autotune_context(
                        runner, run_lm_head=False
                    ):
                        general.assert_called_once_with(
                            True,
                            cache=None,
                            skip_ops=skip_ops,
                        )
                        context = mega_autotune._active_context
                        if prefill is None:
                            self.assertIs(context, previous)
                        else:
                            self.assertIsNot(context, previous)
                            runner.max_decode_logits_rows.assert_called_with()
                            runner.decode_num_tokens_per_req.assert_called_with(
                                num_draft_tokens=width
                            )
                            self.assertEqual(context.decode_num_tokens, decode)
                            self.assertEqual(context.prefill_num_tokens, prefill)
                    self.assertIs(mega_autotune._active_context, previous)

            # Cache-disabled startups in the same second must still write to
            # different files, even if they use the same device and rank.
            with (
                get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
                patch.dict(sys.modules, modules),
                patch.dict(os.environ, SGLANG_FLASHINFER_AUTOTUNE_CACHE="0"),
                patch.multiple(
                    autotune,
                    flashinfer_autotune_cache_path=lambda _: cache_path,
                    get_flashinfer_autotune_skip_ops=lambda _: set(),
                    get_exec=lambda: SimpleNamespace(
                        moe=SimpleNamespace(moe_runner_backend="triton")
                    ),
                ),
                patch.object(autotune.datetime, "datetime") as clock,
                patch.object(
                    autotune.torch.cuda, "current_stream", return_value=Mock()
                ),
                patch.object(
                    autotune.torch,
                    "get_device_module",
                    return_value=SimpleNamespace(stream=lambda _: nullcontext()),
                ),
            ):
                clock.now.return_value.strftime.return_value = "20260922_010000"
                paths = []
                for _ in range(2):
                    with autotune.flashinfer_autotune_context(
                        runner, run_lm_head=False
                    ):
                        paths.append(Path(general.call_args.kwargs["cache"]))
                self.assertNotEqual(*paths)
                for path in paths:
                    self.assertEqual(path.parent, cache_path.parent / "runs")
                    self.assertTrue(path.name.startswith("rank.20260922_010000."))


class TestAutotuneCacheDigest(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _write(self, name: str, configs) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(configs))
        return path

    def _digest(self, path: Path, env=ENV) -> str:
        return _autotune_cache_digest(path, env)

    def test_unusable_caches_read_as_empty(self):
        # Files yielding no loadable entries must digest alike, whichever way
        # they are unusable; a non-dict also has to not raise.
        self.assertEqual(self._digest(self.dir / "absent.json"), "")
        corrupt = self.dir / "corrupt.json"
        corrupt.write_text("{not json")
        self.assertEqual(self._digest(corrupt), "")
        self.assertEqual(self._digest(self._write("null.json", None)), "")
        self.assertEqual(self._digest(self._write("list.json", [])), "")

    def test_metadata_stamp_decides_whether_entries_load(self):
        # Equal tactics, different stamps: one rank loads them, the other
        # ignores the file.
        rank0 = self._write("rank0.json", {"_metadata": {"cublas": "12.8"}, "op": 7})
        rank1 = self._write("rank1.json", {"_metadata": {"cublas": "12.9"}, "op": 7})
        self.assertNotEqual(self._digest(rank0), self._digest(rank1))

    def test_environment_is_part_of_the_load_decision(self):
        # Same file, drifted environment on one rank: that rank loads nothing.
        cache = self._write("rank.json", {"_metadata": {"cublas": "12.8"}, "op": 7})
        self.assertNotEqual(
            self._digest(cache), self._digest(cache, {**ENV, "gpu": "NVIDIA B200"})
        )

    def test_key_order_does_not_matter(self):
        # Pins sort_keys: the same tactics must digest alike in any order.
        rank0 = self._write("rank0.json", {"a": 1, "b": 2})
        rank1 = self._write("rank1.json", {"b": 2, "a": 1})
        self.assertEqual(self._digest(rank0), self._digest(rank1))


class TestDropDivergedAutotuneCache(CustomTestCase):
    """The gate itself, over a real gloo group and real files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _run_gate(self, scenarios) -> list:
        """Gate each scenario's per-rank caches in one spawn; return what each
        rank kept, per scenario."""
        world_size = len(scenarios[0])
        port = find_available_port(23456)
        ctx = multiprocessing.get_context("spawn")
        procs, readers = [], []
        for rank in range(world_size):
            paths = []
            for index, per_rank_configs in enumerate(scenarios):
                path = self.dir / f"scenario{index}_rank{rank}.json"
                path.write_text(json.dumps(per_rank_configs[rank]))
                paths.append(str(path))
            reader, writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_gate_worker,
                args=(rank, world_size, port, paths, writer),
            )
            proc.start()
            writer.close()
            procs.append(proc)
            readers.append(reader)
        results = [r.recv() for r in readers]
        for proc in procs:
            proc.join(timeout=120)
        for status, value in results:
            self.assertEqual(status, "ok", msg=value)
        return [list(kept) for kept in zip(*(value for _, value in results))]

    def test_matching_caches_are_kept_and_diverged_ones_dropped_everywhere(self):
        # A rank that kept a diverged cache would skip profiles its peer still runs.
        meta = {"_metadata": {"cublas": "12.8"}}
        matching = {**meta, "op": 7}
        kept = self._run_gate(
            [[matching, matching], [{**meta, "op": 7}, {**meta, "op": 8}]]
        )
        self.assertEqual(kept, [[True, True], [False, False]])


class TestModelPrefillAutotune(CustomTestCase):
    """Model kernel warmup must cover prefill without a speculative dummy batch."""

    def setUp(self):
        self.hook = Mock(return_value=1)
        self.mr = SimpleNamespace(
            model=SimpleNamespace(autotune_prefill_kernels=self.hook),
            is_generation=True,
            is_draft_worker=False,
            dtype=torch.bfloat16,
        )
        self.runner = SimpleNamespace(model_runner=self.mr)
        # No dummy-buffer or attention APIs: this path must not build a
        # TARGET_VERIFY batch or mutate request/KV state.
        for target, kwargs in (
            ("max_prefill_buffer_tokens", {"return_value": 65536}),
            (
                "flashinfer_autotune_context",
                {"side_effect": lambda *a, **k: nullcontext()},
            ),
        ):
            p = patch.object(autotune, target, **kwargs)
            setattr(self, target, p.start())
            self.addCleanup(p.stop)
        p = patch.object(
            autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND, "get", return_value=False
        )
        p.start()
        self.addCleanup(p.stop)

    def test_declining_model_never_enters_the_autotune_context(self):
        self.mr.model.wants_prefill_autotune = lambda: False
        autotune.maybe_flashinfer_autotune_extend(self.runner, decode_num_tokens=384)
        self.hook.assert_not_called()
        self.flashinfer_autotune_context.assert_not_called()

    def test_extend_pass_is_opt_in(self):
        # A draft worker keeps its own warmup; a model without the hook opts out.
        for draft, has_hook in ((True, True), (False, False)):
            with self.subTest(draft=draft, has_hook=has_hook):
                self.mr.is_draft_worker = draft
                if not has_hook:
                    del self.mr.model.autotune_prefill_kernels
                autotune.maybe_flashinfer_autotune_extend(
                    self.runner, decode_num_tokens=384
                )
                self.hook.assert_not_called()
                self.flashinfer_autotune_context.assert_not_called()


class TestMoePrefillAutotune(CustomTestCase):
    """SGLANG_OPT_MOE_PREFILL_AUTOTUNE: one routed-experts call at the prefill ceiling."""

    def setUp(self):
        from collections import namedtuple
        from threading import Lock

        from sglang.srt.layers.moe import prefill_autotune

        self.module = prefill_autotune
        # AutoTuner keys and (tactic, profile) winners, as the tuning-file settle reads them.
        Key = namedtuple("Key", "custom_op nearest_profile extras")

        def key(op, bucket):
            return Key(op, ((bucket,),) * 5, _moe_extras())

        moe_op = "flashinfer::trtllm_fp4_block_scale_moe"
        self.winners = {key(moe_op, 128): ([8, 1], None)}
        self.tuner = SimpleNamespace(_winner_cache=lambda: self.winners, _lock=Lock())

        def route(hidden_states, *, precomputed_router):
            self.calls.append((hidden_states, precomputed_router))
            rows = hidden_states.shape[0]
            for bucket in (1024, 2048, rows):
                self.winners[key(moe_op, bucket)] = ([32, 1], None)
            self.winners[key("flashinfer::bmm_bf16", rows)] = (0, None)

        self.calls = []
        self.block = SimpleNamespace(
            tp_size=1,
            gate=SimpleNamespace(weight=torch.empty((512, 64), dtype=torch.bfloat16)),
            experts=SimpleNamespace(w13_input_scale_quant=torch.ones(1)),
            _forward_router_experts=route,
        )
        from flashinfer import autotuner

        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

        for target, attr, kwargs in (
            (autotuner.AutoTuner, "get", {"return_value": self.tuner}),
            (prefill_autotune, "_eligible_block", {"return_value": self.block}),
            (
                flashinfer_trtllm,
                "quantize_nvfp4_routed_input",
                {"side_effect": lambda x, scale: ("packet", x.shape)},
            ),
            (torch.cuda, "current_stream", {}),
            (torch.cuda, "empty_cache", {}),
        ):
            p = patch.object(target, attr, **kwargs)
            p.start()
            self.addCleanup(p.stop)

    def test_switch_and_backend_gate_the_hook(self):
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled  # noqa: F401
        from sglang.srt.layers.moe import utils as moe_utils

        backend = SimpleNamespace(is_flashinfer_trtllm=lambda: True)
        a2a = SimpleNamespace(is_none=lambda: True)
        for master, switch, expected in (
            (False, None, False),
            (True, "0", False),
            (True, None, True),
        ):
            with (
                self.subTest(master=master, switch=switch),
                autotune.envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(
                    master
                ),
                patch.object(moe_utils, "get_moe_runner_backend", return_value=backend),
                patch.object(moe_utils, "get_moe_a2a_backend", return_value=a2a),
                patch(
                    "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                    return_value=False,
                ),
            ):
                if switch is not None:
                    autotune.envs.SGLANG_OPT_MOE_PREFILL_AUTOTUNE.set(switch)
                try:
                    self.assertIs(
                        self.module.wants_moe_prefill_autotune(object()), expected
                    )
                finally:
                    autotune.envs.SGLANG_OPT_MOE_PREFILL_AUTOTUNE.clear()

    def test_one_routed_call_at_the_ceiling_counts_new_moe_buckets(self):
        for prequant in (True, False):
            with (
                self.subTest(prequant=prequant),
                autotune.envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                autotune.envs.SGLANG_OPT_MOE_NVFP4_PREQUANT_OVERLAP.override(prequant),
                autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.override(False),
            ):
                self.calls.clear()
                tuned = self.module.autotune_moe_prefill(
                    object(), 16384, dtype=torch.bfloat16
                )
                ((hidden, (logits, packet)),) = self.calls
                self.assertEqual(tuple(hidden.shape), (16384, 64))
                self.assertEqual(tuple(logits.shape), (16384, 512))
                self.assertEqual(packet, ("packet", hidden.shape) if prequant else None)
                # Only MoE entries count; the 128 bucket was already tuned.
                self.assertEqual(tuned, 3 if prequant else 1)

    def test_small_ceilings_and_the_extend_pass_return_zero(self):
        self.assertEqual(
            self.module.autotune_moe_prefill(object(), 16, dtype=torch.bfloat16), 0
        )
        self.assertEqual(self.calls, [])
        with autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.override(True):
            self.assertEqual(
                self.module.autotune_moe_prefill(object(), 4096, dtype=torch.bfloat16),
                0,
            )
        self.assertEqual(len(self.calls), 1)


def _moe_extras(hidden=2560, intermediate=640, experts=512, top_k=10, fused_shared=0):
    # MoERunner.get_cache_key_extras layout of FlashInfer 0.7.2 and 0.7.3.
    extras = [0] * 16
    extras[1], extras[2], extras[4] = top_k, experts, fused_shared
    extras[8], extras[10] = hidden, intermediate
    return tuple(extras)


class _PinKey(SimpleNamespace):
    __hash__ = object.__hash__


PIN_SHAPE = (2560, 640, 512, 10)  # hidden, intermediate, experts, top_k
PIN_METADATA = {
    "flashinfer_version": "0.7.3",
    "cuda_version": "13.4",
    "cublas_version": "13.8.1",
    "gpu": "NVIDIA Graphics Device",
    "sm": "10.7",
}
PIN_MOE_SECTION = dict(
    hidden_size=2560, intermediate_size=640, num_experts=512, top_k=10
)
# Synthetic tactics: none of these is a tuned choice.
TA, TB, TC, SPLIT = (64, 7), (64, 9), (256, 3), (4, 11)


def _pin_key(bucket, **shape):
    from sglang.srt.layers.moe.flashinfer_tactic_pins import MOE_OP

    return _PinKey(
        custom_op=MOE_OP, nearest_profile=[(bucket,)] * 5, extras=_moe_extras(**shape)
    )


def _pin_winners(*entries):
    """{key: (tactic, profile)} the way FlashInfer's AutoTuner stores winners."""
    return {_pin_key(bucket): (tactic, "profile") for bucket, tactic in entries}


def _pins(prefill=None, split_k=(), decode=None):
    from sglang.srt.layers.moe.flashinfer_tactic_pins import MoeTacticPins

    return MoeTacticPins(
        flashinfer_version="0.7.3",
        shape=PIN_SHAPE,
        split_k=frozenset(split_k),
        prefill=prefill,
        decode=decode,
    )


class _TuningFileCase(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.moe import flashinfer_tactic_pins as pins_module

        self.pins_module = pins_module
        for cached in (pins_module.read_tuning_file, pins_module._moe_pins_of):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)

    def _file(self, document):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(document, f)
        self.addCleanup(os.unlink, f.name)
        return f.name


class TestTacticPinFile(_TuningFileCase):
    """SGLANG_FLASHINFER_TACTIC_PIN_FILE: the file applies only to the stack it was written for."""

    def test_parses_tables_and_rejects_malformed_entries(self):
        parse = self.pins_module.parse_moe_tactic_pins
        section = dict(PIN_MOE_SECTION, split_k_tactics=[list(SPLIT)])
        section["prefill"] = {"1024": list(TA), "8192": -1}
        section["decode"] = {"4": list(TB), "512": -1}
        pins = parse({"_metadata": PIN_METADATA, "trtllm_fp4_block_scale_moe": section})
        self.assertEqual(pins.shape, PIN_SHAPE)
        self.assertEqual(pins.split_k, frozenset({SPLIT}))
        self.assertEqual(pins.prefill, {1024: TA, 8192: -1})
        self.assertEqual(pins.decode, {4: TB, 512: -1})
        for bad_section in (
            {"prefill": {"1024": [64]}},
            {"prefill": {"1024": [64, -1]}},
            {"prefill": {"0": -1}},
            {"prefill": {"x": -1}},
            {"prefill": [[1024, 64, 7]]},
            {"split_k_tactics": [-1]},
            {"hidden_size": None},
            {"decode": {"3": -1}},
            {"decode": {"1024": -1}},
        ):
            with self.subTest(bad_section=bad_section), self.assertRaises(ValueError):
                parse(
                    {
                        "_metadata": PIN_METADATA,
                        "trtllm_fp4_block_scale_moe": dict(
                            PIN_MOE_SECTION, **bad_section
                        ),
                    }
                )
        for field in PIN_METADATA:
            metadata = {k: v for k, v in PIN_METADATA.items() if k != field}
            with self.subTest(missing=field), self.assertRaises(ValueError):
                parse(
                    {
                        "_metadata": metadata,
                        "trtllm_fp4_block_scale_moe": PIN_MOE_SECTION,
                    }
                )

    def test_pins_apply_only_on_the_exact_stack_and_device(self):
        # Tactic indices move between FlashInfer builds, cuBLAS and CUDA versions,
        # devices and SMs, so every field must match exactly; the device name is
        # what tells a node reporting "NVIDIA Graphics Device" from a VR NVL72.
        path = self._file(
            {"_metadata": PIN_METADATA, "trtllm_fp4_block_scale_moe": PIN_MOE_SECTION}
        )
        cases = [({}, True)] + [
            ({field: value}, False)
            for field, value in (
                ("flashinfer_version", "0.7.3.post1"),
                ("cuda_version", "13.0"),
                ("cublas_version", "13.1.0"),
                ("gpu", "NVIDIA VR NVL72"),
                ("sm", "10.0"),
            )
        ]
        with autotune.envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path):
            for change, applies in cases:
                with (
                    self.subTest(change=change),
                    patch.object(
                        self.pins_module,
                        "_current_metadata",
                        return_value={**PIN_METADATA, **change},
                    ),
                ):
                    self.assertEqual(
                        self.pins_module.active_moe_tactic_pins() is not None, applies
                    )
        self.assertIsNone(self.pins_module.active_moe_tactic_pins())

    def test_only_flashinfer_builds_with_a_known_cache_key_layout(self):
        for version, applies in (("0.7.2", True), ("0.7.3", True), ("0.7.4", False)):
            metadata = {**PIN_METADATA, "flashinfer_version": version}
            path = self._file(
                {"_metadata": metadata, "trtllm_fp4_block_scale_moe": PIN_MOE_SECTION}
            )
            for cached in (
                self.pins_module.read_tuning_file,
                self.pins_module._moe_pins_of,
            ):
                cached.cache_clear()
            with (
                self.subTest(version=version),
                autotune.envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path),
                patch.object(
                    self.pins_module, "_current_metadata", return_value=metadata
                ),
            ):
                self.assertEqual(
                    self.pins_module.active_moe_tactic_pins() is not None, applies
                )

    def test_indeterminate_metadata_never_matches(self):
        for value in ("unknown", "None"):
            unknown = {**PIN_METADATA, "cublas_version": value}
            path = self._file(
                {"_metadata": unknown, "trtllm_fp4_block_scale_moe": PIN_MOE_SECTION}
            )
            for cached in (
                self.pins_module.read_tuning_file,
                self.pins_module._moe_pins_of,
            ):
                cached.cache_clear()
            with (
                self.subTest(value=value),
                autotune.envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path),
                patch.object(
                    self.pins_module, "_current_metadata", return_value=unknown
                ),
            ):
                self.assertIsNone(self.pins_module.active_moe_tactic_pins())

    def test_tuning_file_gets_its_own_autotune_cache(self):
        """With cache reuse on, a boot loads the saved tactics before any tuning, so
        settled tactics must not be found by a boot without the same file."""
        cache_dir = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(cache_dir, ignore_errors=True)
        )
        runner = SimpleNamespace(
            device="cuda",
            dtype=torch.bfloat16,
            is_draft_worker=False,
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )
        parallel = SimpleNamespace(
            tp_size=1,
            pp_size=1,
            attn_dp_size=1,
            moe_ep_size=1,
            tp_rank=0,
            pp_rank=0,
            dp_rank=0,
        )

        def cache_path():
            with (
                patch.dict(
                    sys.modules, {"flashinfer": SimpleNamespace(__version__="0.7.3")}
                ),
                patch.object(torch.cuda, "get_device_capability", return_value=(10, 7)),
                patch.multiple(
                    autotune,
                    get_model=lambda: SimpleNamespace(model_path="m", quantization="q"),
                    get_exec=lambda: SimpleNamespace(
                        moe=SimpleNamespace(moe_runner_backend="flashinfer_trtllm")
                    ),
                    get_parallel=lambda: parallel,
                    get_flashinfer_autotune_skip_ops=lambda mr: set(),
                ),
                autotune.envs.SGLANG_CACHE_DIR.override(cache_dir),
            ):
                return autotune.flashinfer_autotune_cache_path(runner)

        document = {
            "_metadata": PIN_METADATA,
            "trtllm_fp4_block_scale_moe": dict(PIN_MOE_SECTION, decode={"8": list(TA)}),
        }
        stock = cache_path()
        path = self._file(document)
        with autotune.envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(path):
            pinned = cache_path()
            document["trtllm_fp4_block_scale_moe"]["decode"] = {"8": list(TB)}
            with open(path, "w") as f:
                json.dump(document, f)
            changed = cache_path()
        self.assertEqual(len({stock, pinned, changed}), 3)
        self.assertEqual(cache_path(), stock)


class TestSettleTactics(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.moe import flashinfer_tactic_pins as pins_module

        self.pins_module = pins_module

    def _settle_prefill(self, winners, pins, valid):
        before = dict(winners)
        rows = self.pins_module.settle_prefill_tactics(
            winners, list(winners), pins=pins, valid=valid
        )
        changed = {k for k in winners if winners[k] != before[k]}
        return {row[0]: row[1:] for row in rows}, changed

    def test_prefill_without_pins_the_tuner_picks_stand(self):
        winners = _pin_winners((768, list(SPLIT)), (1024, list(TB)))
        rows, changed = self._settle_prefill(winners, None, {})
        self.assertEqual(changed, set())
        self.assertEqual(rows[768], (SPLIT, SPLIT, "tuner"))

    def test_prefill_table_falls_back_to_the_default_tactic_not_the_tuner(self):
        winners = _pin_winners(
            (768, list(SPLIT)),
            (1024, list(TC)),
            (2048, list(TC)),
            (4096, list(TC)),
            (8192, list(TC)),
        )
        pins = _pins(
            prefill={768: TA, 1024: -1, 2048: (64, 999), 4096: SPLIT}, split_k={SPLIT}
        )
        rows, _ = self._settle_prefill(
            winners, pins, {768: {TA}, 2048: {TB}, 4096: {SPLIT}}
        )
        self.assertEqual(rows[768][0::2], (TA, "pin"))
        self.assertEqual(rows[1024][0::2], (-1, "pin_default"))
        self.assertEqual(rows[2048][0::2], (-1, "pin_invalid"))
        self.assertEqual(rows[4096][0::2], (-1, "pin_split_k"))
        self.assertEqual(rows[8192][0::2], (-1, "pin_absent"))
        stored = {k.nearest_profile[4][0]: v[0] for k, v in winners.items()}
        self.assertEqual(
            stored, {768: list(TA), 1024: -1, 2048: -1, 4096: -1, 8192: -1}
        )

    def test_prefill_split_k_is_excluded_from_tuner_picks_too(self):
        winners = _pin_winners((768, list(SPLIT)), (1024, list(TB)))
        rows, _ = self._settle_prefill(winners, _pins(split_k={SPLIT}), {})
        self.assertEqual(rows[768][0::2], (-1, "tuner_split_k"))
        self.assertEqual(rows[1024][0::2], (TB, "tuner"))

    def test_valid_tactics_read_per_bucket_on_both_flashinfer_key_layouts(self):
        def query_072(num_tokens, experts=512):
            return (0, 0, 0, 10, 2560, 2560, 640, experts, 0, 0, 0, 0, num_tokens, 0)

        def query_073(num_tokens, experts=512):
            # 0.7.3 adds the global expert count after the local one.
            return (
                0,
                0,
                0,
                10,
                2560,
                2560,
                640,
                experts,
                512,
                0,
                0,
                0,
                0,
                num_tokens,
                0,
            )

        for query in (query_072, query_073):
            with self.subTest(layout=query.__name__):
                valid = self.pins_module.valid_tactics_by_bucket(
                    {
                        ("runner", query(768)): [list(TA), [-1, -1]],
                        ("runner", query(768, experts=256)): [list(TB)],
                        ("runner", query(1024)): [TC],
                        ("runner", (0,) * 13): [TC],
                    },
                    PIN_SHAPE,
                )
                self.assertEqual(valid, {768: {TA, -1}, 1024: {TC}})

    def test_refused_decode_pins_keep_the_tuner_pick(self):
        # Decode buckets are tuned at startup, so unlike prefill the stock choice a
        # refused pin falls back to is the tuner's pick, not the default tactic.
        winners = _pin_winners(
            (4, list(TC)), (8, list(TB)), (16, list(TB)), (32, list(TC)), (64, list(TB))
        )
        other_shape = _PinKey(**vars(_pin_key(4, intermediate=1024)))
        winners[other_shape] = (list(TC), "profile")
        pins = _pins(
            decode={4: TA, 8: SPLIT, 16: (64, 999), 32: -1, 128: TA}, split_k={SPLIT}
        )
        rows = self.pins_module.settle_decode_tactics(
            winners, pins=pins, valid={4: {TA}, 8: {SPLIT}, 16: {TB}}
        )
        self.assertEqual(
            sorted((row[0], row[3]) for row in rows),
            [
                (4, "pin"),
                (4, "pin_shape_skip"),
                (8, "pin_split_k_refused"),
                (16, "pin_invalid"),
                (32, "pin_default"),
            ],
        )
        stored = {
            (k.nearest_profile[4][0], k is other_shape): v[0]
            for k, v in winners.items()
        }
        self.assertEqual(
            stored,
            {
                (4, False): list(TA),
                (4, True): list(TC),
                (8, False): list(TB),
                (16, False): list(TB),
                (32, False): -1,
                (64, False): list(TB),
            },
        )

    def test_decode_pins_are_settled_before_the_cache_is_saved(self):
        # With cache reuse the next boot serves the saved tactics without tuning,
        # so the pins must be in place when the autotune context saves on exit.
        events = []

        class _Context:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *exc):
                events.append("save")

        with (
            patch.object(
                autotune, "flashinfer_autotune_context", lambda *a, **k: _Context()
            ),
            patch.object(
                self.pins_module,
                "pin_decode_moe_tactics",
                side_effect=lambda: events.append("pin"),
            ),
        ):
            with autotune.envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(None):
                autotune.run_flashinfer_autotune_forward(
                    None, lambda: events.append("tune"), run_lm_head=False
                )
            self.assertEqual(events, ["enter", "tune", "save"])
            events.clear()
            with autotune.envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.override(
                "/tuning.json"
            ):
                autotune.run_flashinfer_autotune_forward(
                    None, lambda: events.append("tune"), run_lm_head=False
                )
            self.assertEqual(events, ["enter", "tune", "pin", "save"])


@unittest.skipUnless(torch.cuda.is_available(), "FlashInfer requires CUDA")
class TestAutotuneCachePhases(CustomTestCase):
    """Loaded target tactics survive draft warmup, unless cache reuse is off."""

    def test_target_and_draft_cache_reuse(self):
        from flashinfer.autotuner import AutoTuner, _collect_metadata

        tuner = AutoTuner.get()
        tuner.clear_cache()
        self.addCleanup(tuner.clear_cache)
        runner = SimpleNamespace(device="cuda", forward_stream=torch.cuda.Stream())
        with tempfile.TemporaryDirectory() as directory:
            target, draft = (
                Path(directory) / name for name in ("target.json", "draft.json")
            )
            for path, key, tactic in (
                (target, "target_prefill", 7),
                (draft, "draft_decode", 3),
            ):
                path.write_text(
                    json.dumps(
                        {"_metadata": _collect_metadata(), key: ["TestRunner", tactic]}
                    )
                )
            with (
                patch.object(
                    autotune,
                    "flashinfer_autotune_cache_path",
                    side_effect=[target, draft, draft],
                ),
                patch.object(
                    autotune, "get_flashinfer_autotune_skip_ops", return_value=set()
                ),
                patch.object(
                    autotune,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(moe_runner_backend="flashinfer_cutedsl")
                    ),
                ),
                autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
                get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
            ):
                with autotune.flashinfer_autotune_context(runner, run_lm_head=False):
                    self.assertEqual(
                        tuner._file_configs["target_prefill"], ("TestRunner", 7)
                    )
                # No profiling: this models a restart that loads tactics from disk.
                self.assertFalse(tuner.profiling_cache)
                with autotune.flashinfer_autotune_context(runner, run_lm_head=False):
                    pass
                saved = json.loads(draft.read_text())
                self.assertEqual(saved["target_prefill"], ["TestRunner", 7])
                self.assertEqual(saved["draft_decode"], ["TestRunner", 3])
                with (
                    autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False),
                    autotune.flashinfer_autotune_context(runner, run_lm_head=False),
                ):
                    self.assertNotIn("target_prefill", tuner._file_configs)
                    self.assertNotIn("draft_decode", tuner._file_configs)


if __name__ == "__main__":
    unittest.main()
