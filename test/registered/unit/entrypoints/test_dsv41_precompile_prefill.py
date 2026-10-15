"""SGLANG_OPT_DSV41_PRECOMPILE_PREFILL: the startup requests cover every shape class
the serving path specializes on, flush the cache, and keep the late-compile watcher
disarmed; sgl-kernel JIT loads after serving started are reported like Triton's."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints import warmup
from sglang.srt.utils import triton_load_watch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(self, flush_failures=0):
        self.requests = []
        self.groups = []
        self.flushes = 0
        self._flush_failures = flush_failures

    async def generate_request(self, obj, request):
        if isinstance(obj.rid, list):
            total = sum(len(ids) for ids in obj.input_ids)
            self.groups.append((len(obj.rid), total % 16 == 0))
            for rid, ids in zip(obj.rid, obj.input_ids):
                self.requests.append(
                    SimpleNamespace(
                        rid=rid,
                        input_ids=ids,
                        sampling_params=obj.sampling_params,
                        stream=False,
                    )
                )
        else:
            self.requests.append(obj)
        for _ in range(3 if obj.stream else 1):
            await asyncio.sleep(0)
            yield {}

    async def flush_cache(self):
        self.flushes += 1
        return SimpleNamespace(success=self.flushes > self._flush_failures)


def _run(tm, mixed_chunk=True):
    asyncio.run(warmup.dsv41_precompile("null", tm, mixed_chunk=mixed_chunk))


def test_requests_cover_the_shape_classes():
    tm = _FakeTokenizerManager()
    _run(tm)
    rids = [r.rid for r in tm.requests]
    assert len(set(rids)) == len(rids)
    assert all(r.startswith(triton_load_watch.PRECOMPILE_RID_PREFIX) for r in rids)

    lengths = [len(r.input_ids) for r in tm.requests]
    assert 1 in lengths
    assert any(n % 16 == 0 and 16 <= n < 4096 for n in lengths)
    assert any(n % 16 and 1 < n < 16 for n in lengths)
    # The mHC prefill path: >= 4,096 rows, divisible by 16 and not.
    assert 4096 in lengths and any(n > 4096 and n % 16 for n in lengths)
    assert max(lengths) > 8192  # chunked prefill with a prefix

    # Prefix hits: a resent prompt extended by 0 (one-token extend), small and >= 4K.
    by_prefix = {}
    for r in tm.requests:
        for s in tm.requests:
            if s is not r and len(s.input_ids) > len(r.input_ids):
                if s.input_ids[: len(r.input_ids)] == r.input_ids:
                    by_prefix.setdefault(id(r), []).append(
                        len(s.input_ids) - len(r.input_ids)
                    )
    extends = sorted(e for v in by_prefix.values() for e in v)
    assert any(e < 16 for e in extends) and any(e >= 4096 for e in extends)
    assert any(
        len(s.input_ids) == len(r.input_ids) and s.input_ids == r.input_ids
        for i, r in enumerate(tm.requests)
        for s in tm.requests[i + 1 :]
    )

    # Greedy and server-default sampling both run; decodes overlap prefills.
    temps = {r.sampling_params.get("temperature") for r in tm.requests}
    assert temps == {None, 0.0}
    streams = [r for r in tm.requests if r.stream]
    assert len(streams) == 3
    assert all(r.sampling_params["max_new_tokens"] == 256 for r in streams)
    # Kernels keyed on next_power_of_2(bs) also specialize the token count on
    # divisibility by 16: every size up to --max-running-requests (16), both classes.
    assert sorted(tm.groups) == [(n, d) for n in range(1, 17) for d in (False, True)]
    assert tm.flushes == 1


def test_mixed_phase_only_with_mixed_chunk():
    tm = _FakeTokenizerManager()
    _run(tm, mixed_chunk=False)
    assert not any(r.stream for r in tm.requests) and tm.flushes == 1


def test_flush_is_retried_then_required(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep(asyncio.sleep))
    tm = _FakeTokenizerManager(flush_failures=2)
    _run(tm)
    assert tm.flushes == 3

    tm = _FakeTokenizerManager(flush_failures=10**6)
    with pytest.raises(RuntimeError, match="flush"):
        _run(tm)


def _no_sleep(real_sleep):
    async def sleep(delay, *args, **kwargs):
        return await real_sleep(0)

    return sleep


def test_skipped_for_disaggregation():
    tm = _FakeTokenizerManager()
    asyncio.run(warmup.dsv41_precompile("prefill", tm))
    assert tm.requests == [] and tm.flushes == 0


def test_jit_load_reported_only_after_serving(monkeypatch, caplog):
    monkeypatch.setattr(triton_load_watch, "_serving_started", False)
    triton_load_watch.arm_on_first_request()
    with caplog.at_level(logging.WARNING, logger=triton_load_watch.__name__):
        triton_load_watch.on_jit_module_load("mod_a", built=True, secs=4.6)
        triton_load_watch.on_request(triton_load_watch.PRECOMPILE_RID_PREFIX + "7")
        triton_load_watch.on_jit_module_load("mod_a", built=True, secs=4.6)
        assert not caplog.records
        triton_load_watch.on_request("HEALTH_CHECK_1")
        triton_load_watch.on_jit_module_load("mod_a", built=True, secs=4.6)
        triton_load_watch.on_jit_module_load("mod_b", built=False, secs=0.01)
    msgs = [r.getMessage() for r in caplog.records]
    assert "'mod_a' was built (nvcc)" in msgs[0] and "4.60 s" in msgs[0]
    assert "'mod_b' was loaded from the JIT cache" in msgs[1]

    monkeypatch.setenv("SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY", "1")
    with pytest.raises(RuntimeError, match="mod_c"):
        triton_load_watch.on_jit_module_load("mod_c", built=False, secs=0.0)
