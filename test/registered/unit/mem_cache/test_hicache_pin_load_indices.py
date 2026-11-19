"""SGLANG_HICACHE_PIN_LOAD_INDICES: HiCache load indices uploaded through pinned buffers give the same indices on the device
as the plain .to(device); a buffer is reused only after its copy's event completed (polled, never waited on), the pool grows
while every buffer is in flight, and writes keep the plain copy (a fenced write does not wait on the uploading stream)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import cache_controller as cc
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _Event:
    def __init__(self):
        self.done = False

    def record(self, *a):
        pass

    def query(self):
        return self.done

    def synchronize(self):
        raise AssertionError("the upload path must never wait on an event")


class TestPinLoadIndices(CustomTestCase):
    def setUp(self):
        self.cuda = torch.cuda.is_available()
        self.events, self.allocs = [], []
        real_empty = torch.empty  # cc.torch is torch: the stand-ins must call the original

        def empty(*a, pin_memory=False, **k):
            self.allocs.append(a)
            return real_empty(*a, pin_memory=pin_memory and self.cuda, **k)

        def event():
            self.events.append(_Event())
            return self.events[-1]

        self.patches = [patch.object(cc.torch, "empty", empty)]
        if not self.cuda:
            self.patches.append(patch.object(cc.device_module, "Event", event))
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _controller(self):
        ctl = cc.HiCacheController.__new__(cc.HiCacheController)
        ctl.io_backend, ctl.device = "kernel", torch.device("cuda" if self.cuda else "cpu")
        return ctl

    def _move(self, ctl, host, flag, load=True):
        ctl._index_pin_load = load
        with envs.SGLANG_HICACHE_PIN_LOAD_INDICES.override(flag):
            return ctl.move_indices(host, host.clone())[0]

    def test_three_uploads_per_load_never_wait_and_match_the_plain_copy(self):
        ctl = self._controller()
        for load in range(3):
            # one load: the KV anchor plus two non-shared pools, all before any copy completes
            for n in (5000, 70000, 3):
                host = torch.randint(0, 1 << 30, (n,), dtype=torch.int64)
                on, off = self._move(ctl, host, True), self._move(ctl, host, False)
                if self.cuda:
                    torch.cuda.synchronize()
                self.assertTrue(torch.equal(on.cpu(), off.cpu()), (load, n))
                self.assertTrue(torch.equal(on.cpu(), host), (load, n))
            if not self.cuda:
                for e in self.events:
                    e.done = True  # the load's copies finished: their buffers return at the next call
        if not self.cuda:
            # the first load grew the pool to 3 buffers (70000 > the 64k initial size); later loads reused them
            self.assertEqual(len(self.allocs), 3)
            self.assertEqual(len(ctl._index_pin_pool["free"]) + len(ctl._index_pin_pool["busy"]), 3)

    def test_writes_and_flag_off_keep_the_plain_copy(self):
        ctl = self._controller()
        with patch.object(ctl, "_pinned_indices_to_device") as staged:
            host = torch.arange(10)
            self._move(ctl, host, False)
            self._move(ctl, host, True, load=False)  # a write path call
            staged.assert_not_called()
            self._move(ctl, host, True)
            staged.assert_called_once()

    def test_start_loading_marks_only_its_own_index_move(self):
        ctl = self._controller()
        seen = {}

        class Stop(Exception):
            pass

        def move(op, key):
            seen[key] = getattr(ctl, "_index_pin_load", False)
            raise Stop

        ctl.load_queue, ctl.write_queue = [object()], [object()]
        ctl.layer_done_counter = SimpleNamespace(update_producer=lambda: 0)
        with patch.object(cc.CacheOperation, "merge_ops", staticmethod(lambda q: object())):
            ctl._move_op_indices = lambda op: move(op, "load")
            with self.assertRaises(Stop):
                ctl.start_loading()
            self.assertFalse(ctl._index_pin_load)
            ctl._move_write_operation = lambda op: move(op, "write")
            with self.assertRaises(Stop):
                ctl.start_writing()
        self.assertEqual(seen, {"load": True, "write": False})


if __name__ == "__main__":
    unittest.main()
