"""SGLANG_HICACHE_PIN_LOAD_INDICES: HiCache host indices uploaded through reused pinned buffers give the same indices on the
device as the plain .to(device); a buffer is refilled only after the copy that last read it finished."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import cache_controller as cc
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _Event:
    def __init__(self):
        self.recorded = self.synced = 0

    def record(self, *a):
        self.recorded += 1

    def synchronize(self):
        self.synced += 1


class TestPinLoadIndices(CustomTestCase):
    def _controller(self, device):
        ctl = cc.HiCacheController.__new__(cc.HiCacheController)
        ctl.io_backend, ctl.device = "kernel", device
        return ctl

    def _move(self, ctl, host, flag):
        with envs.SGLANG_HICACHE_PIN_LOAD_INDICES.override(flag):
            return ctl.move_indices(host, host.clone())[0]

    def test_indices_identical_on_device_with_and_without_the_fix(self):
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        real_empty = torch.empty  # cc.torch is torch: the CPU stand-in for pinned memory must call the original
        empty = real_empty if cuda else (lambda *a, pin_memory=False, **k: real_empty(*a, **k))
        events = []
        def event():
            events.append(_Event())
            return events[-1]
        with patch.object(cc.torch, "empty", empty), patch.object(cc.device_module, "Event", event if not cuda else cc.device_module.Event):
            ctl = self._controller(device)
            sizes = [1, 64, 5000, 70000, 3, 70001, 131072, 7]  # growth past the 64k initial buffer, then reuse
            for i, n in enumerate(sizes):
                host = torch.randint(0, 1 << 30, (n,), dtype=torch.int64)
                off, on = self._move(ctl, host, False), self._move(ctl, host, True)
                if cuda:
                    torch.cuda.synchronize()
                self.assertEqual(on.device.type, device.type)
                self.assertTrue(torch.equal(on.cpu(), off.cpu()), (i, n))
                self.assertTrue(torch.equal(on.cpu(), host), (i, n))
        if not cuda:
            # two buffers alternate; each refill waited for the copy that last read that buffer
            self.assertEqual(len(events), len(sizes))
            self.assertEqual([e.synced for e in events], [1] * (len(sizes) - 2) + [0, 0])

    def test_pinned_input_and_flag_off_keep_the_plain_copy(self):
        ctl = self._controller(torch.device("cpu"))
        with patch.object(ctl, "_pinned_indices_to_device") as staged:
            host = torch.arange(10)
            self._move(ctl, host, False)
            staged.assert_not_called()
            self._move(ctl, host, True)
            staged.assert_called_once()


if __name__ == "__main__":
    unittest.main()
