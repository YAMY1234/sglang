"""Page-wise HiCache L2 loads (SGLANG_HICACHE_PAGEWISE_LOAD): the run and group
helpers, and the order in which L2TransferEngine issues a page-wise load."""

import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import l2_transfer as transfer_module
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.pool_host.pagewise_l2 import (
    contiguous_runs,
    layer_runs,
    split_layer_groups,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self):
        pass

    def wait(self, stream):
        pass


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    def Stream():
        return object()

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


class _FakeHostPool:
    """Records the loads it is asked for into a shared call log."""

    def __init__(self, name, calls, *, pagewise, layer_num=4):
        self.name = name
        self.calls = calls
        self.pagewise = pagewise
        self.layer_num = layer_num

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, device_indices

    def pagewise_load_ok(self, io_backend):
        return self.pagewise and io_backend == "direct"

    def load_to_device_layers_pagewise(
        self, device_pool, host_indices, device_indices, layer_ids, *, is_draft=False
    ):
        self.calls.append((self.name, "pages", list(layer_ids)))

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        self.calls.append((self.name, "layer", layer_id))


class TestPagewiseHelpers(CustomTestCase):
    def test_layer_runs_follow_host_layer_order(self):
        self.assertEqual(
            layer_runs([5, 0, 1, 3, 2, 7]),
            [(0, 4, [1, 2, 4, 3]), (5, 1, [0]), (7, 1, [5])],
        )
        self.assertEqual(layer_runs([2]), [(2, 1, [0])])
        self.assertEqual(layer_runs([]), [])

    def test_split_layer_groups(self):
        self.assertEqual(
            split_layer_groups(list(range(10)), 4),
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
        )
        self.assertEqual(split_layer_groups([0, 1, 2, 3], 1), [[0, 1, 2, 3]])
        self.assertEqual(split_layer_groups([0, 1], 8), [[0], [1]])
        self.assertEqual(split_layer_groups([0, 1], 0), [[0, 1]])
        self.assertEqual(split_layer_groups([], 4), [])

    def test_contiguous_runs(self):
        self.assertEqual(
            contiguous_runs(torch.tensor([3, 4, 5, 9, 10, 2])), ([0, 3, 5], [3, 2, 1])
        )
        self.assertEqual(contiguous_runs(torch.tensor([7])), ([0], [1]))
        self.assertEqual(contiguous_runs(torch.tensor([4, 3])), ([0, 1], [1, 1]))


class TestPagewiseLoadOrder(CustomTestCase):
    def setUp(self):
        transfer_module._timing_events_supported.cache_clear()
        self.addCleanup(transfer_module._timing_events_supported.cache_clear)

    def _load(self, *, enabled, groups=2):
        calls = []
        indices = torch.arange(4)
        transfers = [
            L2Transfer(
                host_pool=_FakeHostPool("kv", calls, pagewise=True),
                device_pool=None,
                host_indices=indices,
                device_indices=indices,
            ),
            L2Transfer(
                host_pool=_FakeHostPool("index", calls, pagewise=False),
                device_pool=None,
                host_indices=indices,
                device_indices=indices,
                layer_mapper={0: 0, 2: 1}.get,
            ),
        ]
        with (
            mock.patch.object(transfer_module, "device_module", _FakeDeviceModule),
            envs.SGLANG_HICACHE_PAGEWISE_LOAD.override(enabled),
            envs.SGLANG_HICACHE_PAGEWISE_LOAD_GROUPS.override(groups),
        ):
            L2TransferEngine("direct").submit_host_to_device(
                transfers,
                transfer_layer_id_max=4,
                on_layer_done=lambda layer_id: calls.append(("done", layer_id)),
            )
        return calls

    def test_groups_are_loaded_before_their_layers_are_done(self):
        self.assertEqual(
            self._load(enabled=True),
            [
                ("kv", "pages", [0, 1]),
                ("index", "layer", 0),
                ("done", 0),
                ("done", 1),
                ("kv", "pages", [2, 3]),
                ("index", "layer", 1),
                ("done", 2),
                ("done", 3),
            ],
        )

    def test_one_group_loads_every_layer_at_once(self):
        self.assertEqual(
            self._load(enabled=True, groups=1),
            [
                ("kv", "pages", [0, 1, 2, 3]),
                ("index", "layer", 0),
                ("index", "layer", 1),
                *(("done", layer_id) for layer_id in range(4)),
            ],
        )

    def test_disabled_keeps_the_per_layer_load(self):
        self.assertEqual(
            self._load(enabled=False),
            [
                ("kv", "layer", 0),
                ("index", "layer", 0),
                ("done", 0),
                ("kv", "layer", 1),
                ("done", 1),
                ("kv", "layer", 2),
                ("index", "layer", 1),
                ("done", 2),
                ("kv", "layer", 3),
                ("done", 3),
            ],
        )


if __name__ == "__main__":
    unittest.main()
