"""Unit tests for MooncakeStoreConfig local_hostname and protocol resolution.

Regression for sgl-project/sglang#23457: runtime attach must resolve
per-node local_hostname from process environment instead of a shared default.
"""

import ctypes
import ctypes.util
import json
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_store
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStoreConfig,
    _resolve_device_name,
)
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestMooncakeStoreConfigLocalHostname(CustomTestCase):
    _BASE_EXTRA_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def _load_from_extra_config(self, extra_config=None):
        config = {**self._BASE_EXTRA_CONFIG, **(extra_config or {})}
        return MooncakeStoreConfig.load_from_extra_config(config)

    def test_load_from_extra_config_uses_mooncake_env_when_omitted(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.2"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "10.0.0.2")

    def test_load_from_extra_config_prefers_env_over_broadcast_override(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.4"):
            cfg = self._load_from_extra_config({"local_hostname": "10.0.0.1"})
        self.assertEqual(cfg.local_hostname, "10.0.0.4")


class TestMooncakeStoreConfigProtocol(CustomTestCase):
    _BASE_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def _load_from_extra_config(self, extra_config=None):
        config = {**self._BASE_CONFIG, **(extra_config or {})}
        return MooncakeStoreConfig.load_from_extra_config(config)

    def test_load_from_extra_config_uses_protocol_env_when_omitted(self):
        with temp_set_env(MOONCAKE_PROTOCOL="efa"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.protocol, "efa")

    def test_load_from_extra_config_prefers_explicit_protocol_over_env(self):
        with temp_set_env(MOONCAKE_PROTOCOL="efa"):
            cfg = self._load_from_extra_config({"protocol": "rdma"})
        self.assertEqual(cfg.protocol, "rdma")


class TestMooncakeDeviceNameResolution(CustomTestCase):
    """One PP stage per GPU leaves every rank at TP rank 0, so a JSON device map
    can name each NUMA node's HCAs instead (both of its rails)."""

    _BY_NUMA = json.dumps(
        {"numa0": "roce_vf0_r1,roce_vf0_r2", "numa1": "roce_vf0_r3,roce_vf0_r4"}
    )

    def _resolve(self, device_name, tp_rank, numa_node):
        with patch.object(mooncake_store, "_local_gpu_numa_node", return_value=numa_node):
            return _resolve_device_name(device_name, tp_rank)

    def test_numa_keys_pick_the_gpu_numa_node_hcas(self):
        self.assertEqual(self._resolve(self._BY_NUMA, 0, 1), "roce_vf0_r3,roce_vf0_r4")
        self.assertEqual(self._resolve(self._BY_NUMA, 0, 0), "roce_vf0_r1,roce_vf0_r2")

    def test_numa_env_wins_over_a_plain_device_list(self):
        by_numa = "numa0:roce_vf0_r1,roce_vf0_r2;numa1:roce_vf0_r3,roce_vf0_r4"
        with envs.SGLANG_MOONCAKE_DEVICE_BY_NUMA.override(by_numa):
            self.assertEqual(
                self._resolve("roce_vf0_r1,roce_vf0_r3", 0, 1), "roce_vf0_r3,roce_vf0_r4"
            )
            # A node the map does not name keeps MOONCAKE_DEVICE.
            self.assertEqual(self._resolve("mlx5_0", 0, 2), "mlx5_0")

    def test_tp_rank_keys_and_plain_names_are_unchanged(self):
        by_rank = json.dumps({"0": "mlx5_0", "1": "mlx5_1"})
        self.assertEqual(self._resolve(by_rank, 1, 0), "mlx5_1")
        self.assertEqual(self._resolve(by_rank, None, 0), "")
        self.assertEqual(self._resolve("mlx5_0,mlx5_1", 0, 1), "mlx5_0,mlx5_1")
        self.assertEqual(self._resolve("{not json", 0, 1), "")


def _mempolicy():
    numa = ctypes.CDLL("libnuma.so.1")
    mode, mask = ctypes.c_int(), (ctypes.c_ulong * 16)()
    numa.get_mempolicy(ctypes.byref(mode), mask, ctypes.c_ulong(1025), None, 0)
    return mode.value, list(mask)


@unittest.skipUnless(ctypes.util.find_library("numa"), "libnuma not installed")
class TestBoundNumaNode(CustomTestCase):
    def test_segment_allocation_binds_to_the_node_then_restores_the_policy(self):
        before = _mempolicy()
        with mooncake_store._bound_numa_node(0):
            mode, mask = _mempolicy()
            self.assertEqual((mode, mask[0]), (2, 1))  # MPOL_BIND, node 0
        self.assertEqual(_mempolicy(), before)

    def test_a_denied_binding_falls_back_to_preferring_the_node(self):
        real = ctypes.CDLL("libnuma.so.1", use_errno=True)

        class DenyBind:
            def __getattr__(self, name):
                return getattr(real, name)

            def set_mempolicy(self, mode, mask, maxnode):
                return -1 if mode == 2 else real.set_mempolicy(mode, mask, maxnode)

        before = _mempolicy()
        with patch.object(mooncake_store.ctypes, "CDLL", return_value=DenyBind()):
            with self.assertLogs(mooncake_store.logger, "WARNING"):
                with mooncake_store._bound_numa_node(0):
                    mode, mask = _mempolicy()
                    self.assertEqual((mode, mask[0]), (1, 1))  # MPOL_PREFERRED
        self.assertEqual(_mempolicy(), before)

    def test_policy_is_restored_when_setup_raises(self):
        before = _mempolicy()
        with self.assertRaises(RuntimeError):
            with mooncake_store._bound_numa_node(0):
                raise RuntimeError("setup failed")
        self.assertEqual(_mempolicy(), before)


if __name__ == "__main__":
    unittest.main()
