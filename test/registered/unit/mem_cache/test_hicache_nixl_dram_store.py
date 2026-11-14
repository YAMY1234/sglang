"""Unit tests for the nixl_dram HiCache L3 backend's pool registration -- no service, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, PoolName
from sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_store import HiCacheNixlDram
from sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_client import NixlTransport
from sglang.test.test_utils import CustomTestCase


def _store(**extra):
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=3,
        pp_size=8,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="org/model",
        extra_config={
            "nixl_dram_registry_dir": "/unused",
            "nixl_dram_expected_services": 1,
            "nixl_dram_reg_chunk_mib": 2048,
            **extra,
        },
    )
    client = mock.Mock()
    client.register_buffer.return_value = 0
    return HiCacheNixlDram(config, client=client)


class TestNixlDramPoolRegistration(CustomTestCase):
    def test_mamba_pool_with_a_list_kv_buffer_registers_every_state_buffer(self):
        # MambaPoolHost.kv_buffer is the list of its state buffers (temporal, conv...).
        temporal = torch.zeros((8, 3, 1, 4, 4), dtype=torch.float32)
        conv = torch.zeros((8, 3, 1, 2, 4), dtype=torch.float32)
        pool = SimpleNamespace(
            page_size=1,
            kv_buffer=[temporal, conv],
            get_hybrid_pool_buffer=lambda: [temporal, conv],
        )
        store = _store()
        store.register_mem_host_pool_v2(pool, PoolName.MAMBA)

        units = store.store.set_page_units.call_args.args[0]
        self.assertEqual(
            units,
            {
                temporal.data_ptr(): temporal.stride(0) * temporal.element_size(),
                conv.data_ptr(): conv.stride(0) * conv.element_size(),
            },
        )
        registered = [c.args for c in store.store.register_buffer.call_args_list]
        self.assertEqual(
            registered,
            [
                (temporal.data_ptr(), temporal.numel() * temporal.element_size()),
                (conv.data_ptr(), conv.numel() * conv.element_size()),
            ],
        )
        self.assertIs(store.registered_pools[PoolName.MAMBA], pool)

    def test_pp_keys_carry_the_stage(self):
        store = _store()
        self.assertEqual(store.mla_suffix, "3")
        self.assertEqual(store.mha_suffix, "0_3")

    def test_write_ledger_context_is_scoped_to_the_physical_write(self):
        store = _store()
        context = {"reason": "state_tombstone", "op_id": 42, "pp_rank": 0, "tp_rank": 0}
        with store.write_ledger_context(context):
            store._put_batch_zero_copy_impl(["state"], [100], [256])
        store.store.batch_put_from.assert_called_once_with(
            ["state"], [100], [256], ledger_context=context
        )
        store.store.batch_put_from.reset_mock()
        store._put_batch_zero_copy_impl(["other"], [200], [64])
        store.store.batch_put_from.assert_called_once_with(["other"], [200], [64])

    def test_v6_timeout_exits_without_releasing_active_dma_handle(self):
        class Exited(BaseException):
            pass

        names = (
            "SGLANG_HICACHE_L3_WRITE_STATE_ON_TOMBSTONE",
            "SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV",
        )
        for enabled in names:
            transport = NixlTransport.__new__(NixlTransport)
            transport.agent = mock.Mock()
            flags = dict.fromkeys(names, "0")
            flags[enabled] = "1"
            with (
                self.subTest(flag=enabled),
                mock.patch.dict(os.environ, flags),
                mock.patch(
                    "sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_client.time.monotonic",
                    return_value=2,
                ),
                mock.patch(
                    "sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_client.os._exit",
                    side_effect=Exited,
                ) as exit_process,
            ):
                with self.assertRaises(Exited):
                    transport.wait(["active-handle", "PROC"], deadline=1)
                exit_process.assert_called_once_with(86)
                transport.agent.release_xfer_handle.assert_not_called()

    def test_v6_terminal_error_releases_handle_and_disabled_timeout_is_unchanged(self):
        flags = {
            "SGLANG_HICACHE_L3_WRITE_STATE_ON_TOMBSTONE": "0",
            "SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV": "0",
        }
        transport = NixlTransport.__new__(NixlTransport)
        transport.agent = mock.Mock()
        with mock.patch.dict(os.environ, flags):
            self.assertEqual(transport.wait(["h", "PROC"], deadline=-1), "TIMEOUT")
        transport.agent.release_xfer_handle.assert_called_once_with("h")
        transport.agent.reset_mock()
        flags["SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV"] = "1"
        with mock.patch.dict(os.environ, flags):
            self.assertEqual(transport.wait(["h", "ERR"], deadline=-1), "ERR")
        transport.agent.release_xfer_handle.assert_called_once_with("h")


if __name__ == "__main__":
    unittest.main()
