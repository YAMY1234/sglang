"""Unit tests for the nixl_dram HiCache L3 backend's pool registration -- no service, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ctypes
import mmap
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, PoolName
from sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_store import HiCacheNixlDram
from sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_client import (
    NixlDramClient,
    NixlDramError,
    NixlTransport,
)
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


class _MemTransport:
    """In-memory stand-in for NIXL: like ShmTransport, it refuses local ranges that were not registered."""

    def __init__(self, pool_bytes):
        self.pool = (ctypes.c_char * pool_bytes)()
        self.base = ctypes.addressof(self.pool)
        self.local, self.posts = [], []
        self.memmove = ctypes.memmove  # bound now, so tests can patch the staging copies' memmove alone

    def register_local(self, ranges):
        self.local.extend((p, p + n) for p, n in ranges)

    def post(self, op, remote, local, remote_descs):
        self.posts.append((op, list(local)))
        for (la, n, _), (ra, _, _) in zip(local, remote_descs):
            la, n, ra = int(la), int(n), int(ra)
            if not any(s <= la and la + n <= e for s, e in self.local):
                raise NixlDramError(f"local range 0x{la:x}+{n} is not registered")
            dst, src = (self.base + ra, la) if op == "WRITE" else (la, self.base + ra)
            self.memmove(dst, src, n)

    def wait(self, posted, deadline):
        return "DONE"

    def close(self):
        pass


def _staged_client(staging_mib, pool_bytes=1 << 20):
    client = NixlDramClient.__new__(NixlDramClient)
    client.cfg = SimpleNamespace(xfer_timeout_s=5.0)
    client.transport = _MemTransport(pool_bytes)
    client.remotes, client.service_names = [None], ["svc0"]
    client.counters = {"xfer_errors": 0}
    client._lock = threading.Lock()
    client._staging = None
    client._init_staging(staging_mib)
    return client


def _host(n, fill):
    b = (ctypes.c_char * n)(*([fill] * n))
    return b, ctypes.addressof(b)


class TestNixlDramStaging(CustomTestCase):
    def test_write_and_read_bounce_through_staging_with_an_unregistered_host_pool(self):
        client = _staged_client(1)
        a, pa = _host(3000, 7)
        b, pb = _host(5000, 9)
        plans = {0: [(0, 0), (1, 8192)]}
        ptrs, sizes = [[pa], [pb]], [[3000], [5000]]
        self.assertEqual(client._transfer("WRITE", plans, ptrs, sizes), {0: "DONE"})
        _, stage, cap = client._staging
        for op, local in client.transport.posts:
            self.assertTrue(all(stage <= la and la + n <= stage + cap for la, n, _ in local))
        ctypes.memset(pa, 0, 3000)
        ctypes.memset(pb, 0, 5000)
        self.assertEqual(client._transfer("READ", plans, ptrs, sizes), {0: "DONE"})
        self.assertEqual(bytes(a), bytes([7]) * 3000)
        self.assertEqual(bytes(b), bytes([9]) * 5000)

    def test_multi_part_object_is_packed_contiguously(self):
        client = _staged_client(1)
        a, pa = _host(100, 1)
        b, pb = _host(200, 2)
        self.assertEqual(client._transfer("WRITE", {0: [(0, 0)]}, [[pa, pb]], [[100, 200]]), {0: "DONE"})
        self.assertEqual(client.transport.pool[0:300], bytes([1]) * 100 + bytes([2]) * 200)

    def test_batch_larger_than_staging_is_sent_in_chunks(self):
        client = _staged_client(1, pool_bytes=2 << 20)
        objs = [_host(400 << 10, k + 1) for k in range(3)]  # 3 x 400 KiB through 1 MiB of staging
        plans = {0: [(k, k * (400 << 10)) for k in range(3)]}
        state = client._transfer("WRITE", plans, [[p] for _, p in objs], [[400 << 10]] * 3)
        self.assertEqual(state, {0: "DONE"})
        self.assertEqual(len(client.transport.posts), 2)
        for k in range(3):
            off = k * (400 << 10)
            self.assertEqual(client.transport.pool[off : off + 16], bytes([k + 1]) * 16)

    def test_object_larger_than_staging_fails_its_service(self):
        client = _staged_client(1, pool_bytes=4 << 20)
        big, pbig = _host(2 << 20, 3)
        self.assertEqual(client._transfer("WRITE", {0: [(0, 0)]}, [[pbig]], [[2 << 20]]), {0: "ERR"})

    def test_without_staging_the_host_pointers_are_posted_directly(self):
        client = _staged_client(0)
        a, pa = _host(64, 5)
        client.transport.register_local([(pa, 64)])
        self.assertEqual(client._transfer("WRITE", {0: [(0, 0)]}, [[pa]], [[64]]), {0: "DONE"})
        self.assertEqual(client.transport.posts[0][1][0][0], pa)

    def test_parts_inside_a_registered_host_buffer_move_as_torch_rows_both_ways(self):
        client = _staged_client(1, pool_bytes=1 << 20)
        page = 4096
        pool, pp = _host(64 * page, 0)  # a host pool buffer of 64 pages, two "pools" of rows inside it
        idx_pool, ip = _host(64 * 512, 0)
        client.register_buffer(pp, 64 * page)
        client.register_buffer(ip, 64 * 512)
        pages = [5, 9, 40]
        for k, pg in enumerate(pages):
            ctypes.memset(pp + pg * page, 10 + k, page)
            ctypes.memset(ip + pg * 512, 20 + k, 512)
        ptrs = [[pp + pg * page, ip + pg * 512] for pg in pages]
        sizes = [[page, 512]] * 3
        plans = {0: [(k, k * (page + 512)) for k in range(3)]}
        with mock.patch("ctypes.memmove", side_effect=AssertionError("row-aligned parts must not use memmove")):
            self.assertEqual(client._transfer("WRITE", plans, ptrs, sizes), {0: "DONE"})
        for k in range(3):
            off = k * (page + 512)
            self.assertEqual(client.transport.pool[off : off + page], bytes([10 + k]) * page)
            self.assertEqual(client.transport.pool[off + page : off + page + 512], bytes([20 + k]) * 512)
        ctypes.memset(pp, 0, 64 * page)
        ctypes.memset(ip, 0, 64 * 512)
        self.assertEqual(client._transfer("READ", plans, ptrs, sizes), {0: "DONE"})
        for k, pg in enumerate(pages):
            self.assertEqual(bytes(pool[pg * page : (pg + 1) * page]), bytes([10 + k]) * page)
            self.assertEqual(bytes(idx_pool[pg * 512 : (pg + 1) * 512]), bytes([20 + k]) * 512)
        self.assertEqual(bytes(pool[0:page]), bytes(page))  # untouched rows stay untouched

    def test_failed_read_service_does_not_overwrite_host_rows(self):
        client = _staged_client(1)
        a, pa = _host(256, 0)
        client.register_buffer(pa, 256)
        client.transport.wait = lambda posted, deadline: "ERR"
        self.assertEqual(client._transfer("READ", {0: [(0, 0)]}, [[pa]], [[256]]), {0: "ERR"})
        self.assertEqual(bytes(a), bytes(256))

    def test_register_buffer_skips_the_host_pool_in_staging_mode(self):
        client = _staged_client(1)
        before = list(client.transport.local)
        host, ph = _host(4096, 1)
        self.assertEqual(client.register_buffer(ph, 4096), 0)
        self.assertEqual(client.transport.local, before)
        self.assertEqual(len(client._host_views), 1)


if __name__ == "__main__":
    unittest.main()
