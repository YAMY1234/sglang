"""Unit tests for the nixl_dram HiCache L3 backend -- no NIXL, no GPU.

The pool service's slab index, NIC selection and exit memory check; the client's settings,
discovery and routing; a client against two in-process pool services over their real ZMQ
control plane (an in-memory transport stands in for NIXL); and the store's Mooncake key layout.
"""

import contextlib
import ctypes
import io
import itertools
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, PoolName
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.nixl_dram import nixl_dram_l3 as l3
from sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_client import (
    ERR_MISSING,
    ERR_SIZE,
    ERR_XFER,
    NixlDramClient,
    NixlDramConfig,
    NixlDramError,
    discover_services,
    registration_ranges,
)
from sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_store import HiCacheNixlDram
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

KIB, MIB = 1 << 10, 1 << 20


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _index(slabs, slab_bytes=64 * KIB):
    clock = _Clock()
    index = l3.SlabIndex(0x10000, slabs * slab_bytes, slab_bytes, clock=clock)
    return index, clock


class TestSlabIndex(CustomTestCase):
    def _put(self, index, key, size):
        [addr] = index.alloc([key], [size], f"w-{key}", 30)
        self.assertGreaterEqual(addr, 0)
        self.assertEqual(index.commit(f"w-{key}", [key], [True]), [True])
        return addr

    def test_write_then_read(self):
        index, _ = _index(4)
        # 5000 B takes an 8 KiB slot, 4096 B a 4 KiB one: two size classes, two slabs.
        addrs = index.alloc(["a", "b"], [5000, 4096], "w", 30)
        self.assertEqual(addrs, [0x10000, 0x10000 + 64 * KIB])
        self.assertEqual(index.exists(["a", "b"]), [0, 0])  # reserved, not committed
        self.assertEqual(index.alloc(["a"], [5000], "w2", 30), [l3.BUSY])
        self.assertEqual(index.commit("w", ["a", "b"], [True, True]), [True, True])
        self.assertEqual(index.exists(["a", "b", "c"]), [1, 1, 0])
        self.assertEqual(index.alloc(["a"], [5000], "w3", 30), [l3.EXISTS])
        self.assertEqual(index.lookup(["a", "c"], "r", 30), ([0x10000, -1], [5000, 0]))
        self.assertEqual(index.release("r"), 1)
        index.check()

    def test_rejected_sizes_and_failed_writes_free_their_slot(self):
        index, _ = _index(1)
        self.assertEqual(
            index.alloc(["big", "empty"], [64 * KIB + 1, 0], "w", 30),
            [l3.TOO_BIG, l3.TOO_BIG],
        )
        [addr] = index.alloc(["a"], [100], "w1", 30)
        self.assertEqual(index.commit("w1", ["a"], [False]), [False])
        self.assertEqual(index.exists(["a"]), [0])
        self.assertEqual(index.used, 0)
        self.assertEqual(index.alloc(["b"], [100], "w2", 30), [addr])
        index.check()

    def test_full_class_evicts_its_least_recently_used_object(self):
        index, clock = _index(1)  # one slab: four 16 KiB slots
        for t, key in enumerate(["k0", "k1", "k2", "k3"]):
            clock.now = t
            self._put(index, key, 16 * KIB)
        clock.now = 4
        index.lookup(["k0"], "r", 30)  # k0 becomes the most recently used
        index.release("r")
        clock.now = 5
        self._put(index, "k4", 16 * KIB)
        self.assertEqual(index.exists(["k0", "k1", "k2", "k3", "k4"]), [1, 0, 1, 1, 1])
        self.assertEqual(index.stats["evicted"], 1)
        index.check()

    def test_objects_being_read_are_never_evicted(self):
        index, _ = _index(1)  # one slab: two 32 KiB slots
        self._put(index, "a", 32 * KIB)
        self._put(index, "b", 32 * KIB)
        index.lookup(["a", "b"], "r", 30)
        self.assertEqual(index.alloc(["c"], [32 * KIB], "w", 30), [l3.NO_SPACE])
        index.release("r")
        self.assertGreaterEqual(index.alloc(["c"], [32 * KIB], "w", 30)[0], 0)
        self.assertEqual(index.exists(["a", "b"]), [0, 1])
        index.check()

    def test_a_colder_slab_of_another_size_class_is_taken_over(self):
        # With "small" last used before "b1", its whole slab moves to the 32 KiB class;
        # once "small" is read again, the 32 KiB class evicts its own oldest object.
        for read_small, evicted in ((False, "small"), (True, "b1")):
            with self.subTest(read_small=read_small):
                index, clock = _index(2)
                clock.now = 1
                self._put(index, "small", 16 * KIB)
                clock.now = 2
                self._put(index, "b1", 32 * KIB)
                clock.now = 3
                self._put(index, "b2", 32 * KIB)  # the 32 KiB slab is now full
                if read_small:
                    clock.now = 3.5
                    index.lookup(["small"], "r", 30)
                    index.release("r")
                clock.now = 4
                self._put(index, "b3", 32 * KIB)
                keys = ["small", "b1", "b2", "b3"]
                present = dict(zip(keys, index.exists(keys)))
                self.assertEqual(present[evicted], 0)
                self.assertEqual(sum(present.values()), 3)
                self.assertEqual(index.stats["slab_moves"], 0 if read_small else 1)
                index.check()

    def test_expired_reservations_and_pins_are_released(self):
        index, _ = _index(1)
        self._put(index, "read", 4096)
        index.alloc(["write"], [4096], "w", 5)
        index.lookup(["read"], "r", 5)
        self.assertEqual(index.expire(now=4), (0, 0))
        self.assertEqual(index.expire(now=5), (1, 1))
        self.assertEqual(index.exists(["read", "write"]), [1, 0])
        # A commit after its lease ran out publishes nothing.
        self.assertEqual(index.commit("w", ["write"], [True]), [False])
        self.assertEqual(index.clear(), 1)
        self.assertEqual(index.entries, {})
        index.check()


class TestPoolServiceDevices(CustomTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.sysfs = tmp.name
        for name, link, state in (
            ("mlx5_0", "InfiniBand", "4: ACTIVE"),
            ("mlx5_1", "InfiniBand", "1: DOWN"),
            ("mlx5_2", "Ethernet", "4: ACTIVE"),
            ("mlx5_3", "InfiniBand", "4: ACTIVE"),
            ("mlx5_10", "InfiniBand", "5: ACTIVE_DEFER"),
        ):
            port = os.path.join(self.sysfs, name, "ports", "1")
            os.makedirs(port)
            for field, value in (("link_layer", link), ("state", state)):
                with open(os.path.join(port, field), "w") as f:
                    f.write(value + "\n")

    def _select(self, spec, **kwargs):
        return l3.select_devices(spec, ib_sysfs=self.sysfs, **kwargs)

    def test_auto_takes_the_active_ports_of_one_link_layer(self):
        devices, seen, bad = self._select("auto")
        self.assertEqual(devices, ["mlx5_0", "mlx5_3"])
        self.assertEqual(bad, [])
        self.assertEqual(len(seen), 5)
        self.assertEqual(self._select("auto", link_layer="Ethernet")[0], ["mlx5_2"])

    def test_preflight_failures(self):
        self.assertIn("--expect-ports 3", self._select("auto", expect=3)[2][0])
        self.assertEqual(
            self._select("mlx5_0,mlx5_1")[2], ["mlx5_1: InfiniBand 1: DOWN"]
        )
        self.assertIn("not one link layer", self._select("mlx5_0,mlx5_2")[2][0])
        self.assertEqual(
            self._select("mlx5_0,mlx5_3", expect=2),
            (["mlx5_0", "mlx5_3"], mock.ANY, []),
        )
        # Empty: no device list, UCX picks.
        devices, _, bad = self._select("")
        self.assertEqual((devices, bad), ([], []))

    def test_devices_default_to_auto(self):
        args = l3.parse_args(["--numa", "1", "--segment-gib", "8"])
        self.assertEqual(args.ucx_devices, "auto")


class TestPoolServiceTeardown(CustomTestCase):
    """The exit check reads per-node meminfo the way /sys/devices/system/node does."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.sysfs = tmp.name
        self._meminfo(0, free_kb=900_000)  # available: 950000 kB
        self._meminfo(1, free_kb=950_000)  # available: 1000000 kB
        self._meminfo(2, free_kb=0, total_kb=0)  # a GPU-memory node: not counted

    def _meminfo(self, node, free_kb, total_kb=1_000_000):
        # available = MemFree + Active(file) + Inactive(file) - Dirty - Writeback + SReclaimable
        rows = {
            "MemTotal": total_kb,
            "MemFree": free_kb,
            "Active(file)": 30_000,
            "Inactive(file)": 20_000,
            "Dirty": 5_000,
            "Writeback": 0,
            "SReclaimable": 5_000,
        }
        os.makedirs(os.path.join(self.sysfs, f"node{node}"), exist_ok=True)
        with open(os.path.join(self.sysfs, f"node{node}", "meminfo"), "w") as f:
            for field, kb in rows.items():
                f.write(f"Node {node} {field}:{' ' * 8}{kb} kB\n")

    def _check(self, baseline, timeout_s=0, sleep=None):
        clock = _Clock()

        def advance(seconds):
            clock.now += seconds
            if sleep is not None:
                sleep()

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = l3.teardown_check(
                "n1", baseline, 95.0, timeout_s, self.sysfs, clock=clock, sleep=advance
            )
        return rc, out.getvalue().splitlines(), clock.now

    def test_memory_back_at_exit_is_ok(self):
        self.assertEqual(
            {n: l3.available_kb(m) for n, m in l3.meminfo(self.sysfs).items()},
            {0: 950_000, 1: 1_000_000},
        )
        rc, lines, _ = self._check({0: 960_000, 1: 1_000_000})
        self.assertEqual(rc, 0)
        self.assertEqual(
            lines,
            [
                "L3B_TEARDOWN ok host n1 numa 0 avail_kb 950000 base_kb 960000",
                "L3B_TEARDOWN ok host n1 numa 1 avail_kb 1000000 base_kb 1000000",
            ],
        )

    def test_memory_still_held_at_the_deadline_fails(self):
        rc, lines, waited = self._check({0: 960_000, 1: 1_100_000}, timeout_s=3)
        self.assertEqual(rc, 1)
        self.assertEqual(waited, 3)
        self.assertEqual(
            lines,
            [
                "L3B_TEARDOWN ok host n1 numa 0 avail_kb 950000 base_kb 960000",
                "L3B_TEARDOWN FAIL host n1 numa 1 avail_kb 1000000 base_kb 1100000",
            ],
        )

    def test_waits_until_the_memory_comes_back(self):
        rc, lines, waited = self._check(
            {0: 1_050_000},
            timeout_s=20,
            sleep=lambda: self._meminfo(0, free_kb=960_000),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(waited, 1)
        self.assertEqual(
            lines, ["L3B_TEARDOWN ok host n1 numa 0 avail_kb 1010000 base_kb 1050000"]
        )

    def test_defaults(self):
        args = l3.parse_args(["--numa", "0", "--segment-gib", "8"])
        self.assertEqual((args.teardown_pct, args.teardown_timeout_s), (95.0, 20.0))


class TestNixlDramClientConfig(CustomTestCase):
    def test_extra_config_keys_are_prefixed_and_checked(self):
        cfg = NixlDramConfig.load(
            {
                "nixl_dram_endpoints": ["10.0.0.1:18800", "10.0.0.2:18800"],
                "nixl_dram_reg_chunk_mib": "2048",
                "prefetch_threshold": 256,
            }
        )
        self.assertEqual(cfg.endpoints, "10.0.0.1:18800,10.0.0.2:18800")
        self.assertEqual(cfg.reg_chunk_mib, 2048)
        self.assertEqual(cfg.devices(), [])  # UCX picks the devices
        with self.assertRaisesRegex(ValueError, "nixl_dram_bogus"):
            NixlDramConfig.load({"nixl_dram_endpoints": "h:1", "nixl_dram_bogus": 1})

    def test_invalid_settings_are_rejected(self):
        for extra, message in (
            ({}, "nixl_dram_registry_dir or nixl_dram_endpoints"),
            ({"nixl_dram_registry_dir": "registry"}, "nixl_dram_expected_services"),
            ({"nixl_dram_endpoints": "h:1", "nixl_dram_lease_s": 10}, "must exceed"),
            ({"nixl_dram_endpoints": "h:1", "nixl_dram_reg_chunk_mib": -1}, "chunk"),
        ):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, message):
                    NixlDramConfig.load(extra)

    def test_registration_ranges_cut_on_whole_pages(self):
        base = 1 << 32
        self.assertEqual(
            registration_ranges(base, 10 * MIB, MIB, 0), [(base, 10 * MIB)]
        )
        self.assertEqual(
            registration_ranges(base, 10 * MIB, MIB, 4 * MIB),
            [(base, 4 * MIB), (base + 4 * MIB, 4 * MIB), (base + 8 * MIB, 2 * MIB)],
        )
        # 96 KiB pages: every cut is on a page and on a 64 KiB OS page (192 KiB steps).
        ranges = registration_ranges(base, 2 * MIB, 96 * KIB, MIB)
        self.assertEqual([n for _, n in ranges], [960 * KIB, 960 * KIB, 128 * KIB])
        for p, _ in ranges:
            self.assertEqual((p - base) % (192 * KIB), 0)

    def test_registry_discovery_waits_for_every_service_and_sorts_them(self):
        with tempfile.TemporaryDirectory() as d:
            for host, numa, port in (
                ("node1", 0, 18800),
                ("node0", 1, 18801),
                ("node0", 0, 18800),
            ):
                name = f"{host}.numa{numa}.p{port}"
                with open(os.path.join(d, f"{name}.json"), "w") as f:
                    json.dump(
                        {"service": name, "host": host, "numa": numa, "port": port},
                        f,
                    )
            with open(os.path.join(d, "partial.json"), "w") as f:
                f.write("{")  # a record being replaced is read on the next pass

            def load(expected, **extra):
                return NixlDramConfig.load(
                    {
                        "nixl_dram_registry_dir": d,
                        "nixl_dram_expected_services": expected,
                        **extra,
                    }
                )

            self.assertEqual(
                [r["service"] for r in discover_services(load(3))],
                ["node0.numa0.p18800", "node0.numa1.p18801", "node1.numa0.p18800"],
            )
            with self.assertRaisesRegex(NixlDramError, "more than"):
                discover_services(load(2))
            with self.assertRaisesRegex(NixlDramError, "only 3 of 4"):
                discover_services(
                    load(4, nixl_dram_discovery_timeout_s=5),
                    clock=itertools.count(0, 10).__next__,
                    sleep=lambda s: None,
                )

    def test_keys_route_by_crc32(self):
        # Every rank must place a key on the same service, so routing is a fixed hash.
        client = NixlDramClient.__new__(NixlDramClient)
        client.services = [{}, {}, {}]
        self.assertEqual(
            client._route(["a", "b", "c", "d", "e", "f"]), {0: [0, 2, 3, 4], 2: [1, 5]}
        )


class _MemTransport:
    """In-process stand-in for NIXL: moves the bytes with memmove and, like NIXL, refuses
    local ranges that were not registered and remote ranges outside the service's pool."""

    def __init__(self):
        self.local, self.posts = [], []
        self.fail_waits = 0

    def register_local(self, ranges):
        self.local.extend((p, p + n) for p, n in ranges)

    def add_remote(self, hello, payload):
        return hello["pool_base"], hello["pool_bytes"]

    def remove_remote(self, remote):
        pass

    def post(self, op, remote, local, remote_descs):
        base, size = remote
        self.posts.append(op)
        for (la, n, _), (ra, _, _) in zip(local, remote_descs):
            la, n, ra = int(la), int(n), int(ra)
            if not any(s <= la and la + n <= e for s, e in self.local):
                raise NixlDramError(f"local range 0x{la:x}+{n} is not registered")
            if not (base <= ra and ra + n <= base + size):
                raise NixlDramError(f"remote range 0x{ra:x}+{n} is outside the pool")
            dst, src = (ra, la) if op == "WRITE" else (la, ra)
            ctypes.memmove(dst, src, n)

    def wait(self, posted, deadline):
        if self.fail_waits:
            self.fail_waits -= 1
            return "ERR"
        return "DONE"

    def close(self):
        pass


class TestNixlDramClientWithServices(CustomTestCase):
    """Two pool services with real slab indexes and ZMQ control planes; pools in process memory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        stop = threading.Event()
        services, threads = [], []
        for numa in range(2):
            pool = (ctypes.c_char * MIB)()
            base = ctypes.addressof(pool)
            export = mock.Mock()
            export.hello.return_value = ({"agent_name": f"agent{numa}"}, b"metadata")
            service = l3.Service(
                SimpleNamespace(
                    port=0,
                    bind="127.0.0.1",
                    numa=numa,
                    registry_dir=tmp.name,
                    stats_interval_s=0,
                ),
                l3.SlabIndex(base, MIB, 64 * KIB),
                export,
                SimpleNamespace(base=base, nbytes=MIB, buffer=pool),
                "node0",
                "127.0.0.1",
                stop,
            )
            service.publish()
            services.append(service)
            threads.append(threading.Thread(target=service.serve, daemon=True))
            threads[-1].start()

        def close():
            stop.set()
            for thread in threads:
                thread.join()
            for service in services:
                service.close()

        self.addCleanup(close)
        self._hosts = []
        self.cfg = NixlDramConfig.load(
            {
                "nixl_dram_registry_dir": tmp.name,
                "nixl_dram_expected_services": 2,
                "nixl_dram_rpc_timeout_s": 5,
                "nixl_dram_xfer_timeout_s": 5,
            }
        )

    def _client(self, transport, staging_mib=0, reconnect=False):
        with (
            envs.SGLANG_HICACHE_NIXL_DRAM_STAGING_MIB.override(staging_mib),
            envs.SGLANG_HICACHE_NIXL_DRAM_RECONNECT.override(reconnect),
        ):
            client = NixlDramClient(self.cfg, transport=transport)
        self.addCleanup(client.close)
        return client

    def _round_trip(self, client, count, part_bytes):
        """Write `count` two-part objects from a host buffer, clear it, read them back."""
        host = (ctypes.c_char * (2 * count * part_bytes))()
        self._hosts.append(host)  # registered memory must outlive the client
        base = ctypes.addressof(host)
        self.assertEqual(client.register_buffer(base, len(host)), 0)
        keys = [f"page-{i}" for i in range(count)]
        self.assertEqual(set(client._route(keys)), {0, 1})  # both services hold some
        # Object i: row 2i whole, then the first half of row 2i + 1.
        ptrs = [
            [base + 2 * i * part_bytes, base + (2 * i + 1) * part_bytes]
            for i in range(count)
        ]
        sizes = [[part_bytes, part_bytes // 2]] * count
        for i in range(count):
            ctypes.memset(ptrs[i][0], i + 1, part_bytes)
            ctypes.memset(ptrs[i][1], i + 101, part_bytes // 2)
        self.assertEqual(client.batch_put_from(keys, ptrs, sizes), [0] * count)
        ctypes.memset(base, 0, len(host))
        self.assertEqual(
            client.batch_get_into(keys, ptrs, sizes), [part_bytes * 3 // 2] * count
        )
        for i in range(count):
            row = 2 * i * part_bytes
            self.assertEqual(host[row : row + part_bytes], bytes([i + 1]) * part_bytes)
            row += part_bytes
            self.assertEqual(
                host[row : row + part_bytes // 2], bytes([i + 101]) * (part_bytes // 2)
            )
        return keys, ptrs, sizes, base

    def test_put_get_exists_and_clear(self):
        client = self._client(_MemTransport())
        self.assertEqual(
            [name.split(".p")[0] for name in client.service_names],
            ["node0.numa0", "node0.numa1"],
        )
        keys, ptrs, sizes, base = self._round_trip(client, 8, 4 * KIB)
        self.assertEqual(client.batch_is_exist(keys + ["other"]), [1] * 8 + [0])
        posts = len(client.transport.posts)
        self.assertEqual(client.batch_put_from(keys, ptrs, sizes), [0] * 8)
        self.assertEqual(len(client.transport.posts), posts)  # stored: no transfer
        self.assertEqual(
            client.batch_get_into(["other", keys[0]], [base, base], [4096, 4096]),
            [ERR_MISSING, ERR_SIZE],
        )
        self.assertEqual(client.remove_all(), 8)
        self.assertEqual(client.batch_is_exist(keys), [0] * 8)

    def test_staging_never_registers_the_host_pool(self):
        transport = _MemTransport()
        client = self._client(transport, staging_mib=1)
        self.assertEqual(len(transport.local), 1)  # the staging region only
        # 24 objects of 48 KiB: more than the 1 MiB region holds at once.
        self._round_trip(client, 24, 32 * KIB)
        self.assertEqual(len(transport.local), 1)

    def test_a_failed_transfer_is_a_failed_write_unless_reconnect_is_on(self):
        for reconnect in (False, True):
            with self.subTest(reconnect=reconnect):
                transport = _MemTransport()
                client = self._client(transport, reconnect=reconnect)
                host = (ctypes.c_char * 4096)()
                client.register_buffer(ctypes.addressof(host), 4096)
                key = f"page-{int(reconnect)}"
                transport.fail_waits = 1
                result = client.batch_put_from([key], [ctypes.addressof(host)], [4096])
                self.assertEqual(result, [0] if reconnect else [ERR_XFER])
                self.assertEqual(client.batch_is_exist([key]), [int(reconnect)])
                self.assertEqual(client.counters["reconnects"], int(reconnect))
                self.assertEqual(client.counters["retried_writes"], int(reconnect))


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
            "nixl_dram_endpoints": "127.0.0.1:18800",
            "nixl_dram_reg_chunk_mib": 2048,
            **extra,
        },
    )
    client = mock.Mock()
    client.register_buffer.return_value = 0
    return HiCacheNixlDram(config, client=client)


class TestHiCacheNixlDramStore(CustomTestCase):
    def test_keys_follow_the_mooncake_layout(self):
        store = _store(extra_backend_tag="tag")
        self.assertEqual(store.config_prefix, "tag_org-model")
        self.assertEqual(store._tag_keys(["h"]), ["tag_org-model_h"])
        self.assertEqual(store.mla_suffix, "3")
        self.assertEqual(store.mha_suffix, "0_3")

    def test_object_layout_is_not_published(self):
        layout = _store().describe_l3_object_layout(
            sidecar_pool_specs=(), window_trailing_pages={}, page_size=64
        )
        self.assertFalse(layout["enabled"])
        self.assertEqual(layout["backend"], "nixl_dram")

    def test_mamba_pool_with_a_list_kv_buffer_registers_every_state_buffer(self):
        # A Mamba host pool's kv_buffer is the list of its state buffers.
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

    def test_zero_copy_io_goes_to_the_client(self):
        store = _store()
        store._put_batch_zero_copy_impl(["k"], [[1, 2]], [[3, 4]], group_ids=["g"])
        store.store.batch_put_from.assert_called_once_with(["k"], [[1, 2]], [[3, 4]])
        store._get_batch_zero_copy_impl(["k"], [5], [6])
        store.store.batch_get_into.assert_called_once_with(["k"], [5], [6])

    def test_factory_builds_it_with_the_host_pool(self):
        entry = StorageBackendFactory._registry["nixl_dram"]
        self.assertEqual(
            (entry["module_path"], entry["class_name"]),
            (
                "sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_store",
                "HiCacheNixlDram",
            ),
        )
        backend_class = mock.Mock()
        StorageBackendFactory._create_builtin_backend(
            "nixl_dram", backend_class, "config", "pool"
        )
        backend_class.assert_called_once_with("config", "pool")


if __name__ == "__main__":
    unittest.main()
