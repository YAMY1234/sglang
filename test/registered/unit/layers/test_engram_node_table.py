"""The node-shared Engram host table's rendezvous (replication node_shared). CPU only."""

import mmap
import multiprocessing
import os
import tempfile
import threading
import unittest
import uuid

from sglang.srt.layers.engram import NodeTable, node_table_address
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NBYTES = 1 << 20


def _address() -> bytes:
    return f"\0sglang-engram-test-{uuid.uuid4().hex}".encode()


def _memfd() -> int:
    fd = os.memfd_create("t", 0)
    os.ftruncate(fd, NBYTES)
    return fd


def _table(address: bytes, nbytes: int = NBYTES) -> NodeTable:
    return NodeTable(address, nbytes, "t", _memfd)


def _view(table: NodeTable) -> mmap.mmap:
    return mmap.mmap(table.fd, table.nbytes, flags=mmap.MAP_SHARED)


def _build_then_exit(address, pipe):
    """A builder that dies before it publishes its table."""
    table = _table(address)
    _view(table)[:4] = b"half"
    pipe.send(table.built_here)
    pipe.recv()
    os._exit(0)


class TestEngramNodeTable(CustomTestCase):
    def test_first_builds_and_readers_wait_for_ready(self):
        address = _address()
        builder = _table(address)
        reader = _table(address)
        self.assertTrue(builder.built_here)
        self.assertFalse(reader.built_here)
        self.assertEqual(reader.owner_pid, os.getpid())

        waiter = threading.Thread(target=reader.wait_ready)
        waiter.start()
        _view(builder)[-8:] = b"lastrow!"
        waiter.join(0.5)
        self.assertTrue(waiter.is_alive(), "a reader must wait for publish()")
        builder.publish()
        waiter.join(10)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(_view(reader)[-8:], b"lastrow!")

        late = NodeTable(address, NBYTES, "t", None)  # attach only, as a TP peer
        self.assertFalse(late.built_here)
        late.wait_ready()  # a ready table serves late readers at once
        self.assertEqual(_view(late)[-8:], b"lastrow!")

    def test_builder_exit_before_ready_fails_readers_and_frees_the_name(self):
        address = _address()
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_build_then_exit, args=(address, child))
        proc.start()
        self.assertTrue(parent.poll(60))
        self.assertTrue(parent.recv())
        reader = _table(address)
        self.assertFalse(reader.built_here)
        parent.send("exit")
        proc.join(30)
        with self.assertRaisesRegex(RuntimeError, "exited before the table was ready"):
            reader.wait_ready()
        # The dead builder's name is free: the next server builds a fresh table.
        rebuilt = _table(address)
        self.assertTrue(rebuilt.built_here)
        self.assertEqual(_view(rebuilt)[:4], b"\0\0\0\0")

    def test_size_mismatch_is_refused(self):
        address = _address()
        builder = _table(address)
        with self.assertRaisesRegex(RuntimeError, "bytes here"):
            _table(address, 2 * NBYTES)
        builder.publish()

    def test_address_keys_on_checkpoint_and_table(self):
        with tempfile.TemporaryDirectory() as model:
            with open(os.path.join(model, "config.json"), "w") as f:
                f.write('{"engram_layer_ids": [1, 14]}')
            first = node_table_address(model, "sglang_engram_1", NBYTES)
            self.assertEqual(
                first, node_table_address(model, "sglang_engram_1", NBYTES)
            )
            self.assertTrue(first.startswith(b"\0sglang-engram-"))
            self.assertLess(len(first), 108)  # sun_path
            self.assertNotEqual(
                first, node_table_address(model, "sglang_engram_14", NBYTES)
            )
            self.assertNotEqual(
                first, node_table_address(model, "sglang_engram_1", 2 * NBYTES)
            )
            with open(os.path.join(model, "config.json"), "w") as f:
                f.write('{"engram_layer_ids": [1, 15]}')
            self.assertNotEqual(
                first, node_table_address(model, "sglang_engram_1", NBYTES)
            )


if __name__ == "__main__":
    unittest.main()
