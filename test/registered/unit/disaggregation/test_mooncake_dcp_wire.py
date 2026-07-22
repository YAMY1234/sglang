import struct
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _registration_message(*, dcp_size=None, dcp_rank=None):
    msg = [
        b"None",
        b"127.0.0.1",
        b"12345",
        b"session",
        struct.pack("Q", 1000),
        struct.pack("Q", 2000),
        pack_int_lists([[3000]], "Q"),
        b"3",
        b"4",
        b"4096",
        pack_int_lists([[128]], "I"),
        pack_int_lists([[16]], "I"),
        b"",
        b"",
    ]
    if dcp_size is not None:
        msg.extend([str(dcp_size).encode("ascii"), str(dcp_rank).encode("ascii")])
    return msg


class TestMooncakeDCPWire(CustomTestCase):
    def test_registration_defaults_old_peer_to_dcp_one(self):
        info = KVArgsRegisterInfo.from_zmq(_registration_message())
        self.assertEqual(info.dst_dcp_size, 1)
        self.assertEqual(info.dst_dcp_rank, 0)

    def test_registration_round_trips_dcp_topology(self):
        info = KVArgsRegisterInfo.from_zmq(
            _registration_message(dcp_size=4, dcp_rank=3)
        )
        self.assertEqual(info.dst_dcp_size, 4)
        self.assertEqual(info.dst_dcp_rank, 3)
        self.assertEqual(info.dst_kv_ptrs, [1000])
        self.assertEqual(info.dst_kv_item_len, 4096)

    def test_dcp_transfer_uses_token_row_addresses(self):
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=4,
            kv_data_ptrs=[1000],
            kv_item_lens=[40],
        )
        manager.enable_custom_mem_pool = False
        manager.get_mla_kv_ptrs_with_pp = lambda src, dst: (src, dst, 1)
        captured = []

        def transfer_data(session_id, blocks):
            captured.append((session_id, blocks))
            return 0

        manager._transfer_data = transfer_data
        status = manager.send_kvcache_dcp(
            "session",
            np.array([10, 11], dtype=np.int32),
            [2000],
            np.array([30], dtype=np.int32),
            dst_kv_item_len=40,
            dst_dcp_size=2,
            dst_dcp_rank=0,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=5,
            executor=None,
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            captured,
            [
                (
                    "session",
                    [
                        (1400, 3200, 10),
                        (1420, 3210, 10),
                        (1440, 3220, 10),
                    ],
                )
            ],
        )

    def test_dcp_transfer_requires_exact_token_count(self):
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(page_size=4)

        with self.assertRaisesRegex(ValueError, "requires num_kv_tokens"):
            manager.send_kvcache_dcp(
                "session",
                np.array([10], dtype=np.int32),
                [2000],
                np.array([30], dtype=np.int32),
                dst_kv_item_len=40,
                dst_dcp_size=2,
                dst_dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=None,
                executor=None,
            )


if __name__ == "__main__":
    unittest.main()
