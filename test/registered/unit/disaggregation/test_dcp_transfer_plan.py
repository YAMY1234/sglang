import asyncio
import json
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    PD_DCP_TRANSFER_VERSION,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.common.utils import build_dcp_token_transfer_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDCPTokenTransferPlan(CustomTestCase):
    def test_even_and_odd_ranks_pack_strided_source_rows(self):
        src_pages = np.array([10, 11], dtype=np.int32)
        dst_pages = np.array([30], dtype=np.int32)

        rank0 = build_dcp_token_transfer_plan(
            src_pages,
            dst_pages,
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=0,
        )
        rank1 = build_dcp_token_transfer_plan(
            src_pages,
            dst_pages,
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=1,
        )

        np.testing.assert_array_equal(
            rank0.src_token_indices, np.array([40, 42, 44, 46])
        )
        np.testing.assert_array_equal(
            rank1.src_token_indices, np.array([41, 43, 45, 47])
        )
        np.testing.assert_array_equal(
            rank0.dst_token_indices, np.array([120, 121, 122, 123])
        )
        np.testing.assert_array_equal(
            rank1.dst_token_indices, np.array([120, 121, 122, 123])
        )

    def test_chunk_offset_preserves_destination_position(self):
        plan = build_dcp_token_transfer_plan(
            np.array([11], dtype=np.int32),
            np.array([30], dtype=np.int32),
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=0,
            src_page_offset=1,
        )

        np.testing.assert_array_equal(plan.src_token_indices, np.array([44, 46]))
        np.testing.assert_array_equal(plan.dst_token_indices, np.array([122, 123]))

    def test_crosses_virtual_page_boundary(self):
        plan = build_dcp_token_transfer_plan(
            np.array([10, 11, 12, 13], dtype=np.int32),
            np.array([30, 47], dtype=np.int32),
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=1,
        )

        np.testing.assert_array_equal(
            plan.src_token_indices,
            np.array([41, 43, 45, 47, 49, 51, 53, 55]),
        )
        np.testing.assert_array_equal(
            plan.dst_token_indices,
            np.array([120, 121, 122, 123, 188, 189, 190, 191]),
        )

    def test_aligned_decode_prefix_restarts_at_destination_row_zero(self):
        plan = build_dcp_token_transfer_plan(
            np.array([7], dtype=np.int32),
            np.array([20], dtype=np.int32),
            physical_page_size=4,
            dcp_size=2,
            dcp_rank=1,
            decode_prefix_len=8,
        )

        np.testing.assert_array_equal(plan.src_token_indices, np.array([29, 31]))
        np.testing.assert_array_equal(plan.dst_token_indices, np.array([80, 81]))

    def test_matches_brute_force_for_ragged_virtual_pages(self):
        rng = np.random.default_rng(1234)
        for physical_page_size in (1, 4, 8):
            for dcp_size in (2, 4):
                for num_src_pages in range(1, 2 * dcp_size + 1):
                    src_pages = rng.choice(
                        1000, size=num_src_pages, replace=False
                    ).astype(np.int32)
                    num_dst_pages = (num_src_pages + dcp_size - 1) // dcp_size
                    dst_pages = rng.choice(
                        1000, size=num_dst_pages, replace=False
                    ).astype(np.int32)

                    for dcp_rank in range(dcp_size):
                        plan = build_dcp_token_transfer_plan(
                            src_pages,
                            dst_pages,
                            physical_page_size=physical_page_size,
                            dcp_size=dcp_size,
                            dcp_rank=dcp_rank,
                        )
                        expected_src = []
                        expected_dst = []
                        for relative_position in range(
                            num_src_pages * physical_page_size
                        ):
                            if relative_position % dcp_size != dcp_rank:
                                continue
                            source_page_ordinal, source_row = divmod(
                                relative_position, physical_page_size
                            )
                            local_row = relative_position // dcp_size
                            destination_page_ordinal, destination_row = divmod(
                                local_row, physical_page_size
                            )
                            expected_src.append(
                                int(src_pages[source_page_ordinal])
                                * physical_page_size
                                + source_row
                            )
                            expected_dst.append(
                                int(dst_pages[destination_page_ordinal])
                                * physical_page_size
                                + destination_row
                            )

                        np.testing.assert_array_equal(
                            plan.src_token_indices, expected_src
                        )
                        np.testing.assert_array_equal(
                            plan.dst_token_indices, expected_dst
                        )

    def test_partial_final_page_transfers_only_valid_rows(self):
        expected_sources = {
            0: [36, 8],
            1: [37],
            2: [38],
            3: [39],
        }
        expected_destinations = {
            0: [28, 29],
            1: [28],
            2: [28],
            3: [28],
        }
        for dcp_rank in range(4):
            plan = build_dcp_token_transfer_plan(
                np.array([9, 2], dtype=np.int32),
                np.array([7], dtype=np.int32),
                physical_page_size=4,
                dcp_size=4,
                dcp_rank=dcp_rank,
                num_kv_tokens=5,
            )
            np.testing.assert_array_equal(
                plan.src_token_indices, expected_sources[dcp_rank]
            )
            np.testing.assert_array_equal(
                plan.dst_token_indices, expected_destinations[dcp_rank]
            )

    def test_empty_chunk_is_empty(self):
        plan = build_dcp_token_transfer_plan(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            physical_page_size=64,
            dcp_size=4,
            dcp_rank=3,
        )
        self.assertEqual(plan.src_token_indices.size, 0)
        self.assertEqual(plan.dst_token_indices.size, 0)

    def test_rejects_unaligned_prefix(self):
        with self.assertRaisesRegex(ValueError, "decode_prefix_len"):
            build_dcp_token_transfer_plan(
                np.array([7], dtype=np.int32),
                np.array([20], dtype=np.int32),
                physical_page_size=4,
                dcp_size=2,
                dcp_rank=0,
                decode_prefix_len=4,
            )

    def test_rejects_missing_destination_page(self):
        with self.assertRaisesRegex(ValueError, "Insufficient destination"):
            build_dcp_token_transfer_plan(
                np.array([10, 11, 12, 13], dtype=np.int32),
                np.array([30], dtype=np.int32),
                physical_page_size=4,
                dcp_size=2,
                dcp_rank=0,
            )

    def test_rejects_token_count_larger_than_source_capacity(self):
        with self.assertRaisesRegex(ValueError, "source pages"):
            build_dcp_token_transfer_plan(
                np.array([10], dtype=np.int32),
                np.array([30], dtype=np.int32),
                physical_page_size=4,
                dcp_size=2,
                dcp_rank=0,
                num_kv_tokens=5,
            )


class TestDCPBootstrapNegotiation(CustomTestCase):
    @staticmethod
    def _server():
        server = object.__new__(CommonKVBootstrapServer)
        server.attn_tp_size = 4
        server.attn_cp_size = 1
        server.dp_size = 1
        server.pp_size = 1
        server.page_size = 64
        server.kv_cache_dtype = "fp8_e4m3"
        server.follow_bootstrap_room = True
        server.enable_dsa_cache_layer_split = False
        server.prefill_http_port = 21100
        server._registered_count = 4
        return server

    @staticmethod
    def _request(*, negotiate: bool):
        query = {
            "prefill_dp_rank": "-1",
            "prefill_cp_rank": "-1",
            "target_tp_rank": "-1",
            "target_pp_rank": "-1",
        }
        if negotiate:
            query["pd_wire_version"] = str(PD_DCP_TRANSFER_VERSION)
        return SimpleNamespace(query=query)

    def test_old_decoder_receives_legacy_bootstrap_schema(self):
        response = asyncio.run(
            self._server()._handle_route_get(self._request(negotiate=False))
        )
        payload = json.loads(response.text)
        self.assertNotIn("pd_dcp_transfer_version", payload)
        self.assertEqual(PrefillServerInfo(**payload).pd_dcp_transfer_version, 0)

    def test_new_decoder_receives_dcp_capability(self):
        response = asyncio.run(
            self._server()._handle_route_get(self._request(negotiate=True))
        )
        payload = json.loads(response.text)
        self.assertEqual(
            payload["pd_dcp_transfer_version"], PD_DCP_TRANSFER_VERSION
        )


if __name__ == "__main__":
    unittest.main()
