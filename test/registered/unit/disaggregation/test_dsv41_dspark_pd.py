import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
    get_dsv41_spec_layout,
)
from sglang.srt.mem_cache.common import retraction_backup
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_layout():
    args = SimpleNamespace(
        mla_compression_ratios=[0, 2, 1],
        kv_layer_ids=[1, 2],
        kv_item_lens=[512, 1024],
        state_types=[StateType.SWA, StateType.C128_STATE, StateType.SWA],
        state_item_lens=[[512], [32768], [512]],
    )
    with get_context().override_server_args(
        speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
    ):
        return get_dsv41_spec_layout(args)


class TestDSV41DSparkPD(CustomTestCase):
    def test_dsv4_ratio_bucket_layer_ids_pair_pp_final_stage_and_draft(self):
        def make_pool(*, start, end, sources):
            pool = object.__new__(DeepSeekV4TokenToKVPool)
            pool._unified_kv = False
            pool._stage_start = start
            pool._stage_end = end
            pool.sources_by_ratio = sources
            pool.kv_pools = {ratio: object() for ratio in sources}
            pool.index_pools = {ratio: object() for ratio in sources if ratio != 128}
            return pool

        prefill_target = make_pool(
            start=20,
            end=40,
            sources={4: [20, 24], 128: [28], 1: [32], 2: [38]},
        )
        decode_target = make_pool(
            start=0,
            end=40,
            sources={4: [2, 8, 20, 24], 128: [14, 28], 1: [32], 2: [38]},
        )
        draft = make_pool(
            start=0,
            end=3,
            sources={4: [0], 128: [1], 2: [2]},
        )

        src = build_kv_layer_ids(
            token_to_kv_pool=prefill_target,
            draft_token_to_kv_pool=draft,
            num_draft_entries=len(draft.get_kv_layer_ids()),
            num_hidden_layers=40,
        )
        dst = build_kv_layer_ids(
            token_to_kv_pool=decode_target,
            draft_token_to_kv_pool=draft,
            num_draft_entries=len(draft.get_kv_layer_ids()),
            num_hidden_layers=40,
        )

        self.assertEqual(
            prefill_target.get_kv_layer_ids(),
            [20, 24, 20, 24, 28, 32, 32, 38, 38],
        )
        self.assertEqual(draft.get_kv_layer_ids(), [0, 0, 1, 2, 2])
        pairs = build_transfer_entry_pairs(src, dst, len(src), len(dst))
        self.assertEqual([src[i] for i, _ in pairs], [dst[j] for _, j in pairs])
        self.assertEqual([src[i] for i, _ in pairs][-5:], [40, 40, 41, 42, 42])

        manager = SimpleNamespace(
            is_mla_backend=True,
            is_hybrid_mla_backend=False,
            pp_size=2,
            enable_custom_mem_pool=False,
            _transfer_data=Mock(return_value=0),
        )
        src_ptrs = [1000 + 100 * i for i in range(len(src))]
        dst_ptrs = [10000 + 100 * i for i in range(len(dst))]
        self.assertEqual(
            MooncakeKVManager._send_kvcache_generic(
                manager,
                mooncake_session_id="session",
                src_data_ptrs=src_ptrs,
                dst_data_ptrs=dst_ptrs,
                item_lens=[16] * len(src),
                prefill_data_indices=np.array([0], dtype=np.int32),
                dst_data_indices=np.array([0], dtype=np.int32),
                executor=None,
                src_layer_ids=src,
                dst_layer_ids=dst,
            ),
            0,
        )
        transfer_blocks = manager._transfer_data.call_args.args[1]
        self.assertEqual(
            transfer_blocks,
            [(src_ptrs[i], dst_ptrs[j], 16) for i, j in pairs],
        )

    def test_unified_dsv4_layer_ids_match_flat_buffer_order(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = True
        pool._stage_start = 20
        pool._stage_end = 25
        pool.compression_ratios = [0] * 20 + [4, 128, 0, 4, 128]
        self.assertEqual(pool.get_kv_layer_ids(), [20, 23, 20, 23, 21, 24])

    def test_partitioned_prefill_layout_is_rank_invariant_and_owner_aware(self):
        common = dict(
            mla_compression_ratios=[0, 2, 1],
        )
        first = SimpleNamespace(
            **common,
            kv_layer_ids=[1],
            kv_item_lens=[512],
            state_types=[StateType.SWA, StateType.C128_STATE],
            state_item_lens=[[512], [32768]],
        )
        final = SimpleNamespace(
            **common,
            kv_layer_ids=[2, 3],
            kv_item_lens=[1024, 512],
            state_types=[StateType.SWA, StateType.C128_STATE, StateType.SWA],
            state_item_lens=[[512], [32768], [512]],
        )
        with (
            get_context().override_server_args(
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=6,
                disaggregation_mode="prefill",
            ),
            get_parallel().override(pp_size=2, pp_rank=0),
        ):
            first_layout = get_dsv41_spec_layout(first)
        with (
            get_context().override_server_args(
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=6,
                disaggregation_mode="prefill",
            ),
            get_parallel().override(pp_size=2, pp_rank=1),
        ):
            final_layout = get_dsv41_spec_layout(final)

        self.assertEqual(first_layout, final_layout)
        self.assertEqual(
            first_layout,
            {
                "num_draft_tokens": 6,
                "compression_ratios": [0, 2, 1],
                "partitioned_prefill": True,
            },
        )
        with (
            get_context().override_server_args(
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=6,
                disaggregation_mode="decode",
            ),
            get_parallel().override(pp_size=1, pp_rank=0),
        ):
            decode_layout = get_dsv41_spec_layout(final, partitioned_prefill=True)
        self.assertEqual(decode_layout, first_layout)

        final.state_types = [StateType.SWA, StateType.C128_STATE]
        with (
            get_context().override_server_args(
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=6,
                disaggregation_mode="prefill",
            ),
            get_parallel().override(pp_size=2, pp_rank=1),
            self.assertRaisesRegex(RuntimeError, "draft SWA state only on the final"),
        ):
            get_dsv41_spec_layout(final)

    def test_bootstrap_validates_before_caching(self):
        layout = make_layout()
        cases = [("matching", layout, layout, 4, True), ("legacy", None, None, 2, True)]
        for key, value in (
            ("num_draft_tokens", 5),
            ("kv_layer_ids", [2, 1]),
            ("kv_item_lens", [256, 1024]),
            ("state_types", ["swa", "c128_state"]),
            ("state_item_lens", [[512], [8192], [512]]),
        ):
            different = copy.deepcopy(layout)
            different[key] = value
            cases.append((key, layout, different, 4, False))
        cases += [
            ("prefill_only", None, layout, 4, False),
            ("decode_only_or_old_prefill", layout, None, 4, False),
            ("tp_mismatch", layout, layout, 2, False),
        ]
        for name, local, peer, tp_size, supported in cases:
            with self.subTest(name=name):
                manager = object.__new__(CommonKVManager)
                manager.prefill_info_table = {}
                manager.kv_args = SimpleNamespace(page_size=256)
                manager.kv_cache_dtype_str = "fp8_e4m3"
                manager.dsv41_spec_layout = local
                manager.attn_tp_size = 4
                manager.dcp_size = 1
                manager._resolve_rank_mapping = Mock()
                response = Mock(status_code=200)
                response.json.return_value = dict(
                    attn_tp_size=tp_size,
                    attn_cp_size=1,
                    dp_size=1,
                    pp_size=1,
                    page_size=256,
                    kv_cache_dtype="fp8_e4m3",
                    follow_bootstrap_room=True,
                    dsv41_spec_layout=peer,
                )
                with patch(
                    "sglang.srt.disaggregation.common.conn.requests.get",
                    return_value=response,
                ) as fetch:
                    if supported:
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )
                        self.assertTrue(
                            manager.try_ensure_parallel_info("prefill:8998")
                        )
                        fetch.assert_called_once()
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "DeepSeek-V4.1 DSpark PD"
                        ):
                            manager.try_ensure_parallel_info("prefill:8998")
                        self.assertFalse(manager.prefill_info_table)
                        manager._resolve_rank_mapping.assert_not_called()

    def test_python_bootstrap_preserves_layout_and_rejects_mixed_ranks(self):
        with patch.object(CommonKVBootstrapServer, "run"):
            server = CommonKVBootstrapServer("127.0.0.1", 8998)
        layout = make_layout()
        payload = dict(
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            pp_size=1,
            pp_rank=0,
            system_dp_size=1,
            system_dp_rank=0,
            rank_ip="127.0.0.1",
            rank_port=1234,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            dsv41_spec_layout=layout,
        )
        request = Mock(json=AsyncMock(return_value=payload))
        self.assertEqual(asyncio.run(server._handle_route_put(request)).status, 200)
        query = Mock(
            query={
                key: "-1"
                for key in (
                    "prefill_dp_rank",
                    "prefill_cp_rank",
                    "target_tp_rank",
                    "target_pp_rank",
                )
            }
        )
        response = asyncio.run(server._handle_route_get(query))
        self.assertEqual(json.loads(response.text)["dsv41_spec_layout"], layout)
        payload["dsv41_spec_layout"] = None
        self.assertEqual(asyncio.run(server._handle_route_put(request)).status, 400)
        self.assertEqual(server._registered_count, 1)
        self.assertEqual(server.dsv41_spec_layout, layout)

    def test_retraction_recomputes_from_prefill_and_replays_boundary_token(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.compression_ratios = [0, 2, 1]
        pool.device = "cuda"
        allocator = Mock(get_kvcache=Mock(return_value=pool))
        for algorithm in (None, "DSPARK"):
            with (
                self.subTest(algorithm=algorithm),
                get_context().override_server_args(speculative_algorithm=algorithm),
                patch("torch.get_device_module") as device_module,
            ):
                req = SimpleNamespace(
                    output_ids=[7, 8],
                    bootstrap_host="prefill",
                    time_stats=Mock(),
                    offload_kv_cache=Mock(),
                )
                request_pool = Mock()
                self.assertTrue(
                    retraction_backup(
                        req, Mock(), request_pool, allocator, "cpu_tensor"
                    )
                )
                queue = SimpleNamespace(
                    token_to_kv_pool_allocator=allocator,
                    _check_if_req_exceed_kv_capacity=Mock(return_value=False),
                    _create_receiver_and_enqueue=Mock(),
                    _resolve_prefill_dp_rank=Mock(return_value=0),
                    retracted_queue=[],
                    pending_reqs=[],
                )
                DecodePreallocQueue.add(queue, req, is_retracted=True)
                if algorithm == "DSPARK":
                    req.offload_kv_cache.assert_not_called()
                    device_module.return_value.synchronize.assert_called_once_with(
                        "cuda"
                    )
                    self.assertEqual(req.output_ids, [7])
                    self.assertEqual(req.pd_rebootstrap_forced_output_id, 8)
                    self.assertTrue(req.pd_rebootstrap_in_progress)
                    queue._create_receiver_and_enqueue.assert_called_once_with(
                        req, is_rebootstrap=True
                    )
                    self.assertFalse(queue.retracted_queue)
                else:
                    req.offload_kv_cache.assert_called_once_with(
                        request_pool, allocator
                    )
                    device_module.assert_not_called()
                    self.assertEqual(req.output_ids, [7, 8])
                    self.assertEqual(queue.retracted_queue, [req])


if __name__ == "__main__":
    unittest.main()
