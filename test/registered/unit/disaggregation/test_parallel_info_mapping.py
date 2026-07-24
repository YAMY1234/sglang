from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    PrefillServerInfo,
)


def test_decode_tp8_maps_to_prefill_tp4_across_pp_stages():
    manager = object.__new__(CommonKVManager)
    manager.attn_tp_size = 8
    manager.attn_cp_size = 1
    manager.attn_cp_rank = 0
    manager.pp_size = 1
    manager.pp_rank = 0
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = True
    manager.enable_all_cp_ranks_for_transfer = False

    for decode_rank in range(8):
        manager.kv_args = SimpleNamespace(engine_rank=decode_rank)
        info = PrefillServerInfo(
            attn_tp_size=4,
            attn_cp_size=1,
            dp_size=1,
            pp_size=2,
            page_size=64,
            kv_cache_dtype="torch.float8_e4m3fn",
            follow_bootstrap_room=True,
        )

        manager._resolve_rank_mapping(info)

        assert info.target_tp_rank == decode_rank // 2
        assert info.target_tp_ranks == [decode_rank // 2]
        assert info.required_dst_info_num == 2
        assert info.target_pp_ranks == [0, 1]
        assert info.required_prefill_response_num == 2
