import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.attention.mamba.track_metadata import (
    build_mamba_track_plan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMambaTrackPlan(unittest.TestCase):
    def _plan(self, *, is_mamba2: bool):
        return build_mamba_track_plan(
            track_mask=[False, True, True, False],  # final row is DP padding
            track_seqlens=[-1, 84, 95, -1],
            extend_seq_lens=[63, 64, 65],
            prefix_lens=[10, 20, 30],
            cache_chunk_size=64,
            conv_state_len=4,
            is_mamba2=is_mamba2,
        )

    def test_gdn_and_mamba2_offsets(self):
        gdn = self._plan(is_mamba2=False)
        mamba2 = self._plan(is_mamba2=True)

        for plan in (gdn, mamba2):
            self.assertEqual(plan.rows, (1, 2))
            self.assertEqual(plan.aligned_rows, (1,))
            self.assertEqual(plan.unaligned_rows, (2,))
            self.assertEqual(
                plan.conv_indices,
                ((123, 124, 125, 126), (187, 188, 189, 190)),
            )
        self.assertEqual(gdn.h_src, (3,))
        self.assertEqual(mamba2.h_src, (2,))

    def test_all_false_and_invalid_metadata(self):
        plan = build_mamba_track_plan(
            track_mask=[False, False],
            track_seqlens=[-1, -1],
            extend_seq_lens=[64, 65],
            prefix_lens=[0, 0],
            cache_chunk_size=64,
            conv_state_len=4,
            is_mamba2=False,
        )
        self.assertEqual(plan.rows, ())
        self.assertEqual(plan.conv_indices, ())

        with self.assertRaisesRegex(ValueError, "cover every real extend row"):
            build_mamba_track_plan(
                track_mask=[True],
                track_seqlens=[64],
                extend_seq_lens=[64, 64],
                prefix_lens=[0, 0],
                cache_chunk_size=64,
                conv_state_len=4,
                is_mamba2=False,
            )

    def test_zero_length_conv_state_keeps_ssm_tracking(self):
        plan = build_mamba_track_plan(
            track_mask=[True],
            track_seqlens=[65],
            extend_seq_lens=[65],
            prefix_lens=[0],
            cache_chunk_size=64,
            conv_state_len=0,
            is_mamba2=False,
        )
        self.assertEqual(plan.rows, (0,))
        self.assertEqual(plan.unaligned_rows, (0,))
        self.assertEqual(plan.h_src, (1,))
        self.assertEqual(plan.conv_indices, ((),))

        backend = SimpleNamespace(device=torch.device("cpu"), conv_states_shape=(8, 0))
        forward_batch = SimpleNamespace(mamba_track_indices=torch.tensor([401]))
        result = MambaAttnBackendBase._init_track_ssm_indices(
            backend,
            torch.tensor([41]),
            forward_batch,
            plan,
        )
        self.assertEqual(result[0].shape, (1, 0))
        self.assertEqual(result[1].tolist(), [1])
        self.assertEqual(result[2].tolist(), [401])

    def test_device_materialization_uses_live_physical_slots(self):
        plan = self._plan(is_mamba2=False)
        backend = SimpleNamespace(device=torch.device("cpu"), conv_states_shape=(8, 4))
        forward_batch = SimpleNamespace(
            mamba_track_indices=torch.tensor([301, 401, 501, 999])
        )

        result = MambaAttnBackendBase._init_track_ssm_indices(
            backend,
            torch.tensor([31, 41, 51, 999]),
            forward_batch,
            plan,
        )
        (
            conv,
            h_src,
            h_dst,
            final_src,
            final_dst,
            rows,
            conv_dst,
        ) = result
        self.assertEqual(conv.tolist(), [[123, 124, 125, 126], [187, 188, 189, 190]])
        self.assertEqual(h_src.tolist(), [3])
        self.assertEqual(h_dst.tolist(), [501])
        self.assertEqual(final_src.tolist(), [41])
        self.assertEqual(final_dst.tolist(), [401])
        self.assertEqual(rows.tolist(), [1, 2])
        self.assertEqual(conv_dst.tolist(), [401, 501])

        # The host plan contains rows/offsets only. A later compaction therefore
        # changes gathered physical src/dst without rebuilding the plan.
        forward_batch.mamba_track_indices = torch.tensor([1301, 1401, 1501, 9999])
        remapped = MambaAttnBackendBase._init_track_ssm_indices(
            backend,
            torch.tensor([131, 141, 151, 9999]),
            forward_batch,
            plan,
        )
        self.assertEqual(remapped[2].tolist(), [1501])
        self.assertEqual(remapped[3].tolist(), [141])
        self.assertEqual(remapped[4].tolist(), [1401])
        self.assertEqual(remapped[6].tolist(), [1401, 1501])

    def test_all_false_forward_metadata_does_not_require_conv_shape(self):
        req_to_token_pool = SimpleNamespace(
            get_mamba_indices=lambda indices: indices.clone()
        )
        backend = SimpleNamespace(
            device=torch.device("cpu"),
            req_to_token_pool=req_to_token_pool,
            _translate_mamba_indices=lambda indices: indices,
            conv_states_shape=None,
        )
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.tensor([1]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1]),
            out_cache_loc=torch.tensor([0]),
            seq_lens_sum=1,
            extend_num_tokens=1,
            extend_seq_lens=torch.tensor([1], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([0]),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[1],
            extend_prefix_lens_cpu=[0],
            mamba_track_indices=torch.tensor([7]),
            mamba_track_mask=torch.tensor([False]),
            mamba_track_seqlens=torch.tensor([-1]),
            mamba_track_mask_cpu=[False],
            mamba_track_seqlens_cpu=[-1],
        )

        metadata = MambaAttnBackendBase._forward_metadata(backend, batch)

        self.assertFalse(metadata.has_mamba_track_mask)
        self.assertIsNone(metadata.track_conv_indices)

    def test_extend_device_mask_without_host_metadata_fails_fast(self):
        req_to_token_pool = SimpleNamespace(
            get_mamba_indices=lambda indices: indices.clone()
        )
        backend = SimpleNamespace(
            device=torch.device("cpu"),
            req_to_token_pool=req_to_token_pool,
            _translate_mamba_indices=lambda indices: indices,
            conv_states_shape=None,
        )
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.tensor([1]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1]),
            out_cache_loc=torch.tensor([0]),
            seq_lens_sum=1,
            extend_num_tokens=1,
            extend_seq_lens=torch.tensor([1], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([0]),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[1],
            extend_prefix_lens_cpu=[0],
            mamba_track_indices=torch.tensor([7]),
            mamba_track_mask=torch.tensor([False]),
            mamba_track_seqlens=torch.tensor([-1]),
        )

        with self.assertRaisesRegex(RuntimeError, "requires host mask/seqlen"):
            MambaAttnBackendBase._forward_metadata(backend, batch)


class TestMamba2TrackMetadataPropagation(unittest.TestCase):
    def _forward_metadata(self):
        return ForwardMetadata(
            query_start_loc=torch.tensor([0, 64], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([11]),
            mamba_track_indices=torch.tensor([21]),
            has_mamba_track_mask=True,
            mamba_track_mask_indices=torch.tensor([0]),
            conv_states_mask_indices=torch.tensor([21]),
        )

    def test_prepare_decode_preserves_track_rows(self):
        source = self._forward_metadata()
        metadata = Mamba2Metadata.prepare_decode(
            source,
            torch.tensor([1]),
            is_target_verify=False,
            draft_token_num=1,
        )
        self.assertIs(
            metadata.mamba_track_mask_indices, source.mamba_track_mask_indices
        )
        self.assertIs(
            metadata.conv_states_mask_indices, source.conv_states_mask_indices
        )

    def test_prepare_mixed_preserves_track_rows(self):
        source = self._forward_metadata()
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.zeros(64, dtype=torch.int64),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([64]),
            out_cache_loc=torch.arange(64),
            seq_lens_sum=64,
            extend_num_tokens=64,
            extend_seq_lens=torch.tensor([64], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens_cpu=[64],
            extend_prefix_lens_cpu=[0],
        )
        metadata = Mamba2Metadata.prepare_mixed(source, 64, batch)
        self.assertIs(
            metadata.mamba_track_mask_indices, source.mamba_track_mask_indices
        )
        self.assertIs(
            metadata.conv_states_mask_indices, source.conv_states_mask_indices
        )


if __name__ == "__main__":
    unittest.main()
