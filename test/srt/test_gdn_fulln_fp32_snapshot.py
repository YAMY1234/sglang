"""docs/223 §11 (A') gate (a'): which h-tracked rows take the fp32 track_state snapshot (CPU).

Drives the real ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend (test_gdn_fulln_checkpoint helpers) and the
real MambaAttnBackendBase._init_fp32_track_rows / _init_track_ssm_indices. The mask must be exactly the h-tracked rows
whose donated depth is the forward's end minus one token: for the fixed full-N route that is P % 64 == 1 (final chunk),
never a legacy N-1 forward (it donates its final state there), never a non-final chunk, and off without the switch.
The mask is computed from declared ForwardBatch fields inside init_forward_metadata, so no ForwardBatch attribute has to
survive EagerRunner.load_batch / PrefillCudaGraphRunner.load_batch (the P3 N = 65 root cause, docs/223 §11).
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gdn_fulln_checkpoint as tc  # noqa: E402
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase  # noqa: E402

CHUNK = tc.CHUNK
SWITCH = "SGLANG_GDN_FULLN_CKPT_FP32_SNAPSHOT"


def fp32_rows(entry, prefix, forward_len):
    fb = SimpleNamespace(mamba_track_mask=torch.tensor([bool(entry.track_mask)]),
                         extend_seq_lens=torch.tensor([forward_len]), mamba_track_indices=torch.tensor([7]),
                         mamba_track_seqlens=torch.tensor([entry.track_seqlen]), extend_prefix_lens=torch.tensor([prefix]))
    me = SimpleNamespace(mamba_chunk_size=CHUNK, device="cpu")
    rows, any_ = MambaAttnBackendBase._init_fp32_track_rows(me, fb)
    out = MambaAttnBackendBase._init_track_ssm_indices(me, torch.tensor([3]), fb)
    return rows, any_, out[1]  # mask, host flag, track_ssm_h_src


class FullNFp32Snapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tc._setup()

    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in (SWITCH, "SGLANG_GDN_FULLN_CKPT_N_MINUS_1")}
        os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = "1"

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _case(self, P, prefix=0, fulln=True, end=None):
        entry, depth = tc._track(list(range(P)), prefix, None, fulln, end=end)
        stop = P if end is None else end
        forward_len = stop - prefix - (0 if fulln or end is not None else 1)
        if not entry.track_mask:
            return None, depth
        return fp32_rows(entry, prefix, forward_len), depth

    def test_mask_is_exactly_the_p_minus_1_rows(self):
        os.environ[SWITCH] = "1"
        lengths = [65, 128, 129, 4096, 4097] + [prefix + ext + (r - (prefix + ext) % CHUNK) % CHUNK
                                                for prefix in (0, 65536) for ext in (130, 1500) for r in (0, 1, 33, 63)]
        for P in lengths:
            prefix = 65536 if P > 65536 else 0
            with self.subTest(P=P):
                got, depth = self._case(P, prefix)
                if got is None:
                    continue
                rows, any_, h_src = got
                expect = (P % CHUNK == 1)
                self.assertEqual(any_, expect)
                if expect:
                    self.assertEqual(rows.tolist(), [True])
                    self.assertEqual(h_src.numel(), 1)
                    self.assertEqual(depth, P - 1)
                else:
                    self.assertIsNone(rows)

    def test_legacy_and_non_final_never_masked(self):
        os.environ[SWITCH] = "1"
        for P in (65, 129, 4097, 1537):
            with self.subTest(P=P, route="legacy"):
                got, _ = self._case(P, fulln=False)
                if got is not None:
                    self.assertFalse(got[1])
        got, depth = self._case(4097, end=128)
        self.assertEqual(depth, 128)
        self.assertFalse(got[1])

    def test_switch_off_is_v9(self):
        os.environ[SWITCH] = "0"
        got, _ = self._case(129)
        self.assertIsNone(got[0])
        self.assertFalse(got[1])


if __name__ == "__main__":
    unittest.main()
