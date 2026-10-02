"""P31 raw boundary and r8/W8 factors survive the real staging copy on CPU.

This tests byte transport and local receive authority, not model needle recall.
The boundary lengths include the two 1151-token needle requests from the
native-release incident, a partial page and the 32K prefill boundary.
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")
import unittest
from types import SimpleNamespace as NS

import numpy as np
import torch

from sglang.srt.disaggregation.flashnext_staging import Catalog, Entry
from sglang.srt.disaggregation.flashnext_staging_kernels import copy_payload
from sglang.srt.disaggregation.flashnext_staging_manifest import Manifest
from sglang.srt.disaggregation.state_handoff import FactorStateHandoff
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool


def peer(seed):
    layers = [i for i in range(48) if i % 4 != 3]
    factor = FactoredGDNPool(
        size=3, cache_params=NS(shape=NS(temporal=(24, 128, 128))),
        mamba_layer_ids=layers, device="cpu",
        cfg=FactoredGDNConfig(r=8, m=8, ring=1, dtype=torch.float16, strict_chunk=True),
    )
    generator = torch.Generator().manual_seed(seed)
    for tensor in (factor.a, factor.U, factor.W):
        tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype) / 8)
    factor.count.copy_(torch.tensor([8 + int(l < 31) for l in layers])[:, None, None])
    raw = torch.randn(4, 10240, generator=generator).bfloat16()
    position = torch.full((4, 1), -1, dtype=torch.int64)
    valid = torch.zeros(4, 1, dtype=torch.int32)
    catalog = Catalog.__new__(Catalog)
    catalog.pool = NS()
    catalog.entries, catalog.by_key = [], {}
    for i, (name, tensor, axis, layer) in enumerate(factor.iter_transfer_state_entries()):
        catalog._add(Entry(layer, "mamba." + name + ".0", tensor, 0, i, 0, slice_axis=axis + 1))
    for name, tensor in (("pd_h31", raw), ("pd_boundary_position", position), ("pd_boundary_valid", valid)):
        catalog._add(Entry(4294967290, "mamba." + name + ".0", tensor, 0, len(catalog.entries), 0))
    return catalog, factor, raw, position, valid


class BoundaryWireTest(unittest.TestCase):
    def test_raw_boundary_factors_positions_and_validity_survive_receive(self):
        p, pf, ph, pp, pv = peer(1656)
        d, df, dh, dp, dv = peer(1657)
        request = NS(kv=NS(mamba_pool_idx=torch.tensor(3),
            mamba_ping_pong_track_buffer=None, mamba_last_track_idx=None,
            mamba_last_track_seqlen=None, mamba_cow_src_index=torch.tensor(2), mamba_needs_clear=True))
        receiver = FactorStateHandoff(df)
        # P sends raw final h31. The codec output for N-1 emitter inputs is not
        # substituted here, and receiving never recomputes U/W from h31.
        for generation, tokens in enumerate((1151, 1151, 65, 32769), 1):
            pp[1] = tokens - 1
            pv[1] = 1
            receiver.prepare_receive(request)
            manifest, source = p.source_payload(
                room=1656, generation=generation, source_rank=0, source_tp=2,
                prompt_tokens=tokens, token_start=0, token_end=tokens,
                kv_indices=np.array([], dtype=np.int64), kv_by_entry=None,
                state_indices=[np.array([1])], chunk_index=0, last_chunk=True, shallow_boundary=True)
            wire = Manifest.from_bytes(manifest.to_bytes())
            self.assertEqual((wire.shallow_count, wire.deep_count), (9, 8))
            self.assertEqual(len(wire.fields), 147)
            target = d.destination_payload(manifest=wire, kv_indices=np.array([], dtype=np.int64),
                state_indices=[np.array([3])], decode_prefix_tokens=0, destination_rank=0, destination_tp=2)
            staging = torch.empty(wire.nbytes, dtype=torch.uint8)
            copy_payload(manifest=manifest, local=source, staging=staging, gather=True)
            copy_payload(manifest=wire, local=target, staging=staging.clone(), gather=False)
            receiver.commit_receive(request)
            for field in wire.fields:
                a, b = source[field.key], target[field.key]
                self.assertTrue(torch.equal(a.tensor[a.rows].view(torch.uint8), b.tensor[b.rows].view(torch.uint8)), field.key)
            self.assertTrue(torch.equal(ph[1], dh[3]))
            self.assertEqual(dp[3].item(), tokens - 1)
            self.assertEqual(dv[3].item(), 1)
            self.assertEqual(df.stale[3].item(), 1)
            self.assertEqual(df.dense_of[3].item(), -1)
            self.assertFalse(request.kv.mamba_needs_clear)


if __name__ == "__main__":
    unittest.main()
