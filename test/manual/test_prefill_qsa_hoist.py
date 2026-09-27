"""Exact packed QSA prefill locations, including long and shared-prefix shapes.

Run on CPU in the serving image before GPU admission, then on the allocated
GPU before loading services. This probe does not replace model numerical gates.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata, build_packed_token_locations, build_prefill_compressed_locs,
)


def probe(device):
    torch.manual_seed(298832)
    rows = []
    for ratio in (4, 8, 16, 128):
        for lengths in ([0, 3, 4, 9, 64, 131], [49152], [254000], [0, 0], []):
            batch = len(lengths)
            width = max(max(lengths, default=0), ratio)
            width = ((width + ratio - 1) // ratio) * ratio
            # Page-aligned, permuted, duplicate request IDs exercise ordering.
            requests = [3, 0, 3, 2, 1, 0][:batch]
            slots = torch.arange(4 * width, device=device).reshape(4, width).int()
            req = torch.tensor(requests, device=device, dtype=torch.long)
            table = slots.index_select(0, req)
            seq = torch.tensor(lengths, device=device, dtype=torch.int32)
            got = build_prefill_compressed_locs(table, seq, lengths, ratio)
            pieces = [table[i, :n // ratio * ratio:ratio].long() // ratio for i, n in enumerate(lengths)]
            expected = torch.cat(pieces) if pieces else got.new_empty(0)
            assert torch.equal(got, expected)
            packed = build_packed_token_locations(slots, req, seq, lengths)
            pieces = [slots[r, :n].long() for r, n in zip(requests, lengths)]
            expected = torch.cat(pieces) if pieces else packed.new_empty(0)
            assert torch.equal(packed, expected)
            blocks = 4 * width // ratio
            buffer = torch.randn(blocks, 1, 8, device=device)
            for prefix in (False, True):
                extend = [min(n, 7) if prefix else n for n in lengths]
                ids = torch.repeat_interleave(torch.arange(batch, device=device, dtype=torch.int32), torch.tensor(extend, device=device, dtype=torch.long), output_size=sum(extend))
                pieces = [torch.arange(n - e, n, device=device) for n, e in zip(lengths, extend)]
                positions = torch.cat(pieces) if pieces else seq.new_empty(0)
                if positions.numel():
                    assert int(positions.max()) == max(lengths) - 1
                for translate in (False, True):
                    pool = SimpleNamespace(get_qsa_compressed_k_buffer=lambda layer: buffer, qsa_index_kv_heads=1, qsa_index_head_dim=8)
                    if translate:
                        pool.physical_page_map = True
                        pool.translate_locations = lambda layer, locs, compressed=False: (locs * 7 + layer) % blocks
                    kwargs = dict(sequence_lengths=seq, token_to_batch_idx=ids, token_slot_table=table,
                                  out_cache_loc=seq.new_empty(0), token_to_kv_pool=pool, compress_ratio=ratio, block_topk=16)
                    ref = QSAIndexerMetadata(**kwargs).get_prefill_mqa_inputs(3, positions)
                    out = QSAIndexerMetadata(**kwargs, prefill_compressed_locs=got).get_prefill_mqa_inputs(3, positions)
                    for a, b in zip(ref, out):
                        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
                    rows.append(dict(ratio=ratio, lengths=lengths, prefix=prefix, physical_translation=translate, bitwise=True))
    return dict(complete=True, passed=True, device=device, cases=len(rows), rows=rows,
                scope="Prefill metadata only; model outputs and performance require independent gates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.device)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
