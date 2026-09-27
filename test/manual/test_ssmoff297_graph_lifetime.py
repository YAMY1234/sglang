"""Reproduce the short-tile dangling index; verify ownership after cache churn."""
import gc
import json
import os
import weakref

import torch
from sglang.kernels.ops.attention.fla.index import (
    prepare_lens, prepare_chunk_indices, prepare_chunk_offsets,
)
from sglang.srt.mem_cache.gdn_prefill_block_graph import pin_chunk_indices

device = os.environ.get('REPLAY_TEST_DEVICE', 'cpu')


def churn():
    for n in range(10):
        cu = torch.tensor([0, 64+n], device=device, dtype=torch.int32)
        prepare_lens(cu)
        prepare_chunk_offsets(cu, 64)
        for size in (16, 32, 64):
            prepare_chunk_indices(cu, size)
    gc.collect()


rows = []
for tokens in (1, 16, 17, 32, 33, 64, 256, 8192, 32768):
    cu = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    size = min(64, max(16, 1 << (tokens-1).bit_length()))
    old = (prepare_lens(cu), prepare_chunk_indices(cu, 64),
           prepare_chunk_offsets(cu, 64))
    ref = weakref.ref(prepare_chunk_indices(cu, size))
    churn()
    old_missing = ref() is None
    assert old_missing == (size != 64), (tokens, old_missing)
    owned = pin_chunk_indices(cu, tokens)
    ref = weakref.ref(prepare_chunk_indices(cu, size))
    before = ref().clone()
    churn()
    assert ref() is not None and torch.equal(ref(), before)
    assert any(x is ref() for x in owned)
    rows.append(dict(tokens=tokens, output_tile=size, old_missing=old_missing,
                     new_retained=True, bitwise=True))
    del owned, old, before
print(json.dumps(dict(complete=True, passed=True, device=device, cases=rows,
    scope='Index allocation lifetime; old missing reference reproduced without unsafe replay')))
