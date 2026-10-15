# SPDX-License-Identifier: Apache-2.0
"""Check the larger value tile across independent replay-state banks."""

import json

import torch

from bench_gdn_midbatch import compare, recurrent_case
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)


def main():
    cases = (
        (0, "normal"),
        (1, "normal"),
        (7, "wrap"),
        (9, "ragged"),
        (9, "noop"),
        (9, "flush"),
    )
    for batch in (4, 6, 8):
        for history, control in cases:
            stock = recurrent_case(batch, history, control)
            candidate = recurrent_case(batch, history, control)
            for case, tile in ((stock, 32), (candidate, 64)):
                gdn_replayssm_spec_decode(
                    **case, block_v=tile, num_warps=2, num_stages=2, nk=2
                )
            torch.cuda.synchronize()
            checks = compare(stock, candidate)
            print(
                json.dumps(
                    dict(batch=batch, history=history, control=control, tensors=checks)
                ),
                flush=True,
            )
            assert all(checks.values()), (batch, history, control, checks)
    print(json.dumps(dict(byte_checks="PASS", cases=18)), flush=True)


if __name__ == "__main__":
    main()
