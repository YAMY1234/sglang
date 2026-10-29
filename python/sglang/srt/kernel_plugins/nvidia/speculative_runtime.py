# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral defaults for speculative runtime hooks.

The runtime owns verification and all cross-rank exchange. Providers may replace
only the pure local-key policy. Missing providers preserve the exact default by
identity.
"""

from __future__ import annotations

import torch
from sglang.kernels.plugin_kernels.nvidia.spec import (
    LocalKeysFn,
    PostVerifyIO,
    PostVerifyOutput,
    PostVerifyStrategy,
    ShardedLogitsView,
    SpeculativeRuntimeHooks,
)


def default_local_keys(view: ShardedLogitsView) -> torch.Tensor | None:
    """Build local keys, or decline before any fallback RNG is consumed."""
    from sglang.srt.speculative.sharded_verify import shard_keys

    # Temperature sampling is an explicit provider numeric policy.  The stock
    # default never selects the distribution-equivalent Gumbel path, even if a
    # caller happens to supply seed material.
    if view.temperatures is not None:
        return None
    return shard_keys(
        view.values,
        view.shard,
        temperatures=view.temperatures,
        seeds=view.seeds,
        splits=view.splits,
    )


def default_post_verify(
    io: PostVerifyIO,
    local_keys: LocalKeysFn,
) -> PostVerifyOutput:
    """Run the existing verifier and preserve its outputs and mutation order."""
    from sglang.srt.speculative.eagle_utils import eagle_sample

    token_ids, accepted_lengths, accepted_indices = eagle_sample(
        io.verify_tree,
        io,
        io.logits,
        io.grammar_mask,
        uno_target_max_top_k=io.sparse_target_max_top_k,
        watermark_state=io.watermark,
        watermark_full_mask=io.watermark_full_mask,
        simulated_acc_len=io.simulated_accept_length,
        simulated_acc_len_rng=io.simulated_accept_length_rng,
        sparse_bonus_argmax=io.sparse_bonus_argmax,
        local_keys=local_keys,
    )
    return PostVerifyOutput(
        token_ids=token_ids,
        accepted_lengths=accepted_lengths,
        accepted_indices=accepted_indices,
    )


def default_speculative_runtime_hooks() -> SpeculativeRuntimeHooks:
    """Build the stock verification strategy."""
    return SpeculativeRuntimeHooks(
        post_verify=PostVerifyStrategy(
            run=default_post_verify,
            local_keys=default_local_keys,
        ),
    )
