# SPDX-License-Identifier: Apache-2.0
"""Target sampling of an EAGLE verify on the vocabulary-sharded LM head output
(SGLANG_Q35MF_SHARDED_VERIFY): every rank reduces its [rows, V / tp] shard to one int64
key per row, an all-gather of the keys and a max pick the token. Greedy rows match
torch.argmax over the gathered FP32 logits bit for bit; temperature-only rows draw a
Gumbel-max sample, the same distribution as softmax(logits / T)."""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import print_info_once
from sglang.srt.environ import envs

_MC_GATHER = envs.SGLANG_PERFLL_VERIFY_MC_GATHER.get()


class VocabShard(msgspec.Struct, frozen=True):
    """Columns [0, valid) of this rank's logits are global token ids start + j."""

    start: int
    valid: int
    vocab_size: int


# key of (-inf, token id 0xFFFFFFFF): below every real column's key
_KEY_FLOOR = -2139095041 << 32


@triton.jit
def _shard_key_kernel(
    logits_ptr,
    stride_row,
    temp_ptr,
    seed_ptr,
    keys_ptr,
    start,
    valid,
    vocab_size,
    KEY_FLOOR: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    if HAS_TEMP:
        temp = tl.load(temp_ptr + row)
        inv_t = 1.0 / tl.maximum(temp, 1e-20)
        seed = tl.load(seed_ptr + row)
    for off in range(0, valid, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        gid = start + cols
        mask = (cols < valid) & (gid < vocab_size)
        x = tl.load(
            logits_ptr + row * stride_row + cols, mask=mask, other=float("-inf")
        )
        x = x.to(tl.float32)
        if HAS_TEMP:
            if temp > 0.0:
                u = tl.rand(seed, gid)
                x = x * inv_t - tl.log(-tl.log(u))
        # NaN never wins, as after sanitize_nan_logits; -0.0 ties +0.0 as in torch.argmax
        x = tl.where(x == x, x, float("-inf")) + 0.0
        bits = x.to(tl.int32, bitcast=True)
        # monotone in x as a signed int32; the low word prefers the smaller token id
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
        key = (ordered << 32) | (4294967295 - gid.to(tl.int64))
        key = tl.where(mask, key, KEY_FLOOR)
        best = tl.maximum(best, key)
    tl.store(keys_ptr + row, tl.max(best, axis=0))


def shard_keys(
    logits: torch.Tensor,
    shard: VocabShard,
    temperatures: Optional[torch.Tensor] = None,
    seeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One int64 key per row of this rank's shard: (order-preserving FP32 bits of the
    row's best value) << 32 | (0xFFFFFFFF - token id). temperatures: [rows] float32;
    rows at T <= 0 take the plain logit, rows at T > 0 logit / T + Gumbel noise drawn
    from seeds [rows] int64 and the global token id."""
    assert logits.stride(-1) == 1
    rows = logits.shape[0]
    keys = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows == 0:
        return keys
    has_temp = temperatures is not None
    _shard_key_kernel[(rows,)](
        logits,
        logits.stride(0),
        temperatures if has_temp else keys,
        seeds if has_temp else keys,
        keys,
        shard.start,
        shard.valid,
        shard.vocab_size,
        KEY_FLOOR=_KEY_FLOOR,
        HAS_TEMP=has_temp,
        BLOCK=2048,
        num_warps=8,
    )
    return keys


def keys_to_tokens(gathered: torch.Tensor, tp_size: int) -> torch.Tensor:
    """[tp * rows] keys of all ranks -> [rows] int64 token ids."""
    best = gathered.view(tp_size, -1).amax(dim=0)
    return 0xFFFFFFFF - (best & 0xFFFFFFFF)


# SGLANG_PERFLL_VERIFY_MC_GATHER: the key exchange's multimem state, built by the first
# (eager, every rank in lockstep) call; rows of 2 int64 = 8 bf16 per rank
_MC_MAX_ROWS = 1024
_mc_state = None


def _mc_gather_keys(keys: torch.Tensor, tp_group) -> torch.Tensor:
    """[rows] int64 keys of every rank -> [rows] best key, by one multimem all-gather."""
    global _mc_state
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    world, rows = tp_group.world_size, keys.shape[0]
    if _mc_state is None:
        _mc_state = triton_symm_mem_ag.create_state(
            tp_group.device_group,
            tp_group.rank_in_group,
            max_tokens=_MC_MAX_ROWS,
            hidden_size=8 * world,
            device=keys.device,
        )
    pair = torch.stack((keys, keys), dim=1)
    out = triton_symm_mem_ag.all_gather_inner(
        _mc_state, pair.view(torch.bfloat16), 8 * world, safe=False
    )
    return out.contiguous().view(torch.int64).view(rows, world, 2)[:, :, 0].amax(dim=1)


def sharded_target_tokens(
    logits: torch.Tensor,
    shard: VocabShard,
    tp_group,
    temperatures: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[rows] token ids over the whole vocabulary from every rank's logits shard."""
    print_info_once(
        "SGLANG_Q35MF_SHARDED_VERIFY: target "
        + ("Gumbel-max" if temperatures is not None else "argmax")
        + " on the vocabulary shards"
    )
    seeds = None
    if temperatures is not None:
        seeds = torch.randint(
            0, 2**31 - 1, (logits.shape[0],), dtype=torch.int64, device=logits.device
        )
    keys = shard_keys(logits, shard, temperatures, seeds)
    if _MC_GATHER and 0 < keys.shape[0] <= _MC_MAX_ROWS:
        return 0xFFFFFFFF - (_mc_gather_keys(keys, tp_group) & 0xFFFFFFFF)
    gathered = torch.empty(
        tp_group.world_size * keys.shape[0], dtype=torch.int64, device=keys.device
    )
    tp_group.all_gather_into_tensor(gathered, keys)
    return keys_to_tokens(gathered, tp_group.world_size)


def gather_full_logits(
    logits: torch.Tensor, shard: VocabShard, tp_group
) -> torch.Tensor:
    """The FP32 [rows, vocab_size] logits the stock verify builds: all-gather, slice
    to the vocabulary, cast."""
    print_info_once(
        "SGLANG_Q35MF_SHARDED_VERIFY: a batch needs the full logits, gathered"
    )
    tp, rows = tp_group.world_size, logits.shape[0]
    # all_gather_into_tensor, not all_gather: the latter allocates in symmetric
    # memory, which the post-verify graph capture cannot record
    out = torch.empty(
        (tp,) + tuple(logits.shape), dtype=logits.dtype, device=logits.device
    )
    tp_group.all_gather_into_tensor(out.view(-1), logits.contiguous().view(-1))
    full = out.movedim(0, 1).reshape(rows, -1)
    return full[:, : shard.vocab_size].float()
