"""Boot-time pipeline layer partition from each GPU's measured speed (SGLANG_PP_PARTITION_AUTO).

Under a per-GPU power cap the same stage runs a few percent faster or slower depending on the GPU it lands on, and with
the stages decoupled the slowest (stage, GPU) pair sets the pipeline beat. Before the weights load, every rank measures
its GPU (a short concurrent GEMM + copy burn, or a fixed list for rigs and tests), the ranks all-gather the speeds, and
each runs the same deterministic search: the contiguous split that minimizes the slowest stage's cost x speed factor,
subject to every stage keeping at least the KV token capacity of the reference split (capacity is the minimum over the
stages). Rank 0's result is broadcast and every rank asserts it computed the same, then it replaces
SGLANG_PP_LAYER_PARTITION, which the model reads when it builds its layer range.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Callable, List, Optional, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)


class StageCostModel(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer forward cost (ms at the reference batch) and per-stage memory (bytes) of a hybrid model. The memory
    fields bound each stage's KV token capacity; dsa_kv_bytes_per_token = 0 turns the capacity constraint off."""

    dsa_ms: float
    kda_ms: float
    first_ms: float = 0.0
    last_ms: float = 0.0
    dsa_weight: float = 0.0
    kda_weight: float = 0.0
    first_weight: float = 0.0
    last_weight: float = 0.0
    kda_state: float = 0.0
    dsa_kv_bytes_per_token: float = 0.0

    @classmethod
    def from_json(cls, path: str) -> "StageCostModel":
        with open(path, "rb") as f:
            return msgspec.json.decode(f.read(), type=cls)


# Kimi-K3.1-Pro prefill, 16k-token chunks: layer costs fitted on five rig windows (three node pairs, P3 on);
# weights from the per-stage load logs; state for 300 Mamba slots per KDA layer; MLA fp8 + DSA indexer KV per token.
KIMI_K31_PREFILL = StageCostModel(
    dsa_ms=27.42,
    kda_ms=10.43,
    first_ms=5.45,
    last_ms=-16.87,
    dsa_weight=15.48e9,
    kda_weight=16.0e9,
    first_weight=-11.5e9,
    last_weight=7.8e9,
    kda_state=1.82e9,
    dsa_kv_bytes_per_token=708.0,
)


def stage_layers(is_dsa: Sequence[bool], start: int, n: int) -> Tuple[int, int]:
    nd = sum(is_dsa[start : start + n])
    return nd, n - nd


def stage_cost(m: StageCostModel, is_dsa: Sequence[bool], s: int, start: int, n: int, pp_size: int) -> float:
    nd, nk = stage_layers(is_dsa, start, n)
    return m.dsa_ms * nd + m.kda_ms * nk + (m.first_ms if s == 0 else 0.0) + (m.last_ms if s == pp_size - 1 else 0.0)


def stage_kv_capacity(
    m: StageCostModel, is_dsa: Sequence[bool], s: int, start: int, n: int, pp_size: int, budget: float
) -> float:
    """Tokens of KV the stage can hold after its weights and Mamba state (inf without KV layers or KV model)."""
    nd, nk = stage_layers(is_dsa, start, n)
    if nd == 0 or m.dsa_kv_bytes_per_token <= 0:
        return math.inf
    used = (
        m.dsa_weight * nd
        + m.kda_weight * nk
        + m.kda_state * nk
        + (m.first_weight if s == 0 else 0.0)
        + (m.last_weight if s == pp_size - 1 else 0.0)
    )
    return (budget - used) / (m.dsa_kv_bytes_per_token * nd)


def even_partition(num_layers: int, pp_size: int) -> Tuple[int, ...]:
    """get_pp_indices' default split: the last `num_layers % pp_size` stages take one extra layer."""
    base, extra = divmod(num_layers, pp_size)
    return tuple(base + (s >= pp_size - extra) for s in range(pp_size))


def min_capacity(
    m: StageCostModel, is_dsa: Sequence[bool], part: Sequence[int], budgets: Sequence[float]
) -> float:
    starts = [sum(part[:s]) for s in range(len(part))]
    return min(stage_kv_capacity(m, is_dsa, s, starts[s], part[s], len(part), budgets[s]) for s in range(len(part)))


def best_partition(
    is_dsa: Sequence[bool],
    speed: Sequence[float],
    m: StageCostModel,
    budgets: Optional[Sequence[float]] = None,
    min_tokens: float = 0.0,
    max_layers: int = 16,
) -> Tuple[Tuple[int, ...], float]:
    """Contiguous split of len(is_dsa) layers over len(speed) stages minimizing max_s cost_s * speed_s, every stage
    keeping >= min_tokens of KV capacity under budgets (bytes per stage; None = no constraint). Ties go to the
    smallest running max, then the smallest prefix, so the result depends only on the inputs. Returns (split, its max
    cost)."""
    L, P = len(is_dsa), len(speed)
    hybrid = not all(is_dsa)
    INF = (math.inf, ())
    # best[s][j]: (max cost, split) placing the first j layers on stages 0..s.
    best: List[List[Tuple[float, Tuple[int, ...]]]] = [[INF] * (L + 1) for _ in range(P)]

    def ok(s: int, start: int, n: int) -> bool:
        # The hybrid HiCache host split sizes a KV and a Mamba pool per stage and fails on an empty one: every stage
        # keeps a full-attention layer, and in a hybrid model a linear-attention layer too.
        nd, nk = stage_layers(is_dsa, start, n)
        if nd == 0 or (hybrid and nk == 0):
            return False
        return budgets is None or stage_kv_capacity(m, is_dsa, s, start, n, P, budgets[s]) >= min_tokens

    for j in range(1, min(max_layers, L) + 1):
        if ok(0, 0, j):
            best[0][j] = (round(stage_cost(m, is_dsa, 0, 0, j, P) * speed[0], 6), (j,))
    for s in range(1, P):
        for j in range(s + 1, L + 1):
            for n in range(1, min(max_layers, j - s) + 1):
                prev = best[s - 1][j - n]
                if prev[0] == math.inf or not ok(s, j - n, n):
                    continue
                cand = (max(prev[0], round(stage_cost(m, is_dsa, s, j - n, n, P) * speed[s], 6)), prev[1] + (n,))
                if cand < best[s][j]:
                    best[s][j] = cand
    value, part = best[P - 1][L]
    if not part:
        raise RuntimeError(f"no PP split of {L} layers over {P} stages meets the KV capacity floor {min_tokens:.0f}")
    return part, value


def burn_unit_ms(seconds: float, device) -> float:
    """Mean time of one burn unit (4 bf16 GEMMs at a prefill shape + a 1 GiB copy) over the second half of a burn of
    `seconds`; callers start it on every GPU at once so each runs at its loaded, power-capped clock."""
    import torch

    a = torch.randn(16384, 7168, device=device, dtype=torch.bfloat16)
    b = torch.randn(7168, 4096, device=device, dtype=torch.bfloat16)
    c = torch.empty(16384, 4096, device=device, dtype=torch.bfloat16)
    x = torch.empty(512 << 20, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    times, t0 = [], time.time()
    while time.time() < t0 + seconds:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.time()
        start.record()
        for _ in range(4):
            torch.matmul(a, b, out=c)
        y.copy_(x)
        end.record()
        end.synchronize()
        times.append((wall, start.elapsed_time(end)))
    half = [ms for w, ms in times if w >= t0 + seconds / 2] or [ms for _, ms in times]
    del a, b, c, x, y
    torch.cuda.empty_cache()
    return sum(half) / len(half)


def resolve(
    *,
    is_dsa: Sequence[bool],
    pp_rank: int,
    pp_size: int,
    world_rank: int,
    all_gather: Callable[[object], List[object]],
    broadcast_from_0: Callable[[object], object],
    measure: Callable[[], float],
    budget: float,
    m: StageCostModel,
    reference: Tuple[int, ...],
    fixed_speed: Sequence[float] = (),
    kv_floor: float = 1.0,
) -> Tuple[Tuple[int, ...], dict]:
    """Collective over the world group: gather (pp_rank, speed, budget) from every rank, take each stage's slowest
    rank, search the split, broadcast rank 0's, assert agreement. Returns the split and a report."""
    speed = float(fixed_speed[pp_rank]) if fixed_speed else measure()
    rows = all_gather((pp_rank, speed, budget))
    stage_speed = [max(v for r, v, _ in rows if r == s) for s in range(pp_size)]
    budgets = [min(b for r, _, b in rows if r == s) for s in range(pp_size)]
    mean = sum(stage_speed) / pp_size
    factors = [v / mean for v in stage_speed]
    # KV token capacity is the minimum over the stages; keep it >= kv_floor x the reference split's.
    floor = kv_floor * min_capacity(m, is_dsa, reference, budgets) if m.dsa_kv_bytes_per_token > 0 else 0.0
    part, value = best_partition(is_dsa, factors, m, budgets if floor > 0 else None, floor)
    chosen = broadcast_from_0(part)
    assert tuple(chosen) == part, f"rank {world_rank}: PP split {part} differs from rank 0's {tuple(chosen)}"
    ref_value = max(
        stage_cost(m, is_dsa, s, sum(reference[:s]), reference[s], pp_size) * factors[s] for s in range(pp_size)
    )
    return part, {
        "factors": [round(x, 4) for x in factors],
        "source": "fixed" if fixed_speed else "burn",
        "reference": list(reference),
        "reference_max_ms": round(ref_value, 2),
        "split": list(part),
        "split_max_ms": round(value, 2),
        "kv_floor_tokens": None if floor <= 0 else int(floor),
        "model": msgspec.structs.asdict(m),
    }


def maybe_apply(*, model_config, is_draft_worker: bool, mem_fraction_static: float) -> None:
    """SGLANG_PP_PARTITION_AUTO: choose the split before the target model's weights load and set
    SGLANG_PP_LAYER_PARTITION for this process (every rank sets the same value). Hybrid KDA/DSA configs only."""
    import torch

    from sglang.srt.distributed.parallel_state import get_world_group
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    par = get_parallel()
    if par.pp_size <= 1 or is_draft_worker or not envs.SGLANG_PP_PARTITION_AUTO.get():
        return
    num_layers = model_config.num_hidden_layers
    is_dsa = [not model_config.hf_text_config.is_kda_layer(i) for i in range(num_layers)]
    path = envs.SGLANG_PP_PARTITION_AUTO_MODEL.get()
    m = StageCostModel.from_json(path) if path else KIMI_K31_PREFILL
    env_part = envs.SGLANG_PP_LAYER_PARTITION.get()
    reference = tuple(int(x) for x in env_part.split(",")) if env_part else even_partition(num_layers, par.pp_size)
    world = get_world_group()
    budget = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory * mem_fraction_static
    burn_s = envs.SGLANG_PP_PARTITION_AUTO_BURN_S.get()

    def gather(obj):
        out = [None] * world.world_size
        torch.distributed.all_gather_object(out, obj, group=world.cpu_group)
        return out

    def bcast(obj):
        box = [obj]
        torch.distributed.broadcast_object_list(box, src=world.ranks[0], group=world.cpu_group)
        return box[0]

    def measure():
        torch.distributed.barrier(group=world.cpu_group)  # every GPU burns at the same time
        return burn_unit_ms(burn_s, torch.device("cuda", torch.cuda.current_device()))

    t0 = time.time()
    part, report = resolve(
        is_dsa=is_dsa,
        pp_rank=par.pp_rank,
        pp_size=par.pp_size,
        world_rank=world.rank_in_group,
        all_gather=gather,
        broadcast_from_0=bcast,
        measure=measure,
        budget=budget,
        m=m,
        reference=reference,
        fixed_speed=[float(x) for x in envs.SGLANG_PP_PARTITION_AUTO_SPEED.get()],
        kv_floor=envs.SGLANG_PP_PARTITION_AUTO_KV_FLOOR.get(),
    )
    envs.SGLANG_PP_LAYER_PARTITION.set(",".join(map(str, part)))
    report["seconds"] = round(time.time() - t0, 1)
    if world.rank_in_group == 0:
        logger.info("PP partition auto (SGLANG_PP_PARTITION_AUTO): %s", json.dumps(report))
