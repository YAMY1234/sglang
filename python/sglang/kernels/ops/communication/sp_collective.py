"""K3 SP-MoE bf16 reduce-scatter and all-gather over MNNVL push memory."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


class Tuning(NamedTuple):
    num_blocks: int
    block_size: int


class Dispatch(NamedTuple):
    strategy: str
    tuning: Tuning


class FusionDispatch(NamedTuple):
    strategy: str
    max_blocks: int


# Safe seed values. GB300 production values are loaded by the layer glue from
# the checked-in JSON after the sweep settles.
DEFAULT_TUNING = Tuning(num_blocks=16, block_size=256)
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "sp_collective")
_TABLES: dict[str, Optional[dict]] = {}


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device).replace(" ", "_").replace("/", "_")


def _push(num_blocks: int, block_size: int) -> dict:
    return {"strategy": "push", "num_blocks": num_blocks, "block_size": block_size}


def _pull(num_blocks: int, block_size: int) -> dict:
    return {"strategy": "pull", "num_blocks": num_blocks, "block_size": block_size}


def _direct(num_blocks: int, block_size: int) -> dict:
    return {"strategy": "direct", "num_blocks": num_blocks, "block_size": block_size}


def _fused_direct(max_blocks: int) -> dict:
    return {"strategy": "fused_direct", "max_blocks": max_blocks}


_SEPARATE = {"strategy": "separate"}

# VR200 NVL72 (SM107), keyed by (world_size, hidden_size); used when no JSON
# names the device. TP8 over two compute trays, bf16, each candidate checked
# against NCCL 2.30 (NVLS off, as served) and taken only where strictly faster.
# Global-token buckets 1024-32768 were measured (us, NCCL -> chosen):
#   RS  104.5->38.7, 162.4->82.4, 201.6->132.6, 263.6->238.4, 517.1->453.1
#   AG  99.6->45.9, 167.1->79.5, 189.9->125.3, 284.7->232.0, 526.6->435.0
#   shard attention-residual read + AG, nvb 7, NCCL -> fused direct AG:
#       128.7->41.1, 173.8->75.0, 206.9->128.3, 323.8->229.2, 579.5->449.5
# The fused pull RS + FFN-side read did not beat the separate path (it lost
# from 16384 rows), so it stays separate. Fewer rows take the 1024 bucket, as
# the GB300 tables' push RS / direct AG do there.
_SM107_TABLES: dict[tuple[int, int], dict] = {
    (8, 7168): {
        "configs": {
            "reduce_scatter": {
                "1024": _push(32, 512),
                "4096": _pull(128, 512),
                "8192": _pull(192, 1024),
                "16384": _pull(128, 1024),
                "32768": _pull(192, 512),
            },
            "all_gather": {
                "1024": _direct(2, 1024),
                "4096": _direct(4, 512),
                "8192": _direct(64, 1024),
                "16384": _direct(32, 512),
                "32768": _direct(32, 1024),
            },
            "reduce_scatter_attn_res": {"1024": _SEPARATE},
            "attn_res_all_gather": {
                "1024": _fused_direct(128),
                "4096": _fused_direct(64),
                "8192": _fused_direct(128),
                "16384": _fused_direct(64),
                "32768": _fused_direct(128),
            },
        }
    },
    # TP4 on one compute tray (decode with attention DP 4), buckets 256-1280
    # (eager, us, NCCL -> chosen): RS 35.4->11.2, 65.9->14.1, 64.2->17.0,
    # 67.2->19.8, 364.6->24.7; AG 31.8->11.8, 65.4->15.2, 66.3->19.4,
    # 66.1->23.3, 67.5->28.3; nvb 7 read + AG 31.9->19.3, 70.4->22.8,
    # 70.3->22.6, 76.1->26.4, 75.4->29.6. Unmeasured sizes from 2048 keep NCCL.
    (4, 7168): {
        "configs": {
            "reduce_scatter": {
                "256": _push(128, 512),
                "512": _push(64, 512),
                "768": _push(128, 512),
                "1024": _push(128, 512),
                "1280": _push(128, 512),
                "2048": {"strategy": "nccl"},
            },
            "all_gather": {
                "256": {"strategy": "push", "num_blocks": 128, "block_size": 512},
                "512": {"strategy": "push", "num_blocks": 128, "block_size": 512},
                "768": {"strategy": "push", "num_blocks": 128, "block_size": 512},
                "1024": {"strategy": "push", "num_blocks": 128, "block_size": 512},
                "1280": {"strategy": "push", "num_blocks": 128, "block_size": 512},
                "2048": {"strategy": "nccl"},
            },
            "reduce_scatter_attn_res": {"256": _SEPARATE},
            "attn_res_all_gather": {
                "256": _fused_direct(64),
                "512": _fused_direct(64),
                "768": _fused_direct(128),
                "1024": _fused_direct(64),
                "1280": _fused_direct(64),
                "2048": _SEPARATE,
            },
        }
    },
}


def _table(world_size: int, hidden_size: int, device: torch.device) -> Optional[dict]:
    path = os.path.join(
        _CONFIG_DIR,
        (f"world={world_size},H={hidden_size},device_name={_device_name(device)}.json"),
    )
    if path not in _TABLES:
        if os.path.exists(path):
            with open(path) as f:
                _TABLES[path] = json.load(f)
        elif tuple(torch.cuda.get_device_capability(device)) == (10, 7):
            _TABLES[path] = _SM107_TABLES.get((world_size, hidden_size))
        else:
            _TABLES[path] = None
    return _TABLES[path]


def get_dispatch(
    kind: str,
    world_size: int,
    hidden_size: int,
    num_tokens: int,
    device: torch.device,
    *,
    element_size: int = 2,
    max_push_size: Optional[int] = None,
) -> Optional[Dispatch]:
    """Return a tuned strategy that fits, or None when NCCL should run."""
    table = _table(world_size, hidden_size, device)
    if table is None:
        return None
    raw_configs = table["configs"].get(kind)
    if raw_configs is None:
        return None
    configs = {int(k): v for k, v in raw_configs.items()}
    buckets = sorted(configs)
    bucket_index = 0
    for index, candidate in enumerate(buckets):
        if candidate <= num_tokens:
            bucket_index = index
        else:
            break
    config = configs[buckets[bucket_index]]
    if (
        max_push_size is not None
        and config["strategy"] == "push"
        and num_tokens * hidden_size * element_size // world_size > max_push_size
    ):
        # The selected push bucket does not fit. Advance to the next measured
        # operation-specific fallback rather than immediately using NCCL.
        fallback_strategy = {
            "reduce_scatter": "pull",
            "all_gather": "direct",
        }.get(kind)
        if fallback_strategy is None:
            return None
        for candidate in buckets[bucket_index + 1 :]:
            config = configs[candidate]
            if config["strategy"] == "nccl":
                return None
            if config["strategy"] == fallback_strategy:
                break
        else:
            return None
    if config["strategy"] == "nccl":
        return None
    tuning = Tuning(config["num_blocks"], config["block_size"])
    override = _PUSH_OVERRIDES.get((kind, world_size, hidden_size))
    if (
        config["strategy"] == "push"
        and override is not None
        and override[0] <= num_tokens < override[1]
    ):
        tuning = override[2]
    return Dispatch(config["strategy"], tuning)


# (kind, world_size, hidden_size) -> (min_tokens, end_tokens, Tuning) replacing a push
# entry's grid; the reduction order does not depend on the grid, so results stay bitwise.
_PUSH_OVERRIDES: dict[tuple[str, int, int], tuple[int, int, Tuning]] = {}


def override_push_tuning(
    kind: str,
    world_size: int,
    hidden_size: int,
    min_tokens: int,
    end_tokens: int,
    tuning: Tuning,
) -> None:
    _PUSH_OVERRIDES[(kind, world_size, hidden_size)] = (min_tokens, end_tokens, tuning)


def get_fusion_dispatch(
    kind: str,
    world_size: int,
    hidden_size: int,
    num_tokens: int,
    device: torch.device,
) -> Optional[FusionDispatch]:
    """Return a measured fused strategy, or None for the separate path."""
    table = _table(world_size, hidden_size, device)
    if table is None:
        return None
    raw_configs = table["configs"].get(kind)
    if raw_configs is None:
        return None
    configs = {int(k): v for k, v in raw_configs.items()}
    bucket = min(configs)
    for candidate in sorted(configs):
        if candidate <= num_tokens:
            bucket = candidate
        else:
            break
    config = configs[bucket]
    if config["strategy"] == "separate":
        return None
    return FusionDispatch(config["strategy"], config["max_blocks"])


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    cls = f"SPCollectiveKernel<{args}>"
    return load_jit(
        "kimi_k3_sp_collective",
        *args,
        cuda_files=["kimi_k3/comm/sp_collective.cuh"],
        cuda_wrappers=[
            ("reduce_scatter_res", f"{cls}::reduce_scatter_res"),
            ("reduce_scatter_pull", f"{cls}::reduce_scatter_pull"),
            ("all_gather", f"{cls}::all_gather"),
            ("all_gather_direct", f"{cls}::all_gather_direct"),
        ],
        extra_cuda_cflags=["-O3"],
    )


_COMM_MAP: dict[int, Communicator] = {}


def register_comm(comm: Communicator) -> None:
    # One communicator per world_size per process -- see the note in
    # communication/all_reduce_residual.py::register_comm. The ops key only on world_size, so an
    # overwrite here would hand the old group's callers the new group's peer
    # pointers.
    prev = _COMM_MAP.get(comm.world_size)
    assert prev is None or prev is comm, (
        f"a different communicator is already registered for world_size="
        f"{comm.world_size}"
    )
    _COMM_MAP[comm.world_size] = comm


@register_custom_op(mutates_args=["output"])
def _reduce_scatter_res_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor],
    residual_is_local: bool,
    num_blocks: int,
    block_size: int,
) -> None:
    _jit_module(world_size).reduce_scatter_res(
        _COMM_MAP[world_size],
        input.view(-1),
        output.view(-1),
        None if residual is None else residual.view(-1),
        residual_is_local,
        num_blocks,
        block_size,
    )


@register_custom_op(mutates_args=["output"])
def _reduce_scatter_pull_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor],
    residual_is_local: bool,
    input_mc_ptr: int,
    num_blocks: int,
    block_size: int,
) -> None:
    _jit_module(world_size).reduce_scatter_pull(
        _COMM_MAP[world_size],
        input.view(-1),
        output.view(-1),
        None if residual is None else residual.view(-1),
        residual_is_local,
        input_mc_ptr,
        num_blocks,
        block_size,
    )


@register_custom_op(mutates_args=["output"])
def _all_gather_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    num_blocks: int,
    block_size: int,
) -> None:
    _jit_module(world_size).all_gather(
        _COMM_MAP[world_size],
        input.view(-1),
        output.view(-1),
        num_blocks,
        block_size,
    )


@register_custom_op(mutates_args=["output"])
def _all_gather_direct_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    output_mc_ptr: int,
    num_blocks: int,
    block_size: int,
) -> None:
    _jit_module(world_size).all_gather_direct(
        _COMM_MAP[world_size],
        input.view(-1),
        output.view(-1),
        output_mc_ptr,
        num_blocks,
        block_size,
    )


def reduce_scatter_res(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    tuning: Tuning = DEFAULT_TUNING,
) -> torch.Tensor:
    residual_is_local = residual is not None and residual.numel() == output.numel()
    _reduce_scatter_res_op(
        world_size,
        input,
        output,
        residual,
        residual_is_local,
        tuning.num_blocks,
        tuning.block_size,
    )
    return output


def reduce_scatter_pull(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    input_mc_ptr: int,
    tuning: Tuning = DEFAULT_TUNING,
) -> torch.Tensor:
    residual_is_local = residual is not None and residual.numel() == output.numel()
    _reduce_scatter_pull_op(
        world_size,
        input,
        output,
        residual,
        residual_is_local,
        input_mc_ptr,
        tuning.num_blocks,
        tuning.block_size,
    )
    return output


def all_gather(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    tuning: Tuning = DEFAULT_TUNING,
) -> torch.Tensor:
    _all_gather_op(
        world_size,
        input,
        output,
        tuning.num_blocks,
        tuning.block_size,
    )
    return output


def all_gather_direct(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    output_mc_ptr: int,
    tuning: Tuning = DEFAULT_TUNING,
) -> torch.Tensor:
    _all_gather_direct_op(
        world_size,
        input,
        output,
        output_mc_ptr,
        tuning.num_blocks,
        tuning.block_size,
    )
    return output
