"""Shadow immutable factors on exact dense model-generated KDA input trajectories.

The shadow never changes model outputs or its FP32 recurrent state. Every KDA
layer/head is covered. This measures state/readout drift, not model NLL drift.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch


def install():
    from safetensors import safe_open
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    from sglang.srt.layers.attention.linear import kda_backend as backend
    from sglang.srt.mem_cache.kda_factored_pool import (
        FactoredKDAConfig,
        FactoredKDAPool,
    )

    root = Path(os.environ["KDA_DRIFT_ROOT"])
    original_forward = backend.KDAAttnBackend.forward_decode
    original_conv = backend.causal_conv1d_update
    active = []
    layers = {}
    sink = None

    def conv(*args, **kwargs):
        result = original_conv(*args, **kwargs)
        if active:
            active[-1]["qkv"] = result
        return result

    def forward(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        nonlocal sink
        if not (root / "armed").exists():
            return original_forward(
                self, layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        assert getattr(self.req_to_token_pool, "factored_kda_pool", None) is None
        state = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id).temporal
        assert state.dtype == torch.float32
        indices = self.forward_metadata.mamba_cache_indices
        ids = indices.cpu().tolist()
        assert ids and min(ids) >= 0 and len(ids) == len(set(ids))
        rank = get_tensor_model_parallel_rank()
        if sink is None:
            with safe_open(
                os.environ["KDA_DRIFT_COMPONENTS"], framework="pt", device="cpu"
            ) as f:
                sink = f.get_tensor("state.sink_dir")
        if layer.layer_id not in layers:
            heads, values, keys = state.shape[-3:]
            params = SimpleNamespace(
                shape=SimpleNamespace(temporal=(heads, values, keys), conv=[])
            )
            pool = FactoredKDAPool(
                state.shape[0] - 1,
                params,
                [layer.layer_id],
                state.device,
                FactoredKDAConfig(32, 32, "bf16"),
            )
            pool.bind_directions(
                {
                    layer.layer_id: sink[
                        layer.layer_id, rank * heads : (rank + 1) * heads
                    ].to(state.device)
                }
            )
            directory = root / f"tp{rank}" / f"layer{layer.layer_id}"
            directory.mkdir(parents=True, exist_ok=True)
            layers[layer.layer_id] = {
                "pool": pool,
                "seen": set(),
                "counts": {},
                "directory": directory,
                "metrics": torch.zeros(state.shape[0], 1024, 4, device=state.device),
                "initial": {},
            }
        info = layers[layer.layer_id]
        fresh = [x for x in ids if x >= 0 and x not in info["seen"]]
        if fresh:
            idx = torch.tensor(fresh, device=state.device, dtype=indices.dtype)
            info["pool"].commit(layer.layer_id, idx, state[idx].clone())
            before = info["pool"].dense(layer.layer_id, idx)
            relative = (
                (before - state[idx]).square().sum((1, 2, 3))
                / state[idx].square().sum((1, 2, 3)).clamp_min(1e-30)
            ).sqrt()
            info["initial"].update(zip(fresh, relative.cpu().tolist()))
            info["seen"].update(fresh)
            info["counts"].update({x: 0 for x in fresh})
        frame = {}
        active.append(frame)
        try:
            result = original_forward(
                self, layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        finally:
            active.pop()
        assert "qkv" in frame, "native path bypassed post-conv capture"
        shadow = info["pool"].decode(layer, frame["qkv"], a, b, indices)
        dense_shadow = info["pool"].dense(layer.layer_id, indices)
        exact = state[indices.long()]
        output = result.reshape_as(shadow)
        difference = (shadow.float() - output.float()).reshape(len(ids), -1)
        reference = output.float().reshape(len(ids), -1)
        values = torch.stack(
            (
                (dense_shadow - exact).square().flatten(1).sum(1),
                exact.square().flatten(1).sum(1),
                difference.square().sum(1),
                reference.square().sum(1),
            ),
            1,
        )
        positions = torch.tensor(
            [info["counts"].get(x, 0) for x in ids], device=state.device
        )
        assert positions.max().item() < 1024, "more than 1024 decode steps"
        info["metrics"][indices.long(), positions] = values
        for x in ids:
            info["counts"][x] += 1
        if len(info["counts"]) == 48 and min(info["counts"].values()) == 1024:
            torch.save(
                {
                    "slots": sorted(info["seen"]),
                    "metrics": info["metrics"].cpu(),
                    "initial_state_relative_error": info["initial"],
                    "columns": [
                        "state_error_sq",
                        "state_reference_sq",
                        "output_error_sq",
                        "output_reference_sq",
                    ],
                },
                info["directory"] / "metrics.pt",
            )
            (info["directory"] / "complete.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "layer": layer.layer_id,
                        "tp_rank": rank,
                        "counts": info["counts"],
                        "reference": "native exact dense FP32; no periodic cut",
                        "scope": "shared model-generated inputs; shadow does not alter model generation; not NLL",
                        "factor": "R32/W32 BF16 immutable columns, FP32 delayed decay and computation",
                    },
                    indent=2,
                )
            )
        return result

    backend.causal_conv1d_update = conv
    backend.KDAAttnBackend.forward_decode = forward
