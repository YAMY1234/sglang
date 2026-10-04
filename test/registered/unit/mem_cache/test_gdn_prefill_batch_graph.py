"""Whole-prefix publication waits for every normal and tracked layer."""
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache import gdn_prefill_batch_graph as module


def fake_pool(layers=6, device="cpu", width=16, heads=2, capacity=40):
    cfg = native.FactoredGDNConfig(init_method="k31", dtype=torch.float16 if str(device).startswith("cuda") else torch.float32, factored_prefix=1)
    omega = torch.randn(1, heads, width, cfg.r + cfg.init_oversample, device=device)
    pool = NS(cfg=cfg, layer_ids=list(range(layers)), layer_map={i: i for i in range(layers)},
              hv=heads, v=width, k=width, prefix_layer_count=lambda: layers,
              init_omega=lambda b: omega.expand(b, -1, -1, -1),
              a=torch.full((layers, capacity, heads, width), .5, device=device),
              U=torch.zeros(layers, capacity, heads, cfg.rmax, width, device=device, dtype=cfg.dtype),
              W=torch.zeros(layers, capacity, heads, cfg.rmax, width, device=device, dtype=cfg.dtype),
              count=torch.full((layers, capacity, heads), 3, dtype=torch.int32, device=device),
              stale=torch.zeros(capacity, dtype=torch.int32, device=device),
              dense_of=torch.full((capacity,), -1, dtype=torch.int32, device=device),
              dense_required=torch.zeros(capacity, dtype=torch.int32, device=device),
              prefix_valid=torch.zeros(capacity, dtype=torch.int32, device=device),
              dense_ring=torch.zeros(layers, 16, heads, width, width, device=device),
              vbar=torch.nn.functional.normalize(torch.randn(layers, heads, width, device=device), dim=-1),
              pside_join=Mock(), invalidate_prefix_dense=Mock(), ring_generation=0, prefix_dense=None, warm_v=None)
