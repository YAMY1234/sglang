"""Real PD install/prewarm dispatch and native checkpoint policy on CPU.

CUDA graph allocation/kernels are replaced, not the installers or collector.
"""

import ast
import copy
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from test_flash_next_native_factor_tail import (
    installed,
    fixture,
    layer,
    IDS,
    ForwardBatch,
    ScheduleBatch,
    ForwardMode,
    torch,
)
from sglang.srt.mem_cache import gdn_prefill_agg_contract as agg
from sglang.srt.mem_cache import gdn_prefill_batch_graph as batch_graph
from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.disaggregation.state_handoff import FactorStateHandoff

ROOT = Path(__file__).resolve().parents[2]


def init_graphs(runner, capture):
    path = ROOT / "python/sglang/srt/model_executor/model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    method = copy.deepcopy(
        next(n for n in cls.body if getattr(n, "name", "") == "init_cuda_graphs")
    )
    from sglang.srt.environ import envs

    scope = dict(os=os, envs=envs, capture_cuda_graphs=capture)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    scope["init_cuda_graphs"](runner)


@contextmanager
def worker(*, native_model=True, flag=True):
    with installed(native=native_model) as (owner, receipt), ExitStack() as stack:
        stack.enter_context(
            patch.object(
                ForwardBatch, "_pfactor_agg_contract_installed", False, create=True
            )
        )
        stack.enter_context(
            patch.dict(
                os.environ,
                {"SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH": "1", agg.FLAG: str(int(flag))},
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=NS(disable_overlap_schedule=True),
            )
        )
        c = fixture()
        p = c.pool
        p.layer_ids = IDS
        p.layer_map = {lid: i for i, lid in enumerate(IDS)}
        p.layer_index = lambda lid: p.layer_map[lid]
        p.batch_prefill = True
        p._prefill_batch_graph.shared = {}

        class NativeLayer:
            pass

        layers = []
        for lid in IDS:
            obj = NativeLayer()
            obj.__dict__.update(vars(layer(lid)))
            layers.append(obj)
        owner.model.model.modules = lambda: iter(layers)
        stack.enter_context(
            patch(
                "sglang.srt.layers.radix_linear_attention.RadixLinearAttention",
                NativeLayer,
            )
        )
        events = []

        def commit_prewarm():
            if flag:
                assert owner._exact_tail_installed and owner._pfactor_agg_installed
            events.append("commit-prewarm")

        p.prewarm_commit_graph = commit_prewarm
