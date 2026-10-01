"""Tier A regression: ownership boundaries and production Qwen3.5 token replay.

Set SGLANG_TIER_A_TEST_MODEL to a small Qwen3.5 checkpoint/config. The GPU
portion uses the real scheduler, verify, KV allocator and graph runners.
"""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def ownership_cases():
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/speculative/extend_draft_state.py"
    )
    spec = importlib.util.spec_from_file_location("tier_a_state", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    pool = SimpleNamespace(req_generation=torch.zeros(9, dtype=torch.int64))
    store = module.PendingExtendStore(pool, 4, 2, torch.float32, "cpu")
    slots = torch.tensor([1, 3, 7])
    hidden = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    tokens = torch.arange(12)
    cache = torch.arange(20, 32)
    store.put(
        slots,
        slots,
        hidden,
        tokens,
        cache,
        torch.tensor([31, 63, 127]),
        torch.tensor([1, 4, 2]),
    )
    hidden.fill_(-99)
    tokens.fill_(0)
    rows = store.take(torch.tensor([7, 1, 0, 5]))
    assert rows.valid_cpu.tolist() == [True, True, False, False]
    assert rows.tokens.tolist() == [8, 9, 10, 11, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    assert rows.cache.tolist() == [28, 29, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert rows.hidden[0].tolist() == [16, 17]
    store.consumed(rows)
    assert not store.ready(torch.tensor([7, 1])).any()
    assert store.ready(torch.tensor([3])).item()
    pool.req_generation[3] += 1
    reused = store.take(torch.tensor([3]))
    assert not reused.valid_cpu.any() and reused.hidden.count_nonzero() == 0
    empty = store.take(torch.empty(0, dtype=torch.int64))
    assert empty.hidden.shape == (0, 2) and empty.cache.numel() == 0
    store.consumed(empty)
    pool.req_generation[3] -= 1
    pool.req_generation_epoch = 1
    assert not store.ready(torch.tensor([3])).any()
    return {
        "ownership": "passed",
        "cases": [
            "reorder",
            "padding",
            "bootstrap",
            "generation-reuse",
            "rejected-KV",
            "empty",
            "detached-storage",
            "pool-clear",
        ],
    }


def install_receipt_rpc():
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    original_draft = EagleDraftWorker.draft

    def observed_draft(self, batch, **kwargs):
        prefixes = batch.seq_lens.cpu().tolist()
        rids = [req.rid for req in batch.reqs]
        result = original_draft(self, batch, **kwargs)
        verify = result[0] if isinstance(result, tuple) else result
        if rids:
            tokens = verify.draft_token.reshape(len(rids), -1).cpu().tolist()
            rank = torch.distributed.get_rank()
            path = os.environ["SGLANG_TIER_A_TEST_CHAINS"] + f".rank{rank}.jsonl"
            with open(path, "a") as stream:
                stream.writelines(
                    json.dumps([rid, prefix, chain]) + "\n"
                    for rid, prefix, chain in zip(rids, prefixes, tokens, strict=True)
                )
        return result

    if os.environ.get("SGLANG_TIER_A_TEST_OBSERVE_CHAINS", "1") == "1":
        EagleDraftWorker.draft = observed_draft

    def receipt(self, path):
        worker = getattr(self.model_worker, "_draft_worker", None)
        coordinator = getattr(worker, "extend_draft_coordinator", None)
        state = {"enabled": coordinator is not None}
        if coordinator is not None:
            state.update(
                fused_replays=coordinator.graph.fused_replays,
                flushes=coordinator.flushes,
                deferred_rounds=coordinator.deferred_rounds,
                stored_rows=coordinator.store.stored_rows,
                consumed_rows=coordinator.store.consumed_rows,
                buckets=coordinator.graph.capture_bs,
            )
        rank = torch.distributed.get_rank()
        Path(path + f".rank{rank}.json").write_text(json.dumps(state))
        if rank == 0:
            Path(path).write_text(json.dumps(state))

    Scheduler.tier_a_regression_receipt = receipt


if os.environ.get("SGLANG_TIER_A_TEST_CHILD") == "1":
    install_receipt_rpc()


def engine_case(model, out):
    from sglang import Engine

    engine = Engine(
        model_path=model,
        tp_size=int(os.environ.get("SGLANG_TIER_A_TEST_TP", "1")),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        random_seed=207,
        mem_fraction_static=0.35,
        max_total_tokens=8192,
        max_running_requests=16,
        context_length=1024,
        attention_backend="trtllm_mha",
        moe_runner_backend="triton",
        speculative_algorithm="NEXTN",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        cuda_graph_bs_decode=[1, 4, 8, 16],
        cuda_graph_backend_decode="full",
        disable_prefill_cuda_graph=True,
        disable_radix_cache=True,
        chunked_prefill_size=512,
        enable_mixed_chunk=False,
        skip_server_warmup=True,
        log_level="info",
    )
    try:
        outputs = []
        # Each completed group returns to empty, then bootstraps reused slots.
        for repeat, count in enumerate((1, 3, 8, 2, 9, 4)):
            prompts = [
                [10 + ((j * 13 + i + repeat) % 480) for i in range(31 + j)]
                for j in range(count)
            ]
            result = engine.generate(
                input_ids=prompts,
                rid=[f"group{repeat}-req{j}" for j in range(count)],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": 48,
                    "ignore_eos": True,
                },
                return_logprob=True,
                top_logprobs_num=2,
            )
            outputs.extend(result if isinstance(result, list) else [result])
        if os.environ.get("SGLANG_TIER_A_TEST_HOT_ARRIVAL") == "1":

            async def hot_arrival():
                stream = await engine.async_generate(
                    input_ids=list(range(10, 41)),
                    rid="hot-main",
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": 256,
                        "ignore_eos": True,
                    },
                    stream=True,
                )
                late = None
                async for chunk in stream:
                    if len(chunk["output_ids"]) >= 8 and late is None:
                        late = asyncio.create_task(
                            engine.async_generate(
                                input_ids=list(range(31, 63)),
                                rid="hot-late",
                                sampling_params={
                                    "temperature": 0,
                                    "max_new_tokens": 48,
                                    "ignore_eos": True,
                                },
                            )
                        )
                    last = chunk
                assert late is not None
                return [last, await late]

            outputs.extend(engine.loop.run_until_complete(hot_arrival()))
        engine.collective_rpc("tier_a_regression_receipt", path=out + ".state.json")
        Path(out).write_text(json.dumps(outputs))
    finally:
        engine.shutdown()


class TestExtendDraftGraph(unittest.TestCase):
    def test_tokens_and_lifecycle(self):
        ownership_cases()
        model = os.environ.get("SGLANG_TIER_A_TEST_MODEL")
        if not model or not torch.cuda.is_available():
            self.skipTest(
                "production graph gate requires CUDA and a small Qwen3.5 fixture"
            )
        with tempfile.TemporaryDirectory(
            prefix="tier-a-regression-", dir=os.environ.get("SGLANG_TIER_A_TEST_OUT")
        ) as directory:
            root = Path(directory)
            scenarios = []
            for acceptance in os.environ.get(
                "SGLANG_TIER_A_TEST_ACCEPTANCE", "-1,2,4"
            ).split(","):
                outputs, states, chains = [], [], []
                for enabled in (0, 1):
                    label = f"al{acceptance}-{enabled}"
                    out = root / f"{label}.json"
                    env = dict(
                        os.environ,
                        SGLANG_SPEC_FUSE_EXTEND_DRAFT=str(enabled),
                        SGLANG_TIER_A_TEST_CHILD="1",
                        SGLANG_TIER_A_TEST_CHAINS=str(out) + ".chains",
                        SGLANG_SIMULATE_ACC_LEN=acceptance,
                        SGLANG_SIMULATE_ACC_TOKEN_MODE="real-draft-token",
                    )
                    with (root / f"{label}.log").open("w") as log:
                        result = subprocess.run(
                            [sys.executable, __file__, "--engine", model, str(out)],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=720,
                            check=False,
                        )
                    if destination := os.environ.get("SGLANG_TIER_A_TEST_OUT"):
                        import shutil

                        for artifact in root.glob(f"{label}*"):
                            shutil.copy2(artifact, Path(destination) / artifact.name)
                    if result.returncode:
                        print((root / f"{label}.log").read_text()[-24000:], flush=True)
                    self.assertEqual(result.returncode, 0)
                    outputs.append(json.loads(out.read_text()))
                    states.append(
                        json.loads(Path(str(out) + ".state.json").read_text())
                    )
                    chain_path = Path(str(out) + ".chains.rank0.jsonl")
                    if chain_path.exists():
                        records = [
                            json.loads(line)
                            for line in chain_path.read_text().splitlines()
                        ]
                        chains.append(
                            {(rid, prefix): tokens for rid, prefix, tokens in records}
                        )
                self.assertFalse(states[0]["enabled"])
                self.assertTrue(
                    states[1]["enabled"], "fusion guard did not admit the fixture"
                )
                self.assertGreater(
                    states[1]["fused_replays"], 0, "a no-op flag cannot pass"
                )
                self.assertGreater(states[1]["consumed_rows"], 0)
                if os.environ.get("SGLANG_TIER_A_TEST_HOT_ARRIVAL") == "1":
                    self.assertGreater(
                        states[1]["flushes"], 0, "exercise mixed bootstrap fallback"
                    )
                checked = 0
                for stock, fused in zip(outputs[0], outputs[1], strict=True):
                    self.assertEqual(stock["output_ids"], fused["output_ids"])
                    checked += len(stock["output_ids"])
                shared = set()
                if os.environ.get("SGLANG_TIER_A_TEST_OBSERVE_CHAINS", "1") == "1":
                    self.assertEqual(len(chains), 2)
                    shared = chains[0].keys() & chains[1].keys()
                    self.assertGreaterEqual(len(shared), 0.95 * max(map(len, chains)))
                    mismatches = [
                        (key, chains[0][key], chains[1][key])
                        for key in sorted(shared)
                        if chains[0][key] != chains[1][key]
                    ]
                    self.assertEqual(
                        mismatches[:20],
                        [],
                        "draft chain changed at an identical prefix",
                    )
                scenarios.append(
                    {
                        "acceptance": acceptance,
                        "checked_tokens": checked,
                        "checked_draft_rows": len(shared),
                        "draft_rows": list(map(len, chains)),
                        "token_mismatch": 0,
                        "chain_mismatch": 0 if chains else None,
                        "per_round_observer_sync": bool(chains),
                        "state": states[1],
                    }
                )
                print(json.dumps({"scenario_pass": scenarios[-1]}), flush=True)
            print(
                json.dumps(
                    {
                        "pass": True,
                        "scenarios": scenarios,
                        "scope": "tiny production Qwen3.5; injected acceptance covers state transitions only; not quality/performance",
                        **ownership_cases(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    if "--engine" in sys.argv:
        engine_case(sys.argv[2], sys.argv[3])
    elif "--state-only" in sys.argv:
        print(json.dumps(ownership_cases()))
    else:
        unittest.main()
