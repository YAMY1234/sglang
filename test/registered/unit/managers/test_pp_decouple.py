"""PP stage decoupling (SGLANG_PP_DECOUPLE_HOST): bounded send windows and the
per-epoch state digest that checks every stage holds the same scheduler state.

Decoupled stages reach a micro-batch epoch at different wall times, but each one
must apply the same allocator and radix tree operations in the same epoch order;
the digest, compared per epoch, is how a run proves that.
"""

import random
import unittest
import weakref
from array import array
from collections import deque

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.managers import pp_decouple  # noqa: E402
from sglang.srt.managers.pp_decouple import PPSendWindow  # noqa: E402
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Payload:
    pass


class _FakeWork:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait(self):
        self.log.append(self.name)


class _FakeP2PWork:
    """P2PWork shape: the work handle and the tensor it keeps alive."""

    def __init__(self, name, log):
        self.work = _FakeWork(name, log)
        self.payload = _Payload()


def _round(name, log, n=2):
    return [_FakeP2PWork(f"{name}.{i}", log) for i in range(n)]


class TestPPSendWindow(CustomTestCase):
    def test_waits_only_rounds_beyond_bound_oldest_first_and_releases_them(self):
        log = []
        window = PPSendWindow(bound=2)
        rounds = [_round(f"r{i}", log) for i in range(4)]
        payloads = [weakref.ref(work.payload) for r in rounds for work in r]
        for r in rounds[:2]:
            self.assertEqual(window.push(works=r), 0)
        self.assertEqual(log, [])
        self.assertEqual(window.push(works=rounds[2]), 1)
        self.assertEqual(window.push(works=rounds[3]), 1)
        self.assertEqual(log, ["r0.0", "r0.1", "r1.0", "r1.1"])
        # Callers clear their work list after handing it over (_pp_commit_comm_work).
        for r in rounds:
            r.clear()
        self.assertEqual([ref() is None for ref in payloads], [True] * 4 + [False] * 4)
        # Empty rounds (ranks that sent nothing) do not occupy the window.
        self.assertEqual(window.push(works=[]), 0)
        self.assertEqual(window.drain(), 2)
        self.assertEqual(log[4:], ["r2.0", "r2.1", "r3.0", "r3.1"])
        self.assertTrue(all(ref() is None for ref in payloads))


class _StageModel:
    """One PP stage's allocator and radix tree, driven by per-epoch operations."""

    def __init__(self, page_count=512):
        self.allocator = TokenToKVPoolAllocator(
            size=page_count,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.req_pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        self.cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
            )
        )
        self.locked = {}

    def apply(self, op):
        kind, rid, arg = op
        if kind == "admit":
            key = RadixKey(array("q", arg))
            kv = self.allocator.alloc(len(arg))
            matched = self.cache.insert(InsertParams(key=key, value=kv)).prefix_len
            self.allocator.free(kv[:matched])
            node = self.cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
            self.cache.inc_lock_ref(node)
            self.locked[rid] = node
        elif kind == "release":
            self.cache.dec_lock_ref(self.locked.pop(rid))
        else:
            self.cache.evict(EvictParams(num_tokens=arg))

    def digest(self):
        sizes, tensors = pp_decouple.collect_pp_state(
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            tree_cache=self.cache,
        )
        return pp_decouple.pp_state_digest(sizes=sizes, tensors=tensors)


def _epoch_script(num_epochs, seed=7):
    """Admissions share a few system prompts (node splits), releases retire the
    oldest locked request, evictions free unlocked leaves."""
    rng = random.Random(seed)
    prompts = [[1000 * p + i for i in range(6)] for p in range(3)]
    active, script = deque(), []
    for epoch in range(num_epochs):
        ops = []
        for _ in range(rng.randint(0, 2)):
            rid = f"e{epoch}r{len(ops)}"
            suffix = [10 * len(script) + i for i in range(rng.randint(2, 6))]
            ops.append(("admit", rid, rng.choice(prompts) + suffix))
            active.append(rid)
        if active and rng.random() < 0.6:
            ops.append(("release", active.popleft(), None))
        if rng.random() < 0.3:
            ops.append(("evict", None, rng.randint(1, 12)))
        script.append(ops)
    return script


def _run_stages(script, num_stages, every, late_release_stage=None, seed=11):
    """Advance stages through the epochs in a random interleaving; each logs a
    digest line every `every` epochs, so log order is not epoch order."""
    rng = random.Random(seed)
    stages = [_StageModel() for _ in range(num_stages)]
    next_epoch = [0] * num_stages
    carried = [[] for _ in range(num_stages)]
    log = []
    while min(next_epoch) < len(script):
        rank = rng.choice([r for r in range(num_stages) if next_epoch[r] < len(script)])
        epoch = next_epoch[rank]
        ops = carried[rank] + [op for op in script[epoch] if op[0] != "release"]
        releases = [op for op in script[epoch] if op[0] == "release"]
        if rank == late_release_stage:
            # Models applying a relayed event when it arrives, not at its epoch.
            carried[rank] = releases
        else:
            ops = ops + releases
            carried[rank] = []
        for op in ops:
            stages[rank].apply(op)
        next_epoch[rank] = epoch + 1
        if (epoch + 1) % every == 0:
            record = pp_decouple.PPStateDigest(
                pp_rank=rank, dp_rank=0, epoch=epoch + 1, digest=stages[rank].digest()
            )
            log.append("INFO scheduler " + pp_decouple.format_digest_line(record))
    return log


class TestPPStateDigest(CustomTestCase):
    NUM_STAGES = 4
    NUM_EPOCHS = 48
    EVERY = 4

    def test_stages_agree_at_equal_epochs_whatever_their_interleaving(self):
        script = _epoch_script(self.NUM_EPOCHS)
        log = _run_stages(script, self.NUM_STAGES, self.EVERY)
        records = pp_decouple.parse_digest_lines(log)
        self.assertEqual(len(records), self.NUM_STAGES * self.NUM_EPOCHS // self.EVERY)
        # The interleaving must actually skew stages, or grouping is untested.
        self.assertNotEqual(
            [r.epoch for r in records], sorted(r.epoch for r in records)
        )
        report = pp_decouple.find_digest_mismatches(records, pp_size=self.NUM_STAGES)
        self.assertEqual((report.mismatched, report.incomplete), ({}, []))
        self.assertGreater(len({r.digest for r in records}), 1)

    def test_an_event_applied_off_epoch_is_flagged_at_exactly_those_epochs(self):
        script = _epoch_script(self.NUM_EPOCHS)
        log = _run_stages(script, self.NUM_STAGES, self.EVERY, late_release_stage=2)
        report = pp_decouple.find_digest_mismatches(
            pp_decouple.parse_digest_lines(log), pp_size=self.NUM_STAGES
        )
        expected = {
            (epoch, 0)
            for epoch in range(self.EVERY, self.NUM_EPOCHS + 1, self.EVERY)
            if any(op[0] == "release" for op in script[epoch - 1])
        }
        self.assertTrue(expected)
        self.assertEqual(set(report.mismatched), expected)
        for stages in report.mismatched.values():
            others = {digest for rank, digest in stages.items() if rank != 2}
            self.assertEqual(len(others), 1)
            self.assertNotIn(stages[2], others)

    def test_free_list_order_is_part_of_the_state(self):
        first, second = _StageModel(), _StageModel()
        for stage, order in ((first, (0, 1)), (second, (1, 0))):
            kv = [stage.allocator.alloc(4), stage.allocator.alloc(4)]
            for i in order:
                stage.allocator.free(kv[i])
        # Same free set, different order: the next allocation would diverge.
        self.assertEqual(
            first.allocator.available_size(), second.allocator.available_size()
        )
        self.assertNotEqual(first.digest(), second.digest())

    def test_a_cut_short_log_is_incomplete_not_mismatched(self):
        records = [
            pp_decouple.PPStateDigest(pp_rank=r, dp_rank=0, epoch=e, digest="d")
            for e in (4, 8)
            for r in range(self.NUM_STAGES)
            if not (e == 8 and r == 3)
        ]
        report = pp_decouple.find_digest_mismatches(records, pp_size=self.NUM_STAGES)
        self.assertEqual((report.mismatched, report.incomplete), ({}, [(8, 0)]))


if __name__ == "__main__":
    unittest.main()
