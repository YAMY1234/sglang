"""CPU contracts for opt-in CP borrowing; no CUDA or model imports required."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
COMPONENTS = ROOT / "python/sglang/srt/managers/scheduler_components"
spec = importlib.util.spec_from_file_location("cp_borrow", COMPONENTS / "cp_borrow.py")
borrow = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = borrow
spec.loader.exec_module(borrow)

EXTEND, IDLE, DECODE = 1, 4, 2
BASE = [0, 0, 1, 0, 1, IDLE, 0, 0, 1, 0, 0, 0, 0]


def rows_for(owner=0, n=1601, prefix=139008, epoch=7, ticket=41, chunk=2):
    rows = []
    for rank in range(4):
        row = BASE.copy()
        flags = borrow.Flag.IDLE_ACCEPT
        if rank == owner:
            row[0], row[3], row[5] = n, 1, EXTEND
            row[10] = n * prefix + n * (n + 1) // 2
            flags = borrow.Flag.CANDIDATE
        vote = borrow.Vote(
            epoch=epoch,
            flags=flags,
            ticket=ticket if rank == owner else 0,
            prefix=prefix if rank == owner else 0,
            chunk=chunk if rank == owner else 0,
        )
        rows.append(row + list(vote.suffix()))
    return rows


def choose(rows):
    return borrow.choose_plan(rows, min_new_tokens=4, extend_mode=EXTEND, idle_mode=IDLE)


def phase_rows(plan, phase, *, reject_rank=None):
    rows = rows_for(plan.owner, plan.n, plan.prefix, plan.epoch, plan.ticket, plan.chunk)
    for rank, row in enumerate(rows):
        flag = borrow.Flag.CANDIDATE if rank == plan.owner else borrow.Flag.IDLE_ACCEPT
        if rank != reject_rank:
            flag |= borrow.Flag.RESERVED
            if phase == borrow.Phase.READY:
                flag |= borrow.Flag.READY
        row[13:] = borrow.Vote(
            epoch=plan.epoch,
            phase=phase,
            flags=flag,
            ticket=plan.ticket,
            prefix=plan.prefix,
            chunk=plan.chunk,
            reject=1 if rank == reject_rank else 0,
        ).suffix()
    return rows


class TestWireUnchanged(unittest.TestCase):
    def build(self, route, fallback=False):
        # Execute the real builder AST, substituting only torch.tensor itself.
        # This keeps the guard independent of optional CUDA import dependencies.
        tree = ast.parse((COMPONENTS / "dp_attn.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLPSyncBatchInfo")
        name = "_get_fallback_tensor" if fallback else "_get_local_tensor"
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        env = dict(
            torch=SimpleNamespace(int64="int64", tensor=lambda value, **kw: value),
            DADP_ROUTE=route,
            DADP_REPLICA_INBOX=SimpleNamespace(envelopes=[object()]),
            ForwardMode=SimpleNamespace(IDLE=SimpleNamespace(value=IDLE)),
            DRAIN_STATE_NO_LISTENER=8,
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), "dp_attn-builder", "exec"), env)
        info = SimpleNamespace(
            num_tokens=17, num_tokens_for_logprob=1,
            can_run_decode_cuda_graph=False, is_extend_in_batch=True,
            local_can_run_tbo=False, local_forward_mode=EXTEND,
            can_run_prefill_cuda_graph=True, prefill_cuda_graph_max_prefix_len=23,
            can_run_draft_cuda_graph=False, is_health_check=False,
            attention_pairs=544, is_shutting_down=False, drain_state=0,
        )
        return env[name](info, device="cpu")

    def test_n_local_13_columns(self):
        self.assertEqual(self.build(""), [17, 1, 0, 1, 0, EXTEND, 1, 23, 0, 0, 544, 0, 0])

    def test_x_local_13_columns(self):
        self.assertEqual(self.build("heavy_to_rank0"), self.build(""))

    def test_replicate_local_14_columns(self):
        self.assertEqual(self.build("replicate"), self.build("") + [1])

    def test_fallback_13_columns(self):
        self.assertEqual(self.build("", True), [0, 0, 1, 0, 1, IDLE, 0, 0, 1, 0, 0, 0, 8])

    def test_replicate_fallback_14_columns(self):
        self.assertEqual(self.build("replicate", True), self.build("", True) + [0])


class TestBorrowPlan(unittest.TestCase):
    def test_every_owner_and_independent_readers_agree(self):
        for owner in range(4):
            rows = rows_for(owner)
            plans = [choose([r.copy() for r in rows]) for _ in range(4)]
            self.assertIsNotNone(plans[0])
            self.assertTrue(all(p == plans[0] for p in plans))
            self.assertEqual(plans[0].owner, owner)

    def test_no_candidate(self):
        self.assertIsNone(choose(rows_for(owner=-1)))

    def test_two_candidates(self):
        rows = rows_for()
        rows[1] = rows_for(owner=1)[1]
        self.assertIsNone(choose(rows))

    def test_busy_or_unavailable_helper(self):
        for col, value in ((0, 1), (5, DECODE), (16, 0), (9, 1), (11, 1), (12, 8)):
            rows = rows_for()
            rows[2][col] = value
            self.assertIsNone(choose(rows), (col, value))

    def test_native_extend_and_threshold_only(self):
        for n in (0, 1, 3):
            self.assertIsNone(choose(rows_for(n=n)))
        rows = rows_for()
        rows[0][5] = DECODE
        self.assertIsNone(choose(rows))

    def test_malformed_wire_is_protocol_error(self):
        for col, value in ((13, 2), (14, 8), (15, 1), (18, -1)):
            rows = rows_for()
            rows[0][col] = value
            with self.assertRaises(borrow.ProtocolError):
                choose(rows)
        with self.assertRaises(borrow.ProtocolError):
            choose([r[:-1] for r in rows_for()])

    def test_plan_and_input_rows_are_immutable(self):
        rows = rows_for()
        original = [r.copy() for r in rows]
        plan = choose(rows)
        self.assertEqual(rows, original)
        with self.assertRaises((AttributeError, TypeError)):
            plan.prefix = 0

    def test_nonzero_ticket_required(self):
        with self.assertRaises(borrow.ProtocolError):
            choose(rows_for(ticket=0))

    def test_logical_counts_remain_at_owner(self):
        for owner in range(4):
            plan = choose(rows_for(owner, n=5))
            self.assertEqual(plan.logical_counts, tuple(5 if r == owner else 0 for r in range(4)))
            self.assertEqual(plan.execution_counts, (2, 1, 1, 1))


class TestBorrowPhases(unittest.TestCase):
    def setUp(self):
        self.plan = choose(rows_for(owner=2))

    def test_reserve_ready_finish(self):
        tx = borrow.Transaction(self.plan)
        self.assertTrue(tx.reserve(phase_rows(self.plan, borrow.Phase.RESERVE)))
        self.assertTrue(tx.commit(phase_rows(self.plan, borrow.Phase.READY)))
        tx.finish()
        self.assertEqual(tx.state, borrow.State.FINISHED)

    def test_reservation_nack_on_every_rank(self):
        for rank in range(4):
            tx = borrow.Transaction(self.plan)
            self.assertFalse(tx.reserve(phase_rows(self.plan, borrow.Phase.RESERVE, reject_rank=rank)))
            self.assertEqual(tx.state, borrow.State.REJECTED)
            self.assertTrue(tx.can_fallback)

    def test_data_nack_before_commit(self):
        tx = borrow.Transaction(self.plan)
        tx.reserve(phase_rows(self.plan, borrow.Phase.RESERVE))
        self.assertFalse(tx.commit(phase_rows(self.plan, borrow.Phase.READY, reject_rank=1)))
        self.assertTrue(tx.can_fallback)

    def test_commit_without_reserve_and_replay_rejected(self):
        tx = borrow.Transaction(self.plan)
        with self.assertRaises(borrow.ProtocolError):
            tx.commit(phase_rows(self.plan, borrow.Phase.READY))
        tx.reserve(phase_rows(self.plan, borrow.Phase.RESERVE))
        with self.assertRaises(borrow.ProtocolError):
            tx.reserve(phase_rows(self.plan, borrow.Phase.RESERVE))

    def test_epoch_ticket_chunk_or_counts_mismatch_is_fatal(self):
        for col in (0, 14, 15, 17, 18, 19):
            tx = borrow.Transaction(self.plan)
            rows = phase_rows(self.plan, borrow.Phase.RESERVE)
            rows[1][col] += 1
            with self.assertRaises(borrow.ProtocolError):
                tx.reserve(rows)
            self.assertEqual(tx.state, borrow.State.FAILED)
            self.assertFalse(tx.can_fallback)

    def test_no_fallback_after_forward_commit(self):
        tx = borrow.Transaction(self.plan)
        tx.reserve(phase_rows(self.plan, borrow.Phase.RESERVE))
        tx.commit(phase_rows(self.plan, borrow.Phase.READY))
        self.assertFalse(tx.can_fallback)
        tx.fail()
        self.assertEqual(tx.state, borrow.State.FAILED)
        self.assertFalse(tx.can_fallback)

    def test_snapshot_survives_nack(self):
        rows = rows_for(owner=2)
        before = [r.copy() for r in rows]
        tx = borrow.Transaction(choose(rows))
        tx.reserve(phase_rows(self.plan, borrow.Phase.RESERVE, reject_rank=0))
        self.assertEqual(rows, before)


class TestInterleaveAndPrivateMapping(unittest.TestCase):
    def test_interleave_input_ids_and_inverse(self):
        for n in (4, 5, 1600, 1601, 5888, 16384):
            layout = borrow.Interleave(n=n)
            ids = tuple(range(100, 100 + n))
            shards = layout.split(ids)
            self.assertEqual(shards[2], ids[2::4])
            self.assertEqual(layout.restore(shards), ids)
            self.assertEqual(sum(layout.counts), n)

    def test_positions_and_pair_conservation(self):
        for prefix in (0, 63, 64, 139008):
            plan = choose(rows_for(n=1601, prefix=prefix))
            for rank in range(4):
                self.assertEqual(plan.positions(rank), tuple(prefix + j for j in range(rank, plan.n, 4)))
                self.assertEqual(plan.attention_pairs(rank), sum(prefix + j + 1 for j in range(rank, plan.n, 4)))
            self.assertEqual(sum(plan.attention_pairs(r) for r in range(4)), 1601 * prefix + 1601 * 1602 // 2)

    def test_wrong_payload_and_padding_rejected(self):
        layout = borrow.Interleave(n=5)
        with self.assertRaises(ValueError):
            layout.split([1, 2])
        with self.assertRaises(ValueError):
            layout.restore(((1, 2), (3,), (4,), (5, 0)))

    def test_private_page_ids_are_not_owner_ids(self):
        owner = borrow.PrivateKVMap(req_slot=7, prefix_locs=(64, 65, 66), suffix_locs=(67, 68))
        helper = borrow.PrivateKVMap(req_slot=13, prefix_locs=(256, 257, 258), suffix_locs=(259, 260))
        self.assertEqual(helper.prefix_copy_pairs(owner), ((64, 256), (65, 257), (66, 258)))
        self.assertEqual(owner.prefix_locs, (64, 65, 66))
        self.assertEqual(helper.req_slot, 13)

    def test_alias_and_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            borrow.PrivateKVMap(req_slot=1, prefix_locs=(8, 9), suffix_locs=(9,))
        helper = borrow.PrivateKVMap(req_slot=2, prefix_locs=(64,), suffix_locs=(65,))
        owner = borrow.PrivateKVMap(req_slot=1, prefix_locs=(8, 9), suffix_locs=(10,))
        with self.assertRaises(ValueError):
            helper.prefix_copy_pairs(owner)

    def test_ep_dispatch_combine_restores_each_token_once(self):
        # CPU row-ownership model, not a claim of testing a real MoE backend.
        for owner in range(4):
            plan = choose(rows_for(owner, n=17))
            layout = borrow.Interleave(n=plan.n)
            source = tuple((owner, plan.ticket, j) for j in range(plan.n))
            dispatched = layout.split(source)
            experts = [[] for _ in range(4)]
            for rank, shard in enumerate(dispatched):
                for local, token in enumerate(shard):
                    experts[token[2] % 4].append((rank, local, token))
            combined = [[None] * n for n in layout.counts]
            for expert in experts:
                for rank, local, token in expert:
                    combined[rank][local] = token
            self.assertEqual(layout.restore(combined), source)
            self.assertEqual(len({t for shard in combined for t in shard}), plan.n)
            self.assertEqual(plan.final_rank, (plan.n - 1) % 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
