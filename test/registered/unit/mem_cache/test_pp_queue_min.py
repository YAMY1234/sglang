"""CPU regression for real cache methods, with in-process mocked collectives.

No CUDA/SGLang import is needed. Real Queue.get() demonstrates the scheduler
block; only tensors and distributed transports are mocked. This does not test
Gloo/NCCL progress or prove cross-rank queue contents are identical.
"""
from __future__ import annotations

import ast
import queue
import threading
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/mem_cache/unified_radix_cache.py"
TREE = ast.parse(SOURCE.read_text())
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedRadixCache")


def method(name, **extra):
    node = next(n for n in ast.walk(CLASS) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = {"torch": types.SimpleNamespace(distributed=types.SimpleNamespace(ReduceOp=types.SimpleNamespace(MIN="MIN")))}
    namespace.update(extra)
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace[name]


class Collective:
    def __init__(self, size):
        self.barrier = threading.Barrier(size, timeout=3)
        self.values = {}

    def reduce(self, rank, data, op):
        assert op == "MIN"
        self.values[rank] = list(data)
        self.barrier.wait()
        result = [min(v[i] for v in self.values.values()) for i in range(len(data))]
        self.barrier.wait()
        data[:] = result

    def broadcast(self, rank, data):
        if rank == 0:
            self.values[0] = list(data)
        self.barrier.wait()
        data[:] = self.values[0]


def synchronized(local, fixed):
    pp, tp = len(local), len(local[0])
    tp_groups = [Collective(tp) for _ in range(pp)]
    pp_groups = [Collective(pp) for _ in range(tp)]
    output, errors = {}, []

    def worker(p, t):
        obj = types.SimpleNamespace(pp_rank=p)
        obj._all_reduce_attn_groups = lambda data, op: tp_groups[p].reduce(t, data, op)
        obj._all_reduce_pp_group = lambda data, op: pp_groups[t].reduce(p, data, op)
        obj._pp_sync = lambda data: pp_groups[t].broadcast(p, data)
        data = list(local[p][t])
        try:
            if fixed:
                method("_all_reduce_queue_counts")(obj, data)
            else:
                method("_all_reduce")(obj, data, "MIN")
            output[p, t] = data
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(p, t)) for p in range(pp) for t in range(tp)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not any(t.is_alive() for t in threads), "collective stalled"
    assert not errors, errors
    return output


class Tensor:
    def __init__(self, values, storage=None, offset=0, length=None):
        self.storage = list(values) if storage is None else storage
        self.offset = offset
        self.length = len(self.storage) if length is None else length
    def __len__(self):
        return self.length
    def __iter__(self):
        return iter(self.tolist())
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.length)
            assert step == 1
            return Tensor([], self.storage, self.offset + start, stop - start)
        if key < 0:
            key += self.length
        if key >= self.length:
            raise IndexError(key)
        return self.storage[self.offset + key]
    def __setitem__(self, key, value):
        assert isinstance(key, slice)
        start, stop, step = key.indices(self.length)
        assert step == 1
        self.storage[self.offset + start:self.offset + stop] = list(value)
    def tolist(self):
        return self.storage[self.offset:self.offset + self.length]
    def clone(self):
        return Tensor(self.tolist())


def ready_result(digests):
    tp_groups = [Collective(len(row)) for row in digests]
    pp_groups = [Collective(len(digests)) for _ in digests[0]]
    results, errors, warnings = {}, {}, []
    torch = types.SimpleNamespace(tensor=lambda values, **kw: Tensor(values), int64="int64", distributed=types.SimpleNamespace(ReduceOp=types.SimpleNamespace(MIN="MIN")))
    def run(p, t):
        cc = types.SimpleNamespace(ack_write_queue=[1, 2], ack_load_queue=[1], prefetch_hit_queue=queue.Queue(), ack_prefetch_queue=queue.Queue(), ack_backup_queue=queue.Queue(), host_mem_release_queue=queue.Queue())
        for i in range(p + 1):
            cc.ack_backup_queue.put(i)
        obj = types.SimpleNamespace(cache_controller=cc, pp_size=len(digests), enable_storage=True, tree_core=types.SimpleNamespace(write_back_duplicate_reclaim_digest=digests[p][t]), _count_ready_acks=len, _l3_tier_stats={})
        obj._all_reduce_attn_groups=lambda data, op: tp_groups[p].reduce(t, data, op)
        obj._all_reduce_pp_group=lambda data, op: pp_groups[t].reduce(p, data, op)
        obj._all_reduce_queue_counts=lambda data: method("_all_reduce_queue_counts")(obj, data)
        try:
            value=method("_sync_hicache_ready_counts",torch=torch,logger=types.SimpleNamespace(warning=lambda *a: warnings.append(a)))(obj)
            results[p,t]=(value,dict(obj._l3_tier_stats))
        except BaseException as exc:
            errors[p,t]=exc
    threads=[threading.Thread(target=run,args=(p,t)) for p,row in enumerate(digests) for t in range(len(row))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not any(t.is_alive() for t in threads), "ready collective stalled"
    return results,errors,warnings


class PPQueueMinTest(unittest.TestCase):
    def test_old_pp0_count_blocks_real_empty_queue(self):
        counts = synchronized([[[1]], [[0]], [[1]], [[2]]], fixed=False)
        self.assertEqual(counts[1, 0], [1])
        q = queue.Queue()
        done = threading.Event()
        drained = []
        def consume():
            drained.extend(method("_drain_queue")(q, counts[1, 0][0]))
            done.set()
        thread = threading.Thread(target=consume)
        thread.start()
        try:
            self.assertFalse(done.wait(0.2), "old code should block at Queue.get")
        finally:
            q.put("test-only unblock")
            thread.join(2)
        self.assertTrue(done.is_set())

    def test_fixed_pp_counts_do_not_block_and_eventually_drain(self):
        local = [[[2, 3, 1, 1, 0]], [[2, 2, 0, 1, 0]], [[3, 2, 2, 1, 0]], [[2, 2, 1, 1, 0]]]
        outputs = synchronized(local, fixed=True)
        for rank, counts in outputs.items():
            self.assertEqual(counts, [2, 2, 0, 1, 0])
            for have, count in zip(local[rank[0]][rank[1]], counts):
                q = queue.Queue()
                for i in range(have):
                    q.put(i)
                self.assertLessEqual(count, q.qsize())
                self.assertEqual(len(list(method("_drain_queue")(q, count))), count)
        ready = synchronized([[[1]], [[2]], [[1]], [[3]]], fixed=True)
        self.assertTrue(all(v == [1] for v in ready.values()))

    def test_min_across_every_pp_and_tp_with_ready_digest(self):
        local = [[[4, 3, 7, -7], [3, 3, 7, -7]], [[2, 5, 7, -7], [1, 2, 7, -7]], [[4, 1, 7, -7], [5, 3, 7, -7]], [[2, 5, 7, -7], [6, 2, 7, -7]]]
        result = synchronized(local, fixed=True)
        self.assertTrue(all(v == [1, 1, 7, -7] for v in result.values()))
        result = synchronized([[[1, 7, -7]], [[1, 8, -8]]], fixed=True)
        self.assertTrue(all(v[-2] != -v[-1] for v in result.values()))

    def test_equal_counts_and_single_pp(self):
        for data in ([[[2]], [[2]], [[2]], [[2]]], [[[3], [1]]], [[[0]]]):
            result = synchronized(data, fixed=True)
            expected = min(v[0] for row in data for v in row)
            self.assertTrue(all(v == [expected] for v in result.values()))

    def test_pp_digest_difference_warns_without_expanding_tp_assert(self):
        results,errors,warnings=ready_result([[7,7],[9,9],[11,11],[12,12]])
        self.assertEqual(errors,{})
        self.assertEqual(len(results),8)
        self.assertEqual(len(warnings),8)
        for value,stats in results.values():
            self.assertEqual(value[:3],(2,1,(0,0,1,0,0)))
            self.assertEqual(stats["pp_reclaim_digest_mismatches"],1)
        results,errors,warnings=ready_result([[7,7],[7,7]])
        self.assertEqual(errors,{})
        self.assertEqual(warnings,[])

    def test_tp_digest_difference_still_asserts(self):
        results,errors,warnings=ready_result([[7,8],[7,8]])
        self.assertEqual(results,{})
        self.assertEqual(len(errors),4)
        self.assertTrue(all(isinstance(e,AssertionError) and "across TP ranks" in str(e) for e in errors.values()))

    def test_both_queue_call_sites_use_global_min(self):
        for name, variable in (("drain_storage_control_queues", "qsizes"), ("_sync_hicache_ready_counts", "ready_counts")):
            node = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_all_reduce_queue_counts"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(ast.unparse(calls[0].args[0]), variable if variable == "qsizes" else "ready_counts[:-2]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
