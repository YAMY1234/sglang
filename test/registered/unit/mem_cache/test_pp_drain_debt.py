"""Exercise the real drain protocol with FIFO queues and modeled P2P transport.

CPU only: no CUDA, Gloo or complete scheduler. The last test deliberately
exhibits the cache-decision counterexample; bounded reads are not consensus.
"""
import ast
import logging
from itertools import islice
import queue
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4] / 'python/sglang/srt/mem_cache/unified_radix_cache.py'
CLASS = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'UnifiedRadixCache')

class Tensor:
    def __init__(self, values): self.values = list(values)
    def clone(self): return Tensor(self.values)
    def tolist(self): return list(self.values)
    def __getitem__(self, index): return self.values[index]
    def __add__(self, other): return Tensor(a + b for a, b in zip(self.values, other.values))
    def __sub__(self, other): return Tensor(a - b for a, b in zip(self.values, other.values))

class Work:
    def wait(self): pass

class Chain:
    def __init__(self, size):
        self.mailboxes = [queue.Queue() for _ in range(size)]
        self.events = []
    def transport(self, rank):
        def send(data, group_dst, **kwargs):
            self.events.append(('send', rank, group_dst))
            self.mailboxes[group_dst].put(data.clone())
            return Work()
        def recv(data, group_src, **kwargs):
            self.events.append(('recv', rank, group_src))
            data.values[:] = self.mailboxes[rank].get(timeout=0.2).values
        return types.SimpleNamespace(isend=send, recv=recv, ReduceOp=types.SimpleNamespace(MIN='MIN'))

def method(name, namespace):
    node = next(n for n in ast.walk(CLASS) if isinstance(n, ast.FunctionDef) and n.name == name)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), 'exec'), namespace)
    return namespace[name]

def cache(chain, rank):
    logger = logging.getLogger('pp-drain-debt-test')
    ns = {'islice': islice, 'torch': types.SimpleNamespace(tensor=lambda data, **kw: Tensor(data), minimum=lambda a,b: Tensor(min(x,y) for x,y in zip(a.values,b.values)), int=int, distributed=chain.transport(rank)), 'logger': logger, 'P2PTag': types.SimpleNamespace(HIRADIX_PP_SYNC=1)}
    obj = types.SimpleNamespace(pp_rank=rank, pp_size=len(chain.mailboxes), pp_group='PP', work_list=[], _pp_drain_state={}, _l3_tier_stats={}, ongoing_backup={}, storage_existence_cache=set(), cache_controller=types.SimpleNamespace(ack_backup_queue=queue.Queue()))
    obj._all_reduce_attn_groups = lambda data, op: None
    for name in ('_pp_sync', '_all_reduce', '_pp_drain_counts'):
        setattr(obj, name, types.MethodType(method(name, ns), obj))
    obj.drain = method('_drain_queue', ns)
    return obj

class DebtTest(unittest.TestCase):
    def test_late_ack_is_not_blocking_and_catches_up_next_round(self):
        chain = Chain(2); a,b = [cache(chain,i) for i in range(2)]
        qa,qb = queue.Queue(), queue.Queue(); qa.put('op0')
        self.assertEqual(list(a.drain(qa, a._pp_drain_counts([qa], ['ack_backup'])[0])), ['op0'])
        self.assertEqual(list(b.drain(qb, b._pp_drain_counts([qb], ['ack_backup'])[0])), [])
        self.assertEqual(b._pp_drain_state['ack_backup'][0], 1)
        qb.put('op0')
        self.assertEqual(a._pp_drain_counts([qa], ['ack_backup'])[0], 0)
        self.assertEqual(list(b.drain(qb, b._pp_drain_counts([qb], ['ack_backup'])[0])), ['op0'])
        self.assertEqual(b._pp_drain_state['ack_backup'][0], 0)

    def test_three_stage_order_is_original_pp_sync_then_local_clip(self):
        chain = Chain(3); ranks = [cache(chain,i) for i in range(3)]
        qs = [queue.Queue() for _ in ranks]; qs[0].put('same'); qs[2].put('same')
        counts = [r._pp_drain_counts([q], ['ack_backup'])[0] for r,q in zip(ranks,qs)]
        self.assertEqual(counts, [1,0,1])  # PP1 must forward PP0 budget, not zero.
        self.assertEqual(chain.events, [('send',0,1),('send',0,1),('recv',1,0),('send',1,2),('recv',1,0),('send',1,2),('recv',2,1),('recv',2,1)])
        self.assertTrue(all(q.empty() for q in chain.mailboxes))

    def test_persistent_debt_warns_once_with_both_operation_snapshots(self):
        chain = Chain(2); a,b = [cache(chain,i) for i in range(2)]
        a.ongoing_backup[17] = (42, 'host-lock'); b.ongoing_backup[18] = (43, 'host-lock')
        a.cache_controller.ack_backup_queue.put(types.SimpleNamespace(id=17,hash_value=['key17']))
        qa,qb=queue.Queue(),queue.Queue();qa.put('one')
        with self.assertLogs('pp-drain-debt-test', level='WARNING') as captured:
            for step in range(20):
                count=a._pp_drain_counts([qa], ['ack_backup'])[0]
                list(a.drain(qa,count)); b._pp_drain_counts([qb], ['ack_backup'])
        self.assertEqual(len(captured.output),1)
        self.assertIn('key17',captured.output[0]); self.assertIn('43',captured.output[0])
        self.assertEqual(b._l3_tier_stats['pp_drain_debt']['ack_backup']['current'],1)

    def test_per_queue_debts_are_independent_and_threshold_is_immediate(self):
        chain=Chain(2);a,b=[cache(chain,i) for i in range(2)]
        qa=[queue.Queue(),queue.Queue()];qb=[queue.Queue(),queue.Queue()]
        for _ in range(64): qa[0].put('x')
        qa[1].put('y');qb[1].put('y')
        a._pp_drain_counts(qa,['ack_backup','prefetch_hit'])
        with self.assertLogs('pp-drain-debt-test',level='WARNING') as cap:
            out=b._pp_drain_counts(qb,['ack_backup','prefetch_hit'])
        self.assertEqual(out.tolist(),[0,1]);self.assertEqual(len(cap.output),1)
        self.assertEqual(b._pp_drain_state['prefetch_hit'][0],0)

    def test_local_tp_min_bounds_every_tp_rank(self):
        chain=Chain(2);a,b=[cache(chain,i) for i in range(2)]
        qa,qb=queue.Queue(),queue.Queue()
        for _ in range(4):qa.put(0);qb.put(0)
        a._pp_drain_counts([qa],['ack_backup'])
        b._all_reduce_attn_groups=lambda data,op:data.values.__setitem__(0,2)
        self.assertEqual(b._pp_drain_counts([qb],['ack_backup'])[0],2)
        self.assertEqual(b._pp_drain_state['ack_backup'][0],2)

    def test_peak_cycles_and_belief_size_mismatch_are_retained(self):
        chain=Chain(2);a,b=[cache(chain,i) for i in range(2)]
        a.storage_existence_cache.add('PP0-only-belief')
        qa,qb=queue.Queue(),queue.Queue();qa.put('one')
        a._pp_drain_counts([qa],['ack_backup']);b._pp_drain_counts([qb],['ack_backup'])
        qa.get();a._pp_drain_counts([qa],['ack_backup']);b._pp_drain_counts([qb],['ack_backup'])
        qb.put('one');a._pp_drain_counts([qa],['ack_backup']);b._pp_drain_counts([qb],['ack_backup'])
        stats=b._l3_tier_stats['pp_drain_debt']['ack_backup']
        self.assertEqual((stats['current'],stats['cycles'],stats['peak'],stats['max_cycles']),(0,0,1,2))
        self.assertEqual(b._l3_tier_stats['pp_belief_size_mismatch'],3)

    def test_digest_is_tp_only_and_no_full_pp_collective_added(self):
        node=next(n for n in CLASS.body if isinstance(n,ast.FunctionDef) and n.name=='_pp_drain_counts')
        self.assertNotIn('_all_reduce_pp_group',ast.unparse(node))
        node=next(n for n in CLASS.body if isinstance(n,ast.FunctionDef) and n.name=='_sync_hicache_ready_counts')
        self.assertIn('self._all_reduce_attn_groups(ready_counts[-2:]',ast.unparse(node))
        self.assertIn('self._all_reduce(ready_counts[:-2]',ast.unparse(node))

    def test_debt_is_not_sufficient_for_replicated_eviction(self):
        # Actual eligibility expression is cd.host_lock_ref == 0. An ACK
        # decrements that ref. One late ACK therefore changes the victim set.
        tree_path=SOURCE.parent/'unified_cache/unified_tree_core.py'
        tree=ast.parse(tree_path.read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_can_reclaim_full_host_duplicate')
        ns={'BASE_COMPONENT_TYPE':0};mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod),str(tree_path),'exec'),ns)
        root=object();obj=types.SimpleNamespace(root_node=root)
        def candidate(lock):return types.SimpleNamespace(component_data=[types.SimpleNamespace(value=1,host_value=1,host_lock_ref=lock)],write_through_pending_id=None,load_back_pending_id=None)
        self.assertTrue(ns[node.name](obj,candidate(0)))
        self.assertFalse(ns[node.name](obj,candidate(1)))

if __name__=='__main__': unittest.main(verbosity=2)
