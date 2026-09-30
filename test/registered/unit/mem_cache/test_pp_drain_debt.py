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

class Scalar(int):
    def item(self): return int(self)

class Tensor:
    def __init__(self, values, dtype='int32'):
        self.values = list(values); self.dtype = dtype; self.shape = (len(self.values),)
    def clone(self): return Tensor(self.values, self.dtype)
    def tolist(self): return list(self.values)
    def __getitem__(self, index):
        return Tensor(self.values[index], self.dtype) if isinstance(index,slice) else Scalar(self.values[index])
    def __add__(self, other): return Tensor(a + b for a, b in zip(self.values, other.values))
    def __sub__(self, other): return Tensor(a - b for a, b in zip(self.values, other.values))
    def numel(self): return len(self.values)
    def element_size(self): return 8 if self.dtype == 'int64' else 4
    def reshape(self, *shape): return self
    def to(self, dtype): return Tensor(self.values, dtype)
    def copy_(self, value): self.values[:] = value.values

class Mailbox(dict):
    def empty(self): return all(q.empty() for q in self.values())
    def channel(self, tag): return self.setdefault(tag,queue.Queue())

class Work:
    def __init__(self): self.done = False
    def wait(self): assert self.done, "Unpaired send must not be treated as complete"
    def is_completed(self): return self.done

class Chain:
    def __init__(self, size):
        self.mailboxes = [Mailbox() for _ in range(size)]
        self.tags = []
        self.events = []
    def transport(self, rank):
        def send(data, group_dst, **kwargs):
            self.events.append(('send', rank, group_dst)); self.tags.append(('send',rank,kwargs['tag']))
            work = Work()
            self.mailboxes[group_dst].channel(kwargs["tag"]).put((data.clone(), work))
            return work
        def recv(data, group_src, **kwargs):
            self.events.append(('recv', rank, group_src)); self.tags.append(('recv',rank,kwargs['tag']))
            received, work = self.mailboxes[rank].channel(kwargs["tag"]).get(timeout=0.2)
            data.values[:] = received.values
            work.done = True
        return types.SimpleNamespace(isend=send, recv=recv, ReduceOp=types.SimpleNamespace(MIN='MIN'))

def method(name, namespace):
    node = next(n for n in ast.walk(CLASS) if isinstance(n, ast.FunctionDef) and n.name == name)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), 'exec'), namespace)
    return namespace[name]

tag_ns = {}
exec((SOURCE.parents[1]/'distributed/communication_tags.py').read_text(),tag_ns)
P2PTag = tag_ns['P2PTag']

def cache(chain, rank):
    logger = logging.getLogger('pp-drain-debt-test')
    ns = {'islice': islice, 'torch': types.SimpleNamespace(tensor=lambda data, **kw: Tensor(data,kw.get('dtype','int32')), cat=lambda xs:Tensor([z for x in xs for z in x.values],'int64'), int64='int64', minimum=lambda a,b: Tensor(min(x,y) for x,y in zip(a.values,b.values)), int='int32', distributed=chain.transport(rank)), 'logger': logger, 'P2PTag': P2PTag}
    obj = types.SimpleNamespace(pp_rank=rank, pp_size=len(chain.mailboxes), pp_group='PP', work_list=[], _pp_sync_stats=dict(calls=0,sent=0,recv=0,pending=0,pending_peak=0,warned=False), _pp_drain_state={}, _l3_tier_stats={}, ongoing_backup={}, storage_existence_cache=set(), cache_controller=types.SimpleNamespace(ack_backup_queue=queue.Queue()))
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
        self.assertEqual(chain.events, [e for e in [('send',0,1),('recv',1,0),('send',1,2),('recv',2,1)] for _ in range(2)])
        self.assertTrue(all(q.empty() for q in chain.mailboxes))

    def test_persistent_debt_warns_once_with_local_operation_snapshot(self):
        chain = Chain(2); a,b = [cache(chain,i) for i in range(2)]
        a.ongoing_backup[17] = (42, 'host-lock'); b.ongoing_backup[18] = (43, 'host-lock')
        a.cache_controller.ack_backup_queue.put(types.SimpleNamespace(id=17,hash_value=['key17']))
        qa,qb=queue.Queue(),queue.Queue();qa.put('one')
        with self.assertLogs('pp-drain-debt-test', level='WARNING') as captured:
            for step in range(20):
                count=a._pp_drain_counts([qa], ['ack_backup'])[0]
                list(a.drain(qa,count)); b._pp_drain_counts([qb], ['ack_backup'])
        captured.output[:]=[x for x in captured.output if 'PP drain debt' in x]
        self.assertEqual(len(captured.output),1)
        self.assertNotIn('PP0=',captured.output[0]); self.assertIn('43',captured.output[0])
        self.assertEqual(b._l3_tier_stats['pp_drain_debt']['ack_backup']['current'],1)

    def test_per_queue_debts_are_independent_and_threshold_is_immediate(self):
        chain=Chain(2);a,b=[cache(chain,i) for i in range(2)]
        qa=[queue.Queue(),queue.Queue()];qb=[queue.Queue(),queue.Queue()]
        for _ in range(64): qa[0].put('x')
        qa[1].put('y');qb[1].put('y')
        a._pp_drain_counts(qa,['ack_backup','prefetch_hit'])
        with self.assertLogs('pp-drain-debt-test',level='WARNING') as cap:
            out=b._pp_drain_counts(qb,['ack_backup','prefetch_hit'])
        self.assertEqual(out.tolist(),[0,1]);self.assertEqual(sum("PP drain debt" in x for x in cap.output),1)
        self.assertEqual(b._pp_drain_state['prefetch_hit'][0],0)

    def test_local_tp_min_bounds_every_tp_rank(self):
        chain=Chain(2);a,b=[cache(chain,i) for i in range(2)]
        qa,qb=queue.Queue(),queue.Queue()
        for _ in range(4):qa.put(0);qb.put(0)
        a._pp_drain_counts([qa],['ack_backup'])
        b._all_reduce_attn_groups=lambda data,op:data.values.__setitem__(0,2)
        self.assertEqual(b._pp_drain_counts([qb],['ack_backup'])[0],2)
        self.assertEqual(b._pp_drain_state['ack_backup'][0],2)

    def test_peak_cycles_and_local_belief_are_retained(self):
        chain=Chain(2);a,b=[cache(chain,i) for i in range(2)]
        a.storage_existence_cache.add('PP0-only-belief')
        qa,qb=queue.Queue(),queue.Queue();qa.put('one')
        a._pp_drain_counts([qa],['ack_backup']);b._pp_drain_counts([qb],['ack_backup'])
        qa.get();a._pp_drain_counts([qa],['ack_backup']);b._pp_drain_counts([qb],['ack_backup'])
        qb.put('one');a._pp_drain_counts([qa],['ack_backup']);b._pp_drain_counts([qb],['ack_backup'])
        stats=b._l3_tier_stats['pp_drain_debt']['ack_backup']
        self.assertEqual((stats['current'],stats['cycles'],stats['peak'],stats['max_cycles']),(0,0,1,2))
        self.assertEqual(b._l3_tier_stats['pp_belief_size_local'],0)
        self.assertEqual(a._l3_tier_stats['pp_belief_size_local'],1)
        self.assertNotIn('pp_belief_size_mismatch',b._l3_tier_stats)

    def test_each_budget_round_has_one_send_and_recv_per_pipeline_edge(self):
        chain=Chain(4);ranks=[cache(chain,i) for i in range(4)]
        queues=[queue.Queue() for _ in ranks]
        for step in range(10):
            for r,q in zip(ranks,queues): r._pp_drain_counts([q],['ack_backup'])
            self.assertTrue(all(q.empty() for q in chain.mailboxes))
            for rank,r in enumerate(ranks):
                self.assertEqual(r._pp_sync_stats['sent'],(step+1)*(rank<3))
                self.assertEqual(r._pp_sync_stats['recv'],(step+1)*(rank>0))
                for work in r.work_list: work.wait()
                r.work_list.clear()
        self.assertEqual(len(chain.events),10*3*2*2)

    def test_pending_send_warning_is_nonblocking_and_once(self):
        chain=Chain(2);a,b=[cache(chain,i) for i in range(2)]
        with self.assertLogs('pp-drain-debt-test',level='WARNING') as captured:
            for _ in range(3): a._pp_sync(Tensor([0]),site=P2PTag.HIRADIX_PP_SYNC_PREFETCH)
        self.assertEqual(len(captured.output),1)
        self.assertEqual(a._pp_sync_stats['pending_peak'],6)
        for _ in range(3): b._pp_sync(Tensor([0]),site=P2PTag.HIRADIX_PP_SYNC_PREFETCH)
        self.assertTrue(all(work.is_completed() for work in a.work_list))
        self.assertEqual(a._pp_sync_stats['sent'],b._pp_sync_stats['recv'])

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

class SequenceTest(unittest.TestCase):
    def test_three_sites_have_distinct_tags_and_three_rank_sequence(self):
        chain=Chain(3); ranks=[cache(chain,i)for i in range(3)]
        sites=[P2PTag.HIRADIX_PP_SYNC_PREFETCH,P2PTag.HIRADIX_PP_SYNC_QSIZES,P2PTag.HIRADIX_PP_SYNC_READY]
        for seq,site in enumerate(sites,1):
            for rank,r in enumerate(ranks):
                data=Tensor([seq] if rank==0 else [-1])
                r._pp_sync(data,site=site);self.assertEqual(data.tolist(),[seq])
                self.assertEqual(r._pp_sync_stats['calls'],seq)
        self.assertTrue(all(m.empty()for m in chain.mailboxes))
        self.assertEqual({tag for event,rank,tag in chain.tags},set(sites)|{P2PTag.HIRADIX_PP_SYNC_HEADER})

    def test_skipped_middle_site_fails_before_waiting_for_wrong_payload(self):
        chain=Chain(3);a,b,c=[cache(chain,i)for i in range(3)]
        for r in (a,b,c):r._pp_sync(Tensor([0]),site=P2PTag.HIRADIX_PP_SYNC_READY)
        a._pp_sync(Tensor([0]),site=P2PTag.HIRADIX_PP_SYNC_PREFETCH)
        # PP1 omitted PREFETCH and tries the next event's READY. Header detects
        # it even when PP0 cannot reach/send READY until previous work drains.
        with self.assertRaisesRegex(RuntimeError,'PP sync sequence mismatch at HIRADIX_PP_SYNC_READY'):
            b._pp_sync(Tensor([0,0]),site=P2PTag.HIRADIX_PP_SYNC_READY)
        self.assertFalse(any(tag==P2PTag.HIRADIX_PP_SYNC_READY and event=='recv' and rank==1 for event,rank,tag in chain.tags[8:]))

    def test_same_site_sequence_skip_and_shape_mismatch_are_explicit(self):
        for alter in ('sequence','shape'):
            chain=Chain(3);a,b,c=[cache(chain,i)for i in range(3)]
            a._pp_sync(Tensor([7]),site=P2PTag.HIRADIX_PP_SYNC_PREFETCH)
            if alter=='sequence':b._pp_sync_stats['calls']=1
            data=Tensor([0,0]if alter=='shape'else[0])
            with self.assertRaisesRegex(RuntimeError,'sequence mismatch'):
                b._pp_sync(data,site=P2PTag.HIRADIX_PP_SYNC_PREFETCH)

    def test_all_static_sites_supply_explicit_unique_payload_tags(self):
        sites=[]
        for node in ast.walk(CLASS):
            if not isinstance(node,ast.Call)or not isinstance(node.func,ast.Attribute):continue
            if node.func.attr not in ('_pp_sync','_all_reduce'):continue
            keyword=next((kw for kw in node.keywords if kw.arg=='site'),None)
            self.assertIsNotNone(keyword,ast.unparse(node))
            if isinstance(keyword.value,ast.Attribute):sites.append(keyword.value.attr)
        self.assertEqual(len(sites),len(set(sites)))
        self.assertEqual(len(sites),7)

if __name__=='__main__': unittest.main(verbosity=2)
