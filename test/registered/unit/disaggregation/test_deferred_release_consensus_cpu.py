"""Two real CPU/Gloo ranks exercise the production queue release method.

Only the GPU page allocator and ACK source are replaced. The collectives and
the queue method are the actual implementation, with independently timed ACKs.
Run directly with a CPU torch environment; no CUDA or full model is required.
"""
import ast
from datetime import timedelta
from enum import IntEnum
from http import HTTPStatus
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import time
from types import MethodType, SimpleNamespace as NS
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'python/sglang/srt/disaggregation/decode.py'
CONSENSUS = SOURCE.with_name('deferred_release_consensus.py')


def implementation():
    spec = importlib.util.spec_from_file_location('release_consensus', CONSENSUS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = next(n for n in ast.parse(SOURCE.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'DecodeTransferQueue')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('resolve_deferred_releases', 'pop_transferred', '_defer_release')]
    class Poll(IntEnum):
        Failed = 0
        Success = 1
        Bootstrapping = 2
        WaitingForInput = 3
        Transferring = 4
    env = dict(time=time, logger=logging.getLogger('release-consensus-test'),
               agree_deferred_releases=module.agree_deferred_releases,
               agree_deferred_holds=module.agree_deferred_holds, KVPoll=Poll,
               HTTPStatus=HTTPStatus, HiCacheRestoreResult=NS(FAILED=-1, PENDING=1),
               prepare_abort=lambda *a, **k: None,
               release_kv_cache=lambda req, cache, **kw: cache.append(req.rid))
    exec(compile('from __future__ import annotations\n' + '\n'.join(map(ast.unparse, methods)),
                 str(SOURCE), 'exec'), env)
    return module, env


def rank_main(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=25))
    try:
        module, env = implementation()
        method = env['resolve_deferred_releases']
        records = []
        # A local abort notification on either rank must retain on both ranks.
        for notifying_rank in (0, 1):
            holds = module.agree_deferred_holds([rank == notifying_rank, False], dist.group.WORLD)
            assert holds == [1, 0], holds
        # Exercise the true pop_transferred keyword entry, including one-sided
        # abort delivery. It must construct identical deferred lists, not just
        # agree on release after an artificial identical list was supplied.
        pop_records = []
        for enabled in (False, True):
            armed = []
            mgr = NS(enable_deferred_decode_kv_release=True,
                     register_deferred_abort_room=armed.append)
            receiver = NS(kv_mgr=mgr, abort_notified=rank == 0,
                          bootstrap_infos=[{}, {}], failure_exception=lambda: None,
                          clear=lambda: None)
            def abort():
                assert armed == [123], 'ACK tracker must be armed before sending'
                receiver.abort_notified = True
            receiver.abort = abort
            req = NS(rid='aborted', bootstrap_room=123, return_logprob=False)
            dr = NS(req=req, kv_receiver=receiver, hicache_restore_status=0,
                    metadata_buffer_index=7)
            freed = []
            q = NS(queue=[dr], tp_rank=rank, enable_staging=False,
                   enable_deferred_kv_release=enabled, gloo_group=dist.group.WORLD,
                   _deferred_releases=[], deferred_kv_release_timeout=30,
                   _poll_with_metadata_gate=lambda: [env['KVPoll'].Failed],
                   _clean_hicache_prefetch_resources=lambda req: None,
                   scheduler=NS(enable_decode_hicache=False, enable_hisparse=False,
                                output_streamer=NS(stream_output=lambda *args: None),
                                metrics_reporter=NS(enable_metrics=False)),
                   tree_cache=[], metadata_buffers=NS(bootstrap_room={7: 123}),
                   req_to_metadata_buffer_idx_allocator=NS(free=freed.append))
            q._defer_release = MethodType(env['_defer_release'], q)
            env['pop_transferred'](q, rids_to_check=None)
            assert q.queue == []
            assert len(q._deferred_releases) == int(enabled)
            assert freed == ([] if enabled else [7])
            assert q.tree_cache == ([] if enabled else ['aborted'])
            if enabled and rank == 1:
                assert armed == [123] and receiver.abort_notified
            pop_records.append(dict(enabled=enabled, held=len(q._deferred_releases),
                                    metadata_freed=freed))
        for factored, expired, first_rank in ((True, False, 0), (True, False, 1),
                                              (True, True, 0), (False, False, 0)):
            ready = [rank == first_rank]
            receiver = NS(kv_mgr=NS(is_abort_release_safe=lambda room, count: ready[0]))
            req = NS(req=NS(bootstrap_room=123), kv_receiver=receiver)
            deadline = time.monotonic() + (-1 if expired else 30)
            q = NS(_deferred_releases=[(req, deadline, 7, 2)],
                   scheduler=NS(req_to_token_pool=NS(factored_gdn_pool=object() if factored else None)),
                   gloo_group=dist.group.WORLD, deferred_kv_release_timeout=30, released=[])
            q._do_release = lambda req, idx: q.released.append(idx)
            q.resolve_deferred_releases = MethodType(method, q)
            q.resolve_deferred_releases()
            assert q.released == [] and len(q._deferred_releases) == 1
            # Full-pool preallocation now admits the same number on each rank.
            queue = torch.zeros(16 + len(q.released), dtype=torch.uint8)
            dist.all_reduce(queue, group=dist.group.WORLD)
            ready[0] = True
            q.resolve_deferred_releases()
            assert q.released == [7] and q._deferred_releases == []
            queue = torch.zeros(16 + len(q.released), dtype=torch.uint8)
            dist.all_reduce(queue, group=dist.group.WORLD)
            # Empty queue performs no collective and never frees twice.
            q.resolve_deferred_releases()
            assert q.released == [7]
            records.append(dict(factored=factored, expired=expired,
                                first_ack_rank=first_rank, first_release_count=0,
                                final_release_count=1, following_poll_lengths=[16, 17]))
        # Dense-state timeout must also agree: one expired rank alone can't free.
        ready = [False]
        req = NS(req=NS(bootstrap_room=456), kv_receiver=NS(kv_mgr=NS(
            is_abort_release_safe=lambda room, count: ready[0])))
        q = NS(_deferred_releases=[(req, time.monotonic() + (-1 if rank == 0 else 30), 8, 2)],
               scheduler=NS(req_to_token_pool=NS(factored_gdn_pool=None)),
               gloo_group=dist.group.WORLD, deferred_kv_release_timeout=30, released=[])
        q._do_release = lambda req, idx: q.released.append(idx)
        q.resolve_deferred_releases = MethodType(method, q)
        q.resolve_deferred_releases()
        assert not q.released
        q._deferred_releases = [(req, time.monotonic() - 1, 8, 2)]
        q.resolve_deferred_releases()
        assert q.released == [8]
        Path(output, f'rank{rank}.json').write_text(json.dumps(dict(
            rank=rank, complete=True, cases=records, actual_pop_transferred=pop_records,
            one_sided_hold=True,
            dense_timeout_consensus=True, real_gloo=True, actual_queue_method=True)))
    finally:
        dist.destroy_process_group()


class TestDeferredReleaseConsensus(unittest.TestCase):
    def test_staggered_ack_and_abort_with_real_gloo(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(rank_main, args=(str(Path(tmp, 'rendezvous')), tmp), nprocs=2, join=True)
            results = [json.loads(Path(tmp, f'rank{i}.json').read_text()) for i in range(2)]
            self.assertTrue(all(x['complete'] for x in results))
            print(json.dumps(dict(complete=True, ranks=results)))


if __name__ == '__main__':
    unittest.main()
