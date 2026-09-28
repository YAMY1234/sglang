"""Exercise staging against a lazily captured graph on an aliased pooled stream."""
import itertools
import os
import subprocess
import sys
import threading
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


def exercise(operation):
    from sglang.srt.disaggregation import flashnext_staging_transport as transport
    from sglang.srt.mem_cache import gdn_prefill_commit_graph as commit

    torch.cuda.set_device(0)
    endpoint = transport.Endpoint.__new__(transport.Endpoint)
    endpoint.device = 0
    endpoint.stream = torch.cuda.Stream()
    # Pool rollover can return the same CUDA stream even while Endpoint owns it.
    streams = [torch.cuda.Stream() for _ in range(64)]
    capture_stream = next(s for s in streams if s.cuda_stream == endpoint.stream.cuda_stream)
    tensor = torch.zeros((1, 1, 8, 8), device='cuda')
    ready = torch.cuda.Event()
    ready.record()
    torch.cuda.synchronize()
    active, attempted, complete = threading.Event(), threading.Event(), threading.Event()
    failures = []
    manifest = NS(nbytes=256, fields=[], to_bytes=lambda: b'')
    lease = NS(slot=0)
    leases = NS(slot_bytes=4096, acquire=lambda **kw: lease, begin=lambda *a, **kw: None,
                finish=lambda *a, **kw: None, release=lambda *a: None,
                reserved_bytes=4096, peak_slots=1, peak_bytes=256)
    endpoint.storage = NS(leases=leases, buffers=[torch.empty(4096, dtype=torch.uint8, device='cuda')])
    endpoint.catalog = NS(entries=[NS(name='kv', tensor=tensor, tokens_per_row=1)],
                          pool=NS(), source_payload=lambda **kw: (manifest, {}))
    endpoint.manager = NS(attn_tp_size=1, attn_tp_rank=0, get_session_id=lambda: 'test',
                          local_ip='localhost', rank_port=1, _transfer_data=lambda *a: 0,
                          flashnext_staging_metrics=[])
    endpoint.cv = threading.Condition(threading.RLock())
    endpoint.serial = itertools.count()
    endpoint.proof_rooms = {}
    endpoint._send = lambda *a: None
    endpoint._wait = lambda key, kind: dict(nbytes=256, ptr=0, generation=0) if kind == b'SLOT' else {}
    chunk = NS(is_last_chunk=True, index_slice=slice(0, 1), num_kv_tokens=64,
               prefill_kv_indices=[0], prefill_kv_indices_by_entry=None, state_indices=[],
               room=1, wait_event=ready if operation == 'wait' else None)
    request = NS(decode_prefix_len=0, endpoint='localhost', dst_port=1, mooncake_session_id='test')
    target = NS(dst_attn_tp_size=1)

    def worker():
        torch.cuda.set_device(0)
        if not active.wait(10):
            failures.append(RuntimeError('capture never entered'))
            return
        attempted.set()
        try:
            endpoint.transfer(chunk=chunk, request=request, target=target)
        except Exception as exc:
            failures.append(exc)
        finally:
            complete.set()

    real_graph = torch.cuda.graph

    @contextmanager
    def overlap(*args, **kwargs):
        with real_graph(*args, **kwargs):
            active.set()
            assert attempted.wait(10)
            # The worker must be blocked until capture exits; no CUDA work leaks in.
            assert not complete.wait(.1), 'transfer entered an active capture: ' + str(failures)
            yield

    cfg = NS(init_method='other', factored_prefix=True, r=1, rmax=1,
             dtype=torch.float32, init_iters=1, init_oversample=0)
    pool = NS(cfg=cfg, prefix_dense=None, layer_ids=[0])
    plan = NS(pending=[None], slots=torch.zeros(1, dtype=torch.int64, device='cuda'),
              ring_dst=torch.zeros(1, dtype=torch.int64, device='cuda'))
    buffers = NS(evaluate=lambda eager: tensor.add_(1))
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        with patch.object(commit, 'CommitBuffers', return_value=buffers), \
             patch.object(torch.cuda, 'Stream', return_value=capture_stream), \
             patch.object(torch.cuda, 'graph', side_effect=overlap), \
             patch.object(transport, 'copy_payload', return_value=None):
            assert commit.PrefillCommitGraph().run(pool, 0, plan, tensor, None, None,
                                                  eager=None, policy=())
            thread.join(10)
        assert complete.is_set(), 'transfer did not finish after capture'
        assert not failures, str(failures)
        torch.cuda.synchronize()
        assert torch.equal(tensor, torch.full_like(tensor, 2))
    finally:
        active.set()
    print('PASS', operation, 'stream', endpoint.stream.cuda_stream, flush=True)


def cpu_endpoint():
    from sglang.srt.disaggregation.flashnext_staging_transport import Endpoint

    endpoint = Endpoint.__new__(Endpoint)
    endpoint.device = 0
    endpoint.stream = NS(wait_event=lambda *a: None, synchronize=lambda: None)
    endpoint.serial = itertools.count()
    endpoint.proof_rooms = {}
    endpoint.cv = threading.Condition()
    manifest = NS(nbytes=256, fields=[], to_bytes=lambda: b'')
    endpoint.catalog = NS(entries=[NS(name='kv', tensor=[NS(nbytes=256)], tokens_per_row=1)],
                          pool=NS(), source_payload=lambda **kw: (manifest, {}))
    leases = NS(slot_bytes=4096, reserved_bytes=8192, peak_slots=0, peak_bytes=256,
                begin=lambda *a, **kw: None, finish=lambda *a, **kw: None)
    active = set()

    def acquire(**kw):
        with endpoint.cv:
            slot = next(i for i in range(2) if i not in active)
            active.add(slot)
            leases.peak_slots = max(leases.peak_slots, len(active))
            return NS(slot=slot)

    def release(lease):
        with endpoint.cv:
            active.remove(lease.slot)

    leases.acquire, leases.release = acquire, release
    endpoint.storage = NS(leases=leases, buffers=[NS(data_ptr=lambda: 0) for _ in range(2)])
    endpoint.manager = NS(attn_tp_size=1, attn_tp_rank=0, get_session_id=lambda: 'test',
                          local_ip='localhost', rank_port=1, _transfer_data=lambda *a: 0,
                          flashnext_staging_metrics=[])
    endpoint._send = lambda *a: None
    chunk = NS(is_last_chunk=True, index_slice=slice(0, 1), num_kv_tokens=64,
               prefill_kv_indices=[0], prefill_kv_indices_by_entry=None, state_indices=[],
               room=1, wait_event=None)
    request = NS(decode_prefix_len=0, endpoint='localhost', dst_port=1, mooncake_session_id='test')
    return endpoint, dict(chunk=chunk, request=request, target=NS(dst_attn_tp_size=1))


class LockReleaseTest(unittest.TestCase):
    def test_transfer_exception_releases_lock(self):
        from sglang.srt.utils.graph_capture import graph_capture_lock

        endpoint, kwargs = cpu_endpoint()
        with patch.object(torch.cuda, 'set_device'), \
             patch.object(torch.cuda, 'stream', side_effect=ValueError('injected')):
            with self.assertRaisesRegex(ValueError, 'injected'):
                endpoint.transfer(**kwargs)
        acquired = []

        def check():
            ok = graph_capture_lock.acquire(timeout=1)
            acquired.append(ok)
            if ok:
                graph_capture_lock.release()

        thread = threading.Thread(target=check)
        thread.start()
        thread.join(2)
        self.assertEqual(acquired, [True])

    def test_network_waits_overlap(self):
        from sglang.srt.disaggregation import flashnext_staging_transport as transport

        endpoint, kwargs = cpu_endpoint()
        rendezvous = threading.Barrier(2)
        failures, finished = [], []

        def wait(key, kind):
            if kind == b'SLOT':
                rendezvous.wait(timeout=2)
                return dict(nbytes=256, ptr=0, generation=0)
            return {}

        endpoint._wait = wait
        event = NS(record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 0.)

        def worker():
            try:
                finished.append(endpoint.transfer(**kwargs))
            except Exception as exc:
                failures.append(exc)

        with patch.object(torch.cuda, 'set_device'), \
             patch.object(torch.cuda, 'stream', return_value=nullcontext()), \
             patch.object(torch.cuda, 'Event', return_value=event), \
             patch.object(transport, 'copy_payload', return_value=None):
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertFalse(failures, failures)
        self.assertEqual(finished, [0, 0])
        self.assertEqual(endpoint.storage.leases.peak_slots, 2)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class CaptureRaceTest(unittest.TestCase):
    def run_case(self, operation):
        child = subprocess.run([sys.executable, __file__, operation], capture_output=True,
                               text=True, timeout=40, env=os.environ.copy())
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_wait_on_uncaptured_event_during_aliased_capture(self):
        self.run_case('wait')

    def test_transfer_synchronization_during_aliased_capture(self):
        self.run_case('sync')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in ('wait', 'sync'):
        exercise(sys.argv[1])
    else:
        unittest.main()
