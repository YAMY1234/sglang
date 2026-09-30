"""Read the real sidecar path: distinguish absent state from timeout before IO."""

import os
import queue
import types
import unittest
from unittest.mock import Mock, patch

from test_pp_commit_integration import PoolName, method


class SidecarReadTest(unittest.TestCase):
    def exercise(self, terminated, hits):
        f = method(
            "mem_cache/hybrid_cache/hybrid_cache_controller.py",
            "_page_transfer_sidecar",
        )
        log = Mock()
        f.__globals__.update(
            os=os,
            logger=log,
            PrefetchAck=lambda **kw: types.SimpleNamespace(**kw),
            count_pool_hits=lambda value: value,
        )
        transfer = types.SimpleNamespace(
            indices_from_pool=None, name=PoolName.MAMBA, keys=["tail"]
        )
        op = types.SimpleNamespace(
            pool_transfers=[transfer],
            is_terminated=lambda: terminated,
            hash_value=["tail"],
            request_id="request",
        )
        cache = types.SimpleNamespace(
            _sync_trailing_keys=Mock(),
            _resolve_sidecar_nonkv_derived_pool_transfers=Mock(),
            storage_backend=types.SimpleNamespace(batch_get_v2=Mock(return_value=hits)),
            prefetch_sync_queue=queue.Queue(),
        )
        with patch.dict(os.environ, {"SGLANG_HICACHE_PP_PREFETCH_DIAG": "1"}):
            f(cache, op, 1)
        ack = cache.prefetch_sync_queue.get_nowait()
        self.assertEqual(ack.pool_hits, {} if terminated else hits)
        self.assertTrue(cache.prefetch_sync_queue.empty())
        self.assertEqual(
            cache.storage_backend.batch_get_v2.call_count, 0 if terminated else 1
        )
        self.assertEqual(log.info.call_args.args[2], terminated)
        return cache

    def test_complete_aux_read(self):
        self.exercise(False, {"mamba": 1})

    def test_absent_aux_key(self):
        self.exercise(False, {"mamba": 0})

    def test_terminated_before_aux_still_emits_ack(self):
        self.exercise(True, {"mamba": 1})
