"""Exact compact commit keys, changed metadata, and live-buffer transactions."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch


def main():
    path = Path(__file__).with_name('test_factored_prefill_graph.py')
    spec = importlib.util.spec_from_file_location('commit_transactions', path)
    tx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tx)
    cls = tx.graphs.PrefillCommitGraph
    factorize = tx.module.factorize_layers
    policy = (tx.module.ORTH_METHOD, tx.module.ORTH_WARPS_OVERRIDE, tx.module.factorize_dense)
    cases = []
    configurations = [(l, b, t, dt) for l in (2, 36) for b in (1, 2)
                      for t in (False, True) for dt in (torch.float32, torch.float64)]
    configurations += [(36, 16, t, torch.float32) for t in (False, True)]
    for layers, batch, tracked, dtype in configurations:
        pool = tx.pool(layers, 2, 16)
        dense = torch.randn(layers, batch, 2, 16, 16, dtype=dtype)
        extra = torch.randn_like(dense) if tracked else None
        pending = [(dense[i], extra[i] if tracked else None) for i in range(layers)]
        tensors = [x for row in pending for x in row if x is not None]
        full = cls._full_key(pool, tensors, factorize, policy)
        compact, size = cls._uniform_key(pool, pending, factorize, policy)
        n, shape, other, settings = compact
        shapes = ((shape, other) if other is not None else (shape,)) * n
        assert (settings[0], shapes, *settings[1:]) == full
        assert size == sum(x.numel() * x.element_size() for x in tensors) + pool.vbar.nbytes
        changed = list(pending)
        changed[-1] = (pending[-1][0].to(torch.float16), pending[-1][1])
        assert cls._uniform_key(pool, changed, factorize, policy) is None
        changed[-1] = (pending[-1][0][..., :8], pending[-1][1])
        assert cls._uniform_key(pool, changed, factorize, policy) is None
        changed[-1] = (pending[-1][0], None if tracked else pending[-1][0])
        assert cls._uniform_key(pool, changed, factorize, policy) is None
        backing = pool.a
        pool.a = pool.a.clone()
        assert cls._uniform_key(pool, pending, factorize, policy)[0] != compact
        pool.a = backing
        iterations = pool.cfg.init_iters
        pool.cfg.init_iters += 1
        assert cls._uniform_key(pool, pending, factorize, policy)[0] != compact
        pool.cfg.init_iters = iterations
        assert cls._uniform_key(pool, pending, factorize, policy + ('changed',))[0] != compact
        for flag in ('allow_tf32', 'allow_fp16_reduced_precision_reduction',
                     'allow_bf16_reduced_precision_reduction'):
            old = getattr(torch.backends.cuda.matmul, flag)
            try:
                setattr(torch.backends.cuda.matmul, flag, not old)
                assert cls._uniform_key(pool, pending, factorize, policy)[0] != compact
            finally:
                setattr(torch.backends.cuda.matmul, flag, old)
        cases.append(dict(layers=layers, batch=batch, tracked=tracked, dtype=str(dtype),
                          exact_reference_key=True, mutation_checks=9))

    pool = tx.pool(36, 2, 16)
    dense = torch.randn(36, 1, 2, 16, 16)
    extra = torch.randn_like(dense)
    pending = list(zip(dense.unbind(0), extra.unbind(0)))
    tensors = [x for row in pending for x in row]
    key = cls._full_key(pool, tensors, factorize, policy)
    compact, _ = cls._uniform_key(pool, pending, factorize, policy)
    entry = object()
    entries, fast_entries = {key: entry}, {compact: key}

    def reference():
        tensors = [x for row in pending for x in row if x is not None]
        size = sum(x.numel() * x.element_size() for x in tensors) + pool.vbar.nbytes
        return entries.get(cls._full_key(pool, tensors, factorize, policy)), size

    def candidate():
        signature, size = cls._uniform_key(pool, pending, factorize, policy)
        return entries.get(fast_entries.get(signature)), size

    assert reference() == candidate()
    funcs = dict(reference=reference, candidate=candidate)
    values = {name: [] for name in funcs}
    for pair in range(6):
        for name in (('reference', 'candidate') if pair % 2 == 0 else ('candidate', 'reference')):
            fn = funcs[name]
            for _ in range(50):
                fn()
            start = time.perf_counter()
            for _ in range(500):
                fn()
            values[name].append((time.perf_counter() - start) * 1e6 / 500)
    transactions = []
    if tx.GPU:
        os.environ['SGLANG_GDN_PREFILL_COMMIT_LOOKUP'] = '1'
        for dims in ((2, 2, 16), (36, 24, 128)):
            for batch, tracked in ((1, False), (1, True), (2, False)):
                if dims[0] == 36 and batch == 2:
                    continue
                row = tx.case(*dims, batch, tracked)
                assert row['stats']['fast_replayed'] == 2
                transactions.append(row)
    print(json.dumps(dict(complete=True, passed=True, device='cuda' if tx.GPU else 'cpu',
        cases=cases, transactions=transactions, timing=dict(pairs_us=values,
        median_us={name: statistics.median(v) for name, v in values.items()},
        scope='Real metadata lookup only; includes all tensor metadata/pointer/policy checks, excludes bind/replay and model work'))))


if __name__ == '__main__':
    main()
