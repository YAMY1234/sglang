"""Wire equivalence of SGLANG_KV_EVENTS_RAW_TOKENS (bigram token_ids pre-encoded as msgspec.Raw, on top of H466 batching):
stock record_store vs batched vs batched + Raw, PRODUCTION objects (RadixKey over array('l'), TreeNode, recorder-computed
hashes), compared as msgpack bytes of EventBatch(take()), plus the canonical KV-record digest. LEAD 11-17 18:1xZ step 1.
In-image use (probe step 0): import this file and call main() -> (checked, fails); the canonical-digest checks need the kvl
publisher (canonical_kv_records) and are skipped, and counted, when it is absent.
usage: test_raw_tokens_equiv.py <tree/python with the raw-tokens events.py> <dir of test_kv_event_batching_equiv.py>"""
import os, random, sys
from array import array
if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop('SGLANG_HICACHE_BATCH_EVENTS', None); os.environ.pop('SGLANG_KV_EVENTS_RAW_TOKENS', None)
import msgspec
from sglang.srt.mem_cache import events as EV
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.disaggregation.kv_events import EventBatch, StorageMedium
import test_kv_event_batching_equiv as BASE
enc = msgspec.msgpack.Encoder()
checked = fails = 0
stats = dict(raw_events=0, list_events=0)


def check(cond, what):
    global checked, fails
    checked += 1
    if not cond:
        fails += 1; print('FAIL', what)


def run(mode, spec, ops, bigram, extra_key, salt, dynamo, nodes=None):
    nodes = nodes or BASE.tree(spec, bigram, extra_key, salt)
    rec = KVCacheEventRecorder(enabled=True, page_size=64, dynamo_format=dynamo)
    if mode in ('fast', 'raw'):
        rec.record_store = rec._record_store_fast
    rec._raw_tokens = mode == 'raw'
    for kind, i, medium, sess in ops:
        if kind == 'store':
            rec.record_store(nodes[i], medium, session_id=sess)
        elif kind == 'remove':
            rec.record_remove(nodes[i], medium)
        else:
            rec.record_all_cleared()
    ev = rec.take()
    for e in ev:
        t = getattr(e, 'token_ids', None)
        if isinstance(t, msgspec.Raw):
            stats['raw_events'] += 1
        elif isinstance(t, list):
            stats['list_events'] += 1
        check(not isinstance(t, EV._RawTokenParts), 'take() leaves no pending Raw parts')
    return enc.encode(EventBatch(ts=0.0, events=ev)), ev


skipped = dict(digest=0)


def digest(ev):
    """Canonical records of the in-process events (the kvl publisher's canonical_kv_records, Raw-aware); None if absent."""
    try:
        from sglang.srt.managers.scheduler_components.kv_events_publisher import canonical_kv_records
    except ImportError:
        skipped['digest'] += 1
        return None
    return list(canonical_kv_records(ev))


def main():
    global checked, fails
    checked = fails = 0
    # R1: the 240 production-object cases of the H466 equivalence test (same generator, seed and sizes), three ways
    rnd = random.Random(20261117)
    sizes = [1, 7, 63, 64, 65, 127, 128, 129, 640, 4096, 4097, 16384]
    for c in range(240):
        n_nodes = rnd.randint(1, 6)
        spec = [(-1 if i == 0 or rnd.random() < 0.2 else rnd.randrange(i), rnd.choice(sizes)) for i in range(n_nodes)]
        ops = []
        for _ in range(rnd.randint(1, 10)):
            r = rnd.random()
            kind = 'store' if r < 0.7 else ('remove' if r < 0.95 else 'clear')
            ops.append((kind, rnd.randrange(n_nodes), rnd.choice([None, StorageMedium.GPU, StorageMedium.CPU]),
                        rnd.choice([None, None, 'sess-a', 'sess-b'])))
        bigram = rnd.random() < 0.6; extra_key = rnd.choice([None, None, 'lora-x']); salt = rnd.choice([None, None, 'salt-1'])
        dynamo = rnd.random() < 0.8
        a, _ = run('stock', spec, ops, bigram, extra_key, salt, dynamo)
        b, _ = run('fast', spec, ops, bigram, extra_key, salt, dynamo)
        r_, ev_raw = run('raw', spec, ops, bigram, extra_key, salt, dynamo)
        check(a == b == r_, f'case {c}: stock == batched == batched+Raw wire (bigram={bigram})')
    print('R1 240 cases:', stats)
    check(stats['raw_events'] > 0, 'R1 exercised the Raw path')

    # R2 coalescing: a parent-linked chain mixing short (list) and long (Raw) runs -> one BlockStored with pending parts
    parts_raw = [0]
    _raw_fn = EV._RawTokenParts.raw
    def _counted_raw(self):
        parts_raw[0] += 1; return _raw_fn(self)
    EV._RawTokenParts.raw = _counted_raw
    for sizes_chain in ([4096, 7, 640, 65, 16384], [7, 4096], [4096, 4096, 4096], [63, 1, 300]):
        spec = [(-1 if i == 0 else i - 1, n) for i, n in enumerate(sizes_chain)]
        ops = [('store', i, StorageMedium.GPU, None) for i in range(len(spec))]
        a, ev_a = run('stock', spec, ops, True, None, None, True)
        r_, ev_r = run('raw', spec, ops, True, None, None, True)
        check(a == r_, f'R2 chain {sizes_chain}: stock == Raw wire')
        check(digest(ev_a) == digest(ev_r), f'R2 chain {sizes_chain}: canonical records equal (Raw decoded)')
        check(len(ev_r) <= len(ev_a), f'R2 chain {sizes_chain}: coalesced into {len(ev_r)} event(s)')

    check(parts_raw[0] > 0, f'R2 exercised Raw-part coalescing ({parts_raw[0]} joins)')
    EV._RawTokenParts.raw = _raw_fn

    # R3 short final page (n % 64 != 0) and plain keys (non-bigram: untouched list path)
    for n in (300, 4097, 16383):
        for bigram in (True, False):
            spec = [(-1, n)]; ops = [('store', 0, None, None)]
            a, ev_a = run('stock', spec, ops, bigram, None, None, True)
            r_, ev_r = run('raw', spec, ops, bigram, None, None, True)
            check(a == r_, f'R3 n={n} bigram={bigram}: wire equal')
            if not bigram:
                check(all(isinstance(e.token_ids, list) for e in ev_r), f'R3 n={n}: plain keys keep the list form')

    # R4 fallback: a token outside [0, 2**32) keeps the list form and the wire stays equal
    spec = [(-1, 640)]; nodes = BASE.tree(spec, True, None, None); nodes[0].key.token_ids[100] = 1 << 33
    a, _ = run('stock', spec, [('store', 0, None, None)], True, None, None, True, nodes=nodes)
    nodes = BASE.tree(spec, True, None, None); nodes[0].key.token_ids[100] = 1 << 33
    r_, ev_r = run('raw', spec, [('store', 0, None, None)], True, None, None, True, nodes=nodes)
    check(a == r_ and isinstance(ev_r[0].token_ids, list), 'R4 out-of-range token: list fallback, wire equal')

    # R5 mutants: (a) the uint16 marker, (b) a dropped pair -> the wire must differ
    orig = EV._bigram_pairs_body
    def mut_marker(ids, s, e):
        b = orig(ids, s, e); return None if b is None else b.replace(b'\xcd', b'\xce', 1)
    def mut_drop(ids, s, e):
        b = orig(ids, s, e); return None if b is None else b[:-3]
    for name, m in (('marker', mut_marker), ('drop', mut_drop)):
        EV._bigram_pairs_body = m
        spec = [(-1, 4096)]; ops = [('store', 0, None, None)]
        a, _ = run('stock', spec, ops, True, None, None, True)
        r_, _ = run('raw', spec, ops, True, None, None, True)
        check(a != r_, f'R5 mutant {name}: wire differs (caught)')
    EV._bigram_pairs_body = orig

    # R6 in-server verify + stats: good bytes pass and decrement; a mutant raises RuntimeError
    spec = [(-1, 4096), (-1, 300), (-1, 2048)]; ops = [('store', i, None, None) for i in range(3)]
    nodes = BASE.tree(spec, True, None, None)
    rec = KVCacheEventRecorder(enabled=True, page_size=64, dynamo_format=True)
    rec.record_store = rec._record_store_fast; rec._raw_tokens = True; rec._raw_verify_left = 5
    for kind, i, medium, sess in ops:
        rec.record_store(nodes[i], medium, session_id=sess)
    rec.take()
    check(rec._raw_verify_left == 3, f'R6 two Raw runs verified (left {rec._raw_verify_left})')
    check(rec._raw_stats[0] == 2, f'R6 stats count 2 Raw runs ({rec._raw_stats})')
    check(rec._raw_stats[2] >= 1, f'R6 stats count the short run as list ({rec._raw_stats})')
    EV._bigram_pairs_body = mut_marker
    rec = KVCacheEventRecorder(enabled=True, page_size=64, dynamo_format=True)
    rec.record_store = rec._record_store_fast; rec._raw_tokens = True; rec._raw_verify_left = 5
    try:
        rec.record_store(BASE.tree([(-1, 4096)], True, None, None)[0]); raised = False
    except RuntimeError as e:
        raised = 'differ' in str(e)
    check(raised, 'R6 verify raises on mutant bytes')
    EV._bigram_pairs_body = orig
    return checked, fails


if __name__ == '__main__':
    c, f = main()
    print(f'RAW_TOKENS_TESTS checked {c} failed {f} digest_skipped {skipped["digest"]}')
    sys.exit(1 if f or c == 0 else 0)
