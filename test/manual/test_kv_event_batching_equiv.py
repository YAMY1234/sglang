"""Payload equivalence of the ported SGLANG_HICACHE_BATCH_EVENTS recorder (board a10780c76eab) on our base (b8dd9757 =
image 043cfc4 events.py): stock record_store vs _record_store_fast, PRODUCTION objects (radix_cache.RadixKey over
array('l') token ids, TreeNode, hashes computed by the recorder via compute_node_hash_values / event hash path),
compared as msgpack bytes of EventBatch(take()). usage: test_batch_events_equiv.py <tree/python with the ported files>"""
import os, random, sys
from array import array
if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1])
os.environ.pop('SGLANG_HICACHE_BATCH_EVENTS', None)
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.disaggregation.kv_events import EventBatch, StorageMedium
import msgspec
enc = msgspec.msgpack.Encoder()

def tree(spec, bigram, extra_key, salt):
    """spec = list of (parent_index or -1 for root, n_logical_tokens); returns nodes (index-aligned) with the root first."""
    rnd = random.Random(hash((tuple(spec), bigram, extra_key, salt)) & 0xffffffff)
    root = TreeNode(); root.key = RadixKey(array('l'), extra_key, is_bigram=bigram, cache_salt=salt); root.hash_value = []
    nodes = []
    for parent, n in spec:
        nd = TreeNode(); nd.parent = root if parent < 0 else nodes[parent]
        raw = array('l', (rnd.randrange(1, 151000) for _ in range(n + (1 if bigram else 0))))
        nd.key = RadixKey(raw, extra_key, is_bigram=bigram, cache_salt=salt)
        nodes.append(nd)
    return nodes

def run(fast, spec, ops, bigram, extra_key, salt, dynamo):
    nodes = tree(spec, bigram, extra_key, salt)
    rec = KVCacheEventRecorder(enabled=True, page_size=64, dynamo_format=dynamo)
    if fast:
        rec.record_store = rec._record_store_fast
    for kind, i, medium, sess in ops:
        if kind == 'store':
            rec.record_store(nodes[i], medium, session_id=sess)
        elif kind == 'remove':
            rec.record_remove(nodes[i], medium)
        else:
            rec.record_all_cleared()
    ev = rec.take()
    return enc.encode(EventBatch(ts=0.0, events=ev)), len(ev), [n.hash_value for n in nodes]

def main():
    """240 random cases; returns (cases, mismatches). Callable from the in-image probe check."""
    rnd = random.Random(20261117); cases = fails = 0
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
        bigram = rnd.random() < 0.6
        extra_key = rnd.choice([None, None, 'lora-x'])
        salt = rnd.choice([None, None, 'salt-1'])
        dynamo = rnd.random() < 0.8
        a = run(False, spec, ops, bigram, extra_key, salt, dynamo)
        b = run(True, spec, ops, bigram, extra_key, salt, dynamo)
        cases += 1
        if a[0] != b[0] or a[2] != b[2]:
            fails += 1
            print('MISMATCH case', c, dict(spec=spec, bigram=bigram, extra_key=extra_key, salt=salt, dynamo=dynamo, events=(a[1], b[1])))
    return cases, fails


if __name__ == '__main__':
    cases, fails = main()
    print(f'cases checked {cases}; byte-identical {cases - fails}; mismatches {fails}')
    sys.exit(1 if fails or cases == 0 else 0)
