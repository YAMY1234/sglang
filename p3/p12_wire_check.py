"""Fail closed on live P12 before/after D snapshots from one frozen P source.

P: P12=1, SGLANG_GDN_PD_P12_WIRE_REPLAY=1, STAGING_PROOF_DIR and
SGLANG_FLASHNEXT_PD_STAGING_PROOF_TENSORS=1. D: the two staging proof envs only.
Use short (<=2k) and long (>=4k), including non-page-aligned prompt lengths.
The replay gate is intentionally ineligible for timing measurements.
"""
import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path


def check(paths, expected_ranks=(0,1), require_lengths=True, require_cuda=True):
    import torch
    rows=defaultdict(dict)
    for path in paths:
        data=torch.load(path,map_location='cpu',weights_only=True)
        suffix=next((name for name in ('before','after') if data['rid'].endswith('-p12-'+name)),None)
        if suffix is None:
            continue
        if require_cuda and not data.get('source_device','').startswith('cuda'):
            raise ValueError(f'live wire gate requires CUDA staging snapshots: {path}')
        key=(data['rid'].removesuffix('-p12-'+suffix),data['rank'],data['chunk_index'])
        label=(data['role'],suffix)
        if label in rows[key]:raise ValueError(f'duplicate proof {key} {label}')
        rows[key][label]=data
    if not rows:raise ValueError('no P12 frozen-source replay tensor snapshots')
    counts=defaultdict(int);lengths=set();categories=set();ranks=set()
    required={('prefill','before'),('prefill','after'),('decode','before'),('decode','after')}
    for key,pair in rows.items():
        if set(pair)!=required:raise ValueError(f'incomplete P/D before/after proof: {key} {set(pair)}')
        reference=pair['prefill','before']
        for label,data in pair.items():
            if data['prompt_tokens']!=reference['prompt_tokens'] or set(data['fields'])!=set(reference['fields']):
                raise ValueError(f'payload field identity changed: {key} {label}')
            for field,value in reference['fields'].items():
                if not torch.equal(value,data['fields'][field]):
                    raise ValueError(f'wire tensor differs: {key} {label} {field}')
                categories.add(field[1]);counts[label]+=1
        lengths.add(reference['prompt_tokens']);ranks.add(reference['rank'])
    for name in ('gdn_factored_a','gdn_factored_U','gdn_factored_W','gdn_factored_count',
                 'pd_h31','pd_boundary_inputs','pd_boundary_position','pd_boundary_valid'):
        if not any(name in field for field in categories):raise ValueError(f'missing payload category {name}')
    if not {'K','V'}<=categories:raise ValueError('missing KV page payloads')
    if set(expected_ranks)!=ranks:raise ValueError(f'expected TP ranks {expected_ranks}, got {ranks}')
    if require_lengths and not (any(n<=2048 for n in lengths) and any(n>=4096 for n in lengths)
                               and any(n%64 for n in lengths)):
        raise ValueError(f'need <=2k, >=4k and non-page-aligned lengths, got {lengths}')
    return dict(passed=True,groups=len(rows),ranks=sorted(ranks),prompt_lengths=sorted(lengths),
                tensor_comparisons=sum(counts.values()),categories=sorted(categories),
                scope='real received byte tensors; same P source across both transports',timing_eligible=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--snapshots',action='append')
    selection.add_argument('--negative',action='append',help='event + abort live receipt globs')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    paths=sorted({path for pattern in (args.snapshots or args.negative) for path in glob.glob(pattern,recursive=True)})
    try:
        result=check(paths) if args.snapshots else check_negative(paths)
    except (ValueError,RuntimeError) as error:
        result=dict(passed=False,error=str(error))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('P12_LIVE_WIRE_TORCH_EQUAL_'+('PASS' if result['passed'] else 'FAIL'),
          json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)


def check_negative(paths,expected_ranks=(0,1),expected_modes=('event','abort'),require_cuda=True):
    import torch
    groups=defaultdict(dict)
    for path in paths:
        data=torch.load(path,map_location='cpu',weights_only=True)
        if not data.get('rid','').startswith('p12-negative-'):continue
        if 'phase' not in data:continue
        if require_cuda and not data.get('source_device','').startswith('cuda'):
            raise ValueError(f'live negative gate requires CUDA receipts: {path}')
        key=(data['rid'],data['rank'],data['generation'],data['chunk_index'])
        if data['phase'] in groups[key]:raise ValueError(f'duplicate negative phase {key}')
        groups[key][data['phase']]=data
    coverage=set()
    for key,phases in groups.items():
        if set(phases)!={'reserved','injected','drained'}:
            raise ValueError(f'incomplete abort drain or unexpected READY: {key} {set(phases)}')
        injected=phases['injected'];mode=injected['mode']
        if not injected['backend_kv_completed'] or injected['kv_bytes']<=0:
            raise ValueError(f'negative did not exercise backend KV: {key}')
        if mode=='event' and not injected.get('event_rejected'):
            raise ValueError(f'incomplete publication accepted: {key}')
        a,b=phases['reserved']['fields'],phases['drained']['fields']
        if not a or set(a)!=set(b) or any(not torch.equal(value,b[field]) for field,value in a.items()):
            raise ValueError(f'D destination changed before READY: {key}')
        coverage.add((mode,key[1]))
    if coverage!={(mode,rank) for mode in expected_modes for rank in expected_ranks}:
        raise ValueError(f'incomplete negative mode/rank coverage: {coverage}')
    return dict(passed=True,groups=len(groups),coverage=sorted(coverage),
        scope='backend KV completed; incomplete publication/native abort; D rows unchanged after native drain',
        timing_eligible=False)


if __name__=='__main__':main()
