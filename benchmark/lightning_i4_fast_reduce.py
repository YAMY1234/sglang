"""Reduce complete/partial fast-prefill qualification without promoting failed N1."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    result=dict(arms={},errors={})
    for arm in ('original','candidate'):
        root=args.root/arm
        try:
            gates=json.loads((root/'qualification.json').read_text())
            rows=[json.loads(l) for l in (root/'probe/steps.jsonl').read_text().splitlines()]
            stages=[json.loads(l) for l in (root/'probe/i4-stages.jsonl').read_text().splitlines()]
            grouped=collections.defaultdict(list)
            for r in rows:
                assert r['forward_kind']=='prefill' and r['batch_size']==8
                name=r['tag'].split('-B8-',1)[0];grouped[name].append(r)
            result['arms'][arm]={}
            for name,g in gates.items():
                if arm == 'original':
                    g = dict(g, performance_scope='Native control; N1 comparison to factor reference is disclosure, not control invalidation')
                group=grouped.get(name,[])
                complete=len(group)==50
                if complete:
                    assert len(set(r['tag'] for r in group))==50
                    if arm=='candidate':
                        expected_commit={'reference':'reference','shallow':'reference','qr':'qr',
                            'qr_shallow':'qr','device':'device','device_shallow':'device'}[name]
                        assert all(r['handoff']['prefill_commit_mode']==expected_commit for r in group)
                        assert all(r['handoff']['shallow_request_local']==('shallow' not in name) for r in group)
                        assert all(not r['handoff']['emitter_native_batched'] for r in group)
                stage=next((r for r in stages if r['tag']==name+'-stages'),None)
                wall=sum(r['prefill_completed_wall_ms'] for r in group)/(8*len(group)) if group else None
                result['arms'][arm][name]=dict(**g,performance_complete=complete,scored_batches=len(group),
                    completed_prefill_ms_per_request=wall,
                    rounds_ms_per_request=[sum(r['prefill_completed_wall_ms'] for r in group if f'-r{i}-' in r['tag'])/200 for i in (0,1)] if complete else [],
                    n1_and_perf_support=bool(complete and g['n1_passed'] and wall<=50),
                    final_support=False, final_support_note='M1 must independently pass; N1/performance alone cannot admit',
                    representative_stages={k:{field:value/8 for field,value in v.items()} for k,v in stage['stages'].items()} if stage else None,
                    handoff=stage.get('handoff') if stage else None)
        except Exception as e:result['errors'][arm]=repr(e)
    for name in ('collection.json','commit-numerical.json','numerical-exits.json'):
        p=args.root/name
        if p.exists():result[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data=json.loads(p.read_text()))
    (args.root/'fast-reduced.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:{n:{k:v for k,v in r.items() if k not in ('representative_stages','handoff')} for n,r in variants.items()}
        for a,variants in result['arms'].items()},indent=2))
    print('ERRORS',result['errors'])

if __name__=='__main__':main()
