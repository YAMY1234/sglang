"""One loaded model, unscored shadow comparison and prefix/tail ablations."""
import argparse
import json
from pathlib import Path

from lightning_dense_bridge_qualify import atomic, http, worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--modes', default='batched,serial_factor,reference_emitters,reference_emitters_serial_factor,current,shadow')
    args = ap.parse_args()
    plan = dict(json.loads(args.plan.read_text()), batched_extend=True, instrument_forward=True)
    reference = json.loads(args.reference.read_text())['rows']

    def evaluate(url, _, root, env):
        probe = root/'probe'
        probe.mkdir(exist_ok=True)
        result = dict(complete=False, modes={})
        for mode in args.modes.split(','):
            rows=[]
            for i, ref in enumerate(reference):
                atomic(probe/'i4-diagnostic.json', dict(mode=mode, tag=f'{mode}-{i}'))
                response = http(url, '/generate', dict(input_ids=ref['input_ids'],
                    sampling_params=dict(temperature=0, max_new_tokens=65, ignore_eos=True)))
                rows.append(dict(input_ids=ref['input_ids'], responses=response))
                atomic(root/(mode+'.json'), dict(complete=False, rows=rows))
            atomic(root/(mode+'.json'), dict(complete=True, rows=rows))
            exact=first=tokens=0
            for x,y in zip(rows,reference):
                for a,b in zip(x['responses'],y['responses']):
                    u,v=a['output_ids'],b['output_ids']
                    assert len(u)==len(v)==65
                    exact+=u==v; first+=u[0]==v[0]; tokens+=sum(i==j for i,j in zip(u,v))
            result['modes'][mode]=dict(exact=exact,first=first,tokens=tokens,total=1560)
            atomic(root/'diagnostic.json',result)
            print(mode,result['modes'][mode],flush=True)
        (probe/'i4-diagnostic.json').unlink()
        result['complete']=True
        atomic(root/'diagnostic.json',result)
    worker(plan,'factored',0,args.out,micro=False,gsm=False,evaluate=evaluate)


if __name__=='__main__':
    main()
