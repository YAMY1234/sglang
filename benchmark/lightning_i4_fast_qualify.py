"""One allocated node: isolate factor gauge/full factor and packed shallow gates.

Failed variants are still timed ONLY as explicitly unqualified diagnostics.
No failed variant is promoted, and the baseline must reproduce prior 24/24.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from lightning_dense_bridge_qualify import atomic,http,worker


def compare(rows, reference):
    lhs=[r['output_ids'] for b in rows for r in b['responses']]
    rhs=[r['output_ids'] for b in reference for r in b['responses']]
    assert len(lhs)==len(rhs) and all(len(r)==65 for r in lhs+rhs)
    return dict(first=sum(a[0]==b[0] for a,b in zip(lhs,rhs)),exact=sum(a==b for a,b in zip(lhs,rhs)),
        tokens=sum(sum(x==y for x,y in zip(a,b)) for a,b in zip(lhs,rhs)),requests=len(lhs),
        total_tokens=sum(map(len,lhs)),passed=lhs==rhs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();plan=json.loads(args.plan.read_text());source=Path(plan['factored_source'])
    for name,digest in plan['source_file_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    args.out.mkdir(parents=True,exist_ok=True)
    prompts=json.loads((args.reference/'prompts.json').read_text())['input_ids']
    variable=json.loads((args.reference/'batched/variable-n1.json').read_text())['rows']
    standard=json.loads((args.reference/'batched/n1.json').read_text())['rows']
    batches=[prompts[i:i+8] for i in range(0,200,8)]
    atomic(args.out/'input-manifest.json',dict(reference=str(args.reference),sha256={
        str(p.relative_to(args.reference)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (args.reference/'prompts.json',args.reference/'batched/variable-n1.json',args.reference/'batched/n1.json')}))

    def evaluate(label):
        def run(url,_,root,env):
            probe=root/'probe';probe.mkdir(exist_ok=True)
            def generate(inputs,n=1):
                return http(url,'/generate',dict(input_ids=inputs,sampling_params=dict(
                    temperature=0,max_new_tokens=n,ignore_eos=True)))
            modes=[('original',None)] if label=='original' else [
                ('reference',dict(commit='reference',shallow=False)),
                ('qr',dict(commit='qr',shallow=False)),
                ('device',dict(commit='device',shallow=False)),
                ('device_shallow',dict(commit='device',shallow=True)),
                ('qr_shallow',dict(commit='qr',shallow=True)),
                ('shallow',dict(commit='reference',shallow=True))]
            report={}
            for name,config in modes:
                if config is not None:atomic(probe/'i4-config.json',config)
                variable_rows=[];standard_rows=[]
                for ref in variable:
                    variable_rows.append(dict(input_ids=ref['input_ids'],responses=generate(ref['input_ids'],65)))
                variable_check=compare(variable_rows,variable)
                # Same123 standard requests, including repeated slot reuse.
                for repeat in range(3):
                    for b in (1,8,32):
                        standard_rows.append(dict(round=repeat,B=b,responses=generate([[1000]*128]*b,65)))
                standard_check=compare(standard_rows,standard)
                qualified=standard_check['passed'] and variable_check['passed']
                report[name]=dict(variable=variable_check,standard=standard_check,n1_passed=qualified,
                    performance_scope='N1-qualified candidate; M1 separately required' if qualified else 'UNQUALIFIED DIAGNOSTIC ONLY')
                atomic(root/(name+'-n1.json'),dict(variable=variable_rows,standard=standard_rows))
                atomic(root/'qualification.json',report)
                if name=='reference' and not qualified:
                    raise RuntimeError('qualified compatibility baseline did not reproduce; stop all variants')
                for b in batches:generate(b)
                responses=[]
                for repeat in range(2):
                    for i in (range(25) if repeat==0 else reversed(range(25))):
                        tag=f'{name}-B8-r{repeat}-p{i}'
                        atomic(probe/'trigger.json',dict(tag=tag,mode='events',all_forward_modes=True,
                            synchronize_prefill=True))
                        responses.append(dict(tag=tag,responses=generate(batches[i])))
                        (probe/'trigger.json').unlink()
                # Stage events one representative batch, outside scored controls.
                atomic(probe/'trigger.json',dict(tag=name+'-stages',mode='i4-stages',all_forward_modes=True))
                generate(batches[9]);(probe/'trigger.json').unlink()
                atomic(root/(name+'-prefill.json'),dict(complete=True,rows=responses))
                print('I4_FAST',name,json.dumps(report[name]),flush=True)
            if (probe/'i4-config.json').exists():(probe/'i4-config.json').unlink()
            atomic(root/'complete.json',dict(complete=True))
        return run

    def numerical():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            TRITON_CACHE_DIR=str(args.out/'numerical-triton'),NVIDIA_TF32_OVERRIDE='0')
        failed=[]
        for command,logfile in [
            ([sys.executable,str(source/'test/manual/duet/test_mamba2_prefill_commit.py'),'--out',str(args.out/'commit-numerical.json')], 'commit-test.log'),
            ([sys.executable,str(source/'benchmark/lightning_factored_mgs_sweep.py'),'--iterations','3','--out',str(args.out/'m1')], 'm1.log')]:
            with (args.out/logfile).open('w') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:failed.append(dict(logfile=logfile,exit=result.returncode))
        atomic(args.out/'numerical-exits.json',failed)
        if failed:raise RuntimeError(repr(failed))
    failures={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(worker,dict(plan,batched_extend=True,instrument_forward=True),arm,gpu,
            args.out/label,micro=False,gsm=False,evaluate=evaluate(label)):label
            for label,arm,gpu in [('original','original-off',0),('candidate','factored',1)]}
        futures[pool.submit(numerical)]='numerical'
        for f in concurrent.futures.as_completed(futures):
            try:f.result()
            except Exception as e:failures[futures[f]]=repr(e)
            atomic(args.out/'collection.json',dict(complete=False,failures=failures))
    atomic(args.out/'collection.json',dict(complete=not failures,failures=failures))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
