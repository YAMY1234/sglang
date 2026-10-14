"""Diagnostic only: GSM forward segments and N1 for frozen MGS 3/2/1."""
import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

from lightning_dense_bridge_qualify import atomic, http, worker


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(args.plan.read_text())
    # Existing offline GSM data get a writable job-local copy before loading.
    os.environ.update(HF_HOME='/work/hf-cache',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_DATASETS_CACHE=plan['datasets_cache'],HF_MODULES_CACHE=str(args.out/'hf-modules'))
    import sys
    sys.path[:0]=plan['helper_paths']
    from lightning_factored_service_pair import prepare_client
    prep=args.out/'preflight';prep.mkdir(exist_ok=True)
    prepare_client(plan,prep,'gsm')
    import gsm_endpoint

    def run_item(item):
        label,arm,gpu,iters=item
        local=dict(plan,trunc_iterations=iters,instrument_forward=label in ('original','iter3'))
        root=args.out/label
        def evaluate(url,plan,root,env):
            if label in ('original','iter3'):
                probe=root/'probe';probe.mkdir(exist_ok=True)
                atomic(probe/'trigger.json',dict(tag='gsm200',mode='events',all_forward_modes=True))
                gsm_endpoint.run(url,plan['model'],root/'gsm200.json')
                time.sleep(1)  # drain asynchronous event writer before removing trigger
                (probe/'trigger.json').unlink()
            records=[]
            for round_id in range(3):
                for batch in (1,8,32):
                    response=http(url,'/generate',dict(input_ids=[[1000]*128 for _ in range(batch)],
                        sampling_params=dict(temperature=0,max_new_tokens=65,ignore_eos=True)))
                    assert len(response)==batch
                    records.append(dict(round=round_id,B=batch,responses=response))
                    atomic(root/'n1.json',dict(complete=False,iterations=iters,rows=records))
            atomic(root/'n1.json',dict(complete=True,iterations=iters,rows=records))
        worker(local,arm,gpu,root,micro=False,gsm=False,evaluate=evaluate)

    items=[('original','original-off',0,3),('iter3','factored',1,3),
           ('iter2','factored',2,2),('iter1','factored',3,1)]
    failures={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run_item,item):item[0] for item in items}
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as e:failures[futures[future]]=repr(e)
    atomic(args.out/'collection.json',dict(complete=not failures,failures=failures,
        scope='diagnostic only; GSM events add overhead, formal windows remain uninstrumented and MGS=3'))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
