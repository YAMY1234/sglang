"""One full-length diagnostic window on the otherwise idle GPU of the guard job.

This is not a guard, and cannot produce numerical PASS. The running registered
32x256 cell retains its original code and continues to completion.
"""
import argparse
import json
import os
from pathlib import Path
import time

from lightning_sgl_stage2 import now, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--duet',required=True)
    parser.add_argument('--windows',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--variant',choices=['norm','embedding','both'],default='norm')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    os.environ['LIGHTNING_PROBE_VARIANT']=args.variant
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE']='lightning_duet_probe'
    os.environ['TWINSTAR_LIGHTNING_DUET']='1';os.environ['TWINSTAR_LIGHTNING_DUET_DIR']=args.duet
    import sglang as sgl
    row=json.loads(args.windows.read_text())['windows'][0];engine=None;start=time.monotonic()
    rec=dict(cell='diagnostic-rmsnorm-rounding',variant=args.variant,numerical_guard=False,started_at=now(),job_id=os.environ.get('SLURM_JOB_ID'))
    try:
        engine=sgl.Engine(model_path=args.model,tp_size=1,dtype='bfloat16',trust_remote_code=True,
            context_length=8192,max_total_tokens=16384,max_running_requests=2,max_mamba_cache_size=8,
            mem_fraction_static=.7,disable_cuda_graph=True,disable_overlap_schedule=True,
            disable_radix_cache=True,chunked_prefill_size=-1,skip_server_warmup=True,random_seed=20260929,watchdog_timeout=1800)
        out=engine.generate(input_ids=row['prompt']+row['target'],sampling_params={'temperature':0,'max_new_tokens':1,'ignore_eos':True},
                            return_logprob=True,logprob_start_len=3839)
        scores=out['meta_info']['input_token_logprobs']
        assert len(scores)==257 and scores[0][0] is None and [s[1] for s in scores[1:]]==row['target']
        losses=[-s[0] for s in scores[1:]]
        rec.update(status='measured',nll=sum(losses)/256,losses=losses,id=row['id'],prompt_sha256=row['prompt_sha256'])
    except Exception as exc:
        rec.update(status='error',error=f'{type(exc).__name__}: {exc}')
        import traceback;traceback.print_exc()
    finally:
        if engine is not None:engine.shutdown()
        rec.update(finished_at=now(),wall_seconds=time.monotonic()-start);save(args.out/'result.json',rec);print(json.dumps(rec),flush=True)
    return int(rec['status']=='error')


if __name__=='__main__':
    raise SystemExit(main())
