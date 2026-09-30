"""Full 32x256 repeated-reference diagnostic, changing only MoE index_add_.

No SGLang or official GSM process is changed. This does not replace the six-set
acceptance gate or relabel its original reference scores.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

from lightning_sgl_guard import reference_worker, validate_windows
from lightning_sgl_stage2 import save, now


def install_deterministic_reduction():
    import torch
    import twinstar.models.moe as moe
    import twinstar.nemotron_h.model as nemotron
    original='return torch.zeros(n, x2.shape[-1], dtype=torch.float32, device=x2.device).index_add_(0, rows, out_sorted * win[:, None])'
    replacement='return deterministic_reduce(torch.zeros(n, x2.shape[-1], dtype=torch.float32, device=x2.device), rows, out_sorted * win[:, None])'
    source=inspect.getsource(moe.sorted_dispatch)
    if source.count(original)!=1:raise RuntimeError('pinned reference reduction source differs')
    def deterministic_reduce(result, rows, values):
        saved=torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            return result.index_add_(0,rows,values)
        finally:torch.use_deterministic_algorithms(saved)
    scope={**vars(moe),'deterministic_reduce':deterministic_reduce}
    exec(compile(source.replace(original,replacement),'<diagnostic-only-moe-reduction>','exec'),scope)
    nemotron.sorted_dispatch=scope['sorted_dispatch']
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['normal','deterministic-reduction'],required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--duet',required=True)
    parser.add_argument('--windows',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    result=dict(cell='full-reference-stability-diagnostic',arm=args.arm,status='running',started_at=now())
    start=time.monotonic()
    try:
        if args.arm=='deterministic-reduction':result['original_dispatch_sha256']=install_deterministic_reduction()
        windows=validate_windows(json.loads(args.windows.read_text()))
        reference_worker(args,windows)
        cells=[json.loads((args.out/f'reference{i}.json').read_text()) for i in (1,2)]
        if not all(c['complete'] and len(c['windows'])==32 for c in cells):raise RuntimeError('incomplete reference diagnostic')
        scores=[[v for row in c['windows'] for v in row['losses']] for c in cells]
        result.update(status='measured',windows=32,continuation_tokens=len(scores[0]),
                      means=[sum(v)/len(v) for v in scores],bitwise=scores[0]==scores[1],
                      changed_tokens=sum(a!=b for a,b in zip(*scores)),max_token_delta=max(abs(a-b) for a,b in zip(*scores)))
        result['repeat_mean_delta']=abs(result['means'][0]-result['means'][1])
    except Exception as exc:
        result.update(status='error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(finished_at=now(),wall_seconds=time.monotonic()-start)
        save(args.out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
