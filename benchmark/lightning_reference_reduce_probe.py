"""Isolate the pinned reference MoE reduction on actual fixed model activations.

This is an operator diagnostic, not an NLL/accuracy gate. The running reference
and SGLang services are separate processes and are never patched by this tool.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time


def main():
    import torch
    import twinstar.nemotron_h.model as nemotron
    from twinstar.models.ckpt import twinstar_from_spec
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--duet',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    result=dict(cell='actual-reference-moe-reduction-probe',status='running',started_at=datetime.datetime.now().astimezone().isoformat())
    start=time.monotonic()
    audit=json.loads(Path('/task/reference-closure-audit.json').read_text())
    for name,digest in audit['files'].items():
        if hashlib.sha256((Path('/reference')/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'reference source changed: {name}')
    torch.set_grad_enabled(False);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20260929)
    model=twinstar_from_spec(args.model,'all','all',args.duet)
    original=nemotron.sorted_dispatch
    class Collected(Exception):pass
    def probe(x2,w,idx,num_experts,expert,expert_grouped=None):
        if expert_grouped is None:raise RuntimeError('expected actual BF16 CUDA grouped expert path')
        captured={}
        def capture(xs,offs):
            value=expert_grouped(xs,offs);captured['expert_output']=value;return value
        ordinary=original(x2,w,idx,num_experts,expert,capture)
        n,k=idx.shape;order=torch.argsort(idx.reshape(-1),stable=True);rows=order//k
        weighted=captured['expert_output'].float()*w.reshape(-1)[order,None]
        version=weighted._version
        digest=hashlib.sha256(weighted.cpu().contiguous().numpy().tobytes()).hexdigest()
        def repeats(deterministic):
            torch.use_deterministic_algorithms(deterministic)
            first=None;max_delta=0.;fp32_counts=[];bf16_counts=[]
            for _ in range(32):
                value=torch.zeros_like(ordinary).index_add_(0,rows,weighted)
                if first is None:first=value
                fp32_counts.append(int((value!=first).sum().item()))
                bf16_counts.append(int((value.bfloat16()!=first.bfloat16()).sum().item()))
                max_delta=max(max_delta,float((value-first).abs().max().item()))
            return dict(repeats=32,fp32_changed_elements=fp32_counts,bf16_changed_elements=bf16_counts,
                        max_abs_delta=max_delta,bitwise_all_equal=not any(fp32_counts))
        saved=torch.are_deterministic_algorithms_enabled()
        try:
            default=repeats(False);deterministic=repeats(True)
        finally:torch.use_deterministic_algorithms(saved)
        assert weighted._version==version
        result.update(status='measured',reference_sha=audit['reference_sha'],torch_version=torch.__version__,
            gpu_name=torch.cuda.get_device_name(),input_shape=list(x2.shape),routing_shape=list(idx.shape),
            contribution_shape=list(weighted.shape),fixed_contributions_sha256=digest,
            fixed_rows_sha256=hashlib.sha256(rows.cpu().numpy().tobytes()).hexdigest(),
            default=default,deterministic=deterministic,
            interpretation='confirmed repeated fixed-input reduction drift' if not default['bitwise_all_equal'] and deterministic['bitwise_all_equal'] else 'inconclusive')
        raise Collected()
    nemotron.sorted_dispatch=probe
    window=json.loads(Path('/task/nll-windows.json').read_text())['windows'][0]
    try:model.generate_batch([window['prompt']],1,forced=[window['target'][:1]])
    except Collected:pass
    except Exception as exc:result.update(status='error',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        nemotron.sorted_dispatch=original
        result.update(wall_seconds=time.monotonic()-start,finished_at=datetime.datetime.now().astimezone().isoformat())
        (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
