"""R8: exact state/output gate in interpreter or CUDA, plus offline resources.

All artifacts go to --out. --reference is the frozen pre-R8 gdn_factored.py.
GPU mode must run before any timing; changing reduction shape is not assumed
to preserve bytes merely because it is algebraically equivalent.
"""
import argparse
import ast
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def load(a):
    def extract(path, name):
        source = path.read_text(); tree = ast.parse(source)
        node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
        return '\n'.join(source.splitlines()[node.decorator_list[0].lineno-1:node.end_lineno])
    name = '_factored_packed_step_kernel'
    code = (a.source/'gdn_step_vtile.py').read_text()
    code += '\n'+extract(a.source/'gdn_factored.py', name)
    code += '\n'+extract(a.reference, name).replace('def '+name+'(', 'def _reference_step_kernel(', 1)
    path = a.out/'actual_step_kernels.py'; path.write_text(code+'\n')
    spec = importlib.util.spec_from_file_location('r8_actual', path)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def exact_gate(a, m):
    import torch
    cpu = a.mode == 'cpu'
    assert (os.environ.get('TRITON_INTERPRET') == '1') == cpu
    device = 'cpu' if cpu else 'cuda'
    if cpu: assert not torch.cuda.is_available()
    torch.manual_seed(1625); rows = []
    # Every active count (16..31), rank exhaustion, near-span reorthogonalization,
    # zero residual, padding and noncontiguous indices. Snapshot executes before
    # any tiled update; interpreter visits tile zero before later tiles.
    for batch in ((1, 8) if cpu else (1, 8, 64)):
        for dtype in (torch.float32, torch.bfloat16):
            for kind in ('general', 'near_span', 'zero'):
                slots, heads, qheads = batch+2, (2 if cpu else 24), (1 if cpu else 8)
                fa = torch.randn(slots, heads, 128, device=device)
                z = torch.randn(slots, heads, 128, 32, device=device)
                fu = torch.linalg.qr(z).Q.transpose(-2,-1).contiguous()
                fw = torch.randn_like(fu)
                count = (torch.arange(slots*heads,device=device).reshape(slots,heads)%16+16).to(torch.int32)
                if kind == 'near_span': count = (count-16+2)%16+16
                if kind == 'zero': count.zero_()
                stale = torch.zeros(slots, dtype=torch.int32,device=device)
                index = torch.arange(batch, dtype=torch.int64, device=device)
                if batch > 1: index[-1] = -1
                storage = torch.full((batch*2,), -9, dtype=torch.int64, device=device)
                storage[::2] = index; index = storage[::2]
                mixed = torch.randn(batch, (2*qheads+heads)*128, device=device, dtype=dtype)
                if kind == 'near_span':
                    for qhead in range(qheads):
                        key_start=(qheads+qhead)*128
                        mixed[:,key_start:key_start+128] = fu[:batch,qhead*(heads//qheads),0].to(dtype)
                if kind == 'zero': mixed.zero_(); fa.zero_(); fw.zero_()
                gates = [torch.randn(batch,heads,device=device,dtype=dtype) for _ in (0,1)]
                logs = torch.randn(heads,device=device); bias = torch.randn_like(logs)
                vbar = torch.randn(heads,128,device=device)
                states = [[x.clone() for x in (fa,fu,fw,count,stale)] for _ in range(3)]
                outputs = [torch.full((batch,1,heads,128),float('nan'),device=device,dtype=dtype) for _ in range(3)]
                bank=m.StepWorkspaceBank()
                workspace=bank.get(35,batch,heads,device,101,capturing=False) if batch<=8 else None
                for i in range(3):
                    sa,su,sw,sc,ss=states[i]
                    kw=dict(stride_mixed_tok=mixed.stride(0),stride_a_tok=gates[0].stride(0),
                        stride_b_tok=gates[1].stride(0),stride_idx=index.stride(0),
                        H=qheads,HV=heads,K=128,V=128,RMAX=32,SOFTPLUS_THRESHOLD=20.,
                        dst_a=sa,dst_u=su,dst_w=sw,dst_count=sc,
                        OUT_OF_PLACE=False,OUT_ROW_STRIDE=outputs[i].stride(0),num_warps=1)
                    args=(mixed,*gates,logs,bias,vbar,sa,su,sw,sc,ss,index,outputs[i],128**-.5,1e-4)
                    if i == 0: m._reference_step_kernel[(batch*heads,)](*args,**kw)
                    elif i == 1: m._factored_packed_step_kernel[(batch*heads,)](*args,**kw)
                    elif workspace is not None:
                        m.snapshot_step_metadata(sa,sc,index,workspace)
                        m._factored_packed_step_kernel[(batch*heads,1,8)](*args,**kw,
                            V_TILE=16,snapshot_a=workspace[0],snapshot_count=workspace[1])
                    else:
                        m._factored_packed_step_kernel[(batch*heads,)](*args,**kw)
                if not cpu: torch.cuda.synchronize()
                fields=('a','U','W','count','stale','output')
                diffs=[]
                for label,state,out in zip(('default','vtile'),states[1:],outputs[1:]):
                    for field,x,y in zip(fields,[*states[0],outputs[0]],[*state,out]):
                        exact=torch.equal(x.view(torch.uint8),y.view(torch.uint8))
                        if not exact:
                            diffs.append(dict(arm=label,field=field,
                                different_bytes=int((x.view(torch.uint8)!=y.view(torch.uint8)).sum()),
                                max_abs=float((x.float()-y.float()).abs().max())))
                row=dict(batch=batch,dtype=str(dtype),kind=kind,passed=not diffs,diffs=diffs)
                rows.append(row); print(json.dumps(row),flush=True)
    bank=m.StepWorkspaceBank(); live=[]
    for layer in (0,35):
        for batch in (1,8):
            for stream in (101,202):
                buffers=bank.get(layer,batch,24,device,stream,capturing=False);live.extend(buffers)
                assert bank.get(layer,batch,24,device,stream,capturing=True) is buffers
    assert len({x.untyped_storage().data_ptr() for x in live})==len(live)
    try: bank.get(0,2,24,device,101,capturing=True)
    except RuntimeError: pass
    else: raise AssertionError('new scratch allocated inside capture')
    return dict(passed=all(r['passed'] for r in rows),rows=rows,device=device,
                disjoint_buffers=len(live),captured_shape_allocation_rejected=True)


def compile_gate(a,m):
    import math
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    target=GPUTarget('cuda',103,32); rows=[]
    assembler=Path(triton.__file__).parent/'backends/nvidia/bin/ptxas-blackwell'
    for label,kernel,tile in (('original',m._reference_step_kernel,0),
                              ('default',m._factored_packed_step_kernel,0),
                              ('vtile',m._factored_packed_step_kernel,16)):
        constants=dict(stride_mixed_tok=5120,stride_a_tok=24,stride_b_tok=24,stride_idx=1,
            H=8,HV=24,K=128,V=128,RMAX=32,SOFTPLUS_THRESHOLD=20.,OUT_OF_PLACE=False,
            OUT_ROW_STRIDE=3072,WRITE_OUTPUT=True,LAYER_MIXED=0,LAYER_GATE_A=0,
            LAYER_GATE_B=0,LAYER_LOG=0,LAYER_BIAS=0,LAYER_VBAR=0,LAYER_A=0,
            LAYER_U=0,LAYER_W=0,LAYER_COUNT=0)
        signature={n:'*fp32' for n in kernel.arg_names}
        for n in ('mixed_qkv','a_gate','b_gate','o'):signature[n]='*bf16'
        for n in ('cnt_ptr','stale_ptr','dst_count'):signature[n]='*i32'
        signature['ssm_state_indices']='*i64'
        signature['scale']='fp32';signature['gs_eps']='fp32'
        if label != 'original':
            constants['V_TILE']=tile
            if tile:
                signature['snapshot_a']='*fp32';signature['snapshot_count']='*i32'
            else: constants.update(snapshot_a=None,snapshot_count=None)
        for n in constants:signature[n]='constexpr'
        c=triton.compile(ASTSource(kernel,signature=signature,constexprs=constants),
                         target=target,options={'num_warps':1})
        ptx=a.out/(label+'.ptx');ptx.write_text(c.asm['ptx'])
        arch=re.search(r'\.target\s+(\w+)',c.asm['ptx']).group(1)
        r=subprocess.run([str(assembler),'-v','--gpu-name='+arch,str(ptx),'-o',str(a.out/(label+'.cubin'))],
                         capture_output=True,text=True,check=True)
        (a.out/(label+'-ptxas.txt')).write_text(r.stdout+r.stderr)
        regs=int(re.search(r'Used (\d+) registers',r.stderr).group(1)); shared=c.metadata.shared
        row=dict(name=label,registers=regs,shared=shared,
            blocks_per_SM_estimate=min(65536//(math.ceil(regs*32/256)*256),
                233472//(math.ceil(shared/256)*256+1024),32),
            bar_sync_count=len(re.findall(r'\bbar\.sync\b',c.asm['ptx'])))
        rows.append(row);print(json.dumps(row),flush=True)
    return dict(passed=True,rows=rows,gpu_math_executed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('cpu','gpu','compile'),required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    m=load(a);result=compile_gate(a,m) if a.mode=='compile' else exact_gate(a,m)
    result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (a.source/'gdn_factored.py',a.source/'gdn_step_vtile.py',a.reference,Path(__file__))}
    (a.out/(a.mode+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
