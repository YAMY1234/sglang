"""CPU-only Triton interpretation of #443 state IO, not full-model GPU logits.

TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' python this_file --baseline-py OLD.py
The readout logits use a fixed small CPU projection of real GDN kernel outputs.
No GPU driver, model checkpoint, Slurm allocation or serving engine is used.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

if os.environ.get('TRITON_INTERPRET') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('this guard requires CPU interpretation and hidden CUDA devices')

import torch

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT/'python/sglang/srt/layers/attention/linear/kernels'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def same(a, b, label):
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
        raise AssertionError(label)


def main(baseline_path):
    package = ModuleType('copy_guard_kernels'); package.__path__ = [str(KERNELS)]
    sys.modules[package.__name__] = package
    candidate = importlib.import_module('copy_guard_kernels.gdn_factored')
    baseline = load('copy_guard_kernels.frozen_baseline', baseline_path)
    old_decode = baseline.factored_packed_decode
    def old_signature(*args, state_dest=None, **kwargs):
        if state_dest is not None:
            raise AssertionError('baseline must use the original in-place path')
        return old_decode(*args, **kwargs)
    baseline.factored_packed_decode = old_signature
    io = importlib.import_module('copy_guard_kernels.gdn_verify_io')
    tx_module = load('copy_guard_transaction', ROOT/'python/sglang/srt/mem_cache/gdn_factored_spec.py')
    Transaction = tx_module.FactoredGDNVerifyState
    # Execute the actual serving method without importing unrelated GPU backends.
    backend_path = ROOT/'python/sglang/srt/layers/attention/linear/gdn_backend.py'
    tree = ast.parse(backend_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GDNAttnBackend')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_verify_factored')
    scope = {}; exec(compile(ast.Module(body=[method], type_ignores=[]), str(backend_path), 'exec'), scope)
    verify = scope[method.name]
    canonical = 'sglang.srt.layers.attention.linear.kernels.gdn_factored'
    # The method's one import resolves directly to each real interpreted kernel.
    torch.manual_seed(443)
    layers, capacity, batch, heads, key, value = 2, 3, 2, 2, 16, 16
    base_pool = SimpleNamespace(
        cfg=SimpleNamespace(r=8,m=8,rfull=16,kernel_kwargs=lambda: dict(kernel='split',post_order=True)),
        a=torch.randn(layers,8,heads,key)*.01,
        U=torch.randn(layers,8,heads,16,key,dtype=torch.float16)*.01,
        W=torch.randn(layers,8,heads,16,value,dtype=torch.float16)*.01,
        count=torch.full((layers,8,heads),8,dtype=torch.int32),
        stale=torch.zeros(8,dtype=torch.int32),dense_of=torch.arange(8,dtype=torch.int32),
        dense_required=torch.ones(8,dtype=torch.int32),prefix_valid=torch.ones(8,dtype=torch.int32),
        vbar=torch.randn(layers,heads,value)*.01,layer_index=lambda layer:layer)
    base_pool.count[:,2].fill_(14);base_pool.count[:,5].fill_(15)
    pools = [copy.deepcopy(base_pool) for _ in range(3)]
    for i,pool in enumerate(pools):
        pool.spec_state = Transaction(pool,capacity,4,direct_checkpoints=i==2)
    descriptors = [SimpleNamespace(layer_id=i,num_q_heads=1,num_v_heads=heads,
        head_k_dim=key,head_v_dim=value,A_log=torch.randn(heads),dt_bias=torch.randn(heads)) for i in range(layers)]
    mixed=torch.randn(layers,capacity,4,2*key+heads*value,dtype=torch.bfloat16)
    gates_a=torch.randn(layers,capacity,4,heads,dtype=torch.bfloat16);gates_b=torch.randn_like(gates_a)
    slots=torch.tensor([2,5]);head=torch.randn(heads*value,31)
    cases=[]
    for iteration in range(21):
        tickets=[p.spec_state.snapshot_commit(slots) for p in pools]
        # Independently exercise the real indexed snapshot kernel on CPU.
        tx=pools[2].spec_state
        expected={n:t.clone() for n,t in tx.working.items()}
        io.snapshot_factors(pools[2],tx.working,slots)
        for n in tx.names:same(expected[n],tx.working[n],f'snapshot {n}')
        outputs=[]
        for i,pool in enumerate(pools):
            sys.modules[canonical]=baseline if i==0 else candidate
            backend=SimpleNamespace(factored=pool,topk=1)
            outputs.append([verify(backend,d,mixed[li].flatten(0,1),gates_a[li].flatten(0,1),
                                   gates_b[li].flatten(0,1)).view(capacity,4,heads,value)
                            for li,d in enumerate(descriptors)])
        for i in (1,2):
            for n in tx.names:
                same(getattr(pools[0],n),getattr(pools[i],n),'persistent pool before commit')
                same(pools[0].spec_state.checkpoints[n][:,:batch],pools[i].spec_state.checkpoints[n][:,:batch],
                     f'all candidate states {n} round {iteration}')
            for li in range(layers):
                same(outputs[0][li],outputs[i][li],f'kernel output round {iteration}')
                same(outputs[0][li].float().flatten(2)@head,outputs[i][li].float().flatten(2)@head,
                     f'fixed CPU readout logits round {iteration}')
        selected=torch.tensor([iteration%4,(iteration+2)%4] if iteration<4 else [0,0])
        for p,ticket in zip(pools,tickets):
            p.spec_state.commit(ticket,selected,track_slots=torch.tensor([-1,7]),
                                track_steps=torch.tensor([-1,int(selected[1])]))
        for i in (1,2):
            for n in (*tx.names,'stale','dense_of','dense_required','prefix_valid'):
                same(getattr(pools[0],n),getattr(pools[i],n),f'commit {n} round {iteration}')
        cases.append(dict(round=iteration,last_consumed_indices=selected.tolist(),bitwise=True))
    for p in pools:
        before={n:getattr(p,n).clone() for n in tx.names}
        ticket=p.spec_state.snapshot_commit(slots)
        p.spec_state.rollback(ticket)
        for n in tx.names:same(before[n],getattr(p,n),'rollback')
    # Use the production stride-aware scatter kernel, not CPU advanced indexing.
    ptr=ModuleType('sglang.kernels.ops.memory.ptr_table');ptr.make_ptr_table=None
    sys.modules[ptr.__name__]=ptr
    scatter=load('copy_guard_scatter',ROOT/'python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py')
    for n in tx.names:
        src=pools[2].spec_state.checkpoints[n]
        dst=torch.zeros_like(getattr(pools[2],n));indices=torch.tensor([2,5,-1]);steps=torch.tensor([0,3,-1])
        # The public wrapper correctly rejects CPU tensors. Interpret its
        # unchanged device kernel with exactly the wrapper's real strides.
        scatter._require_entry_contiguous_dst(src,3,'CPU source guard')
        entry=dst[0,0].numel()
        scatter._fused_mamba_state_scatter_with_mask_kernel[(indices.numel(),layers,(entry+1023)//1024)](
            src,dst,indices.int(),steps.int(),entry,*src.stride()[:3],*dst.stride()[:2],
            src.shape[1],src.shape[2],dst.shape[1],BLOCK_SIZE=1024)
        same(dst[:,2],src[:,0,0],f'strided scatter {n} first')
        same(dst[:,5],src[:,1,3],f'strided scatter {n} last')
        if torch.count_nonzero(dst[:,0]):raise AssertionError('padded scatter wrote slot zero')
    print(json.dumps(dict(passed=True,device='CPU',triton_interpret=True,
        factor_dtype='float16',layers=layers,head_dim=key,padded_batch=capacity,active_batch=batch,
        cases=cases,consecutive_zero_drafts=17,w8_inside_window=True,
        frozen_baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        flagoff_and_candidate_bitwise=True,indexed_snapshot_and_strided_commit=True,
        logits_scope='fixed small CPU readout of actual interpreted GDN outputs; not 48-layer model logits',
        gpu_model_logits_validated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline-py',type=Path,required=True)
    main(p.parse_args().baseline_py)
