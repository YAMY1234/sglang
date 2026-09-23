"""Independent reference-source parity; CPU or CUDA, real release weights optional.

python test_flashnext_scheme_c.py --reference /path/to/frozen/twinstar \
    --weights /path/to/duet_components.safetensors --device cuda --out receipt.json
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def bitwise(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    if not a.numel():
        return
    assert torch.equal(a.flatten().contiguous().view(torch.uint8), b.flatten().contiguous().view(torch.uint8)), (
        (a.float() - b.float()).abs().max().item() if a.numel() else 0)


def run(reference, device, weights=None):
    root = Path(__file__).resolve().parents[4]
    codec = module('scheme_c_port', root/'python/sglang/srt/mem_cache/flashnext_scheme_c.py')
    fmt_path = reference/'twinstar/duet/latentfmt.py'
    model_path = reference/'twinstar/models/twinstar_model.py'
    fmt = module('scheme_c_oracle_format', fmt_path)
    source = model_path.read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LatentBottleneck')
    ns = dict(torch=torch, nn=torch.nn, os=os, latentfmt=fmt)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(model_path), 'exec'), ns)
    Ref = ns['LatentBottleneck']
    for name in ('LATENT_STORE', 'LATENT_SPARSE_QUANT', 'LATENT_OFF', 'LATENT_SPARSE'):
        os.environ.pop(name, None)
    torch.set_grad_enabled(False)
    torch.manual_seed(373)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tests = []
    # Includes all exact e2m1 edges, sign/underflow, all-zero block and empty rows.
    for n, r in ((0, 4096), (1, 16), (17, 4096), (257, 4096)):
        x = torch.randn(n, r, device=device)
        if n:
            x[0] = 0
        p, sb, g = codec.pack_nvfp4(x)
        for actual, old in zip((p, sb, g), codec._pack_nvfp4_torch(x)):
            bitwise(actual, old)
        bitwise(codec.unpack_nvfp4(p, sb, g), codec._unpack_nvfp4_torch(p, sb, g))
        bitwise(codec.unpack_nvfp4(p, sb, g), fmt.fq_nvfp4(x))
        xf = x.float()
        expected_g = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (6.0 * 448.0)
        expected_sb = ((xf / expected_g).reshape(n, r//16, 16).abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
        bitwise(sb, expected_sb)
        bitwise(g, expected_g)
        tests.append(f'nvfp4-fields-{n}x{r}')
    # The max coordinate fixes g=1/448, sb=448; all edge values remain exact.
    edge = torch.tensor([0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., -0., -.25, -.75, -1.25, -1.75, -2.5, -3.5, 6.], device=device)[None]
    p, sb, g = codec.pack_nvfp4(edge)
    bitwise(codec.unpack_nvfp4(p, sb, g), fmt.fq_nvfp4(edge))
    tests.append('nvfp4-midpoints-negative-zero')
    for scale in (1e-20, 1e-8, 1.0, 1e8, 1e20):
        # Noncontiguous source, dynamic exponents and ties near bin boundaries.
        x=(torch.randn(41,8192,device=device)*scale)[:,::2]
        for actual,old in zip(codec.pack_nvfp4(x),codec._pack_nvfp4_torch(x)):
            bitwise(actual,old)
    tests.append('nvfp4-strides-exponents-byte-parity')
    sets = [torch.tensor([0, 254, 510, 1023, 4096, 10239], device=device),
            torch.tensor([255, 767, 1023, 1535, 1791, 2303], device=device),
            torch.randperm(10240, device=device)[:512],
            torch.arange(512, device=device),
            torch.cat((torch.arange(38,device=device)*256+255, torch.arange(9728,10202,device=device)))]
    for i, idx in enumerate(sets):
        row = idx.flip(0)[None]
        p, lengths, order = codec.pack_gap8(row)
        for actual,old in zip((p,lengths,order),codec._pack_gap8_torch(row)):
            bitwise(actual,old)
        oracle = fmt.pack_gap8(row[0]).to(device)
        bitwise(p[0, :int(lengths[0, 0])], oracle)
        bitwise(codec.unpack_gap8(p, lengths, sparse=idx.numel()), idx.sort().values[None].long())
        bitwise(fmt.unpack_gap8(oracle.cpu(), idx.numel()), idx.cpu().sort().values.long())
        assert int(lengths[0, 0]) == int(fmt.gap8_bytes(row)[0])
        tests.append(f'gap8-reference-{i}')
    try:
        codec.pack_gap8(torch.tensor([[1, 1]], device=device))
    except ValueError:
        tests.append('gap8-duplicate-rejected')
    else:
        raise AssertionError('duplicate accepted')
    assert fmt.LatentFormat('nvfp4','bf16','gap8').bytes_per_token(4096,512,10240,True) == 3848
    nominal = 4096//2 + 4096//16 + 4 + 512*2 + 512 + 4
    assert nominal == codec.FlashNextSchemeCCodec.PAYLOAD_BYTES
    tests.append('nominal-byte-account-3848')
    width, rank, sparse = (10240,4096,512) if weights else (128,32,16)
    Port = type('SizedCodec', (codec.FlashNextSchemeCCodec,), dict(WIDTH=width,RANK=rank,SPIKES=sparse))
    port = Port(device=device)
    ref = Ref(SimpleNamespace(hidden_size=width), SimpleNamespace(latent_rank=rank,latent_sparse=sparse,
        latent_init='',latent_enc_hidden=0,latent_z_format='nvfp4',latent_value_format='bf16',latent_index_format='gap8')).to(device)
    if weights:
        from safetensors import safe_open
        with safe_open(str(weights), framework='pt', device='cpu') as sf:
            matrices = {k:sf.get_tensor('P.latent.core.'+k) for k in ('E','D','mean')}
    else:
        matrices = {'E':torch.randn(rank,width)/width**.5, 'D':torch.randn(width,rank)/rank**.5, 'mean':torch.randn(width)*.01}
    for k, t in matrices.items():
        port.load(k,t)
        getattr(ref,k).copy_(t.to(device=device,dtype=torch.bfloat16).float())
        bitwise(getattr(port,k),getattr(ref,k))
    for n, start in ((0,0),(1,0),(1,8192),(17,0),(17,32768),(257,0)):
        h = torch.randn(n,width,device=device,dtype=torch.bfloat16)
        base = torch.randn_like(h)
        pos = torch.arange(start,start+n,device=device)
        batch = port.encode(h,pos,base)
        output = port.decode(batch,base)
        ref.keep_sink = start == 0
        expected = ref(h[None],base[None])[0]
        bitwise(output,expected)
        reused_batch,reused_output=port.encode_and_decode(h,pos,base)
        bitwise(reused_output,expected)
        for field in vars(batch):bitwise(getattr(reused_batch,field),getattr(batch,field))
        assert batch.sink_rows.numel() == int(start == 0 and n > 0)
        if n and start == 0:
            bitwise(output[0],h[0])
        assert batch.spike_values.dtype == torch.bfloat16
        tests.append(f'ED-embedding-sink-{n}-offset-{start}')
        tests.append(f'ED-reused-reconstruction-{n}-offset-{start}')
    return dict(complete=True, tests=tests, count=len(tests), device=str(device),
                torch_version=torch.__version__, real_release_weights=bool(weights),
                nominal_payload_bytes=nominal, gap8_max_bytes=codec.gap8_capacity(10240,512),
                oracle_format_sha256=hashlib.sha256(fmt_path.read_bytes()).hexdigest(),
                oracle_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                port_sha256=hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest(),
                weights_sha256=hashlib.file_digest(weights.open('rb'),'sha256').hexdigest() if weights else None)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--weights',type=Path)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    start=time.monotonic()
    result=run(a.reference,torch.device(a.device),a.weights)
    result['seconds']=time.monotonic()-start
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
