"""Build the C++ metadata tail into PTX for embedding in the native CuTe grid.

CPU-only NVRTC translation: no CUDA context or module is created here.
"""
import argparse,ctypes as C,re
from pathlib import Path

HERE=Path(__file__).resolve().parent

def source():
    s=(HERE/'route_metadata_reference.cu').read_text()
    s=s.replace('void run(const Params& p) {','void run(const Params& p, uint32_t* arena) {')
    for old,new in [
        ('__shared__ uint32_t candidates[8][32];','auto candidates = reinterpret_cast<uint32_t (*)[32]>(arena);'),
        ('__shared__ uint32_t masks[512];','uint32_t* masks = arena + 256;'),
        ('__shared__ uint32_t bases[512];','uint32_t* bases = arena + 768;'),
        ('__shared__ uint32_t warp_ctas[8];','uint32_t* warp_ctas = arena + 1280;'),
        ('__shared__ uint32_t route_ids[160];','uint32_t* route_ids = arena + 1288;\n  uint32_t* ranks = arena + 1448;'),
        ('asm volatile("griddepcontrol.wait;" ::: "memory");',''),
        ('atomicOr(masks + expert, 1u << token);',''),
        ('const uint32_t count0 = __popc(masks[e0]), count1 = __popc(masks[e1]);','const uint32_t count0 = masks[e0], count1 = masks[e1];'),
        ('const uint32_t rank = __popc(masks[e] & ((1u << token) - 1u));','const uint32_t rank = ranks[expanded];'),
        ('p.permuted_to_token[total_ctas * p.tile_n] = -1;',''),
        ('asm volatile("griddepcontrol.launch_dependents;" ::: "memory");',''),
    ]:
        if old not in s: raise ValueError('Missing source anchor: '+old)
        s=s.replace(old,new)
    # Match ordinary native routingPermutation's per-expanded-thread atomic rank.
    anchor='  // Ascending expert order, arbitrary resolved native tile (including192).'
    s=s.replace(anchor,'  if (tid < p.num_tokens * 10) ranks[tid] = atomicAdd(masks + route_ids[tid], 1u);\n  __syncthreads();\n\n'+anchor)
    # The one-block native router uses shared histogram; its global histogram
    # scratch and guard/capacity tails are left untouched, as in the native run.
    for line in ('  p.histogram[e0] = static_cast<int32_t>(count0);\n',
                 '  p.histogram[e1] = static_cast<int32_t>(count1);\n',
                 '  p.histogram[512 + e0] = 0;\n',
                 '  p.histogram[512 + e1] = 0;\n'):
        assert line in s;s=s.replace(line,'')
    begin=s.index('extern "C" __global__ __launch_bounds__(256, 1)')
    s=s[:begin]+'''extern "C" __global__ __launch_bounds__(256, 1)
void q38_metadata_tail(const q38_native_metadata::Params* p) {
  extern __shared__ uint32_t q38_arena[];
  q38_native_metadata::run(*p, q38_arena);
}
'''
    return '\n'.join(line.rstrip() for line in s.splitlines()) + '\n'

def build(nvrtc,includes):
    text=source();(HERE/'metadata_tail.cu').write_text(text)
    lib=C.CDLL(nvrtc);prog=C.c_void_p()
    rc=lib.nvrtcCreateProgram(C.byref(prog),text.encode(),b'metadata_tail.cu',0,None,None)
    if rc:raise RuntimeError(('create',rc))
    opts=['--gpu-architecture=compute_103a','--std=c++17','--use_fast_math']+['--include-path='+x for x in includes]
    args=(C.c_char_p*len(opts))(*[x.encode() for x in opts]);rc=lib.nvrtcCompileProgram(prog,len(opts),args)
    n=C.c_size_t();lib.nvrtcGetProgramLogSize(prog,C.byref(n));buf=C.create_string_buffer(n.value);lib.nvrtcGetProgramLog(prog,buf)
    (HERE/'tail-build.log').write_bytes(buf.value)
    if rc:raise RuntimeError((rc,buf.value.decode()))
    lib.nvrtcGetPTXSize(prog,C.byref(n));buf=C.create_string_buffer(n.value);lib.nvrtcGetPTX(prog,buf)
    ptx=buf.value.decode();(HERE/'metadata_tail.ptx').write_text(ptx);lib.nvrtcDestroyProgram(C.byref(prog))
    entry=ptx.index('.visible .entry q38_metadata_tail');start=ptx.index('{',entry);end=ptx.rindex('}')
    body=ptx[start+1:end]
    if '.local' in body or re.search(r'\bcall\b',body):
        raise RuntimeError('Tail must inline without static shared, local spills or calls')
    for k in (0,):
        pattern=r'ld\.param\.u64\s+(%rd\d+),\s*\[q38_metadata_tail_param_'+str(k)+r'\];'
        body,count=re.subn(pattern,lambda m:f'mov.u64 {m.group(1)}, Q38_OPERAND_{k};',body)
        if count!=1:raise RuntimeError(('param',k,count))
    body,count=re.subn(r'mov\.(u32|u64)\s+(%rd?\d+),\s*q38_arena;', lambda m:f"{'cvt.u64.u32' if m.group(1)=='u64' else 'mov.u32'} {m.group(2)}, Q38_OPERAND_1;",body)
    if not count:raise RuntimeError(('arena substitution',count))
    body='.reg .b32 %q38_smem;\n'+body.replace('q38_arena','%q38_smem')
    body=re.sub(r'(\.reg \.b64[^;]+;)',r'\1\nmov.u32 %q38_smem,Q38_OPERAND_1;',body,count=1)
    body=body.replace('ret;','bra Q38_TAIL_END;')+'\nQ38_TAIL_END:\n'
    body=body.replace('$','$$')
    for k in (0,1):body=body.replace('Q38_OPERAND_'+str(k),'$'+str(k))
    (HERE/'tail_inline.ptx').write_text('{\n'+body+'\n}\n')
    print('CPU_PTX_READY',len(ptx),'bytes; arena1608 uint32 (6432B), M4/M8/M12')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nvrtc',required=True);p.add_argument('--include',action='append',default=[]);a=p.parse_args();build(a.nvrtc,a.include)
