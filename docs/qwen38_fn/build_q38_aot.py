"""Build the three 2a92c89d MoE bundles without a GPU or private input files."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True, help='SGLang checkout root')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = a.source.absolute()
out = a.output.absolute()
csrc = source / 'python/sglang/kernels/jit/csrc/moe'
ops = source / 'python/sglang/kernels/ops/moe'
shape = [2560, 192, 512, 10, 140, 2, 216064]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root, manifest):
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


lib = C.CDLL('/usr/local/cuda/lib64/libnvrtc.so.13')
common = ['--gpu-architecture=sm_103a', '-std=c++17', '-default-device']
includes = ['-I/usr/local/cuda/include', '-I/usr/local/cuda/include/cccl']


def compile_cuda(path, target, flags):
    program = C.c_void_p()
    def check(rc):
        if rc != 0:
            raise RuntimeError(f'NVRTC error {rc}: {path}')
    check(lib.nvrtcCreateProgram(C.byref(program), path.read_bytes(),
                                path.name.encode(), 0, None, None))
    try:
        options = [*common, *flags, *includes]
        encoded = (C.c_char_p * len(options))(*(o.encode() for o in options))
        rc = lib.nvrtcCompileProgram(program, len(options), encoded)
        size = C.c_size_t()
        check(lib.nvrtcGetProgramLogSize(program, C.byref(size)))
        log = C.create_string_buffer(size.value)
        check(lib.nvrtcGetProgramLog(program, log))
        target.with_suffix('.log').write_bytes(log.value)
        if rc != 0:
            raise RuntimeError(log.value.decode(errors='replace'))
        check(lib.nvrtcGetCUBINSize(program, C.byref(size)))
        binary = C.create_string_buffer(size.value)
        check(lib.nvrtcGetCUBIN(program, binary))
        target.write_bytes(binary.raw)
    finally:
        lib.nvrtcDestroyProgram(C.byref(program))


for name in ('mkm', 'router', 'sf-pack'):
    (out / name).mkdir(parents=True, exist_ok=True)

# Ordered top-10 finalize: compile the exact repository function on the CPU.
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

finalize_source = ops / 'q38_mk_finalize.py'
spec = importlib.util.spec_from_file_location('q38_repro_finalize', finalize_source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
compiled = triton.compile(
    ASTSource(module._q38_mk_mtp_finalize,
              signature={'ExpertRows': '*bf16', 'Weights': '*bf16', 'Out': '*bf16'},
              constexprs={'H': 2560, 'TOPK': 10, 'BLOCK': 256}),
    target=GPUTarget('cuda', 103, 32),
    options={'num_warps': 4, 'enable_fp_fusion': False},
)
root = out / 'mkm'
(root / 'finalize.cubin').write_bytes(compiled.asm['cubin'])
entry = re.search(r'\.entry\s+\w+\s*\((.*?)\)', compiled.asm['ptx'], re.S)
assert entry, 'Could not identify the finalize PTX entry'
params = len(re.findall(r'\.param\s+', entry.group(1)))
assert 3 <= params <= 5
final = dict(symbol=compiled.metadata.name, shared=compiled.metadata.shared,
             params=params, num_warps=4, enable_fp_fusion=False)

expert_flags = ['--generate-line-info', '--ptxas-options=-v',
                '-DQ38_MK_I=192', '-DQ38_MK_TRACE=0',
                '-DQ38_MK_SMEM_BYTES=216064', '-DQ35_MKM_SMEM_BYTES=216064']
base_source = csrc / 'q38_mk_moe_q35_tp4.cu'
compile_cuda(base_source, root / 'q35-port.cubin', expert_flags)
write_manifest(root, dict(
    source_sha256=sha(base_source), finalize_source_sha256=sha(finalize_source),
    shape=shape, sha256={n: sha(root / n) for n in ('q35-port.cubin', 'finalize.cubin')},
    finalize=final,
))

root = out / 'router'
router_source = csrc / 'q38_router_front.cu'
for capacity in (1, 8):
    compile_cuda(router_source, root / f'route-m{capacity}.cubin',
                 [f'-DQ35_ROUTER_MAX_M={capacity}', '-DQ35_ROUTER_KSPLIT=2'])
write_manifest(root, dict(
    source_sha256=sha(router_source), shape=[2560, 512, 2],
    sha256={n: sha(root / n) for n in ('route-m1.cubin', 'route-m8.cubin')},
))

root = out / 'sf-pack'
sf_source = csrc / 'q38_mk_moe_q35_sf_pack_tp4.cu'
compile_cuda(sf_source, root / 'q35-sf-pack.cubin', [*expert_flags, '-DPDL_TRIGGER=0'])
write_manifest(root, dict(
    source_sha256=sha(sf_source), base_manifest_sha256=sha(out / 'mkm/manifest.json'),
    shape=shape, pdl_trigger=0, cubin_sha256=sha(root / 'q35-sf-pack.cubin'),
))
print(json.dumps(dict(status='BUILD_COMPLETE', output=str(out),
                      finalize=final, triton=triton.__version__), indent=2))
