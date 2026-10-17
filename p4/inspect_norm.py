"""Archive the exact service Triton lowering before matching the CUDA epilogue."""
import json
from pathlib import Path
import sys
import torch
from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd_1pass_kernel

def inspect(dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    x = torch.randn(24, 128, device='cuda', dtype=torch.bfloat16)
    y, w, z = torch.empty_like(x), torch.ones(128, device='cuda', dtype=x.dtype), torch.randn_like(x)
    rstd = torch.empty(24, device='cuda')
    for rows in (1, 4):
        k = _layer_norm_fwd_1pass_kernel[(24 // rows, 1)](
            x, y, w, None, z, None, rstd, 128, 128, 128, 0, 0, 24, 128, 1e-6,
            BLOCK_N=128, ROWS_PER_BLOCK=rows, HAS_BIAS=False, HAS_Z=True,
            Z_IS_3D=False, Z_HEADS=0, NORM_BEFORE_GATE=True, IS_RMS_NORM=True,
            ACTIVATION='sigmoid', USE_GDC=True, launch_pdl=True, num_warps=1)
        for fmt in ('ptx','ttgir','llir'):
            (dst / f'norm-r{rows}.{fmt}').write_text(k.asm[fmt])
    (dst / 'runtime.json').write_text(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
       gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability()), indent=2))
    print('NORM_ASM', dst, flush=True)

if __name__ == '__main__':
    inspect(sys.argv[1])
