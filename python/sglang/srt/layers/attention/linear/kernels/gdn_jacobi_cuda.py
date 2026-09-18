"""K3-A one-warp eigensolver probe. Loaded only by the experimental harness."""
import os
from pathlib import Path
from functools import lru_cache
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_extension():
    root=Path(__file__).resolve().parent
    return load(name='twinstar_k3a_jacobi', sources=[str(root/'gdn_jacobi_cuda.cu')],
                extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '--ptxas-options=-v', '-DK3_RANK_TOL_SQ='+os.environ.get('K3_RANK_TOL_SQ','1.e-8f'), '-DK3_FAST_PROJECT='+os.environ.get('K3_FAST_PROJECT','0'), '-DK3_FAST_GZ='+os.environ.get('K3_FAST_GZ','0')], extra_ldflags=['-lcusolver'], verbose=True)
