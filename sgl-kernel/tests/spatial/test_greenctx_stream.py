import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import create_greenctx_stream_by_value, get_sm_available


def test_green_ctx():
    A = torch.randn(5120, 5120).cuda()
    B = torch.randn(5120, 5120).cuda()
    C = torch.matmul(A, B)
    sm_counts = get_sm_available(0)
    stream_group = create_greenctx_stream_by_value(sm_counts // 2, sm_counts // 2, 0)
    with torch.cuda.stream(stream_group[0]):
        for _ in range(100):
            result_0 = torch.matmul(A, B)
    with torch.cuda.stream(stream_group[1]):
        for _ in range(100):
            result_1 = torch.matmul(A, B)
    torch.cuda.synchronize()
    # Note: cuBLAS may select different algorithms based on the number of SMs
    # in the context/green context, which can produce slightly different
    # results. Due to the large matmul, we need to increase the tolerance here.
    assert torch.allclose(result_0, C, rtol=1e-5, atol=1e-3)
    assert torch.allclose(result_1, C, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
