"""GPU preflight: explicit FP32 output must retain the BF16 GEMM accumulator."""
import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import enable_batch_invariant_mode


def main():
    torch.manual_seed(339)
    a=torch.randn(129,193,dtype=torch.bfloat16,device="cuda")
    b=torch.randn(193,65,dtype=torch.bfloat16,device="cuda")
    reference=(a.cpu().float()@b.cpu().float()).cuda()
    enable_batch_invariant_mode()
    actual=torch.mm(a,b,out_dtype=torch.float32)
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual,reference,rtol=1e-5,atol=3e-5)
    assert (actual-actual.bfloat16().float()).abs().max().item()>.001
    assert torch.equal(actual,torch.mm(a,b,out_dtype=torch.float32))
    print("FP32_MM_OUTPUT_PREFLIGHT_OK",flush=True)


if __name__=="__main__":
    main()
