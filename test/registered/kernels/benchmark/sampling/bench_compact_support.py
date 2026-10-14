"""Cold-L2 kernel screen; end-to-end round time remains the Order297 metric."""

import torch
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.sampling.compact_support import compact_top_k_top_p_probs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def native_dense_probs(logits, temperatures, top_k, top_p):
    from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

    rows = logits.shape[0]
    probs = torch.softmax(logits / temperatures, dim=-1)
    probs = top_k_renorm_probs(
        probs, torch.full((rows,), top_k, dtype=torch.int32, device=logits.device)
    )
    return top_p_renorm_probs(
        probs, torch.full((rows,), top_p, dtype=torch.float32, device=logits.device)
    )


def compact_probs(logits, temperatures, top_k, top_p):
    probs = torch.softmax(logits / temperatures, dim=-1)
    return compact_top_k_top_p_probs(probs, top_k, top_p)


@marker.parametrize("vocab", [32768, 248320], [248320])
@marker.benchmark("implementation", ["native", "compact"], unit="us")
def benchmark(vocab: int, implementation: str):
    logits = torch.randn((7, vocab), dtype=torch.float32, device="cuda")
    temperatures = torch.full((7, 1), 0.6, dtype=torch.float32, device="cuda")
    func = native_dense_probs if implementation == "native" else compact_probs
    return marker.do_bench(
        func,
        input_args=(logits, temperatures, 20, 0.95),
        graph_clone_args=(0, 1),
        memory_args=(logits, temperatures),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
