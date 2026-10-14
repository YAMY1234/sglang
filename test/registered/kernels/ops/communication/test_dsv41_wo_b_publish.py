"""DSV4.1 wo_b publishing into the fused all-reduce from its GEMM epilogue
(SGLANG_OPT_DSV41_WO_B_PUBLISH), on 4 SM100 GPUs with NVLink multicast.

Per split-K tactic and row count: the published GEMM's output equals FlashInfer's
low-latency GEMM with the same tactic, and the consumer's outputs equal the separate
publish kernel's, bit for bit, over boundary sequences that cycle the three mailbox
generations, mix both paths, delay one rank in turn and replay from a CUDA graph.
The consumer also matches a torch all-reduce + mHC post + collapse + RMSNorm.

Run with ``python test_dsv41_wo_b_publish.py --num-gpu 4``.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN, HC, K = 5120, 4, 2048


@pytest.fixture(scope="module")
def rank():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield rank
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _linear(device, gen):
    from flashinfer import mxfp8_quantize

    from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

    w = torch.randn(HIDDEN, K, device=device, generator=gen) / math.sqrt(K)
    w_q, w_sf = mxfp8_quantize(w.bfloat16(), is_sf_swizzled_layout=True)
    return SimpleNamespace(
        weight=w_q.contiguous(),
        weight_scale_inv_swizzled=w_sf.reshape(-1).contiguous(),
        bias=None,
        mxfp8_low_latency=True,
        quant_method=SimpleNamespace(
            mxfp8_dense_backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
            use_mxfp8=True,
        ),
    )


def _activation(rows, device, gen):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    x = torch.randn(rows, K, device=device, generator=gen).bfloat16()
    if rows > 1:
        # A zero row: zero outputs, whose negative zeros the publish clears.
        x[-1] = 0
    q, sf = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    return q, sf.reshape(-1)


def _flashinfer_gemm(linear, qa, tactic):
    """FlashInfer's own low-latency GEMM with ``tactic`` (output, valid tactics)."""
    from flashinfer.gemm import gemm_base

    a, a_scale = qa
    out = torch.empty(a.shape[0], HIDDEN, dtype=torch.bfloat16, device=a.device)
    major, minor = torch.cuda.get_device_capability(a.device)
    runner = gemm_base._cutedsl_low_latency_blockscaled_gemm_runner(
        major * 10 + minor, True
    )
    workspace = gemm_base._get_cache_buf(
        "mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, a.device
    )
    inputs = [
        a,
        linear.weight.t(),
        a_scale,
        linear.weight_scale_inv_swizzled,
        torch.bfloat16,
        out,
        workspace,
    ]
    runner.forward(inputs, tactic=tactic)
    return out, runner.get_valid_tactics(inputs, None)


def _mixes(rows, device, rank):
    gen = torch.Generator(device=device).manual_seed(7 + rank)
    return (
        torch.randn(rows, HC, HIDDEN, device=device, generator=gen).bfloat16(),
        torch.rand(rows, HC, device=device, generator=gen),
        torch.randn(rows, HC, HC, device=device, generator=gen) * 0.1,
        torch.rand(rows, HC, device=device, generator=gen),
        torch.randn(HIDDEN, device=device, generator=gen).bfloat16(),
    )


def _torch_boundary(x, mix, group):
    residual, post, comb, pre, norm_weight = mix
    y = x.float()
    dist.all_reduce(y, group=group)
    y = y.bfloat16().float()
    streams = post.unsqueeze(-1) * y.unsqueeze(1) + (
        comb.unsqueeze(-1) * residual.float().unsqueeze(2)
    ).sum(dim=1)
    streams = streams.bfloat16()
    collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=1)
    normed = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-6)
    return streams, (normed * norm_weight.float()).bfloat16()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)
def test_wo_b_publish_matches_separate_publish(rank):
    from sglang.kernels.ops.communication.dsv41_lamport.all_reduce_mhc import (
        AllReduceMHC,
    )
    from sglang.kernels.ops.communication.dsv41_lamport.publishing_gemm import (
        PublishingMxfp8Linear,
    )

    tp_size = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    with torch.inference_mode():
        arm = AllReduceMHC(
            hidden_size=HIDDEN, hc_mult=HC, max_num_tokens=16, top_k=6, device=device
        )
        publisher = PublishingMxfp8Linear(arm)
        gen = torch.Generator(device=device).manual_seed(1000 + rank)
        layers = [_linear(device, gen) for _ in range(2)]

        def boundary(x, mix):
            residual, post, comb, pre, norm_weight = mix
            return arm(x, residual, post, comb, pre, norm_weight, 1e-6)

        # Which path each boundary of a sequence takes (True: published). The
        # sequence crosses the three generations several times on both paths.
        pattern = (True, True, True, False, True, False, False, True, True)
        for rows in (1, 6, 8):
            qas = [_activation(rows, device, gen) for _ in layers]
            mix = _mixes(rows, device, rank)
            assert publisher.supports(layers[0], rows)
            assert publisher.prepare(layers[0], qas[0]) is not None
            _, valid = _flashinfer_gemm(layers[0], qas[0], None)
            by_split = {}
            for tactic in valid:
                by_split.setdefault(tactic[3], tuple(tactic))
            # The default tactic (what an untuned mm_mxfp8 runs) and the first
            # without split-K and with split-K 4 (both epilogue branches).
            tactics = list(
                dict.fromkeys(
                    [tuple(valid[0])] + [by_split[s] for s in (1, 4) if s in by_split]
                )
            )
            for tactic in tactics:
                expected_c = [
                    _flashinfer_gemm(layer, qa, tactic)[0]
                    for layer, qa in zip(layers, qas)
                ]
                for late in range(tp_size):
                    for index, published in enumerate(pattern):
                        li = index % len(layers)
                        if rank == late:
                            torch.cuda._sleep(20_000)
                        if published:
                            prepared = publisher.prepare(layers[li], qas[li])
                            x = publisher(layers[li], prepared, tactic=tactic)
                            assert torch.equal(
                                x.view(torch.int16), expected_c[li].view(torch.int16)
                            ), f"C differs: rows={rows} tactic={tactic}"
                            got = boundary(x, mix)
                        else:
                            got = boundary(expected_c[li].clone(), mix)
                        # The separate publish kernel on the same C, one boundary
                        # later, so the generations must line up across paths.
                        reference = boundary(expected_c[li].clone(), mix)
                        for a, b in zip(got, reference):
                            assert torch.equal(
                                a.view(torch.int16), b.view(torch.int16)
                            ), (
                                f"boundary differs: rows={rows} tactic={tactic} "
                                f"late={late} step={index}"
                            )
                torch.cuda.synchronize()

            # Against torch: the math SGLang's own fused attention boundary runs.
            c = _flashinfer_gemm(layers[0], qas[0], tactics[0])[0]
            streams, normed = boundary(
                publisher(
                    layers[0],
                    publisher.prepare(layers[0], qas[0]),
                    tactic=tactics[0],
                ),
                mix,
            )
            ref_streams, ref_normed = _torch_boundary(
                c, mix, ps.get_tp_group().device_group
            )
            torch.testing.assert_close(
                streams.float(), ref_streams.float(), atol=3e-2, rtol=2e-2
            )
            torch.testing.assert_close(
                normed.float(), ref_normed.float(), atol=6e-2, rtol=3e-2
            )

            # One CUDA graph of published and separate boundaries, replayed.
            tactic = tactics[0]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                captured = []
                for index, published in enumerate(pattern):
                    li = index % len(layers)
                    if published:
                        prepared = publisher.prepare(layers[li], qas[li])
                        x = publisher(layers[li], prepared, tactic=tactic)
                    else:
                        x = _flashinfer_gemm(layers[li], qas[li], tactic)[0]
                    captured.append(boundary(x, mix))
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(3):
                graph.replay()
                torch.cuda.synchronize()
                for index, outputs in enumerate(captured):
                    li = index % len(layers)
                    reference = boundary(
                        _flashinfer_gemm(layers[li], qas[li], tactic)[0], mix
                    )
                    for a, b in zip(outputs, reference):
                        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
            torch.cuda.synchronize()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
