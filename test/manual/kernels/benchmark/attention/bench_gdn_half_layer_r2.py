# SPDX-License-Identifier: Apache-2.0
"""Time the shipped FP8 projection and compensated recurrent verify chain."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics

for flag in ("VIEWS", "LAZY_VALID", "CONV_TILE", "TILED_VERIFY", "NORM_FP8", "PDL"):
    os.environ["SGLANG_Q35MF_GDN_" + flag] = "1"

import torch

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import gdn_replayssm_spec_decode
from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
from sglang.kernels.ops.attention.triton_gdn_fused_proj import qwen3_5_gdn_prefill_projection_views
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_bmm_flashinfer


def emit(**data):
    print(json.dumps(data), flush=True)


class Layer:
    def __init__(self, tp, seed):
        self.h, self.hv = 16 // tp, 64 // tp
        self.dim, self.tokens, slots, ring = 12288 // tp, 7, 4, 16
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def rand(*shape, scale=0.1):
            return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.bfloat16) * scale

        self.x = rand(7, 4096)
        self.wq = rand(self.dim + self.hv * 128, 4096, scale=0.5).to(torch.float8_e4m3fn).t()
        self.wba = rand(2 * self.hv, 4096, scale=0.01)
        self.wo = rand(4096, self.hv * 128, scale=0.5).to(torch.float8_e4m3fn).t()
        self.sx = torch.tensor([0.005], device="cuda")
        self.sw = torch.tensor([0.01], device="cuda")
        self.sy = torch.tensor([0.01], device="cuda")
        self.conv_w = rand(self.dim, 4, scale=0.25)
        self.norm_w = 1.0 + rand(128)
        self.conv_state = rand(slots, self.dim, 3)
        self.physical = rand(slots, self.dim, 9)
        self.window = self.physical.as_strided((slots, 7, self.dim, 3), (self.dim * 9, 1, 9, 1))
        self.case = dict(
            A_log=torch.zeros(self.hv, device="cuda"), dt_bias=torch.zeros(self.hv, device="cuda"),
            checkpoint_state=rand(slots, self.hv, 128, 128),
            d_cache=rand(slots, self.hv, ring, 128), k_cache=rand(slots, self.h, ring, 128),
            g_cache=-torch.rand(slots, self.hv, ring, device="cuda", generator=gen),
            rawv_cache=rand(slots, self.hv, ring, 128), rawk_cache=rand(slots, self.h, ring, 128),
            beta_cache=None, out=rand(7, self.hv, 128),
            query_start_loc=torch.tensor([0, 7], dtype=torch.int32, device="cuda"),
            ssm_state_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
            replay_indices=torch.tensor([2], dtype=torch.int32, device="cuda"),
            write_pos=torch.zeros(slots, dtype=torch.int32, device="cuda"),
            cache_base=torch.full((slots,), 15, dtype=torch.int32, device="cuda"),
            is_flush=torch.zeros(slots, dtype=torch.int8, device="cuda"),
            max_cache_len=ring, max_spec_len=7, null_block_id=-1, launch_mode="verify",
        )
        self.last = None

    def forward(self, alt):
        main = torch.cuda.current_stream()
        alt.wait_stream(main)
        qkvz = apply_fp8_linear_bmm_flashinfer(self.x, self.wq, self.sw, self.sx)
        with torch.cuda.stream(alt):
            ba = torch.nn.functional.linear(self.x, self.wba)
        main.wait_stream(alt)
        mixed, z, b, a = qwen3_5_gdn_prefill_projection_views(qkvz, ba, self.h, self.hv, 128, 128)
        conv = causal_conv1d_update(
            mixed.view(1, 7, self.dim).transpose(1, 2), self.conv_state, self.conv_w,
            activation="silu", conv_state_indices=self.case["ssm_state_indices"],
            intermediate_conv_window=self.window, intermediate_state_indices=self.case["replay_indices"], token_major_out=True, block_n=128,
        ).transpose(1, 2).view(7, self.dim)
        q, k, v = conv.split((self.h * 128, self.h * 128, self.hv * 128), dim=-1)
        gdn_replayssm_spec_decode(
            **self.case, q=q.view(7, self.h, 128), k=k.view(7, self.h, 128),
            v=v.view(7, self.hv, 128), a=a, b=b, block_v=32, num_warps=2, pdl=True,
        )
        (quant, _), _, _ = _layer_norm_fwd(
            self.case["out"].view(-1, 128), self.norm_w, None, 1e-6,
            z=z, is_rms_norm=True, static_quant_scale=self.sy,
        )
        out = apply_fp8_linear_bmm_flashinfer(quant.view(7, self.hv * 128), self.wo, self.sw, self.sy)
        self.last = dict(conv=conv, recurrence=self.case["out"], quant=quant, out=out)

    def bytes(self):
        rows = dict(self.last, state=self.conv_state, physical=self.physical)
        rows.update({k: v for k, v in self.case.items() if isinstance(v, torch.Tensor)})
        return {k: v.contiguous().view(torch.uint8).clone() for k, v in rows.items()}


def capture(layers, alt):
    for layer in layers:
        layer.forward(alt)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in layers:
            layer.forward(alt)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def measure(graph, calls):
    values = []
    for _ in range(9):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (5 * calls))
    return dict(median_us=statistics.median(values), range_us=[min(values), max(values)], samples_us=values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=[4, 8], default=8)
    parser.add_argument("--seed", type=int, default=341)
    parser.add_argument("--layers", type=int, default=45)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    sources = {}
    for fn in (gdn_replayssm_spec_decode, causal_conv1d_update, _layer_norm_fwd, apply_fp8_linear_bmm_flashinfer):
        p = Path(inspect.getsourcefile(fn)); data = p.read_bytes()
        sources[p.name] = dict(bytes=len(data), md5=hashlib.md5(data).hexdigest(), sha256=hashlib.sha256(data).hexdigest())
    emit(kind="metadata", args=vars(args), torch=torch.__version__, cuda=torch.version.cuda,
         runtime_properties=str(props), capability=torch.cuda.get_device_capability(),
         sources=sources, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         mode="single-GPU static microbenchmark; synthetic weights; no serving claim",
         scope="already normalized BF16 input through local unreduced BF16 out_proj; no hop or commit",
         numerics="stock A/A byte control; candidate not implemented", history=0)
    arms = {name: [Layer(args.tp, args.seed + i) for i in range(args.layers)] for name in ("A", "N")}
    alt = torch.cuda.Stream()
    for name in arms:
        arms[name][0].forward(alt)
    torch.cuda.synchronize()
    aa, nn = (arms[name][0].bytes() for name in ("A", "N"))
    checks = {k: torch.equal(v, nn[k]) for k, v in aa.items()}
    emit(kind="stock_bytes", checks=checks, passed=all(checks.values()))
    assert all(checks.values())
    emit(kind="banks", independent_banks_per_arm=args.layers,
         weight_bytes_per_arm=sum(x.wq.numel() + x.wo.numel() + x.wba.numel() * 2 for x in arms["A"]),
         example_weight_strides=dict(qkvz=arms["A"][0].wq.stride(), out=arms["A"][0].wo.stride()))
    graphs = {name: capture(layers, alt) for name, layers in arms.items()}
    for mirror, order in enumerate((("A", "N"), ("N", "A"))):
        for name in order:
            emit(kind="timing", arm=name, mirror=mirror, **measure(graphs[name], args.layers))


if __name__ == "__main__":
    main()
