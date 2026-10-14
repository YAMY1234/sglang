"""CUDA schedule parity: all-layer side-stream expiry vs serial post-order MGS.

Not an alternate numerical gate: the M1 FP64 + periodic optimal-rank oracle
is run separately. This checks stream ownership, changing phases and replay.
"""
import argparse
import json
from pathlib import Path

import torch
from test_mamba2_factored import kernel


def case(batch):
    torch.manual_seed(41)
    layers, heads, groups, p, n, size = 3, 4, 2, 64, 128, batch + 2
    direction = torch.nn.functional.normalize(torch.randn(layers, heads, p, device='cuda'), dim=-1)
    a = torch.zeros(layers, size, heads, n, device='cuda')
    u = torch.zeros(layers, size, heads, 32, p, device='cuda')
    v = torch.zeros(layers, size, heads, 32, n, device='cuda')
    phase = (torch.arange(layers, device='cuda')[:, None, None] +
             torch.arange(size, device='cuda')[None, :, None]) % 16
    count = (16 + phase).expand(layers, size, heads).to(torch.int32).contiguous()
    state = [a, u, v, count]
    x = torch.randn(layers, batch, heads, p, device='cuda').bfloat16()
    b = torch.randn(layers, batch, groups, n, device='cuda').bfloat16()
    c = torch.randn_like(b)
    dt = torch.randn(layers, batch, heads, device='cuda').bfloat16()
    decay_a = torch.full((layers, heads), -0.15, device='cuda')
    bias = torch.randn(layers, heads, device='cuda').bfloat16()
    d = torch.randn_like(bias)
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device='cuda')
    slots = indices.clone()
    out, refout = torch.empty_like(x), torch.empty_like(x)
    side = torch.cuda.Stream()

    def run(buffers, output, batched):
        aa, uu, vv, cc = buffers
        for layer in range(layers):
            kernel.step(x[layer], b[layer], c[layer], dt[layer], decay_a[layer], d[layer],
                        direction[layer], aa[layer], uu[layer], vv[layer], cc[layer], slots,
                        dt_bias=bias[layer], do_truncate=False, out=output[layer])
            if not batched:
                kernel.truncate(uu[layer], vv[layer], cc[layer], slots)
        if batched:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                kernel.truncate_layers(uu, vv, cc, slots)
            torch.cuda.current_stream().wait_stream(side)

    run(state, out, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(state, out, True)
    reference = [tensor.clone() for tensor in state]
    for step in range(64):
        x.mul_(0.99).add_(0.001)
        dt.mul_(0.999)
        slots.copy_(indices.roll(step))
        if step % 3 == 0:
            slots[-1] = -1
        run(reference, refout, False)
        graph.replay()
        torch.testing.assert_close(out, refout, rtol=0, atol=0)
        for actual, expected in zip(state, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return dict(batch=batch, layers=layers, heads=heads, replay_steps=64,
                changing_inputs_slots_padding_phases='PASS', serial_state_output_exact='PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    rows = []
    for batch in (1, 8, 32):
        rows.append(case(batch))
        Path(args.out).write_text(json.dumps(dict(rows=rows), indent=2) + '\n')
        print(json.dumps(rows[-1]), flush=True)


if __name__ == '__main__':
    main()
