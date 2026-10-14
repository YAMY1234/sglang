"""Exercise factor slots, extend commit and changing-slot CUDA graph replay."""

import json
from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.kda_factored_pool import FactoredKDAConfig, FactoredKDAPool


def main():
    torch.manual_seed(206)
    shape = SimpleNamespace(temporal=(2, 128, 128), conv=[(3, 768)])
    cfg = FactoredKDAConfig(32, 32)
    params = SimpleNamespace(shape=shape)
    pool = FactoredKDAPool(5, params, [3, 7], "cuda", cfg)
    pool.bind_directions({lid: torch.randn(2, 128, device="cuda") for lid in (3, 7)})
    slots = torch.tensor([1, 4], device="cuda")
    u = torch.randn(2, 2, 128, 8, device="cuda")
    v = torch.randn(2, 2, 128, 8, device="cuda")
    state = (u @ v.transpose(-1, -2)).contiguous()
    pool.commit(3, slots, state)
    recovered = pool.dense(3, slots)
    error = (recovered - state).norm() / state.norm()
    envelope = (state.bfloat16().float() - state).norm() / state.norm()
    assert error <= envelope, (error, envelope)
    saved = pool.get_cpu_slots(slots)
    pool.copy_slots(slots, slots.flip(0))
    torch.testing.assert_close(pool.dense(3, slots.flip(0)), recovered, rtol=0, atol=0)
    pool.reset_slots(slots)
    assert pool.dense(3, slots).count_nonzero() == 0
    pool.load_cpu_slots(saved, slots)
    torch.testing.assert_close(pool.dense(3, slots), recovered, rtol=0, atol=0)
    # Unfinished prefills retain exact states through slot moves and CPU backup.
    with pool.prefill(slots, [False, False]):
        pool.commit(3, slots, state)
    pending = pool.get_cpu_slots(slots)
    pool.copy_slots(slots, slots.flip(0))
    torch.testing.assert_close(pool.dense(3, slots.flip(0)), state, rtol=0, atol=0)
    pool.reset_slots(slots)
    assert not pool._continuations
    pool.load_cpu_slots(pending, slots)
    torch.testing.assert_close(pool.dense(3, slots), state, rtol=0, atol=0)
    with pool.prefill(slots, [True, True]):
        pool.commit(3, slots, state)
    assert not pool._continuations
    # Prefill emitter graphs call commit, with dynamic dense data and slot IDs.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.commit(7, slots, state)
    slots.copy_(torch.tensor([2, 3], device="cuda"))
    state.mul_(0.5)
    graph.replay()
    recovered = pool.dense(7, slots)
    error2 = (recovered - state).norm() / state.norm()
    assert error2 <= envelope, (error2, envelope)
    # Decode padding, per-slot unsynchronized counts, two expiry boundaries.
    layer = SimpleNamespace(
        layer_id=7,
        A_log=torch.full((2,), -4.0, device="cuda"),
        dt_bias=torch.zeros(256, device="cuda"),
        lower_bound=None,
    )
    indices = torch.tensor([2, -1, 3], device="cuda")
    mixed = torch.randn(3, 768, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(3, 2, device="cuda", dtype=torch.bfloat16)
    pool.count[1, 3] = 40
    padding_before = [x[:, 0].clone() for x in pool.tensors()]
    pool.decode(layer, mixed, gate, beta, indices)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = pool.decode(layer, mixed, gate, beta, indices)
    initial = pool.count[1].clone()
    for t in range(64):
        mixed.normal_()
        graph.replay()
        assert out[0, 1].count_nonzero() == 0
        expected = 32 + (initial[slots] - 32 + t + 1) % 32
        torch.testing.assert_close(pool.count[1, slots], expected, rtol=0, atol=0)
    for x, saved in zip(pool.tensors(), padding_before):
        torch.testing.assert_close(x[:, 0], saved, rtol=0, atol=0)
    actual = sum(x.numel() * x.element_size() for x in pool.tensors())
    assert actual == 2 * 6 * cfg.state_bytes_per_layer(shape)
    print(
        json.dumps(
            {
                "pool_gate": "PASS",
                "actual_slot_bytes_all_layers": actual // 6,
                "commit_rel_error": error.item(),
                "commit_graph_rel_error": error2.item(),
                "bf16_rel_envelope": envelope.item(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
