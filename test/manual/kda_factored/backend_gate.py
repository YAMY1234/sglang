"""Native HybridReqToTokenPool + KDA backend, without model weights."""

import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    KDAKernelDispatcher,
)
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def make_backend(factored):
    os.environ["SGLANG_KDA_FACTORED_STATE"] = "r=32,w=32" if factored else "0"
    shape = KimiLinearStateShape.create(tp_world_size=1, num_heads=2, head_dim=128)
    params = KimiLinearCacheParams(
        shape=shape,
        layers=[0],
        dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
    )
    pool = HybridReqToTokenPool(
        size=2,
        mamba_size=4,
        mamba_spec_state_size=2,
        max_context_len=128,
        device="cuda",
        enable_memory_saver=False,
        cache_params=params,
        mamba_layer_ids=[0],
        enable_mamba_extra_buffer=False,
    )
    backend = KDAAttnBackend.__new__(KDAAttnBackend)
    backend.req_to_token_pool = pool
    backend.state_pruner = None
    backend.accept_lens_pool = None
    backend.kernel_dispatcher = KDAKernelDispatcher(
        *[LinearAttnKernelBackend("triton")] * 3
    )
    backend.forward_metadata = SimpleNamespace(
        mamba_cache_indices=torch.tensor([1, 3], device="cuda", dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32),
        has_mamba_track_mask=False,
        track_chunk_idx=None,
    )
    return backend, params


def main():
    torch.manual_seed(206)
    args = ServerArgs(model_path="dummy", page_size=1, disable_radix_cache=True)
    args._mamba_cache_chunk_size = 64
    set_global_server_args_for_scheduler(args)
    factor, params = make_backend(True)
    stock, _ = make_backend(False)
    pool = factor.req_to_token_pool.factored_kda_pool
    pool.bind_directions({0: torch.randn(2, 128, device="cuda")})
    assert factor.req_to_token_pool.mamba_pool.mamba_cache.temporal.numel() == 0
    assert stock.req_to_token_pool.mamba_pool.mamba_cache.temporal.numel() > 0
    layer = SimpleNamespace(
        layer_id=0,
        num_v_heads=2,
        head_q_dim=128,
        head_k_dim=128,
        head_v_dim=128,
        q_dim=256,
        k_dim=256,
        v_dim=256,
        conv_weights=torch.randn(768, 4, device="cuda") * 0.05,
        bias=None,
        A_log=torch.full((2,), -4.0, device="cuda"),
        dt_bias=torch.zeros(256, device="cuda"),
        lower_bound=None,
    )
    mode = SimpleNamespace(
        is_target_verify=lambda: False, is_draft_extend_v2=lambda: False
    )
    batch = SimpleNamespace(
        forward_mode=mode,
        extend_prefix_lens=torch.zeros(2, device="cuda"),
        extend_seq_lens_cpu=[4, 4],
        mamba_track_mask=None,
    )
    mixed = torch.randn(8, 768, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(1, 8, 2, device="cuda", dtype=torch.bfloat16)
    slots = factor.forward_metadata.mamba_cache_indices.long()
    with pool.prefill(slots, [False, False]):
        f = factor.forward_extend(layer, batch, mixed.clone(), gate, beta)
    d = stock.forward_extend(layer, batch, mixed.clone(), gate, beta)
    torch.testing.assert_close(f, d, rtol=0, atol=0)
    torch.testing.assert_close(
        pool.dense(0, slots),
        stock.req_to_token_pool.mamba2_layer_cache(0).temporal[slots],
        rtol=0,
        atol=0,
    )
    batch.extend_prefix_lens.fill_(4)
    with pool.prefill(slots, [True, True]):
        f = factor.forward_extend(layer, batch, mixed.clone(), gate, beta)
    d = stock.forward_extend(layer, batch, mixed.clone(), gate, beta)
    assert not pool._continuations
    torch.testing.assert_close(f, d, rtol=0, atol=0)
    slots = factor.forward_metadata.mamba_cache_indices.long()
    expected = stock.req_to_token_pool.mamba2_layer_cache(0).temporal[slots]
    error = (pool.dense(0, slots) - expected).norm() / expected.norm()
    envelope = (expected.bfloat16().float() - expected).norm() / expected.norm()
    assert error <= envelope
    for backend in (factor, stock):
        backend.forward_metadata.query_start_loc = torch.arange(
            3, device="cuda", dtype=torch.int32
        )
    max_error = 0.0
    reference = expected.double().transpose(-1, -2)
    low = reference.bfloat16().double()
    captured = {}
    original_decode = stock.kernel_dispatcher.packed_decode

    def capture_decode(**kwargs):
        captured["qkv"] = kwargs["mixed_qkv"].clone()
        return original_decode(**kwargs)

    stock.kernel_dispatcher.packed_decode = capture_decode
    envelope_records = []
    ulp_violations = 0
    for step in range(31):
        mixed = torch.randn(2, 768, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(1, 2, 256, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, 2, 2, device="cuda", dtype=torch.bfloat16)
        f = factor.forward_decode(layer, batch, mixed.clone(), gate, beta)
        d = stock.forward_decode(layer, batch, mixed.clone(), gate, beta)
        max_error = max(max_error, (f.float() - d.float()).abs().max().item())
        query, key, value = captured["qkv"].double().reshape(2, 3, 2, 128).unbind(1)
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() / 128**0.5
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        decay = (
            -layer.A_log.double().exp()[None, :, None]
            * F.softplus(gate.double().reshape(2, 2, 128))
        ).exp()
        bet = beta.double().reshape(2, 2).sigmoid()

        def advance(state, decay=decay, bet=bet, value=value, key=key):
            state = state * decay[..., :, None]
            delta = bet[..., None] * (value - (state * key[..., :, None]).sum(-2))
            return state + key[..., :, None] * delta[..., None, :]

        reference = advance(reference)
        low = advance(low).bfloat16().double()
        out64 = (reference * query[..., :, None]).sum(-2)
        env = (
            ((low * query[..., :, None]).sum(-2).bfloat16().double() - out64)
            .square()
            .mean()
            .sqrt()
        )
        err = (f[0].double() - out64).square().mean().sqrt()
        ulp_violations += (
            (
                (f.float() - d.float()).abs()
                > torch.maximum(d.abs().float(), f.abs().float()) * (2**-7)
            )
            .count_nonzero()
            .item()
        )
        envelope_records.append([err.item(), env.item()])
        assert err <= env, (step, err.item(), env.item(), max_error)
    native = factor.req_to_token_pool.mamba_pool
    source = torch.tensor([1], device="cuda")
    destination = torch.tensor([2], device="cuda")
    native.copy_from(source, destination)
    torch.testing.assert_close(
        pool.dense(0, source), pool.dense(0, destination), rtol=0, atol=0
    )
    native.clear_slots(destination)
    assert pool.dense(0, destination).count_nonzero() == 0
    print(
        json.dumps(
            {
                "backend_gate": "PASS",
                "dense_temporal_numel": 0,
                "extend_relative_error": error.item(),
                "bf16_relative_envelope": envelope.item(),
                "decode_max_abs_error": max_error,
                "fp64_rmse_and_bf16_envelope": envelope_records,
                "per_element_ulp_violations_diagnostic": ulp_violations,
                "accounted_bytes_per_req": pool.cfg.per_req_bytes(params),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
