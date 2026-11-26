import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.speculative.domino_corrector import (
    corrector_act_down_residual,
    corrector_candidate_scores,
    corrector_context_attention,
    corrector_draw,
    corrector_history_attention,
    corrector_norm_matmul,
    corrector_output_head,
    corrector_pool_max,
    corrector_ring_context_attention,
    corrector_topk,
)
from sglang.srt.speculative.domino_corrector import (
    CorrectorContext,
    CorrectorSampling,
    TransformerCorrector,
    _context_attention,
    _draw,
    _history_attention,
    _rope_tables,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HEADS, HEAD_DIM = 4, 64
WIDTH = HEADS * HEAD_DIM


def _assert_matches(out, ref):
    # The kernels accumulate in FP32 and round once; the reference gets the same
    # values widened, so BF16 differs by the final rounding only.
    rtol = 4e-3 if out.dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out.float().cpu(), ref, rtol=rtol, atol=1e-5)


def _randn(shape, dtype, gen):
    return torch.randn(shape, device="cuda", dtype=dtype, generator=gen)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bs,slots", [(1, 7), (3, 7), (33, 4), (0, 7)])
def test_history_attention_matches_reference(dtype, bs, slots):
    gen = torch.Generator(device="cuda").manual_seed(bs * 10 + slots)
    # Unwritten slots must never be read.
    history = torch.full(
        (bs, HEADS, slots, 2 * HEAD_DIM), float("nan"), device="cuda", dtype=dtype
    )
    ref_history = history.float().cpu()
    for slot in range(slots):
        qkv = _randn((bs, 2, 3 * WIDTH), dtype, gen)
        qkv[..., :WIDTH] *= HEAD_DIM**-0.5
        out = corrector_history_attention(qkv, history, slot, heads=HEADS)
        ref = _history_attention(qkv.float().cpu(), ref_history, slot, HEADS)
        _assert_matches(out, ref)
        torch.testing.assert_close(
            history[:, :, : slot + 1].float().cpu(),
            ref_history[:, :, : slot + 1],
            rtol=0,
            atol=0,
        )
        assert history[:, :, slot + 1 :].isnan().all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "bs,window", [(1, 256), (4, 256), (4, 40), (2, 600), (33, 5), (0, 16)]
)
def test_context_attention_matches_reference(dtype, bs, window):
    layers = 2
    gen = torch.Generator(device="cuda").manual_seed(bs * 1000 + window)
    context = _randn((bs, window, layers, 2 * WIDTH), dtype, gen)
    offsets = torch.arange(-window, 0, device="cuda")
    cos, sin = _rope_tables(offsets, HEAD_DIM, torch.float32)
    cos = cos[:, : HEAD_DIM // 2].contiguous()
    sin = sin[:, : HEAD_DIM // 2].contiguous()
    # A request has the newest rows of its window: all, some, or none of them.
    held = torch.arange(bs, device="cuda") * 7 % (window + 1)
    if bs > 1:
        held[0], held[1] = window, 0
    valid = (torch.arange(window, device="cuda") >= window - held[:, None]).to(
        torch.uint8
    )
    for layer in range(layers):
        query = _randn((bs, 2, WIDTH), dtype, gen) * HEAD_DIM**-0.5
        out = corrector_context_attention(
            query, context, valid, cos, sin, layer=layer, heads=HEADS
        )
        ref = _context_attention(
            query.float().cpu(),
            context.float().cpu(),
            valid.cpu(),
            cos.cpu(),
            sin.cpu(),
            layer=layer,
            heads=HEADS,
        )
        _assert_matches(out, ref)
        if bs > 1:
            assert out[1].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("window", [256, 40, 600])
def test_ring_context_attention_matches_the_gathered_window(dtype, window):
    """Reading the window in place equals the dense call on the gathered window,
    bit for bit, including rows before the floor and before position 0."""
    layers, req_pool_rows, bs = 2, 6, 5
    gen = torch.Generator(device="cuda").manual_seed(window)
    ring_kv = _randn((1 + req_pool_rows * window, layers, 2 * WIDTH), dtype, gen)
    slots = torch.tensor([4, 0, 2, 5, 1], device="cuda")
    anchors = torch.tensor(
        [3 * window + 17, 5, window, 2 * window - 1, 0], device="cuda"
    )
    min_pos = torch.zeros(req_pool_rows, dtype=torch.int32, device="cuda")
    min_pos[4], min_pos[2], min_pos[1] = 3 * window, window // 2, 0
    positions = anchors[:, None] + torch.arange(-window, 0, device="cuda")
    valid = positions >= min_pos[slots, None]
    rows = 1 + slots[:, None] * window + torch.remainder(positions, window)
    context = ring_kv[rows].masked_fill_(~valid[:, :, None, None], 0)
    offsets = torch.arange(-window, 0, device="cuda")
    cos, sin = _rope_tables(offsets, HEAD_DIM, torch.float32)
    cos = cos[:, : HEAD_DIM // 2].contiguous()
    sin = sin[:, : HEAD_DIM // 2].contiguous()
    for layer in range(layers):
        query = _randn((bs, 2, WIDTH), dtype, gen) * HEAD_DIM**-0.5
        dense = corrector_context_attention(
            query, context, valid.to(torch.uint8), cos, sin, layer=layer, heads=HEADS
        )
        ring = corrector_ring_context_attention(
            query, ring_kv, slots, anchors, min_pos, cos, sin,
            sink_rows=1, layer=layer, heads=HEADS,
        )  # fmt: skip
        assert torch.equal(ring, dense)
        # Request 4 holds no row: every position of its window is negative.
        assert ring[4].count_nonzero() == 0


@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bs,pool", [(1, 2048), (7, 2048), (3, 100), (0, 64)])
def test_candidate_scores_match_reference(table_dtype, bs, pool):
    gen = torch.Generator(device="cuda").manual_seed(bs * 100 + pool)
    vocab, emb, slots = 5000, 256, 3
    table = _randn((vocab, emb), table_dtype, gen)
    candidate_ids = torch.randint(0, vocab, (bs, pool), device="cuda", generator=gen)
    # One slot of a [batch, slots, K] block: strided rows, as the rollout passes it.
    base = _randn((bs, slots, pool), torch.bfloat16, gen)[:, 1]
    correction = _randn((bs, emb), torch.float32, gen)
    out = corrector_candidate_scores(base, candidate_ids, table, correction)
    weight = F.embedding(candidate_ids.cpu(), table.cpu()).double()
    ref = torch.baddbmm(
        base.cpu().double()[:, :, None], weight, correction.cpu().double()[:, :, None]
    )[:, :, 0]
    assert out.dtype == torch.float32 and out.shape == (bs, pool)
    torch.testing.assert_close(out.cpu().double(), ref, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bs", [1, 116, 0])
def test_output_head_matches_reference(dtype, bs):
    gen = torch.Generator(device="cuda").manual_seed(bs + 3)
    width, emb, slots, eps = 256, 256, 5, 1e-5
    streams = _randn((bs, 2, width), dtype, gen)
    norm_weight = _randn((width,), dtype, gen)
    state_t = _randn((width, emb), torch.float32, gen) * width**-0.5
    hidden_part = _randn((bs, slots, emb), torch.float32, gen)[:, 2]
    out = corrector_output_head(streams, norm_weight, eps, state_t, hidden_part)
    x = streams[:, 1].cpu().double()
    state = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    state = state * norm_weight.cpu().double()
    ref = F.silu(hidden_part.cpu().double() + state @ state_t.cpu().double())
    torch.testing.assert_close(out.cpu().double(), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bs,steps,vocab", [(116, 5, 40960), (3, 1, 1000), (0, 5, 64)])
def test_pool_max_equals_amax(dtype, bs, steps, vocab):
    gen = torch.Generator(device="cuda").manual_seed(steps * vocab)
    # A block's corrected steps: a strided slice of [batch, steps + 1, vocab] logits.
    logits = _randn((bs, steps + 1, vocab), dtype, gen)[:, 1:]
    assert torch.equal(corrector_pool_max(logits), logits.amax(dim=1))


@pytest.mark.parametrize("kind", ["normal", "ties", "zeros"])
@pytest.mark.parametrize(
    "bs,n,k", [(116, 40960, 2048), (116, 8192, 2048), (3, 4096, 1), (2, 4096, 4096)]
)
def test_topk_lays_out_like_torch(kind, bs, n, k):
    """Values and indices equal torch.topk(sorted=False)'s, position by position."""
    gen = torch.Generator(device="cuda").manual_seed(n + k)
    x = _randn((bs, n), torch.bfloat16, gen)
    if kind == "ties":
        # Few distinct values, so the k-th value is shared by many entries.
        x = (x * 2).round() / 2
        x[:, ::97] = -torch.inf
    elif kind == "zeros":
        x = torch.where(
            torch.rand(bs, n, device="cuda", generator=gen) < 0.5, 0.0, -0.0
        )
        x = x.to(torch.bfloat16)
    values, ids = corrector_topk(x, k)
    ref = torch.topk(x, k, dim=-1, sorted=False)
    assert torch.equal(values.view(torch.int16), ref.values.view(torch.int16))
    assert torch.equal(ids, ref.indices)


def _bf16_close(out, ref):
    # Both round FP32 sums of the same products to BF16: at most a final ulp apart.
    torch.testing.assert_close(out.float().cpu(), ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "rows,n", [(232, 768), (232, 256), (232, 2048), (5, 768), (0, 256)]
)
def test_norm_matmul_matches_reference(rows, n):
    gen = torch.Generator(device="cuda").manual_seed(rows + n)
    k, eps = 256, 1e-5
    x = _randn((rows, k), torch.bfloat16, gen) * 3
    norm_weight = _randn((k,), torch.bfloat16, gen)
    weight = _randn((k, n), torch.bfloat16, gen) * k**-0.5
    out = corrector_norm_matmul(x, norm_weight, eps, weight)
    wide = x.float().cpu()
    normed = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps)
    normed = (normed * norm_weight.float().cpu()).bfloat16().double()
    _bf16_close(out, (normed @ weight.double().cpu()).bfloat16())


@pytest.mark.parametrize("rows", [232, 7, 0])
def test_act_down_residual_matches_reference(rows):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    k, n = 1024, 256
    gate_up = _randn((rows, 2 * k), torch.bfloat16, gen)
    weight = _randn((k, n), torch.bfloat16, gen) * k**-0.5
    residual = _randn((rows, n), torch.bfloat16, gen)
    expected_in = residual.clone()
    corrector_act_down_residual(gate_up, weight, residual)
    gate, up = gate_up.double().cpu().chunk(2, dim=-1)
    hidden = (F.silu(gate) * up).bfloat16().double()
    ref = hidden @ weight.double().cpu() + expected_in.double().cpu()
    _bf16_close(residual, ref.bfloat16())


def _rollout_case(dtype):
    torch.manual_seed(5)
    hidden_size, vocab, window, req_pool_rows = 64, 300, 16, 4
    config = {
        "d_model": WIDTH,
        "heads": HEADS,
        "layers": 2,
        "context_window": window,
        "context_attention": True,
    }
    corrector = TransformerCorrector(hidden_size, config, eps=1e-5).cuda().to(dtype)
    embed = nn.Embedding(vocab, hidden_size).cuda().to(dtype)
    embed_proj = (
        nn.Sequential(
            nn.Linear(hidden_size + WIDTH, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, vocab, bias=False),
        )
        .cuda()
        .to(dtype)
    )
    context = CorrectorContext(
        corrector,
        pool=SimpleNamespace(maybe_get_custom_mem_pool=lambda: None),
        req_pool_rows=req_pool_rows,
    )
    context.kv.normal_()
    context.ring.min_pos.copy_(torch.tensor([0, 7, 30, 2], dtype=torch.int32))
    return corrector, embed, embed_proj, context, vocab


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rollout_on_the_ring_window_matches_the_gathered_window(dtype):
    """The rollout reading its context in place proposes what it proposes on the
    gathered window, with the same q, bit for bit."""
    corrector, embed, embed_proj, context, vocab = _rollout_case(dtype)
    bs, slots, pool = 3, 5, 40
    req_pool_indices = torch.tensor([2, 0, 3], device="cuda")
    anchors = torch.tensor([40, 9, 3], device="cuda")
    hidden = torch.randn(bs, slots, 64, device="cuda", dtype=dtype)
    unary = torch.randn(bs, slots, vocab, device="cuda", dtype=dtype)
    candidate_ids = torch.stack(
        [torch.randperm(vocab)[:pool] for _ in range(bs)]
    ).cuda()

    def propose(**context_args):
        torch.manual_seed(11)
        return corrector.propose(
            hidden=hidden,
            anchors=torch.tensor([1, 2, 3], device="cuda"),
            token_inputs=corrector.token_table(embed, vocab),
            embed_proj=embed_proj,
            unary_picks=unary[:, :0].argmax(-1),
            unary_scores=unary.gather(
                2, candidate_ids[:, None, :].expand(-1, slots, -1)
            ),
            candidate_ids=candidate_ids,
            sampling=CorrectorSampling(
                temperatures=torch.tensor([0.7, 1.0, 1.3], device="cuda"),
                greedy_mask=torch.tensor([False, True, False], device="cuda"),
                noise=torch.empty(bs, pool, device="cuda"),
            ),
            **context_args,
        )

    context_kv, context_mask = context.gather(req_pool_indices, anchors)
    dense = propose(context_kv=context_kv, context_mask=context_mask)
    ring = propose(
        context_kv=None,
        context_mask=None,
        context_window=context.window(req_pool_indices, anchors),
    )
    assert torch.equal(ring[0], dense[0]) and torch.equal(ring[1], dense[1])


@pytest.mark.parametrize("bs,pool", [(1, 2048), (5, 2048), (33, 100), (4, 1), (0, 64)])
def test_draw_matches_reference(bs, pool):
    gen = torch.Generator(device="cuda").manual_seed(bs * 10000 + pool)
    scores = 3 * _randn((bs, pool), torch.float32, gen)
    candidate_ids = torch.randint(0, 163840, (bs, pool), device="cuda", generator=gen)
    temperatures = 0.3 + torch.rand(bs, device="cuda", generator=gen)
    greedy_mask = torch.arange(bs, device="cuda") % 3 == 1
    # A greedy row may carry a temperature of 0, clamped just above it.
    temperatures[greedy_mask] = 1e-5
    torch.manual_seed(bs + pool)
    sampling = CorrectorSampling(
        temperatures=temperatures.cpu(),
        greedy_mask=greedy_mask.cpu(),
        noise=torch.empty(bs, pool),
    )
    ref_tokens, ref_q = _draw(scores.cpu(), candidate_ids.cpu(), sampling)
    # The reference drew the noise into its scratch; the kernel takes the same draws.
    tokens, q = corrector_draw(
        scores,
        sampling.noise.cuda(),
        temperatures,
        greedy_mask.view(torch.uint8),
        candidate_ids,
    )
    assert torch.equal(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(q.cpu(), ref_q, rtol=1e-4, atol=1e-7)
    greedy_q = q[greedy_mask]
    assert torch.equal(greedy_q.sum(dim=-1), torch.ones_like(greedy_q[:, 0]))
    assert torch.equal(
        candidate_ids[greedy_mask].gather(1, greedy_q.argmax(dim=-1, keepdim=True)),
        tokens[greedy_mask, None],
    )


def test_draw_survives_a_zero_noise_draw():
    scores = torch.zeros((1, 8), device="cuda")
    noise = torch.ones((1, 8), device="cuda")
    noise[0, 5] = 0.0
    tokens, q = corrector_draw(
        scores,
        noise,
        torch.ones(1, device="cuda"),
        torch.zeros(1, dtype=torch.uint8, device="cuda"),
        torch.arange(8, device="cuda")[None] + 100,
    )
    assert tokens.item() == 105
    torch.testing.assert_close(q, torch.full_like(q, 1 / 8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
