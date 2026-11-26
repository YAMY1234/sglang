"""Steps of the Domino transformer corrector's rollout, one launch each.

The corrector advances a draft block slot by slot with two streams per request
(``sglang.srt.speculative.domino_corrector``). Each slot attends twice per block,
to the slots before it and to a window of committed target context, and ends by
drawing its token from a candidate pool. Each is a small problem per request,
where the cost of the generic route is its launches.

Coverage: CUDA. The attention steps take BF16 or FP32 activations and a
power-of-two head dimension; products and the softmax accumulate in FP32 and the
result is rounded once on store. The output head and the candidate scores run in
FP32 and read their BF16 weights in place. The draw takes FP32 scores over at
most ``DRAW_MAX_POOL`` candidates.
"""

import torch
import triton
import triton.language as tl

# Arbitrary: any finite value below every real score keeps the running maximum finite.
_MASKED = tl.constexpr(-1e30)
_NEG_INF = tl.constexpr(float("-inf"))
# The smallest positive normal FP32: a zero draw would divide by zero.
_TINY = tl.constexpr(1.1754943508222875e-38)
# One program holds a request's whole pool in registers.
DRAW_MAX_POOL = 8192


@triton.jit
def _history_attention_kernel(
    QKV,
    HISTORY,
    OUT,
    slot,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    head = pid % HEADS
    stream = (pid // HEADS) % 2
    request = pid // (2 * HEADS)
    dim = tl.arange(0, HEAD_DIM)
    # [request, stream, 3 * WIDTH]: the query, then each head's key and value.
    token_row = QKV + request * 6 * WIDTH
    query = tl.load(token_row + stream * 3 * WIDTH + head * HEAD_DIM + dim)
    # Only the token stream (0) contributes keys and values, to both streams.
    current = token_row + WIDTH + head * 2 * HEAD_DIM
    key = tl.load(current + dim)
    value = tl.load(current + HEAD_DIM + dim)
    # [request, head, slot, 2 * HEAD_DIM]. The other stream reads this slot from
    # QKV above, never from HISTORY, so the store races with no load.
    past = HISTORY + (request * HEADS + head) * SLOTS * 2 * HEAD_DIM
    append = (dim < HEAD_DIM) & (stream == 0)
    tl.store(past + slot * 2 * HEAD_DIM + dim, key, mask=append)
    tl.store(past + slot * 2 * HEAD_DIM + HEAD_DIM + dim, value, mask=append)

    query = query.to(tl.float32)
    index = tl.arange(0, BLOCK_SLOTS)
    seen = index < slot
    rows = past + index[:, None] * 2 * HEAD_DIM + dim[None, :]
    keys = tl.load(rows, mask=seen[:, None], other=0.0).to(tl.float32)
    values = tl.load(rows + HEAD_DIM, mask=seen[:, None], other=0.0).to(tl.float32)
    scores = tl.where(seen, tl.sum(keys * query[None, :], axis=1), _MASKED)
    score = tl.sum(key.to(tl.float32) * query, axis=0)
    top = tl.maximum(tl.max(scores, axis=0), score)
    weights = tl.where(seen, tl.exp(scores - top), 0.0)
    weight = tl.exp(score - top)
    attended = tl.sum(weights[:, None] * values, axis=0) + weight * value.to(tl.float32)
    attended = attended / (tl.sum(weights, axis=0) + weight)
    tl.store(OUT + (request * 2 + stream) * WIDTH + head * HEAD_DIM + dim, attended)


@triton.jit
def _context_attention_tile(
    query_left,
    query_right,
    key_left,
    key_right,
    value_left,
    value_right,
    cos,
    sin,
    valid,
    top,
    total,
    left,
    right,
):
    # One stream's online-softmax update over a tile of window rows.
    scores = tl.sum(
        query_left[None, :] * (key_left * cos - key_right * sin)
        + query_right[None, :] * (key_right * cos + key_left * sin),
        axis=1,
    )
    scores = tl.where(valid, scores, _MASKED)
    new_top = tl.maximum(top, tl.max(scores, axis=0))
    carry = tl.exp(top - new_top)
    weights = tl.where(valid, tl.exp(scores - new_top), 0.0)[:, None]
    left = left * carry + tl.sum(weights * value_left, axis=0)
    right = right * carry + tl.sum(weights * value_right, axis=0)
    total = total * carry + tl.sum(weights)
    return new_top, total, left, right


@triton.jit
def _context_attention_kernel(
    QUERY,
    CONTEXT,
    VALID,
    SLOTS,
    ANCHORS,
    MIN_POS,
    COS,
    SIN,
    OUT,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    WINDOW: tl.constexpr,
    LAYERS: tl.constexpr,
    LAYER: tl.constexpr,
    TILE: tl.constexpr,
    PAIR: tl.constexpr,
    RING: tl.constexpr,
    SINK_ROWS: tl.constexpr,
):
    # A program serves one head of one stream, or with PAIR of both streams of a
    # request, which then share each window tile they load.
    pid = tl.program_id(0)
    head = pid % HEADS
    row = pid // HEADS
    if PAIR:
        row = row * 2
    request = row // 2
    dim = tl.arange(0, HALF)
    query_0 = QUERY + row * WIDTH + head * 2 * HALF
    query_0_left = tl.load(query_0 + dim).to(tl.float32)
    query_0_right = tl.load(query_0 + HALF + dim).to(tl.float32)
    top_0 = tl.full([], _MASKED, tl.float32)
    total_0 = tl.zeros([], tl.float32)
    left_0 = tl.zeros([HALF], tl.float32)
    right_0 = tl.zeros([HALF], tl.float32)
    if PAIR:
        query_1 = query_0 + WIDTH
        query_1_left = tl.load(query_1 + dim).to(tl.float32)
        query_1_right = tl.load(query_1 + HALF + dim).to(tl.float32)
        top_1 = tl.full([], _MASKED, tl.float32)
        total_1 = tl.zeros([], tl.float32)
        left_1 = tl.zeros([HALF], tl.float32)
        right_1 = tl.zeros([HALF], tl.float32)
    # A window row's key, then its value, for each layer.
    stride = LAYERS * 2 * WIDTH
    if RING:
        slot = tl.load(SLOTS + request).to(tl.int64)
        floor = tl.load(MIN_POS + slot).to(tl.int64)
        anchor = tl.load(ANCHORS + request).to(tl.int64)
    for start in range(0, WINDOW, TILE):
        index = start + tl.arange(0, TILE)
        inside = index < WINDOW
        if RING:
            # Window row w holds position anchor - WINDOW + w, at ring row
            # SINK_ROWS + slot * WINDOW + position % WINDOW (DFlashDraftKVRing.locs).
            position = anchor - WINDOW + index
            valid = inside & (position >= floor)
            rows = SINK_ROWS + slot * WINDOW + position % WINDOW
            # A row before the floor reads as zeros, like the masked dense window.
            take = valid[:, None]
        else:
            valid = inside & (
                tl.load(VALID + request * WINDOW + index, mask=inside, other=0) != 0
            )
            rows = (request * WINDOW + index).to(tl.int64)
            take = inside[:, None]
        keys = CONTEXT + rows[:, None] * stride + LAYER * 2 * WIDTH + head * 2 * HALF
        keys = keys + dim[None, :]
        key_left = tl.load(keys, mask=take, other=0.0).to(tl.float32)
        key_right = tl.load(keys + HALF, mask=take, other=0.0).to(tl.float32)
        value_left = tl.load(keys + WIDTH, mask=take, other=0.0).to(tl.float32)
        value_right = tl.load(keys + WIDTH + HALF, mask=take, other=0.0).to(tl.float32)
        # The rotation of a window row, by its offset from the block.
        table = index[:, None] * HALF + dim[None, :]
        cos = tl.load(COS + table, mask=inside[:, None], other=0.0)
        sin = tl.load(SIN + table, mask=inside[:, None], other=0.0)
        top_0, total_0, left_0, right_0 = _context_attention_tile(
            query_0_left,
            query_0_right,
            key_left,
            key_right,
            value_left,
            value_right,
            cos,
            sin,
            valid,
            top_0,
            total_0,
            left_0,
            right_0,
        )
        if PAIR:
            top_1, total_1, left_1, right_1 = _context_attention_tile(
                query_1_left,
                query_1_right,
                key_left,
                key_right,
                value_left,
                value_right,
                cos,
                sin,
                valid,
                top_1,
                total_1,
                left_1,
                right_1,
            )
    # A request with no valid window row attends to nothing.
    out_0 = OUT + row * WIDTH + head * 2 * HALF
    scale_0 = tl.where(total_0 > 0.0, 1.0 / total_0, 0.0)
    tl.store(out_0 + dim, left_0 * scale_0)
    tl.store(out_0 + HALF + dim, right_0 * scale_0)
    if PAIR:
        out_1 = out_0 + WIDTH
        scale_1 = tl.where(total_1 > 0.0, 1.0 / total_1, 0.0)
        tl.store(out_1 + dim, left_1 * scale_1)
        tl.store(out_1 + HALF + dim, right_1 * scale_1)


@triton.jit
def _candidate_scores_kernel(
    BASE,
    base_stride,
    CANDIDATES,
    TABLE,
    table_stride,
    CORRECTION,
    OUT,
    POOL: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
):
    request = tl.program_id(0)
    index = tl.program_id(1) * BLOCK_POOL + tl.arange(0, BLOCK_POOL)
    inside = index < POOL
    dim = tl.arange(0, DIM)
    ids = tl.load(CANDIDATES + request * POOL + index, mask=inside, other=0)
    rows = TABLE + ids.to(tl.int64)[:, None] * table_stride + dim[None, :]
    weight = tl.load(rows, mask=inside[:, None], other=0.0).to(tl.float32)
    correction = tl.load(CORRECTION + request * DIM + dim)
    base = tl.load(BASE + request * base_stride + index, mask=inside, other=0.0)
    scores = base.to(tl.float32) + tl.sum(weight * correction[None, :], axis=1)
    tl.store(OUT + request * POOL + index, scores, mask=inside)


@triton.jit
def _output_head_kernel(
    STREAMS,
    NORM,
    STATE_T,
    HIDDEN_PART,
    hidden_part_stride,
    OUT,
    eps,
    WIDTH: tl.constexpr,
    EMB: tl.constexpr,
    BLOCK_EMB: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    request = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_EMB + tl.arange(0, BLOCK_EMB)
    # [request, stream, WIDTH]: the hidden stream (1) carries the slot's state.
    state_row = STREAMS + (request * 2 + 1) * WIDTH
    x = tl.load(state_row + tl.arange(0, WIDTH)).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / WIDTH + eps)
    acc = tl.zeros([BLOCK_EMB], tl.float32)
    for start in tl.static_range(0, WIDTH, BLOCK_WIDTH):
        dim = start + tl.arange(0, BLOCK_WIDTH)
        state = tl.load(state_row + dim).to(tl.float32) * inv_rms
        state = state * tl.load(NORM + dim).to(tl.float32)
        weight = tl.load(STATE_T + dim[:, None] * EMB + column[None, :])
        acc += tl.sum(state[:, None] * weight, axis=0)
    acc += tl.load(HIDDEN_PART + request * hidden_part_stride + column)
    tl.store(OUT + request * EMB + column, acc * tl.sigmoid(acc))


@triton.jit
def _pool_max_kernel(
    LOGITS,
    request_stride,
    step_stride,
    OUT,
    VOCAB: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = column < VOCAB
    row = LOGITS + request * request_stride + column
    top = tl.load(row, mask=inside)
    for step in tl.static_range(1, STEPS):
        logits = tl.load(row + step * step_stride, mask=inside)
        top = tl.maximum(top, logits, propagate_nan=tl.PropagateNan.ALL)
    tl.store(OUT + request * VOCAB + column, top, mask=inside)


@triton.jit
def _draw_kernel(
    SCORES,
    NOISE,
    TEMPERATURES,
    GREEDY,
    CANDIDATES,
    TOKENS,
    Q,
    POOL: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
):
    request = tl.program_id(0)
    index = tl.arange(0, BLOCK_POOL)
    inside = index < POOL
    row = request * POOL + index
    scores = tl.load(SCORES + row, mask=inside, other=_NEG_INF)
    greedy = tl.load(GREEDY + request) != 0
    top = tl.argmax(scores, axis=0, tie_break_left=True)
    scaled = scores / tl.load(TEMPERATURES + request)
    weights = tl.exp(scaled - tl.max(scaled, axis=0))
    q = weights / tl.sum(weights, axis=0)
    # Exponential race: argmax(q / E), E ~ Exp(1), is an exact draw from q.
    noise = tl.maximum(tl.load(NOISE + row, mask=inside, other=1.0), _TINY)
    drawn = tl.argmax(q / noise, axis=0, tie_break_left=True)
    choice = tl.where(greedy, top, drawn)
    tl.store(TOKENS + request, tl.load(CANDIDATES + request * POOL + choice))
    point_mass = tl.where(index == top, 1.0, 0.0)
    tl.store(Q + row, tl.where(greedy, point_mass, q), mask=inside)


def corrector_pool_max(logits: torch.Tensor) -> torch.Tensor:
    """``logits.amax(dim=1)`` for [batch, steps, vocab] logits whose rows are
    contiguous: the scores a block's candidate pool is drawn by."""
    bs, steps, vocab = logits.shape
    assert logits.is_cuda and logits.stride(2) == 1 and steps > 0
    out = logits.new_empty((bs, vocab))
    if bs:
        # Arbitrary: 1024 columns x steps rows per program.
        block = min(1024, triton.next_power_of_2(vocab))
        _pool_max_kernel[(bs, triton.cdiv(vocab, block))](
            logits,
            logits.stride(0),
            logits.stride(1),
            out,
            VOCAB=vocab,
            STEPS=steps,
            BLOCK=block,
            num_warps=4,
        )
    return out


def _check(x: torch.Tensor, *, width: int, heads: int) -> int:
    assert x.is_cuda and x.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float32)
    head_dim = width // heads
    assert head_dim * heads == width, (width, heads)
    assert head_dim >= 2 and head_dim & (head_dim - 1) == 0, head_dim
    return head_dim


def corrector_history_attention(
    qkv: torch.Tensor, history: torch.Tensor, slot: int, *, heads: int
) -> torch.Tensor:
    """Attend a slot's two streams to the slots up to it, and record the slot.

    ``qkv`` is [batch, 2 streams, 3 * width]: a stream's query, then each head's
    key and value. ``history`` is [batch, heads, slots, 2 * head_dim], a slot's key
    then its value: slot ``slot`` is written from the token stream (0) and the
    slots before it are read. Returns softmax(query . key) @ value over slots
    ``0 .. slot``, [batch, 2, width]. The query carries the attention scale.
    """
    bs, streams, triple = qkv.shape
    width = triple // 3
    head_dim = _check(qkv, width=width, heads=heads)
    slots = history.shape[2]
    assert streams == 2 and triple == 3 * width and 0 <= slot < slots
    assert history.shape == (bs, heads, slots, 2 * head_dim)
    assert history.dtype == qkv.dtype and history.is_contiguous()
    out = qkv.new_empty((bs, 2, width))
    if bs:
        _history_attention_kernel[(bs * 2 * heads,)](
            qkv,
            history,
            out,
            slot,
            WIDTH=width,
            HEADS=heads,
            HEAD_DIM=head_dim,
            SLOTS=slots,
            BLOCK_SLOTS=triton.next_power_of_2(slots),
            num_warps=1,
        )
    return out


# Window rows per tile, whether a program serves both streams, warps per program.
_CONTEXT_TILE, _CONTEXT_PAIR, _CONTEXT_WARPS = 256, False, 4


def _context_attention(
    query,
    context,
    cos,
    sin,
    *,
    layer,
    heads,
    valid=None,
    ring=None,
    tile=None,
    pair=None,
    num_warps=None,
) -> torch.Tensor:
    bs, streams, width = query.shape
    head_dim = _check(query, width=width, heads=heads)
    half = head_dim // 2
    window = cos.shape[0]
    assert streams == 2 and context.shape[-1] == 2 * width
    assert context.dtype == query.dtype and context.is_contiguous()
    assert cos.shape == sin.shape == (window, half)
    assert cos.dtype == sin.dtype == torch.float32
    assert cos.is_contiguous() and sin.is_contiguous()
    pair = _CONTEXT_PAIR if pair is None else pair
    out = torch.empty_like(query)
    if bs:
        slots, anchors, min_pos, sink_rows = ring if ring is not None else (None,) * 4
        _context_attention_kernel[(bs * heads * (1 if pair else 2),)](
            query,
            context,
            valid,
            slots,
            anchors,
            min_pos,
            cos,
            sin,
            out,
            WIDTH=width,
            HEADS=heads,
            HALF=half,
            WINDOW=window,
            LAYERS=context.shape[-2],
            LAYER=layer,
            # Measured on B300 at window 256: one 256-row pass at 4 warps takes
            # 3.9 us, 32-row tiles 10.3 us; re-tune when the kernel changes.
            TILE=min(tile or _CONTEXT_TILE, triton.next_power_of_2(window)),
            PAIR=pair,
            RING=ring is not None,
            SINK_ROWS=sink_rows or 0,
            num_warps=num_warps or _CONTEXT_WARPS,
        )
    return out


def corrector_context_attention(
    query: torch.Tensor,
    context: torch.Tensor,
    valid: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    layer: int,
    heads: int,
) -> torch.Tensor:
    """Attend a slot's two streams to a window of projected target context.

    ``query`` is [batch, 2 streams, width] and carries the attention scale.
    ``context`` is [batch, window, layers, 2 * width], a window row's key then its
    value for each layer; ``valid`` [batch, window] (uint8) marks the rows a request
    has. ``cos`` / ``sin`` are [window, head_dim / 2] FP32: the rotation applied to
    a row's key, pairing dimension ``d`` with ``d + head_dim / 2``. Returns
    softmax(query . rotated key) @ value over the valid rows, [batch, 2, width],
    and zeros for a request with none.
    """
    bs, window = valid.shape
    assert context.shape[:2] == (bs, window) and valid.dtype == torch.uint8
    assert valid.is_contiguous() and query.shape[0] == bs
    return _context_attention(
        query, context, cos, sin, layer=layer, heads=heads, valid=valid
    )


def corrector_ring_context_attention(
    query: torch.Tensor,
    ring_kv: torch.Tensor,
    req_pool_indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    min_pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sink_rows: int,
    layer: int,
    heads: int,
) -> torch.Tensor:
    """``corrector_context_attention`` reading each window in place from a ring.

    ``ring_kv`` is [rows, layers, 2 * width] with ``window`` rows per request after
    ``sink_rows``: request ``r``'s window row ``w`` is position
    ``p = anchor_positions[r] - window + w`` at row
    ``sink_rows + req_pool_indices[r] * window + p % window``, and is valid when
    ``p >= min_pos[req_pool_indices[r]]``. Equals the dense call on the window
    gathered from the ring with its invalid rows zeroed.
    """
    bs = query.shape[0]
    assert ring_kv.dim() == 3
    assert req_pool_indices.shape == anchor_positions.shape == (bs,)
    for tensor in (req_pool_indices, anchor_positions, min_pos):
        assert tensor.is_contiguous() and tensor.dtype in (torch.int32, torch.int64)
    return _context_attention(
        query,
        ring_kv,
        cos,
        sin,
        layer=layer,
        heads=heads,
        ring=(req_pool_indices, anchor_positions, min_pos, sink_rows),
    )


def corrector_candidate_scores(
    base: torch.Tensor,
    candidate_ids: torch.Tensor,
    table: torch.Tensor,
    correction: torch.Tensor,
) -> torch.Tensor:
    """``base + table[candidate_ids] @ correction``, [batch, K] FP32.

    ``base`` is [batch, K] (rows may be strided), ``candidate_ids`` [batch, K]
    int64, ``table`` [vocab, dim] BF16 or FP32 (rows may be strided) and
    ``correction`` [batch, dim] FP32. The table rows are read in place and widened
    to FP32, so no [batch, K, dim] copy of them is made.
    """
    bs, pool = candidate_ids.shape
    dim = table.shape[1]
    assert correction.shape == (bs, dim) and correction.dtype == torch.float32
    assert correction.is_contiguous() and candidate_ids.is_contiguous()
    assert base.shape == (bs, pool) and base.stride(1) == 1
    assert table.stride(1) == 1 and table.dtype in (torch.bfloat16, torch.float32)
    assert dim & (dim - 1) == 0, dim
    out = torch.empty((bs, pool), dtype=torch.float32, device=base.device)
    if bs:
        # Arbitrary: 32 rows of 256 BF16 per program keep ~7k programs at batch 116.
        block = min(32, triton.next_power_of_2(pool))
        _candidate_scores_kernel[(bs, triton.cdiv(pool, block))](
            base,
            base.stride(0),
            candidate_ids,
            table,
            table.stride(0),
            correction,
            out,
            POOL=pool,
            DIM=dim,
            BLOCK_POOL=block,
            num_warps=4,
        )
    return out


def corrector_output_head(
    streams: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    state_t: torch.Tensor,
    hidden_part: torch.Tensor,
) -> torch.Tensor:
    """``silu(hidden_part + rmsnorm(streams[:, 1]) @ state_t)``, [batch, emb] FP32.

    ``streams`` is [batch, 2, width] BF16 or FP32; the norm takes FP32 statistics
    and its FP32 weight product like ``rmsnorm_fp32``. ``state_t`` is
    [width, emb] FP32 and ``hidden_part`` [batch, emb] FP32 (rows may be strided).
    """
    bs, streams_per_row, width = streams.shape
    emb = state_t.shape[1]
    assert streams_per_row == 2 and streams.is_contiguous()
    assert state_t.shape == (width, emb) and state_t.is_contiguous()
    assert state_t.dtype == hidden_part.dtype == torch.float32
    assert hidden_part.shape == (bs, emb) and hidden_part.stride(1) == 1
    assert norm_weight.shape == (width,) and norm_weight.is_contiguous()
    assert width & (width - 1) == 0 and emb & (emb - 1) == 0, (width, emb)
    out = torch.empty((bs, emb), dtype=torch.float32, device=streams.device)
    if bs:
        block_emb = min(64, emb)
        _output_head_kernel[(bs, emb // block_emb)](
            streams,
            norm_weight,
            state_t,
            hidden_part,
            hidden_part.stride(0),
            out,
            eps,
            WIDTH=width,
            EMB=emb,
            BLOCK_EMB=block_emb,
            BLOCK_WIDTH=min(64, width),
            num_warps=4,
            # The norm then rounds as rmsnorm_fp32 does.
            enable_fp_fusion=False,
        )
    return out


def corrector_draw(
    scores: torch.Tensor,
    noise: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw each request's token from q = softmax(scores / T) over its candidates.

    ``scores`` and ``noise`` are [batch, K] FP32, ``noise`` holding Exp(1) draws;
    ``temperatures`` [batch] FP32 is positive; ``greedy_mask`` [batch] is uint8;
    ``candidate_ids`` [batch, K] is int64. A sampled row takes
    argmax(q / noise) and a greedy row argmax(scores), the first on a tie.
    Returns the tokens, [batch] int64, and each row's q, [batch, K] FP32: the
    point mass on its token for a greedy row.
    """
    bs, pool = scores.shape
    assert scores.is_cuda and 0 < pool <= DRAW_MAX_POOL
    assert scores.dtype == noise.dtype == temperatures.dtype == torch.float32
    assert greedy_mask.dtype == torch.uint8 and candidate_ids.dtype == torch.int64
    assert noise.shape == candidate_ids.shape == (bs, pool)
    assert temperatures.shape == greedy_mask.shape == (bs,)
    for tensor in (scores, noise, temperatures, greedy_mask, candidate_ids):
        assert tensor.is_contiguous()
    tokens = candidate_ids.new_empty((bs,))
    q = torch.empty_like(scores)
    if bs:
        _draw_kernel[(bs,)](
            scores,
            noise,
            temperatures,
            greedy_mask,
            candidate_ids,
            tokens,
            q,
            POOL=pool,
            BLOCK_POOL=triton.next_power_of_2(pool),
            num_warps=8,
        )
    return tokens, q
