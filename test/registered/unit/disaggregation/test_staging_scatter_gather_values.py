"""Per-value correctness of the heterogeneous-TP staging gather/scatter path.

The staging buffer regroups KV heads across a prefill->decode TP-size change.
End-to-end accuracy (gsm8k) cannot catch a head-mapping or byte-packing error,
because a wrong KV often still yields a plausible token. These tests drive the
production head-slice math (``compute_head_slice_params``), the gather/scatter
primitives, and the real ``_scatter_staging_to_kv_torch`` on synthetic KV
seeded so that every cell encodes its global (layer, kv-kind, global-head,
token) identity, then assert the decode pool holds exactly the reference bytes.

Both TP directions are covered (prefill TP > decode TP, i.e. head fan-in, and
prefill TP < decode TP, i.e. head fan-out), plus the decode-side cached-prefix
scatter offset: a scatter into suffix pages must leave the radix-shared prefix
pages untouched.

The tests run on CPU with the torch staging path (no CUDA, no server); the
kernels are device- and dtype-agnostic, so float32 identity values compare
bit-exact with ``torch.equal``.
"""

import unittest

import torch

from sglang.srt.disaggregation.common import staging_buffer as sb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEAD_DIM = 4
NUM_LAYERS = 2
PAGE_SIZE = 1


def _identity(layer: int, is_v: int, global_head: int, token: int, d: int) -> float:
    """A value that uniquely encodes where a KV cell belongs, so a mis-mapped
    head/token/layer is a different number rather than a coincidental match."""
    return float(
        ((((layer * 2 + is_v) * 1024 + global_head) * 4096 + token) * HEAD_DIM) + d
    )


def _make_pool(num_pool_tokens: int, heads_per_rank: int, rank: int, is_v: int):
    """One rank's single-layer-agnostic pool [pool, heads_per_rank, head_dim],
    seeded so local head h == global head (rank * heads_per_rank + h)."""
    pools = []
    for layer in range(NUM_LAYERS):
        t = torch.empty(num_pool_tokens, heads_per_rank, HEAD_DIM)
        for h in range(heads_per_rank):
            gh = rank * heads_per_rank + h
            for tok in range(num_pool_tokens):
                for d in range(HEAD_DIM):
                    t[tok, h, d] = _identity(layer, is_v, gh, tok, d)
        pools.append(t)
    return pools


def _round_trip(prefill_tp, decode_tp, total_kv_heads, num_pages, dst_tp_rank):
    """Gather every writer's head slice into a staging byte buffer exactly as
    ``_gather_all_layers_torch`` lays it out, then scatter with the production
    torch scatter into a fresh decode pool. Returns (decode_k, decode_v)."""
    src_hpr = max(1, total_kv_heads // prefill_tp)
    dst_hpr = max(1, total_kv_heads // decode_tp)
    num_tokens = num_pages * PAGE_SIZE
    if prefill_tp > decode_tp:
        num_writers = prefill_tp // decode_tp
    else:
        num_writers = 1

    # Decode pool for this rank: [pool, dst_hpr, head_dim] per layer, K and V.
    decode_k = [torch.zeros(num_tokens, dst_hpr, HEAD_DIM) for _ in range(NUM_LAYERS)]
    decode_v = [torch.zeros(num_tokens, dst_hpr, HEAD_DIM) for _ in range(NUM_LAYERS)]
    page_idx = torch.arange(num_pages, dtype=torch.int64)

    dtype_size = decode_k[0].element_size()
    per_writer_bytes = num_tokens * src_hpr * HEAD_DIM * dtype_size * NUM_LAYERS * 2
    staging = torch.zeros(num_writers * per_writer_bytes, dtype=torch.uint8)

    for w in range(num_writers):
        src_head_start, nh, _, _ = sb.compute_head_slice_params(
            prefill_tp, decode_tp, w, dst_tp_rank, total_kv_heads
        )
        # Which prefill rank feeds writer slot w of this decode rank.
        if prefill_tp > decode_tp:
            src_rank = dst_tp_rank * num_writers + w
        else:
            src_rank = dst_tp_rank // (decode_tp // prefill_tp)
        k_pool = _make_pool(num_tokens, src_hpr, src_rank, is_v=0)
        v_pool = _make_pool(num_tokens, src_hpr, src_rank, is_v=1)

        gather_idx = page_idx.view(-1, 1, 1).expand(num_tokens, nh, HEAD_DIM)
        per_layer_bytes = num_tokens * nh * HEAD_DIM * dtype_size
        offset = w * per_writer_bytes
        for pool in (k_pool, v_pool):
            for layer in range(NUM_LAYERS):
                dst = (
                    staging[offset : offset + per_layer_bytes]
                    .view(pool[layer].dtype)
                    .reshape(num_tokens, nh, HEAD_DIM)
                )
                sb.gather_kv_head_slices(
                    pool[layer], gather_idx, src_head_start, nh, dst
                )
                offset += per_layer_bytes

    sb._scatter_staging_to_kv_torch(
        staging,
        decode_k,
        decode_v,
        page_idx,
        PAGE_SIZE,
        prefill_tp,
        decode_tp,
        dst_tp_rank,
        total_kv_heads,
    )
    return decode_k, decode_v


def _expected_decode(decode_tp, total_kv_heads, num_pages, dst_tp_rank):
    dst_hpr = max(1, total_kv_heads // decode_tp)
    num_tokens = num_pages * PAGE_SIZE
    exp_k = [torch.empty(num_tokens, dst_hpr, HEAD_DIM) for _ in range(NUM_LAYERS)]
    exp_v = [torch.empty(num_tokens, dst_hpr, HEAD_DIM) for _ in range(NUM_LAYERS)]
    for layer in range(NUM_LAYERS):
        for h in range(dst_hpr):
            gh = dst_tp_rank * dst_hpr + h
            for tok in range(num_tokens):
                for d in range(HEAD_DIM):
                    exp_k[layer][tok, h, d] = _identity(layer, 0, gh, tok, d)
                    exp_v[layer][tok, h, d] = _identity(layer, 1, gh, tok, d)
    return exp_k, exp_v


class TestHeteroTPValueRoundTrip(CustomTestCase):
    def _check(self, prefill_tp, decode_tp, total_kv_heads, num_pages=3):
        for dst_tp_rank in range(decode_tp):
            dk, dv = _round_trip(
                prefill_tp, decode_tp, total_kv_heads, num_pages, dst_tp_rank
            )
            ek, ev = _expected_decode(decode_tp, total_kv_heads, num_pages, dst_tp_rank)
            for layer in range(NUM_LAYERS):
                self.assertTrue(
                    torch.equal(dk[layer], ek[layer]),
                    f"K mismatch tp{prefill_tp}->tp{decode_tp} "
                    f"dst_rank={dst_tp_rank} layer={layer}",
                )
                self.assertTrue(
                    torch.equal(dv[layer], ev[layer]),
                    f"V mismatch tp{prefill_tp}->tp{decode_tp} "
                    f"dst_rank={dst_tp_rank} layer={layer}",
                )

    def test_fan_in_tp4_to_tp2(self):
        self._check(prefill_tp=4, decode_tp=2, total_kv_heads=8)

    def test_fan_out_tp2_to_tp4(self):
        self._check(prefill_tp=2, decode_tp=4, total_kv_heads=8)

    def test_fan_in_tp2_to_tp1(self):
        self._check(prefill_tp=2, decode_tp=1, total_kv_heads=8)


class TestScatterHeadMapping(CustomTestCase):
    """Pin the exact head-index arithmetic of compute_head_slice_params so an
    off-by-one in the mapping fails deterministically."""

    def test_fan_in_tp4_to_tp2_mapping(self):
        # decode rank 0 receives its 4 heads from prefill ranks 0,1 (2 heads
        # each) landing at local [0,2) and [2,4).
        p = sb.compute_head_slice_params(4, 2, 0, 0, 8)
        self.assertEqual(p, (0, 2, 0, 2))
        p = sb.compute_head_slice_params(4, 2, 1, 0, 8)
        self.assertEqual(p, (0, 2, 2, 2))

    def test_fan_out_tp2_to_tp4_mapping(self):
        # decode rank 0 pulls heads [0,2) of prefill rank 0; decode rank 1
        # pulls heads [2,4) of the same prefill rank; both land at local 0.
        p = sb.compute_head_slice_params(2, 4, 0, 0, 8)
        self.assertEqual(p, (0, 2, 0, 2))
        p = sb.compute_head_slice_params(2, 4, 0, 1, 8)
        self.assertEqual(p, (2, 2, 0, 2))


class TestDecodePrefixScatterOffset(CustomTestCase):
    """A scatter addressed at suffix pages must not touch the radix-shared
    prefix pages (the decode-side cached-prefix offset path)."""

    def test_prefix_pages_untouched(self):
        total_kv_heads, decode_tp, dst_tp_rank = 8, 2, 0
        dst_hpr = total_kv_heads // decode_tp
        pool_pages, prefix_pages, suffix_pages = 6, 2, 3
        num_tokens = suffix_pages * PAGE_SIZE

        decode_k = [
            torch.full((pool_pages, dst_hpr, HEAD_DIM), -1.0) for _ in range(NUM_LAYERS)
        ]
        decode_v = [
            torch.full((pool_pages, dst_hpr, HEAD_DIM), -1.0) for _ in range(NUM_LAYERS)
        ]
        # Suffix pages start after the cached prefix (as _scatter_region does
        # via token_start = cache_protected_len + page_start * page_size).
        page_idx = torch.arange(
            prefix_pages, prefix_pages + suffix_pages, dtype=torch.int64
        )

        prefill_tp = 4
        num_writers = prefill_tp // decode_tp
        src_hpr = total_kv_heads // prefill_tp
        dtype_size = decode_k[0].element_size()
        per_writer_bytes = num_tokens * src_hpr * HEAD_DIM * dtype_size * NUM_LAYERS * 2
        staging = (
            torch.arange(num_writers * per_writer_bytes, dtype=torch.float32)
            .view(torch.uint8)[: num_writers * per_writer_bytes]
            .clone()
        )

        sb._scatter_staging_to_kv_torch(
            staging,
            decode_k,
            decode_v,
            page_idx,
            PAGE_SIZE,
            prefill_tp,
            decode_tp,
            dst_tp_rank,
            total_kv_heads,
        )

        for layer in range(NUM_LAYERS):
            # Prefix pages [0, prefix_pages) must be untouched sentinel.
            self.assertTrue(
                torch.all(decode_k[layer][:prefix_pages] == -1.0),
                f"prefix K clobbered layer={layer}",
            )
            self.assertTrue(
                torch.all(decode_v[layer][:prefix_pages] == -1.0),
                f"prefix V clobbered layer={layer}",
            )
            # Suffix pages must have been written (no sentinel left).
            self.assertFalse(
                torch.any(
                    decode_k[layer][prefix_pages : prefix_pages + suffix_pages] == -1.0
                )
            )


if __name__ == "__main__":
    unittest.main()
