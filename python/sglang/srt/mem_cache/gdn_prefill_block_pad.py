"""#ssmoff-opus #779: length-bucketed GDN prefill block graphs for a singleton request.

A prefill of L tokens replays the graph captured for a bucket Lb >= L. Rows [L, Lb) of q/k/v are zero and the gating
inputs a/b are -inf there, so the gating kernel yields g = -0.0 (decay exp(g) = 1) and beta = 0: every padding token
leaves the recurrent state unchanged, and the padded outputs are dropped. One copy kernel rebinds the real rows, the
padding and the layer parameters before each replay (the graph is shared by all layers). The final state and the
chunk states returned for rows < L are compared bytewise with the eager unpadded call (test_opus_prefill_block_pad).
"""
from collections import OrderedDict
import os

import torch
import triton
import triton.language as tl

EXACT = (256, 8192)  # served by the unpadded PrefillBlockGraph
MAX_ENTRIES = int(os.environ.get("SGLANG_GDN_PREFILL_BLOCK_PAD_ENTRIES", "24"))


def bucket(tokens):
    """Graph length for a singleton prefill of `tokens` tokens, or None for the eager path."""
    if tokens in EXACT or tokens <= 0:
        return None
    if tokens <= 8192:
        return -(-tokens // 512) * 512
    if tokens <= 32768:
        return -(-tokens // 4096) * 4096
    return None


@triton.jit
def _bind_padded(sources, destinations, valid_rows, COUNTS: tl.constexpr, WIDTHS: tl.constexpr,
                 STRIDES: tl.constexpr, TOKEN: tl.constexpr, PADS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    begin = 0
    for i in tl.static_range(len(COUNTS)):
        blocks = tl.cdiv(COUNTS[i], BLOCK)
        if pid >= begin and pid < begin + blocks:
            offsets = (pid - begin) * BLOCK + tl.arange(0, BLOCK)
            inside = offsets < COUNTS[i]
            row = offsets // WIDTHS[i]
            col = offsets % WIDTHS[i]
            if TOKEN[i]:
                real = inside & (row < valid_rows)
            else:
                real = inside
            values = tl.load(sources[i] + row * STRIDES[i] + col, mask=real, other=0)
            pad = tl.full([BLOCK], PADS[i], dtype=values.dtype)
            tl.store(destinations[i] + offsets, tl.where(real, values, pad), mask=inside)
        begin += blocks


# per tensor: (token-indexed?, padding value)
_LAYOUT = dict(q=(True, 0.0), k=(True, 0.0), v=(True, 0.0), a=(True, float("-inf")), b=(True, float("-inf")),
               log=(False, 0.0), bias=(False, 0.0), state=(False, 0.0), rows=(False, 0))


def _row_view(x):
    """(rows, width, row stride) of a token tensor [1, T, H, D] / [T, H] whose rows are contiguous inside."""
    if x.ndim == 4:
        assert x.shape[0] == 1 and x.stride(3) == 1 and x.stride(2) == x.shape[3]
        return x.shape[1], x.shape[2] * x.shape[3], x.stride(1)
    assert x.ndim == 2 and x.stride(1) == 1
    return x.shape[0], x.shape[1], x.stride(0)


def bind_padded(buffers, tensors, valid):
    names = tuple(tensors)
    counts, widths, strides, token, pads = [], [], [], [], []
    for n in names:
        is_token, pad = _LAYOUT[n]
        dst = buffers[n]
        counts.append(dst.numel())
        if is_token:
            _, width, stride = _row_view(tensors[n])
        else:
            assert tensors[n].is_contiguous() and tensors[n].numel() == dst.numel()
            width, stride = dst.numel(), 0
        widths.append(width)
        strides.append(stride)
        token.append(is_token)
        pads.append(pad)
    _bind_padded[(sum(triton.cdiv(c, 1024) for c in counts),)](
        tuple(tensors[n] for n in names), tuple(buffers[n] for n in names), valid,
        tuple(counts), tuple(widths), tuple(strides), tuple(token), tuple(pads), 1024, num_warps=4)


class PaddedBlockGraph:
    def __init__(self):
        self.entries = OrderedDict()
        self.stats = dict(captured=0, replayed=0)

    def run(self, tensors, padded, evaluate):
        """tensors: q/k/v [1, L, H, D], a/b [L, HV], log/bias [HV], state [1, HV, V, K], rows [1] (int32)."""
        tokens = tensors['q'].shape[1]
        assert tokens <= padded
        # token tensors enter by their per-token shape; the token count only selects the bucket
        key = (padded,) + tuple((n, tuple(x.shape[2:] if x.ndim == 4 else x.shape[1:]) if _LAYOUT[n][0]
                                 else tuple(x.shape), x.dtype, x.device) for n, x in tensors.items())
        key += (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        if key not in self.entries:
            buffers = {}
            for n, x in tensors.items():
                if _LAYOUT[n][0]:
                    shape = (1, padded) + tuple(x.shape[2:]) if x.ndim == 4 else (padded,) + tuple(x.shape[1:])
                    buffers[n] = torch.empty(shape, dtype=x.dtype, device=x.device)
                else:
                    buffers[n] = torch.empty_like(x, memory_format=torch.contiguous_format)
            buffers['cu'] = torch.tensor([0, padded], dtype=torch.int32, device=tensors['q'].device)
            from sglang.kernels.ops.attention.fla.index import (
                prepare_chunk_indices, prepare_chunk_offsets, prepare_lens,
            )
            cu = buffers['cu']
            # own the FLA index allocations for the graph's lifetime (as PrefillBlockGraph, 127d4cb7b9a)
            pinned = (prepare_lens(cu), prepare_chunk_indices(cu, 64), prepare_chunk_offsets(cu, 64))
            bind_padded(buffers, tensors, tokens)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    bind_padded(buffers, tensors, tokens)
                    evaluate(buffers)
            torch.cuda.current_stream().wait_stream(stream)
            bind_padded(buffers, tensors, tokens)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = evaluate(buffers)
            self.stats['captured'] += 1
            self.entries[key] = (buffers, graph, outputs, pinned)
            while len(self.entries) > MAX_ENTRIES:
                self.entries.popitem(last=False)
        buffers, graph, outputs, _ = self.entries[key]
        self.entries.move_to_end(key)
        bind_padded(buffers, tensors, tokens)
        graph.replay()
        self.stats['replayed'] += 1
        output, last, h = outputs
        state = buffers['state'] if last is None else last
        return output[:, :tokens].clone(), state.clone(), h
