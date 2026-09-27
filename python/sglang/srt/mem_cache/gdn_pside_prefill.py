"""P-only block replay from Opus c290b269e52 with PD shape/ownership checks."""
import json
import os

FLAG = 'SGLANG_GDN_PSIDE_GRAPH'
CHECK = 'SGLANG_GDN_PSIDE_GRAPH_CHECK'
EXACT = (256, 8192, 16384)


def bucket(tokens):
    """P chunks are bounded at 16K; short buckets retain the native output tile."""
    if not 1 <= tokens <= 16384:
        return None
    if tokens in EXACT:
        return tokens
    if tokens <= 16:
        return 16
    if tokens <= 32:
        return 32
    step = 512 if tokens <= 8192 else 4096
    return ((tokens + step - 1) // step) * step


def same(left, right):
    import torch
    if left is None or right is None:
        return left is right
    return (left.shape == right.shape and left.dtype == right.dtype and
            torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def run(backend, layer, query, key, value, a, b, state, rows, cu):
    """Return None only for a shape outside the explicitly admitted P path."""
    import torch
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    if os.environ.get(FLAG) != '1':
        return None
    role = backend._model_runner.server_args.disaggregation_mode
    if role != 'prefill':
        raise RuntimeError('SGLANG_GDN_PSIDE_GRAPH is restricted to P workers')
    n = query.shape[1]
    capacity = bucket(n)
    if (rows.numel() != 1 or cu.numel() != 2 or capacity is None
            or not isinstance(backend.kernel_dispatcher.extend_kernel, TritonGDNKernel)
            or torch.cuda.is_current_stream_capturing()):
        return None
    tensors = dict(q=query, k=key, v=value, a=a, b=b, log=layer.A_log,
                   bias=layer.dt_bias, state=state, rows=rows)

    def eager(t):
        gate, beta = fused_gdn_gating(t['log'], t['a'], t['b'], t['bias'])
        return backend.kernel_dispatcher.extend(q=t['q'], k=t['k'], v=t['v'],
            g=gate, beta=beta, ssm_states=t['state'], cache_indices=t['rows'],
            query_start_loc=t['cu'])

    check = os.environ.get(CHECK) == '1'
    if check:
        reference = dict(tensors, state=state.clone(), cu=cu)
        expected = eager(reference)
    if n in EXACT:
        from .gdn_prefill_block_graph import PrefillBlockGraph
        graph = getattr(backend, '_pside_exact_graph', None)
        if graph is None:
            graph = backend._pside_exact_graph = PrefillBlockGraph(max_entries=3)
        actual = graph.run(dict(tensors, cu=cu), eager)
    else:
        from .gdn_prefill_block_pad import PaddedBlockGraph, chunk_padded
        graph = getattr(backend, '_pside_padded_graph', None)
        if graph is None:
            graph = backend._pside_padded_graph = PaddedBlockGraph()
        def padded(t):
            gate, beta = fused_gdn_gating(t['log'], t['a'], t['b'], t['bias'])
            return chunk_padded(t['q'], t['k'], t['v'], gate, beta,
                                t['state'], t['rows'], t['cu'], t['real_end'])
        actual = graph.run(tensors, capacity, padded)
    if check:
        end = reference['state'] if expected[1] is None else expected[1]
        h = actual[2]
        if h is not None and expected[2] is not None:
            h = h[:, :expected[2].shape[1]]
        verdict = dict(output=same(actual[0], expected[0]),
                       state=same(actual[1], end), checkpoint=same(h, expected[2]))
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        print('PSIDE_BLOCK_CHECK '+json.dumps(dict(rank=get_tensor_model_parallel_rank(),
              layer=layer.layer_id, tokens=n, bucket=capacity, **verdict)), flush=True)
        if not all(verdict.values()):
            raise RuntimeError('P prefill graph differs from frozen eager: '+str(verdict))
    seen = backend.__dict__.setdefault('_pside_block_receipts', set())
    identity = (layer.layer_id, n, capacity)
    if identity not in seen:
        seen.add(identity)
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        print('PSIDE_BLOCK_GRAPH '+json.dumps(dict(rank=get_tensor_model_parallel_rank(),
              layer=layer.layer_id, tokens=n, bucket=capacity, stats=dict(graph.stats))), flush=True)
    return actual
