"""AGG-only M5 entry. Preserve P/D ownership and the native commit path."""
import json
import os

FLAG = 'SGLANG_GDN_PREFILL_BLOCK_GRAPH'


def eligible(backend, query, rows, cu, metadata, *, capturing):
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    role = backend._model_runner.server_args.disaggregation_mode
    if role != 'null':
        return False
    if (query.ndim != 4 or query.shape[0] != 1 or rows.numel() != 1
            or cu.numel() != 2 or capturing
            or not isinstance(backend.kernel_dispatcher.extend_kernel, TritonGDNKernel)):
        return False
    bucketed = os.environ.get('SGLANG_GDN_PREFILL_BLOCK_BUCKETS', '0') == '1'
    limit = 8192 if os.environ.get('SGLANG_GDN_PREFILL_BLOCK_SHORT_ONLY', '0') == '1' else 32768
    tokens = query.shape[1]
    if not (1 <= tokens <= limit if bucketed else tokens in (256,8192)):
        return False
    # The current Triton kernel returns its native chunk checkpoints; alternate
    # checkpoint layouts must use the dispatcher path that owns those contracts.
    if backend.kernel_dispatcher.extend_uses_state_checkpoints:
        return False
    if metadata.track_ssm_recompute_dst is not None and metadata.track_ssm_recompute_dst.numel():
        return False
    return True


def run(backend, layer, query, key, value, a, b, state, rows, cu, metadata, *, qk_prepared=False):
    import torch
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    if os.environ.get(FLAG) != '1':
        return None
    if not eligible(backend,query,rows,cu,metadata,
                    capturing=query.is_cuda and torch.cuda.is_current_stream_capturing()):
        return None
    if any(x.ndim != 4 or x.shape[0] != 1 or x.stride(2) != x.shape[3]*x.stride(3)
           for x in (query,key,value,state)):
        return None
    from .gdn_agg_prefill_block_graph import PrefillBlockGraph, check_result
    bucketed = os.environ.get('SGLANG_GDN_PREFILL_BLOCK_BUCKETS', '0') == '1'
    graph = getattr(backend, '_agg_prefill_block_graph', None)
    if graph is None:
        graph = backend._agg_prefill_block_graph = PrefillBlockGraph(bucketed=bucketed, qk_prepared=qk_prepared)
    if graph.qk_prepared != qk_prepared:
        raise RuntimeError('AGG block graph normalization ownership changed after initialization')
    if graph.bucketed != bucketed:
        raise RuntimeError('AGG block graph bucket policy changed after initialization')
    tensors=dict(q=query,k=key,v=value,a=a,b=b,log=layer.A_log,bias=layer.dt_bias,
                 state=state,rows=rows,cu=cu)
    def evaluate(t):
        gate,beta=fused_gdn_gating(t['log'],t['a'],t['b'],t['bias'])
        return backend.kernel_dispatcher.extend(q=t['q'],k=t['k'],v=t['v'],g=gate,beta=beta,
            ssm_states=t['state'],cache_indices=t['rows'],query_start_loc=t['cu'],
            factored_qk_ready=qk_prepared)
    checked=os.environ.get('SGLANG_GDN_PREFILL_BLOCK_GRAPH_CHECK','0')=='1'
    if checked:
        reference=dict(tensors,state=state.clone())
        expected=evaluate(reference)
    actual=graph.run(tensors,evaluate)
    verdict=check_result(actual,expected,reference['state']) if checked else None
    # First use of each real shape/layer on each rank is explicit evidence that
    # the AGG entry ran; a configured flag alone is never an execution receipt.
    seen=backend.__dict__.setdefault('_agg_block_receipts',set())
    identity=(layer.layer_id,query.shape[1])
    if identity not in seen or checked:
        seen.add(identity)
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        print('AGG_BLOCK_GRAPH '+json.dumps(dict(rank=get_tensor_model_parallel_rank(),
            layer=layer.layer_id,tokens=query.shape[1],bucketed=bucketed,
            stats=dict(graph.stats),checked=verdict)),flush=True)
    return actual
