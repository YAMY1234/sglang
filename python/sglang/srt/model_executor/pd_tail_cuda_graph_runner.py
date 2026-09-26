"""Default-off, isolated B1 full-model tail graph for dense PD prefill.

The graph consumes the existing factor pool after the all-layer prefix commit.
It owns its input buffers, attention metadata, and allocation arena because the
outer extend's hidden/HC outputs remain live through this nested forward.
"""
import copy

import torch

GRAPH_LIMIT_BYTES = 8 << 30
METADATA_LIMIT_BYTES = 2 << 30
FORWARD_RESERVE_BYTES = 4 << 30
HEADROOM_BYTES = 2 << 30


def eligible(batch):
    return (batch.batch_size == 1 and batch.forward_mode.is_decode()
            and batch.input_ids.numel() == 1 and batch.spec_info is None
            and getattr(batch, 'input_embeds', None) is None
            and getattr(batch, 'replace_embeds', None) is None)


def private_stream():
    """A non-blocking CUDA stream that no torch stream pool can hand out.

    PyTorch keys the cuBLAS/cuBLASLt workspace by (handle, stream), and every
    CUDAGraph clears its capture stream's workspace entry when it is destroyed
    (``CUDAGraph::reset`` -> ``clearCublasWorkspacesForStream``). The tail graph
    bakes that workspace. On a pool stream, any other graph captured on the same
    stream (``torch.cuda.Stream()`` recycles 32 pool streams) frees it when that
    graph is destroyed, and the next ``empty_cache`` unmaps it under the tail.
    The stream is never destroyed, so no other graph can clear that workspace.
    """
    import ctypes
    driver = ctypes.CDLL('libcuda.so.1')
    handle = ctypes.c_void_p()
    status = driver.cuStreamCreate(ctypes.byref(handle), ctypes.c_uint(1))  # CU_STREAM_NON_BLOCKING
    if status != 0 or not handle.value:
        raise RuntimeError(f'PD tail private capture stream creation failed: CUresult {status}')
    return torch.cuda.ExternalStream(handle.value, device=torch.cuda.current_device())


def isolated_runner(runner, backend):
    from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
    isolated = copy.copy(runner)
    isolated.attn_backend = backend
    isolated.capture_tail_hooks = []  # The outer model retains logits/scoring ownership.
    # P disables ordinary decode graphs, so its shared output is None. The
    # native decode input builder still requires a logits buffer even though
    # this graph returns only hidden/HC. Keep this B1 buffer private as well.
    isolated.graph_shared_output = GraphSharedOutput(device=runner.device, max_rows=1)
    return isolated


def memory_plan(runner, *, free_bytes, total_bytes):
    from sglang.srt.mem_cache.gdn_stock_dense_commit import scratch_bound_bytes
    pool = runner.req_to_token_pool.factored_gdn_pool
    state = getattr(runner.req_to_token_pool, 'pd_stock_dense_checkpoints', None)
    dense = (len(pool.layer_ids)*(pool.size+1)*pool.hv*pool.v*pool.k*2
             +(pool.size+1)*8) if state is None else 0
    config = runner.model_config.hf_text_config
    tokens = max(runner.server_args.max_prefill_tokens or 0,
                 runner.server_args.chunked_prefill_size or 0)
    split_io = 4*tokens*config.hidden_size*(1+config.hc_count)+tokens*64
    scratch = scratch_bound_bytes(len(pool.layer_ids),pool.hv,pool.v,pool.k,pool.cfg)
    reserved = dense+split_io+scratch+GRAPH_LIMIT_BYTES+METADATA_LIMIT_BYTES+FORWARD_RESERVE_BYTES
    result = dict(device_total_bytes=total_bytes,free_bytes=free_bytes,
        future_dense_bytes=dense,split_input_output_bound_bytes=split_io,
        scratch_bound_bytes=scratch,graph_limit_bytes=GRAPH_LIMIT_BYTES,
        metadata_limit_bytes=METADATA_LIMIT_BYTES,forward_peak_reserve_bytes=FORWARD_RESERVE_BYTES,
        transfer_bytes_already_allocated=4 << 30,rebuild_source_bytes=0,
        projected_free_bytes=free_bytes-reserved,headroom_bytes=HEADROOM_BYTES)
    if tokens <= 0 or result['projected_free_bytes'] < HEADROOM_BYTES:
        raise RuntimeError('PD full tail graph memory budget: '+str(result))
    return result


def tail_runner_type(body):
    """Build the native tail runner class, independently of CUDA allocation.

    Keeping the type factory separate lets CPU integration tests construct the
    real runner, buffers and metadata while doubling only CUDA capture calls.
    """
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.layers.communicator import get_attn_tp_context

    forward = body.forward
    class TailRunner(DecodeCudaGraphRunner):
        def capture(self):
            if (self.enable_pdmux or self.attention_graph_variants is not None
                    or self.require_gathered_buffer or self.pp_size != 1
                    or self._metadata_glue is not None):
                raise ValueError('PD full tail graph has unsupported capture variants')
            self.tail_graph_pool = torch.cuda.graph_pool_handle()
            self.backend = FullCudaGraphBackend(self,graph_pool=self.tail_graph_pool,
                                               reuse_output_buffer=False)
            self.enable_profile_cuda_graph = False
            return super().capture()

        def _capture_one_stream(self, stream_idx=None):
            if stream_idx is not None:
                raise ValueError('PD full tail graph has one serial capture stream')
            self.capture_one_shape(1,self.forward_tail)

        def forward_tail(self,input_ids,positions,forward_batch):
            with get_attn_tp_context().maybe_input_scattered(forward_batch):
                hidden = forward(input_ids,positions,forward_batch)
            values = dict(hidden=hidden)
            if body.last_hc_hidden_states is not None:
                values['hc'] = body.last_hc_hidden_states
            return PPProxyTensors(values)

        def execute_tail(self,batch):
            from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
            if not eligible(batch) or not self.can_run_graph(batch):
                raise ValueError('PD full tail graph requires an eligible B1 decode batch')
            # The outer prefix runner owns a different metadata instance.
            current = copy.copy(batch)
            current.forward_metadata_ready = False
            with forward_context(ForwardContext(attn_backend=self.attn_backend)):
                output = self.execute(current)
            body.last_hc_hidden_states = output.tensors.get('hc')
            self.replays += 1
            return output.tensors['hidden']

    return TailRunner


def make_runner(runner):
    from sglang.srt.runtime_context import get_schedule

    if (runner.device != 'cuda' or not runner.spec_algorithm.is_none()
            or not get_schedule().disable_overlap_schedule or runner.lora_manager is not None):
        raise ValueError('PD full tail graph requires CUDA P without overlap/MTP/LoRA')
    from sglang.srt.disaggregation.flashnext_staging import _RESERVES
    staging = _RESERVES.get(torch.cuda.current_device())
    if staging is None or sum(t.numel()*t.element_size() for t in staging.buffers) != 4 << 30:
        raise ValueError('PD full tail graph requires the already allocated 4 GiB transfer buffers')
    body = runner.model.model.model
    from sglang.srt.model_executor.pd_tail_comm_guard import TailCommunicationLease
    communication = TailCommunicationLease()
    retained_hc = getattr(body,'last_hc_hidden_states',None)
    free,total = torch.cuda.mem_get_info()
    plan = memory_plan(runner,free_bytes=free,total_bytes=total)
    before = torch.cuda.memory_reserved()
    backend = runner._get_attention_backend(init_new_workspace=True)
    isolated = isolated_runner(runner, backend)
    metadata_bytes = torch.cuda.memory_reserved()-before
    if metadata_bytes > METADATA_LIMIT_BYTES:
        raise RuntimeError('PD tail metadata exceeds its predeclared memory budget')

    TailRunner = tail_runner_type(body)

    from sglang.srt.distributed.device_communicators import pynccl_allocator
    previous_pool = pynccl_allocator._graph_pool_id
    stream = private_stream()
    try:
        graph = TailRunner(isolated,attn_backend=backend,capture_bs_override=[1],
                           share_input_buffers=False,capture_stream=stream)
    finally:
        pynccl_allocator.set_graph_pool_id(previous_pool)
        body.last_hc_hidden_states = retained_hc
    torch.cuda.synchronize()
    communication.check()
    graph.communication = communication  # Retain the captured storage owners.
    if graph.stream is not stream:
        raise RuntimeError('PD full tail graph was not captured on its private stream')
    graph.private_stream = stream  # Stream-keyed workspaces stay this graph's.
    graph.replays = 0
    private_bytes = sum(s['total_size'] for s in torch.cuda.memory_snapshot()
                        if tuple(s['segment_pool_id']) == tuple(graph.tail_graph_pool))
    growth = max(0,torch.cuda.memory_reserved()-before)
    if private_bytes > GRAPH_LIMIT_BYTES or growth > GRAPH_LIMIT_BYTES+METADATA_LIMIT_BYTES:
        raise RuntimeError('PD full tail capture exceeds its predeclared memory budget')
    graph.memory_receipt = dict(plan,private_bytes=private_bytes,
        metadata_reserved_growth_bytes=metadata_bytes,total_reserved_growth_bytes=growth,
        graph_pool=list(graph.tail_graph_pool),input_buffers_shared=False,
        private_capture_stream=stream.cuda_stream,
        free_after_capture_bytes=torch.cuda.mem_get_info()[0],buckets=[1],layers=48,
        communication=communication.receipt())
    return graph
