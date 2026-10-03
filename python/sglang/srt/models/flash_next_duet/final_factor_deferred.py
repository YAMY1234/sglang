"""Private dense B1 boundary graph for the default-off final-factor candidate."""
import copy
import logging

import torch

logger = logging.getLogger(__name__)


def startup_rejection(owner, runner, pool):
    args = runner.server_args
    if args.disaggregation_mode != "null":
        return "pd"
    if (not owner.fullstack or owner.fullstack_v3_latent or args.is_embedding
            or args.pp_size != 1 or runner.is_draft_worker):
        return "model"
    if not runner.spec_algorithm.is_none():
        return "spec"
    if (not args.disable_overlap_schedule or args.enable_two_batch_overlap
            or args.enable_torch_compile or args.disable_cuda_graph or runner.lora_manager is not None
            or args.enable_linear_replayssm):
        return "execution"
    if (pool is None or pool._k31_batch_graph is None
            or pool._tracked_factor_side is None
            or not pool._tracked_factor_side.deferred or not pool.cfg.strict_chunk
            or pool.prefix_dense is not None):
        return "requires_k31_tracked_2b"
    return None


def boundary_runner_type(owner):
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend

    class DenseBoundaryRunner(DecodeCudaGraphRunner):
        def capture(self):
            if (self.enable_pdmux or self.attention_graph_variants is not None
                    or self.require_gathered_buffer or self.pp_size != 1
                    or self._metadata_glue is not None):
                raise ValueError("deferred final graph has unsupported capture variants")
            self.dense_boundary_arena = torch.cuda.graph_pool_handle()
            self.backend = FullCudaGraphBackend(
                self, graph_pool=self.dense_boundary_arena, reuse_output_buffer=False)
            self.enable_profile_cuda_graph = False
            return super().capture()

        def _capture_one_stream(self, stream_idx=None):
            if stream_idx is not None:
                raise ValueError("deferred final graph requires a single stream")
            # Capture the complete stock forward, including mix and logits.
            self.capture_one_shape(1, owner.model.forward)

        def execute_dense(self, batch):
            from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

            if batch.batch_size != 1 or not self.can_run_graph(batch):
                raise RuntimeError("deferred final boundary lost its pre-admitted B1 shape")
            current = copy.copy(batch)
            current.forward_metadata_ready = False
            with forward_context(ForwardContext(attn_backend=self.attn_backend)):
                return self.execute(current)

    return DenseBoundaryRunner


def install(owner, runner):
    from sglang.srt.environ import envs

    requested = envs.SGLANG_GDN_FINAL_FACTOR_DEFERRED.get()
    if not requested:
        logger.info("final_factor_deferred=0 (switch off)")
        return
    pool = getattr(runner.req_to_token_pool, "factored_gdn_pool", None)
    reason = startup_rejection(owner, runner, pool)
    if reason is not None:
        logger.warning("final_factor_deferred=0 (requested=1 reason=%s)", reason)
        return
    from sglang.srt.mem_cache.gdn_factored_pool import factorize_layers
    from sglang.srt.mem_cache.gdn_final_factor_deferred import FinalFactorDeferred
    from sglang.srt.model_executor.pd_tail_cuda_graph_runner import isolated_runner, private_stream
    from sglang.srt.model_executor.pd_tail_comm_guard import TailCommunicationLease
    from sglang.srt.distributed.device_communicators import pynccl_allocator

    controller = FinalFactorDeferred(pool, pool._tracked_factor_side)
    controller.prewarm(factorize_layers)
    body = owner.model.model
    retained_hc = body.last_hc_hidden_states
    before = torch.cuda.memory_reserved()
    if torch.cuda.mem_get_info()[0] < 10 << 30:
        raise RuntimeError("deferred final boundary needs 10 GiB startup headroom")
    communication = TailCommunicationLease()
    workspace = runner.init_new_workspace
    try:
        backend = runner._get_attention_backend(init_new_workspace=True)
    finally:
        runner.init_new_workspace = workspace
    linear = backend.linear_attn_backend
    if not linear.kernel_dispatcher.supports_packed_decode:
        raise ValueError("deferred final boundary needs the P arm's packed dense kernel")
    linear._final_boundary_dense = True
    isolated = isolated_runner(runner, backend)
    stream = private_stream()
    previous = pynccl_allocator._graph_pool_id
    try:
        graph = boundary_runner_type(owner)(
            isolated, attn_backend=backend, capture_bs_override=[1],
            share_input_buffers=False, capture_stream=stream,
        )
    finally:
        pynccl_allocator.set_graph_pool_id(previous)
        body.last_hc_hidden_states = retained_hc
    torch.cuda.synchronize()
    communication.check()
    graph.communication = communication
    graph.private_capture_stream = stream
    retained = torch.cuda.memory_reserved()-before
    if retained > 8 << 30:
        raise RuntimeError("deferred final boundary exceeds 8 GiB graph/metadata budget")
    controller.boundary = graph
    owner._final_factor_deferred = pool._final_factor_deferred = controller
    logger.info("final_factor_deferred=1 (AGG B1 generation PP1, graph+2b, "
                "dense S_N then F(S_N), joint=0, count=r, default off) "
                "boundary_retained_bytes=%d input_buffers_shared=0", retained)
