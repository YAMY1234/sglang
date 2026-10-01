"""A single graph for a pending draft extend and the following draft chain."""

import contextlib

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import model_capture_mode
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)


class EAGLEExtendDraftCudaGraphRunner(EAGLEDraftCudaGraphRunner):
    def __init__(self, worker):
        factory = DraftBackendFactory(
            worker.draft_runner, worker.topk, worker.speculative_num_steps
        )
        # Private backends preserve the metadata pointers baked into fallback graphs.
        self.extend = EAGLEDraftExtendCudaGraphRunner(
            worker,
            draft_extend_attn_backend=factory.create_draft_extend_backend(),
            capture=False,
            share_buffers=False,
        )
        super().__init__(
            worker,
            draft_attn_backend=factory.create_decode_backend(),
            capture=False,
            share_buffers=False,
        )
        self.capture_bs = sorted(set(self.capture_bs) & set(self.extend.capture_bs))
        if not self.capture_bs:
            raise ValueError("extend/draft have no common request bucket")
        self.extend.capture_bs = self.capture_bs
        self.max_bs = self.extend.max_bs = max(self.capture_bs)
        self.extend.buffers.reset_index_buffers()
        self.extend.buffers.seq_lens.fill_(self.extend.seq_len_fill_value)
        self.extend.buffers.seq_lens_cpu.fill_(self.extend.seq_len_fill_value)
        self.fused_replays = 0
        self._staged_extend_bs = None
        with model_capture_mode():
            self.capture()

    @contextlib.contextmanager
    def _draft_backend(self):
        worker = self.eagle_worker
        old = worker.draft_attn_backend
        worker.draft_attn_backend = self.draft_attn_backend
        try:
            with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):
                yield
        finally:
            worker.draft_attn_backend = old

    def capture_one_shape(self, size, forward, **kwargs):
        _, extend_body, extend_hook = self.extend.capture_one_shape(
            size, forward, prepare_only=True
        )
        draft_fb, draft_body, draft_hook = super().capture_one_shape(
            size, forward, prepare_only=True
        )

        def body():
            with forward_context(
                ForwardContext(attn_backend=self.extend.draft_extend_attn_backend)
            ):
                output = extend_body()
            self.buffers.topk_index[:size].copy_(
                output.next_token_logits.argmax(dim=-1, keepdim=True)
            )
            self.buffers.topk_p[:size].fill_(1)
            self.buffers.hidden_states[:size].copy_(output.hidden_states)
            with self._draft_backend():
                result = draft_body()
            if self.model_runner.model_config.model_is_mrope:
                draft_fb.mrope_positions.sub_(self.speculative_num_steps - 1)
            return result

        def after_warmup():
            if extend_hook:
                extend_hook()
            if draft_hook:
                draft_hook()

        self.backend.capture_one(
            self._make_graph_key(size),
            body,
            capture_inputs=None,
            post_warmup_hook=after_warmup,
        )

    def stage_extend(self, forward_batch, select_index):
        self._staged_extend_bs = self.extend.execute(
            forward_batch, select_index, stage_only=True
        )

    def _replay_graph(self, shape_key, forward_batch):
        if self._staged_extend_bs != self.bs:
            raise RuntimeError("extend/draft request buckets disagree")
        out = super()._replay_graph(shape_key, forward_batch)
        self._staged_extend_bs = None
        self.fused_replays += 1
        return out
