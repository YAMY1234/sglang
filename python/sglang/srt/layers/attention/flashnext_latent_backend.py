"""QSA routing with independent shared-prefix and private-request page tables."""
import copy
import os

from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend


class FlashNextLatentAttnBackend(QwenSparseAttnBackend):
    def __init__(self, runner):
        super().__init__(runner)
        self.latent_pool = runner.token_to_kv_pool
        private_runner = copy.copy(runner)
        private_runner.req_to_token_pool = self.latent_pool.deep_request_pool
        private_runner.token_to_kv_pool = self.latent_pool.deep
        self.deep_backend = QwenSparseAttnBackend(private_runner)
        self.verify_page_cache = os.environ.get("SGLANG_FLASHNEXT_VERIFY_PAGE_CACHE", "0") == "1"
        self._private_forward = self._private_batch = None

    def _reset_private_batch(self):
        # Metadata boundaries delimit a forward, including graph capture and
        # replay. Never retain addresses across requests or allocation changes.
        self._private_forward = self._private_batch = None

    def _deep_batch(self, forward_batch):
        if not self.verify_page_cache or not forward_batch.forward_mode.is_target_verify():
            return self.latent_pool.deep_batch(forward_batch)
        if self._private_forward is not forward_batch:
            self._private_batch = self.latent_pool.deep_batch(forward_batch)
            self._private_forward = forward_batch
        return self._private_batch

    def init_forward_metadata(self, forward_batch):
        self._reset_private_batch()
        if not getattr(forward_batch, "flashnext_private_locations", False):
            super().init_forward_metadata(forward_batch)
        self.deep_backend.init_forward_metadata(self._deep_batch(forward_batch))

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        self._reset_private_batch()
        if not getattr(forward_batch, "flashnext_private_locations", False):
            super().init_forward_metadata_out_graph(forward_batch, in_capture)
        self.deep_backend.init_forward_metadata_out_graph(
            self._deep_batch(forward_batch), in_capture)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self.deep_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def commit_qsa_verify(self, steps):
        super().commit_qsa_verify(steps)
        self.deep_backend.commit_qsa_verify(steps)

    def init_forward_metadata_in_graph(self, forward_batch):
        self._reset_private_batch()
        if not getattr(forward_batch, "flashnext_private_locations", False):
            super().init_forward_metadata_in_graph(forward_batch)
        self.deep_backend.init_forward_metadata_in_graph(
            self._deep_batch(forward_batch))

    def get_indexer_metadata(self, layer_id, forward_batch):
        if layer_id >= 31:
            return self.deep_backend.get_indexer_metadata(
                layer_id, self._deep_batch(forward_batch))
        return super().get_indexer_metadata(layer_id, forward_batch)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if layer.layer_id >= 31:
            return self.deep_backend.forward_extend(q, k, v, layer,
                self._deep_batch(forward_batch), save_kv_cache, **kwargs)
        return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if layer.layer_id >= 31:
            return self.deep_backend.forward_decode(q, k, v, layer,
                self._deep_batch(forward_batch), save_kv_cache, **kwargs)
        return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
