"""Installed FlashInfer FP4 body, supplied native metadata, unchanged weights.

Construct once outside graph capture for one layer/M/resolved native tactic.
The CUDA front owns the live nine-tensor routing record. No Routing::Runner,
DA selector, weight packing, or weight copy is performed here.
"""
import torch

METADATA_FIELDS = (
    'total_num_padded_tokens', 'expanded_idx_to_permuted_idx',
    'permuted_idx_to_token_idx', 'expert_weights', 'expert_count_histogram',
    'num_tokens_per_expert', 'cta_idx_xy_to_batch_idx',
    'cta_idx_xy_to_mn_limit', 'num_non_exiting_ctas',
)


class NativeFp4Body:
    def __init__(self, native_kwargs, *, tactic):
        from flashinfer.fused_moe import core
        from flashinfer.fused_moe.core import RoutingInputMode

        self.kw = dict(native_kwargs)
        self.tactic = tuple(map(int, tactic))
        if len(self.tactic) != 2 or min(self.tactic) < 0:
            raise ValueError('Pass the existing resolved native (tile_n, config); no -1 fallback')
        if self.kw.get('num_fused_shared_experts', 0) not in (None, 0):
            raise ValueError('This FN binding keeps shared experts separate')
        if self.kw.get('gemm1_lora_delta') is not None:
            raise ValueError('This binding does not carry LoRA auxiliary outputs')
        routing = self.kw['topk_ids']
        if isinstance(routing, tuple):
            ids, weights = routing
        else:
            ids, weights = routing, self.kw['topk_weights']
        if ids.dtype != torch.int32 or weights.dtype != torch.bfloat16:
            raise ValueError('Native FN record requires int32 IDs and BF16 routing weights')
        self.ids, self.weights = ids, weights
        # Installed public allocator shares a top_k<=8 population validator.
        # Our front supplies exact native records itself: allocate the same slot
        # buffers directly, without invoking the incompatible population kernel.
        m, top_k = ids.shape
        experts = int(self.kw['num_experts'])
        tile_n = self.tactic[0]
        if not (1 <= m <= 16 and top_k == self.kw['top_k'] == 10
                and experts == 512 and self.kw['local_expert_offset'] == 0
                and self.kw['local_num_experts'] == 512):
            raise ValueError('FN native metadata binding requires TP1 E512/top10/M1..16')
        # The installed canonical allocation path accepts top10 and uses the
        # identical native capacities/guard. This performs allocation ONLY,
        # never canonicalize_routing or another routing GPU launch.
        shape_only_logits = torch.empty((m, experts), dtype=torch.bfloat16,
                                        device=ids.device)
        self._canonical_raw = tuple(
            core.get_trtllm_moe_sm100_module().allocate_canonical_routing(
                shape_only_logits, top_k, tile_n))
        self._canonical_views = tuple(core._torch_view_of_ffi_tensor(t)
                                      for t in self._canonical_raw)
        raw = self._canonical_views
        if len(raw) != 11:
            raise RuntimeError('Installed canonical allocation must return eleven tensors')
        self.slot = core.TrtllmMoERoutingMetadataSlot(
            tile_n=tile_n,
            total_num_padded_tokens=raw[4],
            expanded_idx_to_permuted_idx=raw[5],
            permuted_idx_to_token_idx=raw[6],
            expert_weights=weights,
            expert_count_histogram=raw[7],
            num_tokens_per_expert=raw[3],
            cta_idx_xy_to_batch_idx=raw[8],
            cta_idx_xy_to_mn_limit=raw[9],
            num_non_exiting_ctas=raw[10],
        )
        self.metadata = tuple(self.slot.tensors())
        self._ffi = core.get_trtllm_moe_sm100_module().moe_op.trtllm_fp4_block_scale_moe
        self._view = core._torch_view_of_ffi_tensor
        self._mode = int(RoutingInputMode.UnpackedPrecomputed)
        self.do_finalize = bool(self.kw['do_finalize'])
        x = self.kw['hidden_states']
        output = self.kw.get('output')
        if output is None:
            output = torch.empty(
                (x.shape[0], x.shape[1] * 2 if self.do_finalize else 0),
                dtype=torch.bfloat16, device=x.device,
            )
        self.output = output
        # Native preparation only allocates/binds the exact selected body buffers.
        self.workspace = tuple(self._view(x) for x in self._call((), True))
        if len(self.workspace) != 7:
            raise RuntimeError('Installed FP4 body ABI must expose seven workspaces')

    def _call(self, workspace, prepare):
        k = self.kw
        return self._ffi(
            self._mode, None, self.ids, self.weights, k.get('routing_bias'),
            k['hidden_states'], k['hidden_states_scale'],
            k['gemm1_weights'], k['gemm1_weights_scale'], k.get('gemm1_bias'),
            k.get('gemm1_lora_delta'), k.get('gemm1_alpha'), k.get('gemm1_beta'),
            k.get('gemm1_clamp_limit'), k['gemm2_weights'], k['gemm2_weights_scale'],
            k.get('gemm2_bias'), k.get('output1_scale_scalar'),
            k.get('output1_scale_gate_scalar'), k.get('output2_scale_scalar'),
            k.get('per_token_scale'), k['num_experts'], k['top_k'],
            k.get('num_fused_shared_experts', 0), k.get('n_group'), k.get('topk_group'),
            k['intermediate_size'], k['local_expert_offset'], k['local_num_experts'],
            k.get('routed_scaling_factor'), k['routing_method_type'], self.do_finalize,
            k['enable_pdl'], k['activation_type'], self.output, list(self.tactic),
            k.get('norm_topk_prob', True), k.get('routing_replay_out'),
            list(self.metadata), list(workspace), prepare,
            k.get('valid_hidden_size'), k.get('valid_intermediate_size'),
        )

    def run(self, packed_x=None, linear_sf=None):
        """After the front writes slot tensors, launch FC1/FC2/native finalize.

        New activation pointers may be rebound in eager/capture; graph replay
        uses its captured pointers. Buffers are serial scratch, never result cache.
        """
        if packed_x is not None:
            self.kw['hidden_states'] = packed_x
        if linear_sf is not None:
            self.kw['hidden_states_scale'] = linear_sf
        self._call(self.workspace, False)
        if self.do_finalize:
            return self.output
        # Installed raw FFI returns this same stable three-tensor deferred ABI.
        return (self.workspace[4], self.slot.expert_weights,
                self.slot.expanded_idx_to_permuted_idx)

    def run_sglang_deferred(self, packed_x=None, linear_sf=None):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _make_deferred_finalize_output,
        )
        if self.do_finalize:
            raise ValueError('Construct with native do_finalize=False for HC deferred combine')
        return _make_deferred_finalize_output(
            self.run(packed_x, linear_sf), top_k=self.kw['top_k'],
            materialized_routing=True,
        )
