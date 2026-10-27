"""Resolve the installed ordinary FN FP4 tactic, without executing its body.

Invoke in the same native warmup/autotune context as the original operation.
No tuning context, custom candidate set, or timing loop is introduced here.
"""
import torch


def resolve_native_tactic(native_kwargs, *, routing_logits=None):
    from flashinfer.fused_moe import core as c

    k = dict(native_kwargs)
    runtime = c.get_trtllm_moe_sm100_module()
    x, sf = k['hidden_states'], k['hidden_states_scale']
    m, h = x.shape[0], x.shape[1] * (2 if x.dtype == torch.uint8 else 1)
    if (h, k['num_experts'], k['top_k'], k['local_num_experts'],
            k['local_expert_offset']) != (2560, 512, 10, 512, 0) or not 1 <= m <= 16:
        raise ValueError('This resolver is scoped to native TP1 FN M1..16')
    if k.get('num_fused_shared_experts', 0) not in (None, 0):
        raise ValueError('FN shared experts remain separate')
    supplied = k.get('topk_ids')
    if routing_logits is not None:
        mode = c.RoutingInputMode.FromLogits
        # Native FromLogits buffers are outputs, not the custom front's inputs.
        ids = torch.empty((m, k['top_k']), dtype=torch.int32, device=x.device)
        weights = torch.empty_like(ids, dtype=torch.bfloat16)
    elif isinstance(supplied, tuple):
        mode = c.RoutingInputMode.UnpackedPrecomputed
        ids, weights = supplied
    else:
        mode = c.RoutingInputMode.UnpackedPrecomputed
        ids, weights = supplied, k['topk_weights']
    output = k.get('output')
    valid_h = k.get('valid_hidden_size')
    if output is None:
        output = c._alloc_trtllm_moe_output(
            m, c._trtllm_moe_output_width(h, valid_h), k['do_finalize'], x.device,
        )
    runner = runtime.MoERunner(
        runtime.moe_op, top_k=k['top_k'], num_local_experts=k['local_num_experts'],
        dtype_act=c.deduce_trtllm_gen_tensor_dtype(x, sf),
        dtype_weights=c.deduce_trtllm_gen_tensor_dtype(
            k['gemm1_weights'], k['gemm1_weights_scale']),
        fp8_quantization_type=c.Fp8QuantizationType.NoneFp8, hidden_size=h,
        hidden_size_output=(c._infer_trtllm_moe_output_hidden_size(h, valid_h)
                            if valid_h is not None else h),
        intermediate_size=k['intermediate_size'], activation_type=k['activation_type'],
        weight_layout=c.WeightLayout.MajorK, use_shuffled_weight=True,
        use_per_token_scaling=k.get('per_token_scale') is not None,
        num_experts=k['num_experts'], num_fused_shared_experts=0,
    )
    inputs = c.MoeRunnerInputs(
        output=output, routing_logits=routing_logits, topk_ids=ids, expert_weights=weights,
        hidden_states=x, hidden_states_scale=sf, gemm1_lora_delta=k.get('gemm1_lora_delta'),
        per_token_scale=k.get('per_token_scale'),
    )
    tuning = runner._make_tuning_config(
        inputs, tune_max_num_tokens=k.get('tune_max_num_tokens', 8192),
        routing_input_mode=mode,
        hidden_states_scale_layout=k.get('hidden_states_scale_layout'),
        use_cold_l2_cache=True, use_cuda_graph=True,
    )
    names = (
        'num_experts routing_bias gemm1_weights gemm1_weights_scale gemm1_bias '
        'gemm1_alpha gemm1_beta gemm1_clamp_limit gemm2_weights gemm2_weights_scale '
        'gemm2_bias output1_scale_scalar output1_scale_gate_scalar output2_scale_scalar '
        'n_group topk_group local_expert_offset routed_scaling_factor routing_method_type '
        'enable_pdl do_finalize activation_type num_fused_shared_experts norm_topk_prob '
        'routing_replay_out valid_hidden_size valid_intermediate_size'
    ).split()
    kwargs = {name: k.get(name) for name in names}
    kwargs['routing_input_mode'] = mode
    kwargs['norm_topk_prob'] = k.get('norm_topk_prob', True)
    if kwargs['enable_pdl'] is None:
        kwargs['enable_pdl'] = c.device_support_pdl(x.device)
    if not c._device_support_moe_pdl(x.device):
        kwargs['enable_pdl'] = False
    _, tactic = c.AutoTuner.get().choose_one(
        'flashinfer::trtllm_fp4_block_scale_moe', [runner], tuning,
        inputs.to_list(), **kwargs,
    )
    if tactic != -1 and min(tactic) >= 0:
        return tuple(map(int, tactic))
    # Exact native fallback for this scoped FN geometry: average tokens/expert
    # <=16*10/512<8, so computeSelectedTileN starts at8 and selectDefaultTileN
    # chooses8. Native factorization API labels getDefaultValidConfigIndex as
    # is_anchor; query metadata only, not a benchmark or alternate tactic.
    rows = runtime.moe_op.trtllm_get_valid_moe_factorizations(
        runner.dtype_act, runner.dtype_weights, runner.fp8_quantization_type,
        runner.top_k, runner.hidden_size, runner.hidden_size_output,
        runner.intermediate_size, runner.num_local_experts, runner.activation_type,
        runner.use_shuffled_weight, runner.weight_layout, runner.use_per_token_scaling,
        m, inputs.gemm1_lora_delta is not None,
    )
    return next((int(tile), int(config)) for tile, config, _, _, anchor in rows
                if int(tile) == 8 and int(anchor) == 1)
