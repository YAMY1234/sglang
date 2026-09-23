"""Final Flash-Next arrival work: plan once and compute only consumed K/V.

The normal model forward and stock/latent-off paths do not call this module.
All five private layers share the same token/group plan. K normalization and
compressed-index normalization retain the existing fused kernels and rounding.
"""
from copy import copy
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _plan(virtual, page_map, positions, rope, kv_locs, compressed_locs, group_rows,
          N: tl.constexpr, G: tl.constexpr, START: tl.constexpr,
          MAP_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    pos = START + i
    loc = tl.load(virtual + i, valid, other=0).to(tl.int64)
    tl.store(positions + i, pos, valid)
    for axis in range(3):
        tl.store(rope + i * 3 + axis, pos, valid)
    for layer in tl.static_range(5):
        page = tl.load(page_map + (loc // 64) * MAP_STRIDE + layer, valid, other=0).to(tl.int64)
        tl.store(kv_locs + layer * N + i, page * 64 + loc % 64, valid)
    group_valid = i < G
    first = tl.load(virtual + i * 4, group_valid, other=0).to(tl.int64)
    for layer in tl.static_range(5):
        page = tl.load(page_map + (first // 64) * MAP_STRIDE + layer, group_valid, other=0)
        tl.store(compressed_locs + layer * G + i, page * 16 + (first % 64) // 4, group_valid)
    for member in range(4):
        tl.store(group_rows + i * 4 + member, i * 4 + member, group_valid)


def make_batch(fb, row, start, stop, token_ids, private_locs, deep_pool, final, *, implementation="kv-only"):
    """No device-to-host reads or per-chunk hybrid attention planning."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    if start % 4 or deep_pool.page_size != 64 or deep_pool.qsa_compress_ratio != 4:
        raise ValueError('final materialization requires group-aligned page64 chunks')
    n = stop-start; groups = n//4
    if n <= 0 or private_locs.numel() != n:
        raise ValueError('invalid materialization chunk')
    device = token_ids.device
    positions = torch.empty(n, dtype=torch.int64, device=device)
    rope = torch.empty((n, 3), dtype=torch.int64, device=device)
    kv_locs = torch.empty((5, n), dtype=torch.int64, device=device)
    compressed = torch.empty((5, groups), dtype=torch.int32, device=device)
    group_rows = torch.empty((groups, 4), dtype=torch.int32, device=device)
    _plan[(triton.cdiv(n, 256),)](private_locs, deep_pool.physical_page_map,
        positions, rope, kv_locs, compressed, group_rows, n, groups, start,
        deep_pool.physical_page_map.stride(0), 256)
    nb = copy(fb)
    nb.forward_mode = ForwardMode.EXTEND
    nb.batch_size = 1
    nb.input_ids = token_ids
    nb.positions = positions
    nb.req_pool_indices = fb.req_pool_indices[row:row+1]
    nb.req_pool_indices_cpu = fb.req_pool_indices_cpu[row:row+1]
    nb.out_cache_loc = private_locs
    nb.extend_seq_lens_cpu = [n]
    nb.extend_prefix_lens_cpu = [start]
    nb.twinstar_prompt_final = [final]
    nb.flashnext_private_locations = True
    nb.flashnext_arrival_plan = SimpleNamespace(
        deep=deep_pool, kv=kv_locs, compressed=compressed, group_rows=group_rows,
        rope=rope, groups=groups, tail=n%4, count=n,
        request_slot=int(fb.req_pool_indices_cpu[row]), implementation=implementation)
    return nb


def emit_kv(emitter, hidden, fb):
    """BF16 projection of trained K/V and index K; Q/gate are never consumed."""
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate
    from sglang.kernels.ops.attention.qsa_indexer import qsa_index_k_compress_store
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    plan=fb.flashnext_arrival_plan
    src=emitter.qsa
    if src is None or hidden.dtype != torch.bfloat16:
        raise ValueError('final arrival requires trained bf16 QSA emitters')
    index=src.indexer; deep=plan.deep
    local=deep._transfer_full_attention_id(emitter.layer_id)
    if index.index_kv_heads != 1 or index.index_head_dim != 128:
        raise ValueError('unsupported final indexer shape')
    q_width=src.q_size * (2 if src.attn_output_gate else 1)
    weight=src.qkv_proj.weight[q_width:q_width+2*src.kv_size]
    index_weight=index.index_qk_proj.weight[index.index_n_heads*index.index_head_dim:]
    if weight.dtype != torch.bfloat16 or index_weight.dtype != torch.bfloat16:
        raise ValueError('emitter loaded with unexpected precision')
    if plan.implementation == "kv-preserve":
        projected, _ = src.qkv_proj(hidden)
        kv = projected[:, q_width:q_width+2*src.kv_size]
    else:
        kv=F.linear(hidden,weight)
    k,v=kv.split(src.kv_size,dim=-1)
    # Reuse the identical K branch of the stock fused norm/RoPE kernel.
    # Zero Q heads removes the discarded Q/gate work without changing K math.
    _,k,_=fused_qk_gemma_rmsnorm_rope_gate(k[:,:0],k,src.q_norm.weight.data,
        src.k_norm.weight.data,src.rotary_emb.cos_sin_cache,fb.positions,
        src.q_norm.variance_epsilon,0,src.num_kv_heads,src.head_dim,
        src.rotary_emb.rotary_dim,has_gate=False)
    attn=emitter.layer.attn
    QSATokenToKVPool.set_kv_buffer(deep,attn,plan.kv[local],
        k.view(-1,attn.tp_k_head_num,attn.qk_head_dim),
        v.view(-1,attn.tp_v_head_num,attn.v_head_dim))
    if plan.implementation == "kv-preserve":
        projected_index, _ = index.index_qk_proj(hidden)
        token_k = projected_index[:, index.index_n_heads*index.index_head_dim:].contiguous().view(plan.count,1,128)
    else:
        token_k=F.linear(hidden,index_weight).view(plan.count,1,128)
    if plan.tail:
        first=plan.count-plan.tail
        slot=plan.request_slot*4
        # Completed groups are consumed directly from token_k. Only the live
        # incomplete group belongs in this request's recurrent pending ring.
        deep.get_qsa_key_state_buffer(emitter.layer_id)[slot:slot+plan.tail].copy_(token_k[first:])
        deep.qsa_rope_position_buffer[slot:slot+plan.tail].copy_(plan.rope[first:])
    if plan.groups:
        buffer=deep.get_qsa_compressed_k_buffer(emitter.layer_id)
        qsa_index_k_compress_store(token_k.view(plan.count,128),plan.group_rows,plan.rope,
            index.rotary_emb.cos_sin_cache,index._rope_axis_map(hidden.device),
            index.k_layernorm.weight.data,plan.compressed[local],buffer.view(buffer.shape[0],-1),
            4,index.rotary_emb.rotary_dim,index.k_layernorm.variance_epsilon,
            index.rotary_emb.is_neox_style)
