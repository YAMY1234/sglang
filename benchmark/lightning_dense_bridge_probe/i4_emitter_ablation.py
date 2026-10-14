"""Unscored emitter stage substitutions; reference arithmetic, no pool changes."""
import torch
import torch.nn.functional as F


def run(stages, components, lid, residual, metadata, pool, plan):
    from sglang.srt.models.lightning_duet.components import final_mamba_state, rms_norm
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    from sglang.kernels.ops.mamba.triton_ops import mamba_chunk_scan_combined

    g = components.geometry
    w = lambda key: components.weights[f'emitters.{lid}.{key}']
    lengths = metadata.mixed_metadata.extend_seq_lens_cpu
    cuts = [0]
    for length in lengths:
        cuts.append(cuts[-1]+length)
    slices = [slice(a,b) for a,b in zip(cuts,cuts[1:])]
    if 'p' in stages:
        normalized = [rms_norm(residual[s].float(), w('norm.weight'), components.eps) for s in slices]
        def project(key):
            return torch.cat([F.linear(x, w(key)) for x in normalized])
    else:
        normalized = rms_norm(residual.float(), w('norm.weight'), components.eps)
        def project(key):
            return F.linear(normalized, w(key))
    if lid in components.attention_emitters:
        return {k: project(f'{k}_proj.weight').reshape(-1,g.kv_heads,g.attn_dim) for k in ('k','v')}
    projected = project('mixer.in_proj.weight')
    _, xbc, raw_dt = projected.split([g.intermediate,g.conv_dim,g.heads],-1)
    mixed = metadata.mixed_metadata
    if 'c' in stages:
        pieces=[]
        for row,s in enumerate(slices):
            history=F.pad(xbc[s].T.unsqueeze(0),(g.conv_width,0))
            pool.conv[pool.layer_map[lid],plan.slots[row]].copy_(history[0,:,-g.conv_width:])
            pieces.append(F.silu(F.conv1d(history,w('mixer.conv1d.weight'),w('mixer.conv1d.bias'),groups=g.conv_dim))[0].T)
        convolved=torch.cat(pieces)
    else:
        convolved=causal_conv1d_fn(xbc.T,w('mixer.conv1d.weight')[:,0],w('mixer.conv1d.bias'),
            conv_states=pool.conv[pool.layer_map[lid]],query_start_loc=metadata.query_start_loc,
            seq_lens_cpu=lengths,cache_indices=plan.slots,has_initial_state=mixed.has_initial_states,activation='silu').T
    x,b,c=convolved.split([g.intermediate,g.bc_dim,g.bc_dim],-1)
    dt=F.softplus(raw_dt.float()+w('mixer.dt_bias'))
    a=-w('mixer.A_log').exp()
    if 's' in stages:
        final=torch.cat([final_mamba_state(x[t].reshape(1,-1,g.heads,g.head_dim),dt[t][None],a,
            b[t].reshape(1,-1,g.groups,g.state_dim),components.chunk) for t in slices])
    else:
        scan_x=x.reshape(1,-1,g.heads,g.head_dim)
        final=mamba_chunk_scan_combined(scan_x,dt[None],a,b.reshape(1,-1,g.groups,g.state_dim),
            c.reshape(1,-1,g.groups,g.state_dim),chunk_size=mixed.chunk_size,seq_idx=mixed.seq_idx,
            chunk_indices=mixed.chunk_indices,chunk_offsets=mixed.chunk_offsets,cu_seqlens=metadata.query_start_loc,
            return_varlen_states=True,return_final_states=False,state_dtype=torch.float32,
            out=torch.empty_like(scan_x),state_input_precision='ieee')
    pool.commit_extend(lid,plan,final)
    return None
