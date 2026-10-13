"""Inference-only final-state adapter for the serving FLA chunk KDA.

Keep the training cumsum/intra/state kernels and their BF16/FP32 boundaries.
Only the subsequent, unconsumed chunk_gla_fwd_o_gk readout is omitted. The
intra kernel still computes Aqk; removing its q-dependent work needs a
separately qualified kernel and is deliberately not claimed here.
"""
from functools import lru_cache
import inspect


@lru_cache(maxsize=1)
def _components():
    from fla.ops.kda import chunk_fwd
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices

    # FLA's internal ABI is not public. Fail closed if it changes, instead of
    # calling a different scan or dropping a state-layout/gating argument.
    required = {
        "chunk_local_cumsum": {"g", "scale", "chunk_size", "cu_seqlens", "chunk_indices"},
        "chunk_kda_fwd_intra": {"q", "k", "v", "gk", "beta", "scale", "cu_seqlens", "chunk_size", "chunk_indices",
                              "safe_gate", "disable_recompute"},
        "chunk_gated_delta_rule_fwd_h": {"k", "w", "u", "gk", "initial_state", "output_final_state", "cu_seqlens",
                                        "cu_seqlens_cpu", "chunk_indices", "chunk_size", "state_v_first"},
    }
    for name, parameters in required.items():
        if not parameters <= inspect.signature(getattr(chunk_fwd, name)).parameters.keys():
            raise RuntimeError(f"GLM lean emitter: unsupported FLA ABI for {name}")
    if not {"cu_seqlens", "chunk_size", "cu_seqlens_cpu"} <= inspect.signature(prepare_chunk_indices).parameters.keys():
        raise RuntimeError("GLM lean emitter: unsupported FLA chunk-index ABI")
    return chunk_fwd, prepare_chunk_indices, RCP_LN2


def kda_final_state(k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu, lower_bound):
    fwd, prepare, rcp_ln2 = _components()
    chunk_size = 64
    indices = prepare(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
    gc = fwd.chunk_local_cumsum(g=g, scale=rcp_ln2, chunk_size=chunk_size,
                                cu_seqlens=cu_seqlens, chunk_indices=indices)
    w, u, _, kg, _, _ = fwd.chunk_kda_fwd_intra(
        q=k, k=k, v=v, gk=gc, beta=beta, scale=k.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=indices,
        safe_gate=True, disable_recompute=False,
    )
    _, _, state = fwd.chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=gc, initial_state=initial_state,
        output_final_state=True, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=indices, chunk_size=chunk_size, state_v_first=False,
    )
    return state
