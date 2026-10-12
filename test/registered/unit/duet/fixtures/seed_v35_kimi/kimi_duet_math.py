"""Kimi arithmetic: the model-specific KDA scan, and the residual code and batched state cut of sglang.srt.duet under the Kimi
line's import path (object aliases, not alternative implementations; the cut's default side="right" is the KDA layout).
"""
import logging

import torch

logger = logging.getLogger(__name__)


# The code and the cut moved verbatim to the fork's common layer sglang.srt.duet.{latent_codec,state_factor}; the Kimi model
# (ResidualCode) and the archived cut_quality probe (project_state_batched) import them from here.  Only the KDA-specific scan
# below stays here.
from sglang.srt.duet import latent_codec as _codec
from sglang.srt.duet import state_factor as _state

ResidualCode = _codec.ResidualCode


def kda_state(k, v, log_decay, beta):
    """The reference scan's final state. Inputs (1,T,H,D), beta (1,T,H).

    Use the same FLA path when available, with the same torch recurrence
    fallback as the reference. This path is used only for fp32 emitters.
    """
    if k.is_cuda:
        try:
            from fla.ops.kda import chunk_kda
        except ImportError:
            logger.warning("fla.ops.kda.chunk_kda unavailable; kda_state falls back to the torch recurrence")
        else:
            _, state = chunk_kda(q=k.to(v.dtype), k=k.to(v.dtype), v=v, g=log_decay,
                                 beta=beta, initial_state=None, output_final_state=True,
                                 use_qk_l2norm_in_kernel=False)
            return state.float()
    state = k.new_zeros((k.shape[0], k.shape[2], k.shape[3], v.shape[3]), dtype=torch.float32)
    for t in range(k.shape[1]):
        state = state * log_decay[:, t].exp()[..., :, None]
        error = v[:, t].float() - torch.einsum("bhk,bhkv->bhv", k[:, t].float(), state)
        state = state + k[:, t].float()[..., :, None] * (beta[:, t, :, None] * error)[..., None, :]
    return state


project_state_batched = _state.project_state_batched   # right side = the KDA layout
