"""Ablate base RMSNorm rounding against the pinned PyTorch reference."""
from types import MethodType
import os
import torch
from lightning_duet.engine import NemotronHForCausalLM as Duet


def reference_norm(self, x, residual=None, post_residual_addition=None, quant_linear=None):
    if quant_linear is not None or post_residual_addition is not None:
        raise ValueError('probe only covers ordinary BF16 Lightning RMSNorm')
    if residual is not None:
        x = x + residual
        residual = x
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.variance_epsilon)
    out = self.weight * normalized.to(x.dtype)
    return (out, residual) if residual is not None else out


class NemotronHForCausalLM(Duet):
    def load_weights(self, weights, is_mtp=False):
        super().load_weights(weights, is_mtp)
        variant = os.environ.get("LIGHTNING_PROBE_VARIANT", "norm")
        if variant not in ("norm", "embedding", "both"):
            raise ValueError("invalid diagnostic variant")
        if variant in ("norm", "both"):
            for norm in [self.model.norm_f, *[layer.norm for layer in self.model.layers]]:
                norm.forward = MethodType(reference_norm, norm)
        if variant in ("embedding", "both"):
            runtime = self.lightning_runtime
            encode = runtime.components.code.encode
            def with_fresh_embedding(hidden, unused_alias, token_ids):
                return encode(hidden, runtime.body.embed_tokens(token_ids), token_ids)
            runtime.components.code.encode = with_fresh_embedding


EntryClass = NemotronHForCausalLM
