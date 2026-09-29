"""Opt-in joint normal/checkpoint factorization for two singleton states.

The two independent fixed random directions remain separate. Publication,
checkpoint positions and recurrent boundary steps belong to the callers.
"""
import os

import torch

FLAG = "SGLANG_GDN_PREFILL_JOIN_BRANCHES"


def configured():
    return os.environ.get(FLAG, "0") == "1"


def enabled():
    marker = os.environ.get(FLAG + "_FILE")
    return configured() and (not marker or os.path.exists(marker))


def eligible(cfg, normal, tracked):
    return cfg.init_method == "k31" and normal == tracked == 1


def modes(cfg, normal, tracked):
    return (False, True) if configured() and eligible(cfg, normal, tracked) else (False,)


class JointInputs:
    """Owned inputs allocated before capture; rebinding adds no concatenation."""
    def __init__(self, normal, tracked, omega, track_omega, *, states=None,
                 pack_heads=False, vbar=None):
        if len(normal) != len(tracked) or not normal:
            raise ValueError("joint factorization needs both complete layer lists")
        if any(a.shape != b.shape or a.shape[0] != 1 or
               a.dtype != torch.float32 or b.dtype != torch.float32
               for a, b in zip(normal, tracked)):
            raise ValueError("joint factorization requires singleton FP32 states")
        if omega is None or track_omega is None or omega.shape != track_omega.shape:
            raise ValueError("joint factorization requires both fixed directions")
        self.states = ([a.new_zeros((2, *a.shape[1:])) for a in normal]
                       if states is None else states)
        if len(self.states) != len(normal) or any(
                x.shape != (2, *a.shape[1:]) or x.dtype != a.dtype or x.device != a.device
                for x, a in zip(self.states, normal)):
            raise ValueError("invalid owned joint input storage")
        self.normal = [x[:1] for x in self.states]
        self.tracked = [x[1:] for x in self.states]
        self.omega = torch.cat((omega, track_omega), dim=0)
        self.head_packed = pack_heads
        if pack_heads:
            self.heads = normal[0].shape[1]
            if vbar is None or vbar.shape != (len(normal), self.heads, normal[0].shape[2]):
                raise ValueError("head packing requires every layer's fixed vbar")
            # B2 layer slices require U/W contiguous copies in factorize_layers.
            # Keep B1 and concatenate independent heads within each layer instead.
            # All views and duplicated fixed bases are owned before graph capture.
            self.head_states = [x.reshape(1, 2*self.heads, *x.shape[2:]) for x in self.states]
            self.head_omega = self.omega.reshape(1, 2*self.heads, *self.omega.shape[2:])
            self.head_vbar = torch.cat((vbar, vbar), dim=1)

    def evaluate(self, eager, vbar, cfg):
        if self.head_packed:
            values = eager(self.head_states, self.head_vbar, cfg, omega=self.head_omega)
            return ([tuple(x[:, :self.heads] for x in layer) for layer in values],
                    [tuple(x[:, self.heads:] for x in layer) for layer in values])
        values = eager(self.states, vbar, cfg, omega=self.omega)
        return ([tuple(x[:1] for x in layer) for layer in values],
                [tuple(x[1:] for x in layer) for layer in values])
