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
    def __init__(self, normal, tracked, omega, track_omega, *, states=None):
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

    def evaluate(self, eager, vbar, cfg):
        values = eager(self.states, vbar, cfg, omega=self.omega)
        return ([tuple(x[:1] for x in layer) for layer in values],
                [tuple(x[1:] for x in layer) for layer in values])
