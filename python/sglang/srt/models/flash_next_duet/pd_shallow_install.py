"""Replace only the two policy-dependent entrypoints of the PD helper package."""

from functools import wraps


def install(cls, stock, legacy):
    from . import pd_shallow as pd

    prepare = cls.prepare_before_cuda_graph_capture
    # Preserve the qualified loading, forward warmup, D boundary completion,
    # factor-only hooks and optional paired diagnostics without copying them.
    legacy.install(cls, stock)
    installed_prepare = cls.prepare_before_cuda_graph_capture
    installed_prefill = cls._twinstar_prefill

    @wraps(installed_prepare)
    def prepare_capture(self, runner):
        if self.pd_shallow_role is not None:
            pd.attach(self, runner)
        return prepare(self, runner)

    @wraps(installed_prefill)
    def dispatch_prefill(self, input_ids, positions, fb):
        if self.pd_shallow_role == "prefill":
            return pd.prefill_extend(self, input_ids, positions, fb)
        return installed_prefill(self, input_ids, positions, fb)

    cls.prepare_before_cuda_graph_capture = prepare_capture
    cls._twinstar_prefill = dispatch_prefill
