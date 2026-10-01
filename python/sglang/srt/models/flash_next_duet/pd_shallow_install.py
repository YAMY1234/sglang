"""Native-release PD dispatch over the existing boundary transport helpers.

The stock class keeps its original constructor, loader and boundary body. Only
the enabled P role selects the shallow constructor/weight filter/extend path.
Guard instrumentation is enabled separately by its bounded-job environment.
"""

import os
from functools import wraps


def install(cls, stock):
    from . import pd_shallow as pd

    init = cls.__init__
    load = cls.load_weights
    prepare = cls.prepare_before_cuda_graph_capture
    forward = cls.forward
    prefill = cls._twinstar_prefill
    boundary = cls._boundary_graph

    @wraps(init)
    def initialize(self, config, quant_config=None, prefix=""):
        text = getattr(config, "text_config", config)
        ts = getattr(config, "twinstar", None) or getattr(text, "twinstar", None)
        if os.environ.get("TWINSTAR_FULLSTACK", "1") == "0":
            ts = None
        self.pd_shallow_role = pd.role(config, ts)
        if self.pd_shallow_role != "prefill":
            return init(self, config, quant_config, prefix)
        factory = stock.Qwen4ExpForConditionalGeneration
        # Model construction is single-threaded. Never leave this substitution
        # installed for the D role or another model after construction.
        stock.Qwen4ExpForConditionalGeneration = lambda c, q, p: pd.build_stock(
            factory, c, q, p, "prefill"
        )
        try:
            return init(self, config, quant_config, prefix)
        finally:
            stock.Qwen4ExpForConditionalGeneration = factory

    @wraps(load)
    def load_weights(self, weights):
        if self.pd_shallow_role == "prefill":
            original = self.model.load_weights
            self.model.load_weights = lambda stream: original(
                (name, value)
                for name, value in stream
                if pd.keep_weight(self.model, name, value, "prefill")
            )
            try:
                result = load(self, weights)
            finally:
                self.model.load_weights = original
            pd.finish_loading(self)
        else:
            result = load(self, weights)
        from twinstar_sgl.pd_shallow_audit import attach_reference, record_loading

        attach_reference(self)
        record_loading(self)
        from sglang.srt.runtime_context import get_disagg

        if (
            getattr(get_disagg(), "flashnext_pd_boundary_graph", False)
            and self.pd_shallow_role == "decode"
        ):
            from twinstar_sgl.pd_boundary_graph import install

            install()
        if getattr(get_disagg(), "flashnext_pd_staging", False):
            from sglang.srt.disaggregation.flashnext_staging import (
                reserve_before_kv_profile,
            )

            reserve_before_kv_profile()
        if os.environ.get("TWINSTAR_CUDA_TIMELINE"):
            # Install in each actual worker, including spawned processes whose
            # import of the guard entry did not inherit the parent's wrapper.
            from twinstar.bench.flashnext_pd_timing_server import install

            install()
            from twinstar_sgl.pd_boundary_observe import install

            install()
        if os.environ.get("TWINSTAR_PD_CPU_TRACE"):
            from twinstar.bench.flashnext_pd_cpu_trace import install

            install()
        if os.environ.get("TWINSTAR_PD_PAIRED_GUARD"):
            from twinstar_sgl.pd_shallow_paired import attach

            attach(self)
        return result

    @wraps(prepare)
    def prepare_capture(self, runner):
        if self.pd_shallow_role is not None:
            pd.attach(self, runner)
        return prepare(self, runner)

    @wraps(forward)
    def dispatch_forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        if self.pd_shallow_role == "prefill" and forward_batch.forward_mode.is_decode():
            return pd.prefill_decode_warmup(self, forward_batch)
        return forward(self, input_ids, positions, forward_batch, *args, **kwargs)

    @wraps(prefill)
    def dispatch_prefill(self, input_ids, positions, fb):
        if self.pd_shallow_role == "prefill":
            return pd.prefill_extend(self, input_ids, positions, fb)
        probe = getattr(self, "_pd_paired_guard", None)
        if probe is not None and probe.selected(fb):
            return probe.prefill(prefill, input_ids, positions, fb)
        return prefill(self, input_ids, positions, fb)

    @wraps(boundary)
    def checked_boundary(self, *args, **kwargs):
        if self.pd_shallow_role == "prefill":
            raise RuntimeError(
                "shallow PD must take h31 from the main extend, not a boundary forward"
            )
        value = boundary(self, *args, **kwargs)
        fb = getattr(self, "_pd_audit_boundary_batch", None)
        if fb is not None:
            from twinstar_sgl.pd_shallow_audit import snapshot

            snapshot(self, fb, "after-deep", logits=value)
            self._pd_audit_boundary_batch = None
        probe = getattr(self, "_pd_paired_guard", None)
        if probe is not None:
            probe.after_legacy_boundary(value)
        return value

    def complete_pd_boundary(self, batch, scheduler):
        if self.pd_shallow_role != "decode":
            raise RuntimeError(
                "boundary completion requires the shallow PD decode role"
            )
        return pd.decode_boundary(self, batch, scheduler)

    cls.__init__ = initialize
    cls.load_weights = load_weights
    cls.prepare_before_cuda_graph_capture = prepare_capture
    cls.forward = dispatch_forward
    cls._twinstar_prefill = dispatch_prefill
    cls._boundary_graph = checked_boundary
    cls.complete_pd_boundary = complete_pd_boundary
    # Only the explicit full-depth factor-only P process installs this adapter.
    # Stock, D and the existing shallow/flag-off paths retain their dispatch.
    if os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL", "0") == "1":
        from twinstar_sgl.pd_factor_only import install as install_factor_only

        install_factor_only(cls)
