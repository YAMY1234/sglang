"""Replace only the two policy-dependent entrypoints of the PD helper package."""

from functools import wraps
import os


def factor_only_contract(owner):
    """Qualify full-depth DUET independently of its constructed emitters.

    Only the factor-only adapter may split this route into N-1 plus the native
    recurrent tail. Latent storage and shallow P loading have other wire
    contracts and cannot use this adapter.
    """
    fs = owner.fullstack
    if (
        not fs
        or "duet_spec" not in fs
        or fs.get("prefill_layer_trim") is not False
        or fs["gdn_rank"] <= 0
        or fs.get("prefill_saving_policy") != "kv-and-ssm"
        or fs.get("qsa_code") != "off"
        or owner.fullstack_v3_latent
        or owner.pd_shallow_role is not None
        or owner.n_layers != 48
        or owner.p_layer_ids != list(range(48))
        or len(owner.model.model.layers) != 48
        or owner._emit_ids()
        or [i for i, kind in enumerate(owner.config.layers_block_type) if kind != "attention"]
        != [i for i in range(48) if i % 4 != 3]
    ):
        raise ValueError(
            "native factor-only requires trim=0, all 48 P layers, 36 native GDN "
            "layers, inactive emitters and the KV/SSM wire contract"
        )
    return dict(
        model_kind="native-duet",
        emitters=len(owner.emitters),
        active_emitters=0,
        fullstack=True,
        prefill_layer_trim=False,
        prefill_route="full-model-factor-only-split",
    )


def install(cls, stock, legacy):
    from . import pd_shallow as pd

    prepare = cls.prepare_before_cuda_graph_capture
    # Preserve the qualified loading, forward warmup, D boundary completion,
    # factor-only hooks and optional paired diagnostics without copying them.
    if os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL", "0") == "1":
        legacy.install(cls, stock, factor_only_contract=factor_only_contract)
        # The native trim=0/r>0 predicate normally selects its own P/boundary
        # split. The factor-only wrapper already owns that split at every GDN
        # layer; select the stock whole-model body, not a second nested split.
        # This class can be constructed only after the above contract passes.
        predicate = cls._is_twinstar_prefill

        @wraps(predicate)
        def factor_only_prefill(self, fb):
            return False

        cls._is_twinstar_prefill = factor_only_prefill
    else:
        # Flag-off also remains compatible with the original helper signature.
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
