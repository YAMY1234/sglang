"""Real imported backend entries: no extracted method globals or injected os."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from source_policy_reference import gdn_front_launch

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.layers.attention.linear import gdn_backend as impl
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class CudaMetadata(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    def data_ptr(self):
        return 64


def tensor(shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta").as_subclass(CudaMetadata)


def setup(rows):
    # Only allocation/device metadata is doubled. Both called methods and their
    # module globals below are the actual imported production module.
    h, hv = 4, 12
    c = (2 * h + hv) * 128
    batch = rows // 4
    state = MambaPool.SpeculativeState(
        conv=[tensor((4, c, 3))],
        temporal=tensor((4, hv, 128, 128)),
        intermediate_ssm=None,
        intermediate_conv_window=[tensor((4, 4, c, 3))],
        replayssm_d=tensor((4, hv, 16, 128)),
        replayssm_k=tensor((4, h, 16, 128)),
        replayssm_g=tensor((4, hv, 16), torch.float32),
        replayssm_rawv=tensor((4, hv, 16, 128)),
        replayssm_rawk=tensor((4, h, 16, 128)),
    )
    backend = object.__new__(impl.GDNAttnBackend)
    backend.mis_metadata = None
    backend.forward_metadata = NS(
        query_start_loc=tensor((batch + 1,), torch.int32),
        mamba_cache_indices=tensor((batch,), torch.int64),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    backend.verify_intermediate_state_indices = tensor((batch,), torch.int32)
    backend.state_layer_cache = lambda layer_id: state
    backend.state_pool = NS(
        replayssm_cache_base=tensor((8,), torch.int32),
        replayssm_spec_write_pos=tensor((8,), torch.int32),
        replayssm_is_flush=tensor((8,), torch.int8),
        replayssm_spec_fold=False,
        replayssm_is_kda=False,
    )
    layer = NS(
        layer_id=0,
        num_k_heads=h,
        num_q_heads=h,
        num_v_heads=hv,
        head_k_dim=128,
        head_q_dim=128,
        head_v_dim=128,
        q_dim=h * 128,
        k_dim=h * 128,
        v_dim=hv * 128,
        activation="silu",
        A_log=tensor((hv,), torch.float32),
        dt_bias=tensor((hv,)),
        conv_weights=tensor((c, 4)),
        bias=None,
    )
    fb = NS(
        spec_info=NS(draft_token_num=4),
        batch_size=batch,
        forward_mode=ForwardMode.TARGET_VERIFY,
        logical_forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=tensor((batch,), torch.int64),
    )
    from sglang.kernels.plugin_kernels.nvidia.numeric import GdnSignature

    layer._numeric_gdn_signatures = frozenset(
        GdnSignature(op, rows, h, hv, 128, 128, 4, 1, variant=v, gate_scalar_type=g)
        for op, v, g in (
            (NumericOperation.GDN_FRONT, 0, 3),
            (NumericOperation.GDN_FRONT, 1, 3),
            (NumericOperation.GDN_CORE, 0, 1),
            (NumericOperation.GDN_CORE, 1, 1),
            (NumericOperation.GDN_VERIFY_GLUE, 0, 1),
        )
    )
    layer._numeric_gdn_standalone_norm_signatures = frozenset(
        s
        for s in layer._numeric_gdn_signatures
        if s.operation == NumericOperation.GDN_CORE and s.variant == 1 and s.rows == 32
    )
    return backend, layer, fb


@pytest.mark.parametrize("enabled", [False, True])
def test_imported_glue_reads_actual_env_and_reaches_real_source_policy(
    enabled, monkeypatch
):
    backend, layer, fb = setup(16)
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1" if enabled else "0")
    assert backend.forward_numeric_verify_glue.__func__.__globals__ is vars(impl)
    assert backend.numeric_verify_glue_supported.__func__.__globals__ is vars(impl)
    with patch.object(numeric_binding, "numeric_operator", return_value=None) as lookup:
        result = backend.forward_numeric_verify_glue(
            layer=layer,
            forward_batch=fb,
            mixed_qkvz=tensor((16, 4096)),
            mixed_ba=tensor((16, 24)),
            norm_weight=tensor((128,)),
            norm_eps=3e-6,
            norm_activation="sigmoid",
        )
    assert result is None
    if enabled:
        lookup.assert_called_once()
        assert lookup.call_args.args[0].operation == NumericOperation.GDN_VERIFY_GLUE
        assert lookup.call_args.args[0].rows == 16
    else:
        lookup.assert_not_called()


@pytest.mark.parametrize("stages,selected", [("6", True), ("12", False)])
def test_imported_front_reads_stage_env_before_any_state_write(
    stages, selected, monkeypatch
):
    backend, layer, fb = setup(4)
    monkeypatch.setenv("Q38_GDNPROJ_FRONT_STAGES", stages)
    from sglang.kernels.plugin_kernels.nvidia.numeric import GdnSignature

    layer._numeric_gdn_signatures = frozenset(
        (
            GdnSignature(
                NumericOperation.GDN_FRONT,
                4,
                4,
                12,
                128,
                128,
                4,
                1,
                variant=0 if stages == "6" else 1,
                gate_scalar_type=3,
            ),
        )
    )
    variant, launch = gdn_front_launch("mk", 4, 2560, None, False)
    assert backend.forward_numeric_inproj_conv.__func__.__globals__ is vars(impl)
    with (
        patch.object(impl, "pp_spec_stable_rows_enabled", return_value=False),
        patch.object(numeric_binding, "numeric_operator", return_value=None) as lookup,
    ):
        result = backend.forward_numeric_inproj_conv(
            layer,
            fb,
            tensor((4, 2560)),
            tensor((4120, 2560)),
            norm_weight=tensor((128,)),
            norm_eps=3e-6,
            front_variant=variant,
            front_launch=launch,
        )
    assert result is None
    if selected:
        assert lookup.call_count >= 1
        assert lookup.call_args_list[0].args[0].operation == NumericOperation.GDN_FRONT
    else:
        lookup.assert_not_called()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
