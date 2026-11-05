"""Preflight only: real canonical IO and tensor metadata, no CUDA execution."""

from dataclasses import replace

import pytest
import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnRingIO,
    GdnSignature,
    NumericOffer,
    NumericOperation,
    NumericRegistry,
    ScalarType,
)
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.kernel_plugins.nvidia.gdn import bind_gdn_core
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


def request(rows=4, key_heads=4, value_heads=12, fused=True, bias_dtype=torch.bfloat16):
    d = tensor((1, value_heads, 16, 128))
    index = tensor((rows // 4,), torch.int32)
    flag = tensor((1,), torch.int8)
    ring = GdnRingIO(d, d, d, d, d, d, d, d, index, index, index, index, flag)
    return dict(
        qkv=tensor((rows, (2 * key_heads + value_heads) * 128)),
        a=tensor((rows, value_heads)),
        b=tensor((rows, value_heads)),
        a_log=tensor((value_heads,), torch.float32),
        dt_bias=tensor((value_heads,), bias_dtype),
        checkpoint=tensor((1, value_heads, 128, 128)),
        ring=ring,
        key_heads=key_heads,
        value_heads=value_heads,
        key_dim=128,
        value_dim=128,
        tokens_per_request=4,
        raw_out=tensor((rows, value_heads, 128)),
        z=tensor((rows, value_heads, 128)) if fused else None,
        norm_weight=tensor((128,)) if fused else None,
        normalized_out=tensor((rows, value_heads, 128)) if fused else None,
        epsilon=0.000003,
    )


@pytest.fixture
def lookup(monkeypatch):
    calls = []
    writes = []

    def find(signature, *, prepared):
        calls.append((signature, prepared))
        return writes.append if prepared else None

    monkeypatch.setattr(numeric_binding, "numeric_operator", find)
    return calls, writes


@pytest.mark.parametrize("rows", [4, 8, 12])
@pytest.mark.parametrize("heads", [(4, 12), (16, 48)])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_exact_variant_caller_identity_no_execution_at_preflight(
    lookup, rows, heads, fused, bias_dtype
):
    calls, writes = lookup
    args = request(rows, *heads, fused=fused, bias_dtype=bias_dtype)
    bound = bind_gdn_core(**args)
    assert bound is not None and writes == []
    signature, prepared = calls[0]
    assert prepared and signature.variant == int(fused)
    assert signature.gate_scalar_type == ScalarType.BF16
    run, io = bound
    assert io.z is args["z"] and io.norm_weight is args["norm_weight"]
    assert io.normalized_out is args["normalized_out"]
    assert io.raw_out is args["raw_out"] and io.ring is args["ring"]
    assert io.dt_bias is args["dt_bias"] and io.dt_bias.dtype == bias_dtype
    assert io.epsilon == args["epsilon"] and io.value_rows == 128
    # Nothing has compiled or written; choosing numeric variant1 is not a raw
    # FLAT_STORES compile-policy change, and dt_bias dtype is retained in IO.
    run(io)
    assert writes == [io]


@pytest.mark.parametrize("absent", ["z", "norm_weight", "normalized_out"])
def test_partial_norm_request_never_reverts_to_raw(lookup, absent):
    args = request()
    args[absent] = None
    assert bind_gdn_core(**args) is None
    assert lookup == ([], [])


@pytest.mark.parametrize(
    "field,shape,dtype",
    [
        ("z", (4, 12, 128), torch.float32),
        ("norm_weight", (128,), torch.float32),
        ("normalized_out", (4, 12, 128), torch.float32),
        ("z", (4, 12, 64), torch.bfloat16),
        ("norm_weight", (12, 128), torch.bfloat16),
        ("checkpoint", (1, 12, 128, 128), torch.float32),
        ("a", (4, 12), torch.float32),
        ("b", (4, 12), torch.float32),
    ],
)
def test_unsupported_norm_view_is_refused_before_writes(lookup, field, shape, dtype):
    args = request()
    args[field] = tensor(shape, dtype)
    assert bind_gdn_core(**args) is None
    assert lookup[1] == [] and lookup[0][0][1] is False


def test_source_packed_z_row_stride_is_preserved(lookup):
    args = request()
    packed = tensor((4, 12 * 128 + 64))
    args["z"] = packed[:, : 12 * 128].view(4, 12, 128)
    assert not args["z"].is_contiguous()
    run, io = bind_gdn_core(**args)
    assert io.z is args["z"] and io.z.stride(0) == 1600
    assert lookup[1] == []


def test_missing_qkv_or_low_part_refuses_before_provider(lookup):
    args = request()
    assert bind_gdn_core(**(args | {"qkv": None})) is None
    assert lookup == ([], [])
    args["ring"] = replace(args["ring"], raw_k=None)
    assert bind_gdn_core(**args) is None
    assert not lookup[0][0][1] and lookup[1] == []


def test_real_registry_facade_norm_and_off_paths(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding

    args = request()
    writes = []
    registry = NumericRegistry()
    signature = GdnSignature(
        NumericOperation.GDN_CORE,
        4,
        4,
        12,
        128,
        128,
        4,
        ScalarType.BF16,
        variant=1,
        gate_scalar_type=ScalarType.BF16,
    )
    registry.register(NumericOffer(signature, writes.append))
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    bound = bind_gdn_core(**args)
    assert bound is not None and writes == []
    run, io = bound
    run(io)
    assert writes == [io]
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "0")
    assert bind_gdn_core(**args) is None and writes == [io]
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.setattr(numeric_binding, "numeric_registry", NumericRegistry())
    assert bind_gdn_core(**args) is None and writes == [io]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
