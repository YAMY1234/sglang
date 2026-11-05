"""Bind declared local kernels to framework-prepared views, with stock fallback."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.kernels.plugin_kernels.nvidia.spec import MegakernelSpec

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    DirectPoolAttentionIO,
    ExpertSignature,
    FinalizePermutedSharedIO,
    FinalizePermutedSignature,
    FinalizeSharedIO,
    GdnCoreIO,
    GdnFrontIO,
    GdnSignature,
    GdnVerifyGlueIO,
    HcApplyNormGateIO,
    HcApplyNormIO,
    HcGateIO,
    HcMixIO,
    HcSignature,
    LocalExpertsIO,
    MtpCaptureSeedIO,
    MtpCaptureSeedSignature,
    MtpDeferredKvIO,
    MtpDeferredKvSignature,
    NativeGdnPdlIO,
    NativeGdnPdlSignature,
    NumericCapabilityQuery,
    NumericIO,
    NumericKernel,
    NumericMetadataProducer,
    NumericOffer,
    NumericOperation,
    NumericSignature,
    OperatorSignature,
    RouterQuantIO,
    RouterQuantSignature,
    SparseScoreIO,
    SparseSelectIO,
    numeric_registry,
)

_IO_TYPES = {
    NumericOperation.MTP_CAPTURE_SEED: MtpCaptureSeedIO,
    NumericOperation.MTP_DEFERRED_KV_PREPARE: MtpDeferredKvIO,
    NumericOperation.GDN_NATIVE_PDL: NativeGdnPdlIO,
    NumericOperation.ROUTER_QUANT: RouterQuantIO,
    NumericOperation.SPARSE_SCORE: SparseScoreIO,
    NumericOperation.SPARSE_SELECT: SparseSelectIO,
    NumericOperation.DIRECT_POOL_ATTENTION: DirectPoolAttentionIO,
    NumericOperation.HC_GATE: HcGateIO,
    NumericOperation.HC_APPLY_NORM: HcApplyNormIO,
    NumericOperation.HC_APPLY_NORM_GATE: HcApplyNormGateIO,
    NumericOperation.HC_MIX: HcMixIO,
    NumericOperation.GDN_FRONT: GdnFrontIO,
    NumericOperation.GDN_CORE: GdnCoreIO,
    NumericOperation.GDN_VERIFY_GLUE: GdnVerifyGlueIO,
    NumericOperation.LOCAL_EXPERTS: LocalExpertsIO,
    NumericOperation.FINALIZE_SHARED: FinalizeSharedIO,
    NumericOperation.FINALIZE_PERMUTED_SHARED: FinalizePermutedSharedIO,
}


def numeric_capability_supported(
    query: NumericCapabilityQuery, *, spec: MegakernelSpec | None
) -> bool:
    """Ask the already selected immutable spec before numeric registration.

    This is startup admission only. Forward calls still require an exact offer
    and complete caller-buffer preflight; they never resolve a spec here.
    """
    from sglang.srt.environ import envs
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled

    if not (
        megakernel_enabled() and envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()
    ):
        return False
    callback = None if spec is None else getattr(spec, "numeric_capability", None)
    if callback is None:
        return False
    supported = callback(query)
    if type(supported) is not bool:
        raise TypeError("numeric capability callback must return bool")
    return supported


def signature_capability_supported(signature, *, spec, **source_facts):
    """Combine actual signature metadata with host-resolved source facts."""
    attributes = tuple(
        (field.name, getattr(signature, field.name))
        for field in fields(signature)
        if field.name != "operation"
    ) + tuple(source_facts.items())
    return numeric_capability_supported(
        NumericCapabilityQuery(signature.operation, attributes),
        spec=spec,
    )


def bind_numeric_operator(
    *,
    enabled: bool,
    signature: OperatorSignature,
    prepared: Collection[OperatorSignature],
    offers: Sequence[NumericOffer],
    stock: NumericKernel,
) -> Callable[[NumericIO], None]:
    """Bind once after the framework adapter has established complete buffers.

    ``enabled`` is the framework's resolved plugin/master gate. ``prepared``
    contains only exact capabilities whose views the adapter can construct and
    refill at replay. An offer alone cannot make an unsupported backend eligible.
    No provider receives the adapter, a runner, or a framework module.
    """
    if not enabled or signature not in prepared or signature.operation not in _IO_TYPES:
        return stock
    matching = [offer.run for offer in offers if offer.signature == signature]
    if not matching:
        return stock
    if len(matching) != 1:
        raise ValueError("a numeric signature must have exactly one provider")
    kernel = matching[0]
    io_type = _IO_TYPES[signature.operation]

    def run(io: NumericIO) -> None:
        if not isinstance(io, io_type):
            raise TypeError("the numeric view does not match the bound operation")
        # Once a provider starts writing, refusal cannot safely replay stock.
        if kernel(io) is not None:
            raise TypeError(
                "a numeric provider must write caller outputs and return None"
            )

    return run


def numeric_operator(
    signature: OperatorSignature, *, prepared: bool
) -> NumericKernel | None:
    """Framework facade: no offer/off/incomplete metadata leaves stock untouched.

    Producers call this only after their ordinary eligibility checks and before
    any provider writes. Captured graphs retain the chosen callable and buffers;
    metadata refill and producer stream joins remain the producer's work.
    """
    if not prepared or signature.operation not in _IO_TYPES:
        return None
    offer = numeric_registry.get(signature)
    if offer is None:
        return None
    from sglang.srt.environ import envs
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled

    if not (
        megakernel_enabled() and envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()
    ):
        return None
    return bind_numeric_operator(
        enabled=True,
        signature=signature,
        prepared=(signature,),
        offers=(offer,),
        stock=lambda io: None,
    )


def collect_numeric_signatures(
    producers: Collection[object],
    widths: tuple[int, ...],
    tokens_per_request: int,
    *,
    capability_spec=None,
) -> tuple[OperatorSignature, ...]:
    """Read framework producers at startup; no producer object reaches a plugin."""
    signatures = {}
    for producer in producers:
        if isinstance(producer, NumericMetadataProducer):
            if capability_spec is not None:
                producer._numeric_capability_spec = capability_spec
            declared = producer.numeric_signatures(widths, tokens_per_request)
            if not isinstance(declared, tuple):
                raise TypeError("numeric declarations must be an immutable tuple")
            for signature in declared:
                if not isinstance(
                    signature,
                    (
                        NumericSignature,
                        HcSignature,
                        GdnSignature,
                        ExpertSignature,
                        FinalizePermutedSignature,
                        RouterQuantSignature,
                        NativeGdnPdlSignature,
                        MtpCaptureSeedSignature,
                        MtpDeferredKvSignature,
                    ),
                ) or not all(
                    isinstance(getattr(signature, field.name), int)
                    for field in fields(signature)
                ):
                    raise TypeError(
                        "numeric signatures may contain only numeric fields"
                    )
                if isinstance(signature, RouterQuantSignature) and any(
                    isinstance(getattr(signature, field.name), bool)
                    for field in fields(signature)
                ):
                    raise TypeError(
                        "router quant signatures cannot contain bool fields"
                    )
                if isinstance(signature, NativeGdnPdlSignature) and any(
                    isinstance(getattr(signature, field.name), bool)
                    for field in fields(signature)
                ):
                    raise TypeError("native GDN signatures cannot contain bool fields")
                if isinstance(signature, MtpCaptureSeedSignature) and any(
                    isinstance(getattr(signature, field.name), bool)
                    for field in fields(signature)
                ):
                    raise TypeError("MTP capture signatures cannot contain bool fields")
                if isinstance(signature, MtpDeferredKvSignature) and any(
                    isinstance(getattr(signature, field.name), bool)
                    for field in fields(signature)
                ):
                    raise TypeError("deferred KV signatures cannot contain bool fields")
                signatures[signature] = None
    return tuple(signatures)
