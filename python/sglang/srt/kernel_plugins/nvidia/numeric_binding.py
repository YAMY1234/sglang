"""Bind declared local kernels to framework-prepared views, with stock fallback."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
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
    PoolUnionAttentionIO,
    PrefillSelectIO,
    RouterQuantIO,
    RouterQuantSignature,
    SparseScoreIO,
    SparseSelectIO,
    numeric_registry,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_OFFER_LOG = envs.SGLANG_NVIDIA_MEGAKERNEL_OFFER_LOG.get()
_offer_log_keys: set = set()


def _log_offer(key, message: str, *args) -> None:
    if key not in _offer_log_keys:
        _offer_log_keys.add(key)
        logger.info("NVIDIA megakernel offer: " + message, *args)


_IO_TYPES = {
    NumericOperation.MTP_CAPTURE_SEED: MtpCaptureSeedIO,
    NumericOperation.MTP_DEFERRED_KV_PREPARE: MtpDeferredKvIO,
    NumericOperation.GDN_NATIVE_PDL: NativeGdnPdlIO,
    NumericOperation.ROUTER_QUANT: RouterQuantIO,
    NumericOperation.SPARSE_SCORE: SparseScoreIO,
    NumericOperation.SPARSE_SELECT: SparseSelectIO,
    NumericOperation.PREFILL_SELECT: PrefillSelectIO,
    NumericOperation.DIRECT_POOL_ATTENTION: DirectPoolAttentionIO,
    NumericOperation.POOL_UNION_ATTENTION: PoolUnionAttentionIO,
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
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled

    if not (
        megakernel_enabled() and envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()
    ):
        return _capability_result(query, False, "plugin or runtime master off")
    callback = None if spec is None else getattr(spec, "numeric_capability", None)
    if callback is None:
        return _capability_result(query, False, "spec has no capability callback")
    supported = callback(query)
    if type(supported) is not bool:
        raise TypeError("numeric capability callback must return bool")
    return _capability_result(
        query, supported, "accepted" if supported else "rejected by the plugin"
    )


def _capability_result(
    query: NumericCapabilityQuery, supported: bool, why: str
) -> bool:
    if _OFFER_LOG:
        _log_offer(
            query,
            "capability %s %s: %s",
            query.operation.name,
            dict(query.attributes),
            why,
        )
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
        if _OFFER_LOG:
            _log_dispatch(signature, "stock: view not prepared or operation unbound")
        return None
    offer = numeric_registry.get(signature)
    if offer is None:
        if _OFFER_LOG:
            _log_dispatch(signature, "stock: no provider offer")
        return None
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_enabled

    if not (
        megakernel_enabled() and envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()
    ):
        if _OFFER_LOG:
            _log_dispatch(signature, "stock: plugin or runtime master off")
        return None
    if _OFFER_LOG:
        _log_dispatch(signature, "provider bound")
    return bind_numeric_operator(
        enabled=True,
        signature=signature,
        prepared=(signature,),
        offers=(offer,),
        stock=lambda io: None,
    )


def _log_dispatch(signature: OperatorSignature, why: str) -> None:
    _log_offer(("dispatch", signature), "dispatch %s: %s", signature, why)


def log_numeric_offers(
    model, declarations: Mapping[object, tuple[OperatorSignature, ...]], *, role: str
) -> None:
    """Per (role, layer, op), the declared rows with and without a provider offer.

    Runs after the plugin registered its offers; a producer that declared nothing
    stopped at its own source gates (env, TP size, shape).
    """
    for name, producer in model.named_modules():
        declared = declarations.get(producer)
        if declared is None:
            continue
        if not declared:
            _log_offer((role, name), "%s %s: declared nothing", role, name)
        rows = {}
        for signature in declared:
            offered = numeric_registry.get(signature) is not None
            rows.setdefault(signature.operation.name, ([], []))[offered].append(
                getattr(signature, "rows", None)
            )
        for operation, (missing, offered) in rows.items():
            _log_offer(
                (role, name, operation),
                "%s %s %s: offered rows %s; no offer for rows %s",
                role,
                name,
                operation,
                offered,
                missing,
            )


def collect_numeric_signatures(
    producers: Collection[object],
    widths: tuple[int, ...],
    tokens_per_request: int,
    *,
    capability_spec=None,
    declarations: dict | None = None,
) -> tuple[OperatorSignature, ...]:
    """Read framework producers at startup; no producer object reaches a plugin.

    ``declarations``, when given, receives each producer's own declaration.
    """
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
            if declarations is not None:
                declarations[producer] = declared
    return tuple(signatures)
