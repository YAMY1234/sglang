"""Explicit host-provider doubles for binding-only source-domain tests.

No plugin, runtime packet reader, CUDA allocation or raw kernel is imported.
The real provider independently tests the same immutable capability contract.
"""

from types import SimpleNamespace as NS

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as Op


def mtp_capability(query):
    a = dict(query.attributes)
    if query.operation == Op.MTP_CAPTURE_SEED:
        return (
            all(
                a.get(k) == v
                for k, v in dict(
                    tp_size=4,
                    sm100_supported=1,
                    is_draft_worker=1,
                    indexer_topk=2048,
                    compress_ratio=4,
                    tail_width=4,
                    tokens_per_request=4,
                    rows=4,
                    columns=2051,
                    seed_columns=2055,
                    front_tokens=0,
                    extend_scalar_type=5,
                    accept_scalar_type=5,
                    request_scalar_type=4,
                    length_scalar_type=5,
                ).items()
            )
            and a.get("trash_row", 0) > 0
        )
    if query.operation == Op.MTP_DEFERRED_KV_PREPARE:
        return all(
            a.get(k) == v
            for k, v in dict(
                tp_size=4,
                sm100_supported=1,
                is_draft_worker=1,
                hidden_size=2560,
                hc_count=4,
                speculative_num_steps=3,
                speculative_eagle_topk=1,
                indexer_topk=2048,
                compress_ratio=4,
                query_heads=6,
                tokens_per_request=4,
                rows=1,
                kv_heads=1,
                head_dim=256,
                scalar_type=1,
                index_columns=2055,
                packed_columns=2112,
                tail_width=4,
                new_k_row_stride=256,
                new_v_row_stride=256,
            ).items()
        ) and a.get("metadata_dtype_mask") in (17, 25)
    return False


def mtp_spec():
    return NS(numeric_capability=mtp_capability)


def contract_geometry(s):
    """Finite host-provider double; no kernel or production plugin imports."""
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        ExpertSignature,
        FinalizePermutedSignature,
        GdnSignature,
        HcSignature,
        NativeGdnPdlSignature,
        NumericSignature,
        RouterQuantSignature,
    )

    if s.operation == Op.PACKED_PREFILL_ATTENTION:
        return (
            s.rows == s.rotary_dim == s.context_bound == s.ctas_per_group == 0
            and 0 < s.kv_heads
            and s.query_heads % s.kv_heads == 0
            and s.query_heads // s.kv_heads <= 16
            and (s.head_dim, s.selected_count, s.compress_ratio, s.scalar_type)
            == (256, 2051, 4, 1)
            and 1 <= s.variant <= 8
        )
    if s.operation == Op.PREFILL_SELECT:
        return (
            s.rows,
            s.query_heads,
            s.kv_heads,
            s.head_dim,
            s.selected_count,
            s.compress_ratio,
            s.scalar_type,
            s.rotary_dim,
            s.variant,
            s.ctas_per_group,
        ) == (0, 4, 1, 128, 512, 4, 1, 0, 0, 0) and 0 < s.context_bound <= 131072
    if s.rows <= 0:
        return False
    if isinstance(s, HcSignature):
        return (
            s.stream_count == 4
            and s.hidden_size == 2560
            and s.scalar_type == 1
            and (s.operation == Op.HC_MIX or (s.rows <= 32 and s.parts in (1, 2, 4)))
        )
    if isinstance(s, GdnSignature):
        if (
            (s.key_heads, s.value_heads) not in ((4, 12), (16, 48))
            or s.key_dim != 128
            or s.value_dim != 128
            or s.tokens_per_request != 4
            or s.scalar_type != 1
        ):
            return False
        if s.operation == Op.GDN_FRONT:
            return (
                (
                    s.rows in (4, 8, 12)
                    or (s.rows == 16 and (s.key_heads, s.value_heads) == (4, 12))
                )
                and s.gate_scalar_type == 3
                and s.variant in (0, 1)
            )
        if s.operation == Op.GDN_VERIFY_GLUE:
            return s.rows in (8, 12, 16) and s.gate_scalar_type == 1
        return (
            s.rows in (4, 8, 12)
            or (
                (s.key_heads, s.value_heads) == (4, 12)
                and (s.rows == 16 or (s.rows == 32 and s.variant == 1))
            )
        ) and s.gate_scalar_type == 1
    if isinstance(s, NumericSignature):
        if s.operation in (Op.SPARSE_SCORE, Op.SPARSE_SELECT):
            return (
                (
                    s.kv_heads,
                    s.head_dim,
                    s.selected_count,
                    s.compress_ratio,
                    s.rotary_dim,
                )
                == (1, 128, 512, 4, 0)
                and 0 < s.context_bound <= 131072
                and s.query_heads in (8, 16)
                and (s.ctas_per_group > 0) == (s.variant in (6, 7))
                and (
                    (
                        s.operation == Op.SPARSE_SELECT
                        and s.scalar_type == 3
                        and (
                            (s.variant == 0 and s.rows == 4 and s.query_heads == 8)
                            or (s.variant == 1 and s.rows <= 64)
                        )
                    )
                    or (
                        s.operation == Op.SPARSE_SCORE
                        and s.scalar_type == 1
                        and (
                            (s.variant in (0, 3) and s.rows == 4 and s.query_heads == 8)
                            or (s.variant == 4 and s.rows <= 64 and s.rows % 4 == 0)
                            or (s.variant == 5 and s.rows <= 64)
                            or (
                                s.variant in (6, 7)
                                and 64 < s.rows <= 576
                                and s.rows % 4 == 0
                                and s.query_heads == 8
                            )
                        )
                    )
                )
            )
        if s.operation == Op.POOL_UNION_ATTENTION:
            return (
                (s.query_heads, s.kv_heads, s.head_dim, s.selected_count)
                == (24, 2, 256, 512)
                and (s.compress_ratio, s.scalar_type, s.variant) == (4, 1, 0)
                and 64 <= s.rows <= 576
                and s.rows % 4 == 0
            )
        return (
            s.operation == Op.DIRECT_POOL_ATTENTION
            and s.head_dim == 256
            and s.selected_count == 512
            and s.compress_ratio == 4
            and s.scalar_type == 1
            and (
                (
                    (s.query_heads, s.kv_heads) == (24, 2)
                    and (
                        (s.variant == 1 and s.rows <= 64 and s.rows % 4 == 0)
                        or (s.variant == 2 and s.rows <= 64)
                    )
                )
                or (
                    (s.query_heads, s.kv_heads) == (6, 1)
                    and s.variant == 0
                    and s.rows == 4
                )
            )
        )
    if isinstance(s, RouterQuantSignature):
        return (
            s.hidden_size == 2560
            and s.expert_count == 512
            and s.scalar_type == 1
            and (
                (s.rows in (1, 4) and s.variant in (0, 1))
                or (s.rows in (8, 16) and s.variant == 2)
            )
        )
    if isinstance(s, FinalizePermutedSignature):
        return (
            s.hidden_size == 2560
            and s.topk == 10
            and s.rows <= 64
            and s.scalar_type == 1
            and s.weight_scalar_type in (1, 3)
        )
    if isinstance(s, NativeGdnPdlSignature):
        return (
            (s.key_heads, s.value_heads) in ((4, 12), (16, 48))
            and s.key_dim == s.value_dim == 128
            and s.rows % 4 == 0
            and s.tokens_per_request == 4
            and s.cache_len == 16
            and s.scalar_type == 1
        )
    if isinstance(s, ExpertSignature):
        if s.operation == Op.FINALIZE_SHARED:
            return (
                s.hidden_size == 2560
                and s.topk == 10
                and s.scalar_type == 1
                and s.variant == 0
            )
        return (
            s.hidden_size == 2560
            and s.intermediate_size == 192
            and s.experts == 512
            and s.topk == 10
            and s.scalar_type == 6
            and (
                (s.variant == 0 and s.rows in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16))
                or (s.variant == 1 and s.rows in (1, 4))
                or (s.variant == 2 and s.rows in (12, 16))
            )
        )
    return False


# Explicit frozen contract-provider fixture; never selected implicitly by production.
from dataclasses import fields

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ExpertSignature,
    FinalizePermutedSignature,
    GdnFrontVariant,
    GdnSignature,
    HcSignature,
    MtpCaptureSeedSignature,
    MtpDeferredKvSignature,
    NativeGdnPdlSignature,
    NumericCapabilityQuery,
    NumericOperation,
    NumericSignature,
    RouterQuantSignature,
)

_FIXED_CONTEXTS = {
    NumericOperation.MTP_CAPTURE_SEED: {
        "tp_size": 4,
        "sm100_supported": 1,
        "is_draft_worker": 1,
        "indexer_topk": 2048,
        "compress_ratio": 4,
        "tail_width": 4,
        "tokens_per_request": 4,
    },
    NumericOperation.MTP_DEFERRED_KV_PREPARE: {
        "tp_size": 4,
        "sm100_supported": 1,
        "is_draft_worker": 1,
        "hidden_size": 2560,
        "hc_count": 4,
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "indexer_topk": 2048,
        "compress_ratio": 4,
        "query_heads": 6,
        "tokens_per_request": 4,
    },
    NumericOperation.FINALIZE_PERMUTED_SHARED: {
        "tp_size": 1,
        "ep_size": 1,
        "expert_count": 512,
    },
}

_GDN_CONTEXT = (
    "tp_size",
    "hidden_size",
    "wide_enabled",
    "m32_norm_enabled",
    "projection_norm_available",
    "m32_norm_available",
    "requested_front_stages",
    "front_stages_specified",
)

_SPARSE_CONTEXT = ("tp_size", "sm100_supported", "page_size", "unpadded_query_heads")

_SCHEMAS = {
    **{
        op: (HcSignature, ())
        for op in (
            NumericOperation.HC_GATE,
            NumericOperation.HC_APPLY_NORM,
            NumericOperation.HC_APPLY_NORM_GATE,
            NumericOperation.HC_MIX,
        )
    },
    **{
        op: (GdnSignature, _GDN_CONTEXT)
        for op in (
            NumericOperation.GDN_FRONT,
            NumericOperation.GDN_CORE,
            NumericOperation.GDN_VERIFY_GLUE,
        )
    },
    **{
        op: (NumericSignature, _SPARSE_CONTEXT)
        for op in (
            NumericOperation.SPARSE_SCORE,
            NumericOperation.SPARSE_SELECT,
            NumericOperation.PREFILL_SELECT,
        )
    },
    NumericOperation.GDN_NATIVE_PDL: (NativeGdnPdlSignature, ()),
    NumericOperation.DIRECT_POOL_ATTENTION: (NumericSignature, ("tp_size",)),
    NumericOperation.POOL_UNION_ATTENTION: (NumericSignature, ("tp_size",)),
    NumericOperation.PACKED_PREFILL_ATTENTION: (NumericSignature, ("tp_size",)),
    NumericOperation.LOCAL_EXPERTS: (
        ExpertSignature,
        (
            "tokens_per_request",
            "sf_pack_enabled",
            "mkm_wide_enabled",
            "mkm_wide_m8_enabled",
            "wide_batch34_enabled",
            "is_target_verify",
            "batch_size",
        ),
    ),
    NumericOperation.FINALIZE_SHARED: (ExpertSignature, ()),
    NumericOperation.FINALIZE_PERMUTED_SHARED: (
        FinalizePermutedSignature,
        tuple(_FIXED_CONTEXTS[NumericOperation.FINALIZE_PERMUTED_SHARED]),
    ),
    NumericOperation.ROUTER_QUANT: (
        RouterQuantSignature,
        ("tp_size", "ep_size", "wide_enabled", "k4_enabled"),
    ),
    NumericOperation.MTP_CAPTURE_SEED: (
        MtpCaptureSeedSignature,
        tuple(_FIXED_CONTEXTS[NumericOperation.MTP_CAPTURE_SEED]),
    ),
    NumericOperation.MTP_DEFERRED_KV_PREPARE: (
        MtpDeferredKvSignature,
        tuple(_FIXED_CONTEXTS[NumericOperation.MTP_DEFERRED_KV_PREPARE]),
    ),
}
_SCHEMAS[NumericOperation.GDN_CORE] = (
    GdnSignature,
    _GDN_CONTEXT + ("normalization_program",),
)


def _source_supported(s, actual):
    op = s.operation
    if op == NumericOperation.HC_MIX:
        return s.rows <= 32
    if op == NumericOperation.PACKED_PREFILL_ATTENTION:
        return actual["tp_size"] > 0
    if op == NumericOperation.PREFILL_SELECT:
        return (
            actual["tp_size"] > 0
            and actual["sm100_supported"] == 1
            and actual["unpadded_query_heads"] == s.query_heads
        )
    fixed = _FIXED_CONTEXTS.get(op, {})
    if any(actual[name] != value for name, value in fixed.items()):
        return False
    if op == NumericOperation.MTP_CAPTURE_SEED:
        return s.trash_row > 0
    if op == NumericOperation.MTP_DEFERRED_KV_PREPARE:
        # The graph producer is narrower than synthetic raw-layout coverage.
        return (
            s.new_k_row_stride == s.new_v_row_stride == 256
            and s.metadata_dtype_mask in (17, 25)
        )
    if isinstance(s, GdnSignature):
        if op == NumericOperation.GDN_CORE:
            program = actual["normalization_program"]
            if s.variant == 0 and program != 0:
                return False
            if s.variant == 1:
                if program == 1 and s.rows == 32:
                    return False
                if program == 2 and s.rows != 32:
                    return False
                if program not in (1, 2):
                    return False
        tp = actual["tp_size"]
        flags = (
            "wide_enabled",
            "m32_norm_enabled",
            "projection_norm_available",
            "m32_norm_available",
            "front_stages_specified",
        )
        if (
            tp <= 0
            or actual["hidden_size"] <= 0
            or any(actual[name] not in (0, 1) for name in flags)
            or (
                not actual["front_stages_specified"]
                and actual["requested_front_stages"] != 0
            )
        ):
            return False
        if op == NumericOperation.GDN_VERIFY_GLUE:
            return tp == 1
        if s.rows == 32:
            return (
                op == NumericOperation.GDN_CORE
                and s.variant == 1
                and tp == 4
                and actual["m32_norm_enabled"] == 1
                and actual["m32_norm_available"] == 1
            )
        if s.rows == 16 and not (tp == 4 and actual["wide_enabled"] == 1):
            return False
        if op == NumericOperation.GDN_FRONT or s.variant == 1:
            if tp not in (1, 4) or actual["projection_norm_available"] != 1:
                return False
            channels = 2 * s.key_heads * s.key_dim + s.value_heads * s.value_dim
            stages = (
                actual["requested_front_stages"]
                if s.rows == 4 and channels == 2560 and actual["front_stages_specified"]
                else 6
            )
            if stages not in (6, 12):
                return False
            if op == NumericOperation.GDN_FRONT:
                variant = (
                    GdnFrontVariant.MK_STAGES_6
                    if stages == 6
                    else GdnFrontVariant.MK_STAGES_12
                )
                return actual["hidden_size"] == 2560 and s.variant == variant
        return True
    if op in (NumericOperation.SPARSE_SCORE, NumericOperation.SPARSE_SELECT):
        if (
            actual["tp_size"] <= 0
            or actual["sm100_supported"] != 1
            or actual["page_size"] != 16
        ):
            return False
        heads = actual["unpadded_query_heads"]
        if not 0 < heads <= 16 or s.query_heads != ((heads + 7) // 8) * 8:
            return False
        if (
            (op == NumericOperation.SPARSE_SCORE and s.variant in (0, 3, 6, 7))
            or (op == NumericOperation.SPARSE_SELECT and s.variant == 0)
        ) and heads != 4:
            return False
        if (op == NumericOperation.SPARSE_SCORE and s.variant in (4, 5, 6, 7)) or (
            op == NumericOperation.SPARSE_SELECT and s.variant == 1
        ):
            return actual["tp_size"] == 1
        return True
    if op == NumericOperation.DIRECT_POOL_ATTENTION:
        return actual["tp_size"] == (4 if s.variant == 0 else 1)
    if op == NumericOperation.POOL_UNION_ATTENTION:
        return actual["tp_size"] == 1
    if op == NumericOperation.LOCAL_EXPERTS:
        flags = (
            "sf_pack_enabled",
            "mkm_wide_enabled",
            "mkm_wide_m8_enabled",
            "wide_batch34_enabled",
            "is_target_verify",
        )
        if (
            any(actual[key] not in (0, 1) for key in flags)
            or actual["tokens_per_request"] <= 0
            or actual["batch_size"] < 0
        ):
            return False
        if s.rows <= 7:
            return s.variant == int(s.rows in (1, 4) and actual["sf_pack_enabled"])
        if not actual["mkm_wide_enabled"]:
            return False
        if actual["wide_batch34_enabled"]:
            return (
                s.variant == 2
                and s.rows in (12, 16)
                and actual["tokens_per_request"] == 4
                and actual["is_target_verify"] == 1
                and actual["batch_size"] in (3, 4)
                and s.rows == 4 * actual["batch_size"]
            )
        return s.variant == 0 and (s.rows != 8 or actual["mkm_wide_m8_enabled"] == 1)
    if op == NumericOperation.ROUTER_QUANT:
        if (
            actual["tp_size"] != 4
            or actual["ep_size"] != 1
            or actual["wide_enabled"] not in (0, 1)
            or actual["k4_enabled"] not in (0, 1)
        ):
            return False
        if s.rows in (1, 4):
            return s.variant == actual["k4_enabled"]
        return actual["wide_enabled"] == 1 and s.variant == 2
    return True


def numeric_capability(query):
    """Answer from host facts only, before registry entries or pools are built."""
    if not isinstance(query, NumericCapabilityQuery):
        return False
    schema = _SCHEMAS.get(query.operation)
    if schema is None:
        return False
    signature_type, context = schema
    names = {field.name for field in fields(signature_type)} - {"operation"}
    actual = dict(query.attributes)
    if actual.keys() != names | set(context):
        return False
    signature = signature_type(
        operation=query.operation,
        **{name: actual[name] for name in names},
    )
    return contract_geometry(signature) and _source_supported(signature, actual)


def contract_spec():
    return NS(numeric_capability=numeric_capability)
