"""DeepSeek DSA kernels (RFC #29630, Phase 2.5)."""

from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    aiter_paged_mqa_logits_persistent_schedule,
    aiter_paged_mqa_logits_verify,
    can_use_cutedsl_paged_mqa_logits_cap8,
    cutedsl_paged_mqa_logits,
    cutedsl_paged_mqa_logits_cap8,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)


def pick_dsl_expand(*args, **kwargs):
    from .cutedsl_paged_mqa_logits import pick_dsl_expand as _pick_dsl_expand

    return _pick_dsl_expand(*args, **kwargs)


__all__ = [
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "aiter_paged_mqa_logits_persistent_schedule",
    "aiter_paged_mqa_logits_verify",
    "can_use_cutedsl_paged_mqa_logits_cap8",
    "cutedsl_paged_mqa_logits",
    "cutedsl_paged_mqa_logits_cap8",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]
