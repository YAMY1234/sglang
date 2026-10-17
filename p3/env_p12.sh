#!/usr/bin/env bash
# Source in P after the frozen, qualified PC+ route0/deferred recipe.
# Both arms keep that recipe verbatim; this changes only P12's local schedule.
case "${1:-1}" in
    0|1) P12_ARM=${1:-1} ;;
    *) echo 'P12 arm must be 0 or 1' >&2; return 2 ;;
esac
if [[ ${SGLANG_GDN_PD_SHALLOW_FULLN:-0} != 1 ||
      ${SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED:-0} != 1 ||
      ${SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD:-0} != 1 ||
      ${SGLANG_GDN_K31_FUSED:-0} != 0 ||
      ${SGLANG_GDN_K31_CHOLQR_MIXED:-0} != 0 ]]; then
    echo 'P12 requires the qualified R1-off shallow full-N/deferred/JOIN P recipe' >&2
    unset P12_ARM
    return 2
fi
export SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE=$P12_ARM
unset P12_ARM
