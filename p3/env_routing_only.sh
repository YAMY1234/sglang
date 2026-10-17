#!/usr/bin/env bash
# Source in P only, after the frozen PD environment:
#   source p3/env_routing_only.sh 0n   # old boundary route
#   source p3/env_routing_only.sh 0    # full-N routing candidate
# Same flags as the scored 0n/0 micro arms. Do not source env.sh afterwards.
case "${1:-0}" in
    0n)
        export SGLANG_GDN_PD_SHALLOW_FULLN=0
        export TWINSTAR_PD_EMITTER_GRAPH=0
        ;;
    0)
        export SGLANG_GDN_PD_SHALLOW_FULLN=1
        export TWINSTAR_PD_EMITTER_GRAPH=1
        ;;
    *)
        echo 'routing-only arm must be 0n or 0' >&2
        return 2
        ;;
esac
export SGLANG_GDN_PREFILL_AGG_CONTRACT=1
export SGLANG_GDN_K31_FUSED=0
export SGLANG_GDN_K31_CHOLQR_MIXED=0
export SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED=0
export SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD=0
export SGLANG_GDN_FACTORED_HOST_SYNC_FREE=1
# The P trunk graph has a distinct unqualified ForwardBatch copy path.
export SGLANG_FLASHNEXT_PD_TRUNK_PREFILL_GRAPH=0
