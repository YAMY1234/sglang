#!/usr/bin/env bash
# Source in the P process only, after the frozen v7 factor recipe.
# Requires --disable-overlap-schedule. D keeps the frozen recipe.
export SGLANG_GDN_K31_FUSED=1
export SGLANG_GDN_K31_CHOLQR_MIXED=1
export SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED=1
export SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD=1
export SGLANG_GDN_FACTORED_HOST_SYNC_FREE=1
