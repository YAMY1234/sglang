#!/usr/bin/env bash
# lead 22:55 PD statistical gate: 0n control / full-N candidate, both R1 off.
# Source in P only after the frozen PD environment. The candidate also enables
# the qualified deferred publication/fence path, so its activation is observable.
# This service arm is "0 + deferred"; r7's route 0 timing used forward publish.
source "$(dirname "${BASH_SOURCE[0]}")/env_routing_only.sh" "${1:-0}" || return
export SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED=1
export SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD=1
