# ARIA (infra/assets/agentx), sourced by the pin's srt_agentic.sh right after its benchmark_lib.sh (agentx.py hook).
# InferenceX's write_agentic_result_json runs unchanged, under errexit as before, under a new name; when it fails, agentx.py
# decides: unreadable per-request records with aiperf's aggregate complete are recorded (agentx-defect.json) and the
# client goes on to InferenceX's validation and exit codes; any other failure stays the client's.
_ARIA_AGENTX=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
eval "_aria_infx_$(declare -f write_agentic_result_json)" \
  || { echo "ERROR: ARIA agentx: InferenceX's benchmark_lib.sh defines no write_agentic_result_json" >&2; exit 1; }
write_agentic_result_json() {
    local rc opts=$-
    set +e
    ( set -e; _aria_infx_write_agentic_result_json "$@" )  # not in a || list, so errexit holds inside
    rc=$?
    [[ $opts != *e* ]] || set -e
    ((rc != 0)) || return 0
    python3 "$_ARIA_AGENTX/agentx.py" defect "$1" "${AGENTIC_OUTPUT_DIR:-$INFMAX_CONTAINER_WORKSPACE}" "${RESULT_FILENAME:-}" "$rc"
}
# InferenceX's install_agentic_deps, unchanged under a new name, installs aiperf from the tree agentx.py aiperf prints: the
# pin's utils/aiperf with upstream's fix for SPAWN subagents' routing (agentx.py AIPERF_FIXES; 28263/28264 host OOM), copied
# to this node; then the venv's aiperf must carry it.
eval "_aria_agentx_$(declare -f install_agentic_deps)" \
  || { echo "ERROR: ARIA agentx: InferenceX's benchmark_lib.sh defines no install_agentic_deps" >&2; exit 1; }
install_agentic_deps() {
    [[ "${AIPERF_DEPS_READY:-}" != 1 ]] || return 0
    AIPERF_DIR=$(python3 "$_ARIA_AGENTX/agentx.py" aiperf "$AIPERF_DIR" "$AIPERF_RUNTIME_DIR/aiperf") || return $?
    _aria_agentx_install_agentic_deps "$@" || return $?
    "$AIPERF_PYTHON" "$_ARIA_AGENTX/agentx.py" aiperf-check
}
# InferenceX's run_server_client, unchanged under a new name; an aiperf replay runs beside agentx.py watch, which copies its
# records to shared storage and bounds its wait for its other result domains after the profile results (in one run, server
# metrics until the job's limit), and copies aiperf's export once it has exited (up to 330 s more here, then it is ended).
eval "_aria_infx_$(declare -f run_server_client)" \
  || { echo "ERROR: ARIA agentx: InferenceX's benchmark_lib.sh defines no run_server_client" >&2; exit 1; }
run_server_client() {
    [[ "${1##*/}" == aiperf && "${2:-}" == profile ]] || { _aria_infx_run_server_client "$@"; return; }
    local shell=$BASHPID watch rc  # this shell's: in the background command it would be the watcher's own
    python3 "$_ARIA_AGENTX/agentx.py" watch "$shell" "$@" &
    watch=$!
    _aria_infx_run_server_client "$@" && rc=0 || rc=$?
    local i
    for ((i = 0; i < 330; i++)); do kill -0 "$watch" 2>/dev/null || break; sleep 1; done
    kill "$watch" 2>/dev/null
    wait "$watch" 2>/dev/null || :
    return "$rc"
}
