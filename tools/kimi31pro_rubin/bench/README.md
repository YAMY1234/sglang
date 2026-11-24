# AgentX benchmark wrapper

The measured run used InferenceX's AgentX client, `inferencex-e2e/benchmarks/srt_agentic.sh` from https://github.com/SemiAnalysisAI/InferenceX at `1570a54f18398e742342ae00eda3924774a42959`, through this wrapper:

- `agentx.py hook` copies `srt_agentic.sh` and adds one line that sources `agentx.sh`.
- `agentx.sh` keeps InferenceX's own functions and their logic. It installs aiperf (InferenceX's pin, `SemiAnalysisAI/aiperf` `754356e9`) with `src/aiperf/credit/sticky_router.py` replaced by the upstream fix in `aiperf-9adff5e4/` ("fix(router): pin SPAWN subagents under their own id"). It also records a run whose per-request file is unreadable while aiperf's aggregate is complete, and bounds the wait for server metrics after the profile results.

```bash
INFX=<WORK_DIR>/InferenceX/inferencex-e2e      # checkout of 1570a54f18398e742342ae00eda3924774a42959
python3 agentx.py hook $INFX/benchmarks/srt_agentic.sh <WORK_DIR>/srt_agentic.sh
export MODEL=moonshotai/Kimi-K3.1-Pro CONC=930 DURATION=3600 THINKING_MODE=thinking_on SCENARIO_TYPE=agentic-coding IS_AGENTIC=1 \
  KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache TOTAL_CPU_DRAM_GB=12472 RESULT_FILENAME=agentx_c930 \
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang: AIPERF_MAX_CONTEXT_LENGTH=996579 \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true ARIA_RESULT_WAIT_S=720
INFMAX_CONTAINER_WORKSPACE=$INFX bash <WORK_DIR>/srt_agentic.sh
```

Run `../decode_warmup/d_warmup.py --dp 4` first. `srt_agentic.sh` also reads the frontend address and the remaining InferenceX inputs (`SPEC_DECODING`, `GPU_COUNT`, worker counts and so on) from the environment; see the script's header.
