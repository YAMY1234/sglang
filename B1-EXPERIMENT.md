# PP PD B1 input preparation experiment (partial implementation)

Base 043cfc4d + GB300 production overlay 7c4ed474. The default-off
SGLANG_PP_PD_OVERLAP_B1 switch prepares immutable pinned token storage for
one disjoint fresh text request after the current forward is submitted.
Admission, prefix/cache matching, KV/row allocation, HiCache, batch metadata
and sampling remain at the original scheduler boundary after the old result.
The next input generation is validated at admission; old output slot generation
is validated before result consumption. The prepared backing remains owned by
the live batch and PPBatchMetadata through result consumption.

This is not a complete Increment B implementation: multi-request batches and
continuing chunks fall back, and scheduling/positions/lengths/sampling are not
prepared ahead. Runtime results qualify only this narrower experiment.

CPU contract test:
python3 test/registered/unit/managers/test_pp_pd_lookahead.py -v

Required runtime gates: actual committed-generation logs on the participating
PP ranks; identical first token and top5 logprobs; >=200 GSM8K questions with
no loss; matched-work AgentX OFF/ON/OFF windows. No GPU correctness or speed
claim follows from CPU tests.
