"""Prompt/logprob handoff planning, without GPU or SGLang dependencies."""


def prefill_count(length, logprob_start=None):
    if length < 1:
        raise ValueError("Lightning DUET requires a nonempty prompt")
    if logprob_start is not None:
        if not 0 <= logprob_start < length:
            raise ValueError("DUET input logprobs require 0 <= logprob_start_len < prompt length")
        return logprob_start
    return length - 1
