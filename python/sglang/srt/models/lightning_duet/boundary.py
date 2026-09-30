"""Prompt/logprob handoff planning, without GPU or SGLang dependencies."""


def prefill_count(length, logprob_start=None):
    if length < 1:
        raise ValueError("Lightning DUET requires a nonempty prompt")
    if logprob_start is not None:
        if not 0 <= logprob_start < length:
            raise ValueError("DUET input logprobs require 0 <= logprob_start_len < prompt length")
        return logprob_start
    return length - 1


def mark_runner_dummy_batches(model_runner):
    """Use the runner's explicit dummy-batch hook, never infer from slot IDs."""
    if getattr(model_runner, "_lightning_dummy_marker_installed", False):
        return
    original = model_runner.prepare_dummy_forward_batch

    def prepare(batch):
        batch = original(batch)
        batch._lightning_dummy_batch = True
        return batch

    model_runner.prepare_dummy_forward_batch = prepare
    model_runner._lightning_dummy_marker_installed = True
