"""Same-image CPU gate: native chat conversion and ready-output consumption.

Uses archived AgentX messages and the exact input IDs observed by the server.
No model execution, GPU, network server or service-performance claim.
"""

import argparse
import asyncio
import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace


async def run(args):
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    from fastapi import Request
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager
    from sglang.srt.runtime_context import get_context, publish, reset_context
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    compressed = args.fixture.read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == args.fixture_sha256
    fixture = json.loads(gzip.decompress(compressed))
    assert len(fixture["points"]) == 6
    model_path = "/work/assets/views-k31u/model-lmo"
    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        dtype="bfloat16",
        device="cpu",
        trust_remote_code=True,
        reasoning_parser="qwen3",
        context_length=262144,
        disable_cuda_graph=True,
    )
    reset_context()
    publish(server_args, role="tokenizer")
    model_config = ModelConfig.from_server_args(server_args)
    tokenizer = get_tokenizer(model_path, tokenizer_mode="auto", trust_remote_code=True)
    manager = SimpleNamespace(
        server_args=server_args,
        model_config=model_config,
        model_path=model_path,
        served_model_name=model_path,
        tokenizer=tokenizer,
        processor=None,
        request_logger=SimpleNamespace(log_requests=False, log_requests_level=0),
        incremental_streaming_output=False,
    )
    manager.config_value = lambda name: get_context().config_leaf(name)
    template = TemplateManager()
    template.load_chat_template(manager, None, model_path)
    os.environ["SGLANG_OPENAI_CHAT_CONVERSION_WORKER"] = "0"
    baseline = OpenAIServingChat(manager, template)
    os.environ["SGLANG_OPENAI_CHAT_CONVERSION_WORKER"] = "1"
    candidate = OpenAIServingChat(manager, template)
    assert baseline._conversion_worker is None
    isolated = (
        candidate._conversion_worker._convert.__self__.tokenizer_manager.tokenizer
    )
    assert isolated is not tokenizer
    # Initialization is a snapshot; later env changes must not change the path.
    os.environ["SGLANG_OPENAI_CHAT_CONVERSION_WORKER"] = "0"

    async def capture_conversion(self, adapted, request, raw):
        return adapted

    for handler in (baseline, candidate):
        handler._handle_streaming_request = MethodType(capture_conversion, handler)

    result = dict(
        passed=False,
        device="cpu",
        fixture_sha256=args.fixture_sha256,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_is_fast=tokenizer.is_fast,
        tokenizer_isolated=True,
        rows=[],
        scope="Native conversion + native TokenizerManager stream consumer with pre-ready CPU output. Generation dispatch replaced by a capture endpoint. Not a GPU or C48 performance gate.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def measured(handler, body):
        event = asyncio.Event()
        state = SimpleNamespace(
            event=event,
            out_list=[dict(text="ready", meta_info={})],
            finished=False,
            time_stats=SimpleNamespace(response_sent_to_client_time=1),
        )
        generated = GenerateReqInput(input_ids=[1], stream=True, rid="ready-output")
        stream = TokenizerManager._stream_one_response(manager, generated, state)
        output = {}

        async def consume():
            value = await stream.__anext__()
            output.update(time_ns=time.perf_counter_ns(), value=value)

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        request = ChatCompletionRequest(**body)
        raw = Request(
            {
                "type": "http",
                "headers": [(b"x-request-id", b"q19-cpu")],
                "method": "POST",
                "path": "/v1/chat/completions",
            }
        )
        event.set()
        begin = time.perf_counter_ns()
        try:
            adapted = await handler.handle_request(request, raw)
            end = time.perf_counter_ns()
            await consumer
        finally:
            if not consumer.done():
                consumer.cancel()
            await stream.aclose()
        assert isinstance(adapted, GenerateReqInput), repr(adapted)
        assert output["value"]["text"] == "ready"
        fields = dataclasses.asdict(adapted)
        fields.pop("received_time", None)
        return fields, dict(
            conversion_wall_ms=(end - begin) / 1e6,
            ready_output_delay_ms=(output["time_ns"] - begin) / 1e6,
            output_before_conversion_done=output["time_ns"] < end,
        )

    try:
        for repeat in (1, 2):
            for point in fixture["points"]:
                measured_rows = {}
                order = (
                    (("off", baseline), ("on", candidate))
                    if repeat == 1
                    else (("on", candidate), ("off", baseline))
                )
                for label, handler in order:
                    fields, timing = await measured(handler, point["request"])
                    actual = fields["input_ids"]
                    token_equal = actual == point["expected_input_ids"]
                    row = dict(
                        job=point["job"],
                        xrid=point["xrid"],
                        repeat=repeat,
                        setting=label,
                        expected_tokens=point["expected_tokens"],
                        actual_tokens=len(actual),
                        observed_token_ids_equal=token_equal,
                        **timing,
                    )
                    if not token_equal:
                        row["different_positions"] = sum(
                            a != b for a, b in zip(actual, point["expected_input_ids"])
                        )
                    result["rows"].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    assert token_equal, row
                    measured_rows[label] = (fields, row)
                assert measured_rows["off"][0] == measured_rows["on"][0], (
                    "Converted request fields differ"
                )
                assert not measured_rows["off"][1]["output_before_conversion_done"]
                assert measured_rows["on"][1]["output_before_conversion_done"]
                for label in ("off", "on"):
                    measured_rows[label][1]["all_request_fields_equal"] = True
                print(
                    json.dumps(
                        dict(
                            job=point["job"],
                            repeat=repeat,
                            tokens=point["expected_tokens"],
                            off=measured_rows["off"][1],
                            on=measured_rows["on"][1],
                        )
                    ),
                    flush=True,
                )
        result["passed"] = len(result["rows"]) == 24
    finally:
        await candidate.aclose_conversion_worker()
        reset_context()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    asyncio.run(run(args))
