"""Diagnostic A/B for the GLM-5.3-Flash GSM8K request contract.

This is intentionally a manual test: it compares the registered CI contract
with the model-native chat contract without changing a production threshold.
"""

import json
import statistics
import time
import unittest
from collections import Counter

import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.simple_eval_common import ChatCompletionSampler, CompletionSampler
from sglang.test.simple_eval_mixed_prefix_gsm8k import (
    INVALID,
    GSM8KEval,
    get_answer_value,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    try_cached_model,
)

MODEL_PATH = "zai-org/GLM-5.3-Flash"
SERVER_LAUNCH_TIMEOUT = 3600


def _observe_responses(sampler, api):
    records = []
    resource = (
        sampler.client.chat.completions if api == "chat" else sampler.client.completions
    )
    original_create = resource.create

    def observed_create(*args, **kwargs):
        response = original_create(*args, **kwargs)
        choice = response.choices[0]
        usage = response.usage
        if api == "chat":
            message = choice.message
            visible = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or ""
        else:
            visible = choice.text or ""
            reasoning = ""
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        records.append(
            {
                "finish_reason": choice.finish_reason,
                "completion_tokens": (
                    usage.completion_tokens if usage is not None else None
                ),
                "reasoning_tokens": (
                    getattr(details, "reasoning_tokens", None)
                    if details is not None
                    else None
                ),
                "visible_chars": len(visible),
                "reasoning_chars": len(reasoning),
            }
        )
        return response

    resource.create = observed_create
    return records


def _numeric_stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


class TestGLM53GSM8KProtocolAB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--dsa-prefill-backend",
                "trtllm",
                "--dsa-decode-backend",
                "trtllm",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--moe-runner-backend",
                "deep_gemm",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--enable-dp-attention",
                "--dp-size",
                "4",
                "--moe-a2a-backend",
                "deepep",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        _wait_for_gpu_idle_in_ci(timeout=120)

    def _run_case(self, *, label, api, max_tokens, num_examples, num_shots):
        requests.get(self.base_url + "/flush_cache").raise_for_status()
        common = {
            "base_url": self.base_url + "/v1",
            "model": self.model,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
        if api == "chat":
            sampler = ChatCompletionSampler(
                **common,
                reasoning_effort="max",
                record_meta_info=True,
            )
        else:
            sampler = CompletionSampler(
                **common,
                stop=["Question", "Assistant:", "<|separator|>"],
            )
        response_records = _observe_responses(sampler, api)
        evaluation = GSM8KEval(
            num_examples=num_examples,
            num_threads=128,
            num_shots=num_shots,
        )

        started = time.perf_counter()
        result = evaluation(sampler)
        latency = time.perf_counter() - started

        wrong = []
        invalid = 0
        for index, convo in enumerate(result.convos):
            response_text = convo[-1]["content"]
            predicted = get_answer_value(response_text)
            expected = get_answer_value(evaluation._lines[index]["answer"])
            if predicted == INVALID:
                invalid += 1
            if predicted != expected:
                wrong.append(
                    {
                        "index": index,
                        "question": evaluation._lines[index]["question"],
                        "expected": expected,
                        "predicted": predicted,
                        "response_chars": len(response_text),
                        "response_tail": response_text[-600:],
                    }
                )

        summary = {
            "label": label,
            "api": api,
            "reasoning_effort": "max" if api == "chat" else None,
            "max_tokens": max_tokens,
            "num_examples": len(evaluation._lines),
            "num_shots": num_shots,
            "score": float(result.score),
            "invalid_answers": invalid,
            "latency_seconds": latency,
            "finish_reasons": dict(
                Counter(record["finish_reason"] for record in response_records)
            ),
            "completion_tokens": _numeric_stats(
                record["completion_tokens"] for record in response_records
            ),
            "reasoning_tokens": _numeric_stats(
                record["reasoning_tokens"] for record in response_records
            ),
            "visible_chars": _numeric_stats(
                record["visible_chars"] for record in response_records
            ),
            "reasoning_chars": _numeric_stats(
                record["reasoning_chars"] for record in response_records
            ),
            "wrong_count": len(wrong),
        }
        print("GLM53_GSM8K_AB_SUMMARY " + json.dumps(summary, sort_keys=True))
        for item in wrong:
            print(
                "GLM53_GSM8K_AB_WRONG "
                + json.dumps({"label": label, **item}, sort_keys=True)
            )
        return summary

    def test_protocol_ab(self):
        cases = [
            {
                "label": "ci_completion_20shot_512",
                "api": "completion",
                "max_tokens": 512,
                "num_examples": 500,
                "num_shots": 20,
            },
            {
                "label": "chat_max_20shot_512",
                "api": "chat",
                "max_tokens": 512,
                "num_examples": 500,
                "num_shots": 20,
            },
            {
                "label": "chat_max_20shot_2048",
                "api": "chat",
                "max_tokens": 2048,
                "num_examples": 500,
                "num_shots": 20,
            },
            {
                "label": "chat_max_zeroshot_full_2048",
                "api": "chat",
                "max_tokens": 2048,
                "num_examples": None,
                "num_shots": 0,
            },
        ]
        summaries = [self._run_case(**case) for case in cases]
        print("GLM53_GSM8K_AB_ALL " + json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
