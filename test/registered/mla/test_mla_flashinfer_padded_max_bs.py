"""FlashInfer MLA backend must serve the padded max batch under MLP sync.

Regression for: with DP attention (or an EP a2a backend) the eager and
cuda-graph runners pad the request count to the attn-tp alignment, so a
warm-up / capture batch can be wider than req_to_token_pool.size. The
FlashInfer MLA backend used to size kv_indptr to the raw pool size and
crash at startup with "The expanded size of the tensor (N) must match the
existing size (N+k)". TP4 + DP-attention 2 gives attn_tp_size=2, and
--max-running-requests 3 leaves each DP rank a pool of 1 request while the
padded warm-up / capture batch is 2.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-c", runner_config="4-gpu-h100")


class TestMLAFlashInferPaddedMaxBs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                # attn_tp_size = tp / dp = 2, so MLP sync pads the batch to a
                # multiple of 2; an odd request cap makes pool < padded batch.
                # (dp=1 would have DP attention resolved away.)
                "--enable-dp-attention",
                "--dp",
                "2",
                "--attention-backend",
                "flashinfer",
                # The ragged prefill wrapper rejects the DP-padded q rows
                # (q.shape[0] != qo_indptr[-1]); that is a separate issue,
                # keep this test on the paged prefill path.
                "--flashinfer-mla-disable-ragged",
                "--max-running-requests",
                "3",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generate_at_request_cap(self):
        # Startup (warm-up + graph capture at the padded batch) is the real
        # check; the generate below confirms decode at the request cap works.
        responses = [
            requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0},
                },
            )
            for _ in range(3)
        ]
        for response in responses:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["text"].strip())


if __name__ == "__main__":
    unittest.main()
