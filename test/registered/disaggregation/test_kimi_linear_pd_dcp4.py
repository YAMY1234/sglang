"""Eight-Blackwell Kimi-Linear PD disaggregation + decode DCP acceptance."""

import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_cuda_ci(est_time=1200, suite="nightly-8-gpu-b200", nightly=True)

KIMI_LINEAR_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


def _has_eight_blackwell_gpus() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        return False
    return all(
        torch.cuda.get_device_capability(device_index) >= (10, 0)
        for device_index in range(8)
    )


@unittest.skipUnless(
    _has_eight_blackwell_gpus(),
    "Kimi-Linear PD+DCP acceptance requires eight Blackwell GPUs",
)
class TestKimiLinearPDDCP4(GSM8KMixin, PDDisaggregationServerBase):
    model = KIMI_LINEAR_MODEL
    # The monolithic TP4+DCP4 baseline on this runner is 0.89 for this
    # deterministic 200-example slice.
    gsm8k_score_threshold = 0.88
    gsm8k_num_examples = 200
    gsm8k_num_threads = 4
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()

    @classmethod
    def start_prefill(cls):
        args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "4",
            "--ep-size",
            "4",
            "--attention-backend",
            "tokenspeed_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dtype",
            "bfloat16",
            "--random-seed",
            "0",
            "--mem-fraction-static",
            "0.80",
        ]
        args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=args,
        )

    @classmethod
    def start_decode(cls):
        args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "4",
            "--dcp-size",
            "4",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "tokenspeed_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dcp-comm-backend",
            "a2a",
            "--dcp-replicate-q-proj",
            "--dtype",
            "bfloat16",
            "--random-seed",
            "0",
            "--cuda-graph-max-bs-decode",
            "64",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--mem-fraction-static",
            "0.80",
        ]
        args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=args,
        )

    def _assert_batch_completes(self, batch_size: int):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": [
                    f"Reply with one short word for request {index}: the sky is"
                    for index in range(batch_size)
                ],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        outputs = response.json()
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), batch_size)
        self.assertTrue(all(output["text"].strip() for output in outputs))

    def test_decode_cuda_graph_and_eager_batch(self):
        self._assert_batch_completes(2)
        self._assert_batch_completes(2)
        self._assert_batch_completes(65)

    def test_decode_physical_capacity_sanity(self):
        response = requests.get(self.decode_url + "/server_info", timeout=30)
        response.raise_for_status()
        self.assertGreater(response.json()["max_total_num_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
