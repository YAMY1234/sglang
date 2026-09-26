"""CPU coverage for continuing a streaming session without new prompt tokens."""
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session


def received(rid, ids):
    return SimpleNamespace(rid=rid, input_ids=array("q", ids), mm_inputs=None,
        session_params=SimpleNamespace(id="s", rid=None, offset=None, replace=False, drop_previous_output=False),
        sampling_params=SamplingParams(max_new_tokens=2), lora_id=None, custom_logit_processor=None,
        stream=False, return_logprob=False, top_logprobs_num=0, token_ids_logprob=None,
        return_sampling_mask=False, require_reasoning=False, return_hidden_states=False,
        return_routed_experts=False, routed_experts_start_len=0, priority=None, routing_key=None,
        extra_key=None, cache_salt=None, http_worker_ipc=None, time_stats=None)


class TestEmptySessionInput(unittest.TestCase):
    def test_normalizes_only_session_empty_input(self):
        req = GenerateReqInput(input_ids=[], session_params={"id": "s"})
        req.normalize_batch_and_arguments()
        self.assertTrue(req.is_single)
        self.assertEqual(req.input_ids, [])
        for params in (None, {}, {"id": ""}):
            with self.assertRaises(ValueError):
                GenerateReqInput(input_ids=[], session_params=params).normalize_batch_and_arguments()
        with self.assertRaises(ValueError):
            GenerateReqInput(input_ids=[], session_params={"id": "s"}, sampling_params={"n": 2}).normalize_batch_and_arguments()

    def test_append_consumes_committed_output(self):
        session = Session(capacity_of_str_len=0, session_id="s", streaming=True)
        first = session.create_req(received("first", [10, 20]), tokenizer=None, vocab_size=100)
        first.output_ids.extend([30, 40])
        first._refresh_fill_ids()
        session.finish_req(first)
        second = session.create_req(received("second", []), tokenizer=None, vocab_size=100)
        self.assertEqual(list(second.origin_input_ids), [10, 20, 30, 40])
        self.assertIsNone(second.to_finish)

    def test_empty_session_without_history_aborts(self):
        session = Session(capacity_of_str_len=0, session_id="s", streaming=True)
        with patch("sglang.srt.managers.schedule_batch.get_parallel", return_value=SimpleNamespace(tp_rank=0)):
            req = session.create_req(received("empty", []), tokenizer=None, vocab_size=100)
        self.assertIsNotNone(req.to_finish)
        self.assertFalse(session._inflight)


if __name__ == "__main__":
    unittest.main()
