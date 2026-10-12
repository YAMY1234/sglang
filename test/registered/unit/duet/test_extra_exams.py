"""HumanEval+ and LongBench v2 in the exam harness (benchmark/duet_eval/lmeval): prompts, answer extraction, sanitizing and grading
on hand-made replies, no GPU.

LongBench v2 (longbench_v2_utils.py): the template is the official file byte for byte, the prompt is filled in upstream's order,
the middle truncation is pred.py's (a literal copy of its lines is the reference), the extraction is pred.py's.  The thinking block
is removed by model.py before any grader sees a reply (needs lm-eval; skipped without it).  HumanEval+ (evalplus_utils.py,
humaneval_grade.py): the prompt is EvalPlus's chat prompt; with evalplus installed (requirements-evalplus.txt) the sanitizer is run
on replies with and without a code block, and the grader on two fake arms of all 164 tasks (canonical solutions, one wrong, one
reply without code), which also computes EvalPlus's ground truth on first use (DUET_DATA_ROOT/evalplus-cache).

Run from this directory:  python -m unittest test_extra_exams
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmark.duet_eval.lmeval import evalplus_utils, longbench_v2_utils as lb  # noqa: E402

HAVE_EVALPLUS = importlib.util.find_spec("evalplus") is not None and importlib.util.find_spec("tree_sitter_python") is not None
HAVE_LMEVAL = importlib.util.find_spec("lm_eval") is not None
HAVE_HUMANEVAL_DATA = os.path.isfile(evalplus_utils.DATA)


class WordTokenizer:
    """A whitespace tokenizer with a BOS special token that encode() adds and decode(skip_special_tokens=True) drops, as HF
    tokenizers with add_bos do."""
    BOS = "<s>"

    def __init__(self):
        self.vocab = {self.BOS: 0}

    def encode(self, text):
        ids = [0]
        for w in text.split(" "):
            ids.append(self.vocab.setdefault(w, len(self.vocab)))
        return ids

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids if not (skip_special_tokens and i == 0))


def official_truncation(prompt, tokenizer, max_len):
    """pred.py query_llm, the model_map branch, verbatim apart from the function frame."""
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt


class LongBenchV2(unittest.TestCase):
    def test_template_is_the_official_file(self):
        # sha256 of prompts/0shot.txt at THUDM/LongBench 2e00731f8d0bff23dc4325161044d0ed8af94c1e
        self.assertEqual(hashlib.sha256(lb.TEMPLATE_0SHOT.encode()).hexdigest(),
                         "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950")
        self.assertEqual(lb.MAX_LEN, 120000)

    def test_prompt_fields_are_stripped_and_replaced_in_upstream_order(self):
        item = {"context": "\n  A text that mentions $Q$ literally.  \n", "question": " Which? ", "choice_A": " a ", "choice_B": "b",
                "choice_C": "c\n", "choice_D": "d"}
        p = lb.official_prompt(item)
        # upstream replaces $DOC$ first, so a "$Q$" inside the context is replaced by the question too
        self.assertIn("<text>\nA text that mentions Which? literally.\n</text>", p)
        self.assertIn("What is the correct answer to this question: Which?\nChoices:\n(A) a\n(B) b\n(C) c\n(D) d\n\n", p)
        self.assertTrue(p.endswith('Format your response as follows: "The correct answer is (insert answer here)".'))

    def test_middle_truncation_is_pred_py(self):
        for words, max_len in ((30, 10), (30, 9), (9, 10), (10, 10)):
            text = " ".join(f"w{i}" for i in range(words))
            got, n = lb.truncate(text, WordTokenizer(), max_len)
            self.assertEqual(got, official_truncation(text, WordTokenizer(), max_len))
            self.assertEqual(n, words + 1)                      # the BOS counts, as tokenizer.encode returns it
        got, _ = lb.truncate(" ".join(f"w{i}" for i in range(30)), WordTokenizer(), 10)
        self.assertEqual(got, "w0 w1 w2 w3 w25 w26 w27 w28 w29")    # BOS + 4 head ids, 5 tail ids; BOS dropped by decode

    def test_short_prompt_is_not_re_decoded(self):
        tok = WordTokenizer()
        text = "a  b"                                          # a decode round trip would change the double space
        self.assertEqual(lb.truncate(text, tok, 10), (text, 4))

    def test_extract_answer(self):
        cases = {"The correct answer is (B)": "B", "**The correct answer is (C)**": "C", "The correct answer is D.": "D",
                 "I think (A). The correct answer is (B), not (C)": "B", "The correct answer is (E)": None, "Answer: A": None,
                 "the correct answer is (A)": None, "The correct answer is A or B": "A", "": None,
                 "The correct answer is *(A)*": "A"}
        for reply, want in cases.items():
            self.assertEqual(lb.extract_answer(reply), want, reply)

    def test_process_results(self):
        doc = {"answer": "C"}
        self.assertEqual(lb.process_results(doc, ["  The correct answer is (C)\n"]), {"accuracy": 1.0, "no_answer": 0.0})
        self.assertEqual(lb.process_results(doc, ["The correct answer is (A)"]), {"accuracy": 0.0, "no_answer": 0.0})
        self.assertEqual(lb.process_results(doc, [""]), {"accuracy": 0.0, "no_answer": 1.0})


@unittest.skipUnless(HAVE_LMEVAL, "lm-eval is not installed (model.py)")
class ThinkingBlock(unittest.TestCase):
    """model.py removes the thinking block before lm-eval stores and grades a reply: the graders read the final answer only."""

    def strip(self, text, think_end="</think>", required=True):
        from benchmark.duet_eval.lmeval import model
        stub = types.SimpleNamespace(think_end=think_end, think_required=required)
        return model.SGLangDuetAPI._strip_thinking(stub, text)

    def test_answer_after_the_thinking_block(self):
        answer, unclosed = self.strip("Maybe the correct answer is (A)?</think>The correct answer is (B)")
        self.assertEqual((lb.extract_answer(answer.strip()), unclosed), ("B", False))
        answer, _ = self.strip("plan: ```python\ndef f():\n    return 0\n```</think>```python\ndef f():\n    return 1\n```")
        self.assertNotIn("return 0", answer)

    def test_unclosed_thinking_is_an_empty_answer(self):
        answer, unclosed = self.strip("The correct answer is (A) ... still thinking")
        self.assertEqual((answer, unclosed), ("", True))
        self.assertIsNone(lb.extract_answer(answer))

    def test_model_without_thinking_keeps_the_reply(self):
        answer, unclosed = self.strip("The correct answer is (D)", think_end=None, required=False)
        self.assertEqual((lb.extract_answer(answer), unclosed), ("D", False))


class HumanEvalPrompt(unittest.TestCase):
    def test_prompt_is_evalplus_chat_prompt(self):
        doc = {"prompt": '\n\ndef add(a, b):\n    """Add two numbers."""\n', "task_id": "HumanEval/x", "entry_point": "add"}
        self.assertEqual(evalplus_utils.doc_to_text(doc),
                         "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n"
                         '```python\ndef add(a, b):\n    """Add two numbers."""\n```')

    @unittest.skipUnless(HAVE_EVALPLUS, "evalplus is not installed")
    def test_prefix_and_message_are_evalplus_source(self):
        root = Path(importlib.util.find_spec("evalplus").submodule_search_locations[0])
        self.assertIn(f'instruction_prefix = "{evalplus_utils.INSTRUCTION_PREFIX}"', (root / "codegen.py").read_text())
        self.assertIn('prompt = self.instruction_prefix + f"\\n```python\\n{prompt.strip()}\\n```"',
                      (root / "provider" / "openai.py").read_text())
        self.assertIn('prompt = task["prompt"].strip() + "\\n"', (root / "codegen.py").read_text())


@unittest.skipUnless(HAVE_EVALPLUS, "evalplus is not installed")
class HumanEvalSanitizer(unittest.TestCase):
    def sanitize(self, reply, entry="add"):
        from evalplus.sanitize import sanitize
        return sanitize(reply, entrypoint=entry)

    def test_code_block_with_prose(self):
        reply = ("Here is the function.\n\n```python\nimport math\n\ndef add(a, b):\n    return a + b\n```\n\n"
                 "It adds the numbers.")
        self.assertEqual(self.sanitize(reply), "import math\ndef add(a, b):\n    return a + b")

    def test_reply_without_a_code_block(self):
        reply = "def add(a, b):\n    return a + b"
        self.assertEqual(self.sanitize(reply), reply)

    def test_tests_and_main_are_dropped_helpers_kept(self):
        reply = ("```python\ndef helper(x):\n    return x\n\ndef add(a, b):\n    return helper(a) + b\n\n"
                 "assert add(1, 2) == 3\nif __name__ == '__main__':\n    print(add(1, 2))\n```")
        self.assertEqual(self.sanitize(reply), "def helper(x):\n    return x\ndef add(a, b):\n    return helper(a) + b")

    def test_prose_only_is_no_code(self):
        from benchmark.duet_eval.lmeval.humaneval_grade import defines_entry_point
        # EvalPlus's sanitizer falls back to the reply's first line when it finds no code; the grader flags such a solution
        self.assertEqual(self.sanitize("I cannot solve this problem."), "I cannot solve this problem.")
        self.assertEqual(self.sanitize(""), "")
        self.assertFalse(defines_entry_point("I cannot solve this problem.", "add"))
        self.assertFalse(defines_entry_point("", "add"))
        self.assertFalse(defines_entry_point("def other(a, b):\n    return a", "add"))
        self.assertTrue(defines_entry_point(self.sanitize("```python\ndef add(a, b):\n    return a + b\n```"), "add"))


@unittest.skipUnless(HAVE_EVALPLUS and HAVE_HUMANEVAL_DATA, "evalplus or the HumanEval+ data (prepare_data.sh) is missing")
class HumanEvalGrader(unittest.TestCase):
    def write_arm(self, root, arm, replies):
        d = root / arm / "rep0" / "humaneval_plus" / "__models__base"
        d.mkdir(parents=True)
        with open(d / "samples_duet_humaneval_plus_2026-10-10T00-00-00.000000.jsonl", "w") as fh:
            for i, (task, reply) in enumerate(replies):
                fh.write(json.dumps({"doc_id": i, "doc": {"task_id": task}, "resps": [[reply]], "filtered_resps": [reply]}) + "\n")
        return str(root / arm)

    def test_two_arms_in_one_call(self):
        from benchmark.duet_eval.lmeval import humaneval_grade
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
        canonical = [(t, f"Solution:\n```python\n{p['prompt']}{p['canonical_solution']}\n```") for t, p in problems.items()]
        other = list(canonical)
        other[0] = ("HumanEval/0", "```python\ndef has_close_elements(numbers, threshold):\n    return False\n```")
        other[1] = ("HumanEval/1", "I am not sure how to write this.")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.write_arm(root, "stock", canonical), self.write_arm(root, "duet", other)
            humaneval_grade.grade([a, b], str(root / "grade"), parallel=8, passes=1)
            lines = {arm: [json.loads(l) for l in open(next((root / arm).rglob("samples_*.jsonl")))] for arm in ("stock", "duet")}
            # every canonical solution passes the base tests; all but HumanEval/32's the plus tests (its known oracle failure),
            # which needs the raised memory limit for HumanEval/15 and /130
            failed = {d["doc"]["task_id"] for d in lines["stock"] if not d["humaneval_plus"]}
            self.assertEqual(failed, set(humaneval_grade.KNOWN_ORACLE_FAILURES))
            self.assertEqual(sum(d["humaneval"] for d in lines["stock"]), len(problems))
            self.assertEqual([lines["duet"][i]["humaneval_plus"] for i in range(3)], [0, 0, 1])
            self.assertEqual([lines["duet"][i]["humaneval"] for i in range(3)], [0, 0, 1])
            self.assertEqual([lines["duet"][i]["no_answer"] for i in range(3)], [0, 1, 0])
            self.assertTrue(all("humaneval_grade" in d for d in lines["duet"]))
            record = json.loads((root / "grade" / "grade.json").read_text())
            self.assertEqual(record["per_dir"][b]["humaneval_plus"], len(problems) - 3)


if __name__ == "__main__":
    unittest.main()
