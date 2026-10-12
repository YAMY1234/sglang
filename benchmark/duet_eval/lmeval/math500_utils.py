"""MATH-500 grading for chat-style responses (copied next to the generated duet_math500.yaml by tasks.py).

lm-eval's hendrycks_math500 grader takes the text between the first and the last `$` of the response as the answer, which
was written for completion-style outputs and scores visibly correct chat responses as wrong (a response quoting `$` twice
anywhere loses).  Here the answer is the LAST \\boxed{} of the response (lm-eval's own extractor) and two scores are reported:
  math_verify  -- symbolic equivalence (math-verify, the grader used by current MATH-500 leaderboards): primary
  exact_match  -- lm-eval's hendrycks_math string normalisation (`is_equiv`) on the boxed answer: secondary
A response without a \\boxed{} answer scores 0 on both and is counted in `no_boxed`."""
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed


def boxed_answer(text):
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except (AssertionError, IndexError):
        return None


def symbolic_match(answer, gold):
    from math_verify import parse, verify
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${answer}$")))
    except Exception:          # noqa: BLE001 - math-verify raises assorted parser errors on malformed latex
        return False


def process_results(doc, results):
    answer = boxed_answer(results[0])
    gold = doc["answer"]
    if answer is None:
        return {"math_verify": 0, "exact_match": 0, "no_boxed": 1}
    return {"math_verify": int(symbolic_match(answer, gold) or is_equiv(answer, gold)),
            "exact_match": int(is_equiv(answer, gold)), "no_boxed": 0}
