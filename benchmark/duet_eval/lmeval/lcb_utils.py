"""LiveCodeBench (code generation) for the DUET exams: the official LiveCodeBench prompt and code extraction
(lcb_runner/prompts/code_generation.py PromptConstants + get_generic_question_template_answer, lcb_runner/utils/
extraction_utils.extract_code for chat models), with the grading done afterwards in parallel by lcb_grade.py (the official
run_test on public + private tests, 6 s per test).  lm-eval records only whether a code block was produced; lcb_grade.py
adds `pass@1` to every sample line.  Copied next to the generated duet_lcb.yaml by tasks.py."""

SYSTEM_MESSAGE = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate "
                  "a correct Python program that matches the specification and passes all tests.")
FORMAT_WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMAT_WITHOUT_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample "
                          "inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the "
                          "inputs, runs the algorithm and writes output to STDOUT.")


def doc_to_text(doc):
    prompt = f"### Question:\n{doc['question_content']}\n\n"
    if doc.get("starter_code"):
        prompt += f"### Format: {FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{doc['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_code(text):
    """Official rule for chat models: the block between the last two ``` fences; '' when fewer than two fences."""
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1:fences[-1]])


def process_results(doc, results):
    return {"code_extracted": int(bool(extract_code(results[0]).strip()))}
