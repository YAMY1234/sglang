"""OpenAI simple-evals multiple-choice GPQA-Diamond and MMLU for the paired DUET exams.

The prompt template, the answer patterns and the two normalisers below are copied verbatim from simple-evals
(github.com/openai/simple-evals, commit 652c89d0ca9df547706735883097e9537d40dc47, common.py lines 15-75 and 355-396; MIT licence);
the scoring follows its gpqa_eval.py / mmlu_eval.py.  lm-eval only sends the prompt as the user message and stores the reply
(the thinking segment removed by model.py, as for every exam).
  GPQA-Diamond: the 198-row csv with GPQA's original columns (Hub dataset zai-org/glm-simple-evals-dataset, gpqa_diamond.csv,
    sha256 41d1213c...; simple-evals' own copy on openaipublic is not reachable from the cluster, so its row order is unverified),
    options permuted as simple-evals permutes them: random.Random(0), one rng.sample(range(4), 4) per row and repeat over
    examples * n_repeats; repetition k of the exam uses repeat k.  Extraction: the first match of ANSWER_PATTERN_MULTICHOICE in
    the raw reply.
  MMLU: the 14042 test questions (cais/mmlu 'all' test split, revision c30699e8; the Hendrycks test set of simple-evals'
    mmlu.csv), one pass.  Extraction: normalize_response, then the first MULTILINGUAL_ANSWER_REGEXES pattern that matches,
    normalize_extracted_answer.
Every doc carries se_id (GPQA: csv row, MMLU: test-set index), the key report_lmeval pairs on: lm-eval renumbers doc_id inside
a shard.  run.py passes the repetition and the shard ("i/n": docs i, i+n, ...) through lm-eval's --metadata.  The verbatim
block keeps upstream's non-raw "\\s" strings (Python warns once at import; the regexes are unchanged).
Copied next to the generated yamls by tasks.py."""
import csv
import os
import random
import re

import datasets

DATA = os.path.join(os.environ.get("DUET_DATA_ROOT", "/src/results/hfcache"), "simple-evals")
GPQA_MAX_REPEATS = 8          # the permutation stream of 8 repeats; its first 4 repeats are simple-evals' default n_repeats=4

# ---------------------------------------------------------------- simple-evals common.py (verbatim)
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:",
    "Answer\s*:​​​​​​",  # Korean invisible character
    "উত্তর\s*:",
    "उत्तर\s*:",
    "উত্তরঃ",
    "উত্তর\s*:",
    "Antwort\s*:",
    "답변\s*:",
    "정답\s*:",
    "답\s*:",
    "答案\s*：",
    "答案\s*:",
    "答\s*：",
    "答\s*:",
    "答复\s*：",
    "答曰\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "الجواب النهائي:",
    "Respuesta\s*:",
    "Risposta\s*:",
    "答え\s*:",
    "答え\s*：",
    "回答\s*:",
    "回答\s*：",
    "解答\s*:",
    "Jawaban\s*:",
    "Réponse\s*:",
    "Resposta\s*:",
    "Jibu\s*:",
    "Idahun\s*:",
    "Ìdáhùn\s*:",
    "Idáhùn\s*:",
    "Àmọ̀nà\s*:",
    "Àdáhùn\s*:",
    "Ànúgọ\s*:",
    "Àṣàyàn\s*:",
]


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


# ---------------------------------------------------------------- data
def _shard(docs, shard):
    if not shard:
        return docs
    i, n = (int(x) for x in str(shard).split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard {shard}: need 0 <= i < n")
    return docs[i::n]


def gpqa_docs(rep=0, shard=None, **kwargs):
    with open(os.path.join(DATA, "gpqa_diamond.csv"), encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rng = random.Random(0)
    perms = [rng.sample(range(4), 4) for _ in range(len(rows) * GPQA_MAX_REPEATS)]   # gpqa_eval.py: examples * n_repeats, then one each
    rep = int(rep)
    if not 0 <= rep < GPQA_MAX_REPEATS:
        raise ValueError(f"GPQA simple-evals repeat {rep} outside 0..{GPQA_MAX_REPEATS - 1}")
    docs = []
    for i, row in enumerate(rows):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        choices = [choices[j] for j in perms[rep * len(rows) + i]]
        docs.append({"Question": row["Question"], "A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3],
                     "answer": "ABCD"[choices.index(row["Correct Answer"])], "se_id": i, "repeat": rep})
    return {"test": datasets.Dataset.from_list(_shard(docs, shard))}


def mmlu_docs(shard=None, **kwargs):
    import pyarrow.parquet as pq
    rows = pq.read_table(os.path.join(DATA, "mmlu_test.parquet")).to_pylist()
    docs = [{"Question": r["question"], "A": r["choices"][0], "B": r["choices"][1], "C": r["choices"][2], "D": r["choices"][3],
             "answer": "ABCD"[r["answer"]], "Subject": r["subject"], "se_id": i} for i, r in enumerate(rows)]
    return {"test": datasets.Dataset.from_list(_shard(docs, shard))}


def doc_to_text(doc):
    return QUERY_TEMPLATE_MULTICHOICE.format(Question=doc["Question"], A=doc["A"], B=doc["B"], C=doc["C"], D=doc["D"])


# ---------------------------------------------------------------- scoring (gpqa_eval.py / mmlu_eval.py)
def process_results_gpqa(doc, results):
    match = re.search(ANSWER_PATTERN_MULTICHOICE, results[0])
    extracted = match.group(1) if match else None
    return {"accuracy": 1.0 if extracted == doc["answer"] else 0.0, "no_answer": float(extracted is None)}


def process_results_mmlu(doc, results):
    response_text = normalize_response(results[0])
    extracted = None
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted = normalize_extracted_answer(match.group(1))
            break
    return {"accuracy": 1.0 if extracted == doc["answer"] else 0.0, "no_answer": float(extracted is None)}
