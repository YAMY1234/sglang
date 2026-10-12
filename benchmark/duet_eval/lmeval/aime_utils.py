"""AIME 2024 from the HuggingFaceH4/aime_2024 mirror (copied next to the generated duet_aime24.yaml by tasks.py).  The mirror's
`answer` column is a string, zero-padded for 7 of the 30 problems ("025"), while lm-eval's own source (Maxwell-Jia/AIME_2024) holds
integers; lm-eval's grader compares strings, so a correct "25" would be scored wrong.  The integer form is restored before grading,
which makes the task score exactly as lm-eval's aime24 task does."""


def process_docs(dataset):
    return dataset.map(lambda doc: {"answer": str(int(doc["answer"]))})
