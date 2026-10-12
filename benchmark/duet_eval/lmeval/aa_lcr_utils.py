"""AA-LCR for the DUET exams (aa_lcr_data.py builds the questions with the official prompt).  The official score is an LLM
equality check of the reply against the official answer, run afterwards by aa_lcr_judge.py over the sample logs; lm-eval records
only whether the reply (thinking removed) is non-empty.  Copied next to the generated duet_aa_lcr.yaml by tasks.py."""


def process_results(doc, results):
    return {"answered": int(bool(results[0].strip()))}
