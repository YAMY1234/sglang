"""lm-eval (EleutherAI lm-evaluation-harness) driver for the paired DUET exams on SGLang servers (README.md: the protocol and how to run it).

Exams: tasks.py (the protocol and the task yamls: lm-eval's AIME / MATH-500 / IFEval / RULER with the exam generation settings),
se_utils.py (GPQA-Diamond and MMLU as OpenAI simple-evals runs them), lcb_utils.py + lcb_grade.py + lcb/ (LiveCodeBench v6, the
official prompt and grader), aa_lcr_data.py + aa_lcr_utils.py + aa_lcr_judge.py (AA-LCR, not in the protocol), prepare_data.sh
(the offline exam data).  Serving side: model.py (lm-eval model `sglang-duet`: paired seeds, sidecar), lb.py (least-outstanding
balancer over data-parallel servers; replays an interrupted run's replies), run.py (one repetition of the exams = one lm-eval process), exam_lmeval_job.sh (whole node:
servers, balancers, reps, report).  Results: report_lmeval.py (paired statistics over the sample logs; common.py: the sample-log
readers), forced_pairs.py (teacher-forced NLL gap); the one-off diagnostics are in ../archive/."""
