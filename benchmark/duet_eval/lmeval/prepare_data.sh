#!/bin/bash
# Build DUET_DATA_ROOT, the exam data the paired lm-eval exams read offline (exam_lmeval_job.sh), on a machine with internet.
#   bash benchmark/duet_eval/lmeval/prepare_data.sh <DUET_DATA_ROOT>          (run from the repository root; ~7 GB)
# Needs curl, sha256sum and python3 with datasets, pyyaml and nltk (the exam image has them); network: huggingface.co (with its
# file CDN), rajpurkar.github.io (SQuAD), raw.githubusercontent.com (NLTK's punkt_tab) and github.com (EvalPlus's release file).
# ruler/      RULER sources lm-eval would download: the Paul Graham essay haystack (HF dataset baber/paul_graham_essays, as one
#             parquet), SQuAD 2.0 dev and HotpotQA dev (distractor) -- run.py serves them to lm-eval's RULER tasks
# lcb/        LiveCodeBench v6 (release files test3..test6 of livecodebench/code_generation_lite at a pinned revision) ->
#             lcb_v6_2408_2505.jsonl (454 problems) + lcb_v6_2408_2505_tests.pkl (public + private tests, lcb/build_lcb_data.py)
# aa-lcr/     only with AA_LCR=1 (AA-LCR is not part of the exam protocol): AA-LCR v1.1 (ArtificialAnalysis/AA-LCR at a pinned
#             revision: questions csv + extracted document text) -> aa_lcr_v1.1.jsonl (100 questions, official prompt, aa_lcr_data.py)
# simple-evals/  GPQA-Diamond (the 198-row csv with GPQA's original columns -- Question, Correct Answer, Incorrect Answer 1-3 -- that
#             simple-evals reads, from zai-org/glm-simple-evals-dataset) and the MMLU test set (cais/mmlu 'all' test parquet), pinned
# evalplus-cache/  HumanEval+ v0.1.10 (the EvalPlus release file humanevalplus_release v0.1.10, decompressed as EvalPlus caches it:
#             evalplus-cache/evalplus/HumanEvalPlus-v0.1.10.jsonl; humaneval_grade.py runs EvalPlus with XDG_CACHE_HOME=evalplus-cache,
#             where EvalPlus also keeps its ground-truth outputs and times)
# longbench-v2/  LongBench v2 (THUDM/LongBench-v2 data.json at a pinned revision); the prompts of each tokenizer are built from it
#             into longbench-v2/prompts/ at exam time or ahead (python -m benchmark.duet_eval.lmeval.longbench_v2_utils --tokenizer <base>)
# datasets/   HF datasets cache of the lm-eval exam sets (AIME 2024 / 2025, MATH-500, IFEval) at the revisions of tasks.HF_SETS,
#             the revisions the generated task yamls load
# nltk_data/  punkt_tab (IFEval checkers, RULER sentence split)
# The RULER sources, the simple-evals files, the LiveCodeBench release files and the built problem file, and the HumanEval+ and
# LongBench v2 files are checked against the sha256 of the copies the
# reported exams used (a mismatch stops the script); the HF exam sets are loaded at the revisions the reported exams used; the built
# LiveCodeBench tests pickle and NLTK's punkt_tab are not hashed.  report_lmeval.py --check-prompts compares every prompt of a run with
# the reported ones (reference/*-prompts.json).
set -euo pipefail
D=${1:?DUET_DATA_ROOT}; mkdir -p "$D/ruler" "$D/simple-evals" "$D/lcb" "$D/datasets" "$D/nltk_data"
check() {   # check <file> <sha256>
  local got
  got=$(sha256sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "sha256 mismatch for $1: $got (expected $2) -- the upstream file changed"; exit 1; }
  echo "ok  $1"
}
get() {     # get <url> <file> <sha256>
  if [ ! -f "$2" ]; then curl -fL --retry 3 -o "$2.part" "$1"; mv "$2.part" "$2"; fi
  check "$2" "$3"
}
HF=https://huggingface.co/datasets
get "$HF/baber/paul_graham_essays/resolve/792d672b77d1a67720f7e9820d646358fc18792e/essays.parquet" "$D/ruler/essays.parquet" \
    825448033be010042ffd6fc77c32bbcab61f2ef63196f0fc4bfbf5b388a1660e
get "$HF/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json" \
    "$D/ruler/hotpot_dev_distractor_v1.json" e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8
get https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json "$D/ruler/dev-v2.0.json" \
    80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8
get "$HF/zai-org/glm-simple-evals-dataset/resolve/df520e38a2d589df9d4e24f8ea1f1af962cbf6cc/gpqa/gpqa_diamond.csv" \
    "$D/simple-evals/gpqa_diamond.csv" 41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305
get "$HF/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/all/test-00000-of-00001.parquet" \
    "$D/simple-evals/mmlu_test.parquet" 74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7
LCB=$HF/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505
get $LCB/test3.jsonl "$D/lcb/test3.jsonl" 28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79
get $LCB/test4.jsonl "$D/lcb/test4.jsonl" d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da
get $LCB/test5.jsonl "$D/lcb/test5.jsonl" 7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214
get $LCB/test6.jsonl "$D/lcb/test6.jsonl" bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5
[ -f "$D/lcb/lcb_v6_2408_2505_tests.pkl" ] || python -m benchmark.duet_eval.lmeval.lcb.build_lcb_data --src "$D/lcb" --out "$D/lcb"
check "$D/lcb/lcb_v6_2408_2505.jsonl" 2faba4bd0f3c6efc1eb51f76356db2339e98255976b720fa3ac5cba21c5e78df
mkdir -p "$D/evalplus-cache/evalplus" "$D/longbench-v2"
get https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz \
    "$D/evalplus-cache/HumanEvalPlus-v0.1.10.jsonl.gz" 272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101
[ -f "$D/evalplus-cache/evalplus/HumanEvalPlus-v0.1.10.jsonl" ] || gunzip -c "$D/evalplus-cache/HumanEvalPlus-v0.1.10.jsonl.gz" \
    > "$D/evalplus-cache/evalplus/HumanEvalPlus-v0.1.10.jsonl"
check "$D/evalplus-cache/evalplus/HumanEvalPlus-v0.1.10.jsonl" 42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f
get "$HF/THUDM/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json" "$D/longbench-v2/data.json" \
    15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2
if [ "${AA_LCR:-0}" = 1 ]; then
  mkdir -p "$D/aa-lcr"
  AALCR=$HF/ArtificialAnalysis/AA-LCR/resolve/9a77ef56b717057ade24ceab4d273712a0b4f19e
  get $AALCR/AA-LCR_Dataset.csv "$D/aa-lcr/AA-LCR_Dataset.csv" aea5198982436fa774f616964bbeb34716de9e94368a123275b9355765c9b73b
  get $AALCR/extracted_text/AA-LCR_extracted-text.zip "$D/aa-lcr/AA-LCR_extracted-text.zip" 5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a
  [ -f "$D/aa-lcr/aa_lcr_v1.1.jsonl" ] || python -m benchmark.duet_eval.lmeval.aa_lcr_data --src "$D/aa-lcr" --out "$D/aa-lcr"
  check "$D/aa-lcr/aa_lcr_v1.1.jsonl" 255c4654914101ab3f257b815f6d81280a8622983b7d3fc6b227c06154d7873b
fi
python -m nltk.downloader -d "$D/nltk_data" punkt_tab
HF_DATASETS_CACHE="$D/datasets" python - <<'PY'
# the exam sets as lm-eval loads them (dataset path / split of each task, the revision its yaml pins), into the datasets cache
import datasets
from benchmark.duet_eval.lmeval.tasks import HF_SETS
for exam, pin in HF_SETS.items():
    n = len(datasets.load_dataset(pin["path"], split=pin["split"], revision=pin["revision"]))
    print(f"ok  {exam}: {pin['path']}@{pin['revision'][:8]} [{pin['split']}] {n} rows")
PY
echo "PREPARE DATA DONE: export DUET_DATA_ROOT=$D"
