"""The RESUME=1 block of benchmark/duet_eval/lmeval/exam_lmeval_job.sh, run by bash on fake output trees (no GPU, no lm-eval).

The block is cut from the job script between its RESUME=1 and balancers section markers and run with the variables the job sets
before it (OUT, ARMS, REP_LIST, EXAMS, GEN_EXAMS, SE_SHARD, RULER_TAG, RESUME).  Checked: an unfinished repetition directory
(claim.json or sidecar.jsonl, no sample file) moves to <out>/partial/<arm dir>/rep<k>/<kind>-<n> and its sidecar is handed to the
arm's balancer; finished directories, other jobs' RULER units and repetitions outside REP_LIST stay; the RULER unit of the job
(<out>/<arm>-ruler[-<tag>]/rep0/ruler) is resumed like the other kinds.

Run from this directory:  python -m unittest test_exam_resume
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
JOB = ROOT / "benchmark/duet_eval/lmeval/exam_lmeval_job.sh"
BEGIN = "# ---------------------------------------------------------------- RESUME=1"
END = "# ---------------------------------------------------------------- balancers"


def resume_block():
    text = JOB.read_text()
    return text[text.index(BEGIN):text.index(END)]


class ResumeBlock(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "out"
        self.out.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def rep_dir(self, arm_dir, rep, kind, claim=True, sidecar=0, samples=False):
        d = self.out / arm_dir / f"rep{rep}" / kind
        d.mkdir(parents=True)
        if claim:
            (d / "claim.json").write_text(json.dumps({"pid": 1, "job": "1120408"}))
        if sidecar:
            (d / "sidecar.jsonl").write_text("".join(json.dumps({"rid": f"1-k{i}-0"}) + "\n" for i in range(sidecar)))
        if samples:
            (d / "__models__base").mkdir()
            (d / "__models__base" / "samples_duet_x_2026-10-07T01-02-03.456.jsonl").write_text("{}\n")
        return d

    def run_block(self, exams, rep_list, arms="duet", ruler_tag="", se_shard="", resume="1", doc_shard=""):
        own = ("lcb", "aa_lcr", "humaneval_plus", "longbench_v2")                    # tasks.OWN_PROCESS
        gen = ",".join(e for e in exams.split(",") if e != "ruler" and e not in own)
        env = dict(os.environ, OUT=str(self.out), ARMS=arms, REP_LIST=rep_list, EXAMS=exams, GEN_EXAMS=gen,
                   SE_SHARD=se_shard, RULER_TAG=ruler_tag, RESUME=resume, OWN_EXAMS=" ".join(own),
                   DOC_SHARD_EXAMS="lcb humaneval_plus longbench_v2", DOC_SHARD=doc_shard)
        script = resume_block() + 'for a in "${!RESUME_FILES[@]}"; do echo "FILES $a ${RESUME_FILES[$a]}"; done\n'
        proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)
        files = {}
        for line in proc.stdout.splitlines():
            if line.startswith("FILES "):
                _, arm, paths = line.split(" ", 2)
                files[arm] = paths.split(",")
        return files, proc.stdout

    def test_ruler_unit_of_this_job_is_resumed(self):
        self.rep_dir("duet-ruler-128k-e", 0, "ruler", sidecar=3)
        self.rep_dir("duet-ruler-64k", 0, "ruler", sidecar=2)                   # another job's unit
        self.rep_dir("duet-ruler-32k", 0, "ruler", sidecar=5, samples=True)     # finished
        files, log = self.run_block("ruler", "0", ruler_tag="128k-e")
        moved = self.out / "partial/duet-ruler-128k-e/rep0/ruler-1"
        self.assertTrue((moved / "claim.json").is_file())
        self.assertFalse((self.out / "duet-ruler-128k-e/rep0/ruler").exists())
        self.assertEqual(files, {"duet": [str(moved / "sidecar.jsonl")]})
        self.assertTrue((self.out / "duet-ruler-64k/rep0/ruler/claim.json").is_file())
        self.assertTrue((self.out / "duet-ruler-32k/rep0/ruler/claim.json").is_file())
        self.assertIn("-> partial/duet-ruler-128k-e/rep0/ruler-1", (self.out / "job.txt").read_text())
        self.assertIn("duet-ruler-128k-e rep0 ruler: 3 replies", log)

    def test_untagged_ruler_and_both_arms(self):
        self.rep_dir("duet-ruler", 0, "ruler", sidecar=1)
        self.rep_dir("stock-ruler", 0, "ruler", claim=False, sidecar=2)          # a run before claims existed
        files, _ = self.run_block("ruler", "0", arms="duet stock")
        self.assertEqual(files["duet"], [str(self.out / "partial/duet-ruler/rep0/ruler-1/sidecar.jsonl")])
        self.assertEqual(files["stock"], [str(self.out / "partial/stock-ruler/rep0/ruler-1/sidecar.jsonl")])

    def test_second_resume_takes_the_next_number_and_replays_both(self):
        first = self.out / "partial/duet-ruler-128k-e/rep0/ruler-1"
        first.mkdir(parents=True)
        (first / "sidecar.jsonl").write_text("{}\n")
        self.rep_dir("duet-ruler-128k-e", 0, "ruler", sidecar=1)
        files, _ = self.run_block("ruler", "0", ruler_tag="128k-e")
        second = self.out / "partial/duet-ruler-128k-e/rep0/ruler-2"
        self.assertTrue(second.is_dir())
        self.assertEqual(files["duet"], [str(first / "sidecar.jsonl"), str(second / "sidecar.jsonl")])

    def test_claim_without_sidecar_moves_and_replays_nothing(self):
        self.rep_dir("duet-ruler-64k", 0, "ruler")
        files, _ = self.run_block("ruler", "0", ruler_tag="64k")
        self.assertTrue((self.out / "partial/duet-ruler-64k/rep0/ruler-1/claim.json").is_file())
        self.assertEqual(files, {})

    def test_ruler_left_alone_when_the_job_does_not_run_it(self):
        self.rep_dir("duet-ruler-32k", 0, "ruler", sidecar=1)
        self.run_block("gpqa_se,ruler", "4 5", ruler_tag="32k")       # no repetition 0: the job runs no RULER
        self.run_block("gpqa_se", "0", ruler_tag="32k")               # RULER not listed
        self.assertTrue((self.out / "duet-ruler-32k/rep0/ruler/claim.json").is_file())
        self.assertFalse((self.out / "partial").exists())

    def test_other_kinds_keep_their_paths(self):
        self.rep_dir("duet", 0, "gen", claim=False, sidecar=2)
        self.rep_dir("duet", 1, "gen", sidecar=2, samples=True)
        self.rep_dir("duet", 0, "lcb", sidecar=1)
        self.rep_dir("duet", 2, "gen", sidecar=1)                     # not in REP_LIST
        self.rep_dir("duet", 0, "mmlu_se-s1of2", sidecar=1)
        files, _ = self.run_block("aime24,aime25,lcb", "0 1")
        p = self.out / "partial/duet"
        self.assertEqual(files["duet"], [str(p / "rep0/gen-1/sidecar.jsonl"), str(p / "rep0/lcb-1/sidecar.jsonl")])
        self.assertTrue((self.out / "duet/rep1/gen/claim.json").is_file())
        self.assertTrue((self.out / "duet/rep2/gen/claim.json").is_file())
        self.assertTrue((self.out / "duet/rep0/mmlu_se-s1of2/claim.json").is_file())
        files, _ = self.run_block("mmlu_se", "0", se_shard="1/2")
        self.assertEqual(files["duet"], [str(p / "rep0/mmlu_se-s1of2-1/sidecar.jsonl")])

    def test_own_process_exams_with_a_doc_shard(self):
        self.rep_dir("duet", 0, "humaneval_plus-s1of3", sidecar=2)
        self.rep_dir("duet", 0, "longbench_v2-s1of3", sidecar=1)
        self.rep_dir("duet", 0, "humaneval_plus-s0of3", sidecar=1)                # another shard
        files, _ = self.run_block("humaneval_plus,longbench_v2", "0", doc_shard="1/3")
        p = self.out / "partial/duet/rep0"
        self.assertEqual(files["duet"], [str(p / "humaneval_plus-s1of3-1/sidecar.jsonl"), str(p / "longbench_v2-s1of3-1/sidecar.jsonl")])
        self.assertTrue((self.out / "duet/rep0/humaneval_plus-s0of3/claim.json").is_file())

    def test_two_simple_evals_exams_with_a_shard(self):
        # run.py names this process's directory gen-s<i>of<n>
        self.rep_dir("duet", 0, "gen-s0of2", sidecar=1)
        files, _ = self.run_block("gpqa_se,mmlu_se", "0", se_shard="0/2")
        self.assertEqual(files["duet"], [str(self.out / "partial/duet/rep0/gen-s0of2-1/sidecar.jsonl")])

    def test_without_resume_nothing_moves(self):
        self.rep_dir("duet-ruler-128k-e", 0, "ruler", sidecar=1)
        self.rep_dir("duet", 0, "gen", sidecar=1)
        files, _ = self.run_block("aime24,ruler", "0", ruler_tag="128k-e", resume="0")
        self.assertEqual(files, {})
        self.assertFalse((self.out / "partial").exists())


class RulerReplay(unittest.TestCase):
    """lb.py --resume answers a resumed RULER request (greedy: no sampling seed) from the interrupted run's sidecar record."""

    def setUp(self):
        import importlib.util
        import sys
        if importlib.util.find_spec("aiohttp") is None:           # lb.py's server library
            self.skipTest("aiohttp is not installed")
        sys.path.insert(0, str(ROOT))
        from benchmark.duet_eval.lmeval import common, lb
        self.common, self.lb = common, lb

    def test_seedless_record_is_replayed_and_others_are_not(self):
        ids = [11, 12, 13]
        key = self.common.ids_key(ids)
        record = {"rid": f"4242-{key}-0", "finish": "stop", "text_raw": "The special magic number is 7.", "seed": None,
                  "max_new_tokens": 1024, "prompt_tokens": 3, "completion_tokens": 9}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sidecar.jsonl"
            path.write_text(json.dumps(record) + "\n")
            store, cut = self.lb.load_resume([str(path)])
        balancer = self.lb.Balancer(["http://127.0.0.1:1"], store)

        def ask(ids, params, occurrence=0):
            return balancer.replay(json.dumps({"input_ids": ids, "rid": f"77-{self.common.ids_key(ids)}-{occurrence}",
                                               "sampling_params": params}).encode())

        reply = ask(ids, {"max_new_tokens": 1024, "temperature": 0.0})
        self.assertEqual((cut, reply["text"], reply["meta_info"]["duet_resumed"]), (0, record["text_raw"], record["rid"]))
        self.assertIsNone(ask(ids, {"max_new_tokens": 1024, "sampling_seed": 5}))     # a seeded request is another draw
        self.assertIsNone(ask(ids, {"max_new_tokens": 512}))                          # another budget
        self.assertIsNone(ask(ids + [14], {"max_new_tokens": 1024}))                  # another prompt
        self.assertIsNone(ask(ids, {"max_new_tokens": 1024}, occurrence=1))           # another occurrence


if __name__ == "__main__":
    unittest.main()
