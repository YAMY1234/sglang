import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'benchmark'))
from lightning_sgl_accuracy import assess_accuracy


class AccuracyGuardTest(unittest.TestCase):
    def pair(self):
        cell=dict(n_questions=200,reps=1,budget=0,sampling={'temperature':0},
            protocol=dict(gsm8k_file_sha256='pinned',chat_kwargs={},paired_sampling_seed=20260929),
            results=[dict(id=f'gsm8k-{i}',prompt_hash=str(i),gold='18',sampling_seed=i,correct=i<160) for i in range(200)])
        return cell,copy.deepcopy(cell)

    def test_two_percentage_points_is_four_questions(self):
        ref,engine=self.pair()
        for row in engine['results'][156:160]:row['correct']=False
        self.assertEqual(assess_accuracy(ref,engine)['status'],'pass')
        engine['results'][155]['correct']=False
        self.assertEqual(assess_accuracy(ref,engine)['status'],'fail')

    def test_paired_protocol_and_actual_sample_count(self):
        for case in ('count','order','seed','dataset','sampling'):
            ref,engine=self.pair()
            if case=='count':engine['results'].pop()
            if case=='order':engine['results'].reverse()
            if case=='seed':engine['results'][17]['sampling_seed']+=1
            if case=='dataset':engine['protocol']['gsm8k_file_sha256']='changed'
            if case=='sampling':engine['sampling']['temperature']=.6
            with self.assertRaises(ValueError):assess_accuracy(ref,engine)


if __name__=='__main__':unittest.main()
