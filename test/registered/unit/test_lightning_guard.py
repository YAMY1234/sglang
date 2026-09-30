import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'benchmark'))
from lightning_sgl_guard import assess, engine_losses


class LightningGuardTest(unittest.TestCase):
    def cells(self):
        windows = [dict(id=i, prompt_sha256=str(i), target=[42]*256, losses=[2.0]*256, nll=2.0) for i in range(32)]
        return {name: dict(complete=True, windows=copy.deepcopy(windows))
                for name in ('reference1','reference2','duet','stock1','stock2','off')}

    def test_complete_cell_and_fixed_threshold(self):
        cells = self.cells()
        self.assertEqual(assess(cells)['status'], 'pass')
        for row in cells['duet']['windows']:
            row['losses'] = [2.0021]*256; row['nll'] = 2.0021
        self.assertEqual(assess(cells)['status'], 'fail')
        for row in cells['reference2']['windows']:
            row['losses'] = [2.003]*256; row['nll'] = 2.003
        self.assertEqual(assess(cells)['status'], 'pass')

    def test_flag_off_checks_every_token_not_just_mean(self):
        cells = self.cells()
        cells['off']['windows'][0]['losses'][:2] = [2.125,1.875]
        self.assertFalse(assess(cells)['flags_off_same_noise'])
        cells['stock2']['windows'][0]['losses'][:2] = [1.875,2.125]
        self.assertTrue(assess(cells)['flags_off_same_noise'])

    def test_missing_window_misaligned_target_and_nan_rejected(self):
        for corruption in ('missing', 'target', 'nan'):
            cells = self.cells()
            if corruption == 'missing': cells['duet']['windows'].pop()
            elif corruption == 'target': cells['duet']['windows'][0]['target'][0] = 43
            else: cells['duet']['windows'][0]['losses'][0] = float('nan')
            with self.assertRaises(ValueError): assess(cells)


if __name__ == '__main__':
    unittest.main()
