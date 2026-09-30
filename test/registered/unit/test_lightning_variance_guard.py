import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark"))
from lightning_sgl_variance_guard import assess_variance, noise_multiple


def cells():
    windows = [
        dict(id=i, prompt_sha256=str(i), target=[42] * 256, losses=[2.0] * 256, nll=2.0)
        for i in range(32)
    ]
    return {
        name: dict(complete=True, windows=copy.deepcopy(windows))
        for name in (
            "reference1",
            "reference2",
            "duet",
            "duet2",
            "stock1",
            "stock2",
            "off",
            "off2",
        )
    }


def set_loss(data, name, value):
    for row in data[name]["windows"]:
        row["losses"] = [value] * 256
        row["nll"] = value


class VarianceGuardTest(unittest.TestCase):
    def test_zero_noise_is_explicit(self):
        self.assertEqual(noise_multiple(0, 0), 0)
        self.assertIsNone(noise_multiple(0.01, 0))
        data = cells()
        set_loss(data, "off", 2.25)
        report = assess_variance(data)
        self.assertTrue(report["zero_noise_nonzero_difference"]["off_stock"])
        self.assertIsNone(report["off_stock_over_stock_noise"])

    def test_round_means_and_multiples(self):
        data = cells()
        for name, value in (
            ("reference2", 2.25),
            ("duet", 2.5),
            ("duet2", 2.75),
            ("stock2", 2.5),
            ("off", 2.375),
            ("off2", 2.375),
        ):
            set_loss(data, name, value)
        report = assess_variance(data)
        self.assertEqual(report["duet_reference_signed_difference"], 0.5)
        self.assertEqual(report["reference_self_difference"], 0.25)
        self.assertEqual(report["duet_self_difference"], 0.25)
        self.assertEqual(report["duet_reference_over_joint_noise"], 2.0)
        self.assertEqual(report["off_stock_over_stock_noise"], 0.25)
        self.assertFalse(report["duet_reference"]["no_systematic_offset_observed"])
        self.assertEqual(report["per_window"][0]["duet_reference_difference"], 0.5)

    def test_old_threshold_and_bitwise_do_not_decide(self):
        data = cells()
        set_loss(data, "duet", 2.5)
        set_loss(data, "duet2", 2.5)
        set_loss(data, "off", 2.125)
        report = assess_variance(data)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["decision"], "round-to-round-variance")
        self.assertFalse(
            report["legacy_record_only"]["abs_round_mean_difference_le_0p002"]
        )
        self.assertFalse(report["legacy_record_only"]["flags_off_bitwise"])

    def test_same_sign_offset_fails_even_with_single_digit_mean_multiple(self):
        data = cells()
        set_loss(data, "reference2", 2.25)
        set_loss(data, "duet", 2.625)
        set_loss(data, "duet2", 2.625)
        result = assess_variance(data)
        self.assertEqual(result["duet_reference_over_reference_noise"], 2.0)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["duet_reference"]["no_systematic_offset_observed"])

    def test_off_second_round_is_required(self):
        data = cells()
        del data["off2"]
        with self.assertRaises(ValueError):
            assess_variance(data)

    def test_second_duet_pass_must_be_complete_finite_and_identical_input(self):
        for defect in ("missing", "short", "target", "nan"):
            data = cells()
            if defect == "missing":
                data["duet2"]["windows"].pop()
            if defect == "short":
                data["duet2"]["windows"][0]["losses"].pop()
            if defect == "target":
                data["duet2"]["windows"][0]["target"][0] = 43
            if defect == "nan":
                data["duet2"]["windows"][0]["losses"][0] = float("nan")
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                assess_variance(data)


if __name__ == "__main__":
    unittest.main()
