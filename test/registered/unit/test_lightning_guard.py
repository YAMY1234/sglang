import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark"))
import lightning_sgl_stage2 as stage2
from lightning_sgl_guard import assess


class LightningGuardTest(unittest.TestCase):
    def test_native_entry_and_disabled_release_environment(self):
        for mode in ("duet", "stock", "off"):
            process = mock.Mock(pid=123)
            process.poll.return_value = None
            with (
                mock.patch.dict(
                    stage2.os.environ,
                    {
                        "SGLANG_DUET_ENABLED": "1",
                        "TWINSTAR_LIGHTNING_DUET": "1",
                        "SGLANG_DUET_DIR": "/inherited",
                        "TWINSTAR_LIGHTNING_DUET_DIR": "/old",
                    },
                    clear=True,
                ),
                mock.patch.object(
                    stage2.subprocess, "Popen", return_value=process
                ) as launch,
                mock.patch.object(stage2, "save"),
                mock.patch.object(stage2.os, "killpg"),
                mock.patch.object(
                    stage2, "request", return_value={"data": [{"id": "lightning"}]}
                ),
            ):
                args = SimpleNamespace(gpu=0, duet="/checkpoint", model="/model")
                with stage2.server(args, mode, mock.MagicMock()):
                    command = launch.call_args.args[0]
                    env = launch.call_args.kwargs["env"]
                    self.assertEqual(command[1:3], ["-m", "sglang.launch_server"])
                    self.assertNotIn("SGLANG_DUET_ENABLED", env)
                    self.assertNotIn("TWINSTAR_LIGHTNING_DUET", env)
                    self.assertNotIn("TWINSTAR_LIGHTNING_DUET_DIR", env)
                    if mode == "duet":
                        self.assertEqual(
                            command[command.index("--duet-release") + 1], "/checkpoint"
                        )
                        self.assertEqual(env["SGLANG_DUET_DIR"], "/checkpoint")
                    else:
                        self.assertNotIn("--duet-release", command)
                        self.assertNotIn("SGLANG_DUET_DIR", env)

    def cells(self):
        windows = [
            dict(
                id=i,
                prompt_sha256=str(i),
                target=[42] * 256,
                losses=[2.0] * 256,
                nll=2.0,
            )
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

    def test_absolute_threshold_is_record_only(self):
        cells = self.cells()
        for i in range(32):
            cells["reference2"]["windows"][i]["losses"] = [2 + 1 / 256] * 256
            for name in ("duet", "duet2"):
                cells[name]["windows"][i]["losses"] = [
                    2 + 1 / 512 + 1 / 256 + (1 if i % 2 else -1) / 64
                ] * 256
        report = assess(cells)
        self.assertEqual(report["status"], "pass")
        self.assertFalse(
            report["legacy_record_only"]["abs_round_mean_difference_le_0p002"]
        )
        cells = self.cells()
        for name in ("duet", "duet2"):
            for row in cells[name]["windows"]:
                row["losses"] = [2 + 1 / 1024] * 256
        report = assess(cells)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            report["legacy_record_only"]["abs_round_mean_difference_le_0p002"]
        )

    def test_flag_off_checks_every_token_not_just_mean(self):
        cells = self.cells()
        cells["off"]["windows"][0]["losses"][:2] = [2.125, 1.875]
        self.assertFalse(assess(cells)["legacy_record_only"]["flags_off_same_noise"])
        cells["stock2"]["windows"][0]["losses"][:2] = [1.875, 2.125]
        self.assertTrue(assess(cells)["legacy_record_only"]["flags_off_same_noise"])

    def test_missing_window_misaligned_target_and_nan_rejected(self):
        for corruption in ("missing", "target", "nan"):
            cells = self.cells()
            if corruption == "missing":
                cells["duet"]["windows"].pop()
            elif corruption == "target":
                cells["duet"]["windows"][0]["target"][0] = 43
            else:
                cells["duet"]["windows"][0]["losses"][0] = float("nan")
            with self.assertRaises(ValueError):
                assess(cells)


if __name__ == "__main__":
    unittest.main()
